"""DAFTER network: frequency patching, causal transformer, and depatching.

The network operates on CausalMauerSTFT frames without temporal
downsampling. A frequency-only convolutional patcher produces one transformer
token per STFT frame. Every transformer layer owns a bounded KV cache for each
flow evaluation, so repeated evaluations do not mix their streaming states.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import gin
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import (BlockMask, create_block_mask,
                                               flex_attention)

from after.autoencoder.audio import CausalMauerSTFT
from after.diffusion.networks.transformerv2 import PositionalEmbedding


class StreamingCausalSelfAttention(nn.Module):
    """Causal self-attention with fixed-size, per-evaluation KV caches."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        context_frames: int,
        max_flow_evaluations: int,
        max_batch_size: int = 1,
        max_stream_frames: int = 16,
        max_training_frames: int = 1024,
        use_flex_attention: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % n_heads:
            raise ValueError("embed_dim must be divisible by n_heads")
        if context_frames < 1:
            raise ValueError("context_frames must be positive")
        if max_training_frames < 1:
            raise ValueError("max_training_frames must be positive")
        if (embed_dim // n_heads) % 2:
            raise ValueError("rotary attention requires an even head dimension")

        self.embed_dim = int(embed_dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(embed_dim // n_heads)
        self.context_frames = int(context_frames)
        self.max_flow_evaluations = int(max_flow_evaluations)
        self.max_batch_size = int(max_batch_size)
        self.max_stream_frames = int(max_stream_frames)
        self.max_training_frames = int(max_training_frames)
        self.use_flex_attention = bool(use_flex_attention)
        if self.use_flex_attention and self.head_dim & (self.head_dim - 1):
            raise ValueError(
                "FlexAttention requires a power-of-two head dimension; got "
                f"{self.head_dim}")
        if self.use_flex_attention and self.head_dim < 16:
            raise ValueError(
                "FlexAttention requires a head dimension of at least 16; got "
                f"{self.head_dim}")
        self._flex_block_mask: Optional[BlockMask] = None

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)

        cache_shape = (max_flow_evaluations, max_batch_size, n_heads,
                       context_frames, self.head_dim)
        self.register_buffer("k_cache",
                             torch.zeros(cache_shape),
                             persistent=False)
        self.register_buffer("v_cache",
                             torch.zeros(cache_shape),
                             persistent=False)
        self.register_buffer(
            "cache_valid",
            torch.zeros(max_flow_evaluations,
                        max_batch_size,
                        context_frames,
                        dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "position_cache",
            torch.zeros(max_flow_evaluations,
                        max_batch_size,
                        dtype=torch.long),
            persistent=False,
        )
        inv_freq = 1.0 / (10000.0**(
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) /
            float(self.head_dim)))
        self.register_buffer("rotary_inv_freq", inv_freq, persistent=False)

        # Training always starts at position zero, so cache the complete rotary
        # tables and exact bounded-causal mask instead of rebuilding them in
        # every layer on every optimization step. Streaming positions are
        # unbounded and continue to use the dynamic path below.
        positions = torch.arange(max_training_frames, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]
        rotary_cos = angles.cos()
        rotary_sin = angles.sin()
        self.register_buffer("rotary_cos", rotary_cos, persistent=False)
        self.register_buffer("rotary_sin", rotary_sin, persistent=False)
        self.register_buffer("rotary_cos_fp16",
                             rotary_cos.to(torch.float16),
                             persistent=False)
        self.register_buffer("rotary_sin_fp16",
                             rotary_sin.to(torch.float16),
                             persistent=False)
        self.register_buffer("rotary_cos_bf16",
                             rotary_cos.to(torch.bfloat16),
                             persistent=False)
        self.register_buffer("rotary_sin_bf16",
                             rotary_sin.to(torch.bfloat16),
                             persistent=False)
        position_indices = torch.arange(max_training_frames)
        distances = position_indices[:, None] - position_indices[None, :]
        attention_mask = ((distances >= 0) &
                          (distances <= context_frames))
        self.register_buffer("attention_mask",
                             attention_mask,
                             persistent=False)

    @torch.jit.unused
    def prepare_flex_attention(self, device: torch.device) -> None:
        """Build the exact sliding-window block mask once on its CUDA device."""
        if not self.use_flex_attention or device.type != "cuda":
            self._flex_block_mask = None
            return
        context_frames = self.context_frames

        def sliding_window_mask(batch_index, head_index, query_index,
                                key_index):
            del batch_index, head_index
            distance = query_index - key_index
            return ((distance >= 0) & (distance <= context_frames))

        self._flex_block_mask = create_block_mask(
            sliding_window_mask,
            B=None,
            H=None,
            Q_LEN=self.max_training_frames,
            KV_LEN=self.max_training_frames,
            device=str(device),
            BLOCK_SIZE=128,
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, _ = x.shape
        return x.reshape(batch, frames, self.n_heads,
                         self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, frames, _ = x.shape
        return x.transpose(1, 2).reshape(batch, frames, self.embed_dim)

    def _project(self, x: torch.Tensor):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return (self._split_heads(q), self._split_heads(k),
                self._split_heads(v))

    def _apply_rotary(self, x: torch.Tensor,
                      positions: torch.Tensor) -> torch.Tensor:
        angles = (positions.to(dtype=self.rotary_inv_freq.dtype)[..., None] *
                  self.rotary_inv_freq[None, None, :])
        cosines = angles.cos()[:, None].to(dtype=x.dtype)
        sines = angles.sin()[:, None].to(dtype=x.dtype)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cosines - odd * sines
        rotated_odd = even * sines + odd * cosines
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    def _apply_cached_rotary(self, x: torch.Tensor,
                             frames: int) -> torch.Tensor:
        if x.dtype == torch.float16:
            cosines = self.rotary_cos_fp16[:frames]
            sines = self.rotary_sin_fp16[:frames]
        elif x.dtype == torch.bfloat16:
            cosines = self.rotary_cos_bf16[:frames]
            sines = self.rotary_sin_bf16[:frames]
        else:
            cosines = self.rotary_cos[:frames]
            sines = self.rotary_sin[:frames]
        cosines = cosines[None, None]
        sines = sines[None, None]
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cosines - odd * sines
        rotated_odd = even * sines + odd * cosines
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training path with causal, local attention and no cache mutation."""
        q, k, v = self._project(x)
        frames = x.shape[1]
        if frames > self.max_training_frames:
            raise ValueError(
                "training sequence exceeds max_training_frames: "
                f"{frames} > {self.max_training_frames}")
        q = self._apply_cached_rotary(q, frames)
        k = self._apply_cached_rotary(k, frames)
        if (not torch.jit.is_scripting() and self.use_flex_attention and
                x.is_cuda and frames == self.max_training_frames and
                self._flex_block_mask is not None):
            y = flex_attention(q, k, v, block_mask=self._flex_block_mask)
        else:
            mask = self.attention_mask[:frames, :frames]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.output(self._merge_heads(y))

    @torch.jit.export
    def reset_stream(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_valid.zero_()
        self.position_cache.zero_()

    @torch.jit.export
    def forward_stream(self, x: torch.Tensor,
                       cache_index: int) -> torch.Tensor:
        if cache_index < 0 or cache_index >= self.max_flow_evaluations:
            raise ValueError("cache_index is outside the configured range")
        batch = x.shape[0]
        frames = x.shape[1]
        if batch > self.max_batch_size:
            raise ValueError("streaming batch exceeds max_batch_size")
        if frames < 1 or frames > self.max_stream_frames:
            raise ValueError("unsupported number of streaming frames")

        q, k, v = self._project(x)
        positions = (self.position_cache[cache_index, :batch, None] +
                     torch.arange(frames, device=x.device)[None, :])
        q = self._apply_rotary(q, positions)
        k = self._apply_rotary(k, positions)
        past_k = self.k_cache[cache_index, :batch]
        past_v = self.v_cache[cache_index, :batch]
        full_k = torch.cat((past_k, k), dim=2)
        full_v = torch.cat((past_v, v), dim=2)

        past_allowed = self.cache_valid[cache_index, :batch, None, :].expand(
            batch, frames, self.context_frames)
        local_positions = torch.arange(frames, device=x.device)
        current_allowed = (local_positions[None, :] <=
                           local_positions[:, None])
        current_allowed = current_allowed[None].expand(batch, frames, frames)
        allowed = torch.cat((past_allowed, current_allowed), dim=-1)

        mask = torch.zeros(batch,
                           1,
                           frames,
                           self.context_frames + frames,
                           device=x.device,
                           dtype=x.dtype)
        mask = mask.masked_fill(~allowed[:, None, :, :], float("-inf"))
        y = F.scaled_dot_product_attention(q,
                                           full_k,
                                           full_v,
                                           attn_mask=mask)

        if frames >= self.context_frames:
            self.k_cache[cache_index, :batch].copy_(
                k[:, :, -self.context_frames:].detach())
            self.v_cache[cache_index, :batch].copy_(
                v[:, :, -self.context_frames:].detach())
            self.cache_valid[cache_index, :batch].fill_(True)
        else:
            self.k_cache[cache_index, :batch].copy_(torch.cat(
                (past_k[:, :, frames:], k.detach()), dim=2))
            self.v_cache[cache_index, :batch].copy_(torch.cat(
                (past_v[:, :, frames:], v.detach()), dim=2))
            valid = torch.cat((
                self.cache_valid[cache_index, :batch, frames:],
                torch.ones(batch,
                           frames,
                           device=x.device,
                           dtype=torch.bool),
            ), dim=1)
            self.cache_valid[cache_index, :batch].copy_(valid)

        self.position_cache[cache_index, :batch].add_(frames)

        return self.output(self._merge_heads(y))


class ConditionedTransformerBlock(nn.Module):

    def __init__(self, embed_dim: int, n_heads: int, mlp_multiplier: int,
                 condition_width: int, context_frames: int,
                 max_flow_evaluations: int, max_batch_size: int,
                 max_stream_frames: int,
                 max_training_frames: int = 1024,
                 use_flex_attention: bool = False,
                 use_conditioning: bool = True) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(embed_dim,
                                           elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.modulation = (nn.Linear(condition_width, 4 * embed_dim)
                           if use_conditioning else None)
        self.attention = StreamingCausalSelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            context_frames=context_frames,
            max_flow_evaluations=max_flow_evaluations,
            max_batch_size=max_batch_size,
            max_stream_frames=max_stream_frames,
            max_training_frames=max_training_frames,
            use_flex_attention=use_flex_attention,
        )
        hidden_dim = embed_dim * mlp_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def _modulations(self, condition: torch.Tensor):
        if self.modulation is None:
            raise RuntimeError("this transformer block has no conditioning")
        return self.modulation(condition).chunk(4, dim=-1)

    def forward(self, x: torch.Tensor,
                condition: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm_attention(x)
        if self.modulation is not None:
            if condition is None:
                raise ValueError("condition is required by this transformer block")
            attention_scale, attention_shift, mlp_scale, mlp_shift = (
                self._modulations(condition))
            h = (h * (1.0 + attention_scale[:, None]) +
                 attention_shift[:, None])
        x = x + self.attention(h)
        h = self.norm_mlp(x)
        if self.modulation is not None:
            h = h * (1.0 + mlp_scale[:, None]) + mlp_shift[:, None]
        return x + self.mlp(h)

    @torch.jit.export
    def reset_stream(self) -> None:
        self.attention.reset_stream()

    @torch.jit.export
    def forward_stream(self, x: torch.Tensor, condition: Optional[torch.Tensor],
                       cache_index: int) -> torch.Tensor:
        h = self.norm_attention(x)
        if self.modulation is not None:
            if condition is None:
                raise ValueError("condition is required by this transformer block")
            attention_scale, attention_shift, mlp_scale, mlp_shift = (
                self._modulations(condition))
            h = (h * (1.0 + attention_scale[:, None]) +
                 attention_shift[:, None])
        x = x + self.attention.forward_stream(h, cache_index)
        h = self.norm_mlp(x)
        if self.modulation is not None:
            h = h * (1.0 + mlp_scale[:, None]) + mlp_shift[:, None]
        return x + self.mlp(h)


class CausalFrequencyDownsample(nn.Module):
    """Stride frequency by two while convolving causally over time."""

    def __init__(self, in_channels: int, out_channels: int,
                 input_frequencies: int, time_kernel: int,
                 max_flow_steps: int, max_batch_size: int,
                 condition_width: int) -> None:
        super().__init__()
        if time_kernel < 1:
            raise ValueError("time_kernel must be positive")
        self.time_history = int(time_kernel - 1)
        self.max_flow_steps = int(max_flow_steps)
        self.max_batch_size = int(max_batch_size)
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=(4, time_kernel),
                              stride=(2, 1),
                              padding=(1, 0))
        self.condition_projection = nn.Linear(condition_width, out_channels)
        nn.init.zeros_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_projection.bias)
        self.activation = nn.SiLU()
        self.register_buffer(
            "stream_cache",
            torch.zeros(max_flow_steps, max_batch_size, in_channels,
                        input_frequencies, self.time_history),
            persistent=False,
        )

    def forward(self, x: torch.Tensor,
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.pad(x, (self.time_history, 0, 0, 0))
        x = self.conv(x)
        if condition is not None:
            x = x + self.condition_projection(condition)[:, :, None, None]
        return self.activation(x)

    @torch.jit.export
    def reset_stream(self) -> None:
        self.stream_cache.zero_()

    @torch.jit.export
    def forward_stream(
        self,
        x: torch.Tensor,
        cache_index: int,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cache_index < 0 or cache_index >= self.max_flow_steps:
            raise ValueError("cache_index is outside the configured range")
        batch = x.shape[0]
        if batch > self.max_batch_size:
            raise ValueError("streaming batch exceeds max_batch_size")
        if self.time_history > 0:
            cached = self.stream_cache[cache_index, :batch]
            x = torch.cat((cached, x), dim=-1)
            self.stream_cache[cache_index, :batch].copy_(
                x[..., -self.time_history:].detach())
        x = self.conv(x)
        if condition is not None:
            x = x + self.condition_projection(condition)[:, :, None, None]
        return self.activation(x)


class FrequencyPatcher(nn.Module):

    def __init__(self, spectral_bins: int, patch_ratio: int,
                 patch_channels: int, embed_dim: int, time_kernel: int,
                 max_flow_steps: int, max_batch_size: int,
                 condition_width: int) -> None:
        super().__init__()
        if spectral_bins % patch_ratio:
            raise ValueError("spectral_bins must be divisible by patch_ratio")
        stages = int(math.log2(patch_ratio))
        if 2**stages != patch_ratio:
            raise ValueError("patch_ratio must be a power of two")
        self.patch_channels = int(patch_channels)
        self.patched_bins = int(spectral_bins // patch_ratio)
        self.input_conv = nn.Conv2d(2,
                                    patch_channels,
                                    kernel_size=(3, 1),
                                    stride=1,
                                    padding=(1, 0))
        self.downsample_blocks = nn.ModuleList()
        in_channels = patch_channels
        input_frequencies = spectral_bins
        for _ in range(stages):
            self.downsample_blocks.append(
                CausalFrequencyDownsample(
                    in_channels=in_channels,
                    out_channels=patch_channels,
                    input_frequencies=input_frequencies,
                    time_kernel=time_kernel,
                    max_flow_steps=max_flow_steps,
                    max_batch_size=max_batch_size,
                    condition_width=condition_width,
                ))
            in_channels = patch_channels
            input_frequencies //= 2
        self.project = nn.Linear(patch_channels * self.patched_bins,
                                 embed_dim)

    def forward(
        self,
        spectrum: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.input_conv(spectrum)
        if input_scale is not None:
            x = (1.0 + input_scale) * x
        feature_maps = torch.jit.annotate(List[torch.Tensor], [])
        for block in self.downsample_blocks:
            x = block(x, condition)
            feature_maps.append(x)
        batch, channels, frequencies, frames = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, frames,
                                          channels * frequencies)
        return self.project(x), feature_maps

    @torch.jit.export
    def reset_stream(self) -> None:
        for block in self.downsample_blocks:
            block.reset_stream()

    @torch.jit.export
    def forward_stream(
        self,
        spectrum: torch.Tensor,
        cache_index: int,
        input_scale: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.input_conv(spectrum)
        if input_scale is not None:
            x = (1.0 + input_scale) * x
        feature_maps = torch.jit.annotate(List[torch.Tensor], [])
        for block in self.downsample_blocks:
            x = block.forward_stream(x, cache_index, condition)
            feature_maps.append(x)
        batch, channels, frequencies, frames = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, frames,
                                          channels * frequencies)
        return self.project(x), feature_maps


class ConditionedFrequencyUpsample(nn.Module):

    def __init__(self, channels: int, condition_width: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(2 * channels,
                                       channels,
                                       kernel_size=(4, 1),
                                       stride=(2, 1),
                                       padding=(1, 0))
        self.condition_projection = nn.Linear(condition_width, channels)
        nn.init.zeros_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_projection.bias)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor,
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv(x)
        if condition is not None:
            x = x + self.condition_projection(condition)[:, :, None, None]
        return self.activation(x)


class FrequencyDepatcher(nn.Module):

    def __init__(self, spectral_bins: int, patch_ratio: int,
                 patch_channels: int, embed_dim: int,
                 condition_width: int) -> None:
        super().__init__()
        stages = int(math.log2(patch_ratio))
        if 2**stages != patch_ratio:
            raise ValueError("patch_ratio must be a power of two")
        self.patch_channels = int(patch_channels)
        self.patched_bins = int(spectral_bins // patch_ratio)
        self.project = nn.Linear(embed_dim,
                                 patch_channels * self.patched_bins)
        self.upsample_blocks = nn.ModuleList()
        for _ in range(stages):
            self.upsample_blocks.append(
                ConditionedFrequencyUpsample(patch_channels,
                                             condition_width))
        self.output_conv = nn.Conv2d(patch_channels,
                                     2,
                                     kernel_size=(3, 1),
                                     stride=1,
                                     padding=(1, 0))

    def forward(
        self,
        tokens: torch.Tensor,
        skip_features: List[torch.Tensor],
        output_scale: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(skip_features) != len(self.upsample_blocks):
            raise ValueError("one skip feature is required per upsample block")
        batch, frames, _ = tokens.shape
        x = self.project(tokens)
        x = x.reshape(batch, frames, self.patch_channels,
                      self.patched_bins).permute(0, 2, 3, 1)
        for index, block in enumerate(self.upsample_blocks):
            skip = skip_features[len(skip_features) - index - 1]
            if (skip.shape[2] != x.shape[2] or
                    skip.shape[3] != x.shape[3]):
                raise ValueError("patchifier and depatchifier skips differ in shape")
            x = torch.cat((x, skip), dim=1)
            x = block(x, condition)
        if output_scale is not None:
            x = (1.0 + output_scale) * x
        return self.output_conv(x)


@gin.configurable
class DafterNetwork(nn.Module):
    """The MIDI/style-conditioned spectral vector field used by DAFTER."""

    def __init__(
        self,
        nfft: int = 512,
        hop_size: int = 64,
        stft_alpha: float = 0.5,
        stft_beta: float = 3.0,
        patch_ratio: int = 16,
        patch_channels: int = 16,
        patch_time_kernel: int = 3,
        hidden_channels: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_multiplier: int = 2,
        midi_channels: int = 128,
        style_channels: int = 64,
        use_style: bool = True,
        condition_width: int = 128,
        attention_context_frames: int = 276,
        max_flow_steps: int = 20,
        max_batch_size: int = 16,
        max_stream_frames: int = 16,
        max_training_frames: int = 1024,
        use_flex_attention: bool = False,
        whiten_spectrum: bool = False,
    ) -> None:
        super().__init__()
        self.nfft = int(nfft)
        self.hop_size = int(hop_size)
        self.conditioning_dim = int(midi_channels)
        self.use_style = bool(use_style)
        self.style_dim = int(style_channels) if self.use_style else 0
        self.flow_time_dim = 1
        self.context_frames = int(attention_context_frames)
        self.flow_evaluations = int(max_flow_steps)
        self.spectral_bins = int(nfft // 2)
        self.channels_last = False
        self.max_training_frames = int(max_training_frames)
        self.use_flex_attention = bool(use_flex_attention)
        self.whiten_spectrum = bool(whiten_spectrum)

        self.time_transform = CausalMauerSTFT(
            nfft=nfft,
            hop_size=hop_size,
            synthesis_length=2 * hop_size,
            zero_length=hop_size,
            skip_features=-1,
            normalize=True,
            max_batch_size=max_batch_size,
            alpha_rescale=stft_alpha,
            beta_rescale=stft_beta,
        )
        whitening_mean_shape = (1, 2, self.spectral_bins, 1)
        whitening_std_shape = (1, 1, self.spectral_bins, 1)
        self.register_buffer(
            "spectrum_whitening_mean",
            torch.zeros(whitening_mean_shape),
            persistent=self.whiten_spectrum,
        )
        self.register_buffer(
            "spectrum_whitening_std",
            torch.ones(whitening_std_shape),
            persistent=self.whiten_spectrum,
        )
        self.patcher = FrequencyPatcher(
            self.spectral_bins,
            patch_ratio,
            patch_channels,
            hidden_channels,
            patch_time_kernel,
            max_flow_steps,
            max_batch_size,
            condition_width,
        )
        self.token_condition_fusion = nn.Linear(
            hidden_channels + midi_channels, hidden_channels)
        self.noise_spe = PositionalEmbedding(
            num_channels=condition_width,
            max_positions=10_000,
            factor=100.0,
        )
        self.noise_condition = nn.Sequential(
            nn.Linear(condition_width, condition_width),
            nn.SiLU(),
            nn.Linear(condition_width, condition_width),
            nn.SiLU(),
        )
        # input_scale_output = nn.Linear(condition_width, self.spectral_bins)
        # output_scale_output = nn.Linear(condition_width, self.spectral_bins)
        # nn.init.zeros_(input_scale_output.weight)
        # nn.init.zeros_(input_scale_output.bias)
        # nn.init.zeros_(output_scale_output.weight)
        # nn.init.zeros_(output_scale_output.bias)
        # self.scale_inp = nn.Sequential(
        #     nn.Linear(condition_width, condition_width),
        #     nn.SiLU(),
        #     nn.Linear(condition_width, condition_width),
        #     nn.SiLU(),
        #     input_scale_output,
        # )
        # self.scale_out = nn.Sequential(
        #     nn.Linear(condition_width, condition_width),
        #     nn.SiLU(),
        #     nn.Linear(condition_width, condition_width),
        #     nn.SiLU(),
        #     output_scale_output,
        # )
        self.style_projection = (nn.Linear(style_channels, condition_width)
                                 if self.use_style else None)
        self.blocks = nn.ModuleList([
            ConditionedTransformerBlock(
                embed_dim=hidden_channels,
                n_heads=n_heads,
                mlp_multiplier=mlp_multiplier,
                condition_width=condition_width,
                context_frames=attention_context_frames,
                max_flow_evaluations=max_flow_steps,
                max_batch_size=max_batch_size,
                max_stream_frames=max_stream_frames,
                max_training_frames=max_training_frames,
                use_flex_attention=use_flex_attention,
                use_conditioning=True,
            ) for _ in range(n_layers)
        ])
        # self.final_norm = nn.LayerNorm(hidden_channels)
        self.depatcher = FrequencyDepatcher(self.spectral_bins,
                                            patch_ratio,
                                            patch_channels,
                                            hidden_channels,
                                            condition_width)

    @torch.jit.unused
    def prepare_flex_attention(self, device: torch.device) -> None:
        for block in self.blocks:
            block.attention.prepare_flex_attention(device)

    def _fuse_conditioning(self, tokens: torch.Tensor,
                           conditioning: torch.Tensor) -> torch.Tensor:
        frame_conditioning = conditioning.transpose(1, 2)
        if frame_conditioning.shape[1] != tokens.shape[1]:
            raise ValueError("conditioning and spectrogram lengths differ")
        return self.token_condition_fusion(
            torch.cat((tokens, frame_conditioning), dim=-1))

    def _noise_condition_and_frequency_scales(
            self, flow_time: torch.Tensor):
        noise_level = 1.0 - flow_time
        noise_spe = self.noise_spe(noise_level)
        noise_condition = self.noise_condition(noise_spe)
        # input_scale = self.scale_inp(noise_spe).reshape(
        #     flow_time.shape[0], 1, self.spectral_bins, 1)
        # output_scale = self.scale_out(noise_spe).reshape(
        #     flow_time.shape[0], 1, self.spectral_bins, 1)
        return noise_condition#, input_scale, output_scale

    @torch.no_grad()
    def set_spectrum_whitening_statistics(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        """Install per-channel means and shared per-frequency deviations."""
        expected_mean_shape = tuple(self.spectrum_whitening_mean.shape)
        expected_std_shape = tuple(self.spectrum_whitening_std.shape)
        if (tuple(mean.shape) != expected_mean_shape or
                tuple(std.shape) != expected_std_shape):
            raise ValueError(
                "whitening mean/std must have shapes "
                f"{expected_mean_shape}/{expected_std_shape}; got "
                f"{tuple(mean.shape)}/{tuple(std.shape)}")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("whitening statistics must be finite")
        if (std <= 0).any():
            raise ValueError("whitening standard deviations must be positive")
        self.spectrum_whitening_mean.copy_(
            mean.to(device=self.spectrum_whitening_mean.device,
                    dtype=self.spectrum_whitening_mean.dtype))
        self.spectrum_whitening_std.copy_(
            std.to(device=self.spectrum_whitening_std.device,
                   dtype=self.spectrum_whitening_std.dtype))

    def whiten(self, spectrum: torch.Tensor) -> torch.Tensor:
        if not self.whiten_spectrum:
            return spectrum
        return ((spectrum - self.spectrum_whitening_mean) /
                self.spectrum_whitening_std)

    def unwhiten(self, spectrum: torch.Tensor) -> torch.Tensor:
        if not self.whiten_spectrum:
            return spectrum
        return (spectrum * self.spectrum_whitening_std +
                self.spectrum_whitening_mean)

    def audio_to_spectrum(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.whiten(self.time_transform(waveform))

    def spectrum_to_audio(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.time_transform.inverse(self.unwhiten(spectrum))

    def forward(self, spectrum: torch.Tensor, conditioning: torch.Tensor,
                style: Optional[torch.Tensor],
                flow_time: torch.Tensor) -> torch.Tensor:
        """Training path returning a spectral vector field."""
        if self.channels_last:
            spectrum = spectrum.contiguous(memory_format=torch.channels_last)
        noise_condition = (
            self._noise_condition_and_frequency_scales(flow_time))

        #, input_scale, output_scale


        patch_tokens, skip_features = self.patcher(spectrum,
                                                   input_scale=None,
                                                   condition=noise_condition)
        tokens = self._fuse_conditioning(patch_tokens, conditioning)
        flow_condition = noise_condition
        if self.style_projection is not None:
            if style is None:
                raise ValueError("style is required when use_style=True")
            flow_condition = noise_condition + self.style_projection(style)
        elif style is not None:
            raise ValueError("style must be None when use_style=False")
        for block in self.blocks:
            tokens = block(tokens, flow_condition)
        return self.depatcher(tokens,
                              skip_features,
                              output_scale=None,
                              condition=noise_condition)

    @torch.jit.export
    def reset_stream(self) -> None:
        self.time_transform.reset_stream()
        self.patcher.reset_stream()
        for block in self.blocks:
            block.reset_stream()

    @torch.jit.export
    def denoise_spectrum_stream(self, spectrum: torch.Tensor,
                                conditioning: torch.Tensor,
                                style: torch.Tensor,
                                flow_time: torch.Tensor,
                                cache_index: int) -> torch.Tensor:
        if self.channels_last:
            spectrum = spectrum.contiguous(memory_format=torch.channels_last)
        noise_condition, input_scale, output_scale = (
            self._noise_condition_and_frequency_scales(flow_time))
        patch_tokens, skip_features = self.patcher.forward_stream(
            spectrum, cache_index, input_scale, noise_condition)
        tokens = self._fuse_conditioning(patch_tokens, conditioning)
        flow_condition = (noise_condition + self.style_projection(style))
        for block in self.blocks:
            tokens = block.forward_stream(tokens, flow_condition, cache_index)
        return self.depatcher(tokens,
                              skip_features,
                              output_scale,
                              noise_condition)

    @torch.jit.export
    def forward_stream(self, noise_spectrum: torch.Tensor,
                       conditioning: torch.Tensor,
                       style: torch.Tensor,
                       flow_times: torch.Tensor) -> torch.Tensor:
        if (noise_spectrum.shape[1] != 2 or
                noise_spectrum.shape[2] != self.spectral_bins):
            raise ValueError("noise_spectrum has the wrong spectral shape")
        spectrum = noise_spectrum
        evaluation_count = flow_times.shape[1]
        if evaluation_count < 1 or evaluation_count > self.flow_evaluations:
            raise ValueError(
                "flow_times exceeds the configured evaluation cache count")
        step_size = 1.0 / float(evaluation_count)
        for cache_index in range(evaluation_count):
            velocity = self.denoise_spectrum_stream(
                spectrum, conditioning, style,
                flow_times[:, cache_index],
                cache_index)
            spectrum = spectrum + step_size * velocity
        return self.time_transform.inverse_stream(self.unwhiten(spectrum))

    def cache_size_bytes(self) -> int:
        total = sum(block.stream_cache.numel() * 4
                    for block in self.patcher.downsample_blocks)
        for block in self.blocks:
            total += block.attention.k_cache.numel() * 4
            total += block.attention.v_cache.numel() * 4
            total += block.attention.cache_valid.numel()
            total += block.attention.position_cache.numel() * 8
        return total


def context_frames_for_seconds(seconds: float, sample_rate: int,
                               hop_size: int) -> int:
    return int(math.ceil(seconds * sample_rate / hop_size))
