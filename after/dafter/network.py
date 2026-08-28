"""DAFTER network: frequency patching, causal transformer, and depatching.

The network operates on CausalMauerSTFT frames without temporal
downsampling. A frequency-only convolutional patcher produces one transformer
token per STFT frame. Every transformer layer owns a bounded KV cache for each
flow evaluation, so repeated evaluations do not mix their streaming states.
"""
from __future__ import annotations

import math

import gin
import torch
import torch.nn.functional as F
from torch import nn

from after.autoencoder.audio import CausalMauerSTFT


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
    ) -> None:
        super().__init__()
        if embed_dim % n_heads:
            raise ValueError("embed_dim must be divisible by n_heads")
        if context_frames < 1:
            raise ValueError("context_frames must be positive")
        if (embed_dim // n_heads) % 2:
            raise ValueError("rotary attention requires an even head dimension")

        self.embed_dim = int(embed_dim)
        self.n_heads = int(n_heads)
        self.head_dim = int(embed_dim // n_heads)
        self.context_frames = int(context_frames)
        self.max_flow_evaluations = int(max_flow_evaluations)
        self.max_batch_size = int(max_batch_size)
        self.max_stream_frames = int(max_stream_frames)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training path with causal, local attention and no cache mutation."""
        q, k, v = self._project(x)
        batch = x.shape[0]
        frames = x.shape[1]
        positions = torch.arange(frames, device=x.device)
        rotary_positions = positions[None].expand(batch, frames)
        q = self._apply_rotary(q, rotary_positions)
        k = self._apply_rotary(k, rotary_positions)
        query_positions = positions[:, None]
        key_positions = positions[None, :]
        distances = query_positions - key_positions
        allowed = ((distances >= 0) &
                   (distances <= self.context_frames))
        mask = torch.zeros(frames,
                           frames,
                           device=x.device,
                           dtype=x.dtype)
        mask = mask.masked_fill(~allowed, float("-inf"))
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
                 max_stream_frames: int) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(embed_dim,
                                           elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.modulation = nn.Linear(condition_width, 4 * embed_dim)
        self.attention = StreamingCausalSelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            context_frames=context_frames,
            max_flow_evaluations=max_flow_evaluations,
            max_batch_size=max_batch_size,
            max_stream_frames=max_stream_frames,
        )
        hidden_dim = embed_dim * mlp_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def _modulations(self, condition: torch.Tensor):
        return self.modulation(condition).chunk(4, dim=-1)

    def forward(self, x: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        attention_scale, attention_shift, mlp_scale, mlp_shift = (
            self._modulations(condition))
        h = self.norm_attention(x)
        h = h * (1.0 + attention_scale[:, None]) + attention_shift[:, None]
        x = x + self.attention(h)
        h = self.norm_mlp(x)
        h = h * (1.0 + mlp_scale[:, None]) + mlp_shift[:, None]
        return x + self.mlp(h)

    @torch.jit.export
    def reset_stream(self) -> None:
        self.attention.reset_stream()

    @torch.jit.export
    def forward_stream(self, x: torch.Tensor, condition: torch.Tensor,
                       cache_index: int) -> torch.Tensor:
        attention_scale, attention_shift, mlp_scale, mlp_shift = (
            self._modulations(condition))
        h = self.norm_attention(x)
        h = h * (1.0 + attention_scale[:, None]) + attention_shift[:, None]
        x = x + self.attention.forward_stream(h, cache_index)
        h = self.norm_mlp(x)
        h = h * (1.0 + mlp_scale[:, None]) + mlp_shift[:, None]
        return x + self.mlp(h)


class CausalFrequencyDownsample(nn.Module):
    """Stride frequency by two while convolving causally over time."""

    def __init__(self, in_channels: int, out_channels: int,
                 input_frequencies: int, time_kernel: int,
                 max_flow_steps: int, max_batch_size: int) -> None:
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
        self.activation = nn.SiLU()
        self.register_buffer(
            "stream_cache",
            torch.zeros(max_flow_steps, max_batch_size, in_channels,
                        input_frequencies, self.time_history),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.time_history, 0, 0, 0))
        return self.activation(self.conv(x))

    @torch.jit.export
    def reset_stream(self) -> None:
        self.stream_cache.zero_()

    @torch.jit.export
    def forward_stream(self, x: torch.Tensor,
                       cache_index: int) -> torch.Tensor:
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
        return self.activation(self.conv(x))


class FrequencyPatcher(nn.Module):

    def __init__(self, spectral_bins: int, patch_ratio: int,
                 patch_channels: int, embed_dim: int, time_kernel: int,
                 max_flow_steps: int, max_batch_size: int) -> None:
        super().__init__()
        if spectral_bins % patch_ratio:
            raise ValueError("spectral_bins must be divisible by patch_ratio")
        stages = int(math.log2(patch_ratio))
        if 2**stages != patch_ratio:
            raise ValueError("patch_ratio must be a power of two")
        self.patch_channels = int(patch_channels)
        self.patched_bins = int(spectral_bins // patch_ratio)
        self.downsample_blocks = nn.ModuleList()
        in_channels = 2
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
                ))
            in_channels = patch_channels
            input_frequencies //= 2
        self.project = nn.Linear(patch_channels * self.patched_bins,
                                 embed_dim)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        x = spectrum
        for block in self.downsample_blocks:
            x = block(x)
        batch, channels, frequencies, frames = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, frames,
                                          channels * frequencies)
        return self.project(x)

    @torch.jit.export
    def reset_stream(self) -> None:
        for block in self.downsample_blocks:
            block.reset_stream()

    @torch.jit.export
    def forward_stream(self, spectrum: torch.Tensor,
                       cache_index: int) -> torch.Tensor:
        x = spectrum
        for block in self.downsample_blocks:
            x = block.forward_stream(x, cache_index)
        batch, channels, frequencies, frames = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, frames,
                                          channels * frequencies)
        return self.project(x)


class FrequencyDepatcher(nn.Module):

    def __init__(self, spectral_bins: int, patch_ratio: int,
                 patch_channels: int, embed_dim: int) -> None:
        super().__init__()
        stages = int(math.log2(patch_ratio))
        if 2**stages != patch_ratio:
            raise ValueError("patch_ratio must be a power of two")
        self.patch_channels = int(patch_channels)
        self.patched_bins = int(spectral_bins // patch_ratio)
        self.project = nn.Linear(embed_dim,
                                 patch_channels * self.patched_bins)
        self.upsample_blocks = nn.ModuleList()
        for stage in range(stages):
            is_last = stage == stages - 1
            out_channels = 2 if is_last else patch_channels
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(patch_channels,
                                       out_channels,
                                       kernel_size=(4, 1),
                                       stride=(2, 1),
                                       padding=(1, 0)),
                    nn.Identity() if is_last else nn.SiLU(),
                ))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, frames, _ = tokens.shape
        x = self.project(tokens)
        x = x.reshape(batch, frames, self.patch_channels,
                      self.patched_bins).permute(0, 2, 3, 1)
        for block in self.upsample_blocks:
            x = block(x)
        return x


@gin.configurable
class DafterNetwork(nn.Module):
    """The MIDI/style-conditioned spectral vector field used by DAFTER."""

    def __init__(
        self,
        nfft: int = 512,
        hop_size: int = 64,
        patch_ratio: int = 16,
        patch_channels: int = 16,
        patch_time_kernel: int = 3,
        hidden_channels: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_multiplier: int = 2,
        midi_channels: int = 128,
        style_channels: int = 64,
        condition_width: int = 128,
        attention_context_frames: int = 276,
        max_flow_steps: int = 20,
        max_batch_size: int = 16,
        max_stream_frames: int = 16,
    ) -> None:
        super().__init__()
        self.nfft = int(nfft)
        self.hop_size = int(hop_size)
        self.conditioning_dim = int(midi_channels)
        self.style_dim = int(style_channels)
        self.flow_time_dim = 1
        self.context_frames = int(attention_context_frames)
        self.flow_evaluations = int(max_flow_steps)
        self.spectral_bins = int(nfft // 2)

        self.time_transform = CausalMauerSTFT(
            nfft=nfft,
            hop_size=hop_size,
            synthesis_length=2 * hop_size,
            zero_length=hop_size,
            skip_features=-1,
            normalize=True,
            max_batch_size=max_batch_size,
            alpha_rescale=0.25,
            beta_rescale=1.,
        )
        self.patcher = FrequencyPatcher(
            self.spectral_bins,
            patch_ratio,
            patch_channels,
            hidden_channels,
            patch_time_kernel,
            max_flow_steps,
            max_batch_size,
        )
        self.condition_projection = nn.Linear(midi_channels, hidden_channels)
        self.flow_embedding = nn.Sequential(
            nn.Linear(1, condition_width),
            nn.SiLU(),
            nn.Linear(condition_width, condition_width),
        )
        self.style_projection = nn.Linear(style_channels, condition_width)
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
            ) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_channels)
        self.depatcher = FrequencyDepatcher(self.spectral_bins, patch_ratio,
                                            patch_channels, hidden_channels)

    def _prepare_tokens(self, spectrum: torch.Tensor,
                        conditioning: torch.Tensor) -> torch.Tensor:
        tokens = self.patcher(spectrum)
        frame_conditioning = conditioning.transpose(1, 2)
        if frame_conditioning.shape[1] != tokens.shape[1]:
            raise ValueError("conditioning and spectrogram lengths differ")
        return tokens + self.condition_projection(frame_conditioning)

    def _prepare_stream_tokens(self, spectrum: torch.Tensor,
                               conditioning: torch.Tensor,
                               cache_index: int) -> torch.Tensor:
        tokens = self.patcher.forward_stream(spectrum, cache_index)
        frame_conditioning = conditioning.transpose(1, 2)
        if frame_conditioning.shape[1] != tokens.shape[1]:
            raise ValueError("conditioning and spectrogram lengths differ")
        return tokens + self.condition_projection(frame_conditioning)

    def forward(self, spectrum: torch.Tensor, conditioning: torch.Tensor,
                style: torch.Tensor,
                flow_time: torch.Tensor) -> torch.Tensor:
        """Training path returning a spectral vector field."""
        tokens = self._prepare_tokens(spectrum, conditioning)
        flow_condition = (self.flow_embedding(flow_time) +
                          self.style_projection(style))
        for block in self.blocks:
            tokens = block(tokens, flow_condition)
        return self.depatcher(self.final_norm(tokens))

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
        tokens = self._prepare_stream_tokens(spectrum, conditioning,
                                             cache_index)
        flow_condition = (self.flow_embedding(flow_time) +
                          self.style_projection(style))
        for block in self.blocks:
            tokens = block.forward_stream(tokens, flow_condition, cache_index)
        return self.depatcher(self.final_norm(tokens))

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
        return self.time_transform.inverse_stream(spectrum)

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
