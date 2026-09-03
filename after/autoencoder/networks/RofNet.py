"""Streaming axial-transformer autoencoder for complex spectrograms."""

from __future__ import annotations

import math
from typing import Optional

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bottlenecks import ReluBottleneck, TanhBottleneck, VAEBottleneck


class SharedAttention(nn.Module):
    """QKV attention with interchangeable SDPA and explicit implementations."""

    def __init__(self, dim: int, heads: int, attention_impl: str = "sdpa"):
        super().__init__()
        if dim % heads:
            raise ValueError(f"embedding dimension {dim} is not divisible by {heads} heads")
        if attention_impl not in ("sdpa", "manual"):
            raise ValueError("attention_impl must be 'sdpa' or 'manual'")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.attention_impl = attention_impl
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def set_attention_impl(self, attention_impl: str) -> None:
        if attention_impl not in ("sdpa", "manual"):
            raise ValueError("attention_impl must be 'sdpa' or 'manual'")
        self.attention_impl = attention_impl

    def project(self, x: torch.Tensor):
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        return (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None and mask.ndim == q.ndim - 1:
            mask = mask.unsqueeze(0).expand(q.shape[0], -1, -1, -1)
        if self.attention_impl == "sdpa":
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        queries, keys = q.shape[-2], k.shape[-2]
        flat_q = q.reshape(-1, queries, self.head_dim)
        flat_k = k.reshape(-1, keys, self.head_dim)
        flat_v = v.reshape(-1, keys, self.head_dim)
        scores = torch.bmm(flat_q, flat_k.transpose(1, 2)) * self.scale
        if mask is not None:
            scores = scores + mask.reshape(-1, queries, keys)
        return torch.bmm(torch.softmax(scores, dim=-1), flat_v).reshape(q.shape)

    def merge(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, tokens, _ = x.shape
        return self.out(x.transpose(1, 2).reshape(batch, tokens, self.dim))

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q, k, v = self.project(x)
        return self.merge(self.attend(q, k, v, mask))


class FrequencyAttention(nn.Module):
    def __init__(self, dim: int, heads: int, attention_impl: str):
        super().__init__()
        self.attention = SharedAttention(dim, heads, attention_impl)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, frequencies, dim = x.shape
        y = self.attention(x.reshape(batch * frames, frequencies, dim))
        return y.reshape(batch, frames, frequencies, dim)


class TemporalAttention(nn.Module):
    """Local causal attention; cache layout is ``(..., H, W-1, Dh)``."""

    def __init__(
        self,
        dim: int,
        heads: int,
        time_window: int,
        attention_impl: str,
        byblock: bool = False,
        block_size: Optional[int] = None,
    ):
        super().__init__()
        if time_window < 1:
            raise ValueError("time_window must be positive")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.time_window = int(time_window)
        self.byblock = bool(byblock)
        self.block_size = int(
            2 * self.time_window if block_size is None else block_size
        )
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        self.attention = SharedAttention(dim, heads, attention_impl)
        self.relative_bias = nn.Parameter(torch.zeros(heads, time_window))
        self.register_buffer("k_cache", torch.empty(0), persistent=False)
        self.register_buffer("v_cache", torch.empty(0), persistent=False)

    def _mask(
        self,
        queries: int,
        cached: int,
        device: torch.device,
        dtype: torch.dtype,
        cache_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_positions = torch.arange(queries, device=device)
        key_positions = torch.arange(-cached, queries, device=device)
        lag = query_positions[:, None] - key_positions[None, :]
        allowed = torch.logical_and(lag >= 0, lag < self.time_window)
        if cache_valid is not None and cached:
            cache_indices = torch.arange(cached, device=device)
            valid_past = cache_indices >= cached - cache_valid.to(torch.long)
            allowed = torch.logical_and(
                allowed,
                torch.cat(
                    (
                        valid_past,
                        torch.ones(queries, device=device, dtype=torch.bool),
                    )
                )[None, :],
            )
        bias = self.relative_bias[:, lag.clamp(0, self.time_window - 1)]
        invalid = torch.full_like(bias, torch.finfo(dtype).min)
        return torch.where(allowed[None, :, :], bias.to(dtype), invalid)

    def _project(self, x: torch.Tensor):
        leading = x.shape[:-2]
        frames, dim = x.shape[-2:]
        q, k, v = self.attention.project(x.reshape(-1, frames, dim))
        cache_shape = (*leading, self.heads, frames, self.head_dim)
        return (
            q.reshape(cache_shape),
            k.reshape(cache_shape),
            v.reshape(cache_shape),
        )

    def _local_mask(
        self,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        offsets = torch.arange(self.time_window, device=device)
        positions = torch.arange(frames, device=device)[:, None]
        allowed = offsets[None, :] >= self.time_window - 1 - positions
        bias = self.relative_bias.flip(-1)[:, None, :].expand(-1, frames, -1)
        invalid = torch.full_like(bias, torch.finfo(dtype).min)
        return torch.where(allowed[None, :, :], bias.to(dtype), invalid)

    def _block_mask(
        self,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
        first: bool,
    ) -> torch.Tensor:
        history = self.time_window - 1
        query_positions = torch.arange(block_size, device=device)[:, None]
        key_positions = torch.arange(-history, block_size, device=device)[None, :]
        lag = query_positions - key_positions
        allowed = torch.logical_and(lag >= 0, lag < self.time_window)
        if first and history:
            allowed = torch.logical_and(allowed, key_positions >= 0)
        bias = self.relative_bias.to(dtype)[:, lag.clamp(0, self.time_window - 1)]
        invalid = torch.full_like(bias, torch.finfo(dtype).min)
        return torch.where(allowed[None, :, :], bias, invalid)

    def _attend_by_block(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        frames: int,
    ) -> torch.Tensor:
        batch = q.shape[0]
        block = self.block_size
        blocks = (frames + block - 1) // block
        padded_frames = blocks * block
        right_padding = padded_frames - frames
        history = self.time_window - 1
        context = block + history

        q = F.pad(q, (0, 0, 0, right_padding))
        k = F.pad(k, (0, 0, history, right_padding))
        v = F.pad(v, (0, 0, history, right_padding))

        q = q.reshape(batch, self.heads, blocks, block, self.head_dim)
        q = q.permute(0, 2, 1, 3, 4)
        k = k.unfold(-2, context, block).transpose(-1, -2)
        v = v.unfold(-2, context, block).transpose(-1, -2)
        k = k.permute(0, 2, 1, 3, 4)
        v = v.permute(0, 2, 1, 3, 4)

        first_mask = self._block_mask(block, q.device, q.dtype, first=True)
        first_output = self.attention.attend(
            q[:, 0].contiguous(),
            k[:, 0].contiguous(),
            v[:, 0].contiguous(),
            first_mask,
        )[:, None]

        if blocks > 1:
            steady_mask = self._block_mask(block, q.device, q.dtype, first=False)
            rest_output = self.attention.attend(
                q[:, 1:].reshape(-1, self.heads, block, self.head_dim),
                k[:, 1:].reshape(-1, self.heads, context, self.head_dim),
                v[:, 1:].reshape(-1, self.heads, context, self.head_dim),
                steady_mask,
            ).reshape(batch, blocks - 1, self.heads, block, self.head_dim)
            output = torch.cat((first_output, rest_output), dim=1)
        else:
            output = first_output

        output = output.permute(0, 2, 1, 3, 4)
        return output.reshape(batch, self.heads, padded_frames, self.head_dim)[
            :, :, :frames
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-2]
        frames = x.shape[-2]
        q, k, v = self._project(x)
        q = q.reshape(-1, self.heads, frames, self.head_dim)
        k = k.reshape(-1, self.heads, frames, self.head_dim)
        v = v.reshape(-1, self.heads, frames, self.head_dim)

        if self.byblock:
            y = self._attend_by_block(q, k, v, frames)
            y = self.attention.merge(y)
            return y.reshape(*leading, frames, self.dim)

        history = self.time_window - 1
        k = F.pad(k, (0, 0, history, 0))
        v = F.pad(v, (0, 0, history, 0))
        k = k.unfold(-2, self.time_window, 1).transpose(-1, -2)
        v = v.unfold(-2, self.time_window, 1).transpose(-1, -2)

        q = q.permute(0, 2, 1, 3).unsqueeze(-2)
        k = k.permute(0, 2, 1, 3, 4)
        v = v.permute(0, 2, 1, 3, 4)
        mask = self._local_mask(frames, x.device, x.dtype)
        mask = mask.permute(1, 0, 2)[None, :, :, None, :]

        if self.attention.attention_impl == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            scores = (q * k).sum(dim=-1) * self.attention.scale
            weights = torch.softmax(scores + mask.squeeze(-2), dim=-1)
            y = (weights.unsqueeze(-1) * v).sum(dim=-2)

        y = y.squeeze(-2).permute(0, 2, 1, 3)
        y = self.attention.merge(y)
        return y.reshape(*leading, frames, self.dim)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_valid: Optional[torch.Tensor] = None,
    ):
        leading = x.shape[:-2]
        frames = x.shape[-2]
        cached = k_cache.shape[-2]
        q, k, v = self._project(x)
        full_k = torch.cat((k_cache, k), dim=-2)
        full_v = torch.cat((v_cache, v), dim=-2)
        mask = self._mask(frames, cached, x.device, x.dtype, cache_valid)
        y = self.attention.attend(
            q.reshape(-1, self.heads, frames, self.head_dim),
            full_k.reshape(-1, self.heads, cached + frames, self.head_dim),
            full_v.reshape(-1, self.heads, cached + frames, self.head_dim),
            mask,
        )
        y = self.attention.merge(y).reshape(*leading, frames, self.dim)
        keep = self.time_window - 1
        if keep:
            new_k = full_k[..., -keep:, :]
            new_v = full_v[..., -keep:, :]
        else:
            new_k = full_k[..., :0, :]
            new_v = full_v[..., :0, :]
        return y, new_k, new_v

    def forward_stream(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-2]
        expected = (*leading, self.heads, self.head_dim)
        if (
            self.k_cache.numel() == 0
            or self.k_cache.shape[:-2] != expected[:-1]
            or self.k_cache.shape[-1] != expected[-1]
            or self.k_cache.device != x.device
            or self.k_cache.dtype != x.dtype
        ):
            shape = (*leading, self.heads, 0, self.head_dim)
            self.k_cache = x.new_empty(shape)
            self.v_cache = x.new_empty(shape)
        y, k, v = self.forward_with_cache(x, self.k_cache, self.v_cache)
        self.k_cache = k.detach()
        self.v_cache = v.detach()
        return y

    def reset_stream(self) -> None:
        if self.k_cache.ndim >= 2:
            self.k_cache = self.k_cache[..., :0, :]
            self.v_cache = self.v_cache[..., :0, :]


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiplier: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * multiplier),
            nn.GELU(),
            nn.Linear(dim * multiplier, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AxialBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        time_window: int,
        ff_mult: int,
        attention_impl: str,
        byblock: bool = False,
        block_size: Optional[int] = None,
    ):
        super().__init__()
        self.freq_norm = nn.LayerNorm(dim)
        self.freq_attention = FrequencyAttention(dim, heads, attention_impl)
        self.time_norm = nn.LayerNorm(dim)
        self.time_attention = TemporalAttention(
            dim, heads, time_window, attention_impl, byblock, block_size
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, ff_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.freq_attention(self.freq_norm(x))
        x = x + self.time_attention(self.time_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.mlp(self.mlp_norm(x))

    def forward_stream(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.freq_attention(self.freq_norm(x))
        h = self.time_norm(x).transpose(1, 2)
        x = x + self.time_attention.forward_stream(h).transpose(1, 2)
        return x + self.mlp(self.mlp_norm(x))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_valid: torch.Tensor,
    ):
        x = x + self.freq_attention(self.freq_norm(x))
        h, new_k, new_v = self.time_attention.forward_with_cache(
            self.time_norm(x).transpose(1, 2), k_cache, v_cache, cache_valid
        )
        x = x + h.transpose(1, 2)
        return x + self.mlp(self.mlp_norm(x)), new_k, new_v


class TemporalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        time_window: int,
        ff_mult: int,
        attention_impl: str,
        byblock: bool = False,
        block_size: Optional[int] = None,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = TemporalAttention(
            dim, heads, time_window, attention_impl, byblock, block_size
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, ff_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))

    def forward_stream(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention.forward_stream(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_valid: torch.Tensor,
    ):
        h, new_k, new_v = self.attention.forward_with_cache(
            self.attention_norm(x), k_cache, v_cache, cache_valid
        )
        x = x + h
        return x + self.mlp(self.mlp_norm(x)), new_k, new_v


class FrequencyMerge(nn.Module):
    def __init__(self, frequencies: int, ratio: int, dim: int, out_dim: int):
        super().__init__()
        if ratio < 1 or frequencies % ratio:
            raise ValueError(
                f"frequency size {frequencies} is not divisible by ratio {ratio}"
            )
        self.ratio = int(ratio)
        self.projection = nn.Linear(dim * ratio, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, frequencies, dim = x.shape
        x = x.reshape(batch, frames, frequencies // self.ratio, dim * self.ratio)
        return self.projection(x)


class FrequencyExpand(nn.Module):
    def __init__(self, ratio: int, dim: int, out_dim: int):
        super().__init__()
        self.ratio = int(ratio)
        self.out_dim = int(out_dim)
        self.projection = nn.Linear(dim, ratio * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, frequencies, _ = x.shape
        x = self.projection(x)
        return x.reshape(batch, frames, frequencies * self.ratio, self.out_dim)


class CausalSmoothingConv(nn.Module):
    def __init__(self, channels: int, frequencies: int, kernel_size):
        super().__init__()
        frequency_kernel, time_kernel = tuple(kernel_size)
        if frequency_kernel % 2 != 1 or time_kernel < 1:
            raise ValueError("smoothing requires an odd frequency kernel and positive time kernel")
        self.frequencies = int(frequencies)
        self.time_history = int(time_kernel - 1)
        self.frequency_padding = frequency_kernel // 2
        self.conv = nn.Conv2d(channels, channels, (frequency_kernel, time_kernel))
        self.register_buffer("cache", torch.empty(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(
            x,
            (self.time_history, 0, self.frequency_padding, self.frequency_padding),
        )
        return self.conv(x)

    def forward_with_cache(self, x: torch.Tensor, cache: torch.Tensor):
        if self.time_history:
            x = torch.cat((cache, x), dim=-1)
            new_cache = x[..., -self.time_history :]
        else:
            new_cache = x[..., :0]
        x = F.pad(x, (0, 0, self.frequency_padding, self.frequency_padding))
        return self.conv(x), new_cache

    def forward_stream(self, x: torch.Tensor) -> torch.Tensor:
        expected = (x.shape[0], x.shape[1], x.shape[2], self.time_history)
        if (
            self.cache.numel() == 0
            or tuple(self.cache.shape) != expected
            or self.cache.device != x.device
            or self.cache.dtype != x.dtype
        ):
            self.cache = x.new_zeros(expected)
        y, cache = self.forward_with_cache(x, self.cache)
        self.cache = cache.detach()
        return y

    def reset_stream(self) -> None:
        if self.cache.numel():
            self.cache.zero_()


@gin.configurable
class RofNet(nn.Module):
    """RoFormer-style streaming spectrogram autoencoder."""

    def __init__(
        self,
        in_size: int = 2,
        out_size=None,
        bottleneck_size: int = 16,
        audio_channels: int = 1,
        freq_size: int = 128,
        dims=(32, 64, 96, 128),
        freq_ratios=(2, 2, 2),
        depths=(1, 1, 1, 1),
        heads=(4, 4, 4, 4),
        middle_dim: int = 128,
        middle_layers: int = 2,
        middle_heads: int = 4,
        time_window: int = 8,
        ff_mult: int = 2,
        time_transform=None,
        bottleneck=None,
        use_vae: bool = False,
        smoothing_conv: bool = False,
        smoothing_kernel_size=(3, 3),
        attention_impl: str = "sdpa",
        byblock: bool = False,
        block_size: Optional[int] = None,
    ):
        super().__init__()
        dims = tuple(dims)
        depths = tuple(depths)
        heads = tuple(heads)
        freq_ratios = tuple(freq_ratios)
        if not (len(dims) == len(depths) == len(heads)):
            raise ValueError("dims, depths and heads must have equal lengths")
        if len(freq_ratios) != len(dims) - 1:
            raise ValueError("freq_ratios must contain one entry between each stage")
        if not dims:
            raise ValueError("at least one spectral stage is required")
        for dim, stage_heads in zip(dims, heads):
            if dim % stage_heads:
                raise ValueError(f"embedding dimension {dim} is not divisible by {stage_heads} heads")
        if middle_dim % middle_heads:
            raise ValueError("middle_dim must be divisible by middle_heads")
        if attention_impl not in ("sdpa", "manual"):
            raise ValueError("attention_impl must be 'sdpa' or 'manual'")
        if time_transform is None:
            raise ValueError("time_transform is required")

        self.in_size = int(in_size)
        self.out_size = int(in_size if out_size is None else out_size)
        self.bottleneck_size = int(bottleneck_size)
        self.audio_channels = int(audio_channels)
        self.freq_size = int(freq_size)
        self.dims = dims
        self.depths = depths
        self.heads = heads
        self.freq_ratios = freq_ratios
        self.middle_dim = int(middle_dim)
        self.middle_layers = int(middle_layers)
        self.middle_heads = int(middle_heads)
        self.time_window = int(time_window)
        self.byblock = bool(byblock)
        self.block_size = int(2 * time_window if block_size is None else block_size)
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        self.use_vae = bool(use_vae)
        self.attention_impl = attention_impl
        self.time_transform = time_transform
        if bottleneck is None:
            bottleneck = VAEBottleneck() if use_vae else nn.Identity()
        self.bottleneck = bottleneck

        transform_frequencies = time_transform.nfft // 2 + 1
        if time_transform.skip_features is not None:
            transform_frequencies -= abs(time_transform.skip_features)
        if transform_frequencies != self.freq_size:
            raise ValueError(
                f"freq_size={self.freq_size}, but time_transform produces "
                f"{transform_frequencies} frequency bins"
            )

        frequencies = [self.freq_size]
        for ratio in freq_ratios:
            if ratio < 1 or frequencies[-1] % ratio:
                raise ValueError(
                    f"frequency size {frequencies[-1]} is not divisible by ratio {ratio}"
                )
            frequencies.append(frequencies[-1] // ratio)
        self.frequencies = tuple(frequencies)
        self.freq_final_dim = frequencies[-1]

        self.input_projection = nn.Linear(in_size, dims[0])
        self.frequency_positions = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, f, d)) for f, d in zip(frequencies, dims)]
        )
        self.encoder_stages = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for dim, depth, stage_heads in zip(dims, depths, heads):
            self.encoder_stages.append(
                nn.ModuleList(
                    [
                        AxialBlock(
                            dim,
                            stage_heads,
                            time_window,
                            ff_mult,
                            attention_impl,
                            self.byblock,
                            self.block_size,
                        )
                        for _ in range(depth)
                    ]
                )
            )
            self.decoder_stages.append(
                nn.ModuleList(
                    [
                        AxialBlock(
                            dim,
                            stage_heads,
                            time_window,
                            ff_mult,
                            attention_impl,
                            self.byblock,
                            self.block_size,
                        )
                        for _ in range(depth)
                    ]
                )
            )

        self.frequency_merges = nn.ModuleList(
            [
                FrequencyMerge(frequencies[i], freq_ratios[i], dims[i], dims[i + 1])
                for i in range(len(freq_ratios))
            ]
        )
        self.frequency_expands = nn.ModuleList(
            [
                FrequencyExpand(freq_ratios[i], dims[i + 1], dims[i])
                for i in range(len(freq_ratios))
            ]
        )

        flat_dim = frequencies[-1] * dims[-1]
        self.middle_encode_projection = nn.Linear(flat_dim, middle_dim)
        self.middle_encoder = nn.ModuleList(
            [
                TemporalBlock(
                    middle_dim,
                    middle_heads,
                    time_window,
                    ff_mult,
                    attention_impl,
                    self.byblock,
                    self.block_size,
                )
                for _ in range(middle_layers)
            ]
        )
        encoded_dim = 2 * bottleneck_size if use_vae else bottleneck_size
        self.latent_projection = nn.Linear(middle_dim, encoded_dim)
        self.middle_decode_projection = nn.Linear(bottleneck_size, middle_dim)
        self.middle_decoder = nn.ModuleList(
            [
                TemporalBlock(
                    middle_dim,
                    middle_heads,
                    time_window,
                    ff_mult,
                    attention_impl,
                    self.byblock,
                    self.block_size,
                )
                for _ in range(middle_layers)
            ]
        )
        self.spectral_projection = nn.Linear(middle_dim, flat_dim)
        self.output_projection = nn.Linear(dims[0], self.out_size)
        self.smoothing = (
            CausalSmoothingConv(self.out_size, freq_size, smoothing_kernel_size)
            if smoothing_conv
            else None
        )

        # Kept for compatibility with the existing trainer interface.
        object.__setattr__(
            self,
            "encoder",
            nn.ModuleList([
                self.input_projection,
                self.encoder_stages,
                self.frequency_merges,
                self.middle_encode_projection,
                self.middle_encoder,
                self.latent_projection,
            ]),
        )
        object.__setattr__(
            self,
            "decoder",
            nn.ModuleList([
                self.middle_decode_projection,
                self.middle_decoder,
                self.spectral_projection,
                self.frequency_expands,
                self.decoder_stages,
                self.output_projection,
            ]),
        )

    def set_attention_impl(self, attention_impl: str) -> None:
        if attention_impl not in ("sdpa", "manual"):
            raise ValueError("attention_impl must be 'sdpa' or 'manual'")
        self.attention_impl = attention_impl
        for module in self.modules():
            if isinstance(module, SharedAttention):
                module.set_attention_impl(attention_impl)

    def _pack_audio(self, x: torch.Tensor) -> torch.Tensor:
        if self.audio_channels == 1:
            return x
        batch, channels, samples = x.shape
        return x.reshape(batch * channels, 1, samples)

    def _unpack_audio(self, x: torch.Tensor) -> torch.Tensor:
        if self.audio_channels == 1:
            return x
        return x.reshape(x.shape[0] // self.audio_channels, self.audio_channels, -1)

    def _encode_features(self, spectrum: torch.Tensor, streaming: bool) -> torch.Tensor:
        x = spectrum.permute(0, 3, 2, 1)
        x = self.input_projection(x)
        for index, stage in enumerate(self.encoder_stages):
            x = x + self.frequency_positions[index]
            for block in stage:
                x = block.forward_stream(x) if streaming else block(x)
            if index < len(self.frequency_merges):
                x = self.frequency_merges[index](x)
        batch, frames, frequencies, dim = x.shape
        x = self.middle_encode_projection(x.reshape(batch, frames, frequencies * dim))
        for block in self.middle_encoder:
            x = block.forward_stream(x) if streaming else block(x)
        return self.latent_projection(x).transpose(1, 2)

    def _decode_features(self, latent: torch.Tensor, streaming: bool) -> torch.Tensor:
        x = self.middle_decode_projection(latent.transpose(1, 2))
        for block in self.middle_decoder:
            x = block.forward_stream(x) if streaming else block(x)
        batch, frames, _ = x.shape
        x = self.spectral_projection(x).reshape(
            batch, frames, self.frequencies[-1], self.dims[-1]
        )
        last = len(self.decoder_stages) - 1
        for stage_index in range(last, -1, -1):
            x = x + self.frequency_positions[stage_index]
            for block in self.decoder_stages[stage_index]:
                x = block.forward_stream(x) if streaming else block(x)
            if stage_index:
                x = self.frequency_expands[stage_index - 1](x)
        x = self.output_projection(x).permute(0, 3, 2, 1)
        if self.smoothing is not None:
            x = self.smoothing.forward_stream(x) if streaming else self.smoothing(x)
        return x

    def _apply_bottleneck(self, x: torch.Tensor, return_mean: bool = False):
        if isinstance(self.bottleneck, nn.Identity):
            return x, x.new_zeros(())
        if isinstance(self.bottleneck, TanhBottleneck):
            return self.bottleneck(x)
        return self.bottleneck(x, return_mean=return_mean)

    def _stream_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.bottleneck, VAEBottleneck):
            return x.chunk(2, dim=1)[0]
        if isinstance(self.bottleneck, TanhBottleneck):
            return self.bottleneck.scale * torch.tanh(x)
        if isinstance(self.bottleneck, (ReluBottleneck, nn.Identity)):
            return x
        return self.bottleneck.forward_stream(x)

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = True,
        freeze_encoder: bool = False,
        look_ahead_steps: int = 0,
    ):
        packed = self._pack_audio(x)
        spectrum = self.time_transform(packed)
        x_multiband = spectrum.clone()
        if freeze_encoder:
            with torch.no_grad():
                encoded = self._encode_features(spectrum, False)
                encoded, regloss = self._apply_bottleneck(encoded)
        else:
            encoded = self._encode_features(spectrum, False)
            encoded, regloss = self._apply_bottleneck(encoded)
        z = encoded.clone()
        if look_ahead_steps > 0:
            z = torch.cat((z[..., look_ahead_steps:], torch.zeros_like(z[..., :look_ahead_steps])), dim=-1)
        y_multiband = self._decode_features(z, False)
        y = self._unpack_audio(self.time_transform.inverse(y_multiband))
        if return_all:
            return y, y_multiband, z, regloss, x_multiband
        return y

    def encode(self, x: torch.Tensor, with_multi: bool = False, return_mean: bool = False):
        spectrum = self.time_transform(self._pack_audio(x))
        encoded = self._encode_features(spectrum, False)
        result = self._apply_bottleneck(encoded, return_mean)
        if return_mean and len(result) == 3:
            latent, regloss, mean = result
            return (latent, spectrum, mean) if with_multi else (latent, regloss, mean)
        latent, regloss = result
        return (latent, spectrum) if with_multi else (latent, regloss)

    def decode(self, z: torch.Tensor, with_multi: bool = False):
        spectrum = self._decode_features(z, False)
        y = self._unpack_audio(self.time_transform.inverse(spectrum))
        return (y, spectrum) if with_multi else y

    def encode_stream(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = self.time_transform.forward_stream(self._pack_audio(x))
        encoded = self._encode_features(spectrum, True)
        return self._stream_bottleneck(encoded)

    def decode_stream(self, z: torch.Tensor) -> torch.Tensor:
        spectrum = self._decode_features(z, True)
        return self._unpack_audio(self.time_transform.inverse_stream(spectrum))

    def forward_stream(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_stream(self.encode_stream(x))

    def reset_stream_state(self) -> None:
        self.time_transform.reset_stream()
        for module in self.modules():
            if isinstance(module, TemporalAttention):
                module.reset_stream()
        if self.smoothing is not None:
            self.smoothing.reset_stream()

    reset_stream = reset_stream_state


class StatelessStreamingRofNet(nn.Module):
    """Functional fixed-callback streaming wrapper with one flat state tensor."""

    def __init__(
        self, model: RofNet, callback_samples: int, mode: str = "forward"
    ):
        super().__init__()
        if mode not in ("encode", "decode", "forward"):
            raise ValueError("mode must be 'encode', 'decode' or 'forward'")
        self.model = model.eval()
        self.transform = model.time_transform
        self.mode = mode
        self.input_name = "latent" if mode == "decode" else "audio"
        self.output_name = "latent_out" if mode == "encode" else "audio_out"
        self.stream_batch = model.audio_channels
        if callback_samples <= 0 or callback_samples % self.transform.hop_size:
            raise ValueError("callback_samples must be a positive multiple of hop_size")
        self.frames_per_call = callback_samples // self.transform.hop_size

        spectral_bins = self.transform.nfft // 2 + 1
        skip = self.transform.skip_features
        if skip is not None:
            spectral_bins -= abs(skip)
        if spectral_bins != model.freq_size:
            raise ValueError(
                f"freq_size={model.freq_size}, but the time transform produces {spectral_bins} bins"
            )

        nfft = self.transform.nfft
        frequencies = nfft // 2 + 1
        samples = torch.arange(nfft, dtype=torch.float32)
        bins = torch.arange(frequencies, dtype=torch.float32).unsqueeze(1)
        phase = 2.0 * math.pi * bins * samples / nfft
        weights = torch.full((frequencies,), 2.0 / nfft)
        weights[0] = 1.0 / nfft
        if nfft % 2 == 0:
            weights[-1] = 1.0 / nfft
        self.register_buffer("analysis_cos", torch.cos(phase))
        self.register_buffer("analysis_sin", -torch.sin(phase))
        self.register_buffer("synthesis_cos", weights[:, None] * torch.cos(phase))
        self.register_buffer("synthesis_sin", -weights[:, None] * torch.sin(phase))

        self.cache_shapes: list[tuple[int, ...]] = []
        self.cache_names: list[str] = []
        if mode != "decode":
            self._add_cache(
                "analysis",
                (self.stream_batch, 1, nfft - self.transform.hop_size),
            )
        self._add_cache("attention_valid", (1,))
        if mode != "decode":
            self._collect_encoder_caches()
        if mode != "encode":
            self._collect_decoder_caches()
            if model.smoothing is not None:
                self._add_cache(
                    "smoothing",
                    (
                        self.stream_batch,
                        model.out_size,
                        model.freq_size,
                        model.smoothing.time_history,
                    ),
                )
            self._add_cache(
                "synthesis",
                (self.stream_batch, 1, self.transform.hop_size),
            )
        offsets = [0]
        for shape in self.cache_shapes:
            offsets.append(offsets[-1] + math.prod(shape))
        self.cache_offsets = offsets
        self.state_size = offsets[-1]

    def _add_cache(self, name: str, shape: tuple[int, ...]) -> None:
        self.cache_names.append(name)
        self.cache_shapes.append(shape)

    def _add_attention_cache(
        self, name: str, attention: TemporalAttention, leading: tuple[int, ...]
    ) -> None:
        shape = (
            *leading,
            attention.heads,
            attention.time_window - 1,
            attention.head_dim,
        )
        self._add_cache(name + ".k", shape)
        self._add_cache(name + ".v", shape)

    def _collect_encoder_caches(self) -> None:
        for stage_index, stage in enumerate(self.model.encoder_stages):
            for block_index, block in enumerate(stage):
                self._add_attention_cache(
                    f"encoder.{stage_index}.{block_index}",
                    block.time_attention,
                    (self.stream_batch, self.model.frequencies[stage_index]),
                )
        for index, block in enumerate(self.model.middle_encoder):
            self._add_attention_cache(
                f"middle_encoder.{index}", block.attention, (self.stream_batch,)
            )

    def _collect_decoder_caches(self) -> None:
        for index, block in enumerate(self.model.middle_decoder):
            self._add_attention_cache(
                f"middle_decoder.{index}", block.attention, (self.stream_batch,)
            )
        for stage_index in range(len(self.model.decoder_stages) - 1, -1, -1):
            for block_index, block in enumerate(self.model.decoder_stages[stage_index]):
                self._add_attention_cache(
                    f"decoder.{stage_index}.{block_index}",
                    block.time_attention,
                    (self.stream_batch, self.model.frequencies[stage_index]),
                )

    def initial_state(self, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(1, self.state_size, device=device, dtype=dtype)

    def _cache(
        self, state: torch.Tensor | tuple[torch.Tensor, ...], index: int
    ) -> torch.Tensor:
        if isinstance(state, tuple):
            return state[index]
        return state[:, self.cache_offsets[index] : self.cache_offsets[index + 1]].reshape(
            self.cache_shapes[index]
        )

    def _normalize(self, real: torch.Tensor, imag: torch.Tensor):
        magnitude = torch.sqrt(real * real + imag * imag).clamp_min(1e-8)
        scale = self.transform.beta_rescale * magnitude.pow(
            self.transform.alpha_rescale - 1.0
        )
        return real * scale, imag * scale

    def _denormalize(self, real: torch.Tensor, imag: torch.Tensor):
        magnitude = torch.sqrt(real * real + imag * imag).clamp_min(1e-8)
        target = (magnitude / self.transform.beta_rescale).pow(
            1.0 / self.transform.alpha_rescale
        )
        scale = target / magnitude
        return real * scale, imag * scale

    def _analysis(
        self, x: torch.Tensor, state: torch.Tensor, index: int, updates: list[torch.Tensor]
    ):
        history = self._cache(state, index)
        signal = torch.cat((history, x), dim=-1)
        updates.append(signal[..., -(self.transform.nfft - self.transform.hop_size) :])
        frames = torch.stack(
            [
                signal[..., start : start + self.transform.nfft]
                for start in range(
                    0,
                    self.frames_per_call * self.transform.hop_size,
                    self.transform.hop_size,
                )
            ],
            dim=-2,
        )
        frames = frames * self.transform.analysis_window
        real = torch.matmul(frames, self.analysis_cos.t()).transpose(-1, -2)
        imag = torch.matmul(frames, self.analysis_sin.t()).transpose(-1, -2)
        if self.transform.normalize:
            real, imag = self._normalize(real, imag)
        skip = self.transform.skip_features
        if skip is not None:
            if skip > 0:
                real, imag = real[:, :, skip:], imag[:, :, skip:]
            elif skip < 0:
                real, imag = real[:, :, :skip], imag[:, :, :skip]
        batch, channels, frequencies, frame_count = real.shape
        spectrum = torch.stack((real, imag), dim=2).reshape(
            batch, channels * 2, frequencies, frame_count
        )
        return spectrum, index + 1

    def _synthesis(
        self,
        spectrum: torch.Tensor,
        state: torch.Tensor,
        index: int,
        updates: list[torch.Tensor],
    ):
        batch, packed_channels, frequencies, frame_count = spectrum.shape
        channels = packed_channels // 2
        values = spectrum.reshape(batch, channels, 2, frequencies, frame_count)
        real, imag = values[:, :, 0], values[:, :, 1]
        skip = self.transform.skip_features
        if skip is not None:
            if skip > 0:
                real, imag = F.pad(real, (0, 0, skip, 0)), F.pad(imag, (0, 0, skip, 0))
            elif skip < 0:
                real, imag = F.pad(real, (0, 0, 0, -skip)), F.pad(imag, (0, 0, 0, -skip))
        if self.transform.normalize:
            real, imag = self._denormalize(real, imag)
        frames = torch.matmul(real.transpose(-1, -2), self.synthesis_cos)
        frames = frames + torch.matmul(imag.transpose(-1, -2), self.synthesis_sin)
        length = self.transform.synthesis_length
        support = frames[..., -length:] * self.transform.synthesis_window[-length:]
        hop = self.transform.hop_size
        first, second = support[..., :hop], support[..., hop:]
        pending = self._cache(state, index)
        if self.frames_per_call == 1:
            output = first + pending.unsqueeze(-2)
        else:
            output = torch.cat(
                (
                    first[..., :1, :] + pending.unsqueeze(-2),
                    first[..., 1:, :] + second[..., :-1, :],
                ),
                dim=-2,
            )
        updates.append(second[..., -1, :])
        return output.reshape(batch, channels, frame_count * hop), index + 1

    def _cached_axial(
        self,
        block: AxialBlock,
        x: torch.Tensor,
        state: torch.Tensor,
        index: int,
        valid: torch.Tensor,
        updates: list[torch.Tensor],
    ):
        k_cache, v_cache = self._cache(state, index), self._cache(state, index + 1)
        x, k_cache, v_cache = block.forward_with_cache(x, k_cache, v_cache, valid)
        updates.extend((k_cache, v_cache))
        return x, index + 2

    def _cached_temporal(
        self,
        block: TemporalBlock,
        x: torch.Tensor,
        state: torch.Tensor,
        index: int,
        valid: torch.Tensor,
        updates: list[torch.Tensor],
    ):
        k_cache, v_cache = self._cache(state, index), self._cache(state, index + 1)
        x, k_cache, v_cache = block.forward_with_cache(x, k_cache, v_cache, valid)
        updates.extend((k_cache, v_cache))
        return x, index + 2

    def _encode_cached(
        self,
        spectrum: torch.Tensor,
        state: torch.Tensor,
        index: int,
        valid: torch.Tensor,
        updates: list[torch.Tensor],
    ):
        x = self.model.input_projection(spectrum.permute(0, 3, 2, 1))
        for stage_index, stage in enumerate(self.model.encoder_stages):
            x = x + self.model.frequency_positions[stage_index]
            for block in stage:
                x, index = self._cached_axial(block, x, state, index, valid, updates)
            if stage_index < len(self.model.frequency_merges):
                x = self.model.frequency_merges[stage_index](x)
        batch, frames, frequencies, dim = x.shape
        x = self.model.middle_encode_projection(
            x.reshape(batch, frames, frequencies * dim)
        )
        for block in self.model.middle_encoder:
            x, index = self._cached_temporal(block, x, state, index, valid, updates)
        x = self.model.latent_projection(x).transpose(1, 2)
        return self._bottleneck(x), index

    def _decode_cached(
        self,
        latent: torch.Tensor,
        state: torch.Tensor,
        index: int,
        valid: torch.Tensor,
        updates: list[torch.Tensor],
    ):
        x = self.model.middle_decode_projection(latent.transpose(1, 2))
        for block in self.model.middle_decoder:
            x, index = self._cached_temporal(block, x, state, index, valid, updates)
        batch, frames, _ = x.shape
        x = self.model.spectral_projection(x).reshape(
            batch, frames, self.model.frequencies[-1], self.model.dims[-1]
        )
        for stage_index in range(len(self.model.decoder_stages) - 1, -1, -1):
            x = x + self.model.frequency_positions[stage_index]
            for block in self.model.decoder_stages[stage_index]:
                x, index = self._cached_axial(block, x, state, index, valid, updates)
            if stage_index:
                x = self.model.frequency_expands[stage_index - 1](x)
        spectrum = self.model.output_projection(x).permute(0, 3, 2, 1)
        if self.model.smoothing is not None:
            cache = self._cache(state, index)
            spectrum, cache = self.model.smoothing.forward_with_cache(spectrum, cache)
            updates.append(cache)
            index += 1
        return spectrum, index

    def _bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.model.bottleneck
        if isinstance(bottleneck, VAEBottleneck):
            return x.chunk(2, dim=1)[0]
        if isinstance(bottleneck, TanhBottleneck):
            return bottleneck.scale * torch.tanh(x)
        if isinstance(bottleneck, (ReluBottleneck, nn.Identity)):
            return x
        raise TypeError(f"unsupported export bottleneck: {type(bottleneck).__name__}")

    def forward_with_caches(
        self,
        x: torch.Tensor,
        state: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        updates: list[torch.Tensor] = []
        index = 0
        if self.mode != "decode":
            if self.model.audio_channels == 2:
                x = x.reshape(x.shape[0] * x.shape[1], 1, x.shape[-1])
            x, index = self._analysis(x, state, index, updates)
        valid = self._cache(state, index)
        index += 1
        updates.append(
            torch.clamp(valid + self.frames_per_call, max=self.model.time_window - 1)
        )
        if self.mode != "decode":
            x, index = self._encode_cached(x, state, index, valid, updates)
            if self.mode == "encode":
                return x, updates
        x, index = self._decode_cached(x, state, index, valid, updates)
        x, index = self._synthesis(x, state, index, updates)
        if self.model.audio_channels == 2:
            x = x.reshape(-1, self.model.audio_channels, x.shape[-1])
        return x, updates

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        x, updates = self.forward_with_caches(x, state)
        new_state = torch.cat([value.reshape(1, -1) for value in updates], dim=1)
        return x, new_state


def print_shape_summary(model: RofNet, frames: int = 1) -> None:
    """Print the static hierarchy for a compact architecture sanity check."""
    rows = [("STFT", f"(B, {model.in_size}, {model.freq_size}, {frames})")]
    for index, (frequency, dim, depth) in enumerate(
        zip(model.frequencies, model.dims, model.depths)
    ):
        rows.append((f"encoder stage {index} x{depth}", f"(B, {frames}, {frequency}, {dim})"))
    rows.extend(
        [
            ("middle", f"(B, {frames}, {model.middle_dim})"),
            ("latent", f"(B, {model.bottleneck_size}, {frames})"),
        ]
    )
    for index in range(len(model.dims) - 1, -1, -1):
        rows.append(
            (
                f"decoder stage {index} x{model.depths[index]}",
                f"(B, {frames}, {model.frequencies[index]}, {model.dims[index]})",
            )
        )
    rows.append(("ISTFT", f"(B, {model.audio_channels}, {frames * model.time_transform.hop_size})"))
    width = max(len(name) for name, _ in rows)
    print("RofNet shape summary")
    for name, shape in rows:
        print(f"  {name:<{width}}  {shape}")
