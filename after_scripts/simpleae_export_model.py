"""Portable, stateless streaming wrapper for ``AutoEncoder2D``.

The wrapper uses real-valued DFT matrices and carries every streaming cache in
one flat tensor.  Its only contract is::

    audio, new_state = model(audio, state)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from after.autoencoder.networks.bottlenecks import (
    ReluBottleneck,
    TanhBottleneck,
    VAEBottleneck,
)


def remove_weight_norm(module: nn.Module) -> None:
    """Resolve legacy weight normalization before graph export."""
    for child in module.modules():
        if hasattr(child, "weight_g") and hasattr(child, "weight_v"):
            torch.nn.utils.remove_weight_norm(child)


class StatelessStreamingSimpleAE(nn.Module):
    """Functional streaming view of an initialized ``AutoEncoder2D``."""

    def __init__(
        self,
        model: nn.Module,
        callback_samples: int,
        mode: str = "forward",
    ):
        super().__init__()
        if mode not in ("encode", "decode", "forward"):
            raise ValueError(f"Unknown model mode: {mode}")
        self.model = model.eval()
        self.transform = model.time_transform
        self.stream_batch = model.audio_channels
        self.mode = mode
        self.input_name = "latent" if mode == "decode" else "audio"
        self.output_name = "latent_out" if mode == "encode" else "audio_out"
        if callback_samples % self.transform.hop_size:
            raise ValueError("callback_samples must be a multiple of hop_size")
        self.frames_per_call = callback_samples // self.transform.hop_size

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
            self._collect_encoder_caches()
        if mode != "encode":
            self._collect_decoder_caches()
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
        if math.prod(shape) > 0:
            self.cache_names.append(name)
            self.cache_shapes.append(shape)

    def _add_padding_caches(self, name: str, module: nn.Module) -> None:
        delay = module.downsampling_delay
        if delay.padding:
            self._add_cache(
                f"{name}.delay", tuple(delay.pad[:self.stream_batch].shape)
            )
        cache = module.cache
        if cache.padding:
            self._add_cache(
                f"{name}.cache", tuple(cache.pad[:self.stream_batch].shape)
            )

    def _add_conv_transpose_cache(self, name: str, module: nn.Module) -> None:
        if module.use_cache and module.time_pad:
            self._add_cache(
                f"{name}.cache", tuple(module.cache[:self.stream_batch].shape)
            )

    def _add_encoder_block_caches(self, name: str, block: nn.Module) -> None:
        self._add_padding_caches(f"{name}.conv", block.conv)
        self._add_padding_caches(f"{name}.pool", block.pool)
        self._add_padding_caches(f"{name}.proj", block.proj)

    def _add_decoder_block_caches(self, name: str, block: nn.Module) -> None:
        self._add_padding_caches(f"{name}.conv", block.conv)
        self._add_conv_transpose_cache(f"{name}.up", block.up)
        self._add_padding_caches(f"{name}.proj", block.proj)

    def _collect_encoder_caches(self) -> None:
        self._add_padding_caches("preconv", self.model.preconv)
        for index, block in enumerate(self.model.down_layers):
            self._add_encoder_block_caches(f"down.{index}", block)
        if self.model.audio_channels == 2:
            self._add_encoder_block_caches("stereo_merge", self.model.stereo_merge)
        self._add_padding_caches(
            "middle_encode", self.model.middle_block_encode.project
        )

    def _collect_decoder_caches(self) -> None:
        self._add_padding_caches(
            "middle_decode", self.model.middle_block_decode.project
        )
        if self.model.audio_channels == 2:
            self._add_decoder_block_caches("stereo_split", self.model.stereo_split)
        for index, block in enumerate(self.model.up_layers):
            self._add_decoder_block_caches(f"up.{index}", block)
        self._add_padding_caches("outconv", self.model.outconv)

    def initial_state(self, *, device=None, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(1, self.state_size, device=device, dtype=dtype)

    def initial_cache_states(
        self, *, device=None, dtype=torch.float32
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.zeros(shape, device=device, dtype=dtype)
            for shape in self.cache_shapes
        )

    def _cache(
        self,
        state: torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        start = self.cache_offsets[index]
        end = self.cache_offsets[index + 1]
        return state[:, start:end].reshape(self.cache_shapes[index])

    def _cached_padding(
        self,
        x: torch.Tensor,
        module: nn.Module,
        state: torch.Tensor,
        index: int,
        updates: list[torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        padding = module.padding
        if not padding:
            return x, index
        x = torch.cat((self._cache(state, index), x), dim=-1)
        updates.append(x[..., -padding:])
        if module.crop:
            x = x[..., :-padding]
        return x, index + 1

    def _conv2d(
        self,
        x: torch.Tensor,
        module: nn.Module,
        state: torch.Tensor,
        index: int,
        updates: list[torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        x, index = self._cached_padding(
            x, module.downsampling_delay, state, index, updates
        )
        x, index = self._cached_padding(x, module.cache, state, index, updates)
        conv = module.causal_conv
        vertical = conv.padding_vert
        x = F.pad(x, (0, 0, vertical[0], vertical[1]))
        x = F.conv2d(
            x,
            conv.weight,
            conv.bias,
            conv.stride,
            0,
            conv.dilation,
            conv.groups,
        )
        return x, index

    def _conv1d(
        self,
        x: torch.Tensor,
        module: nn.Module,
        state: torch.Tensor,
        index: int,
        updates: list[torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        x, index = self._cached_padding(
            x, module.downsampling_delay, state, index, updates
        )
        x, index = self._cached_padding(x, module.cache, state, index, updates)
        return (
            F.conv1d(
                x,
                module.weight,
                module.bias,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
            ),
            index,
        )

    def _conv_transpose2d(
        self,
        x: torch.Tensor,
        module: nn.Module,
        state: torch.Tensor,
        index: int,
        updates: list[torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        x = F.conv_transpose2d(
            x,
            module.weight,
            None,
            module.stride,
            module._padding,
            module.output_padding,
            module.groups,
            module.dilation,
        )
        padding = 2 * module.time_pad
        if module.use_cache and padding:
            cached = self._cache(state, index)
            x = torch.cat((x[..., :padding] + cached, x[..., padding:]), dim=-1)
            updates.append(x[..., -padding:])
            index += 1
        if padding:
            x = x[..., :-padding]
        if module.bias is not None:
            x = x + module.bias[None, :, None, None]
        return x, index

    def _encoder_block(
        self,
        x: torch.Tensor,
        block: nn.Module,
        state: torch.Tensor,
        index: int,
        updates: list[torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        residual = x
        x, index = self._conv2d(block.act(x), block.conv, state, index, updates)
        x, index = self._conv2d(block.act(x), block.pool, state, index, updates)
        residual = block.proj_pool(residual)
        residual, index = self._conv2d(
            residual, block.proj, state, index, updates
        )
        return x + residual, index

    def _decoder_block(
        self,
        x: torch.Tensor,
        block: nn.Module,
        state: torch.Tensor,
        index: int,
        updates: list[torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        residual = x
        x, index = self._conv2d(block.act(x), block.conv, state, index, updates)
        x, index = self._conv_transpose2d(
            block.act(x), block.up, state, index, updates
        )
        residual = block.proj_pool(residual)
        residual, index = self._conv2d(
            residual, block.proj, state, index, updates
        )
        return x + residual, index

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
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        updates: list[torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        history = self._cache(state, 0)
        signal = torch.cat((history, x), dim=-1)
        updates.append(signal[..., -(self.transform.nfft - self.transform.hop_size):])
        frames = torch.stack(
            [
                signal[..., start:start + self.transform.nfft]
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
        batch, channels, frequencies, frames_count = real.shape
        packed = torch.stack((real, imag), dim=2).reshape(
            batch, channels * 2, frequencies, frames_count
        )
        return packed, 1

    def _synthesis(
        self,
        packed: torch.Tensor,
        state: torch.Tensor,
        index: int,
        updates: list[torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        batch, packed_channels, frequencies, frames_count = packed.shape
        channels = packed_channels // 2
        spec = packed.reshape(batch, channels, 2, frequencies, frames_count)
        real, imag = spec[:, :, 0], spec[:, :, 1]
        skip = self.transform.skip_features
        if skip is not None:
            if skip > 0:
                real = F.pad(real, (0, 0, skip, 0))
                imag = F.pad(imag, (0, 0, skip, 0))
            elif skip < 0:
                real = F.pad(real, (0, 0, 0, -skip))
                imag = F.pad(imag, (0, 0, 0, -skip))
        if self.transform.normalize:
            real, imag = self._denormalize(real, imag)
        real = real.transpose(-1, -2)
        imag = imag.transpose(-1, -2)
        frames = torch.matmul(real, self.synthesis_cos) + torch.matmul(
            imag, self.synthesis_sin
        )
        length = self.transform.synthesis_length
        support = frames[..., -length:] * self.transform.synthesis_window[-length:]
        hop = self.transform.hop_size
        first, second = support[..., :hop], support[..., hop:]
        pending = self._cache(state, index)
        if self.frames_per_call == 1:
            output = first + pending.unsqueeze(-2)
        else:
            first_frame = first[..., :1, :] + pending.unsqueeze(-2)
            remaining = first[..., 1:, :] + second[..., :-1, :]
            output = torch.cat((first_frame, remaining), dim=-2)
        updates.append(second[..., -1, :])
        return output.reshape(batch, channels, frames_count * hop), index + 1

    def _bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.model.bottleneck
        if isinstance(bottleneck, VAEBottleneck):
            return x.chunk(2, dim=1)[0]
        if isinstance(bottleneck, TanhBottleneck):
            return bottleneck.scale * torch.tanh(x)
        if isinstance(bottleneck, ReluBottleneck) or isinstance(
            bottleneck, nn.Identity
        ):
            return x
        raise TypeError(f"Unsupported export bottleneck: {type(bottleneck).__name__}")

    def forward_with_caches(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        updates: list[torch.Tensor] = []
        if self.mode == "decode":
            batch = x.shape[0]
            index = 0
        else:
            if self.model.audio_channels == 2:
                batch, channels, samples = x.shape
                x = x.reshape(batch * channels, 1, samples)

            x, index = self._analysis(x, state, updates)
            x, index = self._conv2d(x, self.model.preconv, state, index, updates)
            for block in self.model.down_layers:
                x, index = self._encoder_block(x, block, state, index, updates)

            if self.model.audio_channels == 2:
                batch_channels, channels, frequencies, frames = x.shape
                batch = batch_channels // 2
                x = x.reshape(batch, 2 * channels, frequencies, frames)
                x, index = self._encoder_block(
                    x, self.model.stereo_merge, state, index, updates
                )

            batch, channels, frequencies, frames = x.shape
            x = x.reshape(batch, channels * frequencies, frames)
            x, index = self._conv1d(
                self.model.middle_block_encode.activation(x),
                self.model.middle_block_encode.project,
                state,
                index,
                updates,
            )
            x = self._bottleneck(x)

            if self.mode == "encode":
                return x, updates
        x, index = self._conv1d(
            self.model.middle_block_decode.activation(x),
            self.model.middle_block_decode.project,
            state,
            index,
            updates,
        )
        x = x.reshape(
            batch,
            x.shape[1] // self.model.freq_final_dim,
            self.model.freq_final_dim,
            x.shape[-1],
        )

        if self.model.audio_channels == 2:
            x, index = self._decoder_block(
                x, self.model.stereo_split, state, index, updates
            )
            batch, channels, frequencies, frames = x.shape
            x = x.reshape(batch * 2, channels // 2, frequencies, frames)

        for block in self.model.up_layers:
            x, index = self._decoder_block(x, block, state, index, updates)
        x, index = self._conv2d(x, self.model.outconv, state, index, updates)
        x, index = self._synthesis(x, state, index, updates)

        if self.model.audio_channels == 2:
            x = x.reshape(-1, 2, x.shape[-1])
        return x, updates

    def forward(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, updates = self.forward_with_caches(x, state)
        new_state = torch.cat([value.reshape(1, -1) for value in updates], dim=1)
        return x, new_state


class CoreMLStatefulSimpleAE(StatelessStreamingSimpleAE):
    """Core ML export view with one model state per streaming cache."""

    def __init__(self, model: StatelessStreamingSimpleAE):
        callback_samples = model.frames_per_call * model.transform.hop_size
        super().__init__(model.model, callback_samples, model.mode)
        self.state_names = [
            f"cache_{index:02d}_{name.replace('.', '_')}"
            for index, name in enumerate(self.cache_names)
        ]
        for name, value in zip(
            self.state_names,
            self.initial_cache_states(dtype=torch.float16),
        ):
            self.register_buffer(name, value)

    def _cache(
        self,
        state: tuple[torch.Tensor, ...],
        index: int,
    ) -> torch.Tensor:
        return state[index]

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        caches = tuple(getattr(self, name).float() for name in self.state_names)
        output, next_caches = self.forward_with_caches(audio, caches)
        for name, value in zip(self.state_names, next_caches):
            getattr(self, name).copy_(value.to(torch.float16))
        return output


class CoreMLStatefulExport(nn.Module):
    """Generic Core ML state adapter for a portable per-cache model."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.state_names = [
            f"cache_{index:02d}_{name.replace('.', '_')}"
            for index, name in enumerate(model.cache_names)
        ]
        for name, shape in zip(self.state_names, model.cache_shapes):
            self.register_buffer(name, torch.zeros(shape, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        caches = tuple(getattr(self, name).float() for name in self.state_names)
        output, next_caches = self.model.forward_with_caches(x, caches)
        for name, value in zip(self.state_names, next_caches):
            getattr(self, name).copy_(value.to(torch.float16))
        return output
