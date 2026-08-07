"""Double-encoder 2D spectral autoencoder."""
import math
from typing import Optional

import gin
import torch
import torch.nn as nn
import cached_conv as cc
from einops.layers.torch import Rearrange

from .SimpleNet2D import (Conv2d, EncoderBlock2D, DecoderBlock2D,
                          ConvBlock1d)


def _regloss_like(regloss, reference):
    if torch.is_tensor(regloss):
        return regloss.to(device=reference.device, dtype=reference.dtype)
    return reference.new_tensor(regloss)


@gin.configurable
class Encoder2D(nn.Module):
    """Encoder half of the 2D spectral autoencoder."""

    def __init__(self,
                 in_size: int = 2,
                 bottleneck_size: int = 0,
                 audio_channels: int = 1,
                 channels=None,
                 time_ratios=None,
                 freq_ratios=None,
                 freq_size: int = 1024,
                 kernel_size: int = 3,
                 bottleneck=None,
                 time_transform=None,
                 use_vae: bool = False):
        super().__init__()

        if channels is None:
            channels = [128, 128, 128, 128]
        if time_ratios is None:
            time_ratios = [1, 1, 1, 2, 1]
        if freq_ratios is None:
            freq_ratios = [2, 2, 2, 2, 1]
        if bottleneck is None:
            bottleneck = nn.Identity()

        self.time_transform = time_transform
        self.channels = channels
        self.bottleneck = bottleneck
        self.in_size = in_size
        self.bottleneck_size = bottleneck_size
        self.audio_channels = audio_channels

        self.pack_audio = (Rearrange('b ch t -> (b ch) 1 t',
                                     ch=audio_channels)
                           if audio_channels == 2 else nn.Identity())
        self.merge_stereo_features = (
            Rearrange('(b ch) c f t -> b (ch c) f t', ch=audio_channels)
            if audio_channels == 2 else nn.Identity())

        n = len(channels)
        self.down_layers = nn.ModuleList()

        self.preconv = Conv2d(in_size,
                              channels[0],
                              kernel_size=7,
                              padding_vert=(7 - 1) // 2,
                              padding_time=cc.get_padding(kernel_size=7,
                                                          stride=1,
                                                          mode="causal"))

        for i in range(n - 1):
            self.down_layers.append(
                EncoderBlock2D(in_c=channels[i],
                               out_c=channels[i + 1],
                               kernel_size=kernel_size,
                               ratio=time_ratios[i],
                               freq_ratio=freq_ratios[i]))

        self.stereo_merge = (EncoderBlock2D(in_c=2 * channels[-1],
                                            out_c=channels[-1],
                                            kernel_size=kernel_size,
                                            ratio=1,
                                            freq_ratio=1)
                             if audio_channels == 2 else nn.Identity())

        freq_total_ratio = math.prod(freq_ratios)
        self.freq_final_dim = freq_size // freq_total_ratio

        self.middle_block_encode = ConvBlock1d(
            in_channels=self.freq_final_dim * channels[-1],
            out_channels=bottleneck_size * 2 if use_vae else bottleneck_size,
            kernel_size=3)
        self.rearrange_encode = Rearrange('b c f t -> b (c f) t')

        self.encoder = nn.ModuleList([
            self.preconv, self.down_layers, self.stereo_merge,
            self.middle_block_encode, self.bottleneck
        ])

    def _encode_features(self, h):
        h = self.preconv(h)
        for layer in self.down_layers:
            h = layer(h)
        h = self.merge_stereo_features(h)
        h = self.stereo_merge(h)
        h = self.rearrange_encode(h)
        h = self.middle_block_encode(h)
        return h

    def _apply_bottleneck(self, h, return_mean: bool = False):
        if return_mean:
            try:
                out = self.bottleneck(h, return_mean=True)
            except TypeError:
                out = self.bottleneck(h)
        else:
            out = self.bottleneck(h)

        if not isinstance(out, tuple):
            return out, h.new_tensor(0.)
        if len(out) == 3:
            z, regloss, mean = out
            return z, _regloss_like(regloss, z), mean
        z, regloss = out
        return z, _regloss_like(regloss, z)

    def _encode_with_multi(self, x):
        x = self.pack_audio(x)
        x_multiband = self.time_transform(x)
        h = self._encode_features(x_multiband.clone())
        z, regloss = self._apply_bottleneck(h)
        return z, regloss, x_multiband

    @torch.jit.ignore
    def encode(self, x, with_multi: bool = False, return_mean: bool = False):
        if with_multi and not return_mean:
            z, _, x_multiband = self._encode_with_multi(x)
            return z, x_multiband

        x = self.pack_audio(x)
        x_multiband = self.time_transform(x)
        h = self._encode_features(x_multiband.clone())
        out = self._apply_bottleneck(h, return_mean=return_mean)

        if return_mean:
            z, regloss, mean = out
            if with_multi:
                return z, x_multiband, mean
            return z, regloss, mean

        z, regloss = out
        if with_multi:
            return z, x_multiband
        return z, regloss


@gin.configurable
class Decoder2D(nn.Module):
    """Decoder half of the 2D spectral autoencoder."""

    def __init__(self,
                 in_size: int = 2,
                 out_size=None,
                 bottleneck_size: int = 0,
                 audio_channels: int = 1,
                 channels=None,
                 time_ratios=None,
                 freq_ratios=None,
                 freq_size: int = 1024,
                 kernel_size: int = 3,
                 time_transform=None,
                 side_channels: int = 0,
                 fusion_after_layers: int = 0):
        super().__init__()

        if channels is None:
            channels = [128, 128, 128, 128]
        if time_ratios is None:
            time_ratios = [1, 1, 1, 2, 1]
        if freq_ratios is None:
            freq_ratios = [2, 2, 2, 2, 1]

        self.time_transform = time_transform
        self.channels = channels
        self.in_size = in_size
        self.bottleneck_size = bottleneck_size
        self.audio_channels = audio_channels
        self.side_channels = side_channels
        self.fusion_after_layers = fusion_after_layers
        out_size = in_size if out_size is None else out_size

        self.unpack_audio = (Rearrange('(b ch) 1 t -> b ch t',
                                       ch=audio_channels)
                             if audio_channels == 2 else nn.Identity())
        self.split_stereo_features = (
            Rearrange('b (ch c) f t -> (b ch) c f t', ch=audio_channels)
            if audio_channels == 2 else nn.Identity())

        n = len(channels)
        self.up_layers = nn.ModuleList()

        freq_total_ratio = math.prod(freq_ratios)
        self.freq_final_dim = freq_size // freq_total_ratio

        self.middle_block_decode = ConvBlock1d(
            in_channels=bottleneck_size,
            out_channels=2 * self.freq_final_dim * channels[-1],
            kernel_size=3)
        self.rearrange_decode = Rearrange('b (c f) t -> b c f t',
                                          f=self.freq_final_dim)

        channels_dec = [2 * c for c in channels]

        for i in range(1, n):
            layer_index = i - 1
            in_channels = channels_dec[n - i]
            if (side_channels > 0
                    and layer_index == fusion_after_layers):
                in_channels += side_channels
            self.up_layers.append(
                DecoderBlock2D(in_c=in_channels,
                               out_c=channels_dec[n - i - 1],
                               kernel_size=kernel_size,
                               ratio=time_ratios[n - i],
                               freq_ratio=freq_ratios[n - i]))

        if not 0 <= fusion_after_layers <= len(self.up_layers):
            raise ValueError(
                "fusion_after_layers must be between 0 and the number of "
                "decoder layers")

        self.stereo_split = (DecoderBlock2D(in_c=channels_dec[-1],
                                            out_c=2 * channels_dec[-1],
                                            kernel_size=kernel_size,
                                            ratio=1,
                                            freq_ratio=1)
                             if audio_channels == 2 else nn.Identity())

        outconv_channels = 2 * channels[0]
        if (side_channels > 0
                and fusion_after_layers == len(self.up_layers)):
            outconv_channels += side_channels
        self.outconv = Conv2d(outconv_channels,
                              out_size,
                              kernel_size=7,
                              padding_vert=(7 - 1) // 2,
                              padding_time=cc.get_padding(kernel_size=7,
                                                          stride=1,
                                                          mode="causal"))

        self.decoder = nn.ModuleList([
            self.middle_block_decode, self.stereo_split, self.up_layers,
            self.outconv
        ])

    def _merge_side(self,
                    h: torch.Tensor,
                    side: Optional[torch.Tensor],
                    drop_fast_mask: Optional[torch.Tensor] = None
                    ) -> torch.Tensor:
        if side is None:
            raise ValueError("This decoder requires a slow feature map")
        if h.shape[0] != side.shape[0] or h.shape[2] != side.shape[2]:
            raise ValueError(
                "Fast and slow decoder maps have incompatible batch or "
                "frequency dimensions")
        if side.shape[-1] < h.shape[-1]:
            raise ValueError(
                "Slow decoder map is shorter than the fast decoder map")

        if drop_fast_mask is not None:
            if h.shape[0] % drop_fast_mask.shape[0] != 0:
                raise ValueError(
                    "Fast dropout mask has an incompatible batch size")
            repeat = h.shape[0] // drop_fast_mask.shape[0]
            if repeat > 1:
                drop_fast_mask = drop_fast_mask.repeat_interleave(repeat,
                                                                   dim=0)
            h = torch.where(drop_fast_mask, torch.zeros_like(h), h)

        side = side[..., :h.shape[-1]]
        return torch.cat((h, side), dim=1)

    def _decode_features(
            self,
            h: torch.Tensor,
            side: Optional[torch.Tensor] = None,
            drop_fast_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.middle_block_decode(h)
        h = self.rearrange_decode(h)
        h = self.stereo_split(h)
        h = self.split_stereo_features(h)

        if self.side_channels > 0 and self.fusion_after_layers == 0:
            h = self._merge_side(h, side, drop_fast_mask=drop_fast_mask)

        for i, layer in enumerate(self.up_layers):
            h = layer(h)
            if (self.side_channels > 0
                    and i + 1 == self.fusion_after_layers):
                h = self._merge_side(h,
                                     side,
                                     drop_fast_mask=drop_fast_mask)
        h = self.outconv(h)
        return h

    @torch.jit.ignore
    def decode(self,
               z,
               with_multi: bool = False,
               side=None,
               drop_fast_mask=None):
        h = self._decode_features(z,
                                  side=side,
                                  drop_fast_mask=drop_fast_mask)
        y = self.time_transform.inverse(h)
        y = self.unpack_audio(y)
        if with_multi:
            return y, h
        return y


@gin.configurable
class SlowMapDecoder(nn.Module):
    """Decode slow codes into a 2D map for the fast decoder."""

    def __init__(self,
                 bottleneck_size: int,
                 audio_channels: int,
                 channels,
                 time_ratios,
                 freq_ratios,
                 freq_size: int,
                 upsample_layers: int = None,
                 output_channels: int = None,
                 extra_time_ratios=None,
                 decoder_time_ratios=None,
                 decoder_freq_ratios=None,
                 kernel_size: int = 3):
        super().__init__()
        if extra_time_ratios is None:
            extra_time_ratios = []

        explicit_ratios = (decoder_time_ratios is not None
                           or decoder_freq_ratios is not None)
        if explicit_ratios:
            if decoder_time_ratios is None or decoder_freq_ratios is None:
                raise ValueError(
                    "decoder_time_ratios and decoder_freq_ratios must both be "
                    "provided")
            if len(decoder_time_ratios) != len(decoder_freq_ratios):
                raise ValueError(
                    "decoder time and frequency ratios must have equal length")
            if not 0 < len(decoder_time_ratios) < len(channels):
                raise ValueError(
                    "decoder ratio count must be smaller than the channel count")
            if extra_time_ratios:
                raise ValueError(
                    "extra_time_ratios cannot be combined with explicit decoder "
                    "ratios")
            layer_count = len(decoder_time_ratios)
        else:
            if upsample_layers is None:
                raise ValueError(
                    "upsample_layers is required without explicit decoder ratios")
            if not 0 <= upsample_layers < len(channels):
                raise ValueError(
                    "upsample_layers must be smaller than the channel count")
            layer_count = upsample_layers

        channels_dec = (list(channels) if explicit_ratios else
                        [2 * c for c in channels])
        freq_final_dim = freq_size // math.prod(freq_ratios)
        self.middle_block_decode = ConvBlock1d(
            in_channels=bottleneck_size,
            out_channels=freq_final_dim * channels_dec[-1],
            kernel_size=kernel_size)
        self.rearrange_decode = Rearrange('b (c f) t -> b c f t',
                                          f=freq_final_dim)

        self.stereo_split = (DecoderBlock2D(in_c=channels_dec[-1],
                                            out_c=2 * channels_dec[-1],
                                            kernel_size=kernel_size,
                                            ratio=1,
                                            freq_ratio=1)
                             if audio_channels == 2 else nn.Identity())
        self.split_stereo_features = (
            Rearrange('b (ch c) f t -> (b ch) c f t', ch=audio_channels)
            if audio_channels == 2 else nn.Identity())

        self.up_layers = nn.ModuleList()
        n = len(channels)
        for i in range(1, layer_count + 1):
            if explicit_ratios:
                time_ratio = decoder_time_ratios[i - 1]
                freq_ratio = decoder_freq_ratios[i - 1]
            else:
                time_ratio = time_ratios[n - i]
                freq_ratio = freq_ratios[n - i]
            self.up_layers.append(
                DecoderBlock2D(in_c=channels_dec[n - i],
                               out_c=channels_dec[n - i - 1],
                               kernel_size=kernel_size,
                               ratio=time_ratio,
                               freq_ratio=freq_ratio))

        natural_output_channels = channels_dec[n - layer_count - 1]
        output_channels = (natural_output_channels
                           if output_channels is None else output_channels)
        self.output_channels = output_channels
        self.output_proj = (Conv2d(natural_output_channels,
                                   output_channels,
                                   kernel_size=1,
                                   padding=0)
                            if output_channels != natural_output_channels else
                            nn.Identity())
        self.time_layers = nn.ModuleList([
            DecoderBlock2D(in_c=output_channels,
                           out_c=output_channels,
                           kernel_size=kernel_size,
                           ratio=ratio,
                           freq_ratio=1) for ratio in extra_time_ratios
        ])
        if explicit_ratios:
            self.total_time_ratio = math.prod(decoder_time_ratios)
        else:
            self.total_time_ratio = math.prod(
                time_ratios[len(channels) - layer_count:]) * math.prod(
                    extra_time_ratios)

    def forward(self, z_slow):
        x = self.middle_block_decode(z_slow)
        x = self.rearrange_decode(x)
        x = self.stereo_split(x)
        x = self.split_stereo_features(x)
        for layer in self.up_layers:
            x = layer(x)
        x = self.output_proj(x)
        for layer in self.time_layers:
            x = layer(x)
        return x


@gin.configurable
class DoubleAE(nn.Module):
    """Autoencoder with fast and slow encoders feeding one decoder."""

    def __init__(self,
                 fast_encoder: nn.Module,
                 slow_encoder: nn.Module,
                 decoder: nn.Module,
                 slow_decoder: nn.Module = None,
                 slow_shift_steps: int = 1,
                 regularisation_ratio: float = 1.,
                 freeze_mode: str = "both",
                 drop_fast_probability: float = 0.):
        super().__init__()
        if not 0. <= drop_fast_probability <= 1.:
            raise ValueError("drop_fast_probability must be between 0 and 1")
        self.fast_encoder = fast_encoder
        self.slow_encoder = slow_encoder
        self.decoder = decoder
        self.slow_decoder = slow_decoder
        self.slow_shift_steps = slow_shift_steps
        
        self.regloss_ratio = regularisation_ratio
        self.freeze_mode = freeze_mode
        self.drop_fast_probability = drop_fast_probability

        self.encoder = nn.ModuleList([self.fast_encoder, self.slow_encoder])

    def _shift_slow_to_past(self, z_slow):
        if self.slow_shift_steps <= 0:
            return z_slow
        if self.slow_shift_steps >= z_slow.shape[-1]:
            return torch.zeros_like(z_slow)
        pad = torch.zeros_like(z_slow[..., :self.slow_shift_steps])
        return torch.cat((pad, z_slow[..., :-self.slow_shift_steps]), dim=-1)

    def _match_slow_to_fast(self, z_slow, fast_steps: int):
        repeat = math.ceil(fast_steps / z_slow.shape[-1])
        z_slow = z_slow.repeat_interleave(repeat, dim=-1)
        return z_slow[..., :fast_steps]

    def _decode_slow_map(self, z_slow, fast_steps: int):
        if self.slow_decoder is None:
            raise ValueError("No slow map decoder is configured")
        z_slow = self._shift_slow_to_past(z_slow)
        slow_map = self.slow_decoder(z_slow)
        if slow_map.shape[-1] < fast_steps:
            raise ValueError(
                "Slow decoder map is shorter than the fast code: "
                f"{slow_map.shape[-1]} < {fast_steps}")
        return slow_map[..., :fast_steps]

    def _combine_latents(self, z_fast, z_slow):
        z_slow = self._shift_slow_to_past(z_slow)
        z_slow = self._match_slow_to_fast(z_slow, z_fast.shape[-1])
        return torch.cat((z_fast, z_slow), dim=1)

    @torch.jit.ignore
    def combine_codes(self, z_fast, z_slow):
        """Pack raw fast and slow codes for the configured decoder layout."""
        if self.slow_decoder is not None:
            return z_fast, z_slow
        return self._combine_latents(z_fast, z_slow)

    @torch.jit.ignore
    def split_codes(self, z):
        """Split a combined decoder latent into fast and aligned slow parts."""
        if isinstance(z, (tuple, list)):
            return z
        fast_size = self.fast_encoder.bottleneck_size
        return z[:, :fast_size], z[:, fast_size:]

    def _encode_branches(self, x, freeze_encoder: bool = False):
        if freeze_encoder and self.freeze_mode in ["both", "fast"]:
            with torch.no_grad():
                z_fast, fast_regloss, fast_multiband = self.fast_encoder._encode_with_multi(
                    x)
        else:
             z_fast, fast_regloss, fast_multiband = self.fast_encoder._encode_with_multi(
                    x)
             
        if freeze_encoder and self.freeze_mode in ["both", "slow"]:
            with torch.no_grad():   
                z_slow, slow_regloss, _ = self.slow_encoder._encode_with_multi(x)
        else:
            z_slow, slow_regloss, _ = self.slow_encoder._encode_with_multi(x)

        fast_regloss *= self.regloss_ratio
        regloss = fast_regloss + slow_regloss
        return z_fast, z_slow, regloss, fast_multiband

    def _encode_combined(self,
                         x,
                         with_multi: bool = False,
                         freeze_encoder: bool = False):
        z_fast, z_slow, regloss, fast_multiband = self._encode_branches(
            x, freeze_encoder=freeze_encoder)
        z = self.combine_codes(z_fast, z_slow)

        if with_multi:
            return z, regloss, fast_multiband

        return z, regloss

    @torch.jit.ignore
    def encode_fast(self,
                    x,
                    with_multi: bool = False,
                    return_mean: bool = False):
        """Encode audio with the fast branch only."""
        return self.fast_encoder.encode(x,
                                        with_multi=with_multi,
                                        return_mean=return_mean)

    @torch.jit.ignore
    def encode_slow(self,
                    x,
                    with_multi: bool = False,
                    return_mean: bool = False):
        """Encode audio with the slow branch only."""
        return self.slow_encoder.encode(x,
                                        with_multi=with_multi,
                                        return_mean=return_mean)

    @torch.jit.ignore
    def encode_codes(self, x, with_multi: bool = False):
        """Return raw fast and slow branch codes without concatenating them."""
        z_fast, fast_regloss, fast_multiband = self.fast_encoder._encode_with_multi(
            x)
        z_slow, slow_regloss, slow_multiband = self.slow_encoder._encode_with_multi(
            x)
        regloss = fast_regloss + slow_regloss

        if with_multi:
            return z_fast, z_slow, regloss, fast_multiband, slow_multiband
        return z_fast, z_slow, regloss

    @torch.jit.ignore
    def decode_codes(self, z_fast, z_slow, with_multi: bool = False):
        """Decode from raw fast and slow branch codes."""
        return self.decode((z_fast, z_slow), with_multi=with_multi)

    @torch.jit.ignore
    def forward(self,
                x,
                return_all: bool = True,
                freeze_encoder: bool = False,
                look_ahead_steps: int = 0,
                apply_branch_dropout: bool = False):
        z, regloss, x_multiband = self._encode_combined(
            x, with_multi=True, freeze_encoder=freeze_encoder)

        if self.slow_decoder is not None:
            z_fast, z_slow = z
            side = self._decode_slow_map(z_slow, z_fast.shape[-1])
            drop_fast_mask = None
            if apply_branch_dropout and self.drop_fast_probability > 0.:
                drop_fast_mask = torch.rand(
                    x.shape[0], 1, 1, 1,
                    device=x.device) < self.drop_fast_probability

            if look_ahead_steps > 0:
                z_fast = z_fast[..., look_ahead_steps:]
                z_fast = torch.cat(
                    (z_fast,
                     torch.zeros_like(z_fast[..., :look_ahead_steps])),
                    dim=-1)
                side = side[..., look_ahead_steps:]
                side = torch.cat(
                    (side, torch.zeros_like(side[..., :look_ahead_steps])),
                    dim=-1)

            y, y_multiband = self.decoder.decode(z_fast,
                                                  with_multi=True,
                                                  side=side,
                                                  drop_fast_mask=drop_fast_mask)
            z = z_fast
        else:
            if look_ahead_steps > 0:
                z = z[..., look_ahead_steps:]
                z = torch.cat(
                    (z, torch.zeros_like(z[..., :look_ahead_steps])), dim=-1)

            y, y_multiband = self.decoder.decode(z, with_multi=True)

        if return_all:
            return y, y_multiband, z, regloss, x_multiband
        return y

    @torch.jit.ignore
    def encode(self, x, with_multi: bool = False, return_mean: bool = False):
        if return_mean:
            raise NotImplementedError(
                "return_mean is not implemented for combined DoubleAE latents")
        if with_multi:
            z, _, x_multiband = self._encode_combined(x, with_multi=True)
            return z, x_multiband
        return self._encode_combined(x)

    @torch.jit.ignore
    def decode(self, z, with_multi: bool = False):
        if isinstance(z, (tuple, list)):
            z_fast, z_slow = z
            side = self._decode_slow_map(z_slow, z_fast.shape[-1])
            return self.decoder.decode(z_fast,
                                       with_multi=with_multi,
                                       side=side)
        return self.decoder.decode(z, with_multi=with_multi)
