"""Double-encoder 2D spectral autoencoder."""
import math

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
                 time_transform=None):
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
            self.up_layers.append(
                DecoderBlock2D(in_c=channels_dec[n - i],
                               out_c=channels_dec[n - i - 1],
                               kernel_size=kernel_size,
                               ratio=time_ratios[n - i],
                               freq_ratio=freq_ratios[n - i]))

        self.stereo_split = (DecoderBlock2D(in_c=channels_dec[-1],
                                            out_c=2 * channels_dec[-1],
                                            kernel_size=kernel_size,
                                            ratio=1,
                                            freq_ratio=1)
                             if audio_channels == 2 else nn.Identity())

        self.outconv = Conv2d(2 * channels[0],
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

    def _decode_features(self, h):
        h = self.middle_block_decode(h)
        h = self.rearrange_decode(h)
        h = self.stereo_split(h)
        h = self.split_stereo_features(h)
        for layer in self.up_layers:
            h = layer(h)
        h = self.outconv(h)
        return h

    @torch.jit.ignore
    def decode(self, z, with_multi: bool = False):
        h = self._decode_features(z)
        y = self.time_transform.inverse(h)
        y = self.unpack_audio(y)
        if with_multi:
            return y, h
        return y


@gin.configurable
class DoubleAE(nn.Module):
    """Autoencoder with fast and slow encoders feeding one decoder."""

    def __init__(self,
                 fast_encoder: nn.Module,
                 slow_encoder: nn.Module,
                 decoder: nn.Module,
                 slow_shift_steps: int = 1,
                 regularisation_ratio: float = 1.):
        super().__init__()
        self.fast_encoder = fast_encoder
        self.slow_encoder = slow_encoder
        self.decoder = decoder
        self.slow_shift_steps = slow_shift_steps
        
        self.regloss_ratio = regularisation_ratio

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

    def _combine_latents(self, z_fast, z_slow):
        z_slow = self._shift_slow_to_past(z_slow)
        z_slow = self._match_slow_to_fast(z_slow, z_fast.shape[-1])
        return torch.cat((z_fast, z_slow), dim=1)

    @torch.jit.ignore
    def combine_codes(self, z_fast, z_slow):
        """Combine raw fast and slow codes into the decoder latent layout."""
        return self._combine_latents(z_fast, z_slow)

    @torch.jit.ignore
    def split_codes(self, z):
        """Split a combined decoder latent into fast and aligned slow parts."""
        fast_size = self.fast_encoder.bottleneck_size
        return z[:, :fast_size], z[:, fast_size:]

    def _encode_combined(self, x, with_multi: bool = False):
        z_fast, fast_regloss, fast_multiband = self.fast_encoder._encode_with_multi(
            x)
        z_slow, slow_regloss, _ = self.slow_encoder._encode_with_multi(x)
        z = self._combine_latents(z_fast, z_slow)
        
        fast_regloss *= self.regloss_ratio

        if with_multi:
            return z, fast_regloss + slow_regloss, fast_multiband

        return z, fast_regloss + slow_regloss

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
        z = self.combine_codes(z_fast, z_slow)
        return self.decode(z, with_multi=with_multi)

    @torch.jit.ignore
    def forward(self,
                x,
                return_all: bool = True,
                freeze_encoder: bool = False,
                look_ahead_steps: int = 0):
        if freeze_encoder:
            with torch.no_grad():
                z, regloss, x_multiband = self._encode_combined(
                    x, with_multi=True)
        else:
            z, regloss, x_multiband = self._encode_combined(x,
                                                            with_multi=True)

        if look_ahead_steps > 0:
            z = z[..., look_ahead_steps:]
            z = torch.cat((z, torch.zeros_like(z[..., :look_ahead_steps])),
                          dim=-1)

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
        return self.decoder.decode(z, with_multi=with_multi)
