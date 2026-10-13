"""Causal waveform-to-mel structure conditioning at the codec frame rate."""
import gin
import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.functional import melscale_fbanks

from after.autoencoder.networks.cachedConv2D import CausalConv2d


class ChannelNorm2D(nn.Module):
    """Normalize channels independently at each frequency/time location."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MelEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride):
        super().__init__()
        self.time_stride = time_stride
        self.net = nn.Sequential(
            CausalConv2d(in_channels, out_channels, (3, 3), stride=(2, 1),
                         padding_vert=(1, 1), padding_time=(2, 0)),
            ChannelNorm2D(out_channels), nn.SiLU(),
            CausalConv2d(out_channels, out_channels, (3, 3),
                         padding_vert=(1, 1), padding_time=(2, 0)),
            ChannelNorm2D(out_channels), nn.SiLU())

    def forward(self, x):
        x = self.net(x)
        # Select interval endpoints rather than the start of each stride.
        return x[..., self.time_stride - 1::self.time_stride]


@gin.configurable
class CausalMelStructureEncoder(nn.Module):
    """Encode [B, audio channels, samples] into [B, out_dim, codec frames].

    Output j uses audio only through sample (j+1)*codec_hop-1. This is a
    stateless training encoder: history before the supplied crop is zero.
    """
    input_representation = "waveform"

    def __init__(self, sample_rate=44100, codec_hop=1024, n_fft=2048,
                 hop_length=256, n_mels=128, f_min=20., f_max=20000.,
                 log_offset=1e-5, channels=(32, 64, 128, 192, 256),
                 out_dim=8, ac_regularisation=True):
        super().__init__()
        assert len(channels) == 5, "Expected stem and four convolution stages"
        assert n_fft >= hop_length and codec_hop % (2 * hop_length) == 0
        self.codec_hop = codec_hop
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_offset = log_offset
        self.out_size = out_dim
        self.ac_regularisation = ac_regularisation
        self.register_buffer("window", torch.hann_window(n_fft))
        self.register_buffer("mel_filters", melscale_fbanks(
            n_fft // 2 + 1, f_min, f_max, n_mels, sample_rate,
            norm="slaney", mel_scale="slaney"))
        self.stem = nn.Sequential(nn.Conv2d(1, channels[0], 1),
                                  ChannelNorm2D(channels[0]), nn.SiLU())
        strides = (1, 2, 1, codec_hop // (2 * hop_length))
        self.blocks = nn.Sequential(*[
            MelEncoderBlock(channels[i], channels[i+1], strides[i])
            for i in range(4)])
        self.output = nn.Conv1d(channels[-1], out_dim, 1)

    def spectrogram(self, waveform):
        assert waveform.ndim == 3, "Expected waveform [B, C, samples]"
        assert waveform.shape[-1] % self.codec_hop == 0, "Crop must contain whole codec frames"
        # Right-aligned analysis: frame j ends at (j+1)*hop_length-1.
        # Disable autocast for FFT and mel projection, then resume for convs.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            mono = waveform.float().mean(dim=1)
            mono = F.pad(mono, (self.n_fft - self.hop_length, 0))
            spectrum = torch.stft(mono, n_fft=self.n_fft,
                                  hop_length=self.hop_length,
                                  win_length=self.n_fft, window=self.window.float(),
                                  center=False, return_complex=True)
            power = spectrum.abs().square()
            mel = torch.matmul(power.transpose(1, 2), self.mel_filters.float())
            return torch.log(mel.transpose(1, 2) + self.log_offset).unsqueeze(1)

    def forward(self, waveform, return_full=False):
        x = self.blocks(self.stem(self.spectrogram(waveform)))
        # Pool frequency only; time resolution stays explicit.
        z = self.output(x.mean(dim=2))
        if return_full:
            regularisation = F.relu(z.abs() - 1).mean() if self.ac_regularisation else z.new_zeros(())
            return z, z, regularisation
        return z
