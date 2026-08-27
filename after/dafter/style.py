"""Style-conditioning encoders for DAFTER."""
from typing import Sequence

import gin
import torch
from torch import nn

from after.autoencoder.audio import CausalMauerSTFT


@gin.configurable
class SpectralStyleEncoder(nn.Module):
    """Small trainable style encoder that can later receive pretrained weights."""

    def __init__(self,
                 style_channels: int = 64,
                 channels: Sequence[int] = (16, 32, 64, 128),
                 nfft: int = 512,
                 hop_size: int = 128) -> None:
        super().__init__()
        self.style_channels = int(style_channels)
        self.transform = CausalMauerSTFT(nfft=nfft,
                                         hop_size=hop_size,
                                         synthesis_length=2 * hop_size,
                                         zero_length=hop_size,
                                         skip_features=-1,
                                         normalize=True)
        blocks = []
        in_channels = 2
        for out_channels in channels:
            blocks.extend((
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(),
            ))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.projection = nn.Linear(channels[-1], style_channels)
        self.output_norm = nn.LayerNorm(style_channels)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[1] != 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        spectrum = self.transform(waveform)
        features = self.blocks(spectrum).mean(dim=(-2, -1))
        return self.output_norm(self.projection(features))
