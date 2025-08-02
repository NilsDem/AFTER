from music2latent import EncoderDecoder
import numpy as np
import torch.nn.functional as F


class M2LWrapper():
    """Wrapper for the EncoderDecoder model to use it with the AudioExample class."""

    def __init__(self, device="cpu"):
        self.model = EncoderDecoder(device=device)

    def to(self, device):
        self.model = EncoderDecoder(device=device)
        return self

    def cpu(self):
        self.model = EncoderDecoder(device="cpu")
        return self

    def eval(self):
        return self

    def encode(self, x):
        x = x.squeeze(1)
        x_padded = F.pad(x, (0, 1536))  # pad left and right

        return self.model.encode(x_padded)

    def decode(self, z):
        x = self.model.decode(z).unsqueeze(1)
        x = x[..., :-1536]  # remove padding
        return x

    def __call__(self, x):
        return self.decode(self.encode(x))


import torchaudio


class AudioGenWrapper():
    """Wrapper for the AudioGen model to use it with the AudioExample class."""

    def __init__(self, device="cpu"):
        from agc import AGC
        self.agc = AGC.from_pretrained(
            "Audiogen/agc-continuous")  # or "agc-discrete"
        self.device = device

        self.fwd_resampler = torchaudio.transforms.Resample(orig_freq=44100,
                                                            new_freq=48000).to(
                                                                self.device)
        self.inverse_resampler = torchaudio.transforms.Resample(
            orig_freq=48000, new_freq=44100).to(self.device)

    def cpu(self):
        self.agc.cpu()
        return self

    def eval(self):
        self.agc.eval()
        return self

    def encode(self, x):
        x = self.fwd_resampler(x)
        x = self.agc.encode(x)
        return x

    def decode(self, z):
        x = self.agc.decode(z)
        x = self.inverse_resampler(x)
        return x

    def __call__(self, x):
        return self.decode(self.encode(x))
