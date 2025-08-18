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

        return self.model.encode(x_padded).float()

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
        self.agc = AGC.from_pretrained("Audiogen/agc-continuous").to(
            device)  # or "agc-discrete"
        self.device = device

        self.fwd_resampler = torchaudio.transforms.Resample(orig_freq=44100,
                                                            new_freq=48000).to(
                                                                self.device)
        self.inverse_resampler = torchaudio.transforms.Resample(
            orig_freq=48000, new_freq=44100).to(self.device)

    def cpu(self):
        self.agc.cpu()
        self.inverse_resampler.cpu()
        self.fwd_resampler.cpu()
        return self

    def to(self, device):
        self.agc.to(device)
        self.inverse_resampler.to(device)
        self.fwd_resampler.to(device)
        return self

    def eval(self):
        self.agc.eval()
        return self

    def encode(self, x):
        x = self.fwd_resampler(x.repeat(1, 2, 1))  # repeat to stereo
        x = self.agc.encode(x)
        return x

    def decode(self, z):
        x = self.agc.decode(z)[:, :1]
        x = self.inverse_resampler(x)
        return x

    def __call__(self, x):
        return self.decode(self.encode(x))


from dac import DAC
import dac
import torch


class DACWrapper:
    """Wrapper around Descript's DAC model."""

    def __init__(self, device="cpu", model_type="44khz"):
        self.device = device

        self.model = DAC.load(
            "/data/nils/repos/codecs_benchmark/autoencoder_runs/weights_44khz_8kbps_0.0.1.pth"
        )
        self.model.to(device).eval()

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def cpu(self):
        self.to("cpu")
        return self

    def eval(self):
        self.model.eval()
        return self

    def encode(self, x):
        # expects (B,1,T)
        with torch.no_grad():
            z, codes, latents, _, _ = self.model.encode(x.to(self.device))
            return z

    def decode(self, codes):
        with torch.no_grad():
            return self.model.decode(codes)

    def __call__(self, x):
        return self.decode(self.encode(x))


from encodec import EncodecModel
import torch.nn.functional as F

import torchaudio


class EncodecWrapper:
    """Wrapper around Facebook's Encodec model."""

    def __init__(self, device="cpu", bandwidth=6.0):
        self.device = device
        self.model = EncodecModel.encodec_model_24khz()  # or 48khz
        self.model.set_target_bandwidth(bandwidth)
        self.model.to(device).eval()

        self.fwd_resampler = torchaudio.transforms.Resample(orig_freq=44100,
                                                            new_freq=24000).to(
                                                                self.device)
        self.inverse_resampler = torchaudio.transforms.Resample(
            orig_freq=24000, new_freq=44100).to(self.device)

    def to(self, device):
        self.model.to(device)
        self.inverse_resampler.to(device)
        self.fwd_resampler.to(device)
        self.device = device
        return self

    def cpu(self):
        self.to("cpu")
        return self

    def eval(self):
        self.model.eval()
        return self

    def encode(self, x):
        # expects x: (B,1,T)
        x = self.fwd_resampler(x)  # repeat to stereo
        with torch.no_grad():
            return self.model.encode(x.to(self.device))[0][0]

    def decode(self, codes):

        with torch.no_grad():
            x = self.model.decode([(codes, None)])

        x = self.inverse_resampler(x)
        return x

    def __call__(self, x):
        return self.decode(self.encode(x))


from stable_audio_tools.models.autoencoders import AudioAutoencoder
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
import json
import torch


def copy_state_dict(model, state_dict):
    """Load state_dict to model, but only for keys that match exactly.

    Args:
        model (nn.Module): model to load state_dict.
        state_dict (OrderedDict): state_dict to load.
    """
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict and state_dict[
                key].shape == model_state_dict[key].shape:
            if isinstance(state_dict[key], torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                state_dict[key] = state_dict[key].data
            model_state_dict[key] = state_dict[key]

    model.load_state_dict(model_state_dict, strict=False)


class SAOWrapper():
    """Wrapper for the EncoderDecoder model to use it with the AudioExample class."""

    def __init__(self, device="cpu"):

        ckpt_path = "/data/nils/repos/codecs_benchmark/autoencoder_runs/sao/vae_model.ckpt"
        model_config = "/data/nils/repos/codecs_benchmark/autoencoder_runs/sao/vae_model_config.json"
        with open(model_config) as f:
            model_config = json.load(f)

        model = create_model_from_config(model_config)

        copy_state_dict(model, load_ckpt_state_dict(ckpt_path))

        self.model = model
        self.to(device)

    def to(self, device):
        self.model.to(device)
        return self

    def cpu(self):
        self.model.cpu()
        return self

    def eval(self):
        self.model.eval()
        return self

    def encode(self, x):
        x = x.repeat(1, 2, 1)
        z = self.model.encode(x)
        return z

    def decode(self, z):
        x = self.model.decode(z)
        x = x.mean(dim=1, keepdim=True)
        return x

    def __call__(self, x):
        return self.decode(self.encode(x))
