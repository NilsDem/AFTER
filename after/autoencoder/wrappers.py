import numpy as np
import torch.nn.functional as F
import torchaudio
import torch.nn.functional as F


class M2LWrapper():
    """Wrapper for the EncoderDecoder model to use it with the AudioExample class."""

    def __init__(self, device="cpu"):
        from music2latent import EncoderDecoder
        self.model = EncoderDecoder(device=device)

    def to(self, device):
        from music2latent import EncoderDecoder
        self.model = EncoderDecoder(device=device)
        return self

    def cpu(self):
        from music2latent import EncoderDecoder
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


class DACWrapper:
    """Wrapper around Descript's DAC model."""

    def __init__(self, device="cpu", model_type="44khz"):
        from dac import DAC
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


class EncodecWrapper:
    """Wrapper around Facebook's Encodec model."""

    def __init__(self, device="cpu", bandwidth=6.0):
        from encodec import EncodecModel
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


import json, math
import torch
from einops import rearrange


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


class SAOWrapper:
    """SAO wrapper with chunked overlap-add encode/decode (batched)."""

    def __init__(
        self,
        device="cpu",
        ckpt_path="/data/nils/repos/codecs_benchmark/autoencoder_runs/sao/vae_model.ckpt",
        model_config="/data/nils/repos/codecs_benchmark/autoencoder_runs/sao/vae_model_config.json"
    ):
        from stable_audio_tools.models.autoencoders import AudioAutoencoder
        from stable_audio_tools.models import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict

        with open(model_config) as f:
            model_config = json.load(f)
        model = create_model_from_config(model_config)
        copy_state_dict(model, load_ckpt_state_dict(ckpt_path))

        self.model = model.to(device).eval()
        self.device = device

        # convenience
        self.sr = model.sample_rate
        self.compress_ratio = model.downsampling_ratio
        self.latent_dim = model.latent_dim
        self.in_channels = model.in_channels
        self.out_channels = model.out_channels

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def cpu(self):
        return self.to("cpu")

    def eval(self):
        self.model.eval()
        return self

    def encode(self,
               audio,
               chunked=False,
               chunk_size=32,
               overlap=4,
               max_batch_size=1):
        """
        Encode batched audio with optional chunking.
        audio: (B, C, T), already preprocessed to correct channels and sample rate
        Returns latents: (B, latent_dim, latent_length)
        """
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        bs, n_ch, sample_length = audio.shape
        compress_ratio = self.compress_ratio

        if (audio.shape[-1] // compress_ratio) < 4 * chunk_size:
            chunked = False

        assert n_ch == self.in_channels
        assert sample_length % compress_ratio == 0, "Audio length must be multiple of compression ratio."

        latent_length = sample_length // compress_ratio
        hopsize_l = chunk_size - overlap
        win = torch.bartlett_window(overlap * 2, device=audio.device)

        if not chunked:
            return self.model.encode(audio)

        # --- chunked path ---
        chunk_size_s = chunk_size * compress_ratio
        overlap_s = overlap * compress_ratio
        hopsize_s = hopsize_l * compress_ratio

        n_chunk = int(math.ceil(
            (sample_length - chunk_size_s) / hopsize_s)) + 1
        pad_len = chunk_size_s + hopsize_s * (n_chunk - 1) - sample_length
        audio = F.pad(audio, (0, pad_len))

        chunks = []
        for i in range(n_chunk):
            head = i * hopsize_s
            chunk = audio[..., head:head + chunk_size_s]
            chunks.append(chunk)
        chunks = torch.stack(chunks, dim=1)  # (B, n_chunk, C, chunk_size_s)
        chunks = rearrange(chunks, "b n c l -> (b n) c l")

        # batched encoding
        zs = []
        for i in range(0, chunks.shape[0], max_batch_size):
            z_ = self.model.encode(chunks[i:i + max_batch_size])
            zs.append(z_)
        zs = torch.cat(zs, dim=0)
        zs = rearrange(zs, "(b n) c l -> b n c l", b=bs)

        # overlap-add crossfade in latent domain
        latents = torch.zeros(
            (bs, self.latent_dim, audio.shape[-1] // compress_ratio),
            device=audio.device)
        for i in range(n_chunk):
            z_ = zs[:, i]
            if i != 0:
                z_[:, :, :overlap] *= win[:overlap]
            if i != n_chunk - 1:
                z_[:, :, -overlap:] *= win[-overlap:]
            head = i * hopsize_l
            latents[..., head:head + chunk_size] += z_

        return latents[..., :latent_length]

    def decode(self,
               latents,
               chunked=False,
               chunk_size=32,
               overlap=4,
               max_batch_size=1):
        """
        Decode latents with optional chunking.
        latents: (B, latent_dim, latent_length)
        Returns audio: (B, C, T)
        """
        bs, latent_dim, latent_length = latents.shape
        compress_ratio = self.compress_ratio
        assert latent_dim == self.latent_dim

        if latents.shape[-1] < 4 * chunk_size:
            chunked = False

        hopsize = chunk_size - overlap
        chunk_size_s = chunk_size * compress_ratio
        overlap_s = overlap * compress_ratio
        hopsize_s = hopsize * compress_ratio
        sample_length = latent_length * compress_ratio

        win = torch.bartlett_window(overlap_s * 2, device=latents.device)

        if not chunked:
            audios = self.model.decode(latents)
            return audios[:, :1, :sample_length]

        # --- chunked path ---
        n_chunk = int(math.ceil((latent_length - chunk_size) / hopsize)) + 1
        pad_len = chunk_size + hopsize * (n_chunk - 1) - latent_length
        latents = F.pad(latents, (0, pad_len), mode="reflect")

        chunks = []
        for i in range(n_chunk):
            head = i * hopsize
            chunk = latents[..., head:head + chunk_size]
            chunks.append(chunk)
        chunks = torch.stack(chunks, dim=1)
        chunks = rearrange(chunks, "b n c l -> (b n) c l")

        # batched decoding
        xs = []
        for i in range(0, chunks.shape[0], max_batch_size):
            x_ = self.model.decode(chunks[i:i + max_batch_size])
            xs.append(x_)
        xs = torch.cat(xs, dim=0)
        xs = rearrange(xs, "(b n) c l -> b n c l", b=bs)

        audios = torch.zeros(
            (bs, xs.shape[2], latents.shape[-1] * compress_ratio),
            device=latents.device)
        for i in range(n_chunk):
            x_ = xs[:, i]
            if i != 0:
                x_[..., :overlap_s] *= win[:overlap_s]
            if i != n_chunk - 1:
                x_[..., -overlap_s:] *= win[-overlap_s:]
            head = i * hopsize_s
            audios[..., head:head + chunk_size_s] += x_
        print("hi")
        return audios[:, :1, :sample_length]

    def __call__(self, x, **kwargs):
        """shortcut: encode+decode"""
        z = self.encode(x, **kwargs)
        return self.decode(z, **kwargs)
