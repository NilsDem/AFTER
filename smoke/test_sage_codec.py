"""SAGE grid ordering and the offline/online representation contract."""
import sys
import types

import gin
import numpy as np
import pytest
import torch
from torch import nn

from after_scripts.prepare_dataset import encode_batch
from after.autoencoder.codecs import load_codec
from after.diffusion.model import Base
from after.encoder_ssl.transforms import CodecEncoderTransform


class FakeSAGE:
    def __init__(self, folded=False):
        self.device = "cpu"
        self.autoencoder = nn.Module()
        self.autoencoder.weight = nn.Parameter(torch.zeros(()))
        self.autoencoder.register_buffer("window", torch.ones(4))
        self.autoencoder.encoder = types.SimpleNamespace(
                fold_freq_to_channels=folded, _final_resolution=(4, 32),
                depths=[2, 6, 2],
                patch_embed=types.SimpleNamespace(patch_size=(64, 1)))
        self.autoencoder._stft_config = types.SimpleNamespace(hop_length=512)

    def to(self, device):
        self.autoencoder.to(device)
        self.device = torch.device(device)
        return self

    def pad(self, audio):
        frames = audio.shape[-1] // 512 + 1
        length = (frames + (-frames) % 4 - 1) * 512
        return torch.nn.functional.pad(audio, (0, length - audio.shape[-1])), audio.shape[-1]

    def encode(self, audio):
        frames = (audio.shape[-1] // 512 + 1) // 4
        bands = 1 if self.autoencoder.encoder.fold_freq_to_channels else 4
        self.native = torch.arange(2 * bands * frames).reshape(1, 2, bands * frames).float()
        return self.native.expand(audio.shape[0], -1, -1)

    def decode(self, latent, target_length):
        self.decoded = latent
        return latent.new_zeros(latent.shape[0], 2, target_length)


@pytest.mark.parametrize("folded", [False, True])
def test_sage_preprocessing(monkeypatch, folded):
    model = FakeSAGE(folded)
    monkeypatch.setitem(sys.modules, "sage", types.SimpleNamespace(
        SAGE=types.SimpleNamespace(from_checkpoint=lambda *args, **kwargs: model)))
    gin.clear_config()
    gin.bind_parameter("load_codec.codec_type", "sage")
    codec = load_codec("unused.ckpt")
    audio = torch.ones(2, 2, 8192)
    latent = codec.encode(audio)
    bands = 1 if folded else 4
    assert latent.shape == (2, 2 * bands, 5)
    for c in range(2):
        for f in range(bands):
            torch.testing.assert_close(latent[:, c * bands + f], model.native[:, c, f*5:(f+1)*5].expand(2, -1))
    reconstructed = codec.decode(latent)
    torch.testing.assert_close(model.decoded, model.native.expand(2, -1, -1))
    assert reconstructed.shape == (2, 2, 9728)
    assert codec.compression_ratio == 2048
    codec.to("cpu")
    assert model.device == torch.device("cpu")
    assert next(codec.parameters()) is model.autoencoder.weight
    assert not model.autoencoder.weight.requires_grad
    codec.train()
    assert not model.autoencoder.training
    codec.to(dtype=torch.float64)
    assert model.autoencoder.window.dtype == torch.float64
    transform = CodecEncoderTransform("unused.ckpt")
    torch.testing.assert_close(transform(audio), latent)
    trainer = Base.__new__(Base)
    nn.Module.__init__(trainer)
    trainer.emb_model = codec
    trainer.encoder_time = None
    trainer.ssl_representation = "sage"
    x, cond, time = trainer.prep_data(
        {"waveform": audio, "timbre_waveform": audio}, device="cpu")
    assert x.shape[-1] == 4
    torch.testing.assert_close(time, x)
    # Stored offline latents retain the boundary frame for subsequent cropping.
    stored = encode_batch(codec, list(audio.numpy()), "cpu")
    np.testing.assert_array_equal(stored[..., :4], x.numpy())
    gin.clear_config()


def test_sage_configs():
    gin.clear_config()
    gin.parse_config_file("after/diffusion/configs/audio_simdino_sage.gin")
    assert gin.query_parameter("load_codec.codec_type") == "sage"
    gin.clear_config()
    gin.parse_config_file("after/encoder_ssl/configs/ssl_sage.gin")
    assert gin.query_parameter("%AUDIO_CHANNELS") == 2
    assert gin.query_parameter("%REPRESENTATION") == "sage"
    gin.clear_config()
