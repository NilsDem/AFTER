"""Causal mel analysis, frame alignment, and hybrid diffusion integration."""
import gin
import numpy as np
import pytest
import torch
from torch import nn

from after.diffusion.model import Base
from after.diffusion.networks.mel_structure import CausalMelStructureEncoder
from after.diffusion.networks.transformerv2 import DenoiserV2
from after.diffusion.utils import collate_fn, make_key_sampler


@pytest.mark.parametrize("codec_hop", [1024, 2048])
def test_shapes_and_future_independence(codec_hop):
    torch.manual_seed(1)
    encoder = CausalMelStructureEncoder(codec_hop=codec_hop).train()
    assert 1_800_000 < sum(p.numel() for p in encoder.parameters()) < 2_000_000
    waveform = torch.randn(1, 2, codec_hop * 8)
    changed = waveform.clone()
    changed[..., codec_hop * 4:] += torch.randn_like(changed[..., codec_hop * 4:]) * 5
    mel = encoder.spectrogram(waveform)
    altered_mel = encoder.spectrogram(changed)
    assert mel.shape == (1, 1, 128, waveform.shape[-1] // 256)
    torch.testing.assert_close(mel[..., :codec_hop * 4 // 256], altered_mel[..., :codec_hop * 4 // 256])
    output = encoder(waveform)
    altered_output = encoder(changed)
    assert output.shape == (1, 8, 8)
    torch.testing.assert_close(output[..., :4], altered_output[..., :4])
    assert not torch.allclose(output[..., 4:], altered_output[..., 4:])
    # Crop extension must not change its already-complete codec frames.
    torch.testing.assert_close(encoder(waveform[..., :codec_hop * 4]), output[..., :4])


def test_analysis_matches_right_aligned_window():
    encoder = CausalMelStructureEncoder()
    waveform = torch.randn(1, 2, 2048)
    mono = waveform.mean(dim=1)
    frame = torch.nn.functional.pad(mono[:, :256], (1792, 0))
    power = torch.fft.rfft(frame * encoder.window).abs().square()
    expected = torch.log(power @ encoder.mel_filters + encoder.log_offset)
    torch.testing.assert_close(encoder.spectrogram(waveform)[:, 0, :, 0], expected)


class TinyCodec(nn.Module):
    def encode(self, waveform):
        return waveform.reshape(waveform.shape[0], waveform.shape[1], -1, 1024).mean(-1)


@torch.enable_grad()
def test_hybrid_batch_to_diffusion_backward():
    torch.manual_seed(2)
    encoder = CausalMelStructureEncoder()
    trainer = Base.__new__(Base)
    nn.Module.__init__(trainer)
    trainer.emb_model = TinyCodec()
    trainer.encoder_time = encoder
    trainer.ssl_representation = "codec"
    waveform = torch.randn(1, 2, 8192)
    x, cond, structure_input = trainer.prep_data(
        {"waveform": waveform, "timbre_waveform": waveform + 1}, device="cpu")
    torch.testing.assert_close(structure_input, waveform)
    assert x.shape == cond.shape == (1, 2, 8)
    structure, mean, regularisation = encoder(structure_input, return_full=True)
    net = DenoiserV2(n_channels=2, seq_len=8, embed_dim=32, cond_dim=2,
                     tcond_dim=8, noise_embed_dims=16, n_layers=1,
                     dropout=0., causal=True, attention_chunk_size=1)
    prediction = net(x, torch.full((1,), .5), cond.mean(-1), structure)
    assert prediction.shape == x.shape
    prediction.square().mean().backward()
    assert encoder.blocks[0].net[0].weight.grad.abs().sum() > 0


def test_offline_waveform_crop_and_key_selection(monkeypatch):
    gin.clear_config()
    gin.bind_parameter("%SR", 44100)
    # Deterministic latent offset 2, waveform offset 2*codec_hop.
    monkeypatch.setattr(np.random, "randint", lambda *args: 2)
    waveform = np.arange(10240, dtype=np.float32)[None]
    batch = [{"z": np.zeros((2, 10), dtype=np.float32), "waveform": waveform}]
    output = collate_fn(batch, 4, "mel", 1024, structure_keys=[], timbre_keys=[])
    np.testing.assert_array_equal(output["x_time_cond"].numpy(), waveform[None, ..., 2048:6144])
    encoder = CausalMelStructureEncoder()
    assert encoder(output["x_time_cond"]).shape == (1, 8, 4)
    assert "waveform" in make_key_sampler("mel", [], [])()[2]
    gin.clear_config()


def test_config_builds_mel_encoder():
    gin.clear_config()
    gin.parse_config_file("after/diffusion/configs/audio_simdino_mel.gin")
    gin.bind_parameter("encoder_time/CausalMelStructureEncoder.codec_hop", 2048)
    reference = gin.query_parameter("diffusion.model.Base.encoder_time")
    encoder = reference.scoped_configurable_fn()
    assert encoder.codec_hop == 2048
    assert encoder.out_size == 8
    assert gin.query_parameter("%STRUCTURE_TYPE") == "mel"
    assert gin.query_parameter("load_codec.codec_type") == "sage"
    gin.clear_config()


def test_lazy_mel_dataset_routes_aligned_waveform(tmp_path):
    import json
    import soundfile as sf
    from after.diffusion.online_data import OnlineWaveformDataset, collate_online

    waveform = np.random.default_rng(1).normal(0, .1, (12288, 2)).astype(np.float32)
    sf.write(tmp_path / "audio.wav", waveform, 44100, subtype="FLOAT")
    (tmp_path / "dataset.json").write_text(json.dumps({
        "format": "after_lazy_waveform", "source_root": ".", "sample_rate": 44100,
        "channels": 2, "epoch_chunk_size": 8192}))
    (tmp_path / "audio_manifest.jsonl").write_text(json.dumps({
        "path": "audio.wav", "num_frames": 12288, "sample_rate": 44100,
        "channels": 2, "duration": 12288 / 44100}) + "\n")
    dataset = OnlineWaveformDataset(tmp_path, n_signal=4, ae_ratio=2048,
                                    sample_rate=44100, structure_type="mel",
                                    audio_channels=2, augment=False)
    batch = collate_online([dataset[0]])
    assert batch["waveform"].shape == (1, 2, 8192)
    encoder = CausalMelStructureEncoder(codec_hop=2048)
    assert encoder(batch["waveform"]).shape == (1, 8, 4)
