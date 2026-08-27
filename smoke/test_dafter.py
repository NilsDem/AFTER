"""Smoke tests for MIDI-conditioned audio-space rectified flow."""
import pathlib
import pickle
import tempfile

import gin
import lmdb
import numpy as np
import pretty_midi
import torch

from after.dafter.data import collate_dafter, get_dafter_datasets
from after.dafter.model import DafterRectifiedFlow
from after.dafter.network import DafterNetwork
from after.dafter.style import SpectralStyleEncoder
from after.dafter.summary import format_model_summary, model_summary
from after.dafter.trainer import DafterTrainer
from after.dataset.audio_example import AudioExample


def tiny_model(midi_dropout=0.0,
               style_dropout=0.0,
               style_condition_source="encode"):
    network = DafterNetwork(
        nfft=64,
        hop_size=16,
        patch_ratio=4,
        patch_channels=4,
        hidden_channels=32,
        n_layers=1,
        n_heads=4,
        mlp_multiplier=2,
        midi_channels=128,
        style_channels=8,
        condition_width=16,
        attention_context_frames=8,
        max_batch_size=1,
        max_stream_frames=4,
    )
    style_encoder = (SpectralStyleEncoder(style_channels=8,
                                          channels=(4, 8),
                                          nfft=64,
                                          hop_size=16)
                     if style_condition_source == "encode" else None)
    return DafterRectifiedFlow(network=network,
                              style_encoder=style_encoder,
                              style_condition_source=style_condition_source,
                              midi_dropout=midi_dropout,
                              style_dropout=style_dropout)


def test_collate_builds_hop_aligned_128_note_roll():
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(
        pretty_midi.Note(velocity=100,
                         pitch=60,
                         start=0.0,
                         end=0.02))
    midi.instruments.append(instrument)
    waveform = np.linspace(-1.0, 1.0, 256,
                           dtype=np.float32)[None]

    batch = collate_dafter(
        [{"waveform": waveform, "midi": midi, "metadata": {"id": 1}}],
        n_frames=8,
        hop_size=16,
        sample_rate=16000,
        style_crop_samples=64,
        style_embedding_dim=8,
        style_condition_source="encode",
        style_embedding_key=None,
    )
    assert batch["waveform"].shape == (1, 1, 128)
    assert batch["midi"].shape == (1, 128, 8)
    assert batch["style_waveform"].shape == (1, 1, 64)
    assert batch["midi"][0, 60].max() == 100.0 / 127.0
    assert "style_embedding" not in batch


def test_waveform_midi_lmdb_loads_without_codec_or_style_condition():
    with tempfile.TemporaryDirectory() as path:
        midi = pretty_midi.PrettyMIDI()
        example = AudioExample()
        example.put_array("waveform",
                          np.zeros((1, 128), dtype=np.int16),
                          dtype=np.int16)
        example.put_buffer("midi", pickle.dumps(midi), shape=None)
        example.put_metadata({"path": "test.wav"})
        environment = lmdb.open(path, map_size=16 * 1024**2)
        with environment.begin(write=True) as transaction:
            transaction.put(b"00000000", bytes(example))
        environment.close()

        dataset, validation, sampler, validation_sampler = (
            get_dafter_datasets([path],
                                use_validation=False,
                                style_condition_source="none"))
        assert len(dataset) == 1
        assert validation is None
        assert validation_sampler is None
        batch = collate_dafter([dataset[0]],
                               n_frames=8,
                               hop_size=16,
                               sample_rate=16000,
                               style_crop_samples=64,
                               style_embedding_dim=8,
                               style_condition_source="none")
        assert batch["waveform"].shape == (1, 1, 128)
        assert batch["midi"].shape == (1, 128, 8)


def test_rectified_flow_loss_backward_and_euler_sampling_shapes():
    torch.manual_seed(0)
    model = tiny_model()
    waveform = torch.randn(2, 1, 128)
    midi = torch.randn(2, 128, 8)
    style_waveform = torch.randn(2, 1, 256)
    output = model(waveform=waveform,
                   midi=midi,
                   style_waveform=style_waveform)
    assert output["loss"].ndim == 0
    output["loss"].backward()
    assert model.network.patcher.project.weight.grad is not None
    assert model.style_encoder.projection.weight.grad is not None

    model.eval()
    with torch.no_grad():
        style = model.resolve_style(style_waveform=style_waveform)
        noise = torch.randn(2, 2, 32, 8)
        spectrum = model.sample_spectrogram(midi,
                                            style,
                                            num_steps=2,
                                            initial_noise=noise)
        audio = model.sample_audio(midi,
                                   style,
                                   num_steps=2,
                                   initial_noise=noise)
    assert spectrum.shape == (2, 2, 32, 8)
    assert audio.shape == waveform.shape


def test_midi_and_style_dropout_are_independent_config_entries():
    model = tiny_model(midi_dropout=1.0, style_dropout=0.0)
    midi = torch.ones(4, 128, 8)
    style = torch.ones(4, 8)
    dropped_midi, dropped_style, midi_mask, style_mask = (
        model.drop_conditions(midi, style))
    assert not dropped_midi.any()
    assert torch.equal(dropped_style, style)
    assert midi_mask.all()
    assert not style_mask.any()


def test_style_source_is_one_of_encode_data_or_none_for_the_whole_run():
    model = tiny_model(style_condition_source="data")
    provided = torch.randn(2, 8)
    reference = torch.randn(2, 128, 8)
    resolved = model.resolve_style(style_embedding=provided,
                                   reference=reference)
    torch.testing.assert_close(resolved, provided)
    assert model.style_encoder is None

    no_style_model = tiny_model(style_condition_source="none")
    resolved = no_style_model.resolve_style(reference=reference)
    assert resolved.shape == (2, 8)
    assert not resolved.any()
    assert no_style_model.style_encoder is None

    item = {
        "waveform": np.zeros((1, 128), dtype=np.float32),
        "midi": None,
        "stored_style": np.ones(8, dtype=np.float32),
    }
    data_batch = collate_dafter([item],
                                n_frames=8,
                                hop_size=16,
                                sample_rate=16000,
                                style_crop_samples=64,
                                style_embedding_dim=8,
                                style_condition_source="data",
                                style_embedding_key="stored_style")
    assert "style_waveform" not in data_batch
    assert data_batch["style_embedding"].shape == (1, 8)

    none_batch = collate_dafter([item],
                                n_frames=8,
                                hop_size=16,
                                sample_rate=16000,
                                style_crop_samples=64,
                                style_embedding_dim=8,
                                style_condition_source="none")
    assert "style_waveform" not in none_batch
    assert "style_embedding" not in none_batch


def test_layer_summary_has_parameters_and_runtime_shapes():
    model = tiny_model()
    summary = model_summary(model, n_frames=8, style_crop_samples=256)
    report = format_model_summary(summary)
    assert summary["total_parameters"] > 0
    assert summary["network_parameters"] > 0
    assert summary["style_encoder_parameters"] > 0
    assert "network.patcher.downsample_blocks.0.conv" in report
    assert "[1, 2, 32, 8]" in report


def test_trainer_runs_one_optimizer_step():
    trainer = DafterTrainer(tiny_model(), device="cpu")
    before = trainer.model.network.patcher.project.weight.detach().clone()
    metrics = trainer.training_step({
        "waveform": torch.randn(2, 1, 128),
        "midi": torch.randn(2, 128, 8),
        "style_waveform": torch.randn(2, 1, 256),
    })
    assert metrics["loss"] > 0
    assert not torch.equal(before,
                           trainer.model.network.patcher.project.weight)


def test_audio_logging_generates_four_examples_at_5_and_20_steps():
    class RecordingLogger:
        def __init__(self):
            self.tags = []

        def add_audio(self, tag, *args, **kwargs):
            self.tags.append(tag)

    trainer = DafterTrainer(tiny_model(), device="cpu")
    logger = RecordingLogger()
    trainer.log_audio(logger, {
        "waveform": torch.randn(4, 1, 128),
        "midi": torch.randn(4, 128, 8),
        "style_waveform": torch.randn(4, 1, 256),
    }, sample_rate=16000, sample_steps=(5, 20), examples=4)
    assert sum(tag.startswith("audio/target/") for tag in logger.tags) == 4
    assert sum("generated_5_steps" in tag for tag in logger.tags) == 4
    assert sum("generated_20_steps" in tag for tag in logger.tags) == 4


def test_reference_gin_config_is_64_hop_and_256_bins():
    gin.clear_config()
    config = (pathlib.Path(__file__).parents[1] / "after" / "dafter" /
              "configs" / "midi_audio_64.gin")
    gin.parse_config_file(str(config))
    network = DafterNetwork()
    assert network.hop_size == 64
    assert network.nfft == 512
    assert network.spectral_bins == 256
    assert network.patcher.patched_bins == 16
    assert network.blocks[0].attention.embed_dim == 256
    gin.clear_config()


if __name__ == "__main__":
    test_collate_builds_hop_aligned_128_note_roll()
    test_waveform_midi_lmdb_loads_without_codec_or_style_condition()
    test_rectified_flow_loss_backward_and_euler_sampling_shapes()
    test_midi_and_style_dropout_are_independent_config_entries()
    test_style_source_is_one_of_encode_data_or_none_for_the_whole_run()
    test_layer_summary_has_parameters_and_runtime_shapes()
    test_trainer_runs_one_optimizer_step()
    test_audio_logging_generates_four_examples_at_5_and_20_steps()
    test_reference_gin_config_is_64_hop_and_256_bins()
