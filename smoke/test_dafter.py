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
from after.dafter.model import DafterRectifiedFlow, pseudo_huber_loss
from after.dafter.network import DafterNetwork
from after.dafter.style import SpectralStyleEncoder
from after.dafter.summary import format_model_summary, model_summary
from after.dafter.trainer import DafterTrainer
from after.dataset.audio_example import AudioExample
from after.diffusion.networks.transformerv2 import PositionalEmbedding


def tiny_model(midi_dropout=0.0,
               style_dropout=0.0,
               style_condition_source="encode",
               whiten_spectrum=False):
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
        use_style=style_condition_source != "none",
        condition_width=16,
        attention_context_frames=8,
        max_batch_size=1,
        max_stream_frames=4,
        whiten_spectrum=whiten_spectrum,
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


def test_spectrum_whitening_is_fitted_applied_and_checkpointed():
    torch.manual_seed(7)
    model = tiny_model(style_condition_source="none", whiten_spectrum=True)
    trainer = DafterTrainer(model=model)
    batches = [
        {"waveform": torch.randn(3, 1, 128) + offset}
        for offset in (-0.25, 0.5)
    ]

    fitted_batches = trainer.fit_spectrum_whitening(batches)
    assert fitted_batches == len(batches)

    raw_spectra = torch.cat([
        model.network.time_transform(batch["waveform"])
        for batch in batches
    ])
    whitened = model.network.whiten(raw_spectra)
    torch.testing.assert_close(
        whitened.mean(dim=(0, 3)),
        torch.zeros_like(whitened.mean(dim=(0, 3))),
        atol=2e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        whitened.var(dim=(0, 1, 3), correction=0),
        torch.ones_like(whitened.var(dim=(0, 1, 3), correction=0)),
        atol=2e-4,
        rtol=2e-4,
    )
    torch.testing.assert_close(model.network.unwhiten(whitened), raw_spectra)
    first_waveform = batches[0]["waveform"]
    torch.testing.assert_close(
        model.audio_to_spectrum(first_waveform),
        model.network.whiten(model.network.time_transform(first_waveform)),
    )
    candidate_spectrum = torch.randn_like(raw_spectra[:1])
    torch.testing.assert_close(
        model.spectrum_to_audio(candidate_spectrum),
        model.network.time_transform.inverse(
            model.network.unwhiten(candidate_spectrum)),
    )

    state = model.state_dict()
    assert "network.spectrum_whitening_mean" in state
    assert "network.spectrum_whitening_std" in state
    restored = tiny_model(style_condition_source="none", whiten_spectrum=True)
    restored.load_state_dict(state)
    torch.testing.assert_close(
        restored.network.spectrum_whitening_mean,
        model.network.spectrum_whitening_mean,
    )
    torch.testing.assert_close(
        restored.network.spectrum_whitening_std,
        model.network.spectrum_whitening_std,
    )


def test_disabled_spectrum_whitening_is_identity_and_not_checkpointed():
    model = tiny_model(style_condition_source="none", whiten_spectrum=False)
    spectrum = torch.randn(2, 2, model.network.spectral_bins, 8)
    assert model.network.whiten(spectrum) is spectrum
    assert model.network.unwhiten(spectrum) is spectrum
    state = model.state_dict()
    assert "network.spectrum_whitening_mean" not in state
    assert "network.spectrum_whitening_std" not in state


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


def test_midi_conditioning_is_concatenated_then_fused():
    torch.manual_seed(0)
    network = tiny_model(style_condition_source="none").network
    spectrum = torch.randn(2, 2, 32, 8)
    midi = torch.randn(2, 128, 8)

    patch_tokens, _ = network.patcher(spectrum)
    expected = network.token_condition_fusion(
        torch.cat((patch_tokens, midi.transpose(1, 2)), dim=-1))
    actual = network._fuse_conditioning(patch_tokens, midi)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == patch_tokens.shape


def test_frequency_scales_use_spe_noise_embedding_and_start_at_identity():
    torch.manual_seed(0)
    network = tiny_model(style_condition_source="none").network
    flow_time = torch.tensor([[0.0], [0.75]])

    noise_condition, input_scale, output_scale = (
        network._noise_condition_and_frequency_scales(flow_time))

    assert isinstance(network.noise_spe, PositionalEmbedding)
    expected_scale_layers = [torch.nn.Linear, torch.nn.SiLU,
                             torch.nn.Linear, torch.nn.SiLU,
                             torch.nn.Linear]
    assert [type(layer) for layer in network.scale_inp] == expected_scale_layers
    assert [type(layer) for layer in network.scale_out] == expected_scale_layers
    assert noise_condition.shape == (2, 16)
    assert input_scale.shape == (2, 1, 32, 1)
    assert output_scale.shape == (2, 1, 32, 1)
    torch.testing.assert_close(input_scale, torch.zeros_like(input_scale))
    torch.testing.assert_close(output_scale, torch.zeros_like(output_scale))
    assert not torch.equal(noise_condition[0], noise_condition[1])

    prediction = network(
        spectrum=torch.randn(2, 2, 32, 8),
        conditioning=torch.randn(2, 128, 8),
        style=None,
        flow_time=flow_time,
    )
    prediction.square().mean().backward()
    input_scale_gradient = network.scale_inp[-1].weight.grad
    output_scale_gradient = network.scale_out[-1].weight.grad
    assert input_scale_gradient is not None
    assert output_scale_gradient is not None
    assert torch.isfinite(input_scale_gradient).all()
    assert torch.isfinite(output_scale_gradient).all()
    assert input_scale_gradient.abs().sum() > 0
    assert output_scale_gradient.abs().sum() > 0

    assert all(block.condition_projection.weight.grad is not None
               for block in network.patcher.downsample_blocks)
    assert all(block.condition_projection.weight.grad is not None
               for block in network.depatcher.upsample_blocks)


def test_frequency_scales_modulate_hidden_features_not_spectrograms():
    torch.manual_seed(0)
    network = tiny_model(style_condition_source="none").network
    spectrum = torch.randn(2, 2, 32, 8)
    condition = torch.randn(2, 16)
    input_scale = torch.full((2, 1, 32, 1), 0.25)
    patcher_hidden = []

    patcher_hook = network.patcher.downsample_blocks[0].register_forward_pre_hook(
        lambda module, inputs: patcher_hidden.append(inputs[0].detach()))
    try:
        _, skip_features = network.patcher(spectrum,
                                           input_scale,
                                           condition)
    finally:
        patcher_hook.remove()

    expected_patcher_hidden = ((1.0 + input_scale) *
                               network.patcher.input_conv(spectrum))
    torch.testing.assert_close(patcher_hidden[0], expected_patcher_hidden)

    tokens = torch.randn(2, 8, 32)
    output_scale = torch.full((2, 1, 32, 1), -0.25)
    depatcher_hidden = []
    depatcher_hook = network.depatcher.output_conv.register_forward_pre_hook(
        lambda module, inputs: depatcher_hidden.append(inputs[0].detach()))
    try:
        network.depatcher(tokens, skip_features, output_scale, condition)
    finally:
        depatcher_hook.remove()

    hidden = network.depatcher.project(tokens)
    hidden = hidden.reshape(tokens.shape[0], tokens.shape[1],
                            network.depatcher.patch_channels,
                            network.depatcher.patched_bins)
    hidden = hidden.permute(0, 2, 3, 1)
    for index, block in enumerate(network.depatcher.upsample_blocks):
        hidden = torch.cat((hidden, skip_features[-index - 1]), dim=1)
        hidden = block(hidden, condition)
    expected_depatcher_hidden = (1.0 + output_scale) * hidden
    torch.testing.assert_close(depatcher_hidden[0], expected_depatcher_hidden)


def test_each_patcher_feature_is_concatenated_into_matching_depatcher_block():
    torch.manual_seed(0)
    network = tiny_model(style_condition_source="none").network
    spectrum = torch.randn(2, 2, 32, 8)
    midi = torch.randn(2, 128, 8)
    flow_time = torch.rand(2, 1)
    depatcher_block_inputs = []
    handles = [
        block.register_forward_pre_hook(
            lambda module, inputs: depatcher_block_inputs.append(
                inputs[0].detach()))
        for block in network.depatcher.upsample_blocks
    ]
    try:
        noise_condition, input_scale, _ = (
            network._noise_condition_and_frequency_scales(flow_time))
        _, skip_features = network.patcher(spectrum,
                                           input_scale,
                                           noise_condition)

        network(spectrum, midi, None, flow_time)
    finally:
        for handle in handles:
            handle.remove()

    assert len(depatcher_block_inputs) == len(skip_features)
    for index, block_input in enumerate(depatcher_block_inputs):
        assert block_input.shape[1] == 2 * network.patcher.patch_channels
        torch.testing.assert_close(
            block_input[:, network.patcher.patch_channels:],
            skip_features[-index - 1],
        )


def test_pseudo_huber_loss_matches_elementwise_definition_and_is_scalar():
    prediction = torch.tensor([[[[0.0, 1.0], [-2.0, 3.0]]]],
                              requires_grad=True)
    target = torch.zeros_like(prediction)
    transition = 0.00054 * np.sqrt(prediction[0].numel())
    expected = (
        torch.sqrt((prediction - target).square() + transition**2) -
        transition).mean()

    loss = pseudo_huber_loss(prediction, target)

    assert loss.ndim == 0
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_pseudo_huber_transition_scale_is_independent_of_batch_size():
    prediction = torch.linspace(-2.0, 2.0, 24).reshape(1, 2, 4, 3)
    target = torch.zeros_like(prediction)
    single_loss = pseudo_huber_loss(prediction, target)
    repeated_loss = pseudo_huber_loss(prediction.repeat(5, 1, 1, 1),
                                      target.repeat(5, 1, 1, 1))
    torch.testing.assert_close(repeated_loss, single_loss)


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
    assert resolved is None
    assert no_style_model.style_encoder is None
    assert no_style_model.network.style_projection is None
    assert all(block.modulation is not None
               for block in no_style_model.network.blocks)

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
    assert torch.is_tensor(metrics["loss"])
    assert isinstance(trainer._metrics_for_logging(metrics)["loss"], float)
    assert metrics["loss"] > 0
    assert not torch.equal(before,
                           trainer.model.network.patcher.project.weight)


def test_channels_last_execution_and_fused_adamw_device_fallback():
    trainer = DafterTrainer(tiny_model(),
                            device="cpu",
                            use_channels_last=True,
                            use_fused_adamw=True)
    weight = trainer.model.network.patcher.input_conv.weight
    assert trainer.model.network.channels_last
    assert weight.is_contiguous(memory_format=torch.channels_last)
    assert not trainer.fused_adamw


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
    network = DafterNetwork(use_style=False)
    assert network.hop_size == 64
    assert network.nfft == 512
    assert network.time_transform.alpha_rescale == 0.5
    assert network.time_transform.beta_rescale == 3.0
    assert network.spectral_bins == 256
    assert network.patcher.patched_bins == 16
    assert (network.patcher.patch_channels * network.patcher.patched_bins ==
            512)
    assert network.token_condition_fusion.in_features == 384 + 128
    assert network.token_condition_fusion.out_features == 384
    assert isinstance(network.noise_spe, PositionalEmbedding)
    assert network.scale_inp[-1].out_features == 256
    assert network.scale_out[-1].out_features == 256
    assert network.style_dim == 0
    assert network.style_projection is None
    assert network.blocks[0].attention.embed_dim == 384
    assert network.blocks[0].attention.n_heads == 12
    assert network.blocks[0].attention.head_dim == 32
    assert network.blocks[0].attention.use_flex_attention
    gin.clear_config()


if __name__ == "__main__":
    test_collate_builds_hop_aligned_128_note_roll()
    test_waveform_midi_lmdb_loads_without_codec_or_style_condition()
    test_rectified_flow_loss_backward_and_euler_sampling_shapes()
    test_midi_conditioning_is_concatenated_then_fused()
    test_frequency_scales_use_spe_noise_embedding_and_start_at_identity()
    test_frequency_scales_modulate_hidden_features_not_spectrograms()
    test_each_patcher_feature_is_concatenated_into_matching_depatcher_block()
    test_midi_and_style_dropout_are_independent_config_entries()
    test_style_source_is_one_of_encode_data_or_none_for_the_whole_run()
    test_layer_summary_has_parameters_and_runtime_shapes()
    test_trainer_runs_one_optimizer_step()
    test_audio_logging_generates_four_examples_at_5_and_20_steps()
    test_reference_gin_config_is_64_hop_and_256_bins()
