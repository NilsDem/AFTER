"""Correctness smoke tests for the audio-space vector-field network."""
import torch

from after.dafter.network import (
    DafterNetwork,
    StreamingCausalSelfAttention,
    context_frames_for_seconds,
)


def build_model(flow_evaluations: int = 2):
    return DafterNetwork(
        nfft=256,
        hop_size=64,
        patch_ratio=16,
        patch_channels=4,
        hidden_channels=64,
        n_layers=2,
        n_heads=4,
        mlp_multiplier=2,
        midi_channels=8,
        style_channels=12,
        condition_width=16,
        attention_context_frames=8,
        max_flow_steps=flow_evaluations,
        max_batch_size=1,
        max_stream_frames=4,
    ).eval()


def test_cached_attention_matches_offline_and_is_bounded():
    torch.manual_seed(0)
    attention = StreamingCausalSelfAttention(
        embed_dim=32,
        n_heads=4,
        context_frames=4,
        max_flow_evaluations=2,
        max_batch_size=1,
        max_stream_frames=12,
    ).eval()
    x = torch.randn(1, 12, 32)

    with torch.no_grad():
        offline = attention(x)
        attention.reset_stream()
        streaming = torch.cat(
            [attention.forward_stream(x[:, i:i + 1], 0)
             for i in range(x.shape[1])],
            dim=1,
        )
    torch.testing.assert_close(streaming, offline, atol=1e-5, rtol=1e-5)
    positions = torch.arange(12)
    distances = positions[:, None] - positions[None, :]
    expected_mask = (distances >= 0) & (distances <= 4)
    assert torch.equal(attention.attention_mask[:12, :12], expected_mask)
    assert attention.rotary_cos.shape == (1024, 4)
    assert attention.rotary_sin.shape == (1024, 4)
    assert "attention_mask" not in attention.state_dict()
    assert "rotary_cos" not in attention.state_dict()
    assert attention.k_cache.shape[3] == 4
    assert attention.cache_valid[0].sum() == 4
    assert attention.cache_valid[1].sum() == 0
    assert attention.position_cache[0, 0] == 12
    assert attention.position_cache[1, 0] == 0

    changed_old_past = x.clone()
    changed_old_past[:, :-5] += 100.0
    with torch.no_grad():
        changed = attention(changed_old_past)
    torch.testing.assert_close(changed[:, -1],
                               offline[:, -1],
                               atol=1e-5,
                               rtol=1e-5)


def test_model_training_and_streaming_shapes_and_cache_reset():
    torch.manual_seed(1)
    model = build_model(flow_evaluations=2)
    spectrum = torch.randn(2, 2, 128, 6)
    conditioning = torch.randn(2, 8, 6)
    style = torch.randn(2, 12)
    flow_time = torch.rand(2, 1)
    vector_field = model(spectrum, conditioning, style, flow_time)
    assert vector_field.shape == spectrum.shape
    vector_field.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    model.zero_grad(set_to_none=True)
    noise_spectrum = torch.randn(1, 2, 128, 2)
    stream_conditioning = torch.randn(1, 8, 2)
    stream_style = torch.randn(1, 12)
    flow_times = torch.rand(1, 2, 1)
    with torch.no_grad():
        audio = model.forward_stream(noise_spectrum, stream_conditioning,
                                     stream_style, flow_times)
    assert audio.shape == (1, 1, 128)
    for block in model.blocks:
        assert block.attention.cache_valid[:, 0].sum(dim=1).tolist() == [2, 2]

    model.reset_stream()
    for block in model.blocks:
        assert not block.attention.cache_valid.any()

    scripted = torch.jit.script(model)
    with torch.no_grad():
        scripted_audio = scripted.forward_stream(noise_spectrum,
                                                  stream_conditioning,
                                                  stream_style,
                                                  flow_times)
    assert scripted_audio.shape == (1, 1, 128)


def test_patcher_strides_frequency_but_not_time_and_is_causal():
    model = build_model(flow_evaluations=1)
    assert len(model.patcher.downsample_blocks) == 4
    assert len(model.depatcher.upsample_blocks) == 4
    for block in model.patcher.downsample_blocks:
        assert block.conv.stride == (2, 1)
        assert block.conv.kernel_size == (4, 3)
    for block in model.depatcher.upsample_blocks:
        assert block.conv.stride == (2, 1)
        assert block.conv.in_channels == 2 * model.patcher.patch_channels

    spectrum = torch.randn(1, 2, model.spectral_bins, 9)
    with torch.no_grad():
        offline_tokens, offline_features = model.patcher(spectrum)
        model.patcher.reset_stream()
        stream_outputs = [
            model.patcher.forward_stream(spectrum[..., start:start + 3], 0)
            for start in range(0, spectrum.shape[-1], 3)
        ]
        streaming_tokens = torch.cat([output[0] for output in stream_outputs],
                                     dim=1)
        streaming_features = [
            torch.cat([output[1][level] for output in stream_outputs], dim=-1)
            for level in range(len(offline_features))
        ]
    torch.testing.assert_close(streaming_tokens,
                               offline_tokens,
                               atol=1e-5,
                               rtol=1e-5)
    for streaming, offline in zip(streaming_features, offline_features):
        torch.testing.assert_close(streaming,
                                   offline,
                                   atol=1e-5,
                                   rtol=1e-5)


def test_four_tenths_context_conversion():
    assert context_frames_for_seconds(0.4, 44100, 64) == 276


if __name__ == "__main__":
    test_cached_attention_matches_offline_and_is_bounded()
    test_model_training_and_streaming_shapes_and_cache_reset()
    test_patcher_strides_frequency_but_not_time_and_is_causal()
    test_four_tenths_context_conversion()
