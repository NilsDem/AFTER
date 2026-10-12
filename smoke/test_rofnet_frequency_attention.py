"""Frequency attention can be omitted from high-resolution RofNet stages."""

import torch

from after.autoencoder.audio import CausalMauerSTFT
from after.autoencoder.networks.RofNet import FrequencyAttention, RofNet


def make_model(skipped_stages):
    return RofNet(
        dims=(8, 8, 8),
        depths=(1, 1, 1),
        heads=(2, 2, 2),
        freq_ratios=(4, 4),
        middle_dim=8,
        middle_heads=2,
        middle_layers=1,
        bottleneck_size=4,
        freq_size=128,
        skip_frequency_attention_stages=skipped_stages,
        time_transform=CausalMauerSTFT(
            nfft=256,
            hop_size=64,
            synthesis_length=128,
            zero_length=64,
            skip_features=-1,
        ),
    ).eval()


def test_frequency_attention_is_skipped_symmetrically():
    model = make_model(2)

    for stages in (model.encoder_stages, model.decoder_stages):
        assert isinstance(stages[0][0].freq_attention, torch.nn.Identity)
        assert isinstance(stages[1][0].freq_attention, torch.nn.Identity)
        assert isinstance(stages[2][0].freq_attention, FrequencyAttention)

    audio = torch.randn(1, 1, 1024)
    with torch.no_grad():
        reconstruction = model(audio, return_all=False)
    assert reconstruction.shape == audio.shape


def test_skipped_frequency_attention_is_not_an_extra_residual():
    block = make_model(1).encoder_stages[0][0]
    x = torch.randn(1, 2, 128, 8)
    with torch.no_grad():
        expected = x + block.time_attention(
            block.time_norm(x).transpose(1, 2)).transpose(1, 2)
        expected = expected + block.mlp(block.mlp_norm(expected))
        actual = block(x)
    torch.testing.assert_close(actual, expected)


def test_frequency_attention_is_enabled_by_default():
    model = make_model(0)
    for stages in (model.encoder_stages, model.decoder_stages):
        assert all(isinstance(stage[0].freq_attention, FrequencyAttention)
                   for stage in stages)


def test_invalid_number_of_skipped_stages_is_rejected():
    try:
        make_model(4)
    except ValueError as error:
        assert "skip_frequency_attention_stages" in str(error)
    else:
        raise AssertionError("Expected an invalid skipped-stage count to fail")


if __name__ == "__main__":
    test_frequency_attention_is_skipped_symmetrically()
    test_skipped_frequency_attention_is_not_an_extra_residual()
    test_frequency_attention_is_enabled_by_default()
    test_invalid_number_of_skipped_stages_is_rejected()
