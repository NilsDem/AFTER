"""Tests for dense sliding-window latent extraction."""
import numpy as np

from after_scripts.prepare_dataset import encode_sliding_stats


class WindowMeanEncoder:

    def __init__(self, window_size):
        self.window_size = window_size
        self.batch_sizes = []

    def encode_stats(self, audio):
        self.batch_sizes.append(audio.shape[0])
        mean = audio.reshape(audio.shape[0], audio.shape[1], -1,
                             self.window_size).mean(dim=-1)
        return mean, mean.square()


def test_dense_latents_use_causal_padding_and_end_on_the_full_last_window():
    model = WindowMeanEncoder(window_size=4)
    audio = np.arange(1, 9, dtype=np.float32)[None, :]

    dense_mean, dense_variance = encode_sliding_stats(model, [audio], "cpu",
                                                      window_size=4,
                                                      hop_size=2,
                                                      batch_size=3)

    # Windows are [0, 0, 1, 2], [1, 2, 3, 4], [3, 4, 5, 6],
    # and [5, 6, 7, 8]. The result has n / hop time steps.
    expected = np.array([[[0.75, 2.5, 4.5, 6.5]]], dtype=np.float32)
    np.testing.assert_allclose(dense_mean, expected)
    np.testing.assert_allclose(dense_variance, expected**2)
    # The two phase-shifted streams have two outputs each and share one batch.
    assert model.batch_sizes == [2]


def test_dense_latent_length_is_original_length_times_ratio_over_hop():
    ratio = 16
    hop = 4
    original_latent_length = 5
    model = WindowMeanEncoder(window_size=ratio)
    audio = np.zeros((1, original_latent_length * ratio), dtype=np.float32)

    dense_mean, dense_variance = encode_sliding_stats(
        model, [audio], "cpu",
        window_size=ratio,
        hop_size=hop,
        batch_size=7)

    expected_shape = (1, 1, original_latent_length * ratio // hop)
    assert dense_mean.shape == expected_shape
    assert dense_variance.shape == expected_shape
    assert max(model.batch_sizes) <= 7


def test_dense_latents_require_one_code_per_window():
    model = WindowMeanEncoder(window_size=2)
    audio = np.zeros((1, 8), dtype=np.float32)

    try:
        encode_sliding_stats(model, [audio], "cpu",
                             window_size=4,
                             hop_size=2,
                             batch_size=2)
    except ValueError as error:
        assert "Expected encoder temporal downsampling ratio" in str(error)
    else:
        raise AssertionError("Expected multi-step window output to be rejected")


if __name__ == "__main__":
    test_dense_latents_use_causal_padding_and_end_on_the_full_last_window()
    test_dense_latent_length_is_original_length_times_ratio_over_hop()
    test_dense_latents_require_one_code_per_window()
