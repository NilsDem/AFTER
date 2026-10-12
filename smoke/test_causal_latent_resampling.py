"""Causal teacher-latent interpolation must not expose future frames."""

import torch

from after.autoencoder.latent_resampling import causal_linear_upsample


def test_first_teacher_frame_waits_for_its_complete_window():
    latent = torch.tensor([[[4.0, 8.0, 12.0]]])
    dense = causal_linear_upsample(latent, ratio=4, output_length=12)

    torch.testing.assert_close(dense[..., :3], torch.zeros(1, 1, 3))
    torch.testing.assert_close(dense[..., 3:8], torch.full((1, 1, 5), 4.0))
    torch.testing.assert_close(
        dense[..., 8:12], torch.tensor([[[5.0, 6.0, 7.0, 8.0]]]))


def test_future_teacher_frame_cannot_change_earlier_conditioning():
    original = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    changed = original.clone()
    changed[..., 2] = 1000.0

    original_dense = causal_linear_upsample(original, 4, 16)
    changed_dense = causal_linear_upsample(changed, 4, 16)

    # Teacher frame 2 is complete at dense step 11 and only affects the
    # delayed interpolation beginning on the following step.
    torch.testing.assert_close(original_dense[..., :12], changed_dense[..., :12])
    assert not torch.equal(original_dense[..., 12:], changed_dense[..., 12:])


if __name__ == "__main__":
    test_first_teacher_frame_waits_for_its_complete_window()
    test_future_teacher_frame_cannot_change_earlier_conditioning()
