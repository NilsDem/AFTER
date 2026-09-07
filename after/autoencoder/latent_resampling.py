"""Resampling helpers for teacher-latent conditioning."""

import torch


def causal_linear_upsample(
    latent: torch.Tensor,
    ratio: int,
    output_length: int,
) -> torch.Tensor:
    """Linearly upsample without reading a teacher frame before it is ready.

    Teacher frame ``k`` represents the audio window ending at
    ``(k + 1) * ratio`` student steps.  Linear interpolation is delayed by one
    teacher interval, allowing both endpoints to be known.  Values before the
    first complete teacher window are zero.
    """
    if latent.ndim < 1 or latent.shape[-1] == 0:
        raise ValueError("latent must contain at least one teacher frame")
    if ratio < 1:
        raise ValueError("ratio must be positive")
    if output_length < 0:
        raise ValueError("output_length must be non-negative")

    position = torch.arange(output_length, device=latent.device)
    delayed = (position - (ratio - 1)).clamp_min(0)
    right = torch.div(delayed, ratio, rounding_mode="floor")
    right = right.clamp_max(latent.shape[-1] - 1)
    left = (right - 1).clamp_min(0)
    phase = torch.remainder(delayed, ratio).to(latent.dtype) / ratio
    phase = phase.reshape((1,) * (latent.ndim - 1) + (output_length,))

    start = latent.index_select(-1, left)
    end = latent.index_select(-1, right)
    dense = torch.lerp(start, end, phase)
    valid = position >= ratio - 1
    return dense * valid.to(latent.dtype)
