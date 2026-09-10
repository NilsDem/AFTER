"""Reconstruction and embedding metrics used by the evaluation."""

from __future__ import annotations

import math

import numpy as np
import torch
import torchaudio
from scipy.linalg import sqrtm

from after.autoencoder.core import k_weighting


def si_sdr(reference: torch.Tensor, estimate: torch.Tensor,
           epsilon: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB, returned once per batch item."""
    reference = reference.flatten(1)
    estimate = estimate.flatten(1)
    reference = reference - reference.mean(dim=1, keepdim=True)
    estimate = estimate - estimate.mean(dim=1, keepdim=True)
    scale = ((estimate * reference).sum(dim=1, keepdim=True) /
             (reference.square().sum(dim=1, keepdim=True) + epsilon))
    target = scale * reference
    noise = estimate - target
    return 10.0 * torch.log10(
        (target.square().sum(dim=1) + epsilon) /
        (noise.square().sum(dim=1) + epsilon))


class MelSTFTDistance(torch.nn.Module):
    """Per-example version of AFTER's multi-scale mel-STFT training distance."""

    def __init__(self, sample_rate: int,
                 scales=(64, 128, 256, 512, 1024, 2048),
                 mel_bands: int = 64, use_k_weighting: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.scales = tuple(scales)
        self.use_k_weighting = use_k_weighting
        self.spectrograms = torch.nn.ModuleList([
            torchaudio.transforms.Spectrogram(
                n_fft=scale, hop_length=scale // 4, power=None,
                normalized=True, center=False, pad_mode=None)
            for scale in self.scales
        ])
        self.mel_scales = torch.nn.ModuleList([
            torchaudio.transforms.MelScale(
                n_mels=mel_bands, sample_rate=sample_rate,
                n_stft=scale // 2 + 1)
            for scale in self.scales
        ])

    def forward(self, reference: torch.Tensor,
                estimate: torch.Tensor) -> torch.Tensor:
        if self.use_k_weighting:
            reference = k_weighting(reference, self.sample_rate)
            estimate = k_weighting(estimate, self.sample_rate)
        distance = reference.new_zeros(reference.shape[0])
        for scale, spectrogram, mel_scale in zip(
                self.scales, self.spectrograms, self.mel_scales):
            reference_mel = mel_scale(spectrogram(reference).abs())
            estimate_mel = mel_scale(spectrogram(estimate).abs())
            magnitude = (reference_mel - estimate_mel).abs().flatten(1).mean(1)
            log_magnitude = ((torch.log1p(reference_mel) -
                              torch.log1p(estimate_mel)).square()
                             .flatten(1).mean(1))
            distance += magnitude + math.sqrt(scale / 2) * log_magnitude
        return distance


def clap_fad(reference: torch.Tensor, estimate: torch.Tensor,
             epsilon: float = 1e-6) -> float:
    """Fréchet distance between two matrices of CLAP embeddings."""
    if reference.ndim != 2 or estimate.ndim != 2:
        raise ValueError("CLAP embeddings must have shape [examples, embedding]")
    if len(reference) < 2 or len(estimate) < 2:
        raise ValueError("CLAP FAD needs at least two embeddings per distribution")
    reference = reference.double().cpu().numpy()
    estimate = estimate.double().cpu().numpy()
    mean_reference = reference.mean(axis=0)
    mean_estimate = estimate.mean(axis=0)
    covariance_reference = np.cov(reference, rowvar=False)
    covariance_estimate = np.cov(estimate, rowvar=False)
    identity = np.eye(reference.shape[1])
    covariance_mean = sqrtm(
        (covariance_reference + epsilon * identity) @
        (covariance_estimate + epsilon * identity))
    if np.iscomplexobj(covariance_mean):
        imaginary = np.abs(covariance_mean.imag).max()
        if imaginary > 1e-5:
            raise ValueError(f"FAD covariance square root has imaginary part {imaginary}")
        covariance_mean = covariance_mean.real
    mean_distance = np.square(mean_reference - mean_estimate).sum()
    covariance_distance = np.trace(
        covariance_reference + covariance_estimate - 2 * covariance_mean)
    return float(max(0.0, mean_distance + covariance_distance))


def classification_metrics(logits: torch.Tensor,
                           labels: torch.Tensor) -> dict[str, float]:
    prediction = logits.argmax(dim=1)
    accuracy = (prediction == labels).float().mean().item()
    recalls = []
    for label in labels.unique():
        mask = labels == label
        recalls.append((prediction[mask] == label).float().mean())
    return {
        "accuracy": accuracy,
        "balanced_accuracy": torch.stack(recalls).mean().item(),
    }

