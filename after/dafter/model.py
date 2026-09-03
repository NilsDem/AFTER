"""Rectified-flow training and sampling for DAFTER."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import gin
import torch
from torch import nn


def pseudo_huber_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean pseudo-Huber penalty with sample-size-dependent transition scale."""
    if x.shape != y.shape:
        raise ValueError(
            f"pseudo-Huber inputs must have the same shape; got {tuple(x.shape)} "
            f"and {tuple(y.shape)}")
    if x.ndim < 1 or x.shape[0] == 0:
        raise ValueError("pseudo-Huber inputs must contain a batch dimension")

    transition = 0.00054 * math.sqrt(x[0].numel())
    residual = x - y
    elementwise_loss = torch.sqrt(residual.square() + transition**2) - transition
    return elementwise_loss.mean()


@gin.configurable
class DafterRectifiedFlow(nn.Module):
    """MIDI/style-conditioned rectified flow in complex STFT space.

    The flow follows the convention already used by AFTER: ``x0`` is Gaussian
    noise, ``x1`` is the clean target, and the network predicts ``x1 - x0`` at
    points on the straight interpolation between them.
    """

    def __init__(
        self,
        network: nn.Module,
        style_encoder: Optional[nn.Module] = None,
        style_condition_source: str = "encode",
        midi_dropout: float = 0.1,
        style_dropout: float = 0.1,
        freeze_style_encoder: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= midi_dropout <= 1.0:
            raise ValueError("midi_dropout must be between zero and one")
        if not 0.0 <= style_dropout <= 1.0:
            raise ValueError("style_dropout must be between zero and one")
        if style_condition_source not in {"encode", "data", "none"}:
            raise ValueError(
                "style_condition_source must be 'encode', 'data', or 'none'")
        if style_condition_source == "encode" and style_encoder is None:
            raise ValueError("style source 'encode' requires a style encoder")
        if style_condition_source != "encode" and style_encoder is not None:
            raise ValueError(
                "a style encoder is only used when style source is 'encode'")
        network_uses_style = bool(getattr(network, "use_style", True))
        if style_condition_source == "none" and network_uses_style:
            raise ValueError(
                "style source 'none' requires a network with use_style=False")
        if style_condition_source != "none" and not network_uses_style:
            raise ValueError(
                "style source 'encode' or 'data' requires use_style=True")

        self.network = network
        self.style_encoder = style_encoder
        self.style_condition_source = style_condition_source
        self.midi_dropout = float(midi_dropout)
        self.style_dropout = float(style_dropout)
        self.freeze_style_encoder = bool(freeze_style_encoder)

        if self.style_encoder is not None and self.freeze_style_encoder:
            self.style_encoder.requires_grad_(False)
            self.style_encoder.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def train(self, mode: bool = True):
        super().train(mode)
        if self.style_encoder is not None and self.freeze_style_encoder:
            self.style_encoder.eval()
        return self

    def audio_to_spectrum(self, waveform: torch.Tensor) -> torch.Tensor:
        """Transform clean waveform targets without constructing a grad graph."""
        with torch.no_grad():
            return self.network.audio_to_spectrum(waveform)

    def spectrum_to_audio(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Undo optional whitening and synthesize a waveform."""
        return self.network.spectrum_to_audio(spectrum)

    def resolve_style(
        self,
        style_waveform: Optional[torch.Tensor] = None,
        style_embedding: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Resolve the single style-conditioning source selected for this run."""
        if self.style_condition_source == "none":
            return None
        elif self.style_condition_source == "encode":
            if style_waveform is None:
                raise ValueError(
                    "style_waveform is required for style source 'encode'")
            if self.freeze_style_encoder:
                with torch.no_grad():
                    style = self.style_encoder(style_waveform)
            else:
                style = self.style_encoder(style_waveform)
        else:
            if style_embedding is None:
                raise ValueError(
                    "style_embedding is required for style source 'data'")
            style = style_embedding

        if style.ndim != 2 or style.shape[-1] != self.network.style_dim:
            raise ValueError(
                f"style must have shape [batch, {self.network.style_dim}]; "
                f"got {tuple(style.shape)}")
        return style

    def drop_conditions(
        self,
        midi: torch.Tensor,
        style: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Apply independent classifier-free dropout to MIDI and style."""
        batch_size = midi.shape[0]
        midi_mask = torch.rand(batch_size, device=midi.device) < self.midi_dropout
        if style is None:
            if self.style_condition_source != "none":
                raise ValueError("style cannot be None when style is enabled")
            style_mask = torch.zeros(batch_size,
                                     device=midi.device,
                                     dtype=torch.bool)
            dropped_style = None
        else:
            style_mask = (torch.rand(batch_size, device=style.device) <
                          self.style_dropout)
            dropped_style = style.masked_fill(style_mask[:, None], 0.0)
        dropped_midi = midi.masked_fill(midi_mask[:, None, None], 0.0)
        return dropped_midi, dropped_style, midi_mask, style_mask

    def forward(
        self,
        waveform: torch.Tensor,
        midi: torch.Tensor,
        style_waveform: Optional[torch.Tensor] = None,
        style_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        clean_spectrum = self.audio_to_spectrum(waveform)
        if clean_spectrum.shape[-1] != midi.shape[-1]:
            raise ValueError(
                "waveform STFT and MIDI conditioning have different lengths: "
                f"{clean_spectrum.shape[-1]} and {midi.shape[-1]}")

        style = self.resolve_style(style_waveform, style_embedding, midi)
        if self.training:
            midi, style, midi_mask, style_mask = self.drop_conditions(
                midi, style)
        else:
            midi_mask = torch.zeros(waveform.shape[0],
                                    device=waveform.device,
                                    dtype=torch.bool)
            style_mask = torch.zeros_like(midi_mask)

        noise = torch.randn_like(clean_spectrum)
        flow_time = torch.rand(clean_spectrum.shape[0],
                               1,
                               device=clean_spectrum.device,
                               dtype=clean_spectrum.dtype)
        broadcast_time = flow_time[:, :, None, None]
        interpolant = ((1.0 - broadcast_time) * noise +
                       broadcast_time * clean_spectrum)
        target_velocity = clean_spectrum - noise
        predicted_velocity = self.network(interpolant, midi, style, flow_time)
        # flow_loss = pseudo_huber_loss(predicted_velocity, target_velocity)

        flow_loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
        return {
            "loss": flow_loss,
            "flow_loss": flow_loss.detach(),
            "midi_drop_fraction": midi_mask.float().mean().detach(),
            "style_drop_fraction": style_mask.float().mean().detach(),
        }

    @torch.no_grad()
    def sample_spectrogram(
        self,
        midi: torch.Tensor,
        style: Optional[torch.Tensor],
        num_steps: int,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Integrate the learned vector field from noise (t=0) to audio (t=1)."""
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        expected_shape = (midi.shape[0], 2, self.network.spectral_bins,
                          midi.shape[-1])
        if initial_noise is None:
            spectrum = torch.randn(expected_shape,
                                   device=midi.device,
                                   dtype=midi.dtype)
        else:
            if tuple(initial_noise.shape) != expected_shape:
                raise ValueError(
                    f"initial_noise must have shape {expected_shape}; got "
                    f"{tuple(initial_noise.shape)}")
            spectrum = initial_noise.clone()

        step_size = 1.0 / float(num_steps)
        for step in range(num_steps):
            flow_time = torch.full((midi.shape[0], 1),
                                   step * step_size,
                                   device=midi.device,
                                   dtype=midi.dtype)
            spectrum.add_(step_size * self.network(spectrum, midi, style,
                                                   flow_time))
        return spectrum

    @torch.no_grad()
    def sample_audio(
        self,
        midi: torch.Tensor,
        style: Optional[torch.Tensor],
        num_steps: int,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        spectrum = self.sample_spectrogram(midi, style, num_steps,
                                           initial_noise)
        return self.spectrum_to_audio(spectrum)
