"""Small causal priors used to regularise autoencoder latents."""

import math
from typing import Optional

import gin
import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.reshape(-1)
        frequencies = torch.exp(
            -math.log(10_000.) *
            torch.arange(self.dim // 2, device=value.device,
                         dtype=value.dtype) / (self.dim // 2 - 1))
        phase = value[:, None] * frequencies[None] * 100.
        return torch.cat((phase.cos(), phase.sin()), dim=-1)


def _sinusoidal_positions(length: int, dim: int, reference: torch.Tensor):
    positions = torch.arange(length,
                             device=reference.device,
                             dtype=reference.dtype)
    frequencies = torch.exp(
        -math.log(10_000.) *
        torch.arange(dim // 2, device=reference.device,
                     dtype=reference.dtype) / (dim // 2 - 1))
    phase = positions[:, None] * frequencies[None]
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class CausalTransformerBlock(nn.Module):
    """Pre-LN transformer block with optional adaptive layer normalisation."""

    def __init__(self, hidden_dim: int, condition_dim: Optional[int]):
        super().__init__()
        self.condition_dim = condition_dim
        self.attention_norm = nn.LayerNorm(
            hidden_dim, elementwise_affine=condition_dim is None)
        self.mlp_norm = nn.LayerNorm(
            hidden_dim, elementwise_affine=condition_dim is None)
        self.modulation = (None if condition_dim is None else
                           nn.Linear(condition_dim, 4 * hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim,
                                               num_heads=4,
                                               batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor,
                condition: Optional[torch.Tensor]) -> torch.Tensor:
        attention_input = self.attention_norm(tokens)
        if self.modulation is not None:
            attention_scale, attention_shift, mlp_scale, mlp_shift = (
                self.modulation(condition).chunk(4, dim=-1))
            attention_input = (attention_input *
                               (1. + attention_scale[:, None]) +
                               attention_shift[:, None])
        attended = self.attention(attention_input,
                                  attention_input,
                                  attention_input,
                                  attn_mask=attention_mask,
                                  need_weights=False)[0]
        tokens = tokens + attended

        mlp_input = self.mlp_norm(tokens)
        if self.modulation is not None:
            mlp_input = (mlp_input * (1. + mlp_scale[:, None]) +
                         mlp_shift[:, None])
        return tokens + self.mlp(mlp_input)


class CausalLatentTransformer(nn.Module):

    def __init__(self, latent_size: int, num_layers: int, hidden_dim: int,
                 attention_window: int,
                 condition_dim: Optional[int] = None):
        super().__init__()
        self.attention_window = attention_window
        self.input_projection = nn.Linear(latent_size, hidden_dim)
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(hidden_dim, condition_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_size)

    def _attention_mask(self, length: int, device: torch.device):
        positions = torch.arange(length, device=device)
        distance = positions[:, None] - positions[None, :]
        return (distance < 0) | (distance >= self.attention_window)

    def forward(self, latent: torch.Tensor,
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.input_projection(latent.transpose(1, 2))
        tokens = tokens + _sinusoidal_positions(
            tokens.shape[1], tokens.shape[2], tokens)[None]
        attention_mask = self._attention_mask(tokens.shape[1], tokens.device)
        for block in self.blocks:
            tokens = block(tokens, attention_mask, condition)
        return self.output_projection(self.final_norm(tokens)).transpose(1, 2)


@gin.configurable
class RectifiedFlowLatentPrior(nn.Module):
    """Causal rectified-flow vector field over continuous latent sequences."""

    def __init__(self,
                 latent_size: int,
                 num_layers: int = 2,
                 hidden_dim: int = 128,
                 noise_levels_dim: int = 64,
                 attention_window: int = 64):
        super().__init__()
        self.noise_embedding = SinusoidalEmbedding(noise_levels_dim)
        self.noise_mlp = nn.Sequential(
            nn.Linear(noise_levels_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.transformer = CausalLatentTransformer(
            latent_size=latent_size,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            attention_window=attention_window,
            condition_dim=hidden_dim,
        )

    def predict_velocity(self, interpolant: torch.Tensor,
                         flow_time: torch.Tensor) -> torch.Tensor:
        noise_level = 1. - flow_time
        condition = self.noise_mlp(self.noise_embedding(noise_level))
        return self.transformer(interpolant, condition)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(latent)
        flow_time = torch.rand(latent.shape[0],
                               device=latent.device,
                               dtype=latent.dtype)
        interpolant = ((1. - flow_time[:, None, None]) * noise +
                       flow_time[:, None, None] * latent)
        target_velocity = latent - noise
        prediction = self.predict_velocity(interpolant, flow_time)
        return F.mse_loss(prediction, target_velocity)


@gin.configurable
class AutoregressiveLatentPrior(nn.Module):
    """Continuous next-latent prediction with a causal transformer."""

    def __init__(self,
                 latent_size: int,
                 num_layers: int = 2,
                 hidden_dim: int = 128,
                 attention_window: int = 64):
        super().__init__()
        self.transformer = CausalLatentTransformer(
            latent_size=latent_size,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            attention_window=attention_window,
        )

    def predict_next(self, latent: torch.Tensor) -> torch.Tensor:
        return self.transformer(latent)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        prediction = self.predict_next(latent[..., :-1])
        return F.mse_loss(prediction, latent[..., 1:])
