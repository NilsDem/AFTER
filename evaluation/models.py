"""Evaluation models and autoencoder checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cached_conv as cc
import gin
import torch
from torch import nn

from after.autoencoder.latent_priors import SinusoidalEmbedding
from after.diffusion.networks.rotary_embedding import RotaryEmbedding


@dataclass
class AutoencoderRun:
    name: str
    path: Path
    model: nn.Module
    sample_rate: int
    hop_size: int
    latent_size: int
    checkpoint_step: int


def checkpoint_steps(path: Path) -> list[int]:
    steps = []
    for checkpoint in path.glob("checkpoint*.pt"):
        suffix = checkpoint.stem.removeprefix("checkpoint")
        if suffix.isdigit():
            steps.append(int(suffix))
    return sorted(steps)


def load_autoencoder(name: str, path: Path, device: torch.device,
                     step: int | None = None) -> AutoencoderRun:
    config_path = path / "config.gin"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")
    steps = checkpoint_steps(path)
    if not steps:
        raise FileNotFoundError(f"No checkpoint*.pt files in {path}")
    step = max(steps) if step is None else step
    checkpoint_path = path / f"checkpoint{step}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    gin.clear_config()
    cc.use_cached_conv(False)
    try:
        gin.parse_config_file(str(config_path))
        reference = gin.query_parameter("Trainer.model")
        model = reference.scoped_configurable_fn()
        sample_rate = int(gin.query_parameter("%SR"))
    finally:
        gin.clear_config()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = dict(checkpoint["model_state"])
    expected = set(model.state_dict())
    for key in list(state):
        if key not in expected and key.endswith(".cache"):
            del state[key]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Incompatible checkpoint {checkpoint_path}: missing "
            f"{incompatible.missing_keys}, unexpected {incompatible.unexpected_keys}")
    del checkpoint
    model = model.to(device).eval().requires_grad_(False)
    return AutoencoderRun(
        name=name,
        path=path,
        model=model,
        sample_rate=sample_rate,
        hop_size=int(model.time_transform.hop_size),
        latent_size=int(model.bottleneck_size),
        checkpoint_step=step,
    )


class RotarySelfAttention(nn.Module):
    """Full non-causal self-attention with rotary position embeddings."""

    def __init__(self, hidden_dim: int, rotary_embedding: RotaryEmbedding,
                 num_heads: int = 4):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        head_dim = hidden_dim // num_heads
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.rotary_embedding = rotary_embedding

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, hidden_dim = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self.rotary_embedding.rotate_queries_with_cached_keys(
            query, key)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, length, hidden_dim)
        return self.output(attended)


class NonCausalTransformerBlock(nn.Module):
    """Pre-LN transformer block with adaptive conditioning."""

    def __init__(self, hidden_dim: int, condition_dim: int,
                 rotary_embedding: RotaryEmbedding):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim,
                                           elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = nn.Linear(condition_dim, 4 * hidden_dim)
        self.attention = RotarySelfAttention(hidden_dim, rotary_embedding)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        attention_scale, attention_shift, mlp_scale, mlp_shift = (
            self.modulation(condition).chunk(4, dim=-1))
        attention_input = (self.attention_norm(tokens) *
                           (1.0 + attention_scale[:, None]) +
                           attention_shift[:, None])
        tokens = tokens + self.attention(attention_input)
        mlp_input = (self.mlp_norm(tokens) * (1.0 + mlp_scale[:, None]) +
                     mlp_shift[:, None])
        return tokens + self.mlp(mlp_input)


class NonCausalLatentTransformer(nn.Module):

    def __init__(self, latent_size: int, num_layers: int, hidden_dim: int,
                 condition_dim: int):
        super().__init__()
        num_heads = 4
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by 4 attention heads")
        self.rotary_embedding = RotaryEmbedding(hidden_dim // num_heads)
        self.input_projection = nn.Linear(latent_size, hidden_dim)
        self.blocks = nn.ModuleList([
            NonCausalTransformerBlock(
                hidden_dim, condition_dim, self.rotary_embedding)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_size)

    def forward(self, latent: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        tokens = self.input_projection(latent.transpose(1, 2))
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.output_projection(self.final_norm(tokens)).transpose(1, 2)


class ConditionalRectifiedFlow(nn.Module):
    """Instrument-conditioned non-causal latent flow with rotary attention."""

    def __init__(self, latent_size: int, num_instruments: int,
                 num_layers: int = 4, hidden_dim: int = 128,
                 noise_embedding_dim: int = 64):
        super().__init__()
        self.noise_embedding = SinusoidalEmbedding(noise_embedding_dim)
        self.noise_mlp = nn.Sequential(
            nn.Linear(noise_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.instrument_embedding = nn.Embedding(num_instruments, hidden_dim)
        self.transformer = NonCausalLatentTransformer(
            latent_size=latent_size,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            condition_dim=hidden_dim,
        )

    def predict_velocity(self, interpolant: torch.Tensor,
                         flow_time: torch.Tensor,
                         instrument: torch.Tensor) -> torch.Tensor:
        noise_level = 1.0 - flow_time
        condition = (self.noise_mlp(self.noise_embedding(noise_level)) +
                     self.instrument_embedding(instrument))
        return self.transformer(interpolant, condition)

    def loss(self, latent: torch.Tensor, instrument: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(latent)
        flow_time = torch.rand(latent.shape[0], device=latent.device,
                               dtype=latent.dtype)
        interpolant = ((1.0 - flow_time[:, None, None]) * noise +
                       flow_time[:, None, None] * latent)
        target = latent - noise
        return torch.nn.functional.mse_loss(
            self.predict_velocity(interpolant, flow_time, instrument), target)


class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_size: int, num_classes: int,
                 hidden_size: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.layers(embedding)


class LatentProbe(nn.Module):
    """Two small temporal convolutions followed by average pooling."""

    def __init__(self, latent_size: int, num_classes: int, hidden_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(latent_size, hidden_size, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(latent).squeeze(-1))
