"""
Unified gin-configurable bottlenecks for PQMF and spectral autoencoders.
Replaces the scattered bottleneck definitions in SimpleNetsStream.py and AE2D_bottlenecks.py.

All bottlenecks implement:
  forward(x, **kwargs) -> (z, reg_loss)  — used during training
  forward_stream(x) -> z                 — used during streaming export
"""
import gin
import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor

from after.autoencoder.core import SimpleLatentReg


@gin.configurable
class TanhBottleneck(nn.Module):
    """Tanh-compressed bottleneck with optional noise injection."""

    def __init__(self, scale: int = 3, sigma: float = 0.) -> None:
        super().__init__()
        self.sigma = sigma
        self.scale = scale

    def forward(self, x: Tensor,
                apply_noise: bool = True) -> Tuple[Tensor, Tensor]:
        if apply_noise:
            x = self.scale * torch.tanh(x) + self.sigma * torch.randn_like(x)
        else:
            x = self.scale * torch.tanh(x)
        return x, torch.tensor(0.)

    def forward_stream(self, x: Tensor) -> Tensor:
        return self.scale * torch.tanh(x)


@gin.configurable
class ReluBottleneck(nn.Module):
    """Bottleneck with L2 regularisation loss (no tanh compression)."""

    def __init__(self, scale: int = 3, sigma: float = 0.) -> None:
        super().__init__()
        self.sigma = sigma
        self.reg_loss = SimpleLatentReg(scale=scale)

    def forward(self, x: Tensor, apply_noise: bool = False,
                return_mean: bool = False) -> Tuple[Tensor, Tensor]:
        reg_loss = self.reg_loss(x)
        fullx = x + self.sigma * torch.randn_like(x)
        if return_mean:
            return fullx, reg_loss, x
        return fullx, reg_loss

    def forward_stream(self, x: Tensor) -> Tensor:
        return x


@gin.configurable
class VAEBottleneck(nn.Module):
    """Variational bottleneck with KL divergence loss."""

    def __init__(self) -> None:
        super().__init__()

    @torch.jit.ignore
    def forward(self, z: Tensor,
                return_mean: bool = False) -> Tuple[Tensor, Tensor]:
        mean, scale = z.chunk(2, 1)
        std = nn.functional.softplus(scale) + 1e-2
        var = std * std
        logvar = torch.log(var)
        z = torch.randn_like(mean) * std + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()
        if return_mean:
            return z, kl, mean
        return z, kl

    def forward_stream(self, z: Tensor) -> Tensor:
        mean, scale = z.chunk(2, 1)
        std = nn.functional.softplus(scale) + 1e-2
        z = torch.randn_like(mean) * std + mean
        return z
