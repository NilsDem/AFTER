"""Distribution regularisers shared by SSL and diffusion encoders."""
import gin
import torch
from torch import nn


@gin.configurable
class SIGReg(nn.Module):
    """Match random 1D projections to a standard Gaussian characteristic function.

    Inputs are [B,D], or [...,B,D]. Both projection and statistic run in FP32.
    """
    def __init__(self, knots=17, num_projections=64):
        super().__init__()
        assert knots > 1 and num_projections > 0
        self.num_projections = num_projections
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # Static device names also make these contexts TorchScript-compatible.
        with torch.autocast('cuda', enabled=False), torch.autocast('cpu', enabled=False):
            directions = torch.randn(proj.size(-1), self.num_projections,
                                     device=proj.device, dtype=torch.float32)
            directions = directions / directions.norm(p=2, dim=0)
            x_t = (proj.float() @ directions).unsqueeze(-1) * self.t
            err = ((x_t.cos().mean(-3) - self.phi).square()
                   + x_t.sin().mean(-3).square())
            statistic = (err @ self.weights) * proj.size(-2)
            return statistic.mean()
