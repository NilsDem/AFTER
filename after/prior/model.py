from typing import Dict

import gin
import torch
from einops import rearrange
from torch import nn


def add_noise_convex(x: torch.Tensor):
    amount = torch.rand(x.shape[0], 1, x.shape[-1], device=x.device)
    return amount * torch.randn_like(x) + (1 - amount) * x


def add_noise_linear(x: torch.Tensor):
    amount = torch.rand(x.shape[0], 1, x.shape[-1], device=x.device)
    return x + amount * torch.randn_like(x)


@gin.configurable
class Prior(nn.Module):

    def __init__(self,
                 net: nn.Module,
                 predictor: nn.Module,
                 sos_token: nn.Module,
                 conditioner: nn.Module = None,
                 rf_batch_mul: int = 2,
                 p_drop_sos: float = 0.1,
                 add_train_noise: str = "convex"):
        super().__init__()
        if add_train_noise not in ("none", "linear", "convex"):
            raise ValueError(f"Unknown noise mode: {add_train_noise}")
        self.net = net
        self.predictor = predictor
        self.sos_token = sos_token
        self.conditioner = conditioner
        self.rf_batch_mul = rf_batch_mul
        self.p_drop_sos = p_drop_sos
        self.add_train_noise = add_train_noise

    def encode_condition(self, x: torch.Tensor):
        if self.conditioner is None:
            condition = torch.empty(x.shape[0], 0, device=x.device)
            regularization = torch.zeros((), device=x.device)
        else:
            condition, _, regularization = self.conditioner(x,
                                                             return_full=True)
        return condition, regularization

    def noise(self, x: torch.Tensor):
        if self.add_train_noise == "convex":
            return add_noise_convex(x)
        if self.add_train_noise == "linear":
            return add_noise_linear(x)
        return x

    def rf_loss(self, target: torch.Tensor, z: torch.Tensor,
                cond: torch.Tensor):
        target = rearrange(target, "b c t -> (b t) c")
        z = rearrange(z, "b d t -> (b t) d")
        if cond.shape[1] > 0:
            cond = cond.unsqueeze(-1).repeat(1, 1, target.shape[0] //
                                             cond.shape[0])
            cond = rearrange(cond, "b d t -> (b t) d")

        z = self.sos_token(z, replace_ratio=self.p_drop_sos)
        if self.rf_batch_mul > 1:
            target = target.repeat(self.rf_batch_mul, 1)
            z = z.repeat(self.rf_batch_mul, 1)
            cond = cond.repeat(self.rf_batch_mul, 1)

        time = torch.rand(target.shape[0], 1, device=target.device)
        base = torch.randn_like(target)
        x_t = (1 - time) * base + time * target
        velocity = self.predictor(x_t, z, time.squeeze(1), cond=cond)
        return (velocity - (target - base)).square().mean()

    def training_step(self, x: torch.Tensor,
                      x_condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        condition, regularization = self.encode_condition(x_condition)
        z = self.net(self.noise(x)[..., :-1], cond=condition)
        flow = self.rf_loss(x[..., 1:], z, condition)
        return {"flow": flow, "regularization": regularization}

    @torch.no_grad()
    def sample_token(self,
                     z: torch.Tensor,
                     cond: torch.Tensor,
                     steps: int = 20,
                     temperature: float = 1.0):
        x = temperature * torch.randn(z.shape[0],
                                      self.predictor.in_channels,
                                      device=z.device)
        dt = 1.0 / steps
        for step in range(steps):
            time = torch.full((x.shape[0], ),
                              step / steps,
                              device=x.device,
                              dtype=x.dtype)
            x = x + dt * self.predictor(x, z, time, cond=cond)
        return x

    @torch.no_grad()
    def sample(self,
               cond: torch.Tensor,
               target_len: int,
               steps: int = 20,
               temperature: float = 1.0):
        z = self.sos_token.sos.repeat(cond.shape[0], 1)
        x = self.sample_token(z, cond, steps, temperature).unsqueeze(-1)
        for _ in range(target_len - 1):
            z = self.net(x, cond=cond)[..., -1]
            x_next = self.sample_token(z, cond, steps, temperature)
            x = torch.cat((x, x_next.unsqueeze(-1)), dim=-1)
        return x

