import gin
import torch
from torch import nn


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PositionalEmbedding(nn.Module):

    def __init__(self,
                 num_channels: int,
                 max_positions: int = 10000,
                 factor: float = 1000.0):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.factor = factor

    def forward(self, x: torch.Tensor):
        x = x * self.factor
        freqs = torch.arange(self.num_channels // 2,
                             device=x.device).float()
        freqs = freqs / (self.num_channels // 2 - 1)
        freqs = (1 / self.max_positions)**freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat((x.cos(), x.sin()), dim=1)


class ResBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels))

    def forward(self, x, y):
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        return x + gate * self.mlp(modulate(self.in_ln(x), shift, scale))


class FinalLayer(nn.Module):

    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels)
        self.linear = nn.Linear(model_channels, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels))

    def forward(self, x, condition):
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


@gin.configurable
class SimpleMLPAdaLN(nn.Module):

    def __init__(self,
                 in_channels: int,
                 model_channels: int,
                 out_channels: int,
                 z_channels: int,
                 cond_channels: int,
                 num_res_blocks: int,
                 noise_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.time_embed = nn.Sequential(
            PositionalEmbedding(noise_dim),
            nn.Linear(noise_dim, noise_dim),
            nn.SiLU(),
            nn.Linear(noise_dim, model_channels),
        )
        self.cond_embed = nn.Linear(z_channels + cond_channels,
                                    model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList(
            [ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.initialize_weights()

    def forward(self, x, c, t, cond):
        x = self.input_proj(x)
        if self.cond_channels > 0:
            c = torch.cat((c, cond), dim=-1)
        y = self.time_embed(t) + self.cond_embed(c)
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.res_blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
