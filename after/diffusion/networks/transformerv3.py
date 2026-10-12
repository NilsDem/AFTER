"""SwiGLU DiT with bounded, per-solver-stage RoPE history.

Training forward is stateless. Streaming stores unrotated keys and values,
then rotates the current window together, as in llc_spectral's prior.
"""
import math
from typing import Dict, Optional

import gin
import torch
from torch import nn
from torch.nn import functional as F


class RotaryAttention(nn.Module):
    keys: Dict[int, torch.Tensor]
    values: Dict[int, torch.Tensor]

    def __init__(self, dim, heads, window, causal):
        super().__init__()
        assert dim % heads == 0 and (dim // heads) % 2 == 0
        assert window > 0
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.causal = causal
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.output = nn.Linear(dim, dim)
        self.register_buffer('frequencies', 10000 ** (
            -torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.keys = torch.jit.annotate(Dict[int, torch.Tensor], {})
        self.values = torch.jit.annotate(Dict[int, torch.Tensor], {})

    def rope(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        positions = torch.arange(offset, offset + x.shape[-2], device=x.device)
        angles = positions[:, None] * self.frequencies[None, :]
        cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cosine - odd * sine,
                            even * sine + odd * cosine), -1).flatten(-2)

    def forward(self, tokens: torch.Tensor, stage: int = -1) -> torch.Tensor:
        batch, frames, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, frames, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        history = 0
        if stage >= 0:
            assert self.causal, 'Streaming requires causal attention'
            if stage in self.keys:
                previous = self.keys[stage]
                assert previous.shape[0] == batch, 'Reset caches when changing batch size'
                history = previous.shape[-2]
                k = torch.cat((previous, k), -2)
                v = torch.cat((self.values[stage], v), -2)
        lag = (torch.arange(frames, device=tokens.device)[:, None]
               - torch.arange(-history, frames, device=tokens.device)[None, :])
        mask = lag.abs() < self.window
        if self.causal:
            mask = mask & (lag >= 0)
        attended = F.scaled_dot_product_attention(
            self.rope(q, history), self.rope(k, 0), v, attn_mask=mask)
        if stage >= 0:
            start = max(0, k.shape[-2] - self.window + 1)
            self.keys[stage] = k[..., start:, :].detach().clone()
            self.values[stage] = v[..., start:, :].detach().clone()
        return self.output(attended.transpose(1, 2).reshape(batch, frames, dim))

    @torch.jit.export
    def reset_stream(self):
        self.keys.clear()
        self.values.clear()


class SwiGLU(nn.Module):
    def __init__(self, dim, multiplier, dropout):
        super().__init__()
        hidden = int(dim * multiplier)
        self.input = nn.Linear(dim, 2 * hidden)
        self.output = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(x).chunk(2, -1)
        return self.dropout(self.output(value * F.silu(gate)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, window, causal, multiplier, dropout):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = RotaryAttention(dim, heads, window, causal)
        self.mlp = SwiGLU(dim, multiplier, dropout)
        self.modulation = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor,
                stage: int) -> torch.Tensor:
        ashift, ascale, agate, mshift, mscale, mgate = self.modulation(condition).chunk(6, -1)
        x = x + agate * self.attention(self.norm_attention(x) * (1 + ascale) + ashift, stage)
        return x + mgate * self.mlp(self.norm_mlp(x) * (1 + mscale) + mshift)


@gin.configurable
class DenoiserV3(nn.Module):
    """[B,C,T] latents, [B,Ds,T] structure, [B,Dt] timbre -> velocity.

    meanflow=True adds an interval-width embedding. guidance_embed=True adds
    a trained-in MIDI guidance strength omega (default 1). streaming=True makes forward
    commit history immediately; roll_cache remains a no-op for existing exports.
    Reset history before changing solver steps, guidance batch, or sequence.
    """
    def __init__(self, n_channels, cond_dim, tcond_dim, embed_dim=384,
                 n_layers=6, num_heads=8, noise_embed_dims=64,
                 mlp_multiplier=3, dropout=0.1, causal=True,
                 local_attention_size=64, meanflow=False, streaming=False,
                 guidance_embed=False):
        super().__init__()
        assert noise_embed_dims >= 4 and noise_embed_dims % 2 == 0
        self.meanflow = meanflow
        self.guidance_embed = guidance_embed
        self.streaming = streaming
        self.input = nn.Linear(n_channels + tcond_dim, embed_dim)
        self.timbre = nn.Linear(cond_dim, embed_dim)
        frequencies = torch.exp(-math.log(10000.) * torch.arange(
            noise_embed_dims // 2) / (noise_embed_dims // 2))
        self.register_buffer('time_frequencies', frequencies)
        self.time_mlp = nn.Sequential(nn.Linear(noise_embed_dims, embed_dim), nn.SiLU(),
                                      nn.Linear(embed_dim, embed_dim))
        self.interval_mlp = nn.Sequential(nn.Linear(noise_embed_dims, embed_dim), nn.SiLU(),
                                          nn.Linear(embed_dim, embed_dim))
        self.omega_mlp = (nn.Sequential(
            nn.Linear(noise_embed_dims, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)) if guidance_embed else None)
        if self.omega_mlp is not None:
            nn.init.zeros_(self.omega_mlp[-1].weight)
            nn.init.zeros_(self.omega_mlp[-1].bias)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, local_attention_size, causal,
                             mlp_multiplier, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.output = nn.Linear(embed_dim, n_channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def embedding(self, t: torch.Tensor, scale: float = 100.) -> torch.Tensor:
        angles = t.reshape(-1, 1).float() * scale * self.time_frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), -1)

    def forward(self, x: torch.Tensor, time: torch.Tensor, cond: torch.Tensor,
                time_cond: torch.Tensor, cache_index: int = 0,
                interval: Optional[torch.Tensor] = None,
                omega: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.shape[-1] == time_cond.shape[-1], 'Structure must match latent frames'
        condition = self.time_mlp(self.embedding(time)) + self.timbre(cond)
        if self.meanflow:
            if interval is None:
                interval = torch.zeros_like(time)
            condition = condition + self.interval_mlp(self.embedding(interval))
        if self.omega_mlp is not None:
            strength = torch.ones_like(time) if omega is None else omega
            condition = condition + self.omega_mlp(self.embedding(strength, scale=1.))
        elif omega is not None:
            raise ValueError('omega requires guidance_embed=True')
        tokens = self.input(torch.cat((x, time_cond), 1).transpose(1, 2))
        stage = cache_index if self.streaming else -1
        for block in self.blocks:
            tokens = block(tokens, condition.unsqueeze(1), stage)
        return self.output(self.norm(tokens)).transpose(1, 2)

    @torch.jit.export
    def reset_stream(self):
        for block in self.blocks:
            block.attention.reset_stream()

    @torch.jit.export
    def roll_cache(self, size: int, cache_index: int):
        pass
