from functools import partial
import torch
from einops.layers.torch import Rearrange
from torch import nn
from einops import rearrange
import gin
from .rotary_embedding import RotaryEmbedding


class PositionalEmbedding(nn.Module):

    def __init__(
        self,
        num_channels: int,
        max_positions: int,
        factor: float,
        endpoint: bool = False,
        rearrange: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.factor = factor
        self.rearrange = (Rearrange("b (f c) -> b (c f)", f=2)
                          if rearrange else nn.Identity())

    def forward(self, x: torch.Tensor):
        x = x.view(-1)
        x = x * self.factor
        freqs = torch.arange(
            start=0,
            end=self.num_channels // 2,
            device=x.device,
        ).float()
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions)**freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return self.rearrange(x)


def sliding_window_causal_mask(seq_len: int, window_size: int, device):
    """
    Returns a [seq_len, seq_len] attention mask (float), where each token attends
    only to the last `window_size` tokens including itself.
    """
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    dist = i - j  # shape [seq_len, seq_len]
    mask = (dist < 0) | (dist >= window_size)  # bool: True if outside window
    return mask.masked_fill(mask, float('-inf'))


class RotaryMHSelfAttention(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 is_causal: bool,
                 p_drop: float,
                 head_dim: int,
                 qkv_bias: bool = False,
                 window_size: None | int = None):
        super().__init__()
        self.is_causal = is_causal
        self.window_size = window_size
        self.p_drop = p_drop
        self.n_heads = embed_dim // head_dim
        self.rotary_emb = RotaryEmbedding(head_dim)
        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        nn.init.xavier_normal_(self.qkv_linear.weight)
        if qkv_bias:
            nn.init.constant_(self.qkv_linear.bias, 0)

    def forward(self, x, attn_mask=None):
        bs, seq_len, _ = x.shape

        if self.window_size is not None:
            # Create sliding window mask
            causal_mask = sliding_window_causal_mask(seq_len, self.window_size,
                                                     x.device)
            # Expand to [bs * heads, seq_len, seq_len]
            causal_mask = causal_mask.unsqueeze(0).expand(
                bs, self.n_heads, -1, -1)
            attn_mask = causal_mask

        q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        q, k, v = [
            rearrange(x, "bs n (h d) -> bs h n d", h=self.n_heads)
            for x in (q, k, v)
        ]
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        out = nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            is_causal=False if self.window_size is not None else
            self.is_causal,  # because we're doing masking ourselves
            dropout_p=self.p_drop if self.training else 0,
        )
        out = rearrange(out, "bs h n d -> bs n (h d)", h=self.n_heads)
        return out


class MLP(nn.Module):

    def __init__(self, embed_dim: int, mlp_multiplier: int, p_drop: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return self.mlp(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        is_causal: bool,
        mlp_multiplier: int,
        p_drop: float,
        head_dim: int,
        norm_cls,
        window_size: int,
    ):
        super().__init__()
        self.self_attention = RotaryMHSelfAttention(embed_dim,
                                                    is_causal,
                                                    p_drop,
                                                    head_dim,
                                                    window_size=window_size)
        self.mlp = MLP(embed_dim, mlp_multiplier, p_drop)
        self.norm1 = norm_cls(embed_dim)
        self.norm2 = norm_cls(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attention(self.norm1(x)) + x
        x = self.mlp(self.norm2(x)) + x
        return x


@gin.configurable
class TransformerEncoder(nn.Module):

    def __init__(
        self,
        n_channels: int,
        mlp_multiplier: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        head_dim: int,
        seq_len: int | None = None,
        is_causal: bool = True,
        output_dim: int = None,
        norm_cls=nn.LayerNorm,
        window_size: int | None = None,
        out_fn: str = None,
        use_tanh=False,
        regularisation: str = "none",
    ):
        super().__init__()
        self._out_dim = output_dim if output_dim is not None else n_channels
        self._seq_len = seq_len
        if use_tanh:
            self.out_fn = torch.tanh
        else:
            self.out_fn = nn.Identity()
        if regularisation == "vae":
            self._out_dim *= 2
        self.regularisation = regularisation

        # partial(
        #     crop_to_seq_len,
        #     seq_len=seq_len) if seq_len is not None else identity

        self.patchify_and_embed = nn.Sequential(
            Rearrange("b c t -> b t c"),
            nn.Linear(n_channels, embed_dim),
            norm_cls(embed_dim),
        )
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                embed_dim=embed_dim,
                mlp_multiplier=mlp_multiplier,
                is_causal=is_causal,
                p_drop=dropout,
                head_dim=head_dim,
                norm_cls=norm_cls,
                window_size=window_size,
            ) for _ in range(n_layers)
        ])
        self.out_proj = nn.Sequential(nn.Linear(embed_dim, self.out_dim),
                                      Rearrange("b t c -> b c t"))

    def forward(self, x, return_full=False):

        x = self.patchify_and_embed(x)
        for block in self.decoder_blocks:
            x = block(x)

        x = self.out_fn(self.out_proj(x))

        if self.regularisation == "vae":
            mean, scale = x.chunk(2, 1)

            std = nn.functional.softplus(scale) + 1e-4
            var = std * std
            logvar = torch.log(var)

            x = torch.randn_like(mean) * std + mean
            kl = (mean * mean + var - logvar - 1).sum(1).mean()

        elif self.regularisation == "none":
            kl = torch.tensor(0.).to(x)
            mean = x

        if return_full:
            return x, mean, kl
        return x

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def out_dim(self):
        return self._out_dim


if __name__ == "__main__":

    denoiser = TransformerEncoder(
        n_channels=64,
        mlp_multiplier=4,
        embed_dim=256,
        dropout=0.1,
        n_layers=6,
        head_dim=64,
        seq_len=128,
        is_causal=True,
        window_size=16,
    )

    n_params = sum(p.numel() for p in denoiser.parameters()) / 1e6
    print(f"Number of parameters: {n_params:.2f}M")
    x = torch.randn(16, 64, 128)
    out = denoiser(x)

    print(f"Output shape: {out.shape}")

    traced = torch.jit.trace(denoiser, x)
