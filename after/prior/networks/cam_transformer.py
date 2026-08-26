import gin
import torch
from einops.layers.torch import Rearrange
from torch import nn

from after.diffusion.networks.rotary_embedding import RotaryEmbedding


@gin.configurable
class SOSReplacer(nn.Module):

    def __init__(self, dim, xavier_init: bool = True):
        super().__init__()
        self.sos = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        if xavier_init:
            nn.init.xavier_normal_(self.sos)

    def forward(self, x, replace_ratio: float):
        if replace_ratio == 0.0:
            return x
        batch_size = x.shape[0]
        num_replace = int(batch_size * replace_ratio)
        replace_indices = torch.randperm(batch_size,
                                         device=x.device)[:num_replace]
        x_sos = x.clone()
        x_sos[replace_indices] = self.sos
        return x_sos


def chunk_wise_causal_mask(seq_len: int, chunk_size: int):
    mask = torch.zeros(seq_len, seq_len)
    for i in range(0, seq_len, chunk_size):
        end = min(i + chunk_size, seq_len)
        mask[i:end, i:end] = 1
        mask[i:end, :i] = 1
    return 1 - mask


@gin.configurable
class MHAttention(nn.Module):

    def __init__(self,
                 is_causal: bool = False,
                 dropout_level: float = 0.0,
                 n_heads: int = 4,
                 max_cache_size: int = 0,
                 rotary_emb: nn.Module = None,
                 embed_dim: int = 256,
                 min_chunk_size: int = 1,
                 max_num_cache: int = 1,
                 max_batch_size: int = 4,
                 local_attention_size=None):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads
        if local_attention_size is not None and max_cache_size > 0:
            max_cache_size = local_attention_size
        self.max_cache_size = max_cache_size
        self.min_chunk_size = min_chunk_size
        self.rotary_emb = rotary_emb

        last_shape = (1, n_heads, 0, embed_dim // n_heads)
        self.register_buffer("last_k", torch.zeros(last_shape))
        self.register_buffer("last_v", torch.zeros(last_shape))
        cache_shape = (max_batch_size, max_num_cache, n_heads,
                       max_cache_size, embed_dim // n_heads)
        self.register_buffer("k_cache", torch.zeros(cache_shape))
        self.register_buffer("v_cache", torch.zeros(cache_shape))
        self.rearrange_heads1 = Rearrange("b n (h d) -> b h n d",
                                          h=n_heads)
        self.rearrange_heads2 = Rearrange("b h n d -> b n (h d)",
                                          h=n_heads)

    def get_buffers(self, cache_index: int):
        return (self.k_cache[:, cache_index],
                self.v_cache[:, cache_index])

    def set_buffers(self, k: torch.Tensor, v: torch.Tensor,
                    cache_index: int):
        self.k_cache[:k.shape[0], cache_index] = k
        self.v_cache[:v.shape[0], cache_index] = v

    def roll_cache(self, roll_size: int, cache_index: int):
        k_cache, v_cache = self.get_buffers(cache_index)
        k_cache = torch.cat(
            (k_cache[:self.last_k.shape[0]], self.last_k[:, :, :roll_size]),
            dim=2)
        v_cache = torch.cat(
            (v_cache[:self.last_v.shape[0]], self.last_v[:, :, :roll_size]),
            dim=2)
        k_cache = k_cache[:, :, -self.max_cache_size:]
        v_cache = v_cache[:, :, -self.max_cache_size:]
        self.set_buffers(k_cache, v_cache, cache_index)

    def reset_cache(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                cache_index: int):
        q = self.rearrange_heads1(q)
        k = self.rearrange_heads1(k)
        v = self.rearrange_heads1(v)

        if self.max_cache_size > 0:
            k_cache, v_cache = self.get_buffers(cache_index)
            full_k = torch.cat((k_cache[:k.shape[0]], k), dim=2)
            full_v = torch.cat((v_cache[:v.shape[0]], v), dim=2)
            full_k = full_k[:, :, -self.max_cache_size:]
            full_v = full_v[:, :, -self.max_cache_size:]
            self.last_k = k
            self.last_v = v
        else:
            full_k = k
            full_v = v

        if self.is_causal:
            mask = chunk_wise_causal_mask(full_k.shape[2],
                                          self.min_chunk_size)
            mask = mask[-q.shape[2]:]
            mask = mask.masked_fill(mask == 1, float("-inf")).to(k)
        else:
            mask = None

        if self.rotary_emb is not None:
            q, full_k = self.rotary_emb.rotate_queries_with_cached_keys(
                q, full_k)

        out = nn.functional.scaled_dot_product_attention(
            q,
            full_k,
            full_v,
            attn_mask=mask,
            is_causal=False,
            dropout_p=self.dropout_level if self.training else 0.0)
        return self.rearrange_heads2(out)


class SelfAttention(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 is_causal: bool,
                 dropout_level: float,
                 n_heads: int,
                 rotary_emb=None,
                 max_cache_size: int = 0,
                 min_chunk_size: int = 1):
        super().__init__()
        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.mha = MHAttention(is_causal=is_causal,
                               dropout_level=dropout_level,
                               n_heads=n_heads,
                               rotary_emb=rotary_emb,
                               embed_dim=embed_dim,
                               max_cache_size=max_cache_size,
                               min_chunk_size=min_chunk_size)

    def roll_cache(self, roll_size: int, cache_index: int):
        self.mha.roll_cache(roll_size, cache_index)

    def reset_cache(self):
        self.mha.reset_cache()

    def forward(self, x: torch.Tensor, cache_index: int):
        q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        return self.mha(q, k, v, cache_index)


class MLP(nn.Module):

    def __init__(self, embed_dim: int, multiplier: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class DecoderBlock(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 cond_dim: int,
                 is_causal: bool,
                 mlp_multiplier: int,
                 dropout: float,
                 rotary_emb=None,
                 max_cache_size: int = 0,
                 min_chunk_size: int = 1):
        super().__init__()
        self.cond_dim = cond_dim
        self.self_attention = SelfAttention(
            embed_dim,
            is_causal,
            dropout,
            n_heads=embed_dim // 64,
            rotary_emb=rotary_emb,
            max_cache_size=max_cache_size,
            min_chunk_size=min_chunk_size)
        self.mlp = MLP(embed_dim, mlp_multiplier, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.linear = (nn.Linear(cond_dim, 2 * embed_dim)
                       if cond_dim > 0 else nn.Identity())

    def roll_cache(self, roll_size: int, cache_index: int):
        self.self_attention.roll_cache(roll_size, cache_index)

    def reset_cache(self):
        self.self_attention.reset_cache()

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                cache_index: int):
        x = self.self_attention(self.norm1(x), cache_index) + x
        x = self.norm2(x)
        if self.cond_dim > 0:
            alpha, beta = self.linear(cond).chunk(2, dim=-1)
            x = x * (1 + alpha.unsqueeze(1)) + beta.unsqueeze(1)
        return self.mlp(self.norm3(x)) + x


class DenoiserTransBlock(nn.Module):

    def __init__(self,
                 n_channels: int,
                 out_dim: int,
                 seq_len: int,
                 mlp_multiplier: int,
                 embed_dim: int,
                 cond_dim: int,
                 dropout: float,
                 n_layers: int,
                 is_causal: bool,
                 pos_emb_type: str,
                 max_cache_size: int,
                 min_chunk_size: int):
        super().__init__()
        self.patchify_and_embed = nn.Sequential(
            Rearrange("b c t -> b t c"),
            nn.Linear(n_channels, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        if pos_emb_type == "learnable":
            self.pos_embed = nn.Embedding(seq_len, embed_dim)
            rotary_emb = None
        elif pos_emb_type == "rotary":
            self.pos_embed = None
            rotary_emb = RotaryEmbedding(32)
        elif pos_emb_type == "none":
            self.pos_embed = None
            rotary_emb = None
        else:
            raise ValueError(f"Unknown positional embedding: {pos_emb_type}")

        self.register_buffer("precomputed_pos_enc",
                             torch.arange(seq_len).long())
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embed_dim=embed_dim,
                         cond_dim=cond_dim,
                         is_causal=is_causal,
                         mlp_multiplier=mlp_multiplier,
                         dropout=dropout,
                         rotary_emb=rotary_emb,
                         max_cache_size=max_cache_size,
                         min_chunk_size=min_chunk_size)
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Sequential(nn.Linear(embed_dim, out_dim),
                                      Rearrange("b t c -> b c t"))

    def roll_cache(self, size: int, cache_index: int):
        for block in self.decoder_blocks:
            block.roll_cache(size, cache_index)

    def reset_cache(self):
        for block in self.decoder_blocks:
            block.reset_cache()

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                cache_index: int):
        x = self.patchify_and_embed(x)
        if self.pos_embed is not None:
            positions = self.precomputed_pos_enc[:x.shape[1]]
            x = x + self.pos_embed(positions).unsqueeze(0)
        for block in self.decoder_blocks:
            x = block(x, cond, cache_index)
        return self.out_proj(x)


@gin.configurable
class Denoiser(nn.Module):

    def __init__(self,
                 n_channels: int,
                 out_dim: int,
                 seq_len: int = 32,
                 embed_dim: int = 256,
                 cond_dim: int = 64,
                 n_layers: int = 6,
                 mlp_multiplier: int = 2,
                 dropout: float = 0.1,
                 causal: bool = False,
                 pos_emb_type: str = "learnable",
                 max_cache_size: int = 0,
                 min_chunk_size: int = 1):
        super().__init__()
        self.denoiser_trans_block = DenoiserTransBlock(
            out_dim=out_dim,
            n_channels=n_channels,
            seq_len=seq_len,
            mlp_multiplier=mlp_multiplier,
            embed_dim=embed_dim,
            dropout=dropout,
            n_layers=n_layers,
            cond_dim=cond_dim,
            is_causal=causal,
            pos_emb_type=pos_emb_type,
            max_cache_size=max_cache_size,
            min_chunk_size=min_chunk_size)

    def roll_cache(self, size: int, cache_index: int):
        self.denoiser_trans_block.roll_cache(size, cache_index)

    def reset_cache(self):
        self.denoiser_trans_block.reset_cache()

    def forward(self,
                x: torch.Tensor,
                cond: torch.Tensor,
                cache_index: int = 0):
        return self.denoiser_trans_block(x, cond, cache_index)
