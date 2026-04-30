import torch
from einops.layers.torch import Rearrange
from torch import nn
from typing import Optional
import gin
from .rotary_embedding import RotaryEmbedding


# -----------------------------
# Fast, scriptable positional embedding
# -----------------------------
class PositionalEmbedding(nn.Module):
    """
    Fourier features for scalar time inputs.
    Optimizations:
      - precompute freqs buffer once
      - avoid torch.arange + ger each forward
    """

    def __init__(
        self,
        num_channels: int,
        max_positions: int,
        factor: float,
        endpoint: bool = False,
        rearrange: bool = False,
    ):
        super().__init__()
        assert num_channels % 2 == 0, "num_channels must be even"
        self.num_channels = int(num_channels)
        self.max_positions = int(max_positions)
        self.endpoint = bool(endpoint)
        self.factor = float(factor)

        self.rearrange = (Rearrange("b (f c) -> b (c f)", f=2)
                          if rearrange else nn.Identity())

        # Precompute frequency exponents once (buffer)
        half = self.num_channels // 2
        denom = float(half - (1 if self.endpoint else 0))
        freqs = torch.arange(0, half, dtype=torch.float32) / denom
        freqs = (1.0 / float(self.max_positions))**freqs  # shape [half]
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: scalar or [B] or [B, ...] -> flatten to [B]
        x = x.reshape(-1).to(dtype=self.freqs.dtype) * self.factor  # [B]
        # Outer product via broadcasting: [B, 1] * [1, half] -> [B, half]
        ang = x[:, None] * self.freqs[None, :]
        out = torch.cat([ang.cos(), ang.sin()], dim=1)  # [B, num_channels]
        return self.rearrange(out)


# -----------------------------
# Vectorized mask precompute (no Python loops in forward)
# -----------------------------
@torch.jit.ignore
def _build_chunkwise_mask_maxlen(max_len: int, chunk_size: int,
                                 window_size: int) -> torch.Tensor:
    """
    Builds a mask of shape [max_len, max_len] with True = masked.
    Semantics match your original:
      - within chunk: full attention (bidirectional) inside each chunk
      - past chunks:
          window_size < 0  => attend to all past
          window_size >= 0 => attend to [q - window + 1, chunk_start - 1]
    """
    L = int(max_len)
    cs = int(chunk_size)
    ws = int(window_size)

    q = torch.arange(L)[:, None]  # [L,1]
    k = torch.arange(L)[None, :]  # [1,L]

    cq = q // cs
    ck = k // cs
    same_chunk = cq.eq(ck)

    if ws < 0:
        past_allowed = ck.lt(cq)  # all past chunks
    else:
        chunk_start = (cq * cs)  # [L,1]
        sliding_start = (q - (ws - 1)).clamp(min=0)  # [L,1]
        # keys must be before current chunk start, and within sliding window
        past_allowed = (k < chunk_start) & (k >= sliding_start)

    allowed = same_chunk | past_allowed
    masked = ~allowed  # True = masked
    return masked


class CacheModule(nn.Module):

    def __init__(self, max_cache_size: int = 0):
        super().__init__()
        self.max_cache_size = int(max_cache_size)
        self.register_buffer("k_cache", torch.zeros(1), persistent=False)
        self.register_buffer("v_cache", torch.zeros(1), persistent=False)

    @torch.jit.export
    def get_cache(self):
        return self.k_cache, self.v_cache

    @torch.jit.export
    def set_cache(self, k: torch.Tensor, v: torch.Tensor):
        self.k_cache = k
        self.v_cache = v


@gin.configurable
class MHAttention(nn.Module):

    def __init__(
        self,
        is_causal: bool = False,
        dropout_level: float = 0.0,
        n_heads: int = 4,
        streaming: bool = False,
        rotary_emb: Optional[nn.Module] = None,
        embed_dim: int = 256,
        attention_chunk_size: int = 4,
        local_attention_size: Optional[int] = None,
        max_diffusion_steps: int = 16,
        max_batch_size: int = 4,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.is_causal = bool(is_causal)
        self.dropout_level = float(dropout_level)
        self.n_heads = int(n_heads)
        self.rotary_emb = rotary_emb
        self.min_chunk_size = int(attention_chunk_size)
        self.local_attention_size = local_attention_size

        if streaming:
            if local_attention_size is None:
                raise ValueError(
                    "streaming=True requires local_attention_size to be set")
            self.max_cache_size = int(local_attention_size)
        else:
            self.max_cache_size = 0

        # Cache buffers
        if self.max_cache_size > 0:
            self.register_buffer(
                "last_k",
                torch.zeros(max_batch_size, self.n_heads, 1,
                            embed_dim // n_heads))
            self.register_buffer(
                "last_v",
                torch.zeros(max_batch_size, self.n_heads, 1,
                            embed_dim // n_heads))

            k_cache = torch.zeros(
                (max_batch_size, max_diffusion_steps, self.n_heads,
                 self.max_cache_size, embed_dim // n_heads),
                dtype=torch.float32,
            )
            v_cache = torch.zeros_like(k_cache)
            self.register_buffer("k_cache", k_cache)
            self.register_buffer("v_cache", v_cache)

        self.rearrange_heads1 = Rearrange("bs n (h d) -> bs h n d",
                                          h=self.n_heads)
        self.rearrange_heads2 = Rearrange("bs h n d -> bs n (h d)",
                                          h=self.n_heads)

        # -----------------------------
        # Precompute mask once (CPU), register as buffer.
        # We precompute for max_total_len = cache + seq + sink.
        # Then slice in forward.
        # -----------------------------
        self.max_seq_len = int(max_seq_len)
        max_total_len = self.max_seq_len + self.max_cache_size
        self.max_total_len = int(max_total_len)

        if self.is_causal:
            ws = -1 if self.local_attention_size is None else int(
                self.local_attention_size)
            # build boolean mask once; will be moved to device automatically as buffer
            masked_bool = _build_chunkwise_mask_maxlen(self.max_total_len,
                                                       self.min_chunk_size, ws)
            # store as bool; convert to -inf in forward with correct dtype/device
            self.register_buffer("_attn_mask_bool",
                                 masked_bool,
                                 persistent=False)
        else:
            self.register_buffer("_attn_mask_bool",
                                 torch.zeros(1, 1, dtype=torch.bool),
                                 persistent=False)

    def get_buffers(self, i: int):
        return self.k_cache[:, i], self.v_cache[:, i]

    def set_buffers(self, k, v, i: int):
        self.k_cache[:k.shape[0], i] = k
        self.v_cache[:v.shape[0], i] = v

    def roll_cache(self, roll_size: int, cache_index: int):
        if self.last_k is None or self.last_v is None:
            return
        k_cache, v_cache = self.get_buffers(cache_index)

        if roll_size < self.min_chunk_size:
            print("warming - roll size is smaller than min chunk size")

        k_cache = torch.cat(
            [k_cache[:self.last_k.shape[0]], self.last_k[:, :, :roll_size]],
            dim=2)
        v_cache = torch.cat(
            [v_cache[:self.last_v.shape[0]], self.last_v[:, :, :roll_size]],
            dim=2)

        if k_cache.shape[2] > self.max_cache_size:
            k_cache = k_cache[:, :, -self.max_cache_size:]
            v_cache = v_cache[:, :, -self.max_cache_size:]

        self.set_buffers(k_cache, v_cache, cache_index)

    def _get_attn_mask(self, Tq: int, Tk: int, device: torch.device,
                       dtype: torch.dtype) -> torch.Tensor:
        """
        Slice precomputed bool mask and convert to -inf/0 float mask.
        Shape expected by SDPA: [Tq, Tk] (broadcastable).
        """
        m = self._attn_mask_bool[:Tk, :Tk]  # [Tk, Tk] for safety (max)
        m = m[-Tq:, :Tk]  # [Tq, Tk]
        attn = torch.zeros((Tq, Tk), device=device, dtype=dtype)
        attn = attn.masked_fill(m.to(device=device), float("-inf"))
        return attn

    def forward(self, q, k, v, cache_index: int):
        # [B, N, C] -> [B, H, N, D]
        q, k, v = [self.rearrange_heads1(x) for x in (q, k, v)]

        if self.max_cache_size > 0:
            k_cache, v_cache = self.get_buffers(cache_index)
            full_k = torch.cat([k_cache[:k.shape[0]], k], dim=2)
            full_v = torch.cat([v_cache[:v.shape[0]], v], dim=2)
            self.last_k = k
            self.last_v = v
        else:
            full_k = k
            full_v = v

        if self.rotary_emb is not None:
            q, full_k = self.rotary_emb.rotate_queries_with_cached_keys(
                q, full_k)

        # ----------- attention mask -----------
        attn_mask: Optional[torch.Tensor] = None
        if self.is_causal:
            Tq = q.shape[2]
            Tk = full_k.shape[2]
            attn_mask = self._get_attn_mask(Tq,
                                            Tk,
                                            device=q.device,
                                            dtype=q.dtype)

        out = nn.functional.scaled_dot_product_attention(
            q,
            full_k,
            full_v,
            attn_mask=attn_mask,
            is_causal=False,  # we handle causal via mask
            dropout_p=self.dropout_level if self.training else 0.0,
        )

        out = self.rearrange_heads2(out)
        return out


class SelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        is_causal: bool = True,
        dropout_level: float = 0.0,
        n_heads: int = 8,
        rotary_emb=None,
        local_attention_size: Optional[int] = None,
        attention_chunk_size: int = 4,
        streaming: bool = False,
        max_diffusion_steps: int = 16,
        max_batch_size: int = 16,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.qkv_linear = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        self.mha = MHAttention(
            is_causal=is_causal,
            dropout_level=dropout_level,
            n_heads=n_heads,
            streaming=streaming,
            rotary_emb=rotary_emb,
            embed_dim=embed_dim,
            attention_chunk_size=attention_chunk_size,
            local_attention_size=local_attention_size,
            max_diffusion_steps=max_diffusion_steps,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

    def roll_cache(self, roll_size: int, cache_index: int):
        self.mha.roll_cache(roll_size, cache_index=cache_index)

    def forward(self, x, cache_index: int):
        q, k, v = self.qkv_linear(x).chunk(3, dim=2)
        return self.mha(q, k, v, cache_index)


class MLP(nn.Module):

    def __init__(self, embed_dim, mlp_multiplier, dropout_level):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout_level),
        )

    def forward(self, x):
        return self.mlp(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        cond_dim: int,
        tcond_dim: int,
        is_causal: bool,
        mlp_multiplier: int,
        dropout_level: float,
        rotary_emb=None,
        local_attention_size: Optional[int] = None,
        attention_chunk_size: int = 1,
        streaming: bool = False,
        max_diffusion_steps: int = 0,
        max_batch_size: int = 0,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.tcond_dim = tcond_dim

        self.self_attention = SelfAttention(
            embed_dim=embed_dim,
            is_causal=is_causal,
            dropout_level=dropout_level,
            n_heads=max(1, embed_dim // 64),
            rotary_emb=rotary_emb,
            attention_chunk_size=attention_chunk_size,
            local_attention_size=local_attention_size,
            streaming=streaming,
            max_diffusion_steps=max_diffusion_steps,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        self.mlp = MLP(embed_dim, mlp_multiplier, dropout_level)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        if self.cond_dim > 0:
            self.linear = nn.Linear(cond_dim, 2 * embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)

        if self.tcond_dim > 0:
            self.tcond_linear = nn.Linear(tcond_dim, 2 * embed_dim)
            self.norm0 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        else:
            self.tcond_linear = self.norm0 = nn.Identity()

    def roll_cache(self, size: int, cache_index: int = 0):
        self.self_attention.roll_cache(size, cache_index)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor],
        tcond: Optional[torch.Tensor],
        cache_index: int,
    ) -> torch.Tensor:
        if self.tcond_dim > 0:
            x = self.norm0(x)
            assert tcond is not None

            alpha, beta = self.tcond_linear(tcond).chunk(2, dim=-1)
            x = x * (1 + alpha) + beta

        x = self.self_attention(self.norm1(x), cache_index=cache_index) + x

        if self.cond_dim > 0:
            x = self.norm2(x)
            assert cond is not None
            alpha, beta = self.linear(cond).chunk(2, dim=-1)
            x = x * (1 + alpha.unsqueeze(1)) + beta.unsqueeze(1)

        x = self.mlp(self.norm3(x)) + x
        return x


class DenoiserTransBlock(nn.Module):

    def __init__(
        self,
        n_channels: int = 64,
        seq_len: int = 32,
        mlp_multiplier: int = 4,
        embed_dim: int = 256,
        cond_dim: int = 128,
        tcond_dim: int = 0,
        dropout: float = 0.1,
        n_layers: int = 4,
        is_causal: bool = True,
        local_attention_size: Optional[int] = None,
        attention_chunk_size: int = 4,
        use_out_proj: bool = True,
        streaming: bool = False,
        max_diffusion_steps: int = 16,
        max_batch_size: int = 16,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier
        self.tcond_dim = tcond_dim
        self.seq_len = int(seq_len)

        self.patchify_and_embed = nn.Sequential(
            Rearrange("b c t -> b t c"),
            nn.Linear(n_channels, self.embed_dim),
            nn.GELU(),
        )

        if tcond_dim > 0:
            self.patchify_and_embed_tcond = nn.Sequential(
                Rearrange("b c t -> b t c"),
                nn.Linear(tcond_dim, tcond_dim),
                nn.GELU(),
            )
        else:
            self.patchify_and_embed_tcond = nn.Identity()

        self.rotary_emb = RotaryEmbedding(32)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                embed_dim=self.embed_dim,
                mlp_multiplier=self.mlp_multiplier,
                cond_dim=0 if cond_dim == 0 else self.embed_dim,
                tcond_dim=tcond_dim,
                is_causal=is_causal,
                dropout_level=self.dropout,
                rotary_emb=self.rotary_emb,
                attention_chunk_size=attention_chunk_size,
                local_attention_size=local_attention_size,
                streaming=streaming,
                max_diffusion_steps=max_diffusion_steps,
                max_batch_size=max_batch_size,
                max_seq_len=self.seq_len,
            ) for _ in range(self.n_layers)
        ])

        self.rearrange2 = Rearrange("b t c -> b c t")
        if use_out_proj:
            self.out_proj = nn.Sequential(
                nn.Linear(self.embed_dim, n_channels), self.rearrange2)
        else:
            self.out_proj = self.rearrange2

    def roll_cache(self, size: int, cache_index: int):
        for block in self.decoder_blocks:
            block.roll_cache(size, cache_index)

    def forward(
        self,
        x: torch.Tensor,
        features: Optional[torch.Tensor],
        time_cond: Optional[torch.Tensor],
        cache_index: int,
    ):
        x = self.patchify_and_embed(x)  # [B, T, C]

        if time_cond is not None and self.tcond_dim > 0:
            time_cond = self.patchify_and_embed_tcond(time_cond)

        for block in self.decoder_blocks:
            x = block(x,
                      cond=features,
                      tcond=time_cond,
                      cache_index=cache_index)

        return self.out_proj(x)


@gin.configurable
class DenoiserV2(nn.Module):

    def __init__(self,
                 n_channels: int,
                 seq_len: int = 32,
                 embed_dim: int = 256,
                 cond_dim: int = 64,
                 tcond_dim: int = 0,
                 noise_embed_dims: int = 128,
                 n_layers: int = 6,
                 mlp_multiplier: int = 2,
                 dropout: float = 0.1,
                 causal: bool = False,
                 local_attention_size: Optional[int] = None,
                 attention_chunk_size: int = 4,
                 use_out_proj: bool = True,
                 streaming: bool = False,
                 max_diffusion_steps: int = 16,
                 max_batch_size: int = 16,
                 **kwargs):
        super().__init__()
        self.noise_embed_dims = int(noise_embed_dims)
        self.embed_dim = int(embed_dim)
        self.n_channels = int(n_channels)

        self.fourier_feats = PositionalEmbedding(num_channels=noise_embed_dims,
                                                 max_positions=10_000,
                                                 factor=100.0)

        if cond_dim > 0:
            self.embedding = nn.Sequential(
                nn.Linear(cond_dim + noise_embed_dims, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
        else:
            self.embedding = nn.Identity()

        self.denoiser_trans_block = DenoiserTransBlock(
            n_channels=n_channels,
            seq_len=seq_len,
            mlp_multiplier=mlp_multiplier,
            embed_dim=embed_dim,
            dropout=dropout,
            n_layers=n_layers,
            cond_dim=0 if cond_dim == 0 else self.embed_dim,
            tcond_dim=tcond_dim,
            is_causal=causal,
            attention_chunk_size=attention_chunk_size,
            local_attention_size=local_attention_size,
            use_out_proj=use_out_proj,
            streaming=streaming,
            max_diffusion_steps=max_diffusion_steps,
            max_batch_size=max_batch_size,
        )

    def roll_cache(self, size: int, cache_index: int):
        self.denoiser_trans_block.roll_cache(size, cache_index)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        time_cond: Optional[torch.Tensor] = None,
        cache_index: int = 0,
    ) -> torch.Tensor:
        # retro-compat: sometimes time comes as [B,1,T], keep scalar per batch
        if time.dim() > 1:
            time = time[..., 0]
        time = time.reshape(-1)

        noise_level = self.fourier_feats(time)

        if cond is not None:
            embedding_in = torch.cat([noise_level, cond], dim=-1)
        else:
            embedding_in = noise_level

        features = self.embedding(embedding_in)

        x = self.denoiser_trans_block(x,
                                      features=features,
                                      time_cond=time_cond,
                                      cache_index=cache_index)
        return x


if __name__ == "__main__":
    # quick check
    denoiser = DenoiserV2(
        n_channels=16,
        seq_len=32,
        tcond_dim=24,
        cond_dim=32,
        causal=True,
        local_attention_size=64,
        attention_chunk_size=4,
        streaming=True,
        max_diffusion_steps=16,
        max_batch_size=16,
    )

    B = 4
    tcond = torch.randn((B, 24, 32))
    x = torch.randn((B, 16, 32))
    cond = torch.randn((B, 32))
    time = torch.randn(B, 1)

    y = denoiser(x, time, cond, tcond, cache_index=0)
    print("out:", y.shape)

    # TorchScript check
    ts = torch.jit.script(denoiser.eval())
    y2 = ts(x, time, cond, tcond, 0)
    print("ts out:", y2.shape)
