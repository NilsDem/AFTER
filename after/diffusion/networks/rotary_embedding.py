from __future__ import annotations
from math import pi, log

import torch
from torch.nn import Module
from torch import nn, broadcast_tensors, Tensor
from typing import Dict, Tuple, Optional, List
from einops.layers.torch import Rearrange

from typing import Literal

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# broadcat, as tortoise-tts was using it


def broadcat(tensors, dim=-1):
    broadcasted_tensors = torch.broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


#def slice_at_dim(t, dim_slice: slice, *, dim):
#    dim += (t.ndim if dim < 0 else 0)
#    colons = [slice(None)] * t.ndim
#    colons[dim] = dim_slice
#    return t[tuple(colons)]


def slice_at_dim(t, start: int, end: int, *, dim: int):
    dim += (t.ndim if dim < 0 else 0)
    return t.narrow(dim, start, end - start)


class RotaryEmbedding(Module):

    def __init__(self,
                 dim,
                 custom_freqs: Tensor | None = None,
                 freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
                 theta=10000,
                 max_freq=10,
                 num_freqs=1,
                 learned_freq=False,
                 use_xpos=False,
                 xpos_scale_base=512,
                 interpolate_factor=1.,
                 theta_rescale_factor=1.,
                 seq_before_head_dim=False,
                 cache_if_possible=True,
                 cache_max_seq_len=8192):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor**(dim / (dim - 2))

        self.freqs_for = freqs_for

        if custom_freqs is not None:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta**
                          (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer('cached_freqs',
                             torch.zeros(cache_max_seq_len, dim),
                             persistent=False)
        self.register_buffer('cached_freqs_seq_len',
                             torch.tensor(0),
                             persistent=False)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.register_buffer('dummy', torch.tensor(0), persistent=False)

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.rearrange_freq_scale = Rearrange('n d -> n 1 d')

        self.rearrange_rotate_half1 = Rearrange('... (d r) -> ... d r', r=2)

        self.rearrange_rotate_half2 = Rearrange('... d r -> ... (d r)')

        if not use_xpos:
            return

        self.register_buffer('scale', scale, persistent=False)
        self.register_buffer('cached_scales',
                             torch.zeros(cache_max_seq_len, dim),
                             persistent=True)
        self.register_buffer('cached_scales_seq_len',
                             torch.tensor(0),
                             persistent=True)

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

        # rotary embedding helper functions

    def rotate_half(self, x):
        #layer = Rearrange('... (d r) -> ... d r', r = 2)
        x = self.rearrange_rotate_half1(x)
        #x = einops.rearrange(x, '... (d r) -> ... d r', r = 2)

        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        #layer = Rearrange('... d r -> ... (d r)')
        return self.rearrange_rotate_half2(x)

    #@autocast('cuda', enabled = False)
    def apply_rotary_emb(self,
                         freqs,
                         t,
                         start_index: int = 0,
                         scale: float = 1.,
                         seq_dim: int = -2,
                         freqs_seq_dim: Optional[int] = None):
        dtype = t.dtype

        #if t.ndim == 3 or freqs_seq_dim is not None:
        #    freqs_seq_dim = 0 if freqs_seq_dim is None else freqs_seq_dim
        #    seq_len = t.shape[seq_dim]
        #    freqs = slice_at_dim(freqs, freqs.shape[freqs_seq_dim], dim = freqs_seq_dim)

        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim

        assert rot_dim <= t.shape[
            -1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

        # Split t into three parts: left, middle (to be transformed), and right
        t_left = t[..., :start_index]
        t_middle = t[..., start_index:end_index]
        t_right = t[..., end_index:]

        # Apply rotary embeddings without modifying t in place
        t_transformed = (t_middle * freqs.cos() * scale) + (
            self.rotate_half(t_middle) * freqs.sin() * scale)

        out = torch.cat((t_left, t_transformed, t_right), dim=-1)

        return out.type(dtype)

    # learned rotation helpers

    def apply_learned_rotations(self,
                                rotations,
                                t,
                                start_index=0,
                                freq_ranges=None):
        if freq_ranges is not None:
            rotations = torch.einsum('..., f -> ... f', rotations, freq_ranges)
            layer = Rearrange('... r f -> ... (r f)')
            rotations = layer(rotations)
            #rotations = einops.rearrange(rotations, '... r f -> ... (r f)')

        #rotations = einops.repeat(rotations, '... n -> ... (n r)', r = 2)
        rotations = rotations.unsqueeze(-1).expand(
            [rotations.shape[0], rotations.shape[1]],
            2).reshape(rotations.shape[0], -1)

        return self.apply_rotary_emb(rotations, t, start_index=start_index)


# classes

    def get_seq_pos(self,
                    seq_len: int,
                    device: torch.device,
                    dtype: torch.dtype,
                    offset: int = 0):
        return (torch.arange(seq_len, device=device, dtype=dtype) +
                offset) / self.interpolate_factor

    def rotate_queries_or_keys(self,
                               t,
                               seq_dim: int | None = None,
                               offset: int = 0,
                               scale: float | None = None):
        #seq_dim = default(seq_dim, self.default_seq_dim)
        seq_dim = self.default_seq_dim if seq_dim is None else seq_dim

        assert not self.use_xpos or scale is not None, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len,
                               device=device,
                               dtype=dtype,
                               offset=offset)

        freqs = self.forward(seq, seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            #layer = Rearrange('n d -> n 1 d')
            freqs = self.rearrange_freq_scale(freqs)
            #freqs = einops.rearrange(freqs, 'n d -> n 1 d')

        return self.apply_rotary_emb(freqs,
                                     t,
                                     scale=1. if scale is None else scale,
                                     seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self,
                                        q,
                                        k,
                                        seq_dim: int | None = None,
                                        offset: int = 0):
        dtype, device, seq_dim = q.dtype, q.device, self.default_seq_dim if seq_dim is None else seq_dim

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        q_scale = k_scale = 1.

        #if self.use_xpos:
        #    seq = self.get_seq_pos(k_len, dtype = dtype, device = device)
        #
        #    q_scale = self.get_scale(seq[-q_len:]).type(dtype)
        #    k_scale = self.get_scale(seq).type(dtype)
        #

        rotated_q = self.rotate_queries_or_keys(q,
                                                seq_dim=seq_dim,
                                                scale=q_scale,
                                                offset=k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k,
                                                seq_dim=seq_dim,
                                                scale=k_scale**-1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim: int | None = None):
        seq_dim = self.default_seq_dim if seq_dim is None else seq_dim
        #default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:

            freqs = self.rearrange_freq_scale(freqs)
            scale = self.rearrange_freq_scale(scale)
            #freqs = einops.rearrange(freqs, 'n d -> n 1 d')
            #scale = einops.rearrange(scale, 'n d -> n 1 d')

        rotated_q = self.apply_rotary_emb(freqs,
                                          q,
                                          scale=scale,
                                          seq_dim=seq_dim)
        rotated_k = self.apply_rotary_emb(freqs,
                                          k,
                                          scale=scale**-1,
                                          seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(self,
                  t: Tensor,
                  seq_len: int | None = None,
                  offset: int = 0):
        assert self.use_xpos

        should_cache = (self.cache_if_possible and seq_len is not None
                        and (offset + seq_len) <= self.cache_max_seq_len)

        #if (
        #    should_cache and \
        #    self.cached_scales is not None and \
        #    (seq_len + offset) <= self.cached_scales_seq_len.item()
        #):
        #    return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        #if self.use_xpos:
        #    power = (t - len(t) // 2) / self.scale_base
        #    scale = self.scale ** einops.rearrange(power, 'n -> n 1')
        #    #scale = einops.repeat(scale, 'n d -> n (d r)', r = 2)
        #    scale = scale.unsqueeze(-1).expand(-1, -1, 2).reshape(scale.shape[0], -1)
        #if should_cache and offset == 0:
        #    self.cached_scales[:seq_len] = scale
        #    self.cached_scales_seq_len.copy_(seq_len)

        return scale

    def get_axial_freqs(self, *dims):
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = slice(None)

            new_axis_indices = tuple(Ellipsis if i is None else slice(0, None)
                                     for i in all_axis)
            all_freqs.append(freqs[new_axis_indices])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    #@autocast('cuda', enabled = False)
    def forward(self,
                t: Tensor,
                seq_len: Optional[int] = None,
                offset: int = 0) -> Tensor:

        if seq_len is not None:
            seq_len = int(seq_len)
        else:
            seq_len = 0

        a = offset + seq_len

        should_cache = (self.cache_if_possible and not self.learned_freq
                        and seq_len is not None and self.freqs_for != 'pixel'
                        and (offset + seq_len) <= self.cache_max_seq_len)

        if (
            should_cache and \
            self.cached_freqs is not None and \
            (offset + seq_len) <= self.cached_freqs_seq_len.item()
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        #freqs = einops.repeat(freqs, '... n -> ... (n r)', r = 2)

        freqs = freqs.unsqueeze(-1).expand(freqs.shape[0], freqs.shape[1],
                                           2).reshape(freqs.shape[0], -1)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len.copy_(torch.tensor(seq_len))

        return freqs
