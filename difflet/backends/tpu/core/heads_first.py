"""Attention inputs in TPU-friendly layout: heads first and split-halves RoPE.

Shared by the TPU sequence-parallel paths of FLUX, Qwen-Image, HunyuanVideo and Wan.

The diffusers-style processors keep q/k/v as [B, S, H, D] through the per-head
RMSNorm and an interleaved-pair RoPE (``x.reshape(..., D/2, 2)``), then transpose
for the attention kernel. Free on a GPU (strides); on a TPU each step is a
physical relayout -- the minor dims (H, D) or (D/2, 2) do not fill a (8, 128) tile
-- and they were 4-14 ms of a step per model (traces 2026-11-12/13). Two changes
remove them, both exact:

* **Heads first**: ``[B, S, H*D] -> [B, H, S, D]`` right away (:func:`to_heads_first`;
  with D = 128 each tile belongs to one head), norms, RoPE and stream concats
  then run on (S, D) tiles.
* **Split-halves RoPE**: permuting each head's q/k channels by
  :func:`pair_split_permutation` (``[0, 2, .., D-2, 1, 3, .., D-1]``) -- the
  output rows of the q/k projections and the qk-norm weights, once, on the
  host (:func:`permute_qk_`) -- turns pair ``(2i, 2i+1)`` into ``(i, D/2 + i)``:
  the rotation is two half-width slices (:func:`rope_split_halves`). q . k is
  unchanged by a permutation common to q and k; v is not touched.

The host-side permutation assigns new parameter data: an on-device ``copy_`` into
the old parameter left Wan a weight layout that every forward copied back.
"""

from __future__ import annotations

import torch
from torch import nn


def pair_split_permutation(head_dim: int) -> torch.Tensor:
    """``[0, 2, ..., D-2, 1, 3, ..., D-1]``: interleaved RoPE pairs -> split halves."""
    if int(head_dim) % 2:
        raise ValueError(f"rotary head_dim must be even, got {head_dim}")
    idx = torch.arange(int(head_dim))
    return torch.cat([idx[0::2], idx[1::2]])


def _on_host_then_back(tensor: torch.Tensor, fn) -> torch.Tensor:
    return fn(tensor.detach().cpu()).contiguous().to(tensor.device)


def permute_rows_per_head_(linear: nn.Module, head_dim: int, perm: torch.Tensor) -> None:
    """Permute each head's output rows (and bias) of a q/k projection in place."""
    def per_head(host):
        rows = host.shape[0]
        if rows % head_dim:
            raise ValueError(f"{type(linear).__name__} rows {rows} are not whole heads of {head_dim}")
        return host.view(rows // head_dim, head_dim, *host.shape[1:]).index_select(1, perm) \
                   .reshape(host.shape)

    linear.weight.data = _on_host_then_back(linear.weight, per_head)
    if getattr(linear, "bias", None) is not None:
        linear.bias.data = _on_host_then_back(linear.bias, per_head)


def permute_norm_(norm: nn.Module | None, perm: torch.Tensor) -> None:
    """Permute a per-head RMSNorm's weight (and bias) the same way."""
    for name in ("weight", "bias"):
        param = getattr(norm, name, None) if norm is not None else None
        if param is not None:
            param.data = _on_host_then_back(param, lambda h: h.index_select(0, perm))


def permute_qk_(attn: nn.Module, head_dim: int,
                projections=("to_q", "to_k", "add_q_proj", "add_k_proj"),
                norms=("norm_q", "norm_k", "norm_added_q", "norm_added_k")) -> None:
    """Every present q/k projection and qk norm of ``attn``, for split-halves RoPE."""
    perm = pair_split_permutation(head_dim)
    for name in projections:
        linear = getattr(attn, name, None)
        if linear is not None:
            permute_rows_per_head_(linear, head_dim, perm)
    for name in norms:
        permute_norm_(getattr(attn, name, None), perm)


def to_heads_first(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    """[B, S, H*D] -> [B, H, S, D]; a 4-D tensor is taken as already heads first
    (the AG kernel writes it so with ``head_dims``)."""
    if tensor.dim() == 4:
        return tensor
    return tensor.unflatten(-1, (int(heads), -1)).transpose(1, 2)


def rope_split_halves(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *,
                      compute_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Rotate [B, H, S, D] (q/k channels in split-halves order) by ``(cos, sin)``.

    ``cos``/``sin`` are [S, D] in the same split order (low half for channels
    ``:D/2``, high half for ``D/2:``) or [S, D/2] (one angle per pair, both halves).
    Per pair ``(lo, hi)``: ``lo*c_lo - hi*s_lo | hi*c_hi + lo*s_hi`` -- the products
    and sums of the interleaved ``x*cos + rot(x)*sin`` (negation is exact, a sum
    commutes), so with the same ``compute_dtype`` the result is bit-identical to
    the interleaved form on unpermuted channels. ``compute_dtype``: what x is
    cast to first (diffusers' ``x.float()``); the products then promote with the
    tables' dtype, as the interleaved forms do. None: x's own dtype."""
    xf = x.to(compute_dtype) if compute_dtype is not None else x
    lo, hi = xf.chunk(2, dim=-1)
    cos = cos.to(device=x.device)[None, None]
    sin = sin.to(device=x.device)[None, None]
    if cos.shape[-1] * 2 == x.shape[-1]:
        cos_lo = cos_hi = cos
        sin_lo = sin_hi = sin
    else:
        cos_lo, cos_hi = cos.chunk(2, dim=-1)
        sin_lo, sin_hi = sin.chunk(2, dim=-1)
    return torch.cat([lo * cos_lo - hi * sin_lo, hi * cos_hi + lo * sin_hi], dim=-1).to(x.dtype)


__all__ = [
    "pair_split_permutation",
    "permute_norm_",
    "permute_qk_",
    "permute_rows_per_head_",
    "rope_split_halves",
    "to_heads_first",
]
