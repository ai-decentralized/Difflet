"""FLUX's RoPE in the split-halves layout, by permuting q/k head channels once at load.

diffusers' FLUX rotary embedding pairs channels ``(2i, 2i+1)`` of every head:
``x.reshape(..., D/2, 2).unbind(-1)`` then ``stack(...).flatten`` — a relayout
of the minor dimension on every q and k of every block, ~4-7% of a compiled
TPU denoise step (trace 2026-11-03: ``embeddings.py`` ``apply_rotary_emb`` and
most of the "data formatting" time).

Attention only sees q and k through ``q . k``, which is unchanged when both are
permuted the same way along the head dim. Permuting each head's channels to
``P = [0, 2, 4, ..., D-2, 1, 3, ..., D-1]`` — the output rows of ``to_q``/``to_k``
(and the text stream's ``add_q_proj``/``add_k_proj``), and the weights of the
per-head qk RMSNorms that follow them — turns each pair ``(2i, 2i+1)`` into
``(i, D/2 + i)``. That is the split-halves rotary (``use_real_unbind_dim=-2``):
a slice and a concat, no interleave. The cos/sin tables are permuted the same
way once per forward (``PermutedFluxPosEmbed``). V, the attention output and
everything after are untouched, so the result equals the stock layout's up to
the RMSNorm's summation order.
"""

from __future__ import annotations

import torch
from torch import nn

ROPE_UNBIND_DIM = -2


def pair_split_permutation(head_dim: int) -> torch.Tensor:
    """``[0, 2, ..., D-2, 1, 3, ..., D-1]``: interleaved pairs -> split halves."""
    if int(head_dim) % 2:
        raise ValueError(f"rotary head_dim must be even, got {head_dim}")
    idx = torch.arange(int(head_dim))
    return torch.cat([idx[0::2], idx[1::2]])


class PermutedFluxPosEmbed(nn.Module):
    """``FluxPosEmbed`` whose (cos, sin) come out in the split-halves channel order."""

    def __init__(self, inner: nn.Module, head_dim: int) -> None:
        super().__init__()
        self.inner = inner
        self.register_buffer("perm", pair_split_permutation(head_dim), persistent=False)

    def forward(self, ids: torch.Tensor):
        cos, sin = self.inner(ids)
        perm = self.perm.to(cos.device)
        return cos.index_select(-1, perm), sin.index_select(-1, perm)


def _permute_rows_per_head_(linear: nn.Module, head_dim: int, perm: torch.Tensor) -> None:
    weight = linear.weight
    out = int(weight.shape[0])
    if out % int(head_dim):
        raise ValueError(f"{type(linear).__name__} rows {out} are not whole heads of {head_dim}")
    heads = out // int(head_dim)
    perm = perm.to(weight.device)
    with torch.no_grad():
        w = weight.view(heads, int(head_dim), -1).index_select(1, perm).reshape(out, -1)
        weight.copy_(w)
        bias = getattr(linear, "bias", None)
        if bias is not None:
            bias.copy_(bias.view(heads, int(head_dim)).index_select(1, perm).reshape(out))


def _permute_norm_(norm: nn.Module, perm: torch.Tensor) -> None:
    weight = getattr(norm, "weight", None)
    if weight is None:
        return
    with torch.no_grad():
        weight.copy_(weight.index_select(0, perm.to(weight.device)))


def permute_flux_rope_layout_(transformer: nn.Module, processor_cls=None) -> int:
    """Rewrite a loaded (TP-sharded or not) FLUX transformer in place for split-halves RoPE.

    Returns the number of attention modules rewritten. ``processor_cls`` builds
    each attention's processor with ``rope_unbind_dim=-2``; it defaults to the
    TP processor of ``models/flux/tp_sharding.py``.
    """
    if processor_cls is None:
        from difflet.models.flux.tp_sharding import FluxTPAttnProcessor as processor_cls
    if isinstance(transformer.pos_embed, PermutedFluxPosEmbed):
        raise RuntimeError("FLUX RoPE layout is already permuted")
    head_dim = None
    count = 0
    processor = processor_cls(rope_unbind_dim=ROPE_UNBIND_DIM)
    for block in list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks):
        attn = block.attn
        dim = int(attn.to_q.weight.shape[0]) // int(attn.heads)
        head_dim = dim if head_dim is None else head_dim
        perm = pair_split_permutation(dim)
        for name in ("to_q", "to_k", "add_q_proj", "add_k_proj"):
            linear = getattr(attn, name, None)
            if linear is not None:
                _permute_rows_per_head_(linear, dim, perm)
        for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            norm = getattr(attn, name, None)
            if norm is not None:
                _permute_norm_(norm, perm)
        attn.processor = processor
        count += 1
    transformer.pos_embed = PermutedFluxPosEmbed(transformer.pos_embed, head_dim)
    return count


__all__ = [
    "PermutedFluxPosEmbed",
    "ROPE_UNBIND_DIM",
    "pair_split_permutation",
    "permute_flux_rope_layout_",
]
