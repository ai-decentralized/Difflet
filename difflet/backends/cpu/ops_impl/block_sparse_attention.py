"""Experiment E2: static block-sparse attention on the CPU backend, the masked fp32 reference
(keys outside each 128-query group's 512-key tiles at -inf), for routing checks."""

from __future__ import annotations

from difflet.backends.neuron.ops_impl.attn_tile_masks import kv_tiles_for, masked_attention_reference


def block_sparse_attention(q, k, v, *, scale: float, kv_mask: str):
    tiles = kv_tiles_for(kv_mask, q.shape[1])
    return masked_attention_reference(q, k, v, scale=scale, kv_tiles=tiles).to(q.dtype)
