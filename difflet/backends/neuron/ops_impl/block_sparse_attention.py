"""Experiment E2 (``docs/plans/2026-11-18-attn-constmax-sparse-experiments.md``): static
block-sparse attention on the neuron backend. Difflet's attention_cte fork
(``mx_attention_kernel.mx_attention_fwd``) with its ``kv_tiles`` mask, registered as the torch op
``difflet_neuron::block_sparse_attention`` (the pattern of feat/mxfp8-sage-attention's
``ops_impl/mx_attention.py``). The exp runs on the Vector Engine (``allow_vector_exp=True``), as
shipped attention_cte runs bf16 attention, so the mask is the only difference. Imported only when
``DIFFLET_EXP_ATTN_SPARSE`` is set (``modeling_wan``), which registers the op.
"""

from __future__ import annotations

import torch

from difflet.backends.neuron.ops_impl.attn_tile_masks import kv_tiles_for


def load_op():
    """``difflet_neuron::block_sparse_attention(q, k, v, scale, kv_mask) -> bf16`` on ``[BH, S,
    128]`` bf16 (attention_cte's ``tp_q``/``tp_k`` layout), ``kv_mask`` a ``kv_tiles_for`` spec;
    registered once."""
    try:
        return torch.ops.difflet_neuron.block_sparse_attention  # already registered
    except (AttributeError, RuntimeError):
        pass
    from torch_neuronx import wrap_nki
    from torch_neuronx.nki_hop import nki_op

    from difflet.backends.neuron.ops_impl.fp8_kernels import logical_cores
    from difflet.backends.neuron.ops_impl.mx_attention_kernel import mx_attention_fwd

    kernel = wrap_nki(mx_attention_fwd)

    def fake(q, k, v, scale, kv_mask):
        return torch.empty((q.shape[0], q.shape[1], v.shape[2]), dtype=torch.bfloat16,
                           device=q.device)

    @nki_op("difflet_neuron::block_sparse_attention", mutates_args={}, fake_fn=fake)
    def block_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
                               kv_mask: str) -> torch.Tensor:
        return kernel[(logical_cores(),)](q, k, v, scale=scale, pv_mx=False, qk_mx=False,
                                          allow_vector_exp=True,
                                          kv_tiles=kv_tiles_for(kv_mask, q.shape[1]))

    return block_sparse_attention


_OP = load_op()


def block_sparse_attention(q, k, v, *, scale: float, kv_mask: str):
    """Self-attention on ``[BH, S, d]`` computing, per 128-query group, only the 512-key tiles
    ``kv_mask`` names. The spec's grid must hold ``S`` tokens: ``kv_tiles_for`` raises when the
    op's kernel is traced (not here: this runs inside the compiled block)."""
    return _OP(q.contiguous(), k.contiguous(), v.contiguous(), scale, kv_mask)


__all__ = ["block_sparse_attention", "load_op"]
