"""Wan on TPU with sequence parallelism over collective matmuls (``DIFFLET_TPU_SP=1``).

``difflet/models/wan/modeling_wan.py`` already implements Megatron-SP for
Trainium (``sp_enabled``): the model scatters the patch sequence across the TP
group on entry and gathers it on exit, and inside each attention and FFN it
all-gathers before the column-parallel projections and reduce-scatters after
the row-parallel one. That file is shared with Trainium, so it is left as is.
On TPU the separate gather / reduce-scatter are slower than the all-reduce they
replace; what pays is fusing each into its matmul
(``ops_impl.collective_matmul``). :func:`install` therefore swaps the forward
of every block's ``attn1``, ``attn2`` and ``ffn`` for the fused equivalent, on
this module instance only.

What runs where, per block (B, S, H = batch, sequence, model dim):

    self-attn   x (B,S/tp,H) -AG+[to_q,to_k,to_v]-> q,k,v (B,S,heads/tp)
                attention as before (full sequence, local heads), then
                to_out[0] -matmul+RS-> (B,S/tp,H)
    cross-attn  as above with AG+[to_q] only (k, v come from the replicated text)
    FFN         x -AG+net_in-> gelu -net_out+RS-> (B,S/tp,H)

The per-token work between them (residual, adaLN, the three LayerNorms) is the
shared block forward, now on S/tp tokens. The across-heads q/k RMSNorm keeps its
(B,S,1) all-reduce: it needs every head's sum of squares.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from difflet.backends.tpu.ops_impl import collective_matmul as cm


def _attention_forward(self, hidden_states, encoder_hidden_states=None, rotary_emb=None):
    from difflet.models.wan.modeling_wan import (
        _attn_kernel,
        apply_rotary_emb,
        apply_rotary_emb_half_split,
    )

    if encoder_hidden_states is None:
        q, k, v = cm.all_gather_matmul(hidden_states, (self.to_q, self.to_k, self.to_v))
    else:
        (q,) = cm.all_gather_matmul(hidden_states, (self.to_q,))
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

    q = self._global_rms_norm(self.norm_q, q)
    k = self._global_rms_norm(self.norm_k, k)

    q = q.unflatten(-1, (self.local_heads, self.head_dim))
    k = k.unflatten(-1, (self.local_heads, self.head_dim))
    v = v.unflatten(-1, (self.local_heads, self.head_dim))

    if rotary_emb is not None and not self.is_cross_attention:
        cos, sin = rotary_emb
        # The model's rope emits tables in its rope_layout, and a module whose q/k rows were
        # reordered (use_half_split_rope_) must rotate in that layout too, as WanAttention does.
        half_split = getattr(self, "rope_layout", "interleaved") == "half_split"
        rotate = apply_rotary_emb_half_split if half_split else apply_rotary_emb
        q = rotate(q, cos, sin)
        k = rotate(k, cos, sin)

    out = _attn_kernel(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                       head_dim=self.head_dim)
    out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
    out = cm.matmul_reduce_scatter(out, self.to_out[0])
    return self.to_out[1](out)


def _ffn_forward(self, x):
    (h,) = cm.all_gather_matmul(x, (self.net_in,))
    h = F.gelu(h, approximate="tanh")
    return cm.matmul_reduce_scatter(h, self.net_out)


def install(module: torch.nn.Module, *, seq_len: int) -> bool:
    """Swap the fused forwards into a ``WanTransformer3DModel`` built with
    ``sp_enabled``. Returns False (and changes nothing) when the shape does not
    split, so the caller keeps the shared Megatron-SP path."""
    if not getattr(module, "sp_enabled", False):
        raise ValueError("install() needs a Wan model built with sp_enabled=True")
    from difflet.backends.tpu.ops_impl import parallel_mesh

    tp = parallel_mesh.get_tp_size()
    if not cm.supports(seq_len, tp):
        print(f"[difflet tpu] {cm.SP_ENV}: sequence {seq_len} does not split into "
              f"2 x tp={tp}; using the unfused Megatron-SP path", flush=True)
        return False
    cm.prepare()
    for block in module.blocks:
        for attn in (block.attn1, block.attn2):
            if getattr(attn, "context_parallel_enabled", False):
                raise NotImplementedError("sequence parallelism excludes context parallelism")
            attn.forward = types.MethodType(_attention_forward, attn)
        block.ffn.forward = types.MethodType(_ffn_forward, block.ffn)
    print(f"[difflet tpu] Wan sequence-parallel with collective matmuls "
          f"({len(module.blocks)} blocks, S={seq_len}, tp={tp})", flush=True)
    return True


__all__ = ["install"]
