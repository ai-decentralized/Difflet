"""HunyuanVideo on TPU with sequence parallelism over collective matmuls (``DIFFLET_TPU_SP=1``).

``difflet/models/hunyuan_video/modeling_hunyuan_video.py`` implements a latent-only
Megatron-SP for Trainium: the video stream is scattered by sequence on entry and
gathered before the first single-stream block; in each of the 20 dual-stream blocks
the latent attention and feed-forward all-gather before their column-parallel
projections and reduce-scatter after the row-parallel ones, while the text stream
stays full (dense TP) and the 40 single-stream blocks run dense. That file is shared
with Trainium and left as is; :func:`install` swaps, on this module instance only,
the dual-stream blocks' attention and latent feed-forward for versions that fuse each
gather / reduce-scatter into its matmul (``ops_impl.collective_matmul``):

    attention   latent (B,S/tp,D) -AG+[to_q,to_k,to_v]-> (B,S,D/tp) x3
                the shared attention forward, unchanged, on the precomputed projections
                latent context -to_out[0]+RS-> ; text as before (all-reduce)
    latent FFN  x -AG+net[0].proj-> act -net[2]+RS->

The shared forward is reused by calling it with a stand-in ``self`` whose latent
projections return the fused all-gathers' outputs, whose SP flag is off (no unfused
gather / reduce-scatter) and whose ``to_out[0]`` is the identity.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from difflet.backends.tpu.ops_impl import collective_matmul as cm


class _Precomputed:
    """The ``self`` the shared attention forward sees."""

    def __init__(self, attn, qkv):
        self._attn = attn
        self.to_q, self.to_k, self.to_v = (lambda _x, t=t: t for t in qkv)
        self.to_out = [nn.Identity(), *list(attn.to_out)[1:]]
        self.sp_enabled = False

    def __getattr__(self, name):  # heads, norms, add_*_proj, to_add_out, _shape, ...
        return getattr(self._attn, name)


def _attention_forward(self, *, hidden_states, encoder_hidden_states=None,
                       attention_mask=None, image_rotary_emb=None):
    from difflet.models.hunyuan_video.modeling_hunyuan_video import HunyuanVideoAttention

    qkv = cm.all_gather_matmul(hidden_states, (self.to_q, self.to_k, self.to_v))
    latent_ctx, text_out = HunyuanVideoAttention.forward(
        _Precomputed(self, qkv),
        hidden_states=qkv[0],  # read for nothing but shapes
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
    )
    return cm.matmul_reduce_scatter(latent_ctx, self.to_out[0]), text_out


def _ffn_forward(self, hidden_states):
    act = self.net[0]
    (h,) = cm.all_gather_matmul(hidden_states, (act.proj,))
    # The activation module's own forward, minus its projection.
    h = act.__class__.forward(_Projected(act, h), hidden_states)
    return cm.matmul_reduce_scatter(h, self.net[2])


class _Projected:
    """An activation module whose ``proj`` returns an already-projected tensor."""

    def __init__(self, act, projected):
        self._act = act
        self.proj = lambda _x: projected

    def __getattr__(self, name):  # approximate, ...
        return getattr(self._act, name)


def install(module: torch.nn.Module, *, latent_seq_len: int) -> bool:
    """Swap the fused forwards into the dual-stream blocks of a model built with
    ``sp_enabled``. Returns False (and changes nothing) when the latent sequence does
    not split, so the caller keeps the shared Megatron-SP path."""
    from difflet.backends.tpu.ops_impl import parallel_mesh
    from difflet.models.hunyuan_video.modeling_hunyuan_video import HunyuanVideoTransformerBlock

    blocks = [m for m in module.modules() if isinstance(m, HunyuanVideoTransformerBlock)]
    if not blocks or not all(b.sp_enabled for b in blocks):
        raise ValueError("install() needs a HunyuanVideo model built with sp_enabled=True")
    tp = parallel_mesh.get_tp_size()
    if not cm.supports(latent_seq_len, tp):
        print(f"[difflet tpu] {cm.SP_ENV}: latent sequence {latent_seq_len} does not split "
              f"into 2 x tp={tp}; using the unfused Megatron-SP path", flush=True)
        return False
    cm.prepare()
    for block in blocks:
        block.attn.forward = types.MethodType(_attention_forward, block.attn)
        block.ff.forward = types.MethodType(_ffn_forward, block.ff)
    print(f"[difflet tpu] HunyuanVideo sequence-parallel with collective matmuls "
          f"({len(blocks)} dual-stream blocks, S_latent={latent_seq_len}, tp={tp}; "
          f"single-stream blocks stay dense)", flush=True)
    return True


__all__ = ["install"]
