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

Heads first, split-halves RoPE (``DIFFLET_TPU_HV_HEADS_FIRST``, default on). The
shared attention keeps q/k/v as [B, S, H, D] through the qk RMSNorm and the
interleaved RoPE (``copy bf16[1,10240,6,64,2]``, ``reshape [10496,6,64,2]``,
...: ~30 ms of relayout and RoPE/norm fusions of a 476 ms step, trace
2026-11-13). With it, the q/k channels of every attention whose forward is
replaced here are permuted once (``core/heads_first``), the AG kernel writes q/k/v
as [B, H, S, D], and :func:`_dual_attention` / :func:`_single_attention` run the
norms, the RoPE and the joint attention (same bounds from the same mask) on that
layout. Single-stream blocks are permuted only when they run the SP forward.
"""

from __future__ import annotations

import math
import os
import types

import torch
import torch.nn as nn

from difflet.backends.tpu.core import heads_first
from difflet.backends.tpu.ops_impl import collective_matmul as cm

HEADS_FIRST_ENV = "DIFFLET_TPU_HV_HEADS_FIRST"


def _heads_first_enabled() -> bool:
    return os.environ.get(HEADS_FIRST_ENV, "1").strip().lower() not in {"0", "false", "no"}


def _rope(x, image_rotary_emb):
    """``_apply_hunyuan_rotary`` (pairs take ``cos[..., 0::2]`` / ``sin[..., 1::2]``,
    computed in x's and the tables' promoted type) on split-halves [B, H, S, D]."""
    cos, sin = image_rotary_emb
    if cos.dim() != 2:
        raise ValueError(f"HunyuanVideo rotary tables must be [S, D], got {tuple(cos.shape)}")
    return heads_first.rope_split_halves(x, cos[..., 0::2], sin[..., 1::2])


def _joint_attention(latent, context, attention_mask, scale, attention=None):
    """``dual_stream_attention`` on [B, H, S, D] streams: [latent | context] along S,
    the key bounds from the mask, one attention; returns both streams' outputs as
    [B, S, H*D]. ``attention``: ``difflet.ops.attention``, replaceable in tests."""
    from difflet.models.hunyuan_video.modeling_hunyuan_video import (
        _flatten_attention_mask,
        _keypad_bounds_from_mask,
    )

    if attention is None:
        from difflet.ops import attention
    (lq, lk, lv), (cq, ck, cv) = latent, context
    b, h, s_lat, d = lq.shape
    q, k, v = (torch.cat(pair, dim=2) for pair in ((lq, cq), (lk, ck), (lv, cv)))
    q_len, kv_len = q.shape[2], k.shape[2]
    flat = [t.reshape(b * h, t.shape[2], d) for t in (q, k, v)]
    mask_flat = _flatten_attention_mask(attention_mask, batch=b, heads=h)
    kwargs = {}
    if mask_flat is not None:
        # Bounds per query row: the fused path reduces them to one window per row.
        # (Bounds with one query row, broadcast, hung on the chips right after the
        # compile, HunyuanVideo and LTX-2 alike, 2026-11-13.)
        kwargs["bound_min"], kwargs["bound_max"] = _keypad_bounds_from_mask(mask_flat, q_len)
    out = attention(*flat, scale=scale, causal=False, tp_q=True, tp_k=True, tp_out=False,
                    **kwargs)
    out = out.reshape(b, h, q_len, d).transpose(1, 2).flatten(2, 3).to(lq.dtype)
    del kv_len
    return out[:, :s_lat], out[:, s_lat:]


def _dual_attention(attn, qkv, encoder_hidden_states, attention_mask, image_rotary_emb,
                    attention=None):
    """The shared dual-stream attention, heads first: (latent context before the
    row-parallel output projection, text output)."""
    heads = int(attn.heads)
    lq, lk, lv = (heads_first.to_heads_first(t, heads) for t in qkv)
    lq, lk = attn.norm_q(lq), attn.norm_k(lk)
    if image_rotary_emb is not None:
        lq, lk = _rope(lq, image_rotary_emb), _rope(lk, image_rotary_emb)
    cq, ck, cv = (heads_first.to_heads_first(proj(encoder_hidden_states), heads)
                  for proj in (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj))
    if attn.norm_added_q is not None:
        cq = attn.norm_added_q(cq)
    if attn.norm_added_k is not None:
        ck = attn.norm_added_k(ck)
    latent, text = _joint_attention((lq, lk, lv), (cq, ck, cv), attention_mask,
                                    1.0 / math.sqrt(attn.head_dim), attention)
    if len(attn.to_out) > 1:
        latent = attn.to_out[1](latent)
    if attn.to_add_out is not None:
        text = attn.to_add_out(text)
    return latent, text


def _single_attention(attn, qkv, text_len, attention_mask, image_rotary_emb, attention=None):
    """The shared pre-only (single-stream) attention on the joint [latent | text]
    q/k/v, heads first: (latent output, text output) as [B, S, H*D]."""
    heads = int(attn.heads)
    q, k, v = (heads_first.to_heads_first(t, heads) for t in qkv)
    q, k = attn.norm_q(q), attn.norm_k(k)
    s_lat = q.shape[2] - int(text_len)
    lq, lk, lv = q[:, :, :s_lat], k[:, :, :s_lat], v[:, :, :s_lat]
    cq, ck, cv = q[:, :, s_lat:], k[:, :, s_lat:], v[:, :, s_lat:]
    if image_rotary_emb is not None:
        lq, lk = _rope(lq, image_rotary_emb), _rope(lk, image_rotary_emb)
    return _joint_attention((lq, lk, lv), (cq, ck, cv), attention_mask,
                            1.0 / math.sqrt(attn.head_dim), attention)


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

    hd = getattr(self, "_difflet_kernel_heads", 0)
    qkv = cm.all_gather_matmul(hidden_states, (self.to_q, self.to_k, self.to_v), (hd, hd, hd))
    if getattr(self, "_difflet_heads_first", False):
        latent_ctx, text_out = _dual_attention(self, qkv, encoder_hidden_states,
                                               attention_mask, image_rotary_emb)
        return cm.matmul_reduce_scatter(latent_ctx, self.to_out[0]), text_out
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


#: ``DIFFLET_TPU_SP_SINGLE`` (with ``DIFFLET_TPU_SP=1``; default on, ``0`` turns it
#: off): the 40 single-stream blocks run sequence-parallel too, instead of dense TP
#: with an all-reduce per block (1.0 s of a 10.3 s, 20-step denoise on v5e-4 at
#: 320x512x61, 2026-11-11). Step 516.1 -> 500.7 ms (ring) / 478.0 ms (kernel).
SINGLE_ENV = "DIFFLET_TPU_SP_SINGLE"

#: The single-stream blocks' ops: q|k|v|mlp share one all-gather, the two row-parallel
#: projections one reduce-scatter (as FLUX's single blocks).
SINGLE_OPS = (("ag", 4), ("rs", 2))


def _single_requested() -> bool:
    import os

    return os.environ.get(SINGLE_ENV, "1").strip().lower() in ("1", "true", "yes", "on")


class _PrecomputedSingle(_Precomputed):
    """The ``self`` the shared attention sees in a single-stream block: the joint
    q/k/v are precomputed and the attention has no output projection (pre_only)."""

    def __init__(self, attn, qkv):
        self._attn = attn
        self.to_q, self.to_k, self.to_v = (lambda _x, t=t: t for t in qkv)
        self.to_out = None
        self.sp_enabled = False


def _single_block_forward(self, hidden_states, encoder_hidden_states, temb,
                          attention_mask=None, image_rotary_emb=None, *args, **kwargs):
    """A single-stream block on this rank's contiguous chunk of the joint
    [latent | text] sequence (B, (S_lat + S_txt)/tp, H).

    The shared model forward hands the first single block the full latent and text
    (it gathers the latent at the dual -> single transition) and expects full
    streams back from the last; in between, the blocks pass the chunk along as
    ``hidden_states``. A single-stream block treats every joint token alike outside
    attention, so a chunk may straddle the latent / text boundary; attention runs on
    the gathered q/k/v in the natural joint order, with the shared mask and rotary.
    """
    del args, kwargs
    from difflet.models.hunyuan_video.modeling_hunyuan_video import HunyuanVideoAttention
    from difflet.ops import gather_from_sequence_parallel_region

    first, last, text_len = self._difflet_sp_single
    if first:
        joint = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        # This rank's rows come from a buffer (runtime data), not a Python-constant
        # slice: with a constant, XLA folded the concatenation away on ranks 0-2
        # (their chunk lies inside the latent) but not on rank 3 (its chunk holds
        # the text), the per-rank programs diverged and the first forward hung in a
        # collective (v5e-4, 2026-11-11) -- the LTX-2 failure, see collective_matmul.
        hidden_states = joint.index_select(1, self._difflet_sp_rows)
    residual = hidden_states
    norm, gate = self.norm(hidden_states, emb=temb)
    attn = self.attn
    hd = getattr(attn, "_difflet_kernel_heads", 0)
    q, k, v, mlp = cm.all_gather_matmul(norm, (attn.to_q, attn.to_k, attn.to_v, self.proj_mlp),
                                        (hd, hd, hd, 0))
    if getattr(attn, "_difflet_heads_first", False):
        latent_out, text_out = _single_attention(attn, (q, k, v), text_len, attention_mask,
                                                 image_rotary_emb)
    else:
        s_lat = q.shape[1] - text_len
        latent_out, text_out = HunyuanVideoAttention.forward(
            _PrecomputedSingle(attn, (q, k, v)),
            hidden_states=q[:, :s_lat],  # read for nothing but the latent length
            encoder_hidden_states=q[:, s_lat:],
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
    attn_out = torch.cat([latent_out, text_out], dim=1)
    # proj_out_attn has the bias (skip_bias_add, added once after the reduction);
    # proj_out_mlp has none.
    proj = cm.matmul2_reduce_scatter(attn_out, self.proj_out_attn, self.act_mlp(mlp),
                                     self.proj_out_mlp, bias=self.proj_out_attn.bias)
    hidden_states = residual + gate.unsqueeze(1) * proj
    if last:
        full = gather_from_sequence_parallel_region(hidden_states, dim=1)
        return full[:, :-text_len], full[:, -text_len:]
    return hidden_states, encoder_hidden_states  # the text is not read again until the last


def _install_single(module, *, latent_seq_len: int, text_seq_len: int, tp: int) -> bool:
    from difflet.backends.tpu.ops_impl import parallel_mesh

    joint = int(latent_seq_len) + int(text_seq_len)
    if not cm.supports(joint, tp):
        print(f"[difflet tpu] {SINGLE_ENV}: joint sequence {joint} does not split into "
              f"2 x tp={tp}; single-stream blocks stay dense", flush=True)
        return False
    hd = _setup_heads_first([b.attn for b in module.single_transformer_blocks])
    cm.prepare(SINGLE_OPS + ((("ag", 4, (hd, hd, hd, 0)),) if hd else ()))
    blocks = list(module.single_transformer_blocks)
    rank, n = parallel_mesh.get_tp_rank(), joint // tp
    device = next(module.parameters()).device
    for i, block in enumerate(blocks):
        block._difflet_sp_single = (i == 0, i == len(blocks) - 1, int(text_seq_len))
        block.forward = types.MethodType(_single_block_forward, block)
    blocks[0].register_buffer(
        "_difflet_sp_rows", torch.arange(rank * n, (rank + 1) * n, device=device), persistent=False)
    print(f"[difflet tpu] HunyuanVideo single-stream blocks sequence-parallel too "
          f"({len(blocks)} blocks, joint S={joint}, tp={tp})", flush=True)
    return True


def _setup_heads_first(attns) -> int:
    """Permute these attentions' q/k for split-halves RoPE and mark them for the
    heads-first forward; returns the head size the AG kernel then writes q/k/v
    with (0: switched off)."""
    if not _heads_first_enabled():
        return 0
    head_dim = int(attns[0].head_dim)
    for attn in attns:
        heads_first.permute_qk_(attn, int(attn.head_dim))
        attn._difflet_heads_first = True
        attn._difflet_kernel_heads = head_dim
    return head_dim


def install(module: torch.nn.Module, *, latent_seq_len: int,
            text_seq_len: int | None = None) -> bool:
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
    # The Pallas AG kernel: with single-stream SP, -4.5% per step at 320x512x61
    # (v5e-4, 2026-11-11); DIFFLET_TPU_SP_KERNEL=jax keeps the ring.
    cm.set_kernel_default("pallas")
    hd = _setup_heads_first([b.attn for b in blocks])
    cm.prepare(cm.DEFAULT_OPS + ((("ag", 3, (hd, hd, hd)),) if hd else ()))
    for block in blocks:
        block.attn.forward = types.MethodType(_attention_forward, block.attn)
        block.ff.forward = types.MethodType(_ffn_forward, block.ff)
    single = (_single_requested() and text_seq_len is not None
              and _install_single(module, latent_seq_len=latent_seq_len,
                                  text_seq_len=text_seq_len, tp=tp))
    print(f"[difflet tpu] HunyuanVideo sequence-parallel with collective matmuls "
          f"({len(blocks)} dual-stream blocks, S_latent={latent_seq_len}, tp={tp}; "
          f"single-stream blocks {'SP' if single else 'stay dense'}"
          f"{'; heads-first split-halves RoPE' if _heads_first_enabled() else ''})", flush=True)
    return True


__all__ = ["HEADS_FIRST_ENV", "install"]
