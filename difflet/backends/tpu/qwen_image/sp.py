"""Qwen-Image on TPU with sequence parallelism over collective matmuls (``DIFFLET_TPU_SP=1``).

``difflet/models/qwen_image/modeling_qwen.py`` implements Megatron-SP for Trainium
(``QwenImageSPTransformer2DModel``): both streams, image and text, are scattered by
sequence on entry and gathered on exit, and each block all-gathers before the
column-parallel projections and reduce-scatters after the row-parallel ones. That
file is shared with Trainium, so it is left as is; :func:`install` swaps each
block's forward, on this module instance only, for one that fuses every gather /
reduce-scatter into its matmul (``ops_impl.collective_matmul``):

    attention   img (B,S_img/tp,D) -AG+[to_q,to_k,to_v]->           (B,S_img,D/tp) x3
                txt (B,S_txt/tp,D) -AG+[add_q,add_k,add_v]->        (B,S_txt,D/tp) x3
                joint attention as before (the shared processor, unchanged)
                img context -to_out[0]+RS-> ; txt context -to_add_out+RS->
    MLP (each)  x -AG+net[0].proj-> gelu(tanh) -net[2]+RS->

The joint attention is not duplicated here: the shared processor is called with a
stand-in ``attn`` whose projections hand back the tensors already computed by the
fused all-gathers and whose output projections are the identity, so norms, RoPE,
the text mask and the kernel are exactly the shared code's.

Heads first, split-halves RoPE (``DIFFLET_TPU_QWEN_HEADS_FIRST``, default on).
The shared processor keeps q/k/v as [B, S, H, D] through the per-head RMSNorm
and the interleaved (complex-pair) RoPE before transposing for the attention: on
TPU every step of that is a relayout (``copy f32[1,4096,6,64,2]``,
``reshape f32[4096,6,64,2]``, ...; ~14 ms of relayout and ~6 ms of RoPE
fusions of a 173 ms step, trace 2026-11-13). As for FLUX and Wan, each head's q/k
channels are permuted once at install (``[0, 2, .., 1, 3, ..]``: the rows of
to_q/to_k/add_q_proj/add_k_proj and the qk RMSNorm weights, on the host) so the
rotation is two half-width slices, and :func:`_attention_heads_first` runs the
norm, RoPE and the [text | image] concat on [B, H, S, D]. q . k is unchanged by
the common permutation; v and everything after are untouched. Unmasked joint
attention only (the TPU path passes no text mask); with a mask the shared
processor runs.
"""

from __future__ import annotations

import math
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.backends.tpu.core import heads_first
from difflet.backends.tpu.core.heads_first import pair_split_permutation  # noqa: F401 (re-export)
from difflet.backends.tpu.ops_impl import collective_matmul as cm


class _Precomputed:
    """The ``attn`` the processor sees: projections return precomputed tensors,
    the output projections pass through, everything else is the real module."""

    def __init__(self, attn, img_qkv, txt_qkv):
        self._attn = attn
        self.to_q, self.to_k, self.to_v = (lambda _x, t=t: t for t in img_qkv)
        self.add_q_proj, self.add_k_proj, self.add_v_proj = (lambda _x, t=t: t for t in txt_qkv)
        self.to_out = [nn.Identity(), *list(attn.to_out)[1:]]
        self.to_add_out = nn.Identity()

    def __getattr__(self, name):  # heads, norm_q, norm_k, norm_added_q, ...
        return getattr(self._attn, name)


HEADS_FIRST_ENV = "DIFFLET_TPU_QWEN_HEADS_FIRST"


def _heads_first_enabled() -> bool:
    return os.environ.get(HEADS_FIRST_ENV, "1").strip().lower() not in {"0", "false", "no"}


def _half_tables(freqs):
    """(cos, sin) at half width [S, D/2] from the processor's RoPE input: the real
    (cos, sin) pair (each value repeated twice, interleaved) or the complex table."""
    if isinstance(freqs, tuple):
        cos, sin = freqs
        return cos[..., 0::2], sin[..., 0::2]
    return freqs.real, freqs.imag


def _rope_halves(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Qwen's real RoPE (float32, as ``_apply_qwen_rope_real``) on [B, H, S, D] in
    the split-halves channel order, from half-width tables."""
    return heads_first.rope_split_halves(x, cos, sin, compute_dtype=torch.float32)


def _attention_heads_first(attn, img_qkv, txt_qkv, image_rotary_emb=None, attention=None):
    """The shared processor's unmasked joint attention with q/k/v as [B, H, S, D]
    from the start; q/k in the split-halves channel order (see the module docstring).
    Returns (image context, text context) as [B, S, H*D], before the output
    projections. ``attention``: ``difflet.ops.attention``, replaceable in tests."""
    if attention is None:
        from difflet.ops import attention
    heads = int(attn.heads)
    img_q, img_k, img_v = (heads_first.to_heads_first(t, heads) for t in img_qkv)
    txt_q, txt_k, txt_v = (heads_first.to_heads_first(t, heads) for t in txt_qkv)
    if attn.norm_q is not None:
        img_q = attn.norm_q(img_q)
    if attn.norm_k is not None:
        img_k = attn.norm_k(img_k)
    if attn.norm_added_q is not None:
        txt_q = attn.norm_added_q(txt_q)
    if attn.norm_added_k is not None:
        txt_k = attn.norm_added_k(txt_k)
    if image_rotary_emb is not None:
        img_freqs, txt_freqs = image_rotary_emb
        (ic, is_), (tc, ts) = _half_tables(img_freqs), _half_tables(txt_freqs)
        img_q, img_k = _rope_halves(img_q, ic, is_), _rope_halves(img_k, ic, is_)
        txt_q, txt_k = _rope_halves(txt_q, tc, ts), _rope_halves(txt_k, tc, ts)
    q = torch.cat([txt_q, img_q], dim=2)
    k = torch.cat([txt_k, img_k], dim=2)
    v = torch.cat([txt_v, img_v], dim=2)
    b, h, s_q, d = q.shape
    out = attention(q.reshape(b * h, s_q, d), k.reshape(b * h, s_q, d), v.reshape(b * h, s_q, d),
                    scale=1.0 / math.sqrt(d), causal=False, attention_mask=None,
                    tp_q=True, tp_k=True, tp_out=False)
    out = out.reshape(b, h, s_q, d).transpose(1, 2).flatten(2, 3).to(txt_q.dtype)
    seq_txt = txt_q.shape[2]
    img_out = out[:, seq_txt:]
    if len(attn.to_out) > 1:
        img_out = attn.to_out[1](img_out)
    return img_out, out[:, :seq_txt]


def install_split_rope_(blocks) -> int:
    """Permute every block's q/k head channels (projections and qk norms) for the
    split-halves RoPE of :func:`_attention_heads_first`. Returns the head size."""
    head_dim = None
    for block in blocks:
        attn = block.attn
        dim = int(attn.to_q.weight.shape[0]) // int(attn.heads)
        head_dim = head_dim or dim
        heads_first.permute_qk_(attn, dim)
    return head_dim


def _mlp(mlp, x):
    act = mlp.net[0]
    (h,) = cm.all_gather_matmul(x, (act.proj,))
    h = F.gelu(h, approximate=getattr(act, "approximate", "tanh"))
    return cm.matmul_reduce_scatter(h, mlp.net[2])


def _block_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    encoder_hidden_states_mask,
    temb,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
    modulate_index=None,
):
    # Streams arrive sequence-sharded: img (B, S_img/tp, D), txt (B, S_txt/tp, D).
    img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
    if self.zero_cond_t:
        temb = torch.chunk(temb, 2, dim=0)[0]
    txt_mod1, txt_mod2 = self.txt_mod(temb).chunk(2, dim=-1)

    img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1,
                                              modulate_index)
    txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)

    attn = self.attn
    img_qkv = cm.all_gather_matmul(img_modulated, (attn.to_q, attn.to_k, attn.to_v))
    txt_qkv = cm.all_gather_matmul(txt_modulated,
                                   (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj))
    kwargs = joint_attention_kwargs or {}
    if getattr(self, "_difflet_heads_first", False) and not kwargs and encoder_hidden_states_mask is None:
        img_ctx, txt_ctx = _attention_heads_first(attn, img_qkv, txt_qkv, image_rotary_emb)
    else:
        if getattr(self, "_difflet_heads_first", False):
            raise NotImplementedError(
                f"{HEADS_FIRST_ENV}: q/k are permuted for split-halves RoPE, but this call "
                f"needs the shared processor (attention kwargs {sorted(kwargs)}); set it to 0")
        img_ctx, txt_ctx = attn.processor(
            _Precomputed(attn, img_qkv, txt_qkv),
            hidden_states=img_qkv[0],          # read for nothing but shapes
            encoder_hidden_states=txt_qkv[0],  # the processor splits at its length
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )
    hidden_states = hidden_states + img_gate1 * cm.matmul_reduce_scatter(img_ctx, attn.to_out[0])
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * cm.matmul_reduce_scatter(
        txt_ctx, attn.to_add_out)

    img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2,
                                               modulate_index)
    hidden_states = hidden_states + img_gate2 * _mlp(self.img_mlp, img_modulated2)

    txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
    encoder_hidden_states = encoder_hidden_states + txt_gate2 * _mlp(self.txt_mlp, txt_modulated2)

    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return encoder_hidden_states, hidden_states


def install(module: torch.nn.Module, *, image_seq_len: int, text_seq_len: int) -> bool:
    """Swap the fused forward into every ``QwenImageSPTransformerBlock`` of a model built
    with ``sp_enabled``. Returns False (and changes nothing) when a stream does not
    split, so the caller keeps the shared Megatron-SP path."""
    from difflet.backends.tpu.ops_impl import parallel_mesh
    from difflet.models.qwen_image.modeling_qwen import QwenImageSPTransformerBlock

    blocks = [m for m in module.modules() if isinstance(m, QwenImageSPTransformerBlock)]
    if not blocks or not all(b.sp_enabled for b in blocks):
        raise ValueError("install() needs a Qwen-Image model built with sp_enabled=True")
    tp = parallel_mesh.get_tp_size()
    for name, seq in (("image", image_seq_len), ("text", text_seq_len)):
        if not cm.supports(seq, tp):
            print(f"[difflet tpu] {cm.SP_ENV}: {name} sequence {seq} does not split into "
                  f"2 x tp={tp}; using the unfused Megatron-SP path", flush=True)
            return False
    cm.prepare()
    heads_first = _heads_first_enabled()
    if heads_first:
        install_split_rope_(blocks)
    for block in blocks:
        block.forward = types.MethodType(_block_forward, block)
        block._difflet_heads_first = heads_first
    print(f"[difflet tpu] Qwen-Image sequence-parallel with collective matmuls "
          f"({len(blocks)} blocks, S_img={image_seq_len}, S_txt={text_seq_len}, tp={tp}"
          f"{', heads-first split-halves RoPE' if heads_first else ''})",
          flush=True)
    return True


__all__ = ["HEADS_FIRST_ENV", "install", "install_split_rope_", "pair_split_permutation"]
