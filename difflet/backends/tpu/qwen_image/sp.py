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
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    img_ctx, txt_ctx = attn.processor(
        _Precomputed(attn, img_qkv, txt_qkv),
        hidden_states=img_qkv[0],          # read for nothing but shapes
        encoder_hidden_states=txt_qkv[0],  # the processor splits at its length
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        image_rotary_emb=image_rotary_emb,
        **(joint_attention_kwargs or {}),
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
    for block in blocks:
        block.forward = types.MethodType(_block_forward, block)
    print(f"[difflet tpu] Qwen-Image sequence-parallel with collective matmuls "
          f"({len(blocks)} blocks, S_img={image_seq_len}, S_txt={text_seq_len}, tp={tp})",
          flush=True)
    return True


__all__ = ["install"]
