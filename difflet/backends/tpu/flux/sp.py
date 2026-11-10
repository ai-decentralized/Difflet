"""FLUX.1-dev on TPU with sequence parallelism over collective matmuls (``DIFFLET_TPU_SP=1``).

Under TP every block ends its attention and its MLP in an all-reduce (20% of a
compiled denoise step, trace 2026-11-03). As for Wan and Qwen-Image, each
all-reduce becomes a reduce-scatter fused into the row-parallel matmul and an
all-gather fused into the next column-parallel one
(``ops_impl.collective_matmul``); the per-token work between them runs on this
rank's 1/tp of the tokens. FLUX has no shared Megatron-SP to reuse (diffusers'
modeling, TP-sharded by ``models/flux/tp_sharding.py``), so :func:`install`
swaps, on this module instance only, the model forward and every block forward.

Layout. Each stream is split by sequence on its own: TP rank r holds text rows
``[r*T/tp, (r+1)*T/tp)`` and image rows ``[r*I/tp, (r+1)*I/tp)`` (128 + 1024
at 512 text tokens, 1024x1024). The inputs are sliced before the embedders, so
the embedders already run on the shard; the 64-channel output is gathered at
the end, after ``norm_out`` / ``proj_out``.

    joint block   img -AG+[to_q,to_k,to_v]-> ; txt -AG+[add_q,add_k,add_v]->
                  joint attention (the TP processor, unchanged, full sequence)
                  img -to_out[0]+RS-> ; txt -to_add_out+RS->
                  ff / ff_context: -AG+net[0].proj-> gelu -net[2]+RS->
    single block  x = [txt_r | img_r]  -AG+[to_q,to_k,to_v,proj_mlp]->
                  attention ; gelu(mlp)
                  -[proj_out_attn | proj_out_mlp]+RS-> (one reduce-scatter
                  for the sum of the two row-parallel halves)

Token order. A single block gathers ``[txt_r | img_r]`` from every rank, so its
full sequence is ``[txt_0, img_0, txt_1, img_1, ...]`` rather than ``[txt | img]``.
FLUX's attention has no mask and RoPE is per token, so the attention result is
the same for any order as long as every token keeps its own rotary angle: the
cos/sin rows are permuted to the gathered order once per forward
(``single_order``). The joint blocks gather each stream separately, so they see
the natural order.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.backends.tpu.ops_impl import collective_matmul as cm

#: The collective matmuls FLUX uses: RS over one and two pairs, AG over 1, 3, 4 weights.
FLUX_OPS = (("rs", 1), ("rs", 2), ("ag", 1), ("ag", 3), ("ag", 4))


def single_order(text_len: int, image_len: int, tp: int) -> torch.Tensor:
    """Indices into the natural ``[txt | img]`` order, in the gathered single-block
    order ``[txt_0, img_0, txt_1, img_1, ...]``."""
    t, i = int(text_len) // int(tp), int(image_len) // int(tp)
    parts = []
    for rank in range(int(tp)):
        parts.append(torch.arange(rank * t, (rank + 1) * t))
        parts.append(int(text_len) + torch.arange(rank * i, (rank + 1) * i))
    return torch.cat(parts)


class _Precomputed:
    """The ``attn`` the TP processor sees: projections return tensors already
    computed by the fused all-gathers, the output projections pass through."""

    def __init__(self, attn, qkv, added_qkv=None):
        self._attn = attn
        self.to_q, self.to_k, self.to_v = (lambda _x, t=t: t for t in qkv)
        if added_qkv is not None:
            self.add_q_proj, self.add_k_proj, self.add_v_proj = (lambda _x, t=t: t for t in added_qkv)
        self.to_out = [nn.Identity(), *list(attn.to_out)[1:]] if hasattr(attn, "to_out") else None
        self.to_add_out = nn.Identity()

    def __getattr__(self, name):  # heads, norm_q, norm_k, added_kv_proj_dim, ...
        return getattr(self._attn, name)


def _ff(ff, x):
    act = ff.net[0]
    (h,) = cm.all_gather_matmul(x, (act.proj,))
    h = F.gelu(h, approximate=getattr(act, "approximate", "tanh"))
    return cm.matmul_reduce_scatter(h, ff.net[2])


def _joint_block_forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                         joint_attention_kwargs=None):
    # img (B, I/tp, D), txt (B, T/tp, D)
    norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_e, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb)
    attn = self.attn
    qkv = cm.all_gather_matmul(norm_h, (attn.to_q, attn.to_k, attn.to_v))
    added = cm.all_gather_matmul(norm_e, (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj))
    img_ctx, txt_ctx = attn.processor(
        _Precomputed(attn, qkv, added),
        hidden_states=qkv[0],          # read for nothing but shapes
        encoder_hidden_states=added[0],  # the processor splits at its length
        image_rotary_emb=image_rotary_emb,
        **(joint_attention_kwargs or {}),
    )
    hidden_states = hidden_states + gate_msa.unsqueeze(1) * cm.matmul_reduce_scatter(
        img_ctx, attn.to_out[0])
    norm_h = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * _ff(self.ff, norm_h)

    encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * cm.matmul_reduce_scatter(
        txt_ctx, attn.to_add_out)
    norm_e = self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * _ff(self.ff_context, norm_e)
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    return encoder_hidden_states, hidden_states


def _single_block_forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None,
                          joint_attention_kwargs=None):
    text_len = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [txt_r | img_r]
    residual = hidden_states
    norm, gate = self.norm(hidden_states, emb=temb)
    attn = self.attn
    q, k, v, mlp = cm.all_gather_matmul(norm, (attn.to_q, attn.to_k, attn.to_v, self.proj_mlp))
    attn_out = attn.processor(
        _Precomputed(attn, (q, k, v)), hidden_states=q, image_rotary_emb=image_rotary_emb,
        **(joint_attention_kwargs or {}),
    )
    proj = cm.matmul2_reduce_scatter(attn_out, self.proj_out_attn, self.act_mlp(mlp), self.proj_out_mlp,
                                     bias=self.proj_out_attn.bias)
    hidden_states = residual + gate.unsqueeze(1) * proj
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states[:, :text_len], hidden_states[:, text_len:]


def _rows(x: torch.Tensor, rank: int, tp: int) -> torch.Tensor:
    n = x.shape[1] // tp
    return x[:, rank * n:(rank + 1) * n]


def _model_forward(self, hidden_states, encoder_hidden_states=None, pooled_projections=None,
                   timestep=None, img_ids=None, txt_ids=None, guidance=None,
                   joint_attention_kwargs=None, return_dict=False, **unsupported):
    """``FluxTransformer2DModel.forward`` (diffusers 0.38) with sequence-sharded streams."""
    from difflet.backends.tpu.ops_impl import parallel_mesh
    from difflet.ops import gather_from_sequence_parallel_region

    if any(v is not None for v in unsupported.values()):
        raise NotImplementedError(f"FLUX SP forward does not take {sorted(unsupported)}")
    tp, rank = parallel_mesh.get_tp_size(), parallel_mesh.get_tp_rank()
    # Slice before the embedders: they are per token, so they run on the shard.
    hidden_states = self.x_embedder(_rows(hidden_states, rank, tp))
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    temb = (self.time_text_embed(timestep, pooled_projections) if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections))
    encoder_hidden_states = self.context_embedder(_rows(encoder_hidden_states, rank, tp))

    ids = torch.cat((txt_ids, img_ids), dim=0)
    cos, sin = self.pos_embed(ids)
    joint_rope = (cos, sin)
    order = self._difflet_sp_single_order.to(cos.device)
    single_rope = (cos.index_select(0, order), sin.index_select(0, order))

    for block in self.transformer_blocks:
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
            image_rotary_emb=joint_rope, joint_attention_kwargs=joint_attention_kwargs)
    for block in self.single_transformer_blocks:
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
            image_rotary_emb=single_rope, joint_attention_kwargs=joint_attention_kwargs)

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)  # (B, I/tp, out_channels)
    output = gather_from_sequence_parallel_region(output, dim=1)
    return (output,) if not return_dict else types.SimpleNamespace(sample=output)


def install(module: nn.Module, *, image_seq_len: int, text_seq_len: int) -> bool:
    """Swap the SP forwards into a TP-sharded ``FluxTransformer2DModel``. Returns
    False (and changes nothing) when a stream does not split."""
    from difflet.backends.tpu.ops_impl import parallel_mesh

    tp = parallel_mesh.get_tp_size()
    for name, seq in (("image", image_seq_len), ("text", text_seq_len),
                      ("joint", int(image_seq_len) + int(text_seq_len))):
        if not cm.supports(seq, tp) or int(seq) % tp:
            print(f"[difflet tpu] {cm.SP_ENV}: FLUX {name} sequence {seq} does not split into "
                  f"2 x tp={tp}; SP not installed", flush=True)
            return False
    if not all(hasattr(b, "proj_out_attn") for b in module.single_transformer_blocks):
        raise ValueError("install() needs a FLUX transformer sharded by shard_flux_transformer")
    cm.prepare(FLUX_OPS)
    for block in module.transformer_blocks:
        block.forward = types.MethodType(_joint_block_forward, block)
    for block in module.single_transformer_blocks:
        block.forward = types.MethodType(_single_block_forward, block)
    module.register_buffer("_difflet_sp_single_order",
                           single_order(text_seq_len, image_seq_len, tp), persistent=False)
    module.forward = types.MethodType(_model_forward, module)
    print(f"[difflet tpu] FLUX sequence-parallel with collective matmuls "
          f"({len(module.transformer_blocks)} joint + {len(module.single_transformer_blocks)} single "
          f"blocks, S_img={image_seq_len}, S_txt={text_seq_len}, tp={tp})", flush=True)
    return True


__all__ = ["FLUX_OPS", "install", "single_order"]
