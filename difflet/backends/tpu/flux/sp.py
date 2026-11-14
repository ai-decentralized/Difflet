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

Modulation. Every block's AdaLN ``linear`` (3072 -> 6*3072 joint, 3*3072 single)
is one GEMV per step on a replicated weight: every rank read all ~6 GB of them
each step (four "reduce" fusions, 8.1 ms of a 139 ms step at ~750 GB/s; trace
2026-11-12). With ``DIFFLET_TPU_FLUX_SHARD_MODULATION`` (default on) :func:`install`
keeps rank r's rows ``[r*N/tp, (r+1)*N/tp)`` of each, the model forward computes
every block's local rows up front, and one all-gather of the concatenated
vectors rebuilds them all (a few hundred KB).

Heads first. The TP processor (``models/flux/tp_sharding.py``) unflattens q/k/v
to [B, S, H, D] and runs the per-head RMSNorm, the text/image concat and RoPE in
that shape before permuting to [B, H, S, D] for the attention. On TPU the minor
two dims (H=6, D=128) of [B, S, H, D] fill a (8, 128) tile only 6/8 and every
step through them is a relayout (``reshape [4608,6,128,1]``, ``copy
[1,4608,6,128]``: ~4.3 ms of a 131 ms step, trace 2026-11-12). With
``DIFFLET_TPU_FLUX_HEADS_FIRST`` (default on) :func:`_attention_heads_first`
permutes right after the unflatten, so the norm, the concat and RoPE run on
(S, D) tiles; same ops in the same order otherwise.

RoPE without the rotated copy. ``DIFFLET_TPU_FLUX_FUSED_ROPE`` (default on):
the split-halves rotation is written per half (:func:`_rope_split_halves`)
instead of diffusers' ``x*cos + cat([-x_hi, x_lo])*sin``, whose rotated copy
XLA wrote out; the same float ops, bit-identical (device busy -0.56 ms of a
127 ms step, 2026-11-12). Moving the MLP's tanh-GELU into the reduce-scatter
ring was tried and dropped: XLA fused it into the ring's matmuls, which then
ran slower than the GELU pass they replaced (+1.3 ms).

Token order. A single block gathers ``[txt_r | img_r]`` from every rank, so its
full sequence is ``[txt_0, img_0, txt_1, img_1, ...]`` rather than ``[txt | img]``.
FLUX's attention has no mask and RoPE is per token, so the attention result is
the same for any order as long as every token keeps its own rotary angle: the
cos/sin rows are permuted to the gathered order once per forward
(``single_order``). The joint blocks gather each stream separately, so they see
the natural order.
"""

from __future__ import annotations

import math
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.backends.tpu.core import heads_first
from difflet.backends.tpu.ops_impl import collective_matmul as cm

#: The collective matmuls FLUX uses: RS over one and two pairs, AG over 1, 3, 4 weights.
FLUX_OPS = (("rs", 1), ("rs", 2), ("ag", 1), ("ag", 3), ("ag", 4))

#: ``0`` keeps the AdaLN modulation linears replicated (each rank computes all of them).
SHARD_MODULATION_ENV = "DIFFLET_TPU_FLUX_SHARD_MODULATION"


#: ``0`` keeps the TP processor's [B, S, H, D] order around the attention.
HEADS_FIRST_ENV = "DIFFLET_TPU_FLUX_HEADS_FIRST"


def _heads_first_enabled() -> bool:
    return os.environ.get(HEADS_FIRST_ENV, "1").strip().lower() not in {"0", "false", "no"}


def _attention_heads_first(attn, qkv, added=None, image_rotary_emb=None, attention=None):
    """The TP processor's math on [B, H, S, D]: per-head RMSNorm, ``[text | image]``
    concat, RoPE, attention; returns [B, S, H*D] (text rows first when ``added``).
    ``attention``: ``difflet.ops.attention``, replaceable in tests."""
    from diffusers.models.embeddings import apply_rotary_emb

    if attention is None:
        from difflet.ops import attention
    heads = int(attn.heads)
    query, key, value = (heads_first.to_heads_first(t, heads) for t in qkv)
    query, key = attn.norm_q(query), attn.norm_k(key)
    if added is not None:
        eq, ek, ev = (heads_first.to_heads_first(t, heads) for t in added)
        eq, ek = attn.norm_added_q(eq), attn.norm_added_k(ek)
        query = torch.cat([eq, query], dim=2)
        key = torch.cat([ek, key], dim=2)
        value = torch.cat([ev, value], dim=2)
    if image_rotary_emb is not None:
        unbind = int(getattr(attn.processor, "rope_unbind_dim", -1))
        if unbind == -2 and getattr(attn, "_difflet_fused_rope", False):
            cos, sin = image_rotary_emb
            query = _rope_split_halves(query, cos.to(query.device), sin.to(query.device))
            key = _rope_split_halves(key, cos.to(key.device), sin.to(key.device))
        else:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=2,
                                     use_real_unbind_dim=unbind)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=2, use_real_unbind_dim=unbind)
    bsz, _, seq, head_dim = query.shape
    out_dtype = query.dtype
    out3 = attention(
        query.reshape(bsz * heads, seq, head_dim), key.reshape(bsz * heads, seq, head_dim),
        value.reshape(bsz * heads, seq, head_dim), scale=1.0 / math.sqrt(head_dim), causal=False,
        tp_q=True, tp_k=True, tp_out=False,
    )
    out = out3.reshape(bsz, heads, seq, head_dim).transpose(1, 2).reshape(bsz, seq, heads * head_dim)
    return out.to(out_dtype)


FUSED_ROPE_ENV = "DIFFLET_TPU_FLUX_FUSED_ROPE"
#: ``0``: the AG kernel returns q/k/v as [B, S, H*D] (relaid out before the
#: heads-first attention) instead of [B, H, S, D].
KERNEL_HEADS_ENV = "DIFFLET_TPU_FLUX_KERNEL_HEADS"


def _env_on(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in {"0", "false", "no"}


def _rope_split_halves(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """diffusers' ``apply_rotary_emb(x, (cos, sin), sequence_dim=2,
    use_real_unbind_dim=-2)`` for x [B, H, S, D], cos/sin [S, D], without the
    rotated copy (``core/heads_first.rope_split_halves``, float32 as diffusers)."""
    return heads_first.rope_split_halves(x, cos, sin, compute_dtype=torch.float32)


def _shard_modulation_enabled() -> bool:
    return os.environ.get(SHARD_MODULATION_ENV, "1").strip().lower() not in {"0", "false", "no"}


def _modulation_norms(module: nn.Module) -> list[nn.Module]:
    """The AdaLN norms whose ``linear`` is sharded, in the order the forward uses."""
    norms = []
    for block in module.transformer_blocks:
        norms += [block.norm1, block.norm1_context]
    norms += [block.norm for block in module.single_transformer_blocks]
    return norms


def _shard_rows_(linear: nn.Linear, rank: int, tp: int) -> None:
    """Keep rank's contiguous 1/tp of the output rows. Sliced on the host and
    assigned as new parameters (not ``copy_`` into the old ones), so the old
    replicated weight is freed and the new one gets a plain layout."""
    rows = linear.out_features // tp

    def part(tensor: torch.Tensor) -> nn.Parameter:
        host = tensor.detach().cpu()[rank * rows:(rank + 1) * rows].contiguous()
        return nn.Parameter(host.to(tensor.device), requires_grad=False)

    linear.weight = part(linear.weight)
    if linear.bias is not None:
        linear.bias = part(linear.bias)
    linear.out_features = rows


def _modulations(norms: list[nn.Module], temb: torch.Tensor, tp: int,
                 gather=None) -> list[torch.Tensor]:
    """Every norm's full modulation ``linear(silu(temb))`` [B, N]: each rank computes
    its row shard of all of them, one all-gather along dim 1 brings the rest
    (``gather``: the TP all-gather, replaceable in tests)."""
    if gather is None:
        from difflet.ops import gather_from_sequence_parallel_region as gather

    act = F.silu(temb)
    local = torch.cat([norm.linear(act) for norm in norms], dim=1)  # [B, sum N/tp]
    gathered = gather(local, dim=1)                                  # [B, tp * sum N/tp]
    gathered = gathered.view(local.shape[0], tp, local.shape[1])
    widths = [norm.linear.out_features for norm in norms]
    return [part.reshape(local.shape[0], tp * width)  # rank r's rows come r-th
            for part, width in zip(gathered.split(widths, dim=2), widths)]


def _adaln(norm: nn.Module, x: torch.Tensor, temb: torch.Tensor, modulation):
    """``norm(x, emb=temb)`` (AdaLayerNormZero / ...Single), from a precomputed
    modulation vector when there is one."""
    if modulation is None:
        return norm(x, emb=temb)
    if modulation.shape[1] == 3 * x.shape[-1]:  # AdaLayerNormZeroSingle
        shift_msa, scale_msa, gate_msa = modulation.chunk(3, dim=1)
        return norm.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None], gate_msa
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)
    x = norm.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
    return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


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
                         joint_attention_kwargs=None, modulation=(None, None)):
    # img (B, I/tp, D), txt (B, T/tp, D)
    norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = _adaln(
        self.norm1, hidden_states, temb, modulation[0])
    norm_e, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = _adaln(
        self.norm1_context, encoder_hidden_states, temb, modulation[1])
    attn = self.attn
    hd = getattr(self, "_difflet_kernel_heads", 0)
    qkv = cm.all_gather_matmul(norm_h, (attn.to_q, attn.to_k, attn.to_v), (hd, hd, hd))
    added = cm.all_gather_matmul(norm_e, (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj),
                                 (hd, hd, hd))
    if getattr(self, "_difflet_heads_first", False):
        out = _attention_heads_first(attn, qkv, added, image_rotary_emb)
        text_len = added[0].shape[-2] if added[0].dim() == 4 else added[0].shape[1]
        txt_ctx, img_ctx = out[:, :text_len], attn.to_out[1](out[:, text_len:])
    else:
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
                          joint_attention_kwargs=None, modulation=None):
    text_len = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [txt_r | img_r]
    residual = hidden_states
    norm, gate = _adaln(self.norm, hidden_states, temb, modulation)
    attn = self.attn
    hd = getattr(self, "_difflet_kernel_heads", 0)
    q, k, v, mlp = cm.all_gather_matmul(norm, (attn.to_q, attn.to_k, attn.to_v, self.proj_mlp),
                                        (hd, hd, hd, 0))
    if getattr(self, "_difflet_heads_first", False):
        attn_out = _attention_heads_first(attn, (q, k, v), None, image_rotary_emb)
    else:
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

    joint_count = len(self.transformer_blocks)
    if self._difflet_sp_modulation_norms:
        mods = _modulations(self._difflet_sp_modulation_norms, temb, tp)
        joint_mods = [(mods[2 * i], mods[2 * i + 1]) for i in range(joint_count)]
        single_mods = mods[2 * joint_count:]
    else:
        joint_mods = [(None, None)] * joint_count
        single_mods = [None] * len(self.single_transformer_blocks)
    for block, modulation in zip(self.transformer_blocks, joint_mods):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
            image_rotary_emb=joint_rope, joint_attention_kwargs=joint_attention_kwargs,
            modulation=modulation)
    for block, modulation in zip(self.single_transformer_blocks, single_mods):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
            image_rotary_emb=single_rope, joint_attention_kwargs=joint_attention_kwargs,
            modulation=modulation)

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
    # The Pallas AG kernel: -10.2% per step at 1024x1024 (v5e-4, 2026-11-11);
    # DIFFLET_TPU_SP_KERNEL=jax keeps the ring.
    cm.set_kernel_default("pallas")
    # The text stream's AG kernels (128 rows per device, 64-row halves) keep their
    # weights in VMEM instead of streaming them once per hop: device time q|k|v
    # 80 -> 67 us, FFN up 105 -> 88 us; busy -1.1 ms of a 126.5 ms step, output
    # bit-identical (v5e-4, 2026-11-12). The image stream (1024+ rows) streams.
    cm.set_resident_rows_default(128)
    heads_first = _heads_first_enabled()
    # The AG kernel writes q/k/v heads first for the heads-first attention:
    # the [B, S, H*D] -> [B, H, S, D] relayouts after it go away.
    head_dim = int(module.transformer_blocks[0].attn.head_dim)
    kernel_heads = head_dim if heads_first and _env_on(KERNEL_HEADS_ENV) else 0
    hf3 = (kernel_heads,) * 3
    cm.prepare(FLUX_OPS + ((("ag", 3, hf3), ("ag", 4, hf3 + (0,))) if kernel_heads else ()))
    fused_rope = _env_on(FUSED_ROPE_ENV)
    for block in [*module.transformer_blocks, *module.single_transformer_blocks]:
        block._difflet_heads_first = heads_first
        block._difflet_kernel_heads = kernel_heads
        block.attn._difflet_fused_rope = fused_rope
    for block in module.transformer_blocks:
        block.forward = types.MethodType(_joint_block_forward, block)
    for block in module.single_transformer_blocks:
        block.forward = types.MethodType(_single_block_forward, block)
    module.register_buffer("_difflet_sp_single_order",
                           single_order(text_seq_len, image_seq_len, tp), persistent=False)
    norms = _modulation_norms(module) if _shard_modulation_enabled() else []
    if any(norm.linear.out_features % tp for norm in norms):
        norms = []
    for norm in norms:
        _shard_rows_(norm.linear, parallel_mesh.get_tp_rank(), tp)
    # A plain list, not registered: the norms stay where they are in the module tree.
    object.__setattr__(module, "_difflet_sp_modulation_norms", norms)
    module.forward = types.MethodType(_model_forward, module)
    print(f"[difflet tpu] FLUX sequence-parallel with collective matmuls "
          f"({len(module.transformer_blocks)} joint + {len(module.single_transformer_blocks)} single "
          f"blocks, S_img={image_seq_len}, S_txt={text_seq_len}, tp={tp}"
          f"{', sharded modulation' if norms else ''}{', heads-first attention' if heads_first else ''}"
          f"{', fused RoPE' if fused_rope else ''}"
          f"{', q/k/v heads first from the kernel' if kernel_heads else ''})",
          flush=True)
    return True


__all__ = ["FLUX_OPS", "FUSED_ROPE_ENV", "HEADS_FIRST_ENV", "KERNEL_HEADS_ENV", "SHARD_MODULATION_ENV", "install", "single_order"]
