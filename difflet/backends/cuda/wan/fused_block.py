"""Fused eager CUDA kernels for Wan's per-token elementwise work (Triton).

In eager mode each Wan block runs its adaLN modulation, gated residuals and LayerNorms as
chains of fp32 elementwise kernels over the whole ``(tokens, dim)`` sequence (an fp32 copy,
the norm, a multiply, an add and a cast back to bf16 each), and rotates q and k as stride-2
slices of fp32 products. At 480x832x9 on one A100 those chains took about 200 of a DiT
forward's 761 ms. Each kernel below does one chain in a single pass, with the modeling's
fp32 arithmetic and its bf16 rounding points. The LayerNorm and RMS statistics reduce in a
different order, so results agree with the modeling within bf16 rounding, not bit for bit.

``fuse_blocks`` replaces each block's ``forward`` with ``fused_block_forward`` at load. The
CUDA transformer does this in eager mode only: under ``--torch-compile`` inductor fuses the
same chains itself. The fused forward is written for the layout the CUDA backend runs: one
device per process, no tensor, sequence or context parallelism (the platform refuses those
before loading), the interleaved rotary and a timestep per sample.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs: nothing here runs without a CUDA device
    triton = tl = None

HAS_TRITON = triton is not None

if HAS_TRITON:

    @triton.jit
    def _layer_norm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        shift_ptr,
        scale_ptr,
        out_ptr,
        rows_per_batch,
        n,
        eps,
        modulation_batch_stride,
        AFFINE: tl.constexpr,
        MODULATE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n
        x = tl.load(x_ptr + row * n + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / n
        centered = tl.where(mask, x - mean, 0.0)
        y = centered * tl.math.rsqrt(tl.sum(centered * centered, axis=0) / n + eps)
        if AFFINE:
            weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = y * weight + bias
        if MODULATE:
            batch = row // rows_per_batch
            scale = tl.load(
                scale_ptr + batch * modulation_batch_stride + cols, mask=mask, other=0.0
            )
            shift = tl.load(
                shift_ptr + batch * modulation_batch_stride + cols, mask=mask, other=0.0
            )
            y = y * (1.0 + scale.to(tl.float32)) + shift.to(tl.float32)
        tl.store(out_ptr + row * n + cols, y.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _gated_residual_kernel(
        residual_ptr,
        update_ptr,
        gate_ptr,
        out_ptr,
        rows_per_batch,
        n,
        gate_batch_stride,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < n
        residual = tl.load(residual_ptr + row * n + cols, mask=mask, other=0.0).to(tl.float32)
        update = tl.load(update_ptr + row * n + cols, mask=mask, other=0.0).to(tl.float32)
        gate_row = gate_ptr + (row // rows_per_batch) * gate_batch_stride
        gate = tl.load(gate_row + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * n + cols, (residual + update * gate).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _rms_norm_rotary_kernel(
        x_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        x_row_stride,
        rows_per_batch,
        n,
        eps,
        HEAD_DIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n
        base = x_ptr + row * x_row_stride
        x = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / n + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        # The norm stores bf16 before the rotation reads it, as the unfused ops do.
        own = (x * rstd * weight).to(tl.bfloat16).to(tl.float32)
        pair = cols ^ 1
        pair_x = tl.load(base + pair, mask=mask, other=0.0).to(tl.float32)
        pair_weight = tl.load(weight_ptr + pair, mask=mask, other=0.0).to(tl.float32)
        partner = (pair_x * rstd * pair_weight).to(tl.bfloat16).to(tl.float32)
        # Interleaved pairs (2i, 2i+1) of each head: out[2i] = x[2i] cos[2i] - x[2i+1] sin[2i+1],
        # out[2i+1] = x[2i] sin[2i+1] + x[2i+1] cos[2i].
        channel = cols % HEAD_DIM
        even = (cols % 2) == 0
        table = (row % rows_per_batch) * HEAD_DIM
        cos = tl.load(cos_ptr + table + tl.where(even, channel, channel - 1), mask=mask, other=0.0)
        sin = tl.load(sin_ptr + table + tl.where(even, channel + 1, channel), mask=mask, other=0.0)
        out = tl.where(even, own * cos - partner * sin, partner * sin + own * cos)
        tl.store(out_ptr + row * n + cols, out.to(tl.bfloat16), mask=mask)


def _rows(x: torch.Tensor) -> tuple[int, int, int]:
    """``(batch, tokens, channels)`` of a ``(B, S, C)`` activation."""
    batch, tokens, channels = x.shape
    return batch, tokens, channels


def _modulation(t: torch.Tensor, batch: int, channels: int) -> int:
    """The batch stride of a ``(B, 1, C)`` fp32 modulation chunk, which is read in place."""
    if t.dtype != torch.float32 or t.shape != (batch, 1, channels) or t.stride(-1) != 1:
        raise ValueError(f"expected a ({batch}, 1, {channels}) fp32 modulation, got {t.shape}")
    return t.stride(0)


def layer_norm(
    x: torch.Tensor,
    eps: float,
    *,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    shift: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """``bf16(layer_norm_fp32(x) [* weight + bias] [* (1 + scale) + shift])`` for a bf16
    ``(B, S, C)`` ``x``; ``shift``/``scale`` are ``(B, 1, C)`` fp32."""
    batch, tokens, channels = _rows(x)
    x = x.contiguous()
    out = torch.empty_like(x)
    affine, modulate = weight is not None, shift is not None
    stride = _modulation(shift, batch, channels) if modulate else 0
    if modulate and _modulation(scale, batch, channels) != stride:
        raise ValueError("shift and scale must share a layout")
    _layer_norm_kernel[(batch * tokens,)](
        x,
        weight if affine else x,
        bias if affine else x,
        shift if modulate else x,
        scale if modulate else x,
        out,
        tokens,
        channels,
        float(eps),
        stride,
        AFFINE=affine,
        MODULATE=modulate,
        BLOCK=triton.next_power_of_2(channels),
        num_warps=16,
    )
    return out


def gated_residual(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor):
    """``bf16(residual.float() + update.float() * gate)``; bf16 ``(B, S, C)`` residual and
    update, ``(B, 1, C)`` fp32 gate."""
    batch, tokens, channels = _rows(residual)
    residual, update = residual.contiguous(), update.contiguous()
    out = torch.empty_like(residual)
    block = 2048
    _gated_residual_kernel[(batch * tokens, triton.cdiv(channels, block))](
        residual,
        update,
        gate,
        out,
        tokens,
        channels,
        _modulation(gate, batch, channels),
        BLOCK=block,
        num_warps=8,
    )
    return out


def rms_norm_rotary(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """RMS norm over all of ``x``'s channels (fp32, weighted, bf16 store), then the interleaved
    rotary of every head with the ``(1, S, 1, head_dim)`` fp32 tables: what
    ``rms_norm_fp32`` and ``apply_rotary_emb`` compute in sequence, in one pass."""
    batch, tokens, channels = _rows(x)
    if x.stride(-1) != 1:
        x = x.contiguous()
    for table in (cos, sin):
        if table.dtype != torch.float32 or table.numel() != tokens * head_dim:
            raise ValueError(f"expected ({tokens}, {head_dim}) fp32 rotary tables")
    out = torch.empty((batch, tokens, channels), dtype=torch.bfloat16, device=x.device)
    _rms_norm_rotary_kernel[(batch * tokens,)](
        x,
        weight,
        cos.contiguous(),
        sin.contiguous(),
        out,
        x.stride(1),
        tokens,
        channels,
        float(eps),
        HEAD_DIM=head_dim,
        BLOCK=triton.next_power_of_2(channels),
        num_warps=16,
    )
    return out


def _attention(attn, hidden_states, kv_source, rotary_emb) -> torch.Tensor:
    """``WanAttention.forward`` at tp=1 without sequence or context parallelism: the q/k RMS
    norm (with the rotary for self-attention) in one kernel each, and SDPA on the head views
    so q, k and v are not copied into a head-major layout."""
    q = attn.to_q(hidden_states)
    k = attn.to_k(kv_source)
    v = attn.to_v(kv_source)
    if rotary_emb is not None:
        cos, sin = rotary_emb
        q = rms_norm_rotary(q, attn.norm_q.weight, attn.norm_q.eps, cos, sin, attn.head_dim)
        k = rms_norm_rotary(k, attn.norm_k.weight, attn.norm_k.eps, cos, sin, attn.head_dim)
    else:
        q = F.rms_norm(q, (q.shape[-1],), weight=attn.norm_q.weight, eps=float(attn.norm_q.eps))
        k = F.rms_norm(k, (k.shape[-1],), weight=attn.norm_k.weight, eps=float(attn.norm_k.eps))
    batch = q.shape[0]
    q, k, v = (
        t.unflatten(-1, (attn.local_heads, attn.head_dim)).transpose(1, 2) for t in (q, k, v)
    )
    out = F.scaled_dot_product_attention(q, k, v, scale=attn.head_dim**-0.5)
    out = out.transpose(1, 2).reshape(batch, out.shape[2], -1)
    return attn.to_out[1](attn.to_out[0](out))


def fused_block_forward(block, hidden_states, encoder_hidden_states, temb, rotary_emb):
    """``WanTransformerBlock.forward`` with each elementwise chain in one kernel."""
    shift, scale, gate, c_shift, c_scale, c_gate = (block.scale_shift_table + temb.float()).chunk(
        6, dim=1
    )
    norm_h = layer_norm(hidden_states, block.norm1.eps, shift=shift, scale=scale)
    attn_out = _attention(block.attn1, norm_h, norm_h, rotary_emb)
    hidden_states = gated_residual(hidden_states, attn_out, gate)

    norm2 = block.norm2
    norm_h = layer_norm(hidden_states, norm2.eps, weight=norm2.weight, bias=norm2.bias)
    hidden_states = hidden_states + _attention(block.attn2, norm_h, encoder_hidden_states, None)

    norm_h = layer_norm(hidden_states, block.norm3.eps, shift=c_shift, scale=c_scale)
    return gated_residual(hidden_states, block.ffn(norm_h), c_gate)


def fuse_blocks(module) -> int:
    """Give every block of a loaded ``WanTransformer3DModel`` the fused forward. Returns the
    number of blocks changed: 0 without Triton (the blocks keep the modeling's forward)."""
    if not HAS_TRITON:
        return 0
    for block in module.blocks:
        block.forward = types.MethodType(fused_block_forward, block)
    return len(module.blocks)


__all__ = [
    "HAS_TRITON",
    "fuse_blocks",
    "fused_block_forward",
    "gated_residual",
    "layer_norm",
    "rms_norm_rotary",
]
