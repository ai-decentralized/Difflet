"""Fused eager CUDA kernels for MiniMax-H3's per-row elementwise work (Triton).

In eager mode each DiT block spends much of its time on memory-bound elementwise
passes over a ``(rows, hidden)`` sequence of about 61k rows: RMS normalization,
AdaLN modulation, gated residuals, Q/K normalization with rotary embedding and the
SwiGLU gate. Unfused, every step re-reads and re-writes the full sequence, and the
per-row modulation gathers materialize full-size copies of tiny tables. Each kernel
below does one of those groups in a single pass, reading modulation rows by index.

The arithmetic follows the FP32 formulas of ``modeling_minimax_h3``'s native CUDA
path, with the same BF16 rounding points; reduction order differs, so results agree
to within BF16 rounding rather than bit for bit.

Each kernel is registered as a CUDA ``difflet_h3::`` custom op. Compiled blocks call
those opaque ops, so eager and compiled execution share the same elementwise kernels.
Inductor's own fusion would otherwise pick reduction strategies at compile time, and
those choices moved the output from one compilation to the next. Without Triton (a
CPU-only install) the module still imports, and the ops have no kernel.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs: the CUDA ops below have no kernel
    triton = tl = None

HAS_TRITON = triton is not None

if HAS_TRITON:

    def _block(n: int) -> int:
        return triton.next_power_of_2(n)

    @triton.jit
    def _rms_norm_modulate_kernel(
        x_ptr,
        weight_ptr,
        shift_ptr,
        scale_ptr,
        index_ptr,
        out_ptr,
        x_row_stride,
        table_row_stride,
        n,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(x * x, axis=0) / n + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        entry = tl.load(index_ptr + row)
        shift = tl.load(shift_ptr + entry * table_row_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        scale = tl.load(scale_ptr + entry * table_row_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        y = (x * rstd) * weight
        y = y * (1.0 + scale) + shift
        tl.store(out_ptr + row * n + cols, y.to(tl.bfloat16), mask=mask)

    def rms_norm_modulate(x, weight, eps, shift, scale, index):
        """``bf16((rms(x) * w) * (1 + scale[index]) + shift[index])`` per row, FP32 inside."""
        rows, n = x.numel() // x.shape[-1], x.shape[-1]
        flat = x.reshape(rows, n)
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=x.device)
        _rms_norm_modulate_kernel[(rows,)](
            flat,
            weight,
            shift,
            scale,
            index,
            out,
            flat.stride(0),
            shift.stride(0),
            n,
            float(eps),
            BLOCK=_block(n),
            num_warps=16,
        )
        return out.reshape(x.shape)

    @triton.jit
    def _gated_residual_kernel(
        residual_ptr,
        update_ptr,
        gate_ptr,
        index_ptr,
        out_ptr,
        update_row_stride,
        table_row_stride,
        n,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < n
        residual = tl.load(residual_ptr + row * n + cols, mask=mask, other=0.0).to(tl.float32)
        update = tl.load(update_ptr + row * update_row_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        entry = tl.load(index_ptr + row)
        gate = tl.load(gate_ptr + entry * table_row_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        tl.store(out_ptr + row * n + cols, (residual + gate * update).to(tl.bfloat16), mask=mask)

    def gated_residual(residual, gate, index, update):
        """``bf16(residual + gate[index] * update)`` per row, FP32 inside."""
        rows, n = residual.numel() // residual.shape[-1], residual.shape[-1]
        flat_residual = residual.reshape(rows, n).contiguous()
        flat_update = update.reshape(rows, n)
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=residual.device)
        block = 2048
        _gated_residual_kernel[(rows, triton.cdiv(n, block))](
            flat_residual,
            flat_update,
            gate,
            index,
            out,
            flat_update.stride(0),
            gate.stride(0),
            n,
            BLOCK=block,
            num_warps=8,
        )
        return out.reshape(residual.shape)

    @triton.jit
    def _head_norm_rotary_kernel(
        x_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        x_row_stride,
        table_row_stride,
        heads,
        eps,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        HEADS_BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.arange(0, HEADS_BLOCK)[:, None]
        col = tl.arange(0, HEAD_DIM)[None, :]
        mask = (head < heads) & (col < HEAD_DIM)
        base = x_ptr + row * x_row_stride + head * HEAD_DIM
        x = tl.load(base + col, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(x * x, axis=1) / HEAD_DIM + eps)[:, None]
        weight = tl.load(weight_ptr + col).to(tl.float32)
        # The normalization stores BF16 before the rotation reads it.
        normed = (weight * (x * rstd)).to(tl.bfloat16).to(tl.float32)

        half = ROTARY_DIM // 2
        rotary = col < ROTARY_DIM
        partner = tl.where(col < half, col + half, col - half)
        partner_x = tl.load(base + partner, mask=mask & rotary, other=0.0).to(tl.float32)
        partner_weight = tl.load(weight_ptr + partner, mask=rotary, other=0.0).to(tl.float32)
        partner_normed = (partner_weight * (partner_x * rstd)).to(tl.bfloat16).to(tl.float32)
        rotated = tl.where(col < half, -partner_normed, partner_normed)
        table = cos_ptr + row * table_row_stride + col
        # The tables are rounded to the activation dtype first, as in the formula.
        cos = tl.load(table, mask=rotary, other=1.0).to(tl.bfloat16).to(tl.float32)
        sin = tl.load(sin_ptr + row * table_row_stride + col, mask=rotary, other=0.0)
        sin = sin.to(tl.bfloat16).to(tl.float32)
        out = tl.where(rotary, normed * cos + rotated * sin, normed)
        tl.store(
            out_ptr + row * heads * HEAD_DIM + head * HEAD_DIM + col, out.to(tl.bfloat16), mask=mask
        )

    def head_norm_rotary(x, weight, eps, cos, sin):
        """Per-head RMS norm (BF16 store), then rotary on the leading channels; ``(B, S, H, D)``."""
        batch, rows, heads, head_dim = x.shape
        rotary_dim = cos.shape[-1]
        flat = x.reshape(batch * rows, heads, head_dim)
        if flat.stride(2) != 1 or flat.stride(1) != head_dim:
            flat = flat.contiguous()
        cos, sin = cos.contiguous(), sin.contiguous()
        out = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16, device=x.device)
        _head_norm_rotary_kernel[(rows,)](
            flat,
            weight,
            cos,
            sin,
            out,
            flat.stride(0),
            cos.stride(0),
            heads,
            float(eps),
            HEAD_DIM=head_dim,
            ROTARY_DIM=rotary_dim,
            HEADS_BLOCK=_block(heads),
            num_warps=8,
        )
        return out.reshape(x.shape)

    @triton.jit
    def _swiglu_kernel(x_ptr, out_ptr, x_row_stride, n, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < n
        value = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(x_ptr + row * x_row_stride + n + cols, mask=mask, other=0.0).to(tl.float32)
        activated = gate / (1.0 + tl.exp(-gate))
        tl.store(out_ptr + row * n + cols, (value * activated).to(tl.bfloat16), mask=mask)

    def swiglu(x):
        """``value * silu(gate)`` for ``x = [value | gate]`` on the last axis.

        FP32 inside with one BF16 rounding, as a fused gate kernel computes it (the
        unfused formula also rounds the activation before the product).
        """
        rows, n = x.numel() // x.shape[-1], x.shape[-1] // 2
        flat = x.reshape(rows, 2 * n)
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=x.device)
        block = 2048
        _swiglu_kernel[(rows, triton.cdiv(n, block))](
            flat, out, flat.stride(0), n, BLOCK=block, num_warps=8
        )
        return out.reshape(*x.shape[:-1], n)


def _rows(t: torch.Tensor) -> torch.Tensor:
    return t if t.stride(-1) == 1 else t.contiguous()


@torch.library.custom_op("difflet_h3::rms_norm_modulate", mutates_args=(), device_types="cuda")
def rms_norm_modulate_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    shift: torch.Tensor,
    scale: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    return rms_norm_modulate(
        _rows(x), weight.contiguous(), eps, _rows(shift), _rows(scale), index.contiguous()
    )


@rms_norm_modulate_op.register_fake
def _(x, weight, eps, shift, scale, index):
    return torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)


@torch.library.custom_op("difflet_h3::gated_residual", mutates_args=(), device_types="cuda")
def gated_residual_op(
    residual: torch.Tensor, gate: torch.Tensor, index: torch.Tensor, update: torch.Tensor
) -> torch.Tensor:
    return gated_residual(residual, _rows(gate), index.contiguous(), _rows(update))


@gated_residual_op.register_fake
def _(residual, gate, index, update):
    return torch.empty(residual.shape, dtype=torch.bfloat16, device=residual.device)


@torch.library.custom_op("difflet_h3::head_norm_rotary", mutates_args=(), device_types="cuda")
def head_norm_rotary_op(
    x: torch.Tensor, weight: torch.Tensor, eps: float, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return head_norm_rotary(x, weight.contiguous(), eps, cos, sin)


@head_norm_rotary_op.register_fake
def _(x, weight, eps, cos, sin):
    return torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)


@torch.library.custom_op("difflet_h3::swiglu", mutates_args=(), device_types="cuda")
def swiglu_op(x: torch.Tensor) -> torch.Tensor:
    return swiglu(_rows(x))


@swiglu_op.register_fake
def _(x):
    return torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=torch.bfloat16, device=x.device)


__all__ = [
    "gated_residual_op",
    "head_norm_rotary_op",
    "rms_norm_modulate_op",
    "swiglu_op",
]
