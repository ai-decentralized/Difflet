"""Plain PyTorch formulas of MiniMax-H3's fused CUDA ops (``difflet_h3::``).

The CUDA tests compare the Triton kernels with them; the CPU tests register them as the
ops' CPU kernels, so the CUDA arithmetic runs without a GPU.
"""

from __future__ import annotations

import torch


def rms_norm_modulate(x, weight, eps, shift, scale, index):
    value = x.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight.float()
    selected_scale = scale.index_select(0, index).float()
    return (normalized * (1 + selected_scale) + shift.index_select(0, index).float()).to(x.dtype)


def gated_residual(residual, gate, index, update):
    return (residual.float() + gate.index_select(0, index).float() * update.float()).to(
        residual.dtype
    )


def head_norm_rotary(x, weight, eps, cos, sin):
    value = x.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    normed = (weight.float() * normalized).to(x.dtype)
    rotary_dim = cos.shape[-1]
    head = normed[..., :rotary_dim].float()
    c = cos.to(x.dtype)[None, :, None, :].float()
    s = sin.to(x.dtype)[None, :, None, :].float()
    first, second = head.chunk(2, -1)
    rotated = (head * c + torch.cat((-second, first), -1) * s).to(x.dtype)
    return torch.cat((rotated, normed[..., rotary_dim:]), -1).contiguous()


def swiglu(x):
    value, gate = x.float().chunk(2, -1)
    return (value * torch.nn.functional.silu(gate)).to(x.dtype)


_REGISTERED = False


def register_cpu_kernels() -> None:
    """Give the CUDA-only ``difflet_h3::`` ops these formulas as CPU kernels (once)."""
    global _REGISTERED
    if _REGISTERED:
        return
    import difflet.backends.cuda.minimax_h3.fused_ops  # noqa: F401  (defines the ops)

    for name, fn in (
        ("rms_norm_modulate", rms_norm_modulate),
        ("gated_residual", gated_residual),
        ("head_norm_rotary", head_norm_rotary),
        ("swiglu", swiglu),
    ):
        torch.library.register_kernel(f"difflet_h3::{name}", "cpu", fn)
    _REGISTERED = True
