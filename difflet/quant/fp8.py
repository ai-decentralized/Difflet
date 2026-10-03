"""FP8 e4m3 absmax quantization math (pure torch, CPU-runnable).

Same law as FastVideo ``fp8_config.py`` — ``scale = amax / FP8_MAX`` with a
floor so an all-zero tensor divides cleanly; values are clamped to the
representable range and rounded to nearest-even by the dtype cast — except for
the range itself. Tensors are stored as ``torch.float8_e4m3fn`` (max 448), but
Trainium's native fp8 e4m3 tops out at **240** and neuronx-cc's
``--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`` reinterprets the e4m3fn bit
patterns as that format: every encoding above 240 decodes as inf/NaN on the
device (trn2, 2026-10-01: a weight-only tiny probe returned all-NaN with 44.7 %
of its elements above 240). NxD's own quantizers clamp to
``DtypeBound.F8E4M3_MAX = 240`` for the same reason; so does this module, and
the CPU reference therefore matches the device bit-for-bit in value.
Per-channel means one scale per output row of a ``[out, in]`` weight.
Activation quantization is dynamic: one absmax scale over the whole activation
tensor per call (FastVideo's ``granularity="tensor"`` default).
"""

from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn
# Trainium fp8 e4m3 range (NxD DtypeBound.F8E4M3_MAX), not torch.finfo(e4m3fn).max == 448.
FP8_MAX = 240.0
# FastVideo FP8Config.FP8_MIN_SCALE: keeps scale finite/non-zero for zero tensors.
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


def weight_scale(weight: torch.Tensor, granularity: str) -> torch.Tensor:
    """Float32 absmax scale: shape ``[1]`` (tensor) or ``[out, 1]`` (channel)."""
    w = weight.detach().to(torch.float32)
    if granularity == "tensor":
        amax = w.abs().amax().reshape(1)
    elif granularity == "channel":
        if w.ndim != 2:
            raise ValueError(f"per-channel scale needs a 2-D [out, in] weight, got {w.ndim}-D")
        amax = w.abs().amax(dim=1, keepdim=True)
    else:
        raise ValueError(f"unknown weight granularity {granularity!r}")
    return (amax / FP8_MAX).clamp_min(FP8_MIN_SCALE)


def quantize_with_scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x.to(torch.float32) / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


def quantize_weight(weight: torch.Tensor, granularity: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``weight`` (any float dtype) -> (fp8 e4m3fn tensor, float32 scale)."""
    scale = weight_scale(weight, granularity)
    return quantize_with_scale(weight, scale).contiguous(), scale.contiguous()


def dequantize(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return (q.to(torch.float32) * scale.to(torch.float32)).to(dtype)


# Lean activation law (2026-10-03): absmax from the min / max reductions (no abs
# tensor), one fp32 multiply by the reciprocal scale (no divide), a direct
# fp32 -> fp8 cast and no clamp. (A bf16-domain multiply was tried first; XLA
# lowers it as fp32 multiply + an extra bf16 round trip, so fp32 is fewer
# passes.) Without a clamp, rounding could in principle push the absmax element
# above 240 (fp8 e4m3 spacing there is 16: 240 -> 256, and 256 is inf/NaN on
# Trainium), so the scale carries a 2^-7 margin: the scaled absmax lands at
# ~238 and no rounding can reach 248, the round-to-nearest boundary of 256.
# Same law here so CPU-fp8 == device-fp8 in value.
ACT_SCALE_MARGIN = 1.0 + 2.0**-7


def activation_scale(x: torch.Tensor) -> torch.Tensor:
    """Dynamic per-tensor absmax scale of an activation (shape ``[1]``, float32)."""
    lo, hi = torch.aminmax(x.detach())
    amax = torch.maximum(hi, -lo).to(torch.float32).reshape(1)
    return (amax / FP8_MAX * ACT_SCALE_MARGIN).clamp_min(FP8_MIN_SCALE)


def quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` -> (fp8 tensor, float32 scale) with the lean law: one fp32 multiply by
    the reciprocal scale, direct cast to fp8, no clamp."""
    scale = activation_scale(x)
    return (x.to(torch.float32) * (1.0 / scale)).to(FP8_DTYPE), scale


def fake_quant_activation(x: torch.Tensor) -> torch.Tensor:
    """Round-trip an activation through fp8 (dynamic per-tensor scale)."""
    q, scale = quantize_activation(x)
    return dequantize(q, scale, x.dtype)


def fp8_linear_reference(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale_: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference W8A8 ``x @ W^T + b``: dynamic per-tensor fp8 activation, fp8
    weight, fp32 accumulation.

    The activation is quantized in its own dtype (the device law) and the fp8
    values are used exactly; the dequantize-then-matmul form is bit-equivalent
    in value to a true fp8 GEMM with fp32 accumulate (the products are exact
    in fp32); only the accumulation order can differ from the device.
    """
    q, scale = quantize_activation(x)
    x32 = q.to(torch.float32) * scale
    w32 = dequantize(weight_fp8, weight_scale_)
    out = x32 @ w32.t()
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(x.dtype)


__all__ = [
    "ACT_SCALE_MARGIN",
    "FP8_DTYPE",
    "FP8_MAX",
    "FP8_MIN_SCALE",
    "activation_scale",
    "dequantize",
    "fake_quant_activation",
    "fp8_linear_reference",
    "quantize_activation",
    "quantize_weight",
    "quantize_with_scale",
    "weight_scale",
]
