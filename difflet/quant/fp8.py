"""The FP8 quantization law: per-tensor symmetric ``float8_e4m3fn``.

``q = clamp(x / scale, -fp8_max, fp8_max).to(float8_e4m3fn)`` with ``scale = amax / fp8_max`` for
weights and ``scale = amax * margin / fp8_max`` for activations (``amax`` calibrated). Scales are
fp32 ``[1]`` tensors, never below :data:`FP8_MIN_SCALE`: an all-zero weight would otherwise give
``scale = 0`` and ``x / 0``. The clamp is kept in front of the cast on purpose: ``.to(float8)``
of a value past the range is NaN on some backends and saturation on others, and the device must
never see either.

Everything here runs in fp32 on any device, so the CPU reference (``fake_linear.py``), the
checkpoint writer (``checkpoint.py``) and the device layers all quantize with the same code.
"""

from __future__ import annotations

import torch

from difflet.quant.spec import FP8_MAX

FP8_DTYPE = torch.float8_e4m3fn
#: FastVideo's floor (``1 / (448 * 512)``): below it an fp8 weight tensor is all zeros anyway.
FP8_MIN_SCALE = 1.0 / (448.0 * 512.0)


def weight_scale(weight: torch.Tensor, *, fp8_max: float = FP8_MAX) -> torch.Tensor:
    """Per-tensor fp32 ``[1]`` scale of ``weight``: ``amax / fp8_max``, floored."""
    amax = weight.detach().abs().amax().float()
    return (amax / float(fp8_max)).clamp(min=FP8_MIN_SCALE).reshape(1)


def activation_scale(amax: float, *, margin: float, fp8_max: float = FP8_MAX) -> torch.Tensor:
    """Static fp32 ``[1]`` scale for an activation whose calibrated absmax is ``amax``."""
    value = float(amax) * float(margin) / float(fp8_max)
    return torch.tensor([max(value, FP8_MIN_SCALE)], dtype=torch.float32)


def quantize(x: torch.Tensor, scale: torch.Tensor, *, fp8_max: float = FP8_MAX) -> torch.Tensor:
    """``x / scale``, clamped to ``[-fp8_max, fp8_max]``, as ``float8_e4m3fn``."""
    scaled = x.float() * (1.0 / scale.float().reshape(()).to(x.device))
    return scaled.clamp(-float(fp8_max), float(fp8_max)).to(FP8_DTYPE)


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``q * scale`` in fp32."""
    return q.float() * scale.float().reshape(()).to(q.device)


def quantize_weight(weight: torch.Tensor, *, fp8_max: float = FP8_MAX
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(fp8 weight, fp32 [1] scale)`` of a full (unsharded) weight."""
    scale = weight_scale(weight, fp8_max=fp8_max)
    return quantize(weight, scale, fp8_max=fp8_max), scale


def fake_quantize(x: torch.Tensor, scale: torch.Tensor, *, fp8_max: float = FP8_MAX
                  ) -> torch.Tensor:
    """``dequantize(quantize(x))`` in fp32: what the device sees of ``x``, on any device."""
    return dequantize(quantize(x, scale, fp8_max=fp8_max), scale)


__all__ = [
    "FP8_DTYPE",
    "FP8_MIN_SCALE",
    "activation_scale",
    "dequantize",
    "fake_quantize",
    "quantize",
    "quantize_weight",
    "weight_scale",
]
