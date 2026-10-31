"""The CPU reference for an FP8 linear: the same rounding, in fp32 matmuls.

:class:`FakeQuantLinear` holds the fp8 weight and its scale exactly as the quantized checkpoint
stores them, fake-quantizes its input with the static ``input_scale`` and multiplies in fp32:
``(dequant(quant(x)) @ dequant(w_q).T + b)``. That is the arithmetic the device kernel performs
(fp8 operands, fp32 accumulation, one output scale), so a device result that matches this layer
is right, and one that matches the bf16 model instead has skipped the quantization.

:func:`fake_quantize_model_` swaps the target ``nn.Linear`` modules of a model in place, so the
CPU arbiter (``scripts/qwen_image_fp8_cpu_reference.py``) and the unit tests run diffusers'
own model through the quantization law without any backend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.quant.calibration import Calibration
from difflet.quant.fp8 import FP8_DTYPE, dequantize, fake_quantize, quantize_weight
from difflet.quant.spec import QuantSpec


class FakeQuantLinear(nn.Module):
    """``y = fake_quant(x) @ dequant(w_q)^T + b`` in fp32, returned in ``x``'s dtype."""

    def __init__(self, weight_q: torch.Tensor, weight_scale: torch.Tensor,
                 input_scale: torch.Tensor, bias: torch.Tensor | None, *, fp8_max: float) -> None:
        super().__init__()
        if weight_q.dtype != FP8_DTYPE:
            raise TypeError(f"weight_q must be {FP8_DTYPE}, got {weight_q.dtype}")
        self.register_buffer("weight", weight_q.detach().clone())
        self.register_buffer("weight_scale", weight_scale.detach().float().reshape(1).clone())
        self.register_buffer("input_scale", input_scale.detach().float().reshape(1).clone())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())
        self.fp8_max = float(fp8_max)
        self.in_features = int(weight_q.shape[1])
        self.out_features = int(weight_q.shape[0])

    @classmethod
    def from_linear(cls, linear: nn.Linear, input_scale: torch.Tensor, *, fp8_max: float
                    ) -> FakeQuantLinear:
        weight_q, weight_scale = quantize_weight(linear.weight.detach(), fp8_max=fp8_max)
        return cls(weight_q, weight_scale, input_scale, linear.bias, fp8_max=fp8_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = fake_quantize(x, self.input_scale, fp8_max=self.fp8_max)
        w = dequantize(self.weight, self.weight_scale)
        bias = None if self.bias is None else self.bias.float()
        return F.linear(xq, w, bias).to(x.dtype)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, fp8_max={self.fp8_max}, "
                f"weight_scale={self.weight_scale.item():.3e}, "
                f"input_scale={self.input_scale.item():.3e}")


def fake_quantize_model_(model: nn.Module, spec: QuantSpec, calibration: Calibration
                         ) -> list[str]:
    """Replace every ``nn.Linear`` the spec targets with a :class:`FakeQuantLinear`; returns the
    replaced module names. Raises when the calibration lacks a target."""
    targets = [(name, module) for name, module in model.named_modules()
               if spec.matches(name) and isinstance(module, nn.Linear)]
    if not targets:
        raise ValueError("the spec's target patterns match no nn.Linear in the model")
    missing = calibration.covers([name for name, _ in targets])
    if missing:
        raise ValueError(f"calibration lacks {len(missing)} targets, e.g. {missing[:3]}")
    for name, linear in targets:
        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf, FakeQuantLinear.from_linear(
            linear, calibration.input_scale(name, spec), fp8_max=spec.fp8_max))
    return [name for name, _ in targets]


__all__ = ["FakeQuantLinear", "fake_quantize_model_"]
