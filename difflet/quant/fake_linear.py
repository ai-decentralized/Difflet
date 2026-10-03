"""CPU emulation of an FP8-quantized linear layer (same math as the device path).

``FakeQuantLinear`` replaces an ``nn.Linear`` (the CPU backend's
``ColumnParallelLinear`` / ``RowParallelLinear`` are ``nn.Linear`` subclasses):
the weight is stored as fp8 e4m3fn plus its float32 absmax scale, activations
are round-tripped through fp8 with a dynamic per-tensor scale (W8A8), and the
matmul accumulates in fp32. It is the numerical reference the Trainium
NxD quantized layers are checked against, and what CPU metric scripts run.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from difflet.quant.fp8 import fp8_linear_reference, quantize_weight
from difflet.quant.spec import QuantSpec


class FakeQuantLinear(nn.Module):
    def __init__(
        self,
        src: nn.Linear,
        spec: QuantSpec,
        name: str | None = None,
        calibration: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(src.in_features)
        self.out_features = int(src.out_features)
        self.spec = spec
        weight_fp8, scale = quantize_weight(src.weight.detach(), spec.weight_granularity)
        self.register_buffer("weight", weight_fp8)
        self.register_buffer("weight_scale", scale)
        if src.bias is not None:
            self.bias = nn.Parameter(src.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        # Static activation scale (spec.calibration): the layer's calibrated input
        # absmax * STATIC_ACT_MARGIN / 240, looked up like the checkpoint writer does
        # (HF name, Difflet rename, fused proj_out halves). None = dynamic per call.
        if spec.calibration:
            if name is None:
                raise ValueError("a calibrated QuantSpec needs the layer's qualified name")
            from difflet.quant.checkpoint import calibrated_amax
            from difflet.quant.fp8 import FP8_MAX, STATIC_ACT_MARGIN

            layers = calibration if calibration is not None else spec.calibration_layers()
            amax = calibrated_amax(layers, name)
            self.register_buffer("input_scale", torch.tensor(amax * STATIC_ACT_MARGIN / FP8_MAX, dtype=torch.float32))
        else:
            self.input_scale = None

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quant={self.spec.label()}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fp8_linear_reference(x, self.weight, self.weight_scale, self.bias, input_scale=self.input_scale)


def quantize_module_(
    model: nn.Module,
    spec: QuantSpec,
    *,
    linear_types: tuple[type, ...] = (nn.Linear,),
) -> dict[str, Any]:
    """In-place: swap every target ``nn.Linear`` (by dotted name suffix) for a
    ``FakeQuantLinear``. Returns ``{"num_quantized", "quantized": [names]}``."""
    swapped: list[str] = []
    calibration = spec.calibration_layers() if spec.calibration else None
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            qualified = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, linear_types) and spec.matches(qualified):
                setattr(parent, child_name, FakeQuantLinear(child, spec, name=qualified, calibration=calibration))
                swapped.append(qualified)
    return {"num_quantized": len(swapped), "quantized": swapped}


__all__ = ["FakeQuantLinear", "quantize_module_"]
