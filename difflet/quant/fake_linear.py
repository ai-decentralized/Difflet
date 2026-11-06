"""The CPU reference for an MXFP8 linear: the same rounding, in fp32 matmuls.

:class:`FakeQuantMXLinear` holds the fp8 weight and its block scales exactly as the quantized
checkpoint stores them, block-quantizes its input per call (as the kernel does) and multiplies
in fp32: ``mx(x) @ mx(w)^T + b``. That is the arithmetic the device kernel performs (fp8
operands, block scales, fp32 accumulation), so a device result that matches this layer is
right, and one that matches the bf16 model instead has skipped the quantization.

:func:`fake_quantize_model_` swaps the target ``nn.Linear`` modules of a model in place, so the
CPU arbiter (``scripts/qwen_image_fp8_cpu_reference.py``) and the unit tests run diffusers'
own model through the quantization law without any backend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.quant import mx
from difflet.quant.spec import QuantSpec


class FakeQuantMXLinear(nn.Module):
    """``y = mx(x) @ mx(w)^T + b`` in fp32, both operands block-quantized in the device's block
    layout (``difflet/quant/mx.py``), the activation freshly per call (as the kernel does), the
    weight once. ``k_shards`` > 1 quantizes both in that many K ranges: the row-parallel layer
    under TP, where each rank quantizes its own K range."""

    def __init__(self, weight_q: torch.Tensor, weight_scale: torch.Tensor,
                 bias: torch.Tensor | None, *, k_shards: int = 1) -> None:
        super().__init__()
        if weight_q.dtype != mx.FP8_DTYPE or weight_scale.dtype != torch.uint8:
            raise TypeError("FakeQuantMXLinear takes an fp8 weight and uint8 E8M0 scales")
        self.register_buffer("weight", weight_q.detach().clone())
        self.register_buffer("weight_scale", weight_scale.detach().clone())
        self.register_buffer("weight_dequant",
                             mx.dequantize_mx_sharded(weight_q, weight_scale, k_shards))
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())
        self.k_shards = int(k_shards)
        self.in_features = int(weight_q.shape[1])
        self.out_features = int(weight_q.shape[0])

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, k_shards: int = 1) -> FakeQuantMXLinear:
        weight_q, weight_scale = mx.quantize_mx_sharded(linear.weight.detach(), k_shards)
        return cls(weight_q, weight_scale, linear.bias, k_shards=k_shards)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        xq = mx.fake_quantize_mx_sharded(x.reshape(-1, shape[-1]), self.k_shards)
        bias = None if self.bias is None else self.bias.float()
        out = F.linear(xq, self.weight_dequant, bias)
        return out.reshape(*shape[:-1], self.out_features).to(x.dtype)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, mxfp8 block32, "
                f"k_shards={self.k_shards}")


def fake_quantize_model_(model: nn.Module, spec: QuantSpec) -> list[str]:
    """Replace every ``nn.Linear`` the spec targets with its CPU reference layer
    (:class:`FakeQuantMXLinear`); returns the replaced module names."""
    targets = [(name, module) for name, module in model.named_modules()
               if spec.matches(name) and isinstance(module, nn.Linear)]
    if not targets:
        raise ValueError("the spec's target patterns match no nn.Linear in the model")
    for name, linear in targets:
        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf, FakeQuantMXLinear.from_linear(linear, k_shards=spec.k_shards(name)))
    return [name for name, _ in targets]


__all__ = ["FakeQuantMXLinear", "fake_quantize_model_"]
