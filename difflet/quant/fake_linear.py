"""The CPU reference for an MXFP8 linear: the same rounding, in fp32 matmuls.

:class:`FakeQuantMXLinear` holds the fp8 weight and its block scales exactly as the quantized
checkpoint stores them, block-quantizes its input per call (as the kernel does) and multiplies
in fp32: ``mx(x) @ mx(w)^T + b``. That is the arithmetic the device kernel performs (fp8
operands, block scales, fp32 accumulation), so a device result that matches this layer is
right, and one that matches the bf16 model instead has skipped the quantization.

:func:`fake_quantize_model_` swaps the target ``nn.Linear`` modules of a model in place, so the
CPU arbiter (``scripts/qwen_image_fp8_cpu_reference.py``) and the unit tests run diffusers'
own model through the quantization law without any backend. A linear the spec splits
(``QuantSpec.split_linears``: the single-stream ``proj_out`` of FLUX and HunyuanVideo) runs as
its two column ranges (:class:`SplitLinear`), each quantized on its own when a target names it:
the quantized checkpoint stores it so and the neuron model runs it so, and quantizing the fused
weight instead would group other K blocks than the device's.
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


def split_linear_columns(linear: nn.Linear) -> tuple[nn.Linear, nn.Linear]:
    """The halves of a split linear ``[N, K]`` over ``cat([attn, mlp])``, as the quantized
    checkpoint stores them (``checkpoint.py``): the attention half ``Linear(N, N)`` reads the
    first N input columns and holds the bias, the MLP half ``Linear(K - N, N)`` the rest and no
    bias. Copies in the source's dtype and device, so quantizing a half leaves the source as is."""
    n, k = linear.out_features, linear.in_features
    if k <= n:
        raise ValueError(f"a split linear needs K > N (its first N input columns are the "
                         f"attention half), got [{n}, {k}]")
    weight = linear.weight.detach()
    factory = {"dtype": weight.dtype, "device": weight.device}
    attn = nn.Linear(n, n, bias=linear.bias is not None, **factory)
    mlp = nn.Linear(k - n, n, bias=False, **factory)
    with torch.no_grad():
        attn.weight.copy_(weight[:, :n])
        mlp.weight.copy_(weight[:, n:])
        if linear.bias is not None:
            attn.bias.copy_(linear.bias)
    return attn, mlp


class SplitLinear(nn.Module):
    """A split linear run as its halves, each a plain linear or its :class:`FakeQuantMXLinear`:
    ``attn(x[..., :split]) + mlp(x[..., split:])``, the sum the neuron model's two row-parallel
    halves reduce."""

    def __init__(self, attn: nn.Module, mlp: nn.Module, split: int) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.split = int(split)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x[..., :self.split]) + self.mlp(x[..., self.split:])

    def extra_repr(self) -> str:
        return f"split={self.split}"


def fake_quantize_model_(model: nn.Module, spec: QuantSpec) -> list[str]:
    """Replace every ``nn.Linear`` the spec targets with its CPU reference layer
    (:class:`FakeQuantMXLinear`); returns the quantized names in module order. A linear the
    spec splits becomes a :class:`SplitLinear` whose half ``<name>_attn`` / ``<name>_mlp`` is
    quantized when a target names it, over the half's own shard count (the row-parallel
    patterns name the halves, so the fused name's count would be 1), and counts as those names.
    A target pattern that matches no name so counted is refused before anything is replaced, so
    a halves target on a spec without the split is not silently left unquantized."""
    linears = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear)]
    names = []
    for name, _ in linears:
        names.extend((f"{name}_attn", f"{name}_mlp") if spec.is_split(name) else (name,))
    dead = spec.unmatched_targets(names)
    if dead:
        raise ValueError(f"target patterns match no nn.Linear in the model: {'; '.join(dead)}")
    quantized = []
    for name, linear in linears:
        if spec.is_split(name):
            halves = []
            for h, half in zip(("attn", "mlp"), split_linear_columns(linear)):
                part = f"{name}_{h}"
                if spec.matches(part):
                    half = FakeQuantMXLinear.from_linear(half, k_shards=spec.k_shards(part))
                    quantized.append(part)
                halves.append(half)
            layer = SplitLinear(*halves, split=linear.out_features)
        elif spec.matches(name):
            layer = FakeQuantMXLinear.from_linear(linear, k_shards=spec.k_shards(name))
            quantized.append(name)
        else:
            continue
        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, leaf, layer)
    return quantized


__all__ = ["FakeQuantMXLinear", "SplitLinear", "fake_quantize_model_", "split_linear_columns"]
