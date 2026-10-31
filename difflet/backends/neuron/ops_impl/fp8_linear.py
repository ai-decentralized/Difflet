"""FP8 (W8A8, static per-tensor scales) linears for the neuron backend.

The quantized twins of ``nn.Linear`` and of this backend's ``ColumnParallelLinear`` /
``RowParallelLinear`` (``linear.py``): the same constructor arguments, sharding, collectives and
bias semantics, with the weight stored as ``float8_e4m3fn`` beside two fp32 ``[1]`` scales,
``weight_scale`` (the weight's ``amax / fp8_max``) and ``input_scale`` (the calibrated static
activation scale, ``difflet/quant/calibration.py``). The quantized checkpoint
(``difflet/quant/checkpoint.py``) holds exactly these three tensors per linear under the
diffusers key names, so the lazy per-rank loader fills them by name.

Two compute paths, chosen by ``compute``:

* ``"dequant"``: ``F.linear(x, (weight * weight_scale).to(x.dtype))``. The weight is fp8 in HBM
  (half the bytes) and the matmul is bf16: a memory-only mode, measured 5-8 % slower than bf16
  on trn3. It is the path the plumbing is proven with and the fallback for debugging.
* ``"nki"``: the activation is quantized in the graph with ``input_scale`` and the FP8 x FP8
  matmul runs in an NKI kernel on the tensor engine's MX path (``fp8_kernels.py``), fp32
  accumulation, bf16 out, scaled by ``input_scale * weight_scale``.

Decision D42 (``core/config_checks.require_bf16_on_device``) is untouched: activations between
layers and the row-parallel all-reduce stay bf16; fp8 exists only inside the matmul.

Why the fp8 weight is a ``Parameter`` and the scales are buffers: the module is built on
``meta`` and ``materialize_meta_`` (``_torch_megatron/weights.py``) gives every meta parameter
storage in the application dtype, except float8 parameters, which keep theirs; a meta buffer
would keep its dtype but ``build_on_meta`` makes buffers real anyway (they are tiny). The
scales are fp32 so the loader copies them without rounding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from difflet.backends.neuron.ops_impl.collectives import (
    gather_tp_dim,
    get_tp_size,
    reduce_tp,
    scatter_tp_dim,
)
from difflet.backends.neuron.ops_impl.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    _reject_sequence_parallel,
    _split_evenly,
)
from difflet.quant.spec import COMPUTE_MODES, FP8_MAX, QuantSpec

FP8_DTYPE = torch.float8_e4m3fn


def _check_compute(compute: str) -> str:
    if compute not in COMPUTE_MODES:
        raise ValueError(f"fp8 compute must be one of {COMPUTE_MODES}, got {compute!r}")
    return compute


class _FP8LinearBase(nn.Module):
    """The fp8 weight, its scales, the bias, and the two compute paths; subclasses add the
    sharding and the collectives."""

    def _init_tensors(self, out_local: int, in_local: int, bias: bool, *, dtype, device,
                      compute: str, fp8_max: float) -> None:
        self.compute = _check_compute(compute)
        self.fp8_max = float(fp8_max)
        # Never requires_grad: autograd has no float8 kernels, and this is inference.
        self.weight = nn.Parameter(
            torch.empty(out_local, in_local, dtype=FP8_DTYPE, device=device), requires_grad=False
        )
        self.register_buffer("weight_scale", torch.ones(1, dtype=torch.float32, device=device))
        self.register_buffer("input_scale", torch.ones(1, dtype=torch.float32, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_local, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def _matmul(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """``x @ W^T (+ bias)`` on this rank's shard, in ``x``'s dtype. The column layers pass
        their per-partition bias here (fused into the matmul, as ``F.linear`` does for the
        plain layers); the row layer adds its replicated bias after the all-reduce."""
        if self.compute == "dequant":
            # fp8 -> fp32 is exact; one rounding to the activation dtype after the scale.
            weight = (self.weight.float() * self.weight_scale).to(x.dtype)
            return F.linear(x, weight, bias)
        from difflet.backends.neuron.ops_impl import fp8_kernels

        out = fp8_kernels.fp8_linear(x, self.weight, self.weight_scale, self.input_scale,
                                     fp8_max=self.fp8_max)
        return out if bias is None else out + bias

    def _repr_tail(self) -> str:
        return f"compute={self.compute}, fp8_max={self.fp8_max:g}"


class FP8Linear(_FP8LinearBase):
    """``nn.Linear`` with an fp8 weight: the replicated (tp=1) case."""

    def __init__(self, in_features, out_features, bias=True, *, dtype=None, device=None,
                 compute: str, fp8_max: float = FP8_MAX) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self._init_tensors(self.out_features, self.in_features, bias, dtype=dtype, device=device,
                           compute=compute, fp8_max=fp8_max)
        self._difflet_shard = {"weight": None, "bias": None, "weight_scale": None,
                               "input_scale": None}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._matmul(x, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, {self._repr_tail()}"


class FP8ColumnParallelLinear(_FP8LinearBase):
    """``ColumnParallelLinear`` with an fp8 weight ``[out/tp, in]``; the scales are replicated."""

    @property
    def tensor_parallel_group(self):
        from difflet.backends.neuron.ops_impl.collectives import get_tp_group

        return get_tp_group()

    def __init__(self, input_size, output_size, bias=True, gather_output=True, dtype=None,
                 device=None, *, compute: str, fp8_max: float = FP8_MAX, **kwargs) -> None:
        super().__init__()
        if kwargs.get("skip_bias_add"):
            raise NotImplementedError(
                "FP8ColumnParallelLinear(skip_bias_add=True) is not supported on the neuron "
                "backend; no difflet model uses it"
            )
        _reject_sequence_parallel("FP8ColumnParallelLinear", kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.output_size_per_partition = _split_evenly(output_size, "output_size")
        self._init_tensors(self.output_size_per_partition, input_size, bias, dtype=dtype,
                           device=device, compute=compute, fp8_max=fp8_max)
        self._difflet_shard = {"weight": 0, "bias": 0, "weight_scale": None, "input_scale": None}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The bias is per-partition, so it folds straight into the local matmul.
        out = self._matmul(x, self.bias)
        if self.gather_output:
            out = gather_tp_dim(out, dim=-1)
        return out

    def extra_repr(self) -> str:
        return (f"in={self.input_size}, out={self.output_size}, "
                f"out_per_partition={self.output_size_per_partition}, "
                f"gather_output={self.gather_output}, {self._repr_tail()}")


class FP8RowParallelLinear(_FP8LinearBase):
    """``RowParallelLinear`` with an fp8 weight ``[out, in/tp]``; bias and scales replicated."""

    @property
    def tensor_parallel_group(self):
        from difflet.backends.neuron.ops_impl.collectives import get_tp_group

        return get_tp_group()

    def __init__(self, input_size, output_size, bias=True, input_is_parallel=False, dtype=None,
                 device=None, reduce_output=True, skip_bias_add=False, *, compute: str,
                 fp8_max: float = FP8_MAX, **kwargs) -> None:
        super().__init__()
        _reject_sequence_parallel("FP8RowParallelLinear", kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_output = reduce_output
        self.skip_bias_add = skip_bias_add
        self.input_size_per_partition = _split_evenly(input_size, "input_size")
        self._init_tensors(output_size, self.input_size_per_partition, bias, dtype=dtype,
                           device=device, compute=compute, fp8_max=fp8_max)
        self._difflet_shard = {"weight": 1, "bias": None, "weight_scale": None,
                               "input_scale": None}

    def forward(self, x: torch.Tensor):
        if not self.input_is_parallel:
            x = scatter_tp_dim(x, dim=-1)
        out = self._matmul(x)
        if self.reduce_output:
            out = reduce_tp(out)
        if self.skip_bias_add:
            return out, self.bias
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (f"in={self.input_size}, out={self.output_size}, "
                f"in_per_partition={self.input_size_per_partition}, "
                f"input_is_parallel={self.input_is_parallel}, "
                f"reduce_output={self.reduce_output}, skip_bias_add={self.skip_bias_add}, "
                f"{self._repr_tail()}")


# ------------------------------------------------------------------------------- the swap


def fp8_twin(linear: nn.Module, *, compute: str, fp8_max: float) -> _FP8LinearBase:
    """The fp8 layer that replaces ``linear`` (``nn.Linear``, ``ColumnParallelLinear`` or
    ``RowParallelLinear``), same geometry, flags, bias, dtype and device."""
    weight = linear.weight
    factory = {"dtype": weight.dtype if weight.dtype != FP8_DTYPE else torch.bfloat16,
               "device": weight.device}
    has_bias = getattr(linear, "bias", None) is not None
    if isinstance(linear, ColumnParallelLinear):
        return FP8ColumnParallelLinear(
            linear.input_size, linear.output_size, bias=has_bias,
            gather_output=linear.gather_output, compute=compute, fp8_max=fp8_max, **factory)
    if isinstance(linear, RowParallelLinear):
        return FP8RowParallelLinear(
            linear.input_size, linear.output_size, bias=has_bias,
            input_is_parallel=linear.input_is_parallel, reduce_output=linear.reduce_output,
            skip_bias_add=linear.skip_bias_add, compute=compute, fp8_max=fp8_max, **factory)
    if type(linear) is nn.Linear:
        return FP8Linear(linear.in_features, linear.out_features, bias=has_bias, compute=compute,
                         fp8_max=fp8_max, **factory)
    raise TypeError(f"cannot quantize a {type(linear).__name__}")


def quantize_module_(module: nn.Module, spec: QuantSpec) -> list[str]:
    """Replace every linear the ``spec`` targets with its fp8 twin, in place (on the meta
    module, before the checkpoint is loaded). Returns the replaced module names. A target that
    matches no linear is an error: the spec would silently quantize less than it claims."""
    targets = [
        (name, child) for name, child in module.named_modules()
        if spec.matches(name)
        and isinstance(child, (nn.Linear, ColumnParallelLinear, RowParallelLinear))
        and not isinstance(child, _FP8LinearBase)
    ]
    if not targets:
        raise ValueError("the quantization spec's target patterns match no linear in the module")
    for name, child in targets:
        parent_path, _, leaf = name.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        twin = fp8_twin(child, compute=spec.compute, fp8_max=spec.fp8_max)
        if isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(leaf)] = twin
        else:
            setattr(parent, leaf, twin)
    return [name for name, _ in targets]


def fp8_linears(module: nn.Module) -> list[str]:
    """Names of the fp8 linears in ``module``."""
    return [name for name, child in module.named_modules() if isinstance(child, _FP8LinearBase)]


def tp_size() -> int:
    return get_tp_size()


__all__ = [
    "FP8_DTYPE",
    "FP8ColumnParallelLinear",
    "FP8Linear",
    "FP8RowParallelLinear",
    "fp8_linears",
    "fp8_twin",
    "quantize_module_",
]
