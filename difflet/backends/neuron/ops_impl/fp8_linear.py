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
                      compute: str, fp8_max: float, mx: bool) -> None:
        self.compute = _check_compute(compute)
        self.fp8_max = float(fp8_max)
        self.mx = bool(mx)
        # Never requires_grad: autograd has no float8 kernels, and this is inference.
        self.weight = nn.Parameter(
            torch.empty(out_local, in_local, dtype=FP8_DTYPE, device=device), requires_grad=False
        )
        if mx:
            # E8M0 block scales, [out, in/32], sharded with the weight; activations are
            # block-scaled inside the kernel, so there is no input scale.
            if in_local % 32:
                raise ValueError(f"mxfp8 needs in_features a multiple of 32, got {in_local}")
            self.register_buffer("weight_scale",
                                 torch.zeros(out_local, in_local // 32, dtype=torch.uint8,
                                             device=device))
            self.register_buffer("input_scale", None)
        else:
            self.register_buffer("weight_scale", torch.ones(1, dtype=torch.float32, device=device))
            self.register_buffer("input_scale", torch.ones(1, dtype=torch.float32, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_local, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)
        # The kernel path's state, filled by prepare_kernel_ after the load (non-persistent).
        self.register_buffer("weight_packed", None, persistent=False)
        self.register_buffer("out_scale", None, persistent=False)
        self.register_buffer("weight_scales_hw", None, persistent=False)
        self.register_buffer("weight_bf16", None, persistent=False)
        self.kernel_n_chunk = self.kernel_k_full = self.kernel_k_rem = 0
        self.kernel_lnc = 1

    def _matmul(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """``x @ W^T (+ bias)`` on this rank's shard, in ``x``'s dtype. The column layers pass
        their per-partition bias here (fused into the matmul, as ``F.linear`` does for the
        plain layers); the row layer adds its replicated bias after the all-reduce."""
        if self.compute == "dequant":
            if self.weight_bf16 is not None:
                return F.linear(x, self.weight_bf16.to(x.dtype), bias)
            if self.mx:
                from difflet.quant import mx

                weight = mx.dequantize_mx(self.weight, self.weight_scale).to(x.dtype)
            else:
                # fp8 -> fp32 is exact; one rounding to the activation dtype after the scale.
                weight = (self.weight.float() * self.weight_scale).to(x.dtype)
            return F.linear(x, weight, bias)
        if self.mx:
            if self.weight_packed is None:
                raise RuntimeError("the mxfp8 kernel path needs the packed weight: call "
                                   "prepare_fp8_kernels_(module) after the load")
            from difflet.backends.neuron.ops_impl import fp8_kernels

            out = fp8_kernels.mx_linear(x, self.weight_packed, self.weight_scales_hw)
            return out if bias is None else out + bias
        if self.weight_packed is None:
            raise RuntimeError(
                "the fp8 kernel path needs the packed weight: call prepare_fp8_kernels_(module) "
                "after the load (TorchNeuronApplicationBase does, after load_with_store)")
        from difflet.backends.neuron.ops_impl import fp8_kernels

        out = fp8_kernels.fp8_linear(
            x, self.weight_packed, self.out_scale, self.input_scale, fp8_max=self.fp8_max,
            n_chunk=self.kernel_n_chunk, k_full=self.kernel_k_full, k_rem=self.kernel_k_rem,
            lnc=self.kernel_lnc)
        return out if bias is None else out + bias

    def prepare_kernel_(self, *, lnc: int | None = None) -> None:
        """After the load, for the ``nki`` path: pack the fp8 weight into the kernel's
        ``[K/4, N]`` int32 layout, precompute the output scale, and drop the unpacked weight
        (the two would double the layer's HBM). A no-op in dequant mode."""
        from difflet.backends.neuron.ops_impl import fp8_kernels

        if self.mx:
            if self.weight is None:
                return  # already prepared, or fused into FusedFP8Projections
            if self.compute == "nki":
                packed, scales = fp8_kernels.prepare_mx_weight(self.weight, self.weight_scale)
                self.register_buffer("weight_packed", packed, persistent=False)
                self.register_buffer("weight_scales_hw", scales, persistent=False)
            else:
                from difflet.quant import mx

                dequant = mx.dequantize_mx(self.weight, self.weight_scale).to(torch.bfloat16)
                self.register_buffer("weight_bf16", dequant, persistent=False)
            self.weight = None
            return
        if self.compute != "nki" or self.weight_packed is not None:
            return
        packed, out_scale, chunk, k_full, k_rem = fp8_kernels.prepare_weight(
            self.weight, self.weight_scale, self.input_scale)
        self.register_buffer("weight_packed", packed, persistent=False)
        self.register_buffer("out_scale", out_scale, persistent=False)
        self.kernel_n_chunk, self.kernel_k_full, self.kernel_k_rem = chunk, k_full, k_rem
        self.kernel_lnc = fp8_kernels.logical_cores() if lnc is None else int(lnc)
        self.weight = None  # the Parameter slot stays, empty: nothing reloads after this

    def _repr_tail(self) -> str:
        fmt = "mxfp8" if self.mx else f"fp8_max={self.fp8_max:g}"
        return f"compute={self.compute}, {fmt}"


class FP8Linear(_FP8LinearBase):
    """``nn.Linear`` with an fp8 weight: the replicated (tp=1) case."""

    def __init__(self, in_features, out_features, bias=True, *, dtype=None, device=None,
                 compute: str, fp8_max: float = FP8_MAX, mx: bool = False) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self._init_tensors(self.out_features, self.in_features, bias, dtype=dtype, device=device,
                           compute=compute, fp8_max=fp8_max, mx=mx)
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
                 device=None, *, compute: str, fp8_max: float = FP8_MAX, mx: bool = False,
                 **kwargs) -> None:
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
                           device=device, compute=compute, fp8_max=fp8_max, mx=mx)
        # The MX block scales follow the weight's rows; a per-tensor scale is replicated.
        self._difflet_shard = {"weight": 0, "bias": 0, "weight_scale": 0 if mx else None,
                               "input_scale": None}

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
                 fp8_max: float = FP8_MAX, mx: bool = False, **kwargs) -> None:
        super().__init__()
        _reject_sequence_parallel("FP8RowParallelLinear", kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_output = reduce_output
        self.skip_bias_add = skip_bias_add
        self.input_size_per_partition = _split_evenly(input_size, "input_size")
        self._init_tensors(output_size, self.input_size_per_partition, bias, dtype=dtype,
                           device=device, compute=compute, fp8_max=fp8_max, mx=mx)
        # The MX block scales follow the weight's K columns (the rank's own K range, quantized
        # as such, difflet/quant/mx.py); a per-tensor scale is replicated.
        self._difflet_shard = {"weight": 1, "bias": None, "weight_scale": 1 if mx else None,
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


def fp8_twin(linear: nn.Module, *, compute: str, fp8_max: float, mx: bool = False
             ) -> _FP8LinearBase:
    """The fp8 layer that replaces ``linear`` (``nn.Linear``, ``ColumnParallelLinear`` or
    ``RowParallelLinear``), same geometry, flags, bias, dtype and device."""
    weight = linear.weight
    factory = {"dtype": weight.dtype if weight.dtype != FP8_DTYPE else torch.bfloat16,
               "device": weight.device, "compute": compute, "fp8_max": fp8_max, "mx": mx}
    has_bias = getattr(linear, "bias", None) is not None
    if isinstance(linear, ColumnParallelLinear):
        return FP8ColumnParallelLinear(
            linear.input_size, linear.output_size, bias=has_bias,
            gather_output=linear.gather_output, **factory)
    if isinstance(linear, RowParallelLinear):
        return FP8RowParallelLinear(
            linear.input_size, linear.output_size, bias=has_bias,
            input_is_parallel=linear.input_is_parallel, reduce_output=linear.reduce_output,
            skip_bias_add=linear.skip_bias_add, **factory)
    if type(linear) is nn.Linear:
        return FP8Linear(linear.in_features, linear.out_features, bias=has_bias, **factory)
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
        twin = fp8_twin(child, compute=spec.compute, fp8_max=spec.fp8_max, mx=spec.is_mx)
        if isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(leaf)] = twin
        else:
            setattr(parent, leaf, twin)
    return [name for name, _ in targets]


def fp8_linears(module: nn.Module) -> list[str]:
    """Names of the fp8 linears in ``module``."""
    return [name for name, child in module.named_modules() if isinstance(child, _FP8LinearBase)]


class FusedFP8Projections(nn.Module):
    """Several mxfp8 projections of one input (attention's q, k and v) as one kernel call.

    The MX kernel quantizes its activation once per call, which at the attention projections'
    small per-rank N (768 at tp4) costs as much as the matmul; three calls on the same input
    pay it three times. Concatenating the three weights along N gives one call whose output is
    split back. Built after the load from the three prepared layers' fp8 weights and scales
    (``prepare_fp8_kernels_``), which then drop theirs; the attention processor calls
    ``attn.fused_qkv`` / ``attn.fused_add_qkv`` when they exist.
    """

    def __init__(self, layers: list[_FP8LinearBase]) -> None:
        super().__init__()
        from difflet.backends.neuron.ops_impl import fp8_kernels

        if not layers or any(not (l.mx and l.compute == "nki") for l in layers):
            raise ValueError("fused projections need mxfp8 layers on the kernel path")
        if any(l.weight is None for l in layers):
            raise ValueError("fuse before the layers drop their weights (prepare_kernel_)")
        if len({int(l.weight.shape[1]) for l in layers}) != 1:
            raise ValueError("fused projections must share the input size")
        biases = [l.bias for l in layers]
        if any(b is None for b in biases) and not all(b is None for b in biases):
            raise ValueError("fused projections must all have a bias or none")
        weight = torch.cat([l.weight for l in layers], dim=0)
        scale = torch.cat([l.weight_scale for l in layers], dim=0)
        packed, scales = fp8_kernels.prepare_mx_weight(weight, scale)
        self.register_buffer("weight_packed", packed, persistent=False)
        self.register_buffer("weight_scales_hw", scales, persistent=False)
        self.register_buffer("bias", None if biases[0] is None else torch.cat(biases, dim=0).detach(),
                             persistent=False)
        self.splits = [int(l.weight.shape[0]) for l in layers]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        from difflet.backends.neuron.ops_impl import fp8_kernels

        out = fp8_kernels.mx_linear(x, self.weight_packed, self.weight_scales_hw)
        if self.bias is not None:
            out = out + self.bias
        return tuple(out.split(self.splits, dim=-1))

    def extra_repr(self) -> str:
        return f"splits={self.splits}, mxfp8 nki"


#: Attention projections that consume one input: the attribute the fused layer gets, and the
#: layers it fuses (``difflet.models.qwen_image.modeling_qwen_image`` reads the attributes).
FUSED_PROJECTION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fused_qkv", ("to_q", "to_k", "to_v")),
    ("fused_add_qkv", ("add_q_proj", "add_k_proj", "add_v_proj")),
)


def fuse_projections_(module: nn.Module) -> list[str]:
    """Give every submodule whose ``to_q``/``to_k``/``to_v`` (and ``add_*_proj``) are mxfp8
    kernel-path layers with their weights still loaded a :class:`FusedFP8Projections`; the
    three layers then drop their weights. Returns the fused attributes' names. Only layers
    whose output is not gathered (``gather_output=False`` or plain) fuse: a gathered output
    would have to be gathered per split."""
    fused: list[str] = []
    for name, child in list(module.named_modules()):
        for attr, members in FUSED_PROJECTION_GROUPS:
            layers = [getattr(child, m, None) for m in members]
            if not all(isinstance(l, _FP8LinearBase) and l.mx and l.compute == "nki"
                       and l.weight is not None for l in layers):
                continue
            if any(getattr(l, "gather_output", False) for l in layers):
                continue
            child.add_module(attr, FusedFP8Projections(layers))
            for l in layers:
                l.weight = None
                l.weight_scale = None
            fused.append(f"{name}.{attr}" if name else attr)
    return fused


def prepare_fp8_kernels_(module: nn.Module, *, lnc: int | None = None, fuse: bool = False) -> int:
    """``prepare_kernel_`` on every fp8 linear of ``module`` and tie the input scales of
    siblings that quantize the same activation; returns how many layers were packed.

    The quantize-and-pack of the activation is the kernel path's fixed cost (~0.13 ms on a
    trn3 core at 5120 x 3072, as much as a small projection's matmul). Attention's q, k and v
    projections (and the text stream's add_q/k/v) consume one tensor, so their calibrated
    scales are equal; when the three modules hold the *same* ``input_scale`` tensor object the
    three quantize subgraphs are identical and the compiler folds them into one (measured:
    0.555 ms -> 0.295 ms for the three 3072x768 projections, against 0.565 ms in bf16).
    Tying by equal value under one parent changes no number: a layer whose input differs
    keeps its own quantize op either way.
    """
    count = 0
    if fuse:
        # Only for a model whose attention calls ``fused_qkv`` / ``fused_add_qkv`` when present
        # (Qwen-Image's processor): the fused layers' three members drop their weights.
        fuse_projections_(module)
    by_parent: dict[str, list[_FP8LinearBase]] = {}
    for name, child in module.named_modules():
        if isinstance(child, _FP8LinearBase) and (child.compute == "nki" or child.mx):
            by_parent.setdefault(name.rpartition(".")[0], []).append(child)
    for siblings in by_parent.values():
        for child in siblings:
            if child.mx:
                continue
            for other in siblings:
                if other is child:
                    break
                if (not other.mx and other.input_scale is not child.input_scale
                        and torch.equal(other.input_scale, child.input_scale)):
                    child.input_scale = other.input_scale  # the same tensor object
                    break
    for siblings in by_parent.values():
        for child in siblings:
            child.prepare_kernel_(lnc=lnc)
            count += 1
    return count


def tp_size() -> int:
    return get_tp_size()


__all__ = [
    "FP8_DTYPE",
    "FP8ColumnParallelLinear",
    "FP8Linear",
    "FP8RowParallelLinear",
    "FUSED_PROJECTION_GROUPS",
    "FusedFP8Projections",
    "fuse_projections_",
    "fp8_linears",
    "fp8_twin",
    "prepare_fp8_kernels_",
    "quantize_module_",
]
