"""MXFP8 (W8A8, OCP microscaling) linears for the neuron backend.

The quantized twins of ``nn.Linear`` and of this backend's ``ColumnParallelLinear`` /
``RowParallelLinear`` (``linear.py``): the same constructor arguments, sharding, collectives and
bias semantics, with the weight stored as ``float8_e4m3fn`` beside its uint8 E8M0 block scales
``[out, in/32]`` (``difflet/quant/mx.py``). The quantized checkpoint
(``difflet/quant/checkpoint.py``) holds exactly these two tensors per linear under the diffusers
key names, so the lazy per-rank loader fills them by name. Activations are block-scaled inside
the kernel: there is no input scale and no calibration.

Two compute paths, chosen by ``compute``:

* ``"dequant"``: the weight is dequantized to bf16 once after the load and the matmul is bf16:
  a memory-only mode (half the weight bytes in the checkpoint), the path the plumbing is proven
  with and the fallback for debugging.
* ``"nki"``: nkilib's ``matmul_mxfp8`` kernel on the tensor engine's MX path
  (``fp8_kernels.py``): the bf16 activation goes in, is block-quantized in the kernel, and the
  fp8 x fp8 matmul accumulates in fp32, bf16 out.

Decision D42 (``core/config_checks.require_bf16_on_device``) is untouched: activations between
layers and the row-parallel all-reduce stay bf16; fp8 exists only inside the matmul.

Why the fp8 weight is a ``Parameter`` and the scales are buffers: the module is built on
``meta`` and ``materialize_meta_`` (``_torch_megatron/weights.py``) gives every meta parameter
storage in the application dtype, except float8 parameters, which keep theirs; a meta buffer
would keep its dtype but ``build_on_meta`` makes buffers real anyway (they are small).
"""

from __future__ import annotations

from collections.abc import Callable

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
from difflet.quant.mx import BLOCK, FP8_DTYPE, LAYOUT_K
from difflet.quant.spec import COMPUTE_MODES, QuantSpec


def _check_compute(compute: str) -> str:
    if compute not in COMPUTE_MODES:
        raise ValueError(f"fp8 compute must be one of {COMPUTE_MODES}, got {compute!r}")
    return compute


class _FP8LinearBase(nn.Module):
    """The fp8 weight, its block scales, the bias, and the two compute paths; subclasses add
    the sharding and the collectives."""

    def _init_tensors(self, out_local: int, in_local: int, bias: bool, *, dtype, device,
                      compute: str, split: str = "") -> None:
        self.compute = _check_compute(compute)
        # The per-rank K, checked at this process's tp (the launch's inside a rank; 1 on a bare
        # host, where a row layer's K is whole). ``split`` is how a row layer's K got here, so
        # the refusal names the full K and the tp, not the per-rank number alone.
        if in_local % LAYOUT_K:
            raise ValueError(f"mxfp8 needs in_features a multiple of {LAYOUT_K} (the MX layout), "
                             f"got {in_local}" + (f" ({split})" if split else ""))
        # Never requires_grad: autograd has no float8 kernels, and this is inference.
        self.weight = nn.Parameter(
            torch.empty(out_local, in_local, dtype=FP8_DTYPE, device=device), requires_grad=False
        )
        # E8M0 block scales, [out, in/32], sharded with the weight.
        self.register_buffer("weight_scale",
                             torch.zeros(out_local, in_local // BLOCK, dtype=torch.uint8,
                                         device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_local, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)
        # Filled by prepare_kernel_ after the load (non-persistent): the kernel's layout, or
        # the dequantized bf16 weight.
        self.register_buffer("weight_packed", None, persistent=False)
        self.register_buffer("weight_scales_hw", None, persistent=False)
        self.register_buffer("weight_bf16", None, persistent=False)

    def _matmul(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """``x @ W^T (+ bias)`` on this rank's shard, in ``x``'s dtype. The column layers pass
        their per-partition bias here (fused into the matmul, as ``F.linear`` does for the
        plain layers); the row layer adds its replicated bias after the all-reduce."""
        if self.compute == "dequant":
            if self.weight_bf16 is not None:
                return F.linear(x, self.weight_bf16.to(x.dtype), bias)
            from difflet.quant import mx

            return F.linear(x, mx.dequantize_mx(self.weight, self.weight_scale).to(x.dtype), bias)
        if self.weight_packed is None:
            raise RuntimeError("the mxfp8 kernel path needs the packed weight: call "
                               "prepare_fp8_kernels_(module) after the load")
        from difflet.backends.neuron.ops_impl import fp8_kernels

        out = fp8_kernels.mx_linear(x, self.weight_packed, self.weight_scales_hw)
        return out if bias is None else out + bias

    def prepare_kernel_(self) -> None:
        """After the load: on the ``nki`` path lay the weight out as the kernel reads it, on the
        ``dequant`` path dequantize it once to bf16 (a block-scale dequantization traced into
        the block graph ran at 6.9 s per block); either way drop the fp8 weight (the two would
        double the layer's HBM). A no-op once done, or for a layer fused into
        :class:`FusedFP8Projections`."""
        if self.weight is None:
            return
        if self.compute == "nki":
            from difflet.backends.neuron.ops_impl import fp8_kernels

            packed, scales = fp8_kernels.prepare_mx_weight(self.weight, self.weight_scale)
            self.register_buffer("weight_packed", packed, persistent=False)
            self.register_buffer("weight_scales_hw", scales, persistent=False)
        else:
            from difflet.quant import mx

            dequant = mx.dequantize_mx(self.weight, self.weight_scale).to(torch.bfloat16)
            self.register_buffer("weight_bf16", dequant, persistent=False)
        self.weight = None  # the Parameter slot stays, empty: nothing reloads after this

    def _repr_tail(self) -> str:
        return f"compute={self.compute}, mxfp8"


class FP8Linear(_FP8LinearBase):
    """``nn.Linear`` with an mxfp8 weight: the replicated (tp=1) case."""

    def __init__(self, in_features, out_features, bias=True, *, dtype=None, device=None,
                 compute: str) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self._init_tensors(self.out_features, self.in_features, bias, dtype=dtype, device=device,
                           compute=compute)
        self._difflet_shard = {"weight": None, "bias": None, "weight_scale": None}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._matmul(x, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, {self._repr_tail()}"


class FP8ColumnParallelLinear(_FP8LinearBase):
    """``ColumnParallelLinear`` with an mxfp8 weight ``[out/tp, in]``; the block scales follow
    the weight's rows."""

    @property
    def tensor_parallel_group(self):
        from difflet.backends.neuron.ops_impl.collectives import get_tp_group

        return get_tp_group()

    def __init__(self, input_size, output_size, bias=True, gather_output=True, dtype=None,
                 device=None, *, compute: str, **kwargs) -> None:
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
                           device=device, compute=compute)
        self._difflet_shard = {"weight": 0, "bias": 0, "weight_scale": 0}

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
    """``RowParallelLinear`` with an mxfp8 weight ``[out, in/tp]``: the block scales follow the
    weight's K columns (the rank's own K range, quantized as such, ``difflet/quant/mx.py``);
    the bias is replicated."""

    @property
    def tensor_parallel_group(self):
        from difflet.backends.neuron.ops_impl.collectives import get_tp_group

        return get_tp_group()

    def __init__(self, input_size, output_size, bias=True, input_is_parallel=False, dtype=None,
                 device=None, reduce_output=True, skip_bias_add=False, *, compute: str,
                 **kwargs) -> None:
        super().__init__()
        _reject_sequence_parallel("FP8RowParallelLinear", kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_output = reduce_output
        self.skip_bias_add = skip_bias_add
        self.input_size_per_partition = _split_evenly(input_size, "input_size")
        self._init_tensors(output_size, self.input_size_per_partition, bias, dtype=dtype,
                           device=device, compute=compute,
                           split=f"{input_size} split over tp={get_tp_size()}")
        self._difflet_shard = {"weight": 1, "bias": None, "weight_scale": 1}

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


def _parallel_kind(linear: nn.Module) -> str | None:
    """``"column"`` / ``"row"`` for this backend's parallel linears, and for another backend's
    classes of the same name: a model module binds ``difflet.ops.ColumnParallelLinear`` when it
    is first imported, from whichever backend was selected then (in a CPU-only process that is
    the CPU backend's, a plain ``nn.Linear`` subclass at tp=1), so the class identity alone
    does not say which layer it is."""
    if isinstance(linear, ColumnParallelLinear):
        return "column"
    if isinstance(linear, RowParallelLinear):
        return "row"
    name = type(linear).__name__
    if isinstance(linear, nn.Linear) and name == "ColumnParallelLinear":
        return "column"
    if isinstance(linear, nn.Linear) and name == "RowParallelLinear":
        return "row"
    return None


def fp8_twin(linear: nn.Module, *, compute: str) -> _FP8LinearBase:
    """The mxfp8 layer that replaces ``linear`` (``nn.Linear``, ``ColumnParallelLinear`` or
    ``RowParallelLinear``), same geometry, flags, bias, dtype and device. A parallel linear of
    another backend (``_parallel_kind``) keeps the flags it carries and the defaults of the
    ones it does not (the CPU backend's drop them: it is tp=1)."""
    weight = linear.weight
    factory = {"dtype": weight.dtype if weight.dtype != FP8_DTYPE else torch.bfloat16,
               "device": weight.device, "compute": compute}
    has_bias = getattr(linear, "bias", None) is not None
    kind = _parallel_kind(linear)
    if kind == "column":
        return FP8ColumnParallelLinear(
            getattr(linear, "input_size", getattr(linear, "in_features", None)),
            getattr(linear, "output_size", getattr(linear, "out_features", None)), bias=has_bias,
            gather_output=getattr(linear, "gather_output", True), **factory)
    if kind == "row":
        return FP8RowParallelLinear(
            getattr(linear, "input_size", getattr(linear, "in_features", None)),
            getattr(linear, "output_size", getattr(linear, "out_features", None)), bias=has_bias,
            input_is_parallel=getattr(linear, "input_is_parallel", False),
            reduce_output=getattr(linear, "reduce_output", True),
            skip_bias_add=getattr(linear, "skip_bias_add", False), **factory)
    if type(linear) is nn.Linear:
        return FP8Linear(linear.in_features, linear.out_features, bias=has_bias, **factory)
    raise TypeError(f"cannot quantize a {type(linear).__name__}")


def quantize_module_(module: nn.Module, spec: QuantSpec, *,
                     key_fn: Callable[[str], str] | None = None) -> list[str]:
    """Replace every linear the ``spec`` targets with its mxfp8 twin, in place (on the meta
    module, before the checkpoint is loaded). Returns the replaced module names. A target that
    matches no linear is an error: the spec would silently quantize less than it claims.

    ``key_fn`` maps a module name to its checkpoint key before the spec is consulted: target
    patterns are written against the checkpoint's key names (the quantized checkpoint writer
    matches those), and a model whose module names differ from its keys (Wan's ``ffn.net_in``
    is ``ffn.net.0.proj`` on disk, ``wan/transformer.checkpoint_key``) would otherwise quantize
    a different set of layers on the device than in the checkpoint. The returned names stay
    module names: they address this module.

    A twin that refuses its geometry (``_init_tensors``: a per-rank K with no MX layout) is
    re-raised with the layer's name, since the bare message cannot say which of a model's
    hundreds of linears it is about; the usual cause is a row-parallel layer whose whole K
    divides by 128 but whose per-rank K at this tp does not."""
    key = key_fn if key_fn is not None else (lambda name: name)
    targets = [
        (name, child) for name, child in module.named_modules()
        if spec.matches(key(name))
        and isinstance(child, (nn.Linear, ColumnParallelLinear, RowParallelLinear))
        and not isinstance(child, _FP8LinearBase)
    ]
    if not targets:
        raise ValueError("the quantization spec's target patterns match no linear in the module")
    for name, child in targets:
        parent_path, _, leaf = name.rpartition(".")
        parent = module.get_submodule(parent_path) if parent_path else module
        try:
            twin = fp8_twin(child, compute=spec.compute)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc} (tp too high for this layer at this spec?)") from exc
        if isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(leaf)] = twin
        else:
            setattr(parent, leaf, twin)
    return [name for name, _ in targets]


def fp8_linears(module: nn.Module) -> list[str]:
    """Names of the mxfp8 linears in ``module``."""
    return [name for name, child in module.named_modules() if isinstance(child, _FP8LinearBase)]


class FusedFP8Projections(nn.Module):
    """Several mxfp8 projections of one input (attention's q, k and v) as one kernel call.

    The MX kernel quantizes its activation once per call; concatenating the three weights
    along N gives one call whose output is split back. Built after the load from the three
    prepared layers' fp8 weights and scales (``prepare_fp8_kernels_``), which then drop theirs;
    the attention processor calls ``attn.fused_qkv`` / ``attn.fused_add_qkv`` when they exist.
    Measured step-neutral on Qwen-Image at tp4 (the kernel's cost is its load path, not the
    quantize), kept because it is correct and costs nothing.
    """

    def __init__(self, layers: list[_FP8LinearBase]) -> None:
        super().__init__()
        from difflet.backends.neuron.ops_impl import fp8_kernels

        if not layers or any(l.compute != "nki" for l in layers):
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
            if not all(isinstance(l, _FP8LinearBase) and l.compute == "nki"
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


def prepare_fp8_kernels_(module: nn.Module, *, fuse: bool = False) -> int:
    """``prepare_kernel_`` on every mxfp8 linear of ``module`` (after the load, before the
    first forward); returns how many layers were prepared. ``fuse`` first fuses the attention
    q/k/v groups (only for a model whose attention calls ``fused_qkv`` / ``fused_add_qkv``
    when present, Qwen-Image's processor: the fused layers' three members drop their
    weights)."""
    if fuse:
        fuse_projections_(module)
    count = 0
    for child in module.modules():
        if isinstance(child, _FP8LinearBase):
            child.prepare_kernel_()
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
