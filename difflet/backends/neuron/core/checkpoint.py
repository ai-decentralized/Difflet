"""Lazy per-rank safetensors loading onto the neuron device (TorchNeuron, non-AoT).

The module is built on ``meta`` (no memory anywhere), each meta parameter gets
device storage in the target dtype, and then every rank reads only its own slice
of each tensor off disk, casts it on the host and copies it into place. Host
memory peaks at about one full tensor's mapped pages plus one shard in flight,
never the checkpoint: four ranks each materialising a 38 GiB transformer would
need ~152 GiB of host RAM (``difflet/backends/tpu/core/checkpoint.py:3-12``).

Slicing, key mapping and meta materialisation are the TPU backend's, reused by
import rather than copied: those modules depend only on torch and safetensors,
and the split axis comes from each layer's ``_difflet_shard`` declaration, so a
layer and its loader cannot disagree.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import re

import torch
import torch.nn as nn

from difflet.backends.neuron.ops_impl.collectives import get_tp_rank, get_tp_size
from difflet.backends.tpu.core.checkpoint import build_weight_map, load_checkpoint_into
from difflet.backends.tpu.core.weights import (
    load_sharded_state_dict,
    materialize_meta_,
    narrow_to_rank,
    shard_dim,
    shard_state_dict,
)

__all__ = [
    "CheckpointSlice",
    "build_on_meta",
    "build_weight_map",
    "load_checkpoint_into",
    "load_sharded_checkpoint",
    "load_sharded_state_dict",
    "materialize_meta_",
    "narrow_to_rank",
    "shard_dim",
    "shard_state_dict",
    "single_block_proj_out_key",
    "single_block_proj_out_slices",
]


def build_on_meta(factory: Callable[[], nn.Module]) -> nn.Module:
    """Run ``factory()`` with every parameter created on ``meta``; buffers stay real.

    ``include_buffers=False`` matters: buffers computed in ``__init__`` (rotary
    tables, frequency grids) cannot be computed on meta, and non-persistent ones
    are never in a checkpoint, so they must stay real on the host
    (``difflet/backends/tpu/core/weights.py:130-136``). Parallel layers read the
    tp size in ``__init__``, so initialise the parallel mesh before calling this.
    """
    try:
        from accelerate import init_empty_weights
    except ImportError as exc:  # pragma: no cover - pinned in requirements-neuron-native.lock
        raise ImportError(
            "build_on_meta needs accelerate; install requirements-neuron-native.lock"
        ) from exc
    with init_empty_weights(include_buffers=False):
        return factory()


@dataclass(frozen=True)
class CheckpointSlice:
    """A parameter that is the range ``[start, start + length)`` of checkpoint tensor ``key``
    along ``dim``: e.g. Flux's single-stream ``proj_out_attn.weight`` is the first ``inner_dim``
    input columns of the fused ``proj_out.weight``. Each rank reads only its TP shard of that
    range, along the parameter's own declared shard axis."""

    key: str
    dim: int
    start: int
    length: int


def single_block_proj_out_slices(num_single_layers: int, inner_dim: int,
                                 mlp_dim: int) -> dict[str, CheckpointSlice]:
    """The two halves of each single-stream block's fused output projection.

    diffusers fuses a single-stream block's attention and MLP output projections into one
    ``proj_out`` (input columns ``[0, inner_dim)`` attention, ``[inner_dim, inner_dim +
    mlp_dim)`` MLP); Flux's and HunyuanVideo's ports split them so each is row-parallel on its
    own input sharding. ``proj_out_attn.weight`` and ``proj_out_mlp.weight`` are therefore
    column ranges of ``single_transformer_blocks.{i}.proj_out.weight``; the bias is
    ``single_block_proj_out_key``'s rename."""
    slices = {}
    for i in range(int(num_single_layers)):
        block = f"single_transformer_blocks.{i}"
        source = f"{block}.proj_out.weight"
        slices[f"{block}.proj_out_attn.weight"] = CheckpointSlice(source, 1, 0, int(inner_dim))
        slices[f"{block}.proj_out_mlp.weight"] = CheckpointSlice(source, 1, int(inner_dim),
                                                                 int(mlp_dim))
    return slices


def single_block_proj_out_key(name: str) -> str:
    """Checkpoint key of module parameter ``name`` under the split single-stream ``proj_out``:
    ``proj_out_attn.bias`` is the fused ``proj_out.bias`` (``proj_out_mlp`` has none); every
    other name is unchanged."""
    return re.sub(r"\.proj_out_attn\.bias$", ".proj_out.bias", name)


def _read_checkpoint_slice(path: str, spec: CheckpointSlice, *, shard: int | None, tp_size: int,
                           tp_rank: int) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        view = handle.get_slice(spec.key)
        shape = view.get_shape()
        selector = [slice(None)] * len(shape)
        axis = spec.dim % len(shape)
        if spec.start + spec.length > shape[axis]:
            raise ValueError(f"{spec}: range exceeds dim {axis} of size {shape[axis]}")
        lo, hi = spec.start, spec.start + spec.length
        if shard is not None and tp_size > 1:
            shard_axis = shard % len(shape)
            size = spec.length if shard_axis == axis else shape[shard_axis]
            if size % tp_size:
                raise ValueError(f"{spec}: dim {shard_axis} of size {size} is not divisible by tp={tp_size}")
            width = size // tp_size
            if shard_axis == axis:  # this rank's shard of the range
                lo, hi = lo + tp_rank * width, lo + (tp_rank + 1) * width
            else:
                selector[shard_axis] = slice(tp_rank * width, (tp_rank + 1) * width)
        selector[axis] = slice(lo, hi)
        return view[tuple(selector)]


def load_sharded_checkpoint(
    module: nn.Module,
    model_dir: str | Path,
    *,
    device: str | torch.device = "neuron",
    dtype: torch.dtype = torch.bfloat16,
    tp_size: int | None = None,
    tp_rank: int | None = None,
    rename: Callable[[str], str] | None = None,
    prefix: str = "",
    strict: bool = True,
    slices: Mapping[str, CheckpointSlice] | None = None,
    on_tensor: Callable[[str, torch.Tensor], None] | None = None,
) -> dict[str, list[str]]:
    """Load this rank's shards of the safetensors checkpoint in ``model_dir`` onto ``device``.

    Parameters still on ``meta`` get storage on ``device`` in ``dtype``; tensors
    that already hold storage keep their dtype. Each checkpoint entry is sliced on
    the host along the owning layer's declared shard dim, cast on the host to its
    target's dtype (fp32 -> bf16 halves the bytes sent to the device) and copied
    into place. Host buffers are moved to ``device`` last.

    ``tp_size``/``tp_rank`` default to the neuron parallel mesh. ``rename`` maps a
    module name (after stripping ``prefix``) to its checkpoint key. ``slices`` maps a module name
    to a :class:`CheckpointSlice` for parameters that are a range of a fused checkpoint tensor;
    they are read after the renamable ones, and their source keys do not count as unexpected.
    Returns ``{"missing": [...], "unexpected": [...]}``; with ``strict`` a parameter or
    persistent buffer missing from the checkpoint raises ``ValueError``.
    ``on_tensor(name, host_tensor)`` sees every tensor after its copy (the weight store,
    ``core/weight_store.py``, writes them).
    """
    tp_size = get_tp_size() if tp_size is None else int(tp_size)
    if tp_rank is None:
        tp_rank = get_tp_rank() if tp_size > 1 else 0
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank {tp_rank} is out of range for tp_size {tp_size}")
    stranded = _meta_computed_buffers(module)
    if stranded:
        raise ValueError(
            "non-persistent buffers on meta cannot be loaded from a checkpoint: "
            f"{stranded[:5]}; build the module with build_on_meta(), which keeps buffers real"
        )
    target = _resolve_device(device)
    materialize_meta_(module, device=target, dtype=dtype)
    # dtype=None: each entry is cast to its target's dtype -- ``dtype`` for the
    # parameters materialised above, the stored dtype for real buffers, so an fp32
    # buffer is not rounded through bf16 (tpu/core/checkpoint.py:142).
    slices = dict(slices or {})
    report = load_checkpoint_into(
        module,
        model_dir,
        tp_size=tp_size,
        tp_rank=tp_rank,
        prefix=prefix,
        dtype=None,
        strict=strict and not slices,  # sliced parameters show up as missing until read below
        rename=rename,
        on_tensor=on_tensor,
    )
    if slices:
        from difflet.backends.tpu.core.checkpoint import checkpoint_keys

        # The same keys load_checkpoint_into chose its index by, so both read one index.
        keys = checkpoint_keys(module, prefix=prefix, rename=rename)
        report = _load_slices(module, model_dir, slices, report, tp_size=tp_size, tp_rank=tp_rank,
                              strict=strict, on_tensor=on_tensor, keys=keys)
    module.to(target)
    return report


def _load_slices(module, model_dir, slices, report, *, tp_size, tp_rank, strict,
                 on_tensor=None, keys=None):
    from difflet.backends.tpu.core.checkpoint import build_weight_map
    from difflet.backends.tpu.core.weights import shard_dim

    weight_map = build_weight_map(model_dir, keys=keys)
    params = dict(module.named_parameters())
    with torch.no_grad():
        for name, spec in slices.items():
            if name not in params:
                raise ValueError(f"sliced parameter {name!r} is not a parameter of the module")
            if spec.key not in weight_map:
                raise ValueError(f"{name}: checkpoint has no {spec.key!r}")
            target = params[name]
            tensor = _read_checkpoint_slice(weight_map[spec.key], spec, shard=shard_dim(module, name),
                                            tp_size=tp_size, tp_rank=tp_rank)
            if tuple(tensor.shape) != tuple(target.shape):
                raise ValueError(f"{name}: checkpoint slice {tuple(tensor.shape)} != parameter "
                                 f"{tuple(target.shape)} ({spec}, tp={tp_size})")
            tensor = tensor.to(target.dtype)
            target.copy_(tensor)
            if on_tensor is not None:  # the weight store records sliced parameters too
                on_tensor(name, tensor)
    missing = [n for n in report["missing"] if n not in slices]
    sources = {spec.key for spec in slices.values()}
    unexpected = [k for k in report["unexpected"] if k not in sources]
    if strict and missing:
        raise ValueError(f"{len(missing)} parameters had no checkpoint entry: {missing[:5]}")
    return {"missing": missing, "unexpected": unexpected}


def _meta_computed_buffers(module: nn.Module) -> list[str]:
    """Non-persistent buffers left on meta: no checkpoint can ever fill them."""
    persistent = set(module.state_dict().keys())
    return sorted(
        name for name, buf in module.named_buffers() if buf.is_meta and name not in persistent
    )


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device).split(":", 1)[0] == "neuron":
        import torch_neuronx  # noqa: F401  registers the "neuron" device type
    return torch.device(device)
