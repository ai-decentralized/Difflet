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

from collections.abc import Callable
from pathlib import Path

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
    "build_on_meta",
    "build_weight_map",
    "load_checkpoint_into",
    "load_sharded_checkpoint",
    "load_sharded_state_dict",
    "materialize_meta_",
    "narrow_to_rank",
    "shard_dim",
    "shard_state_dict",
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
) -> dict[str, list[str]]:
    """Load this rank's shards of the safetensors checkpoint in ``model_dir`` onto ``device``.

    Parameters still on ``meta`` get storage on ``device`` in ``dtype``; tensors
    that already hold storage keep their dtype. Each checkpoint entry is sliced on
    the host along the owning layer's declared shard dim, cast on the host to its
    target's dtype (fp32 -> bf16 halves the bytes sent to the device) and copied
    into place. Host buffers are moved to ``device`` last.

    ``tp_size``/``tp_rank`` default to the neuron parallel mesh. ``rename`` maps a
    module name (after stripping ``prefix``) to its checkpoint key. Returns
    ``{"missing": [...], "unexpected": [...]}``; with ``strict`` a parameter or
    persistent buffer missing from the checkpoint raises ``ValueError``.
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
    report = load_checkpoint_into(
        module,
        model_dir,
        tp_size=tp_size,
        tp_rank=tp_rank,
        prefix=prefix,
        dtype=None,
        strict=strict,
        rename=rename,
    )
    module.to(target)
    return report


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
