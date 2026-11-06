"""Load a safetensors checkpoint straight into this rank's shards.

``weights.shard_state_dict`` splits an already-materialized state dict. That
is fine for a toy module and impossible for a real one: Qwen-Image's
transformer is 38 GiB, and four ranks each materializing the full tensor set
would need ~152 GiB of host RAM before a single shard is taken.

So this reads *lazily*. ``safetensors``' slice API fetches only the requested
sub-range off disk, so each rank pays roughly ``38 GiB / tp`` instead of the
whole file. The split axis still comes from the layer's own ``_difflet_shard``
declaration on the owning module, exactly as in ``weights.py`` — one source of
truth for how a parameter is partitioned.

Phase 4 of docs/plans/2026-08-16-tpu-backend-support.md.
"""

# Lifted into difflet.backends._torch_megatron (shared with CUDA); re-exported here so the TPU
# and neuron backends and their tests keep their import paths.
from difflet.backends._torch_megatron.checkpoint import (  # noqa: F401
    INDEX_SUFFIX,
    build_weight_map,
    checkpoint_keys,
    load_checkpoint_into,
)

# --- TPU only: one checkpoint tensor feeding several parameters ----------------
#
# The shared loader above is what the neuron backend imports through this module,
# so it stays as is; the windowed variant below is a separate entry point that only
# the TPU FLUX and HunyuanVideo transformers call.

import logging  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Callable  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from difflet.backends._torch_megatron.checkpoint import _load_targets  # noqa: E402
from difflet.backends._torch_megatron.weights import shard_dim  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointSlice:
    """A parameter that comes from a *window* of one checkpoint tensor.

    ``rename`` may return this instead of a bare key when the modeling splits
    an upstream tensor into several parameters — HunyuanVideo's single-stream
    ``proj_out`` becomes ``proj_out_attn`` (columns ``[:inner_dim]``) and
    ``proj_out_mlp`` (columns ``[inner_dim:]``) so each half can be
    row-parallel over its own input sharding. The window is applied before
    the rank shard, and both stay lazy: only the rank's slice of the window is
    read off disk.
    """

    key: str
    dim: int
    start: int
    stop: int | None = None  # None: to the end of the axis


def _read_window_shard(path: str, key: str, *, window: CheckpointSlice | None,
                       dim: int | None, tp_size: int, tp_rank: int):
    """Read one tensor: ``window`` first (if any), then this rank's slice along ``dim``."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        if window is None and (dim is None or tp_size == 1):
            return handle.get_tensor(key)
        view = handle.get_slice(key)
        shape = list(view.get_shape())
        selector = [slice(None)] * len(shape)
        if window is not None:
            w_axis = window.dim % len(shape)
            w_hi = shape[w_axis] if window.stop is None else window.stop
            selector[w_axis] = slice(window.start, w_hi)
            shape[w_axis] = w_hi - window.start
        if dim is not None and tp_size > 1:
            axis = dim % len(shape)
            size = shape[axis]
            if size % tp_size != 0:
                raise ValueError(
                    f"{key}: dim {axis} of size {size} is not divisible by tp={tp_size}"
                )
            width = size // tp_size
            base = selector[axis].start or 0
            selector[axis] = slice(base + tp_rank * width, base + (tp_rank + 1) * width)
        return view[tuple(selector)]


def load_windowed_checkpoint_into(
    module: nn.Module,
    model_dir,
    *,
    tp_size: int,
    tp_rank: int,
    prefix: str = "",
    dtype: torch.dtype | None = None,
    strict: bool = True,
    rename: Callable[[str], "str | CheckpointSlice"] | None = None,
) -> dict[str, list[str]]:
    """``load_checkpoint_into`` where ``rename`` may also return a ``CheckpointSlice``.

    Same contract otherwise: ``prefix`` is stripped from module names, the shard axis
    comes from the layer's ``_difflet_shard``, and the return value is
    ``{"missing": [...], "unexpected": [...]}`` (``strict`` makes missing an error).
    """
    targets = _load_targets(module)

    def source(name: str) -> "str | CheckpointSlice":
        key = name[len(prefix):] if prefix and name.startswith(prefix) else name
        return rename(key) if rename is not None else key

    sources = {name: source(name) for name in targets}
    wanted = {s.key if isinstance(s, CheckpointSlice) else s for s in sources.values()}
    weight_map = build_weight_map(model_dir, keys=wanted)

    missing: list[str] = []
    loaded = 0
    with torch.no_grad():
        for name, target in targets.items():
            src = sources[name]
            window = src if isinstance(src, CheckpointSlice) else None
            key = window.key if window is not None else src
            path = weight_map.get(key)
            if path is None:
                missing.append(name)
                continue
            dim = shard_dim(module, name)
            tensor = _read_window_shard(path, key, window=window, dim=dim,
                                        tp_size=tp_size, tp_rank=tp_rank)
            if tuple(tensor.shape) != tuple(target.shape):
                raise ValueError(
                    f"{name}: checkpoint slice {tuple(tensor.shape)} != parameter "
                    f"{tuple(target.shape)} (window {window}, shard dim {dim}, tp={tp_size})"
                )
            target.copy_(tensor.to(dtype or target.dtype))
            loaded += 1

    unexpected = sorted(set(weight_map) - wanted)
    logger.info(
        "loaded %d tensors from %s (rank %d/%d); %d missing, %d unexpected",
        loaded, model_dir, tp_rank, tp_size, len(missing), len(unexpected),
    )
    if strict and missing:
        raise ValueError(
            f"{len(missing)} parameters had no checkpoint entry: {missing[:5]}"
        )
    return {"missing": missing, "unexpected": unexpected}
