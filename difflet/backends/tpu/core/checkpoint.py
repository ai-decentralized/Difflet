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

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from difflet.backends.tpu.core.weights import shard_dim

logger = logging.getLogger(__name__)

INDEX_SUFFIX = ".safetensors.index.json"


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


def build_weight_map(model_dir) -> dict[str, str]:
    """Map every checkpoint key to the file holding it.

    Handles both layouts: a sharded checkpoint with an index json, and a
    single ``*.safetensors`` with no index.
    """
    from safetensors import safe_open

    directory = Path(model_dir)
    indexes = sorted(directory.glob("*" + INDEX_SUFFIX))
    if indexes:
        body = json.loads(indexes[0].read_text())
        return {k: str(directory / v) for k, v in body["weight_map"].items()}

    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors under {directory}")
    weight_map: dict[str, str] = {}
    for path in files:
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                weight_map[key] = str(path)
    return weight_map


def _read_shard(
    path: str,
    key: str,
    *,
    dim: int | None,
    tp_size: int,
    tp_rank: int,
    window: tuple[int, int, int | None] | None = None,
):
    """Read one tensor, taking only this rank's slice along ``dim``.

    ``window`` is ``(axis, start, stop)`` from a ``CheckpointSlice``: the
    rank shard is taken *within* that range of the checkpoint tensor.
    """
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        if window is None and (dim is None or tp_size == 1):
            return handle.get_tensor(key)

        view = handle.get_slice(key)
        shape = list(view.get_shape())
        selector = [slice(None)] * len(shape)
        if window is not None:
            w_axis, w_lo, w_hi = window
            w_axis %= len(shape)
            if w_hi is None:
                w_hi = shape[w_axis]
            selector[w_axis] = slice(w_lo, w_hi)
            shape[w_axis] = w_hi - w_lo
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
        # safetensors slices with plain indexing; only this range is read.
        return view[tuple(selector)]


def load_checkpoint_into(
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
    """Load ``model_dir`` into ``module``, sharding per ``_difflet_shard``.

    ``prefix`` is stripped from module parameter names to get checkpoint keys
    (difflet's trace modules nest the diffusers model under ``transformer.``).

    ``rename`` maps a (prefix-stripped) *module* parameter name to the
    checkpoint key holding it, for modeling whose attribute names diverge from
    upstream diffusers — Wan's FFN is ``net_in``/``net_out`` where diffusers
    has ``net.0.proj``/``net.2``. It runs in the module→checkpoint direction so
    a parameter that exists has exactly one place to come from; the reverse
    direction would have to guess. Defaults to identity.

    Returns ``{"missing": [...], "unexpected": [...]}`` — with ``strict`` the
    missing list is an error instead.
    """
    weight_map = build_weight_map(model_dir)
    # state_dict() omits non-persistent buffers by definition — they are
    # computed, not stored, so a checkpoint never contains them and their
    # absence is not "missing". (Qwen-Image's static RoPE is exactly this.)
    persistent = set(module.state_dict().keys())
    targets = {
        name: tensor
        for name, tensor in (
            list(module.named_parameters()) + list(module.named_buffers())
        )
        if name in persistent
    }

    missing: list[str] = []
    loaded = 0
    with torch.no_grad():
        for name, target in targets.items():
            key = name[len(prefix):] if prefix and name.startswith(prefix) else name
            window = None
            if rename is not None:
                key = rename(key)
                if isinstance(key, CheckpointSlice):
                    window = (key.dim, key.start, key.stop)
                    key = key.key
            path = weight_map.get(key)
            if path is None:
                missing.append(name)
                continue
            dim = shard_dim(module, name)
            tensor = _read_shard(
                path, key, dim=dim, tp_size=tp_size, tp_rank=tp_rank, window=window
            )
            if tuple(tensor.shape) != tuple(target.shape):
                raise ValueError(
                    f"{name}: checkpoint slice {tuple(tensor.shape)} != parameter "
                    f"{tuple(target.shape)} (shard dim {dim}, tp={tp_size})"
                )
            target.copy_(tensor.to(dtype or target.dtype))
            loaded += 1

    wanted = set()
    for n in targets:
        key = n[len(prefix):] if prefix and n.startswith(prefix) else n
        if rename is not None:
            key = rename(key)
            if isinstance(key, CheckpointSlice):
                key = key.key
        wanted.add(key)
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


__all__ = ["CheckpointSlice", "build_weight_map", "load_checkpoint_into"]
