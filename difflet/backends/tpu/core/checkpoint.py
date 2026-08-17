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
from pathlib import Path

import torch
import torch.nn as nn

from difflet.backends.tpu.core.weights import shard_dim

logger = logging.getLogger(__name__)

INDEX_SUFFIX = ".safetensors.index.json"


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


def _read_shard(path: str, key: str, *, dim: int | None, tp_size: int, tp_rank: int):
    """Read one tensor, taking only this rank's slice along ``dim``."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        if dim is None or tp_size == 1:
            return handle.get_tensor(key)

        view = handle.get_slice(key)
        shape = view.get_shape()
        axis = dim % len(shape)
        size = shape[axis]
        if size % tp_size != 0:
            raise ValueError(
                f"{key}: dim {axis} of size {size} is not divisible by tp={tp_size}"
            )
        width = size // tp_size
        lo, hi = tp_rank * width, (tp_rank + 1) * width
        # safetensors slices with plain indexing; only this range is read.
        selector = [slice(None)] * len(shape)
        selector[axis] = slice(lo, hi)
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
) -> dict[str, list[str]]:
    """Load ``model_dir`` into ``module``, sharding per ``_difflet_shard``.

    ``prefix`` is stripped from module parameter names to get checkpoint keys
    (difflet's trace modules nest the diffusers model under ``transformer.``).
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
            path = weight_map.get(key)
            if path is None:
                missing.append(name)
                continue
            dim = shard_dim(module, name)
            tensor = _read_shard(
                path, key, dim=dim, tp_size=tp_size, tp_rank=tp_rank
            )
            if tuple(tensor.shape) != tuple(target.shape):
                raise ValueError(
                    f"{name}: checkpoint slice {tuple(tensor.shape)} != parameter "
                    f"{tuple(target.shape)} (shard dim {dim}, tp={tp_size})"
                )
            target.copy_(tensor.to(dtype or target.dtype))
            loaded += 1

    unexpected = sorted(set(weight_map) - {
        (n[len(prefix):] if prefix and n.startswith(prefix) else n) for n in targets
    })
    logger.info(
        "loaded %d tensors from %s (rank %d/%d); %d missing, %d unexpected",
        loaded, model_dir, tp_rank, tp_size, len(missing), len(unexpected),
    )
    if strict and missing:
        raise ValueError(
            f"{len(missing)} parameters had no checkpoint entry: {missing[:5]}"
        )
    return {"missing": missing, "unexpected": unexpected}


__all__ = ["build_weight_map", "load_checkpoint_into"]
