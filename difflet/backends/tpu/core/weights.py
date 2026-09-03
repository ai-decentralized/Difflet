"""Split full host checkpoints into this rank's tensor-parallel shards.

This is the TPU analogue of NxD's ``convert_hf_to_neuron_state_dict``. The
plan warned the Trainium version would not transfer, and it does not: NxD
splits some fused projections in its own layout (see the Flux
``proj_out_attn``/``proj_out_mlp`` split), whereas the TPU layers here declare
their own split axis via a module-level ``_difflet_shard`` mapping.

Driving the split off that declaration — rather than a per-model table of
weight names — means a layer and its loader cannot disagree: if
``ColumnParallelLinear`` ever changed which dim it shards, the loader follows
automatically.

Phase 3 of docs/plans/2026-08-16-tpu-backend-support.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def shard_dim(module: nn.Module, name: str) -> int | None:
    """Split axis declared for parameter ``name`` (dotted, from ``module``).

    The declaration lives on the owning *module* (``_difflet_shard`` maps a
    local parameter name to its split axis, or ``None`` for replicated), not
    on the Parameter object — parameters get re-created by `accelerate`'s
    `init_empty_weights`, `to_empty()`, and wrapper layers, which would drop
    an attribute set on the tensor.

    Returns ``None`` for anything that is replicated or undeclared.
    """
    parent_path, _, leaf = name.rpartition(".")
    try:
        owner = module.get_submodule(parent_path) if parent_path else module
    except AttributeError:
        return None
    return (getattr(owner, "_difflet_shard", None) or {}).get(leaf)


def narrow_to_rank(
    full: torch.Tensor, *, dim: int, tp_size: int, tp_rank: int
) -> torch.Tensor:
    if tp_size == 1:
        return full
    size = full.shape[dim]
    if size % tp_size != 0:
        raise ValueError(
            f"weight dim {dim} of size {size} is not divisible by tp={tp_size}"
        )
    shard = size // tp_size
    return full.narrow(dim, tp_rank * shard, shard).contiguous()


def shard_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    tp_size: int,
    tp_rank: int,
    strict: bool = True,
) -> dict[str, torch.Tensor]:
    """Return ``state_dict`` reduced to this rank's shards.

    ``state_dict`` holds FULL (unsharded) host weights, as they come out of a
    HuggingFace checkpoint. Every entry whose owning module declares a split
    axis for it is narrowed; everything else passes through replicated.

    Entries whose shape already matches the module's parameter are passed
    through untouched, so re-loading an already-sharded checkpoint is a no-op
    rather than a double-split — the failure mode that silently produces a
    quarter of a model.
    """
    params = dict(module.named_parameters())
    params.update(dict(module.named_buffers()))
    out: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []

    for name, tensor in state_dict.items():
        target = params.get(name)
        if target is None:
            unexpected.append(name)
            out[name] = tensor
            continue
        if tuple(tensor.shape) == tuple(target.shape):
            out[name] = tensor  # already sharded (or replicated): leave alone
            continue
        dim = shard_dim(module, name)
        if dim is None:
            raise ValueError(
                f"{name}: checkpoint shape {tuple(tensor.shape)} != parameter "
                f"shape {tuple(target.shape)}, and the parameter declares no "
                f"shard axis to reconcile them"
            )
        shard = narrow_to_rank(tensor, dim=dim, tp_size=tp_size, tp_rank=tp_rank)
        if tuple(shard.shape) != tuple(target.shape):
            raise ValueError(
                f"{name}: sharding dim {dim} of {tuple(tensor.shape)} across "
                f"tp={tp_size} gave {tuple(shard.shape)}, expected "
                f"{tuple(target.shape)}"
            )
        out[name] = shard

    if strict and unexpected:
        raise ValueError(
            f"checkpoint has {len(unexpected)} entries with no matching "
            f"parameter: {sorted(unexpected)[:5]}"
        )
    return out


def load_sharded_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    tp_size: int,
    tp_rank: int,
    strict: bool = True,
) -> None:
    sharded = shard_state_dict(
        module, state_dict, tp_size=tp_size, tp_rank=tp_rank, strict=strict
    )
    module.load_state_dict(sharded, strict=strict)


def materialize_meta_(
    module: nn.Module, device="cpu", dtype: torch.dtype | None = None
) -> None:
    """Give storage to meta tensors only, leaving real ones alone.

    NOT ``Module.to_empty()``: that blanks buffers as well, and a module built
    under ``init_empty_weights(include_buffers=False)`` keeps buffers it
    computed for real in ``__init__`` — rotary tables, frequency grids.
    Blanking those yields uninitialized positional embeddings, which corrupts
    every output without raising anything.

    ``dtype`` is applied here rather than after loading, and that is the
    difference between fitting on a v5e chip and not: allocating fp32 and
    converting the *incoming* tensor just upcasts it straight back, doubling
    the per-rank footprint. Buffers keep their computed dtype — the rope math
    runs in fp32 internally.
    """
    for sub in module.modules():
        for name, param in list(sub._parameters.items()):
            if param is not None and param.is_meta:
                sub._parameters[name] = torch.nn.Parameter(
                    torch.empty_like(param, device=device, dtype=dtype),
                    requires_grad=param.requires_grad,
                )
        for name, buf in list(sub._buffers.items()):
            if buf is not None and buf.is_meta:
                sub._buffers[name] = torch.empty_like(buf, device=device)


__all__ = [
    "load_sharded_state_dict",
    "materialize_meta_",
    "narrow_to_rank",
    "shard_dim",
    "shard_state_dict",
]
