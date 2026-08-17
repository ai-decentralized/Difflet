"""TPU tensor-parallel collectives, exportable via ``torch.export``.

Why the custom-op layer (Phase 0 spike finding, 2026-08-16)
-----------------------------------------------------------
Calling ``xm.all_gather`` directly inside a module makes ``torch.export``
fail outright::

    RuntimeError: The tensor has a non-zero number of elements, but its data
    is not allocated yet ... it is likely that we are erroneously tracing
    into a custom kernel.

Direction A (AOT StableHLO export) is the settled compile model, so every
collective must be opaque to the exporter. Wrapping each one in
``torch.library.custom_op`` + ``register_fake`` fixes it, and — verified on a
real 2x2 v5e mesh — the collective is NOT left as an unlowered
``custom_call``: the saved StableHLO contains a genuine ``all-gather`` with
``replica_groups``, and fresh processes load and execute it correctly.

Two mechanical constraints shape the signatures below:

1. The custom-op schema has no ``int[][]`` type, so replica groups cross the
   boundary flattened (``groups_flat`` + ``group_size``) and are rebuilt
   inside.
2. XLA's HLO verifier rejects replica groups that do not cover *every*
   replica (``RET_CHECK ... replica groups should contain 4 replicas, but
   found 1``) — unlike NxD, which tolerates partial groups. ``parallel_mesh``
   always hands us a full partition; ``_groups`` re-checks it rather than
   trusting callers, because the failure mode is a compile-time RET_CHECK
   deep inside XLA rather than anything readable.

Phase 2a of docs/plans/2026-08-16-tpu-backend-support.md.
"""

from __future__ import annotations

import torch

from difflet.backends.tpu.ops_impl.parallel_mesh import (  # noqa: F401 — re-exported
    axis_replica_groups,
    destroy_parallel_mesh,
    get_axis_rank,
    get_cfg_group,
    get_cp_group,
    get_cp_mesh,
    get_mesh_spec,
    get_tp_groups,
    get_tp_rank,
    get_tp_size,
    init_parallel_mesh,
)


def _flatten(groups: list[list[int]]) -> tuple[list[int], int]:
    """Flatten replica groups for the custom-op boundary (no ``int[][]``)."""
    if not groups:
        raise ValueError("replica groups must be non-empty")
    size = len(groups[0])
    if size == 0:
        raise ValueError("replica groups must not contain an empty group")
    if any(len(g) != size for g in groups):
        raise ValueError(f"ragged replica groups are not representable: {groups}")
    flat: list[int] = []
    for group in groups:
        flat.extend(int(r) for r in group)
    return flat, size


def _unflatten(groups_flat: list[int], group_size: int) -> list[list[int]]:
    return [
        list(groups_flat[i : i + group_size])
        for i in range(0, len(groups_flat), group_size)
    ]


def _groups(groups: list[list[int]] | None) -> tuple[list[int], int]:
    """Validate + flatten, defaulting to the tensor-parallel axis."""
    if groups is None:
        groups = get_tp_groups()
    flat, size = _flatten(groups)
    # Must cover EVERY replica exactly once — checking only that the members
    # are contiguous from 0 is not enough: ``[[0]]`` on a 4-replica world is
    # contiguous yet is precisely what XLA rejects.
    world = get_mesh_spec().world_size
    if sorted(flat) != list(range(world)):
        # A partial group compiles into a RET_CHECK failure inside XLA with no
        # reference to difflet code; fail here instead, where it is legible.
        raise ValueError(
            f"TPU replica groups must partition all {world} replicas exactly "
            f"once (got {groups}); XLA rejects partial groups that NxD accepts"
        )
    return flat, size


# --------------------------------------------------------------------------
# Opaque custom ops: the exporter must not trace into these.
# --------------------------------------------------------------------------


@torch.library.custom_op("difflet_tpu::all_gather_dim", mutates_args=())
def _all_gather_dim(
    tensor: torch.Tensor, dim: int, groups_flat: list[int], group_size: int
) -> torch.Tensor:
    import torch_xla.core.xla_model as xm

    return xm.all_gather(
        tensor, dim=dim, groups=_unflatten(groups_flat, group_size), pin_layout=False
    )


@_all_gather_dim.register_fake
def _(
    tensor: torch.Tensor, dim: int, groups_flat: list[int], group_size: int
) -> torch.Tensor:
    shape = list(tensor.shape)
    shape[dim % tensor.dim()] *= group_size
    return tensor.new_empty(shape)


@torch.library.custom_op("difflet_tpu::all_reduce_sum", mutates_args=())
def _all_reduce_sum(
    tensor: torch.Tensor, groups_flat: list[int], group_size: int
) -> torch.Tensor:
    import torch_xla.core.xla_model as xm

    return xm.all_reduce(
        xm.REDUCE_SUM,
        tensor,
        groups=_unflatten(groups_flat, group_size),
        pin_layout=False,
    )


@_all_reduce_sum.register_fake
def _(tensor: torch.Tensor, groups_flat: list[int], group_size: int) -> torch.Tensor:
    return torch.empty_like(tensor)


@torch.library.custom_op("difflet_tpu::reduce_scatter_sum_dim", mutates_args=())
def _reduce_scatter_sum_dim(
    tensor: torch.Tensor, dim: int, groups_flat: list[int], group_size: int
) -> torch.Tensor:
    import torch_xla.core.xla_model as xm

    return xm.reduce_scatter(
        xm.REDUCE_SUM,
        tensor,
        scale=1.0,
        scatter_dim=dim,
        shard_count=group_size,
        groups=_unflatten(groups_flat, group_size),
        pin_layout=False,
    )


@_reduce_scatter_sum_dim.register_fake
def _(
    tensor: torch.Tensor, dim: int, groups_flat: list[int], group_size: int
) -> torch.Tensor:
    shape = list(tensor.shape)
    axis = dim % tensor.dim()
    if shape[axis] % group_size != 0:
        raise ValueError(
            f"cannot reduce-scatter dimension {axis} of size {shape[axis]} "
            f"across {group_size} shards"
        )
    shape[axis] //= group_size
    return tensor.new_empty(shape)


# --------------------------------------------------------------------------
# difflet.ops.collectives surface
# --------------------------------------------------------------------------


def gather_tp_dim(tensor, *, dim: int):
    if get_tp_size() == 1:
        return tensor
    flat, size = _groups(None)
    return torch.ops.difflet_tpu.all_gather_dim(tensor, dim, flat, size)


def reduce_tp(tensor):
    if get_tp_size() == 1:
        return tensor
    flat, size = _groups(None)
    return torch.ops.difflet_tpu.all_reduce_sum(tensor, flat, size)


def scatter_tp_dim(tensor, *, dim: int):
    # Pure slicing — no collective, so no custom op needed. Matches the
    # Trainium implementation, which also narrows locally.
    tp_size = get_tp_size()
    if tp_size == 1:
        return tensor

    dim = dim % tensor.dim()
    dim_size = tensor.shape[dim]
    if dim_size % tp_size != 0:
        raise ValueError(
            f"cannot scatter dimension {dim} of size {dim_size} across tp={tp_size}"
        )

    rank = get_tp_rank()
    shard_size = dim_size // tp_size
    return tensor.narrow(dim, rank * shard_size, shard_size).contiguous()


def scatter_to_sequence_parallel_region(tensor, *, dim: int):
    """Megatron-SP forward entry: chunk the full sequence across the TP group."""
    return scatter_tp_dim(tensor, dim=dim)


def gather_from_sequence_parallel_region(tensor, *, dim: int):
    """Megatron-SP ``g`` operator: all-gather the sequence shard back to full."""
    return gather_tp_dim(tensor, dim=dim)


def reduce_scatter_to_sequence_parallel_region(tensor, *, dim: int):
    """Megatron-SP ``ḡ`` operator: reduce the partial and scatter along ``dim``."""
    if get_tp_size() == 1:
        return tensor
    flat, size = _groups(None)
    return torch.ops.difflet_tpu.reduce_scatter_sum_dim(tensor, dim, flat, size)


__all__ = [
    "axis_replica_groups",
    "destroy_parallel_mesh",
    "gather_from_sequence_parallel_region",
    "gather_tp_dim",
    "get_axis_rank",
    "get_cfg_group",
    "get_cp_group",
    "get_cp_mesh",
    "get_mesh_spec",
    "get_tp_groups",
    "get_tp_rank",
    "get_tp_size",
    "init_parallel_mesh",
    "reduce_scatter_to_sequence_parallel_region",
    "reduce_tp",
    "scatter_to_sequence_parallel_region",
    "scatter_tp_dim",
]


# --------------------------------------------------------------------------
# NxD-shaped compatibility names.
#
# Model code reaches through difflet.ops's __getattr__ passthrough for names
# that only the Trainium backend defined — a genuine leak in the "frozen v1"
# ops surface, which the plan asked to log and patch additively rather than
# edit the model file. Qwen-Image's transformer needs the six below.
#
# The *_spmd family exists because NxD traces ONE graph for all ranks, so the
# rank has to be a runtime tensor input. TPU exports a separate artifact per
# rank, so the rank is a Python constant at export time and the whole
# mechanism collapses to plain integers — which is also what keeps these
# export-safe (a traced .item() would not be).
# --------------------------------------------------------------------------


class SPMDRank(torch.nn.Module):
    """NxD-compatible rank holder; a constant on TPU (per-rank artifacts)."""

    def __init__(self, world_size: int):
        super().__init__()
        self.world_size = int(world_size)

    def get_rank(self) -> int:
        return get_axis_rank("tp") if get_mesh_spec().world_size == 1 else _global_rank()


def _global_rank() -> int:
    import torch_xla.runtime as xr

    return int(xr.global_ordinal())


class _WorldGroup:
    """Minimal stand-in for NxD's world process group (``.size()``/``.rank()``)."""

    def size(self) -> int:
        return get_mesh_spec().world_size

    def rank(self) -> int:
        return 0 if get_mesh_spec().world_size == 1 else _global_rank()


def get_world_group() -> _WorldGroup:
    return _WorldGroup()


def get_tensor_model_parallel_size() -> int:
    return get_tp_size()


def get_tensor_model_parallel_rank() -> int:
    return get_tp_rank()


def get_cp_rank_spmd(global_rank):
    """cp coordinate of a global rank: ``(rank // tp) % cp``.

    Accepts an int (the TPU case) or a tensor (NxD's shape), so model code
    written against either continues to work.
    """
    spec = get_mesh_spec()
    if isinstance(global_rank, torch.Tensor):
        return torch.remainder(
            torch.div(global_rank, spec.tp, rounding_mode="floor"), spec.cp
        ).to(torch.int32)
    return (int(global_rank) // spec.tp) % spec.cp


def gather_from_tensor_model_parallel_region_with_dim(
    tensor, gather_dim: int, process_group=None
):
    """All-gather along ``gather_dim``; ``process_group`` selects the axis.

    On TPU a "process group" is just the replica-group list, so the tp-axis
    default and an explicit cp group go through the same call.
    """
    groups = get_tp_groups() if process_group is None else process_group
    if len(groups[0]) == 1:
        return tensor
    flat, size = _groups(groups)
    return torch.ops.difflet_tpu.all_gather_dim(tensor, gather_dim, flat, size)


def scatter_to_process_group_spmd(tensor, partition_dim: int, rank, process_group=None):
    """Take this rank's slice along ``partition_dim`` — local narrow, no collective."""
    groups = get_tp_groups() if process_group is None else process_group
    group_size = len(groups[0])
    if group_size == 1:
        return tensor
    if isinstance(rank, torch.Tensor):
        raise TypeError(
            "TPU exports one artifact per rank, so the scatter offset must be a "
            "Python int known at export time; got a traced tensor"
        )
    dim = partition_dim % tensor.dim()
    dim_size = tensor.shape[dim]
    if dim_size % group_size != 0:
        raise ValueError(
            f"cannot scatter dim {dim} of size {dim_size} across {group_size} ranks"
        )
    shard = dim_size // group_size
    return tensor.narrow(dim, int(rank) * shard, shard).contiguous()


def reduce_from_tensor_model_parallel_region(tensor):
    return reduce_tp(tensor)


def scatter_to_tensor_model_parallel_region(tensor, dim: int = -1):
    return scatter_tp_dim(tensor, dim=dim)


def get_cfg_rank_spmd(global_rank):
    """cfg coordinate of a global rank: ``(rank // (tp*cp)) % cfg``."""
    spec = get_mesh_spec()
    if isinstance(global_rank, torch.Tensor):
        return torch.remainder(
            torch.div(global_rank, spec.tp * spec.cp, rounding_mode="floor"), spec.cfg
        ).to(torch.int32)
    return (int(global_rank) // (spec.tp * spec.cp)) % spec.cfg


__all__ += [
    "gather_from_tensor_model_parallel_region_with_dim",
    "get_cfg_rank_spmd",
    "reduce_from_tensor_model_parallel_region",
    "scatter_to_tensor_model_parallel_region",
    "get_cp_rank_spmd",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_size",
    "get_world_group",
    "scatter_to_process_group_spmd",
    "SPMDRank",
]
