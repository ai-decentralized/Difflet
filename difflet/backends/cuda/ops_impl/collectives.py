"""Collectives for the CUDA backend.

At world size 1 (D5) every collective returns its input or takes this rank's
local slice (scatters, which were always local narrows on Trainium and TPU too).
At world size > 1 (follow-up F2) the group-taking collectives run
``torch.distributed`` all-gather / all-reduce over the group ``parallel_mesh``
built (``get_cfg_group()`` / ``get_cp_group()``). Tensor parallel across
processes — the tp-group collectives and the sequence-parallel operators —
raises ``NotImplementedError`` naming follow-up F6. Semantics match
``difflet/backends/cpu/ops_impl/collectives.py`` line for line at tp=1 and
``difflet/backends/tpu/ops_impl/collectives.py`` for the surface and error
behaviour.
"""

from __future__ import annotations

import torch

from difflet.backends.cuda.ops_impl.parallel_mesh import (  # noqa: F401  (re-exported)
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

_F6 = "tensor parallel across PCIe is follow-up F6 of the CUDA backend plan"


def _not_single_process(what: str):
    raise NotImplementedError(
        f"{what}: {_F6} (the CUDA backend runs tp=1; multi-process collectives take "
        "the torch.distributed group from get_cfg_group() / get_cp_group())"
    )


def _global_rank() -> int:
    from difflet.backends.cuda.ops_impl.parallel_mesh import _runtime_ordinal

    return _runtime_ordinal()


def _is_replica_lists(group) -> bool:
    return isinstance(group, (list, tuple))


def _group_size(group) -> int:
    """Size of a replica-list partition or of a ``torch.distributed`` group."""
    if _is_replica_lists(group):
        return len(group[0])
    return torch.distributed.get_world_size(group)


def _group_ranks(group) -> list[int]:
    """Global ranks of this rank's group: its replica list, or the group's ranks."""
    if _is_replica_lists(group):
        rank = _global_rank()
        for ranks in group:
            if rank in ranks:
                return list(ranks)
        raise ValueError(f"rank {rank} is in none of the replica groups {group}")
    return list(torch.distributed.get_process_group_ranks(group))


# --------------------------------------------------------------------------
# difflet.ops.collectives surface
# --------------------------------------------------------------------------


def gather_tp_dim(tensor, *, dim: int):
    if get_tp_size() == 1:
        return tensor
    return _not_single_process("gather_tp_dim")


def reduce_tp(tensor):
    if get_tp_size() == 1:
        return tensor
    return _not_single_process("reduce_tp")


def scatter_tp_dim(tensor, *, dim: int):
    # Pure slicing — no collective. Matches Trainium and TPU, which narrow locally.
    tp_size = get_tp_size()
    if tp_size == 1:
        return tensor
    dim = dim % tensor.dim()
    dim_size = tensor.shape[dim]
    if dim_size % tp_size != 0:
        raise ValueError(f"cannot scatter dimension {dim} of size {dim_size} across tp={tp_size}")
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
    return _not_single_process("reduce_scatter_to_sequence_parallel_region")


# --------------------------------------------------------------------------
# NxD-compat surface (names model code imports from difflet.ops)
# --------------------------------------------------------------------------


class SPMDRank(torch.nn.Module):
    """NxD-compatible rank holder; a constant on a single-process backend."""

    def __init__(self, world_size: int = 1):
        super().__init__()
        self.world_size = int(world_size)

    def get_rank(self) -> int:
        return get_axis_rank("tp") if get_mesh_spec().world_size == 1 else _global_rank()


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
    """cp coordinate of a global rank: ``(rank // tp) % cp``; int or tensor."""
    spec = get_mesh_spec()
    if isinstance(global_rank, torch.Tensor):
        return torch.remainder(torch.div(global_rank, spec.tp, rounding_mode="floor"), spec.cp).to(
            torch.int32
        )
    return (int(global_rank) // spec.tp) % spec.cp


def gather_from_tensor_model_parallel_region_with_dim(tensor, gather_dim: int, process_group=None):
    """All-gather over ``process_group`` (default the TP group), concatenated in
    group-rank order along ``gather_dim``."""
    group = get_tp_groups() if process_group is None else process_group
    size = _group_size(group)
    if size == 1:
        return tensor
    if _is_replica_lists(group):
        return _not_single_process("gather_from_tensor_model_parallel_region_with_dim")
    tensor = tensor.contiguous()
    parts = [torch.empty_like(tensor) for _ in range(size)]
    torch.distributed.all_gather(parts, tensor, group=group)
    return torch.cat(parts, dim=gather_dim)


def scatter_to_process_group_spmd(tensor, partition_dim: int, rank, process_group=None):
    """Take this rank's slice along ``partition_dim`` — local narrow, no collective."""
    group = get_tp_groups() if process_group is None else process_group
    group_size = _group_size(group)
    if group_size == 1:
        return tensor
    if isinstance(rank, torch.Tensor):
        rank = int(rank.item())
    dim = partition_dim % tensor.dim()
    dim_size = tensor.shape[dim]
    if dim_size % group_size != 0:
        raise ValueError(f"cannot scatter dim {dim} of size {dim_size} across {group_size} ranks")
    shard = dim_size // group_size
    return tensor.narrow(dim, int(rank) * shard, shard).contiguous()


def reduce_from_tensor_model_parallel_region(tensor, process_group=None):
    """Sum-all-reduce over ``process_group`` (default the TP group), on a copy.

    ``process_group`` is replica lists or a ``torch.distributed`` group; it is what
    a ``RowParallelLinear`` exposes as ``tensor_parallel_group`` (Flux's single
    block merges two reduces into this one).
    """
    group = get_tp_groups() if process_group is None else process_group
    if _group_size(group) == 1:
        return tensor
    if _is_replica_lists(group):
        return _not_single_process("reduce_from_tensor_model_parallel_region")
    out = tensor.clone()
    torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM, group=group)
    return out


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


__all__ = [
    "SPMDRank",
    "axis_replica_groups",
    "destroy_parallel_mesh",
    "gather_from_sequence_parallel_region",
    "gather_from_tensor_model_parallel_region_with_dim",
    "gather_tp_dim",
    "get_axis_rank",
    "get_cfg_group",
    "get_cfg_rank_spmd",
    "get_cp_group",
    "get_cp_mesh",
    "get_cp_rank_spmd",
    "get_mesh_spec",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_size",
    "get_tp_groups",
    "get_tp_rank",
    "get_tp_size",
    "get_world_group",
    "init_parallel_mesh",
    "reduce_from_tensor_model_parallel_region",
    "reduce_scatter_to_sequence_parallel_region",
    "reduce_tp",
    "scatter_to_process_group_spmd",
    "scatter_to_sequence_parallel_region",
    "scatter_to_tensor_model_parallel_region",
    "scatter_tp_dim",
]
