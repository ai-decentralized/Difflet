"""Tensor-parallel collectives for the neuron (TorchNeuron) backend.

Every collective is a functional collective
(``torch.distributed._functional_collectives``), so one code path serves eager
mode and ``torch.compile(backend="neuron", fullgraph=True)``. torch_neuronx
registers PrivateUse1 kernels for ``all_gather_into_tensor``,
``reduce_scatter_tensor`` and ``all_reduce``
(``torch_neuronx/distributed/ops/functional_collectives.py``), and its dynamo
backend rewrites the group name into replica groups
(``neuron_dynamo_backend/fx/passes/collective_legalization.py``). Nothing here
uses a barrier, a broadcast or an object collective: the neuron process group
supports barrier only on the default group, has no functional broadcast, and
forces object collectives to graph-break.

Each process is one rank of an MPMD launch, so a rank is a Python constant: the
NxD ``*_spmd`` helpers take an int rank, and ``SPMDRank`` holds no rank buffer.

Tensor and CFG parallelism are supported (see ``parallel_mesh`` for the groups). The CFG
path is the shared model code's: ``scatter_to_process_group_spmd`` over the CFG group is a
local slice by this rank's CFG coordinate (a Python int), and
``gather_from_tensor_model_parallel_region_with_dim`` over it is one all-gather. The CP group
is a size-1 stand-in at degree 1 and raises at a larger degree, so model code that only
imports it keeps importing.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
from torch import nn

from difflet.backends.neuron.ops_impl.parallel_mesh import (
    destroy_parallel_mesh,
    get_axis_group,
    get_axis_size,
    get_mesh_spec,
    get_tp_group,
    get_tp_rank,
    get_tp_size,
    init_parallel_mesh,
    is_mesh_initialized,
)
from difflet.pipeline.parallel_mesh import MeshSpec

_TP_ONLY = "the neuron backend supports tensor and CFG parallelism only"


class _TrivialGroup:
    """Stand-in for the process group of a size-1 axis; no torch group behind it."""

    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0


def _wait(tensor: torch.Tensor) -> torch.Tensor:
    # Eager functional collectives return an AsyncCollectiveTensor; under tracing
    # they already end in wait_tensor and return a plain tensor.
    return tensor.wait() if isinstance(tensor, fc.AsyncCollectiveTensor) else tensor


def _divisible_dim(tensor: torch.Tensor, dim: int, parts: int, what: str) -> int:
    axis = dim % tensor.dim()
    size = tensor.shape[axis]
    if size % parts != 0:
        raise ValueError(f"cannot {what} dimension {axis} of size {size} across {parts} ranks")
    return axis


def _shard(tensor: torch.Tensor, axis: int, rank: int, parts: int) -> torch.Tensor:
    width = tensor.shape[axis] // parts
    return tensor.narrow(axis, rank * width, width).contiguous()


def _gather_dim(tensor, dim: int, group) -> torch.Tensor:
    """All-gather ``tensor`` along ``dim`` over ``group`` as one plain collective."""
    axis = dim % tensor.dim()
    if axis == 0:
        return _wait(fc.all_gather_single(tensor.contiguous(), 0, group))
    # all_gather_single on another dim gathers on dim 0 and then re-concatenates
    # with chunk + cat; move the axis to dim 0 instead, so the gather is one plain
    # collective on the device. The result is made contiguous, as cat's is, so
    # callers can .view() it.
    moved = tensor.movedim(axis, 0).contiguous()
    gathered = _wait(fc.all_gather_single(moved, 0, group))
    return gathered.movedim(0, axis).contiguous()


def _spec_or_trivial() -> MeshSpec:
    return get_mesh_spec() if is_mesh_initialized() else MeshSpec()


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


# --------------------------------------------------------------------------
# difflet.ops collectives surface
# --------------------------------------------------------------------------


def gather_tp_dim(tensor, *, dim: int):
    if get_tp_size() == 1:
        return tensor
    return _gather_dim(tensor, dim, get_tp_group())


def reduce_tp(tensor):
    if get_tp_size() == 1:
        return tensor
    return _wait(fc.all_reduce(tensor, "sum", get_tp_group()))


def scatter_tp_dim(tensor, *, dim: int):
    # Local slice, no collective (as on Trainium and TPU).
    tp_size = get_tp_size()
    if tp_size == 1:
        return tensor
    axis = _divisible_dim(tensor, dim, tp_size, "scatter")
    return _shard(tensor, axis, get_tp_rank(), tp_size)


def get_tensor_model_parallel_size() -> int:
    return get_tp_size()


def get_tensor_model_parallel_rank() -> int:
    return get_tp_rank()


def scatter_to_sequence_parallel_region(tensor, *, dim: int):
    """Megatron-SP forward entry: this rank's chunk of the sequence."""
    return scatter_tp_dim(tensor, dim=dim)


def gather_from_sequence_parallel_region(tensor, *, dim: int):
    """Megatron-SP ``g``: all-gather the sequence shards back to the full sequence."""
    return gather_tp_dim(tensor, dim=dim)


def reduce_scatter_to_sequence_parallel_region(tensor, *, dim: int):
    """Megatron-SP ``g-bar``: sum the row-parallel partials and scatter along ``dim``."""
    tp_size = get_tp_size()
    if tp_size == 1:
        return tensor
    axis = _divisible_dim(tensor, dim, tp_size, "reduce-scatter")
    return _wait(fc.reduce_scatter_single(tensor.contiguous(), "sum", axis, get_tp_group()))


def gather_from_tensor_model_parallel_region_with_dim(tensor, gather_dim: int, process_group=None):
    """All-gather along ``gather_dim`` over ``process_group`` (default: the TP group)."""
    if process_group is None:
        return gather_tp_dim(tensor, dim=gather_dim)
    if int(process_group.size()) == 1:
        return tensor
    return _gather_dim(tensor, gather_dim, process_group)


def reduce_from_tensor_model_parallel_region(tensor):
    return reduce_tp(tensor)


def scatter_to_tensor_model_parallel_region(tensor, dim: int = -1):
    return scatter_tp_dim(tensor, dim=dim)


def scatter_to_process_group_spmd(tensor, partition_dim: int, rank, process_group=None):
    """Slice ``rank`` of ``partition_dim`` split over ``process_group``: a local narrow, no
    collective.

    With ``process_group=None`` (the TP group) ``rank`` may be the global rank, as the model
    code passes it (``SPMDRank.get_rank()``); it is mapped to its TP coordinate, ``rank % tp``
    in the mesh layout (a TP-local rank maps to itself). With an explicit group ``rank`` is the
    caller's coordinate in that group (the CFG scatter passes ``get_cfg_rank_spmd``)."""
    size = get_tp_size() if process_group is None else int(process_group.size())
    if size == 1:
        return tensor
    if isinstance(rank, torch.Tensor):
        raise TypeError(
            "the neuron backend runs one process per rank, so the scatter rank must be a "
            "Python int (SPMDRank.get_rank() returns one); got a tensor"
        )
    rank = int(rank)
    if process_group is None and is_mesh_initialized() and 0 <= rank < get_mesh_spec().world_size:
        rank = get_mesh_spec().axis_rank(rank, "tp")
    if not 0 <= rank < size:
        raise ValueError(f"rank {rank} is outside the process group of size {size}")
    axis = _divisible_dim(tensor, partition_dim, size, "scatter")
    return _shard(tensor, axis, rank, size)


class SPMDRank(nn.Module):
    """NxD-compatible rank holder; the rank is a Python constant (one process per rank)."""

    def __init__(self, world_size: int):
        super().__init__()
        self.world_size = int(world_size)

    def get_rank(self) -> int:
        return dist.get_rank() if _dist_ready() else 0


def get_world_group():
    return dist.group.WORLD if _dist_ready() else _TrivialGroup()


def _trivial_axis_group(axis: str) -> _TrivialGroup:
    size = _spec_or_trivial().axis_size(axis)
    if size > 1:
        raise NotImplementedError(f"{_TP_ONLY}; no {axis} group exists ({axis}={size})")
    return _TrivialGroup()


def get_cfg_group():
    """This rank's CFG group (two ranks, one per guidance branch) or a size-1 stand-in."""
    if get_axis_size("cfg") > 1:
        return get_axis_group("cfg")
    return _trivial_axis_group("cfg")


def get_cp_group():
    return _trivial_axis_group("cp")


def get_cfg_rank_spmd(global_rank):
    """cfg coordinate of a global rank: ``(rank // (tp*cp)) % cfg``."""
    spec = _spec_or_trivial()
    if isinstance(global_rank, torch.Tensor):
        return torch.remainder(
            torch.div(global_rank, spec.tp * spec.cp, rounding_mode="floor"), spec.cfg
        ).to(torch.int32)
    return (int(global_rank) // (spec.tp * spec.cp)) % spec.cfg


def get_cp_rank_spmd(global_rank):
    """cp coordinate of a global rank: ``(rank // tp) % cp``."""
    spec = _spec_or_trivial()
    if isinstance(global_rank, torch.Tensor):
        return torch.remainder(
            torch.div(global_rank, spec.tp, rounding_mode="floor"), spec.cp
        ).to(torch.int32)
    return (int(global_rank) // spec.tp) % spec.cp


__all__ = [
    "SPMDRank",
    "destroy_parallel_mesh",
    "gather_from_sequence_parallel_region",
    "gather_from_tensor_model_parallel_region_with_dim",
    "gather_tp_dim",
    "get_cfg_group",
    "get_cfg_rank_spmd",
    "get_cp_group",
    "get_cp_rank_spmd",
    "get_mesh_spec",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_size",
    "get_tp_group",
    "get_tp_rank",
    "get_tp_size",
    "get_world_group",
    "init_parallel_mesh",
    "is_mesh_initialized",
    "reduce_from_tensor_model_parallel_region",
    "reduce_scatter_to_sequence_parallel_region",
    "reduce_tp",
    "scatter_to_process_group_spmd",
    "scatter_to_sequence_parallel_region",
    "scatter_to_tensor_model_parallel_region",
    "scatter_tp_dim",
]
