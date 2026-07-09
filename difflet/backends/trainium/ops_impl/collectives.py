"""Trainium tensor-parallel collective passthroughs."""

from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region_with_dim,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_tensor_model_parallel_region_with_dim,
    scatter_to_process_group_spmd,
)
from neuronx_distributed.parallel_layers.mappings import (
    scatter_to_sequence_parallel_region as _nxd_scatter_to_sequence_parallel_region,
)
from neuronx_distributed.parallel_layers.mappings import (
    scatter_to_tensor_model_parallel_region,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
    get_world_group,
)

from difflet.backends.trainium.core.parallel_mesh import (
    get_cfg_group,
    get_cfg_rank_spmd,
    get_cp_group,
    get_cp_rank_spmd,
    init_parallel_mesh,
)


def gather_tp_dim(tensor, *, dim: int):
    return gather_from_tensor_model_parallel_region_with_dim(tensor, gather_dim=dim)


def reduce_tp(tensor):
    return reduce_from_tensor_model_parallel_region(tensor)


def scatter_tp_dim(tensor, *, dim: int):
    tp_size = get_tensor_model_parallel_size()
    if tp_size == 1:
        return tensor

    dim = dim % tensor.dim()
    dim_size = tensor.shape[dim]
    if dim_size % tp_size != 0:
        raise ValueError(
            f"cannot scatter dimension {dim} of size {dim_size} across tp={tp_size}"
        )

    rank = get_tensor_model_parallel_rank()
    shard_size = dim_size // tp_size
    return tensor.narrow(dim, rank * shard_size, shard_size).contiguous()


def get_tp_size() -> int:
    return get_tensor_model_parallel_size()


def get_tp_rank() -> int:
    return get_tensor_model_parallel_rank()


def scatter_to_sequence_parallel_region(tensor, *, dim: int):
    # Megatron-SP forward entry: chunk the full sequence along ``dim`` across the
    # tensor-parallel group (process_group defaults to the TP group).
    return _nxd_scatter_to_sequence_parallel_region(tensor, sequence_dimension=dim)


def gather_from_sequence_parallel_region(tensor, *, dim: int):
    # Megatron-SP ``g`` operator: all-gather the sequence shard along ``dim``
    # across the TP group back to the full sequence on every rank.
    return gather_from_tensor_model_parallel_region_with_dim(tensor, gather_dim=dim)


def reduce_scatter_to_sequence_parallel_region(tensor, *, dim: int):
    # Megatron-SP ``ḡ`` operator: reduce the row-parallel partial across the TP
    # group and scatter the result along ``dim`` (replaces the all-reduce).
    return reduce_scatter_to_tensor_model_parallel_region_with_dim(tensor, partition_dim=dim)


__all__ = [
    "gather_from_sequence_parallel_region",
    "gather_from_tensor_model_parallel_region_with_dim",
    "gather_tp_dim",
    "get_cfg_group",
    "get_cfg_rank_spmd",
    "get_cp_group",
    "get_cp_rank_spmd",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_size",
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
    "SPMDRank",
]
