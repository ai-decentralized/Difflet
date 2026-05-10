"""Trainium tensor-parallel collective passthroughs."""

from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region_with_dim,
    reduce_from_tensor_model_parallel_region,
    scatter_to_process_group_spmd,
    scatter_to_tensor_model_parallel_region,
)
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
    get_world_group,
)

from nova.backends.trainium.utils.distributed import get_dp_rank_spmd


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


__all__ = [
    "gather_from_tensor_model_parallel_region_with_dim",
    "gather_tp_dim",
    "get_data_parallel_group",
    "get_dp_rank_spmd",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_size",
    "get_tp_rank",
    "get_tp_size",
    "get_world_group",
    "reduce_from_tensor_model_parallel_region",
    "reduce_tp",
    "scatter_to_process_group_spmd",
    "scatter_to_tensor_model_parallel_region",
    "scatter_tp_dim",
    "SPMDRank",
]
