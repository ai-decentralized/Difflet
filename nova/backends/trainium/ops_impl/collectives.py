"""Trainium tensor-parallel collective passthroughs."""

from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region_with_dim,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)

__all__ = [
    "gather_from_tensor_model_parallel_region_with_dim",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_size",
    "reduce_from_tensor_model_parallel_region",
    "scatter_to_tensor_model_parallel_region",
]
