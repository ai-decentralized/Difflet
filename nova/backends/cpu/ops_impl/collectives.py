"""Identity tensor-parallel collectives for CPU numerical checks."""


def gather_tp_dim(tensor, *, dim: int):
    del dim
    return tensor


def gather_from_tensor_model_parallel_region_with_dim(tensor, gather_dim: int):
    del gather_dim
    return tensor


def reduce_tp(tensor):
    return tensor


def reduce_from_tensor_model_parallel_region(tensor):
    return tensor


def scatter_tp_dim(tensor, *, dim: int):
    del dim
    return tensor


def scatter_to_tensor_model_parallel_region(tensor):
    return tensor


def get_tp_size() -> int:
    return 1


def get_tp_rank() -> int:
    return 0


get_tensor_model_parallel_size = get_tp_size
get_tensor_model_parallel_rank = get_tp_rank

__all__ = [
    "gather_from_tensor_model_parallel_region_with_dim",
    "gather_tp_dim",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_size",
    "get_tp_rank",
    "get_tp_size",
    "reduce_from_tensor_model_parallel_region",
    "reduce_tp",
    "scatter_to_tensor_model_parallel_region",
    "scatter_tp_dim",
]
