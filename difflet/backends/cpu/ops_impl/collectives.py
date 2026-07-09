"""Identity tensor-parallel collectives for CPU numerical checks."""


class _SingleProcessGroup:
    def size(self):
        return 1


class SPMDRank:
    def __init__(self, world_size: int = 1):
        self.rank = 0
        self.world_size = world_size

    def get_rank(self):
        return self.rank


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


def scatter_to_process_group_spmd(tensor, *args, **kwargs):
    del args, kwargs
    return tensor


def scatter_to_sequence_parallel_region(tensor, *, dim: int):
    # Megatron-SP forward entry; identity at tp==1 (the full sequence is the shard).
    del dim
    return tensor


def gather_from_sequence_parallel_region(tensor, *, dim: int):
    # Megatron-SP ``g`` operator; identity at tp==1.
    del dim
    return tensor


def reduce_scatter_to_sequence_parallel_region(tensor, *, dim: int):
    # Megatron-SP ``ḡ`` operator; identity at tp==1.
    del dim
    return tensor


def get_tp_size() -> int:
    return 1


def get_tp_rank() -> int:
    return 0


def get_world_group():
    return _SingleProcessGroup()


def init_parallel_mesh(config):
    """Single-process CPU backend: the mesh is trivially (1, 1, 1, 1)."""
    del config


def get_cfg_group():
    return _SingleProcessGroup()


def get_cp_group():
    return _SingleProcessGroup()


def get_cfg_rank_spmd(global_rank):
    del global_rank
    return 0


def get_cp_rank_spmd(global_rank):
    del global_rank
    return 0


get_tensor_model_parallel_size = get_tp_size
get_tensor_model_parallel_rank = get_tp_rank

__all__ = [
    "SPMDRank",
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
]
