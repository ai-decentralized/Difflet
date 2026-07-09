"""Tensor-parallel collective operations."""


def gather_tp_dim(tensor, *, dim: int):
    return _load("gather_tp_dim")(tensor, dim=dim)


def reduce_tp(tensor):
    return _load("reduce_tp")(tensor)


def scatter_tp_dim(tensor, *, dim: int):
    return _load("scatter_tp_dim")(tensor, dim=dim)


def get_tp_size() -> int:
    return _load("get_tp_size")()


def get_tp_rank() -> int:
    return _load("get_tp_rank")()


def scatter_to_sequence_parallel_region(tensor, *, dim: int):
    """Megatron-SP forward entry: scatter a full-sequence tensor along ``dim``
    across the tensor-parallel group, returning this rank's ``[..., S/tp, ...]``
    shard. Identity on the CPU backend (``tp == 1``)."""

    return _load("scatter_to_sequence_parallel_region")(tensor, dim=dim)


def gather_from_sequence_parallel_region(tensor, *, dim: int):
    """Megatron-SP ``g`` operator: all-gather a sequence-sharded tensor along
    ``dim`` across the tensor-parallel group back to the full sequence, so a
    column-parallel projection sees the whole sequence. Identity on CPU."""

    return _load("gather_from_sequence_parallel_region")(tensor, dim=dim)


def reduce_scatter_to_sequence_parallel_region(tensor, *, dim: int):
    """Megatron-SP ``ḡ`` operator: reduce a row-parallel partial across the
    tensor-parallel group and scatter the result along ``dim`` (replacing the
    plain all-reduce). Identity on CPU."""

    return _load("reduce_scatter_to_sequence_parallel_region")(tensor, dim=dim)


def _load(name: str):
    from difflet.ops._dispatch import load_backend_attr

    return load_backend_attr("collectives", name)


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(f"module 'difflet.ops.collectives' has no attribute {name!r}")
    return _load(name)
