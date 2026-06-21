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


def _load(name: str):
    from difflet.ops._dispatch import load_backend_attr

    return load_backend_attr("collectives", name)


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(f"module 'difflet.ops.collectives' has no attribute {name!r}")
    return _load(name)
