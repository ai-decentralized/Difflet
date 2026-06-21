"""Parallel linear layers."""


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(f"module 'difflet.ops.linear' has no attribute {name!r}")
    from difflet.ops._dispatch import load_backend_attr

    return load_backend_attr("linear", name)
