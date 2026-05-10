"""Normalization layers."""


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(f"module 'nova.ops.norm' has no attribute {name!r}")
    from nova.ops._dispatch import load_backend_attr

    return load_backend_attr("norm", name)
