"""Parallel linear layers."""


def __getattr__(name: str):
    from nova.ops._dispatch import load_backend_attr

    return load_backend_attr("linear", name)
