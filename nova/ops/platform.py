"""Backend platform helpers."""

from nova.backends import current_backend


def is_trainium() -> bool:
    return current_backend() == "trainium"


def is_cuda() -> bool:
    return current_backend() == "cuda"


def is_rocm() -> bool:
    return current_backend() == "rocm"


def get_platform_target():
    return _load("get_platform_target")()


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(f"module 'nova.ops.platform' has no attribute {name!r}")
    return _load(name)


def _load(name: str):
    from nova.ops._dispatch import load_backend_attr

    return load_backend_attr("platform", name)
