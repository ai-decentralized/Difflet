"""Backend platform helpers."""

from nova.backends import current_backend


def is_trainium() -> bool:
    return current_backend() == "trainium"


def is_cuda() -> bool:
    return current_backend() == "cuda"


def is_rocm() -> bool:
    return current_backend() == "rocm"
