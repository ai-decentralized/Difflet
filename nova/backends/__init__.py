"""Backend selection and runtime helpers."""

from nova.backends.base import BackendCapabilities, BackendRuntime
from nova.backends.registry import current_backend, get_backend, resolve_backend_name

__all__ = [
    "BackendCapabilities",
    "BackendRuntime",
    "current_backend",
    "get_backend",
    "resolve_backend_name",
]
