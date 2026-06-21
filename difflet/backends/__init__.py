"""Backend selection and runtime helpers."""

from difflet.backends.base import BackendCapabilities, BackendRuntime
from difflet.backends.registry import current_backend, get_backend, resolve_backend_name

__all__ = [
    "BackendCapabilities",
    "BackendRuntime",
    "current_backend",
    "get_backend",
    "resolve_backend_name",
]
