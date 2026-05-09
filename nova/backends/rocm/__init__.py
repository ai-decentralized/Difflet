"""ROCm backend placeholder."""

from nova.backends.rocm.runtime import RocmBackend, create_backend

__all__ = ["RocmBackend", "create_backend"]
