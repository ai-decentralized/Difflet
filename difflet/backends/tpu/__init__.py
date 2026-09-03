"""TPU backend scaffolding (Phase 1 of docs/plans/2026-08-16-tpu-backend-support.md)."""

from difflet.backends.tpu.runtime import TpuBackend, create_backend

__all__ = ["TpuBackend", "create_backend"]
