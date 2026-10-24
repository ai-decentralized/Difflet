"""The tpu runtime: the platform in ``difflet.platforms.tpu`` under its historical names."""

from difflet.platforms.tpu import TpuPlatform as TpuBackend
from difflet.platforms.tpu import create_platform as create_backend

__all__ = ["TpuBackend", "create_backend"]
