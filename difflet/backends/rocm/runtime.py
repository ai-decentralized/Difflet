"""The rocm runtime: the platform in ``difflet.platforms.rocm`` under its historical names."""

from difflet.platforms.rocm import RocmPlatform as RocmBackend
from difflet.platforms.rocm import create_platform as create_backend

__all__ = ["RocmBackend", "create_backend"]
