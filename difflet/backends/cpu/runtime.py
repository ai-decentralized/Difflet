"""The cpu runtime: the platform in ``difflet.platforms.cpu`` under its historical names."""

from difflet.platforms.cpu import CpuPlatform as CpuBackend
from difflet.platforms.cpu import create_platform as create_backend

__all__ = ["CpuBackend", "create_backend"]
