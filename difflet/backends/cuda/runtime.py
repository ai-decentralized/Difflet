"""The cuda runtime: the platform in ``difflet.platforms.cuda`` under its historical names."""

from difflet.platforms.cuda import CudaPlatform as CudaBackend
from difflet.platforms.cuda import create_platform as create_backend

__all__ = ["CudaBackend", "create_backend"]
