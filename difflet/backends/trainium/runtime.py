"""The trainium runtime: the platform in ``difflet.platforms.trainium`` under its historical names."""

from difflet.platforms.trainium import TrainiumPlatform as TrainiumBackend
from difflet.platforms.trainium import create_platform as create_backend

__all__ = ["TrainiumBackend", "create_backend"]
