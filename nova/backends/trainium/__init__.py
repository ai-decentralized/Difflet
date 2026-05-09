"""Trainium backend."""

from nova.backends.trainium.runtime import TrainiumBackend, create_backend

__all__ = ["TrainiumBackend", "create_backend"]
