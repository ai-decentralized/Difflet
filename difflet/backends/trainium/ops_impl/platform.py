"""Trainium platform helpers."""

from neuronx_distributed.utils.utils import hardware
from torch_neuronx.utils import get_platform_target

__all__ = ["get_platform_target", "hardware"]

