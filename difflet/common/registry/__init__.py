"""Common registry helpers layered over the existing `difflet.registry` module."""

from difflet.common.registry.base import CommonModelDescriptor, ServingModelMetadata, describe_model

__all__ = ["CommonModelDescriptor", "ServingModelMetadata", "describe_model"]
