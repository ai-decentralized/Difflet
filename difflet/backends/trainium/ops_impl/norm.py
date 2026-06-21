"""Trainium normalization passthroughs."""

from neuronx_distributed.parallel_layers.layer_norm import LayerNorm

from difflet.backends.trainium.core.modules.custom_calls import CustomRMSNorm

RMSNorm = CustomRMSNorm

__all__ = ["CustomRMSNorm", "LayerNorm", "RMSNorm"]
