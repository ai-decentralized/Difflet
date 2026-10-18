"""Trainium normalization passthroughs."""

from neuronx_distributed.parallel_layers.layer_norm import LayerNorm

from difflet.backends.trainium.core.modules.custom_calls import CustomRMSNorm

RMSNorm = CustomRMSNorm

# The pure-torch reference (the ops WanAttention's q/k norm traced before F4b lifted it
# into an op): the Neuron graph is unchanged.
from difflet.backends._torch_megatron.norm import rms_norm_fp32  # noqa: E402,F401

__all__ = ["CustomRMSNorm", "LayerNorm", "RMSNorm", "rms_norm_fp32"]
