"""Normalization layers: the CPU backend's pure-torch layers, verified on the neuron device."""

from difflet.backends.cpu.ops_impl.norm import CustomRMSNorm, LayerNorm, RMSNorm
# The fp32 reference (fp32 statistics, normalisation and affine, cast back), as Trainium and TPU.
from difflet.backends._torch_megatron.norm import rms_norm_fp32

__all__ = ["CustomRMSNorm", "LayerNorm", "RMSNorm", "rms_norm_fp32"]
