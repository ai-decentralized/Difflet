"""The torch-native Neuron platform is the neuron backend's runtime (the mirror of Trainium,
whose backend runtime aliases its platform)."""

from difflet.backends.neuron.runtime import NeuronBackend as NeuronPlatform
from difflet.backends.neuron.runtime import create_backend as create_platform

__all__ = ["NeuronPlatform", "create_platform"]
