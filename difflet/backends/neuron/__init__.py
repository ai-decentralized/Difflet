"""Neuron backend: TorchNeuron, the PyTorch-native Neuron device (no AoT, one process per core)."""

from difflet.backends.neuron.runtime import NeuronBackend, create_backend

__all__ = ["NeuronBackend", "create_backend"]
