"""TorchNeuron implementations of the difflet.ops surface.

Pure-torch ops run on the ``neuron`` device unchanged, so modules reuse the CPU
backend's implementations wherever those were verified on the device; a module
diverges only where the device needs a different implementation.
"""
