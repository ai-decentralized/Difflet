"""Pytest configuration for all tests.

Real ``torch`` (and the rest of the validated Neuron stack) is importable in the
reference test environment, so tests use the real library. Tests that need to
avoid the heavy Neuron runtime (``torch_xla`` / ``neuronx_distributed``) should
mock only those modules locally rather than replacing ``torch`` globally.
"""
