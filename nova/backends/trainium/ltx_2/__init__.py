"""Trainium LTX-2 component wrappers."""

from nova.backends.trainium.ltx_2.transformer import (
    LTX2TransformerInferenceConfig,
    NeuronLTX2TransformerApplication,
)

__all__ = ["LTX2TransformerInferenceConfig", "NeuronLTX2TransformerApplication"]
