"""Trainium Wan component applications."""

from nova.backends.trainium.wan.backbone import (
    NeuronWanBackboneApplication,
    WanBackboneInferenceConfig,
)
from nova.backends.trainium.wan.text_encoder import (
    NeuronWanTextEncoderApplication,
    WanTextEncoderInferenceConfig,
)
from nova.backends.trainium.wan.vae import (
    NeuronWanVAEDecoderApplication,
    WanVAEDecoderInferenceConfig,
)

__all__ = [
    "NeuronWanBackboneApplication",
    "NeuronWanTextEncoderApplication",
    "NeuronWanVAEDecoderApplication",
    "WanBackboneInferenceConfig",
    "WanTextEncoderInferenceConfig",
    "WanVAEDecoderInferenceConfig",
]
