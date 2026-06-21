"""Trainium Wan component applications."""

from difflet.backends.trainium.wan.backbone import (
    NeuronWanBackboneApplication,
    WanBackboneInferenceConfig,
)
from difflet.backends.trainium.wan.text_encoder import (
    NeuronWanTextEncoderApplication,
    WanTextEncoderInferenceConfig,
)
from difflet.backends.trainium.wan.vae import (
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
