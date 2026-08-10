"""MiniMax-H3 Trainium backend."""

from difflet.backends.trainium.minimax_h3.transformer import (
    MiniMaxH3TransformerInferenceConfig,
    NeuronMiniMaxH3TransformerApplication,
)
from difflet.backends.trainium.minimax_h3.vae import (
    MiniMaxH3AudioVAEDecoderInferenceConfig,
    MiniMaxH3VideoVAEDecoderInferenceConfig,
    NeuronMiniMaxH3AudioVAEDecoderApplication,
    NeuronMiniMaxH3VideoVAEDecoderApplication,
)

__all__ = [
    "MiniMaxH3AudioVAEDecoderInferenceConfig",
    "MiniMaxH3TransformerInferenceConfig",
    "MiniMaxH3VideoVAEDecoderInferenceConfig",
    "NeuronMiniMaxH3AudioVAEDecoderApplication",
    "NeuronMiniMaxH3TransformerApplication",
    "NeuronMiniMaxH3VideoVAEDecoderApplication",
]
