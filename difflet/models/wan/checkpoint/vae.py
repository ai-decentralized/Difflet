"""Wan VAE decoder checkpoint conversion.

The W3c spike compiles decoder-only inference. HuggingFace VAE checkpoints
also contain encoder and quantization-conv tensors, so conversion keeps only
the keys consumed by ``WanVAEDecoderModel``.
"""

from __future__ import annotations


def convert_vae_decoder_state_dict(state_dict: dict, config=None) -> dict:
    """Return decoder-only VAE weights with Difflet-compatible key names.

    Difflet's decoder module intentionally mirrors diffusers names for the active
    path: ``post_quant_conv.*`` and ``decoder.*``. No renaming is required.
    """

    del config
    prefixes = ("post_quant_conv.", "decoder.")
    return {key: value for key, value in state_dict.items() if key.startswith(prefixes)}


__all__ = ["convert_vae_decoder_state_dict"]
