"""HunyuanVideo VAE decoder checkpoint conversion."""

from __future__ import annotations


def convert_vae_decoder_state_dict(state_dict: dict, config=None) -> dict:
    """Return decoder-only VAE weights with Difflet-compatible key names."""

    del config
    prefixes = ("post_quant_conv.", "decoder.")
    return {key: value for key, value in state_dict.items() if key.startswith(prefixes)}


__all__ = ["convert_vae_decoder_state_dict"]
