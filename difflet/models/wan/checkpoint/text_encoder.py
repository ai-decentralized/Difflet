"""HF UMT5EncoderModel → Difflet WanUmT5EncoderModel state dict.

The Difflet ``WanUmT5EncoderModel`` mirrors HF UMT5 attribute names exactly,
so the conversion is the identity. This module exists so callers can use
the same conversion API regardless of which Wan component they're loading.
"""

from __future__ import annotations

from typing import Any, Dict

# Empty by design; mirrors HF UMT5EncoderModel layout 1:1.
TEXT_ENCODER_KEY_RENAMES: list[tuple[str, str]] = []


def convert_text_encoder_state_dict(
    state_dict: Dict[str, Any],
    config: Any | None = None,
) -> Dict[str, Any]:
    """Identity-transform the HF UMT5 state dict.

    Args:
        state_dict: HF-style state dict.
        config: unused; reserved for future variants.

    Returns:
        The same state dict (no key changes).
    """
    del config
    return dict(state_dict)


__all__ = ["TEXT_ENCODER_KEY_RENAMES", "convert_text_encoder_state_dict"]
