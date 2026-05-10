"""Wan checkpoint conversion: HuggingFace diffusers → Nova-loadable.

Public API:
    convert_backbone_state_dict(state_dict, config) -> dict
    convert_text_encoder_state_dict(state_dict, config) -> dict
    convert_vae_decoder_state_dict(state_dict, config) -> dict
    convert_diffusers_checkpoint(model_dir, out_dir, config, components=...) -> dict

Design covered in cclogs/14.
"""

from nova.models.wan.checkpoint.backbone import (
    BACKBONE_KEY_RENAMES,
    convert_backbone_state_dict,
)
from nova.models.wan.checkpoint.cli import convert_diffusers_checkpoint
from nova.models.wan.checkpoint.text_encoder import (
    TEXT_ENCODER_KEY_RENAMES,
    convert_text_encoder_state_dict,
)
from nova.models.wan.checkpoint.vae import convert_vae_decoder_state_dict

__all__ = [
    "BACKBONE_KEY_RENAMES",
    "TEXT_ENCODER_KEY_RENAMES",
    "convert_backbone_state_dict",
    "convert_diffusers_checkpoint",
    "convert_text_encoder_state_dict",
    "convert_vae_decoder_state_dict",
]
