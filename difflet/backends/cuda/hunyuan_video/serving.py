"""HunyuanVideo GPU-specific serving placement policy."""

from dataclasses import replace

from difflet.serving.errors import feature_not_supported


def resolve_clip_profile(options, profile):
    """Refuse explicit placement and discard the staged Trainium CLIP default."""
    if options.clip_placement is not None:
        raise feature_not_supported(
            "HunyuanVideo CUDA does not support --clip-placement; its text encoders reside on the GPU"
        )
    return replace(profile, clip_placement=None)
