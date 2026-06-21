"""Resolve local and Hugging Face model paths."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

# Diffusion repos on HuggingFace commonly publish weights in two formats:
#   1. A diffusers-compatible directory layout (transformer/, text_encoder/, vae/, …)
#   2. The model author's native single-file format (e.g. ``flux1-dev.safetensors``,
#      ``ae.safetensors``) targeted at ComfyUI / kohya-style tools.
# Difflet only reads the diffusers layout, so the single-file copies are 100% dead
# weight (≈23 GB for Flux). DEFAULT_DIFFUSERS_PATTERNS is a conservative
# allow-list that pulls only what diffusers' ``DiffusionPipeline.from_pretrained``
# actually walks, plus license / config metadata. It is suitable for any model
# whose registry entry doesn't override ``download_patterns``.
DEFAULT_DIFFUSERS_PATTERNS: tuple[str, ...] = (
    # Top-level metadata
    "*.json",
    "*.txt",
    "*.md",
    # Diffusers-format component directories
    "transformer/*",
    "transformer_2/*",  # for two-stage pipelines (LTX-2 etc.)
    "text_encoder/*",
    "text_encoder_2/*",
    "text_encoder_3/*",  # SD3 / Helios use a third encoder
    "vae/*",
    "vae_encoder/*",
    "vae_decoder/*",
    "unet/*",            # legacy SD-style models
    "tokenizer/*",
    "tokenizer_2/*",
    "tokenizer_3/*",
    "scheduler/*",
    "feature_extractor/*",
    "safety_checker/*",
    "image_encoder/*",
    "controlnet/*",
)


def resolve_model_path(
    model_id: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    allow_patterns: Sequence[str] | None = None,
) -> str:
    """Return a local filesystem path for ``model_id``.

    Local paths are returned as-is. HuggingFace ids are downloaded via
    ``snapshot_download`` honoring ``allow_patterns``; pass ``None`` to defer
    to ``DEFAULT_DIFFUSERS_PATTERNS`` (which excludes single-file native
    weight bundles like ``flux1-dev.safetensors`` that the diffusers layer
    never reads).
    """
    path = Path(model_id).expanduser()
    if path.exists():
        return str(path.resolve())

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to resolve non-local model ids"
        ) from exc

    patterns = (
        list(allow_patterns)
        if allow_patterns is not None
        else list(DEFAULT_DIFFUSERS_PATTERNS)
    )

    return snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_files_only=local_files_only,
        allow_patterns=patterns,
    )
