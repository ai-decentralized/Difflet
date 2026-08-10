"""Shared MiniMax-H3 staged-CLI constants and cache paths."""

from __future__ import annotations

from pathlib import Path

HF_MODEL_ID = "MiniMaxAI/MiniMax-H3"
MODEL_TYPE = "minimax_h3"
CLI_NAME = "minimax-h3"
VIRTUAL_CORE_SIZE = 2

TEXT_ENCODER_LAYER = 50
TEXT_SEQ_LEN = 1024
VIDEO_VAE_FRAMES_PER_CHUNK = 17
VIDEO_VAE_LATENTS_PER_CHUNK = 5

DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 1344
DEFAULT_NUM_FRAMES = 124
DEFAULT_STEPS = 30


def stage_compiled_dir_from_values(
    stage: str,
    *,
    cache_dir: str | None,
    tp_degree: int,
    height: int,
    width: int,
    num_frames: int,
) -> Path:
    """Return a collision-free cache path for one staged H3 artifact."""

    base = Path(cache_dir or Path.home() / ".cache" / "difflet").expanduser()
    if stage == "text":
        return base / (f"minimax_h3_text_tp{tp_degree}_seq{TEXT_SEQ_LEN}_layer{TEXT_ENCODER_LAYER}")
    if stage == "generate":
        return base / (
            f"minimax_h3_dit_tp{tp_degree}_h{height}w{width}f{num_frames}" f"_text{TEXT_SEQ_LEN}"
        )
    if stage == "video_vae":
        return base / f"minimax_h3_video_vae_h{height}w{width}f{num_frames}"
    if stage == "audio_vae":
        return base / f"minimax_h3_audio_vae_f{num_frames}"
    raise ValueError(f"unknown MiniMax-H3 stage {stage!r}")
