"""Host controller for MiniMax-H3's staged Trainium T2VA path.

Only orchestration runs here: noise creation, the two H3 scheduler states, and
packing/unpacking tensors around the compiled Omni Transformer.  Text encoding
and both decoders remain separate Trainium stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from difflet.models.minimax_h3.contracts import (
    AUDIO_CHANNELS,
    PATCH_SIZE,
    VAE_SPATIAL_COMPRESSION,
    build_padded_t2va_layout,
)
from difflet.models.minimax_h3.scheduling_minimax_h3 import MiniMaxH3Scheduler


@dataclass(frozen=True)
class MiniMaxH3LatentOutput:
    video_latents: torch.Tensor
    audio_latents: torch.Tensor


def patchify_video_latents(
    latents: torch.Tensor,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> torch.Tensor:
    """Pack ``(B, C, T, H, W)`` latents into frame-major H3 rows."""

    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, frames, height, width = latents.shape
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"latent shape {tuple(latents.shape)} is not divisible by {patch_size}")
    latents = latents.reshape(
        batch_size,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(
        batch_size,
        -1,
        channels * patch_t * patch_h * patch_w,
    ).contiguous()


def unpatchify_video_latents(
    rows: torch.Tensor,
    *,
    channels: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> torch.Tensor:
    """Inverse of :func:`patchify_video_latents`."""

    patch_t, patch_h, patch_w = patch_size
    batch_size = rows.shape[0]
    rows = rows.reshape(
        batch_size,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(
        batch_size,
        channels,
        num_latent_frames,
        latent_height,
        latent_width,
    ).contiguous()


def load_minimax_h3_schedulers(
    model_path: str | Path,
) -> tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler]:
    """Load the official shift=12 video and shift=3 audio scheduler configs."""

    model_path = str(model_path)
    return (
        MiniMaxH3Scheduler.from_pretrained(model_path, subfolder="scheduler"),
        MiniMaxH3Scheduler.from_pretrained(model_path, subfolder="audio_scheduler"),
    )


@torch.no_grad()
def denoise_minimax_h3_t2va(
    transformer: Callable[..., Any],
    *,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    num_text_tokens: int,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    seed: int,
    video_scheduler: MiniMaxH3Scheduler,
    audio_scheduler: MiniMaxH3Scheduler,
    model_dtype: torch.dtype = torch.bfloat16,
) -> MiniMaxH3LatentOutput:
    """Run H3's guidance-distilled joint video/audio denoising loop."""

    if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != 1:
        raise ValueError(
            "encoder_hidden_states must have shape (1, text_seq_len, text_dim), got "
            f"{tuple(encoder_hidden_states.shape)}"
        )
    max_text_tokens = int(encoder_hidden_states.shape[1])
    if encoder_attention_mask.shape != (1, max_text_tokens):
        raise ValueError(
            f"encoder_attention_mask must have shape {(1, max_text_tokens)}, got "
            f"{tuple(encoder_attention_mask.shape)}"
        )
    if not 1 <= int(num_text_tokens) <= max_text_tokens:
        raise ValueError(
            f"num_text_tokens must be in [1, {max_text_tokens}], got {num_text_tokens}"
        )
    if int(encoder_attention_mask.sum().item()) != int(num_text_tokens):
        raise ValueError(
            "encoder_attention_mask and num_text_tokens describe different live lengths"
        )
    if num_inference_steps < 2:
        raise ValueError("MiniMax-H3 requires at least two scheduler grid points")

    layout = build_padded_t2va_layout(
        num_text_tokens=int(num_text_tokens),
        max_text_tokens=max_text_tokens,
        height=height,
        width=width,
        num_frames=num_frames,
        sequence_alignment=128,
    )
    latent_height = height // VAE_SPATIAL_COMPRESSION
    latent_width = width // VAE_SPATIAL_COMPRESSION
    latent_channels = 24
    audio_latent_channels = 32

    # The draw order is an official reproducibility contract: video first,
    # then audio, from one request-local CPU generator.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    video_latents = torch.randn(
        1,
        latent_channels,
        layout.num_latent_frames,
        latent_height,
        latent_width,
        generator=generator,
        dtype=torch.float32,
    )
    video_rows = patchify_video_latents(video_latents)
    audio_rows = torch.randn(
        1,
        layout.num_audio_latents * AUDIO_CHANNELS,
        audio_latent_channels,
        generator=generator,
        dtype=torch.float32,
    )

    video_scheduler.set_timesteps(num_inference_steps, device="cpu")
    audio_scheduler.set_timesteps(num_inference_steps, device="cpu")
    if len(video_scheduler.timesteps) != len(audio_scheduler.timesteps):
        raise RuntimeError("MiniMax-H3 video and audio schedules produced different step counts")

    # Static Neuron graphs always receive exactly two timestep embeddings.
    # Audio rows select slot 0 and video/text rows select slot 1.  Keeping both
    # slots when their values happen to be equal avoids a first-step shape fork.
    timestep_indices = torch.ones(layout.sequence_length, dtype=torch.int64)
    timestep_indices[layout.audio_indices] = 0
    fixed_inputs = (
        layout.token_tags.to(torch.int64),
        layout.position_ids.to(torch.float32),
        layout.video_indices.to(torch.int64),
        layout.audio_indices.to(torch.int64),
        layout.text_indices.to(torch.int64),
        encoder_attention_mask.to(torch.bool),
        layout.attention_mask.unsqueeze(0),
    )

    for video_timestep, audio_timestep in zip(
        video_scheduler.timesteps,
        audio_scheduler.timesteps,
    ):
        timestep = torch.stack([audio_timestep, video_timestep]).to(torch.float32)
        output = transformer(
            video_rows.to(model_dtype),
            audio_rows.to(model_dtype),
            encoder_hidden_states.to(model_dtype),
            timestep,
            timestep_indices,
            *fixed_inputs,
        )
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError("MiniMax-H3 Transformer must return video and audio predictions")
        video_prediction, audio_prediction = output
        video_rows = video_scheduler.step(
            video_prediction.float(),
            video_timestep,
            video_rows,
            return_dict=False,
        )[0]
        audio_rows = audio_scheduler.step(
            audio_prediction.float(),
            audio_timestep,
            audio_rows,
            return_dict=False,
        )[0]

    video_latents = unpatchify_video_latents(
        video_rows,
        channels=latent_channels,
        num_latent_frames=layout.num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
    )
    audio_latents = audio_rows.reshape(
        AUDIO_CHANNELS,
        layout.num_audio_latents,
        audio_latent_channels,
    ).permute(0, 2, 1)
    return MiniMaxH3LatentOutput(
        video_latents=video_latents.contiguous(),
        audio_latents=audio_latents.contiguous(),
    )
