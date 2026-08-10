"""Static T2VA geometry shared by the H3 host controller and Neuron graphs.

MiniMax-H3 executes full self-attention over one packed sequence ordered as
``[text | target audio | target video]`` for T2VA.  These helpers reproduce the
official layout without importing the unreleased-in-0.38 diffusers pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2
FPS = 24
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
VAE_SPATIAL_COMPRESSION = 16
PATCH_SIZE = (1, 2, 2)
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


@dataclass(frozen=True)
class MiniMaxH3T2VALayout:
    height: int
    width: int
    num_frames: int
    num_latent_frames: int
    num_audio_latents: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])


def align_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    while num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def video_latent_num_frames(num_frames: int) -> int:
    if num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(
            f"num_frames must be of the form {FRAMES_PER_CHUNK} * n + "
            f"{LATENTS_PER_CHUNK}, got {num_frames}"
        )
    return (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False)
    return torch.from_numpy(grid * _ROPE_SPATIAL_SCALE).to(torch.float64)


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def build_t2va_layout(
    *,
    num_text_tokens: int,
    height: int,
    width: int,
    num_frames: int,
) -> MiniMaxH3T2VALayout:
    """Build the exact official H3 T2VA packed-row and MM-RoPE contract."""

    if num_text_tokens < 1:
        raise ValueError("MiniMax-H3 requires at least one text token")
    if height % 32 or width % 32:
        raise ValueError(f"height and width must be multiples of 32, got {height}x{width}")
    if not 120 <= num_frames <= 360:
        raise ValueError(f"MiniMax-H3 supports 5-15 seconds at 24 fps, got {num_frames} frames")
    if align_num_frames(num_frames) != num_frames:
        raise ValueError(
            f"num_frames must already be aligned to 17 * n + 5 for a static graph, got {num_frames}"
        )

    latent_height = height // VAE_SPATIAL_COMPRESSION
    latent_width = width // VAE_SPATIAL_COMPRESSION
    _, patch_h, patch_w = PATCH_SIZE
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_latent_frames = video_latent_num_frames(num_frames)
    num_audio_latents = audio_latent_num_frames(num_frames)
    num_audio_rows = num_audio_latents * AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_audio_rows + num_video_rows
    audio_start = num_text_tokens
    video_start = audio_start + num_audio_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )

    video_positions = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_positions[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[
        :, None
    ]
    video_positions[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_positions.reshape(-1, 3)

    text_indices = torch.arange(num_text_tokens)
    audio_indices = torch.arange(audio_start, video_start)
    video_indices = torch.arange(video_start, sequence_length)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = TEXT_TAG
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG

    return MiniMaxH3T2VALayout(
        height=height,
        width=width,
        num_frames=num_frames,
        num_latent_frames=num_latent_frames,
        num_audio_latents=num_audio_latents,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
    )
