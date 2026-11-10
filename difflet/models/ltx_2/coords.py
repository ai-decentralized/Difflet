"""LTX-2 RoPE coordinates for the video and audio token grids, shared by every backend.

The bounds follow diffusers' ``LTX2VideoTransformer3DModel`` coordinate preparation: per-patch
[start, end) pixel coordinates (frames in seconds after the causal offset) for video, and mel
start/end times in seconds for audio.
"""

from __future__ import annotations

import torch


def make_video_coords(
    *,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int,
    patch_size_t: int,
    scale_factors: tuple[int, int, int],
    causal_offset: int,
    fps: float,
) -> torch.Tensor:
    frames = torch.arange(0, num_frames, patch_size_t, dtype=torch.float32)
    rows = torch.arange(0, height, patch_size, dtype=torch.float32)
    cols = torch.arange(0, width, patch_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(frames, rows, cols, indexing="ij"), dim=0)
    patch_size_tensor = torch.tensor(
        (patch_size_t, patch_size, patch_size),
        dtype=grid.dtype,
    )
    latent_coords = torch.stack(
        [grid, grid + patch_size_tensor.view(3, 1, 1, 1)],
        dim=-1,
    )
    latent_coords = latent_coords.flatten(1, 3).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    scale_tensor = torch.tensor(scale_factors, dtype=latent_coords.dtype)
    pixel_coords = latent_coords * scale_tensor.view(1, 3, 1, 1)
    pixel_coords[:, 0, ...] = (
        pixel_coords[:, 0, ...] + int(causal_offset) - int(scale_factors[0])
    ).clamp(min=0)
    pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / float(fps)
    return pixel_coords


def make_audio_coords(
    *,
    batch_size: int,
    audio_num_frames: int,
    patch_size_t: int,
    scale_factor: int,
    causal_offset: int,
    sampling_rate: int,
    hop_length: int,
) -> torch.Tensor:
    coords = torch.arange(0, audio_num_frames, patch_size_t, dtype=torch.float32)
    start_mel = (coords * scale_factor + int(causal_offset) - int(scale_factor)).clamp(min=0)
    end_mel = (
        (coords + int(patch_size_t)) * scale_factor + int(causal_offset) - int(scale_factor)
    ).clamp(min=0)
    seconds_per_mel = float(hop_length) / float(sampling_rate)
    coords = torch.stack([start_mel * seconds_per_mel, end_mel * seconds_per_mel], dim=-1)
    return coords.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)


__all__ = ["make_audio_coords", "make_video_coords"]
