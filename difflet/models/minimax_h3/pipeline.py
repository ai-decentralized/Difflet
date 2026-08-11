"""Host controller for MiniMax-H3's staged Trainium T2VA path.

Only orchestration runs here: noise creation, the two H3 scheduler states, and
packing/unpacking tensors around the compiled Omni Transformer.  Text encoding
and both decoders remain separate Trainium stages.
"""

from __future__ import annotations

import math
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


def build_adaln_modulation_table(
    transformer_path: str,
    timesteps: torch.Tensor,
    *,
    num_layers: int,
    freq_dim: int,
    time_embed_hidden_dim: int,
    time_embed_dim: int,
    hidden_size: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Precompute every block's AdaLN modulation for a fixed sampling schedule.

    The 13B `adaln_proj` branch depends only on `(timestep, modality)`, so for a
    frozen schedule its outputs can be built once on the host and streamed to
    the device per step, and the branch's weights never need to be resident.

    Args:
        transformer_path: the HF `transformer/` subfolder holding the sharded
            checkpoint and its safetensors index.
        timesteps: `(num_steps, num_timesteps)` float32 — the per-step timestep
            slots exactly as the denoise loop passes them (slot 0 audio, slot 1
            video for T2VA).

    Returns `(num_steps, num_layers, 6, num_timesteps * MODALITY_NUM,
    hidden_size)` in ``dtype``, matching the row layout `adaln_proj` emits:
    parameters in shift/scale/gate msa-then-mlp order, rows `t * 3 + modality`.
    """
    import json
    import os

    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
    from safetensors import safe_open

    from difflet.models.minimax_h3.modeling_minimax_h3 import MINIMAX_H3_MODALITY_NUM

    if timesteps.ndim != 2:
        raise ValueError(f"timesteps must be (num_steps, num_timesteps), got {list(timesteps.shape)}")
    index_path = os.path.join(
        transformer_path, "diffusion_pytorch_model.safetensors.index.json"
    )
    weight_map = json.load(open(index_path))["weight_map"]
    handles: dict[str, Any] = {}

    def _load(key: str) -> torch.Tensor:
        shard = weight_map[key]
        if shard not in handles:
            handles[shard] = safe_open(
                os.path.join(transformer_path, shard), framework="pt"
            )
        return handles[shard].get_tensor(key)

    time_proj = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
    time_embedder = TimestepEmbedding(
        in_channels=freq_dim, time_embed_dim=time_embed_hidden_dim, out_dim=time_embed_dim
    )
    time_embedder.load_state_dict(
        {
            key[len("time_embedder.") :]: _load(key)
            for key in weight_map
            if key.startswith("time_embedder.")
        }
    )
    time_embedder.float().eval()

    num_steps, num_timesteps = timesteps.shape
    with torch.no_grad():
        temb = time_embedder(time_proj(timesteps.reshape(-1).to(torch.float32)))
        activated = torch.nn.functional.silu(temb)
        rows = num_timesteps * MINIMAX_H3_MODALITY_NUM
        table = torch.empty(num_steps, num_layers, 6, rows, hidden_size, dtype=dtype)
        for layer in range(num_layers):
            weight = _load(f"transformer_blocks.{layer}.adaln_proj.linear.weight")
            bias = _load(f"transformer_blocks.{layer}.adaln_proj.linear.bias")
            # Mirror MiniMaxH3AdaLayerNormModulation exactly: activate at temb
            # precision, cast down to the projection dtype, then the checkpoint's
            # (modality, param, hidden) output layout with rows `t * 3 + modality`.
            out = torch.nn.functional.linear(activated.to(weight.dtype), weight, bias)
            out = out.view(num_steps, num_timesteps, MINIMAX_H3_MODALITY_NUM, 6, hidden_size)
            table[:, layer] = (
                out.permute(0, 3, 1, 2, 4).reshape(num_steps, 6, rows, hidden_size).to(dtype)
            )
    return table


@dataclass(frozen=True)
class MiniMaxH3LatentOutput:
    video_latents: torch.Tensor
    audio_latents: torch.Tensor


def _split_tiles(
    length: int,
    *,
    tile_size: int = 256,
    min_overlap: int = 64,
    alignment: int = 16,
) -> tuple[list[int], list[int], list[int]]:
    """Reproduce the official VAE's aligned tile placement."""

    if tile_size >= length:
        return [0], [length], []
    num_tiles = math.ceil(length / tile_size)
    while tile_size * num_tiles - min_overlap * (num_tiles - 1) < length:
        num_tiles += 1
    overlaps = [min_overlap] * (num_tiles - 1)
    remaining = tile_size * num_tiles - sum(overlaps) - length
    for index in range(remaining // alignment):
        overlaps[index % (num_tiles - 1)] += alignment
    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + tile_size - overlap)
    return starts, [tile_size] * num_tiles, overlaps


def _blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
    extent = min(a.shape[dim], b.shape[dim], extent)
    positions = torch.arange(extent, device=b.device, dtype=b.dtype)
    shape = [1] * a.ndim
    shape[dim] = extent
    weight_a = (1 - positions / extent).view(shape)
    weight_b = (positions / extent).view(shape)
    slice_a = [slice(None)] * a.ndim
    slice_a[dim] = slice(-extent, None)
    slice_b = [slice(None)] * b.ndim
    slice_b[dim] = slice(0, extent)
    blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b
    if extent == b.shape[dim]:
        return blended
    slice_rest = [slice(None)] * b.ndim
    slice_rest[dim] = slice(extent, None)
    return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)


def _stitch_tiles(
    tiles: list[list[torch.Tensor]],
    height_overlaps: list[int],
    width_overlaps: list[int],
) -> torch.Tensor:
    result_rows = []
    for row_index, row in enumerate(tiles):
        result_row = []
        for column_index, tile in enumerate(row):
            if row_index > 0:
                tile = _blend(
                    tiles[row_index - 1][column_index],
                    tile,
                    height_overlaps[row_index - 1],
                    dim=-2,
                )
            if column_index > 0:
                tile = _blend(
                    row[column_index - 1],
                    tile,
                    width_overlaps[column_index - 1],
                    dim=-1,
                )
            if row_index < len(tiles) - 1:
                tile = tile[..., : -height_overlaps[row_index], :]
            if column_index < len(row) - 1:
                tile = tile[..., :, : -width_overlaps[column_index]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


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
    adaln_modulation_table: torch.Tensor | None = None,
) -> MiniMaxH3LatentOutput:
    """Run H3's guidance-distilled joint video/audio denoising loop.

    ``adaln_modulation_table`` is required when the transformer graph was built
    with ``precomputed_adaln=True``: `(num_steps, num_layers, 6, 6, hidden)`,
    step-aligned with the two schedulers' shared timestep sequence.
    """

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

    if adaln_modulation_table is not None and int(adaln_modulation_table.shape[0]) != len(
        video_scheduler.timesteps
    ):
        raise ValueError(
            f"adaln_modulation_table covers {int(adaln_modulation_table.shape[0])} steps, "
            f"but the schedule has {len(video_scheduler.timesteps)}"
        )
    for step_index, (video_timestep, audio_timestep) in enumerate(
        zip(
            video_scheduler.timesteps,
            audio_scheduler.timesteps,
        )
    ):
        timestep = torch.stack([audio_timestep, video_timestep]).to(torch.float32)
        modulation_inputs = (
            ()
            if adaln_modulation_table is None
            else (adaln_modulation_table[step_index].to(model_dtype),)
        )
        output = transformer(
            video_rows.to(model_dtype),
            audio_rows.to(model_dtype),
            encoder_hidden_states.to(model_dtype),
            timestep,
            timestep_indices,
            *fixed_inputs,
            *modulation_inputs,
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


def _first_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


@torch.no_grad()
def decode_minimax_h3_video(
    decoder: Callable[[torch.Tensor], Any],
    normalized_latents: torch.Tensor,
    *,
    latents_mean: list[float] | tuple[float, ...],
    latents_std: list[float] | tuple[float, ...],
    clip_length: int = 17,
    token_drop: int = 3,
    temporal_compression_ratio: int = 4,
    spatial_compression_ratio: int = 16,
    tile_sample_size: int = 256,
    tile_min_overlap: int = 64,
) -> torch.Tensor:
    """Run official H3 temporal chunking and 256px spatial tiling on Neuron."""

    if normalized_latents.ndim != 5 or normalized_latents.shape[0] != 1:
        raise ValueError("MiniMax-H3 video latents must have shape (1, C, T, H, W)")
    mean = torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
    latents = normalized_latents.float() * std + mean
    tokens_chunk_size = math.ceil(clip_length / temporal_compression_ratio)
    token_overlap = (-token_drop) % tokens_chunk_size
    frame_pre_padding = (-clip_length) % temporal_compression_ratio
    frame_overlap = max(
        token_overlap * temporal_compression_ratio - frame_pre_padding,
        0,
    )
    chunk_num_frames = tokens_chunk_size * temporal_compression_ratio

    height = latents.shape[-2] * spatial_compression_ratio
    width = latents.shape[-1] * spatial_compression_ratio
    y_starts, y_lengths, y_overlaps = _split_tiles(
        height,
        tile_size=tile_sample_size,
        min_overlap=tile_min_overlap,
        alignment=spatial_compression_ratio,
    )
    x_starts, x_lengths, x_overlaps = _split_tiles(
        width,
        tile_size=tile_sample_size,
        min_overlap=tile_min_overlap,
        alignment=spatial_compression_ratio,
    )

    def decode_clip(clip: torch.Tensor) -> torch.Tensor:
        rows = []
        for y_start, y_length in zip(y_starts, y_lengths):
            row = []
            for x_start, x_length in zip(x_starts, x_lengths):
                tile = clip[
                    ...,
                    y_start // spatial_compression_ratio : y_start // spatial_compression_ratio
                    + y_length // spatial_compression_ratio,
                    x_start // spatial_compression_ratio : x_start // spatial_compression_ratio
                    + x_length // spatial_compression_ratio,
                ]
                if tile.shape[2:] != (
                    tokens_chunk_size + token_overlap,
                    tile_sample_size // spatial_compression_ratio,
                    tile_sample_size // spatial_compression_ratio,
                ):
                    raise ValueError(
                        "MiniMax-H3 visual VAE fixed graph received an incompatible latent tile, "
                        f"got {tuple(tile.shape[2:])}"
                    )
                row.append(_first_tensor(decoder(tile)).float().cpu())
            rows.append(row)
        return _stitch_tiles(rows, y_overlaps, x_overlaps)

    num_tokens = latents.shape[2] + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
    if pad_tokens:
        latents = torch.cat(
            [latents, latents[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)],
            dim=2,
        )

    decoded_chunks: list[torch.Tensor] = []
    overlap = None
    for index in range(num_chunks):
        start = index * tokens_chunk_size
        clip = decode_clip(latents[:, :, start : start + tokens_chunk_size + token_overlap])
        for subchunk in range(int(token_drop > 0) + 1):
            frame_start = subchunk * chunk_num_frames
            chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
            chunk = chunk[:, :, frame_pre_padding:]
            if subchunk == 0:
                if overlap is not None:
                    chunk = _blend(overlap, chunk, frame_overlap, dim=-3)
                decoded_chunks.append(chunk)
            else:
                overlap = chunk
    if overlap is not None:
        decoded_chunks.append(overlap)
    video = torch.cat(decoded_chunks, dim=2)

    if pad_tokens:
        intra_tail = clip_length % temporal_compression_ratio
        num_tokens_before_pad = latents.shape[2] - pad_tokens
        pad_frames = sum(
            (
                intra_tail
                if intra_tail and (num_tokens_before_pad + offset) % tokens_chunk_size == 0
                else temporal_compression_ratio
            )
            for offset in range(pad_tokens)
        )
        video = video[:, :, :-pad_frames]

    pixel_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1, 1)
    return (video * pixel_std + pixel_mean).clamp(0, 1)


@torch.no_grad()
def decode_minimax_h3_audio(
    decoder: Callable[[torch.Tensor], Any],
    normalized_latents: torch.Tensor,
    *,
    latents_mean: list[float] | tuple[float, ...],
    latents_std: list[float] | tuple[float, ...],
    chunk_latent_frames: int | None = None,
    chunk_core_frames: int = 16,
    hop_length: int = 800,
) -> torch.Tensor:
    """Denormalize stereo-as-batch H3 latents and return ``(1, 2, samples)``."""

    if normalized_latents.ndim != 3 or normalized_latents.shape[0] != AUDIO_CHANNELS:
        raise ValueError("MiniMax-H3 audio latents must have shape (2, C, T)")
    mean = torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1)
    std = torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1)
    latents = normalized_latents.float() * std + mean
    if chunk_latent_frames is None:
        waveform = _first_tensor(decoder(latents))
    else:
        total_frames = int(latents.shape[-1])
        if not total_frames > chunk_latent_frames:
            raise ValueError(
                "chunk_latent_frames must be smaller than the full audio latent length"
            )
        if not 0 < chunk_core_frames <= chunk_latent_frames:
            raise ValueError("chunk_core_frames must be in (0, chunk_latent_frames]")
        pieces = []
        decoded_windows: dict[int, torch.Tensor] = {}
        halo = (chunk_latent_frames - chunk_core_frames) // 2
        for start in range(0, total_frames, chunk_core_frames):
            end = min(start + chunk_core_frames, total_frames)
            window_start = min(
                max(0, start - halo),
                total_frames - chunk_latent_frames,
            )
            window_end = window_start + chunk_latent_frames
            decoded = decoded_windows.get(window_start)
            if decoded is None:
                decoded = _first_tensor(decoder(latents[..., window_start:window_end]))
                decoded_windows[window_start] = decoded
            sample_start = (start - window_start) * hop_length
            sample_end = (end - window_start) * hop_length
            pieces.append(decoded[..., sample_start:sample_end])
        waveform = torch.cat(pieces, dim=-1)
    if waveform.shape[:2] != (AUDIO_CHANNELS, 1):
        raise RuntimeError(
            "MiniMax-H3 audio VAE must return stereo-as-batch mono waveforms, got "
            f"{tuple(waveform.shape)}"
        )
    return waveform.float().permute(1, 0, 2).contiguous()
