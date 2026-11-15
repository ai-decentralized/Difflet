"""MiniMax-H3's denoising loop on CUDA: text-to-video and first-frame image-to-video, each
with audio, over one packed sequence.

The sequence is the shared FL2VA layout (``contracts.build_t2va_layout``),
``[text | keyframe | audio | video]``; T2VA has no keyframe rows. It has no padding rows, so
the transformer gets no attention masks and SDPA can pick its fastest kernel. The keyframe
rows (``keyframe.keyframe_anchor_rows``) are fed unchanged every step and never updated;
every target frame, frame 0 included, is denoised.

Each step evaluates the shared timestep slots (``pipeline.step_timestep_slots``): audio,
video and ``max(video, 0.999)`` for the keyframe rows. As in the reference, the timestep
embedding and AdaLN projections run on the slots' sorted unique values. Those tables and the
rows' indices into them are built on the host before the loop, from one copy of the
schedules, and both schedulers start at index 0, so the loop never waits on the device.

``num_inference_steps`` counts denoising steps (transformer evaluations), as the reference
does: the schedulers get one more grid point than that.
"""

from __future__ import annotations

import torch

from difflet.backends.cuda.minimax_h3.keyframe import LATENT_CHANNELS
from difflet.models.minimax_h3.contracts import (
    AUDIO_CHANNELS,
    VAE_SPATIAL_COMPRESSION,
    build_t2va_layout,
)
from difflet.models.minimax_h3.pipeline import (
    MiniMaxH3LatentOutput,
    patchify_video_latents,
    step_timestep_slots,
    unpatchify_video_latents,
)

_AUDIO_LATENT_CHANNELS = 32
# The rows' timestep slots: audio, video (and text), keyframe.
_AUDIO_SLOT, _VIDEO_SLOT, _KEYFRAME_SLOT = 0, 1, 2


def row_slots(layout) -> torch.Tensor:
    """Each packed row's timestep slot."""
    slots = torch.full((layout.sequence_length,), _VIDEO_SLOT, dtype=torch.int64)
    slots[layout.audio_indices] = _AUDIO_SLOT
    slots[layout.video_indices[: layout.num_condition_video_rows]] = _KEYFRAME_SLOT
    return slots


def _timestep_tables(step_slots: torch.Tensor, rows: torch.Tensor, *, device):
    """Per step: the sorted unique slot values and each row's index into them, on device.

    ``step_slots`` (``(steps, slots)``) and ``rows`` are host tensors, so building the tables
    never waits on the device.
    """
    tables, indices, by_pattern = [], [], {}
    for slots in step_slots:
        unique, inverse = torch.unique(slots, sorted=True, return_inverse=True)
        pattern = tuple(inverse.tolist())
        if pattern not in by_pattern:
            by_pattern[pattern] = inverse.index_select(0, rows).to(device)
        tables.append(unique.to(device))
        indices.append(by_pattern[pattern])
    return tables, indices


@torch.no_grad()
def denoise(
    transformer,
    *,
    encoder_hidden_states: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    seed: int,
    video_scheduler,
    audio_scheduler,
    device: torch.device,
    text_token_tags: torch.Tensor | None = None,
    keyframe_rows: torch.Tensor | None = None,
) -> MiniMaxH3LatentOutput:
    """Denoise one clip; ``encoder_hidden_states`` is ``(1, text_tokens, dim)`` live text.

    ``keyframe_rows`` (``(1, rows_per_frame, 96)`` FP32, ``keyframe.keyframe_anchor_rows``) turns T2VA into first-frame I2VA.
    Returns normalized video ``(1, 24, T, H/16, W/16)`` and audio ``(2, 32, T_audio)`` latents
    on ``device``.
    """
    keyframes = keyframe_rows is not None
    layout = build_t2va_layout(
        num_text_tokens=int(encoder_hidden_states.shape[1]),
        height=height,
        width=width,
        num_frames=num_frames,
        text_token_tags=text_token_tags,
        keyframe_anchors=("first",) if keyframes else (),
    )
    latent_height = height // VAE_SPATIAL_COMPRESSION
    latent_width = width // VAE_SPATIAL_COMPRESSION
    # The reference draws video and audio noise from one CPU generator each restarted at
    # the request seed, in FP32.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    video_shape = (1, LATENT_CHANNELS, layout.num_latent_frames, latent_height, latent_width)
    video_rows = patchify_video_latents(
        torch.randn(video_shape, generator=generator, dtype=torch.float32)
    ).to(device)
    generator.manual_seed(int(seed))
    audio_shape = (1, layout.num_audio_latents * AUDIO_CHANNELS, _AUDIO_LATENT_CHANNELS)
    audio_rows = torch.randn(audio_shape, generator=generator, dtype=torch.float32).to(device)

    for scheduler in (video_scheduler, audio_scheduler):
        scheduler.set_timesteps(num_inference_steps + 1, device=device)
        # Otherwise the first step looks its timestep up on the device.
        scheduler.set_begin_index(0)
    # One host copy of each schedule: the tables below are built without device syncs.
    step_slots = step_timestep_slots(
        video_scheduler.timesteps.cpu(), audio_scheduler.timesteps.cpu(), keyframes=keyframes
    )
    tables, indices = _timestep_tables(step_slots, row_slots(layout), device=device)
    fixed = [
        t.to(device)
        for t in (
            layout.token_tags,
            layout.position_ids.float(),
            layout.video_indices,
            layout.audio_indices,
            layout.text_indices,
        )
    ]
    text = encoder_hidden_states.to(device=device, dtype=torch.bfloat16)
    if keyframe_rows is not None:
        keyframe_rows = keyframe_rows.reshape(1, -1, keyframe_rows.shape[-1]).to(device)
    num_keyframe_rows = layout.num_condition_video_rows
    steps = zip(video_scheduler.timesteps, audio_scheduler.timesteps, tables, indices)
    for video_t, audio_t, table, row_index in steps:
        model_video_rows = (
            video_rows if keyframe_rows is None else torch.cat([keyframe_rows, video_rows], dim=1)
        )
        video_prediction, audio_prediction = transformer(
            model_video_rows, audio_rows, text, table, row_index, *fixed
        )
        video_rows = video_scheduler.step(
            video_prediction[:, num_keyframe_rows:].float(), video_t, video_rows, return_dict=False
        )[0]
        audio_rows = audio_scheduler.step(
            audio_prediction.float(), audio_t, audio_rows, return_dict=False
        )[0]

    video_latents = unpatchify_video_latents(
        video_rows,
        channels=LATENT_CHANNELS,
        num_latent_frames=layout.num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
    )
    audio_latents = audio_rows.reshape(
        AUDIO_CHANNELS, layout.num_audio_latents, _AUDIO_LATENT_CHANNELS
    ).permute(0, 2, 1)
    return MiniMaxH3LatentOutput(video_latents.contiguous(), audio_latents.contiguous())


__all__ = ["denoise", "row_slots"]
