"""Static T2VA / FL2VA geometry shared by the H3 host controller and Neuron graphs.

MiniMax-H3 executes full self-attention over one packed sequence ordered as
``[text | keyframe conditions | target audio | target video]``; T2VA has no
keyframe conditions.  These helpers reproduce the official layout (diffusers
``modular_pipelines/minimax_h3``) without importing the unreleased-in-0.38
diffusers pipeline.
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

MIN_ASPECT_RATIO = 1 / 4
MAX_ASPECT_RATIO = 4
CANVAS_MULTIPLE = 32
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
#: The ``t`` a keyframe anchor is noised to and held at (``t = 1`` is clean).
KEYFRAME_NOISE_AUG = 0.999
#: The seed every keyframe's VAE posterior is sampled under, independent of the request seed.
KEYFRAME_ENCODE_SEED = 42
#: ImageNet pixel normalization of the video VAE's input.
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
KEYFRAME_ANCHORS = ("first", "last")

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
    attention_mask: torch.Tensor
    num_text_tokens: int
    max_text_tokens: int
    #: Leading video rows that are keyframe conditions (``len(keyframe_anchors) * rows_per_frame``).
    num_condition_video_rows: int = 0
    keyframe_anchors: tuple[str, ...] = ()

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])


def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    *,
    canvas_multiple: int = CANVAS_MULTIPLE,
    short_edge: int = CANVAS_SHORT_EDGE,
    max_pixels: int = CANVAS_MAX_PIXELS,
) -> tuple[int, int]:
    """``(height, width)`` of H3's canvas for an aspect ratio (the official ``resolve_canvas_size``)."""

    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")
    ratio = aspect_width / aspect_height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:{1 / MIN_ASPECT_RATIO:g} to "
            f"{MAX_ASPECT_RATIO:g}:1, got {aspect_width}:{aspect_height} ({ratio:g})."
        )
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale
    multiple = canvas_multiple
    return (
        max(multiple, round(height / multiple) * multiple),
        max(multiple, round(width / multiple) * multiple),
    )


def keyframe_anchor_time(anchor: str, *, num_text_tokens: int, num_latent_frames: int) -> float:
    """The rotary time a keyframe block sits at: the first or the last generated latent frame.

    ``"last"`` sums the frame spans with numpy's pairwise summation, as the reference does; the
    order matters in the last ulp from 16 latent frames on.
    """

    if anchor == "first":
        return float(num_text_tokens)
    if anchor == "last":
        spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
        for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
            spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
        return float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
    raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")


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
    text_token_tags: torch.Tensor | None = None,
    keyframe_anchors: tuple[str, ...] = (),
) -> MiniMaxH3T2VALayout:
    """Build the exact official H3 T2VA / FL2VA packed-row and MM-RoPE contract.

    ``text_token_tags`` (``(num_text_tokens,)``) tags the text rows; an FL2VA presentation tags
    the rows of a keyframe's vision block as video.  Default: all text.  ``keyframe_anchors`` adds
    one condition frame block per keyframe (``"first"``/``"last"``) between text and audio.
    """

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
    keyframe_anchors = tuple(keyframe_anchors)
    if len(keyframe_anchors) > len(KEYFRAME_ANCHORS) or len(set(keyframe_anchors)) != len(
        keyframe_anchors
    ):
        raise ValueError(f"at most one 'first' and one 'last' keyframe, got {keyframe_anchors}")
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows
    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    for index, anchor in enumerate(keyframe_anchors):
        rows = slice(
            condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame
        )
        position_ids[rows, 0] = keyframe_anchor_time(
            anchor, num_text_tokens=num_text_tokens, num_latent_frames=num_latent_frames
        )
        position_ids[rows, 1:] = frame_grid

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
    video_indices = torch.cat(
        [torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)]
    )
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    if text_token_tags is None:
        token_tags[text_indices] = TEXT_TAG
    else:
        if tuple(text_token_tags.shape) != (num_text_tokens,):
            raise ValueError(
                f"text_token_tags must be ({num_text_tokens},), got {tuple(text_token_tags.shape)}"
            )
        token_tags[text_indices] = text_token_tags.to(torch.long)
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
        attention_mask=torch.ones(sequence_length, dtype=torch.bool),
        num_text_tokens=num_text_tokens,
        max_text_tokens=num_text_tokens,
        num_condition_video_rows=num_condition_rows,
        keyframe_anchors=keyframe_anchors,
    )


def build_padded_t2va_layout(
    *,
    num_text_tokens: int,
    max_text_tokens: int,
    height: int,
    width: int,
    num_frames: int,
    sequence_alignment: int = 1,
    text_token_tags: torch.Tensor | None = None,
    keyframe_anchors: tuple[str, ...] = (),
) -> MiniMaxH3T2VALayout:
    """Move fixed-bucket text padding to the packed sequence tail.

    H3's released layout has no padding.  A Neuron graph still needs a fixed
    text bucket, so live text and all media stay a contiguous prefix while the
    unused text rows are scattered behind the media.  The resulting key mask is
    exactly ``[0, live_sequence_length)`` and can use attention_cte bounds
    without materializing an ``S x S`` mask.
    """

    if max_text_tokens < num_text_tokens:
        raise ValueError(
            f"max_text_tokens ({max_text_tokens}) is smaller than the live token count "
            f"({num_text_tokens})"
        )
    if sequence_alignment < 1:
        raise ValueError(f"sequence_alignment must be positive, got {sequence_alignment}")
    live = build_t2va_layout(
        num_text_tokens=num_text_tokens,
        height=height,
        width=width,
        num_frames=num_frames,
        text_token_tags=text_token_tags,
        keyframe_anchors=keyframe_anchors,
    )
    text_padding = max_text_tokens - num_text_tokens
    unaligned_length = live.sequence_length + text_padding
    sequence_length = (
        (unaligned_length + sequence_alignment - 1) // sequence_alignment
    ) * sequence_alignment
    graph_padding = sequence_length - unaligned_length
    if text_padding == 0 and graph_padding == 0:
        return live

    position_ids = torch.zeros(sequence_length, 3, dtype=live.position_ids.dtype)
    position_ids[: live.sequence_length] = live.position_ids
    token_tags = torch.full((sequence_length,), TEXT_TAG, dtype=torch.long)
    token_tags[: live.sequence_length] = live.token_tags
    attention_mask = torch.zeros(sequence_length, dtype=torch.bool)
    attention_mask[: live.sequence_length] = True
    text_indices = torch.cat(
        [
            live.text_indices,
            torch.arange(live.sequence_length, live.sequence_length + text_padding),
        ]
    )

    return MiniMaxH3T2VALayout(
        height=height,
        width=width,
        num_frames=num_frames,
        num_latent_frames=live.num_latent_frames,
        num_audio_latents=live.num_audio_latents,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=live.video_indices,
        audio_indices=live.audio_indices,
        text_indices=text_indices,
        attention_mask=attention_mask,
        num_text_tokens=num_text_tokens,
        max_text_tokens=max_text_tokens,
        num_condition_video_rows=live.num_condition_video_rows,
        keyframe_anchors=live.keyframe_anchors,
    )
