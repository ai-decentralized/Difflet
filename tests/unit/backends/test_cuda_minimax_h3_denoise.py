"""MiniMax-H3's CUDA denoising loop on CPU: the rows' timestep slots, the timestep tables and
the loop's contract with the transformer and the schedulers. The packed layout itself is the
shared ``contracts.build_t2va_layout`` (test_fl2va.py)."""

from __future__ import annotations

import pytest
import torch

from difflet.backends.cuda.minimax_h3.denoise import _timestep_tables, denoise, row_slots
from difflet.models.minimax_h3.contracts import (
    KEYFRAME_NOISE_AUG,
    TEXT_TAG,
    VIDEO_TAG,
    build_t2va_layout,
)
from difflet.models.minimax_h3.pipeline import patchify_video_latents, step_timestep_slots
from difflet.models.minimax_h3.scheduling_minimax_h3 import MiniMaxH3Scheduler

# 32x32 pixels: a 2x2 latent grid, one transformer row per latent frame.
_SHAPE = dict(height=32, width=32, num_frames=124)


def test_rows_take_the_audio_video_and_keyframe_slots():
    tags = torch.tensor([TEXT_TAG, VIDEO_TAG, VIDEO_TAG, TEXT_TAG])
    layout = build_t2va_layout(
        num_text_tokens=4, text_token_tags=tags, keyframe_anchors=("first",), **_SHAPE
    )
    slots = row_slots(layout)
    assert layout.num_condition_video_rows == 1  # one row per latent frame here
    assert slots[:4].eq(1).all()  # text rows share the video timestep
    assert slots[4] == 2  # the keyframe row
    assert slots[layout.audio_indices].eq(0).all()
    assert slots[layout.video_indices[1:]].eq(1).all()
    t2va = row_slots(build_t2va_layout(num_text_tokens=4, **_SHAPE))
    assert not t2va.eq(2).any()


def test_timestep_tables_are_sorted_unique_slot_values_with_row_indices():
    video = torch.tensor([0.0, 0.5, 0.9995])
    audio = torch.tensor([0.0, 0.25, 0.9995])
    step_slots = step_timestep_slots(video, audio, keyframes=True)
    rows = torch.tensor([1, 2, 0, 1])  # text, keyframe, audio, video
    tables, indices = _timestep_tables(step_slots, rows, device="cpu")
    keyframe = torch.tensor(KEYFRAME_NOISE_AUG)
    assert torch.equal(tables[0], torch.stack([torch.tensor(0.0), keyframe]))
    assert torch.equal(indices[0], torch.tensor([0, 1, 0, 0]))
    assert torch.equal(tables[1], torch.stack([torch.tensor(0.25), torch.tensor(0.5), keyframe]))
    assert torch.equal(indices[1], torch.tensor([1, 2, 0, 1]))
    # Above 0.999 the keyframe slot takes the video timestep: one value left.
    assert torch.equal(tables[2], torch.tensor([0.9995])) and indices[2].eq(0).all()
    # Rows read the slot values they would read from the full table.
    for step in range(len(video)):
        assert torch.equal(tables[step][indices[step]], step_slots[step][rows])


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, video, audio, text, table, row_index, *rest):
        self.calls.append(
            dict(
                video=video.clone(),
                audio=audio,
                text=text,
                table=table,
                row_index=row_index,
                rest=rest,
            )
        )
        return torch.ones_like(video), torch.zeros_like(audio)


def _denoise(transformer, **over):
    kwargs = dict(
        encoder_hidden_states=torch.randn(1, 3, 8),
        **_SHAPE,
        num_inference_steps=4,
        seed=11,
        video_scheduler=MiniMaxH3Scheduler(shift=12.0),
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
        device=torch.device("cpu"),
    )
    kwargs.update(over)
    return denoise(transformer, **kwargs)


def test_steps_count_transformer_evaluations_with_fp32_latents():
    recorder = _Recorder()
    result = _denoise(recorder)
    assert len(recorder.calls) == 4
    for call in recorder.calls:
        assert call["video"].dtype == call["audio"].dtype == torch.float32
        assert call["text"].dtype == torch.bfloat16
        # The layout's tags, positions and row indices; the transformer takes no masks.
        assert len(call["rest"]) == 5
    layout = build_t2va_layout(num_text_tokens=3, **_SHAPE)
    assert result.video_latents.shape == (1, 24, layout.num_latent_frames, 2, 2)
    assert result.audio_latents.shape == (2, 32, layout.num_audio_latents)


def test_the_anchor_is_fed_first_and_unchanged_while_only_targets_advance():
    recorder = _Recorder()
    anchor = torch.randn(1, 1, 96)
    tags = torch.tensor([TEXT_TAG, VIDEO_TAG, TEXT_TAG])
    _denoise(recorder, keyframe_rows=anchor, text_token_tags=tags)
    first = recorder.calls[0]["video"]
    for call in recorder.calls:
        assert torch.equal(call["video"][:, :1], anchor)  # never updated
        assert call["video"].shape[1] == first.shape[1]
        keyframe_t = call["table"][call["row_index"][3]]  # row 3: the keyframe
        assert keyframe_t >= torch.tensor(KEYFRAME_NOISE_AUG)
    # The targets moved under the all-ones velocity; the anchor did not.
    assert not torch.equal(recorder.calls[-1]["video"][0, 1:], first[0, 1:])


def test_video_and_audio_noise_each_restart_the_request_seed():
    recorder = _Recorder()
    _denoise(recorder, num_inference_steps=1, seed=5)
    video, audio = recorder.calls[0]["video"], recorder.calls[0]["audio"]
    layout = build_t2va_layout(num_text_tokens=3, **_SHAPE)
    shape = (1, 24, layout.num_latent_frames, 2, 2)
    expected = torch.randn(shape, generator=torch.Generator().manual_seed(5))
    assert torch.equal(video, patchify_video_latents(expected))
    shape = (1, 2 * layout.num_audio_latents, 32)
    expected = torch.randn(shape, generator=torch.Generator().manual_seed(5))
    assert torch.equal(audio, expected)


@pytest.mark.parametrize("steps", [1, 50])
def test_the_schedulers_take_one_more_grid_point_than_steps(steps):
    recorder = _Recorder()
    _denoise(recorder, num_inference_steps=steps)
    assert len(recorder.calls) == steps
