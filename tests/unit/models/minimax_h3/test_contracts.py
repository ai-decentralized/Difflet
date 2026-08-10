from __future__ import annotations

import pytest
import torch

from difflet.models.minimax_h3.contracts import (
    AUDIO_TAG,
    TEXT_TAG,
    VIDEO_TAG,
    align_num_frames,
    build_t2va_layout,
)


def test_default_h3_t2va_layout_matches_released_geometry():
    layout = build_t2va_layout(
        num_text_tokens=1024,
        height=768,
        width=1344,
        num_frames=124,
    )

    assert layout.num_latent_frames == 37
    assert layout.num_audio_latents == 207
    assert layout.text_indices.numel() == 1024
    assert layout.audio_indices.numel() == 414
    assert layout.video_indices.numel() == 37 * 24 * 42
    assert layout.sequence_length == 1024 + 414 + 37 * 24 * 42
    assert layout.position_ids.shape == (layout.sequence_length, 3)
    assert layout.position_ids.dtype is torch.float64
    assert torch.all(layout.token_tags[layout.text_indices] == TEXT_TAG)
    assert torch.all(layout.token_tags[layout.audio_indices] == AUDIO_TAG)
    assert torch.all(layout.token_tags[layout.video_indices] == VIDEO_TAG)


def test_h3_layout_is_text_audio_video_in_order():
    layout = build_t2va_layout(
        num_text_tokens=3,
        height=768,
        width=1344,
        num_frames=124,
    )

    assert layout.text_indices.tolist() == [0, 1, 2]
    assert int(layout.audio_indices[0]) == 3
    assert int(layout.video_indices[0]) == 3 + 414
    assert layout.position_ids[:3, 0].tolist() == [0.0, 1.0, 2.0]


def test_h3_static_graph_rejects_unaligned_frames_and_canvas():
    assert align_num_frames(120) == 124
    with pytest.raises(ValueError, match=r"17 \* n \+ 5"):
        build_t2va_layout(num_text_tokens=1, height=768, width=1344, num_frames=120)
    with pytest.raises(ValueError, match="multiples of 32"):
        build_t2va_layout(num_text_tokens=1, height=770, width=1344, num_frames=124)
