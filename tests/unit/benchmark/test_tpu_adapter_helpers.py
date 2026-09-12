"""Chip-free pieces of the TPU benchmark adapter."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmark.adapters import tpu_models
from benchmark.adapters.tpu import _reported_steps


def test_reported_steps_synced_are_the_deltas():
    assert _reported_steps([0.5, 0.6], 20.0, True) == [0.5, 0.6]


def test_reported_steps_natural_is_denoise_wall_over_steps():
    # 3 deltas -> 4 steps; the enqueue deltas are not device time
    assert _reported_steps([0.01, 0.01, 0.01], 4.0, False) == [1.0, 1.0, 1.0]
    assert _reported_steps([], 4.0, False) == []


@pytest.mark.parametrize("layout,shape,value_range,expect", [
    ("BCTHW", (1, 3, 5, 8, 6), "minus_one_to_one", (5, 8, 6, 3)),
    ("BFCHW", (1, 5, 3, 8, 6), "zero_to_one", (5, 8, 6, 3)),
    ("BCHW", (1, 3, 8, 6), "zero_to_one", (1, 8, 6, 3)),
])
def test_frames_uint8_layouts(layout, shape, value_range, expect):
    x = torch.linspace(-1.5, 1.5, int(np.prod(shape))).reshape(shape)
    frames = tpu_models.frames_uint8(x, layout, value_range)
    assert frames.shape == expect and frames.dtype == np.uint8
    assert frames.min() == 0 and frames.max() == 255


def test_frames_uint8_channel_order_is_preserved():
    x = torch.zeros(1, 3, 2, 2, 2)
    x[0, 1] = 1.0  # green everywhere
    frames = tpu_models.frames_uint8(x, "BCTHW", "minus_one_to_one")
    assert frames[0, 0, 0].tolist() == [128, 255, 128]


def test_output_info_fields_match_the_harness_schema():
    info = tpu_models.output_info(torch.tensor([[0.0, 2.0]], dtype=torch.bfloat16), note="n")
    assert set(info) == {"shape", "dtype", "finite", "min", "max", "mean", "std", "note"}
    assert info["shape"] == [1, 2] and info["dtype"] == "torch.bfloat16"
    assert info["finite"] is True and info["max"] == 2.0 and info["note"] == "n"
    assert tpu_models.output_info(torch.tensor([float("nan")]))["finite"] is False


def test_save_media_image_and_video(tmp_path):
    frames = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    assert tpu_models.save_media(frames, str(tmp_path / "img"), kind="image") == [str(tmp_path / "img.png")]
    video = np.zeros((7, 4, 4, 3), dtype=np.uint8)
    written = tpu_models.save_media(video, str(tmp_path / "vid"), kind="video", fps=8)
    assert str(tmp_path / "vid_frames_0_3_6.png") in written
    assert (tmp_path / "vid_frames_0_3_6.png").exists()


def test_make_driver_rejects_unknown_model_type():
    with pytest.raises(NotImplementedError):
        tpu_models.make_driver({"model_type": "nope"}, 0, 4, None)


def test_every_serving_tpu_model_has_a_driver():
    assert set(tpu_models.DRIVERS) == {"qwen_image", "flux", "wan", "hunyuan_video", "ltx_2"}
