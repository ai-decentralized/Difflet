"""Model geometry: the static table, and the token counts derived from it.

The dimensions are a static table so `difflet plan` works before a download.
That only stays true if the table matches the shipped configs, so where a
checkpoint happens to be present locally these tests read it and compare.
"""
from __future__ import annotations

import glob
import json
import os

import pytest

from difflet.planner.model_profile import (
    DIMENSIONS,
    WEIGHTS,
    ModelDimensions,
    device_weight_bytes,
    load_profile,
)
from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.registry import registered_models

MODEL_IDS = {
    "flux": "black-forest-labs/FLUX.1-dev",
    "qwen_image": "Qwen/Qwen-Image",
    "wan": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "hunyuan_video": "hunyuanvideo-community/HunyuanVideo",
    "hunyuan_video_15": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    "ltx_2": "Lightricks/LTX-2",
}


def _shipped_config(model_id: str) -> dict | None:
    """The model's transformer/config.json, if the checkpoint is cached locally."""

    org, _, name = model_id.partition("/")
    pattern = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{org}--{name}/snapshots/*/transformer/config.json"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    with open(matches[0], encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------- static table


def test_every_model_has_dimensions_and_weights():
    names = {entry.name for entry in registered_models()} & set(MODEL_IDS)
    assert names == set(MODEL_IDS)
    for name in MODEL_IDS:
        assert name in DIMENSIONS
        assert name in WEIGHTS


@pytest.mark.parametrize("name,model_id", MODEL_IDS.items())
def test_dimensions_match_the_shipped_config(name, model_id):
    config = _shipped_config(model_id)
    if config is None:
        pytest.skip(f"{model_id} is not in the local HF cache")
    dims = DIMENSIONS[name]
    assert dims.num_attention_heads == config["num_attention_heads"]
    assert dims.attention_head_dim == config["attention_head_dim"]
    assert dims.num_layers == config["num_layers"]
    assert dims.num_single_layers == config.get("num_single_layers", 0)


def test_registry_and_planner_head_counts_cannot_disagree():
    """load_profile refuses to hand back a profile whose two head counts differ."""

    from difflet import registry

    original = DIMENSIONS["flux"]
    DIMENSIONS["flux"] = ModelDimensions(
        num_attention_heads=999, attention_head_dim=128, num_layers=19, num_single_layers=38
    )
    try:
        with pytest.raises(ValueError, match="heads"):
            load_profile(MODEL_IDS["flux"], model_type="flux")
    finally:
        DIMENSIONS["flux"] = original
    assert registry is not None  # import kept meaningful


# ----------------------------------------------------------- sequence lengths


@pytest.mark.parametrize(
    "name,shape,expected_image",
    [
        # 1024x1024 / (vae 8 * patch 2) = 64x64 packed tokens
        ("flux", {"height": 1024, "width": 1024}, 4096),
        ("qwen_image", {"height": 1024, "width": 1024}, 4096),
        # (9-1)//4+1 = 3 latent frames, 480/16 x 832/16 = 30 x 52
        ("wan", {"height": 480, "width": 832, "num_frames": 9}, 3 * 30 * 52),
        # (61-1)//4+1 = 16 latent frames, 320/16 x 512/16 = 20 x 32
        ("hunyuan_video", {"height": 320, "width": 512, "num_frames": 61}, 16 * 20 * 32),
        # LTX-2 compresses 32x spatially and 8x temporally, patch 1
        ("ltx_2", {"height": 512, "width": 768, "num_frames": 121}, 16 * 16 * 24),
    ],
)
def test_token_counts(name, shape, expected_image):
    profile = load_profile(MODEL_IDS[name], model_type=name)
    seq = profile.sequence_lengths(**shape)
    assert seq.image == expected_image
    assert seq.text == DIMENSIONS[name].text_seq_len
    assert seq.joint == seq.image + seq.text


def test_defaults_are_used_when_no_shape_is_given():
    profile = load_profile(MODEL_IDS["flux"], model_type="flux")
    assert profile.sequence_lengths().image == 4096  # registry default 1024x1024


def test_image_models_ignore_the_frame_axis():
    profile = load_profile(MODEL_IDS["flux"], model_type="flux")
    with_frames = profile.sequence_lengths(height=1024, width=1024, num_frames=61)
    assert with_frames.image == 4096


def test_token_count_scales_quadratically_with_resolution():
    profile = load_profile(MODEL_IDS["flux"], model_type="flux")
    small = profile.sequence_lengths(height=512, width=512).image
    large = profile.sequence_lengths(height=1024, width=1024).image
    assert large == small * 4


# --------------------------------------------------------------- weight bytes


def test_tensor_parallelism_does_not_change_device_weight_bytes():
    """TP shards one copy across its group; the device still holds one copy."""

    profile = load_profile(MODEL_IDS["flux"], model_type="flux")
    two = device_weight_bytes(profile, DiffletParallelConfig(tp_degree=2))
    four = device_weight_bytes(profile, DiffletParallelConfig(tp_degree=4))
    assert two == four == profile.weights.total_bytes


@pytest.mark.parametrize(
    "parallel,copies",
    [
        (DiffletParallelConfig(tp_degree=4), 1),
        (DiffletParallelConfig(tp_degree=2, cp_degree=2), 2),
        (DiffletParallelConfig(tp_degree=1, cp_degree=4), 4),
        (DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True), 2),
        (DiffletParallelConfig(tp_degree=2, dp_degree=2), 2),
    ],
)
def test_cp_cfg_and_dp_each_replicate_the_model(parallel, copies):
    profile = load_profile(MODEL_IDS["flux"], model_type="flux")
    assert device_weight_bytes(profile, parallel) == copies * profile.weights.total_bytes
