"""Small top-up coverage for HunyuanVideo VAE modeling helpers.

Targets the pure tensor-repeat helpers and config validation branches not
exercised by the existing AST/config-focused VAE test.
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("DIFFLET_BACKEND", "cpu")

from difflet.models.hunyuan_video.vae import modeling_vae as vm


def test_repeat_nearest_3d_all_axes():
    x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
    out = vm._repeat_nearest_3d(x, (2, 1, 2))
    assert out.shape == (1, 1, 4, 2, 4)


def test_repeat_nearest_3d_identity():
    x = torch.zeros(1, 1, 2, 2, 2)
    out = vm._repeat_nearest_3d(x, (1, 1, 1))
    assert torch.equal(out, x)


def test_repeat_nearest_2d_and_causal_time():
    x = torch.arange(4, dtype=torch.float32).reshape(1, 1, 1, 2, 2)
    out_2d = vm._repeat_nearest_2d(x, (2, 1))
    assert out_2d.shape == (1, 1, 1, 4, 2)

    frames = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 1, 2)
    out_time = vm._repeat_causal_time(frames, 2)
    # first frame kept once, remaining 2 frames duplicated x2 -> 1 + 2*2 = 5
    assert out_time.shape == (1, 1, 5, 1, 2)
    # factor 1 is a no-op
    assert torch.equal(vm._repeat_causal_time(frames, 1), frames)


def test_config_rejects_bad_spatial_ratio():
    with pytest.raises(NotImplementedError, match="spatial compression"):
        vm.HunyuanVideoVAEDecoderConfig(spatial_compression_ratio=4)


def test_config_rejects_bad_temporal_ratio():
    with pytest.raises(NotImplementedError, match="temporal compression"):
        vm.HunyuanVideoVAEDecoderConfig(temporal_compression_ratio=2)


def test_config_rejects_unsupported_up_block():
    with pytest.raises(NotImplementedError, match="HunyuanVideoUpBlock3D"):
        vm.HunyuanVideoVAEDecoderConfig(up_block_types=("SomethingElse",) * 4)


def test_config_tile_latent_properties():
    cfg = vm.HunyuanVideoVAEDecoderConfig()
    assert cfg.tile_latent_height == cfg.tile_sample_min_height // cfg.spatial_compression_ratio
    assert cfg.tile_latent_width == cfg.tile_sample_min_width // cfg.spatial_compression_ratio
    assert cfg.tile_latent_frames == (
        cfg.tile_sample_min_num_frames // cfg.temporal_compression_ratio + 1
    )
