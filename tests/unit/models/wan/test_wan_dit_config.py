"""Unit tests for the backend-neutral Wan DiT config."""

import json

import pytest
import torch

from difflet.backends.tpu.wan.config import TpuWanConfig
from difflet.models.wan.dit_config import WanDiTConfig, checkpoint_key


def test_wan_dit_config_geometry_matches_the_tpu_config():
    cfg = WanDiTConfig(height=480, width=832, num_frames=9)
    # 8x spatial, 4x temporal (+1 for the first frame), patch (1, 2, 2).
    assert (cfg.latent_frames, cfg.latent_height, cfg.latent_width) == (3, 60, 104)
    assert cfg.image_seq_len == 3 * 30 * 52 == 4680


@pytest.mark.parametrize("mode", ["context_parallel_enabled", "cfg_parallel_enabled", "sp_enabled"])
def test_wan_dit_config_validate_checks_geometry_only(mode):
    # The neutral class carries the flags but does not refuse them: whether a
    # parallel mode is implemented is the backend's call, not the geometry's.
    WanDiTConfig(**{mode: True}).validate()
    with pytest.raises(ValueError, match="width=836 must be divisible by 16"):
        WanDiTConfig(height=480, width=836).validate()
    with pytest.raises(ValueError, match="attention heads"):
        WanDiTConfig(num_attention_heads=40, tp_degree=3).validate()


def test_wan_dit_config_from_pretrained_reads_a_diffusers_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "patch_size": [1, 2, 2],
                "num_attention_heads": 40,
                "attention_head_dim": 128,
                "in_channels": 16,
                "out_channels": 16,
                "text_dim": 4096,
                "freq_dim": 256,
                "ffn_dim": 13824,
                "num_layers": 40,
                "_class_name": "WanTransformer3DModel",
            }
        )
    )
    cfg = WanDiTConfig.from_pretrained(tmp_path, tp_degree=4, torch_dtype=torch.float16)
    assert type(cfg) is WanDiTConfig
    assert cfg.patch_size == (1, 2, 2)
    assert cfg.inner_dim == 5120
    assert (cfg.tp_degree, cfg.torch_dtype) == (4, torch.float16)


def test_checkpoint_key_rewrites_only_the_ffn():
    assert checkpoint_key("blocks.0.ffn.net_in.weight") == "blocks.0.ffn.net.0.proj.weight"
    assert checkpoint_key("blocks.0.ffn.net_out.bias") == "blocks.0.ffn.net.2.bias"
    for untouched in (
        "blocks.0.attn1.to_q.weight",
        "patch_embedding.weight",
        "scale_shift_table",
        "proj_out.bias",
    ):
        assert checkpoint_key(untouched) == untouched


def test_tpu_config_is_the_neutral_config_plus_refusals():
    from difflet.backends.tpu.wan import transformer

    assert issubclass(TpuWanConfig, WanDiTConfig)
    assert transformer.checkpoint_key is checkpoint_key
    TpuWanConfig().validate()
    with pytest.raises(NotImplementedError, match="TPU backend does not implement"):
        TpuWanConfig(sp_enabled=True).validate()
    # geometry checks still come from the neutral class
    with pytest.raises(ValueError, match="attention heads"):
        TpuWanConfig(num_attention_heads=40, tp_degree=3).validate()
