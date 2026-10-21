"""difflet.models.qwen_image.dit_config: the backend-neutral Qwen-Image DiT geometry.

Moved out of the TPU backend (CUDA plan PR 3 Task 6) because the arithmetic is
patch/VAE geometry, not hardware. TPU keeps ``TpuQwenImageConfig`` as an alias.
"""

import json

import pytest
import torch

from difflet.models.qwen_image.dit_config import QwenImageDiTConfig


def _config(**overrides):
    base = dict(
        patch_size=2,
        in_channels=64,
        out_channels=16,
        num_layers=2,
        attention_head_dim=128,
        num_attention_heads=24,
        joint_attention_dim=3584,
        guidance_embeds=False,
        axes_dims_rope=(16, 56, 56),
    )
    base.update(overrides)
    cfg = QwenImageDiTConfig(**base)
    cfg.validate()
    return cfg


def test_latent_geometry_follows_vae_scale_factor():
    cfg = _config(height=1024, width=1024)
    assert (cfg.latent_height, cfg.latent_width) == (128, 128)
    assert (cfg.packed_height, cfg.packed_width) == (64, 64)
    assert cfg.image_seq_len == 4096
    assert cfg.torch_dtype is torch.bfloat16


def test_tp_must_divide_the_attention_heads():
    with pytest.raises(ValueError, match="does not divide tp=5"):
        _config(tp_degree=5)


def test_axes_dims_rope_must_sum_to_head_dim():
    with pytest.raises(ValueError, match="sum to attention_head_dim"):
        _config(axes_dims_rope=(16, 56, 40))


def test_pixel_dims_must_be_aligned():
    with pytest.raises(ValueError, match="divisible by 16"):
        _config(height=1000)


def test_from_pretrained_reads_a_diffusers_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "patch_size": 2,
                "in_channels": 64,
                "num_layers": 3,
                "attention_head_dim": 128,
                "num_attention_heads": 24,
                "joint_attention_dim": 3584,
                "guidance_embeds": False,
                "axes_dims_rope": [16, 56, 56],
            }
        )
    )
    cfg = QwenImageDiTConfig.from_pretrained(tmp_path, height=512, width=512, tp_degree=2)
    assert cfg.out_channels == 16  # in_channels // 4 when absent
    assert cfg.num_layers == 3 and cfg.tp_degree == 2
    assert cfg.image_seq_len == 1024


def test_tpu_alias_is_the_same_class():
    from difflet.backends.tpu.qwen_image.config import TpuQwenImageConfig

    assert TpuQwenImageConfig is QwenImageDiTConfig


def test_pipeline_imports_the_bundle_from_contract():
    import difflet.models.qwen_image.pipeline as pipeline

    assert pipeline.QwenImageDiTInputBundle.__module__ == "difflet.models.qwen_image.contract"
