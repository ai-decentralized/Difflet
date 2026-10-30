"""Fixtures for the HunyuanVideo 1.5 AOT loop tests: a tiny diffusers snapshot on disk."""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("DIFFLET_BACKEND", "cpu")


@pytest.fixture(scope="session")
def hv15_snapshot(tmp_path_factory):
    import torch
    from diffusers import (
        AutoencoderKLHunyuanVideo15,
        FlowMatchEulerDiscreteScheduler,
        HunyuanVideo15Transformer3DModel,
    )

    root = tmp_path_factory.mktemp("hv15_snapshot")
    torch.manual_seed(0)
    transformer = HunyuanVideo15Transformer3DModel(
        in_channels=9,
        out_channels=4,
        num_attention_heads=4,
        attention_head_dim=8,
        num_layers=2,
        num_refiner_layers=1,
        mlp_ratio=4.0,
        patch_size=1,
        patch_size_t=1,
        qk_norm="rms_norm",
        text_embed_dim=32,
        text_embed_2_dim=24,
        image_embed_dim=16,
        rope_axes_dim=(2, 2, 4),
        rope_theta=256.0,
        target_size=640,
        task_type="t2v",
        use_meanflow=False,
    )
    vae = AutoencoderKLHunyuanVideo15(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(8, 16, 16, 16, 16),
        layers_per_block=1,
        spatial_compression_ratio=16,
        temporal_compression_ratio=4,
        downsample_match_channel=True,
        upsample_match_channel=True,
        scaling_factor=0.9,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
    transformer.save_pretrained(root / "transformer")
    vae.save_pretrained(root / "vae")
    scheduler.save_pretrained(root / "scheduler")
    (root / "guider").mkdir()
    (root / "guider" / "guider_config.json").write_text(
        json.dumps(
            {
                "_class_name": "ClassifierFreeGuidance",
                "guidance_scale": 6.0,
                "guidance_rescale": 0.0,
                "use_original_formulation": False,
                "start": 0.0,
                "stop": 1.0,
                "enabled": True,
            }
        )
    )
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo15Pipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "transformer": ["diffusers", "HunyuanVideo15Transformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLHunyuanVideo15"],
                "guider": ["diffusers", "ClassifierFreeGuidance"],
            }
        )
    )
    return str(root)
