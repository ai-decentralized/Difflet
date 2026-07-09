"""Unit tests for HunyuanVideo VAE decoder modeling."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
from types import SimpleNamespace

import torch


def test_modeling_hunyuan_video_vae_imports_only_from_allowed_modules():
    src = (_REPO_ROOT / "difflet/models/hunyuan_video/vae/modeling_vae.py").read_text()
    tree = ast.parse(src)
    forbidden_roots = {"neuronx_distributed", "nkilib", "torch_neuronx"}
    forbidden_prefixes = ("difflet.core",)
    offending: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in forbidden_roots or alias.name.startswith(forbidden_prefixes):
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".", 1)[0]
            if root in forbidden_roots or mod.startswith(forbidden_prefixes):
                offending.append(mod)

    assert not offending, f"forbidden imports in modeling_hunyuan_video/vae.py: {offending}"


def test_hunyuan_video_vae_decoder_config_defaults_match_hf_v0():
    from difflet.models.hunyuan_video.vae.modeling_vae import HunyuanVideoVAEDecoderConfig

    cfg = HunyuanVideoVAEDecoderConfig()
    assert cfg.latent_channels == 16
    assert cfg.block_out_channels == (128, 256, 512, 512)
    assert cfg.layers_per_block == 2
    assert cfg.spatial_compression_ratio == 8
    assert cfg.temporal_compression_ratio == 4
    assert cfg.tile_latent_frames == 5
    assert cfg.tile_latent_height == 32
    assert cfg.tile_latent_width == 32


def test_hunyuan_video_vae_decoder_config_filters_diffusers_dict():
    from difflet.models.hunyuan_video.vae.modeling_vae import HunyuanVideoVAEDecoderConfig

    raw = {
        "_class_name": "AutoencoderKLHunyuanVideo",
        "latent_channels": 16,
        "block_out_channels": [128, 256, 512, 512],
        "up_block_types": ["HunyuanVideoUpBlock3D"] * 4,
        "scaling_factor": 0.476986,
    }
    cfg = HunyuanVideoVAEDecoderConfig.from_diffusers_dict(raw)
    assert cfg.block_out_channels == (128, 256, 512, 512)
    assert cfg.up_block_types == ("HunyuanVideoUpBlock3D",) * 4
    assert cfg.extra_config["_class_name"] == "AutoencoderKLHunyuanVideo"


def test_hunyuan_video_vae_decoder_classes_are_importable():
    from difflet.models.hunyuan_video.vae import modeling_vae

    for name in ["HunyuanVideoVAEDecoderConfig", "HunyuanVideoVAEDecoderModel"]:
        cls = getattr(modeling_vae, name)
        assert inspect.isclass(cls), f"{name} should be a class"


def test_hunyuan_video_vae_decoder_state_dict_key_layout_matches_hf_decoder():
    from difflet.models.hunyuan_video.vae.modeling_vae import (
        HunyuanVideoVAEDecoderConfig,
        HunyuanVideoVAEDecoderModel,
    )

    cfg = HunyuanVideoVAEDecoderConfig(
        block_out_channels=(32, 32, 32, 32),
        layers_per_block=1,
    )
    keys = set(HunyuanVideoVAEDecoderModel(cfg).state_dict())

    assert "post_quant_conv.weight" in keys
    assert "decoder.conv_in.conv.weight" in keys
    assert "decoder.mid_block.resnets.0.conv1.conv.weight" in keys
    assert "decoder.up_blocks.0.resnets.0.conv1.conv.weight" in keys
    assert "decoder.conv_norm_out.weight" in keys
    assert "decoder.conv_out.conv.weight" in keys
    assert not any(key.startswith("encoder.") for key in keys)
    assert not any(key.startswith("quant_conv.") for key in keys)


def test_hunyuan_video_vae_decoder_tiny_forward_shape():
    from difflet.models.hunyuan_video.vae.modeling_vae import (
        HunyuanVideoVAEDecoderConfig,
        HunyuanVideoVAEDecoderModel,
    )

    cfg = HunyuanVideoVAEDecoderConfig(
        block_out_channels=(32, 32, 32, 32),
        layers_per_block=1,
    )
    model = HunyuanVideoVAEDecoderModel(cfg).eval()
    latents = torch.randn(1, 16, 1, 2, 2)

    with torch.no_grad():
        out = model(latents)

    assert out.shape == (1, 3, 1, 16, 16)


def test_hunyuan_video_vae_repeat_upsampler_matches_diffusers_nearest():
    from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import (
        HunyuanVideoUpsampleCausal3D,
    )

    from difflet.models.hunyuan_video.vae.modeling_vae import _RepeatNearestUpsampleCausal3D

    for factor in [(1, 2, 2), (2, 2, 2)]:
        torch.manual_seed(123)
        original = HunyuanVideoUpsampleCausal3D(
            in_channels=4,
            out_channels=4,
            upsample_factor=factor,
        ).eval()
        replacement = _RepeatNearestUpsampleCausal3D(
            conv=original.conv,
            upsample_factor=factor,
        ).eval()
        x = torch.randn(1, 4, 5, 4, 4)

        with torch.no_grad():
            expected = original(x)
            actual = replacement(x)

        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected)


def test_hunyuan_video_vae_decoder_inference_config_shapes(tmp_path):
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.backends.trainium.hunyuan_video.vae import (
        HunyuanVideoVAEDecoderInferenceConfig,
        ModelWrapperHunyuanVideoVAEDecoder,
    )
    from difflet.utils.diffusers_adapter import load_diffusers_config

    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "AutoencoderKLHunyuanVideo",
                "out_channels": 3,
                "latent_channels": 16,
                "up_block_types": ["HunyuanVideoUpBlock3D"] * 4,
                "block_out_channels": [128, 256, 512, 512],
                "layers_per_block": 2,
                "act_fn": "silu",
                "norm_num_groups": 32,
                "scaling_factor": 0.476986,
                "spatial_compression_ratio": 8,
                "temporal_compression_ratio": 4,
                "mid_block_add_attention": True,
            }
        )
    )
    nc = NeuronConfig(
        batch_size=1,
        tp_degree=1,
        world_size=1,
        torch_dtype=torch.bfloat16,
        skip_sharding=True,
    )
    cfg = HunyuanVideoVAEDecoderInferenceConfig(
        neuron_config=nc,
        load_config=load_diffusers_config(vae_dir),
        height=320,
        width=512,
        num_frames=61,
    )
    assert cfg.latent_frames == 16
    assert cfg.latent_height == 40
    assert cfg.latent_width == 64
    assert cfg.tile_latent_frames == 5
    assert cfg.tile_latent_height == 32
    assert cfg.tile_latent_width == 32

    mw = ModelWrapperHunyuanVideoVAEDecoder(
        config=cfg, model_cls=object, tag="HunyuanVideoVAEDecoderModel"
    )
    inputs = mw.input_generator()
    assert tuple(inputs[0][0].shape) == (1, 16, 5, 32, 32)
    assert inputs[0][0].dtype == torch.bfloat16


def test_hunyuan_video_vae_segment_specs_materialize_norm_conv_boundaries(tmp_path):
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.backends.trainium.hunyuan_video.vae import (
        NeuronHunyuanVideoVAEDecoderApplication,
        HunyuanVideoVAEDecoderInferenceConfig,
    )
    from difflet.utils.diffusers_adapter import load_diffusers_config

    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "AutoencoderKLHunyuanVideo",
                "out_channels": 3,
                "latent_channels": 16,
                "up_block_types": ["HunyuanVideoUpBlock3D"] * 4,
                "block_out_channels": [128, 256, 512, 512],
                "layers_per_block": 2,
                "act_fn": "silu",
                "norm_num_groups": 32,
                "scaling_factor": 0.476986,
                "spatial_compression_ratio": 8,
                "temporal_compression_ratio": 4,
                "mid_block_add_attention": True,
            }
        )
    )
    cfg = HunyuanVideoVAEDecoderInferenceConfig(
        neuron_config=NeuronConfig(
            batch_size=1,
            tp_degree=1,
            world_size=1,
            torch_dtype=torch.bfloat16,
            skip_sharding=True,
        ),
        load_config=load_diffusers_config(vae_dir),
        height=320,
        width=512,
        num_frames=61,
    )
    app = object.__new__(NeuronHunyuanVideoVAEDecoderApplication)
    app.config = cfg

    specs = app._segment_specs()
    names = [spec.name for spec in specs]

    assert names[0] == "body_up2"
    assert names[-2:] == ["final_norm_act", "final_conv_out"]
    assert "up3_r0_norm1_act" in names
    assert "up3_r0_conv1" in names
    assert "up3_r0_shortcut" in names
    assert "up3_r2_conv2" in names
    assert len(specs) == 16

    by_name = {spec.name: spec for spec in specs}
    assert by_name["body_up2"].input_shape == (1, 16, 5, 32, 32)
    assert by_name["up3_r0_norm1_act"].input_shape == (1, 256, 17, 256, 256)
    assert by_name["up3_r1_norm1_act"].input_shape == (1, 128, 17, 256, 256)
    assert by_name["final_conv_out"].input_shape == (1, 128, 17, 256, 256)


def test_hunyuan_video_vae_host_tiling_reconstructs_v0_shape():
    from difflet.backends.trainium.hunyuan_video.vae import (
        NeuronHunyuanVideoVAEDecoderApplication,
    )

    class FakeVAE(NeuronHunyuanVideoVAEDecoderApplication):
        def __init__(self):
            self.dtype = torch.float32
            self.config = SimpleNamespace(
                latent_channels=16,
                temporal_compression_ratio=4,
                spatial_compression_ratio=8,
                tile_latent_frames=5,
                tile_latent_height=32,
                tile_latent_width=32,
                tile_latent_stride_num_frames=3,
                tile_latent_stride_height=24,
                tile_latent_stride_width=24,
                tile_sample_min_num_frames=16,
                tile_sample_stride_num_frames=12,
                tile_sample_min_height=256,
                tile_sample_min_width=256,
                tile_sample_stride_height=192,
                tile_sample_stride_width=192,
            )

        def forward(self, latents):
            batch, _, frames, height, width = latents.shape
            return torch.zeros(
                (
                    batch,
                    3,
                    (frames - 1) * 4 + 1,
                    height * 8,
                    width * 8,
                ),
                dtype=latents.dtype,
            )

    latents = torch.randn(1, 16, 16, 40, 64)
    decoded = FakeVAE().decode(latents, return_dict=False)[0]

    assert decoded.shape == (1, 3, 61, 320, 512)
