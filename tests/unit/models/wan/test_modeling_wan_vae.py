"""Unit tests for difflet.models.wan.vae.modeling_vae (W3c)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

import pytest
import torch


def test_modeling_vae_imports_only_from_allowed_modules():
    src = (_REPO_ROOT / "difflet/models/wan/vae/modeling_vae.py").read_text()
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

    assert not offending, f"forbidden imports in modeling_vae.py: {offending}"


def test_wan_vae_decoder_config_defaults_match_wan22():
    from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig

    cfg = WanVAEDecoderConfig()
    assert cfg.base_dim == 96
    assert cfg.decoder_base_dim == 96
    assert cfg.z_dim == 16
    assert cfg.dim_mult == [1, 2, 4, 4]
    assert cfg.num_res_blocks == 2
    assert cfg.attn_scales == []
    assert cfg.temperal_downsample == [False, True, True]
    assert cfg.temperal_upsample == [True, True, False]
    assert cfg.scale_factor_temporal == 4
    assert cfg.scale_factor_spatial == 8


def test_wan_vae_decoder_config_from_diffusers_dict_filters_unknown_keys():
    from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig

    raw = {
        "base_dim": 96,
        "z_dim": 16,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "_class_name": "AutoencoderKLWan",
        "_diffusers_version": "0.38.0",
    }
    cfg = WanVAEDecoderConfig.from_diffusers_dict(raw)
    assert cfg.base_dim == 96
    assert cfg.z_dim == 16
    assert cfg.dim_mult == [1, 2, 4, 4]


def test_wan_vae_decoder_config_rejects_unsupported_modes():
    from difflet.models.wan.vae.modeling_vae import WanVAEDecoderConfig

    with pytest.raises(NotImplementedError, match="is_residual"):
        WanVAEDecoderConfig(is_residual=True)
    with pytest.raises(NotImplementedError, match="patchified"):
        WanVAEDecoderConfig(patch_size=2)
    with pytest.raises(ValueError, match="temperal_downsample"):
        WanVAEDecoderConfig(dim_mult=[1, 2, 4], temperal_downsample=[True])


def test_modeling_vae_classes_are_importable():
    from difflet.models.wan.vae import modeling_vae

    expected = [
        "WanCausalConv3d",
        "WanDecoder3d",
        "WanRMSNorm",
        "WanVAEDecoderConfig",
        "WanVAEDecoderModel",
    ]
    for name in expected:
        cls = getattr(modeling_vae, name)
        assert inspect.isclass(cls), f"{name} should be a class"


def test_wan_vae_decoder_state_dict_key_layout_matches_hf_decoder():
    from difflet.models.wan.vae.modeling_vae import (
        WanVAEDecoderConfig,
        WanVAEDecoderModel,
    )

    cfg = WanVAEDecoderConfig(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[True],
    )
    keys = set(WanVAEDecoderModel(cfg).state_dict())

    assert "post_quant_conv.weight" in keys
    assert "decoder.conv_in.weight" in keys
    assert "decoder.mid_block.resnets.0.conv1.weight" in keys
    assert "decoder.up_blocks.0.resnets.0.conv1.weight" in keys
    assert "decoder.norm_out.gamma" in keys
    assert "decoder.conv_out.weight" in keys
    assert not any(key.startswith("encoder.") for key in keys)
    assert not any(key.startswith("quant_conv.") for key in keys)


def test_wan_vae_decoder_tiny_forward_uses_causal_temporal_decode():
    from difflet.models.wan.vae.modeling_vae import (
        WanVAEDecoderConfig,
        WanVAEDecoderModel,
    )

    cfg = WanVAEDecoderConfig(
        base_dim=8,
        z_dim=4,
        dim_mult=[1, 2],
        num_res_blocks=1,
        temperal_downsample=[True],
    )
    model = WanVAEDecoderModel(cfg).eval()
    latents = torch.randn(1, 4, 2, 4, 4)

    with torch.no_grad():
        out = model(latents)

    # Two latent frames with one temporal upsample stage decode to 1 + 2 frames.
    assert out.shape == (1, 3, 3, 8, 8)
    assert out.min() >= -1.0
    assert out.max() <= 1.0


def test_wan_vae_decoder_inference_config_shapes():
    from difflet.backends.trainium.wan.vae import (
        ModelWrapperWanVAEDecoder,
        WanVAEDecoderInferenceConfig,
    )
    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.utils.diffusers_adapter import load_diffusers_config

    snap = (
        "/home/ubuntu/.cache/huggingface/hub/"
        "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/"
        "snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7"
    )
    if not Path(f"{snap}/vae/config.json").exists():
        pytest.skip("Wan2.2 VAE config snapshot not present")

    nc = NeuronConfig(
        batch_size=1,
        tp_degree=1,
        world_size=1,
        torch_dtype=torch.bfloat16,
        skip_sharding=True,
    )
    cfg = WanVAEDecoderInferenceConfig(
        neuron_config=nc,
        load_config=load_diffusers_config(f"{snap}/vae"),
        height=480,
        width=832,
        num_frames=9,
    )
    assert cfg.latent_frames == 3
    assert cfg.latent_height == 60
    assert cfg.latent_width == 104

    mw = ModelWrapperWanVAEDecoder(
        config=cfg, model_cls=object, tag="WanVAEDecoderModel"
    )
    inputs = mw.input_generator()
    assert tuple(inputs[0][0].shape) == (1, 16, 3, 60, 104)
    assert inputs[0][0].dtype == torch.bfloat16
