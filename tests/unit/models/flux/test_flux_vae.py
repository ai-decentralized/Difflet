"""CPU unit tests for difflet.models.flux.vae.modeling_vae.

PatchedGroupNorm is pure torch (no backend needed); we check numeric parity
against nn.GroupNorm. VAEDecoderInferenceConfig is exercised for its schema and
derived decoder_config / vae_scale_factor.
"""

import importlib
import os

import pytest

_PREV_BACKEND = os.environ.get("DIFFLET_BACKEND")
os.environ["DIFFLET_BACKEND"] = "cpu"
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

import torch  # noqa: E402


def _reload(name):
    return importlib.reload(importlib.import_module(name))


import difflet.backends.trainium.core.config as _config_mod  # noqa: E402
vae = _reload("difflet.models.flux.vae.modeling_vae")

NeuronConfig = _config_mod.NeuronConfig

# Restore process-wide backend env so collecting other (trainium-only) test
# modules in the same session is unaffected.
if _PREV_BACKEND is None:
    os.environ.pop("DIFFLET_BACKEND", None)
else:
    os.environ["DIFFLET_BACKEND"] = _PREV_BACKEND


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield


def test_patched_group_norm_matches_reference():
    pg = vae.PatchedGroupNorm(num_groups=2, num_channels=4)
    ref = torch.nn.GroupNorm(num_groups=2, num_channels=4)
    x = torch.randn(2, 4, 5, 5)
    assert torch.allclose(pg(x), ref(x), atol=1e-5)


def test_patched_group_norm_bf16_input_keeps_dtype():
    pg = vae.PatchedGroupNorm(num_groups=2, num_channels=4)
    x = torch.randn(1, 4, 3, 3, dtype=torch.bfloat16)
    out = pg(x)
    assert out.dtype == torch.bfloat16
    # weights are stored in float32 internally
    assert pg.weight.dtype == torch.float32


def test_patched_group_norm_invalid_groups_raises():
    with pytest.raises(ValueError):
        vae.PatchedGroupNorm(num_groups=3, num_channels=4)


def test_patched_group_norm_no_affine():
    pg = vae.PatchedGroupNorm(num_groups=2, num_channels=4, affine=False)
    assert pg.weight is None
    assert pg.bias is None
    out = pg(torch.randn(1, 4, 2, 2))
    assert out.shape == (1, 4, 2, 2)


def test_patched_group_norm_extra_repr():
    pg = vae.PatchedGroupNorm(num_groups=2, num_channels=4)
    rep = pg.extra_repr()
    assert "eps=" in rep and "affine=" in rep


def test_patched_group_norm_reset_parameters():
    pg = vae.PatchedGroupNorm(num_groups=2, num_channels=4)
    pg.weight.data.fill_(3.0)
    pg.bias.data.fill_(5.0)
    pg.reset_parameters()
    assert torch.all(pg.weight == 1.0)
    assert torch.all(pg.bias == 0.0)


def _vae_config(**overrides):
    nc = NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.float32)
    kwargs = dict(
        neuron_config=nc,
        latent_channels=4,
        out_channels=3,
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=[8, 16],
        layers_per_block=1,
        norm_num_groups=2,
        act_fn="silu",
        mid_block_add_attention=True,
        height=64,
        width=64,
    )
    kwargs.update(overrides)
    return vae.VAEDecoderInferenceConfig(**kwargs)


def test_vae_config_required_attributes_and_decoder_config():
    cfg = _vae_config()
    required = cfg.get_required_attributes()
    for attr in required:
        assert hasattr(cfg, attr)
    assert cfg.decoder_config["in_channels"] == 4
    assert cfg.decoder_config["out_channels"] == 3
    assert cfg.decoder_config["block_out_channels"] == [8, 16]


def test_vae_config_vae_scale_factor():
    cfg = _vae_config()
    # 2 ** (len(block_out_channels) - 1)
    assert cfg.vae_scale_factor == 2


def test_tiny_decoder_config_and_scale_factor():
    config = vae.get_decoder_config(
        vae.DecoderTiny,
        {
            "latent_channels": 16,
            "out_channels": 3,
            "num_decoder_blocks": [3, 3, 3, 1],
            "decoder_block_out_channels": [64, 64, 64, 64],
        },
        height=1024,
        width=1024,
    )
    assert config["in_channels"] == 16
    assert config["num_blocks"] == [3, 3, 3, 1]
    assert vae.get_vae_scale_factor(vae.DecoderTiny, config) == 8


def test_vae_config_missing_attribute_raises():
    nc = NeuronConfig(tp_degree=1, world_size=1, torch_dtype=torch.float32)
    with pytest.raises((AssertionError, AttributeError, KeyError)):
        vae.VAEDecoderInferenceConfig(neuron_config=nc, latent_channels=4)


def test_vae_convert_hf_to_neuron_state_dict_strips_decoder_prefix():
    cfg = _vae_config()
    sd = {"decoder.conv_in.weight": torch.randn(2, 2)}
    out = vae.NeuronVAEDecoderApplication.convert_hf_to_neuron_state_dict(dict(sd), cfg)
    assert "conv_in.weight" in out
    assert vae.NeuronVAEDecoderApplication.update_state_dict_for_tied_weights({}) is None
