"""CPU unit tests for difflet.models.flux.application.

Covers the pure parallelism math (get_flux_parallelism_config) and the
create_flux_config factory wiring. The on-disk HF/diffusers config loaders are
replaced with in-memory load hooks so no model checkpoint is required; the
NeuronFluxApplication runtime (compile/load/pipeline) is out of scope.
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


app = _reload("difflet.models.flux.application")

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


# --------------------------------------------------------------------------
# get_flux_parallelism_config
# --------------------------------------------------------------------------
def test_parallelism_config_plain():
    assert app.get_flux_parallelism_config(8) == 8


def test_parallelism_config_context_parallel():
    assert app.get_flux_parallelism_config(8, cp_degree=2) == 16


def test_parallelism_config_cfg_parallel():
    assert app.get_flux_parallelism_config(4, cfg_parallel_enabled=True) == 8


def test_parallelism_config_mutual_exclusive_raises():
    with pytest.raises(ValueError):
        app.get_flux_parallelism_config(4, cp_degree=2, cfg_parallel_enabled=True)


# --------------------------------------------------------------------------
# create_flux_config (with in-memory config loaders)
# --------------------------------------------------------------------------
_CLIP_ATTRS = dict(
    _name_or_path="clip",
    architectures=["CLIPTextModel"],
    attention_dropout=0.0,
    bos_token_id=0,
    dropout=0.0,
    eos_token_id=2,
    hidden_act="gelu",
    hidden_size=32,
    initializer_factor=1.0,
    initializer_range=0.02,
    intermediate_size=64,
    layer_norm_eps=1e-5,
    max_position_embeddings=16,
    model_type="clip_text_model",
    num_attention_heads=4,
    num_hidden_layers=2,
    pad_token_id=1,
    projection_dim=32,
    transformers_version="4.0",
    vocab_size=50,
)

_T5_ATTRS = dict(
    vocab_size=50,
    d_model=32,
    d_kv=8,
    d_ff=64,
    num_layers=2,
    num_decoder_layers=2,
    num_heads=4,
    relative_attention_num_buckets=8,
    relative_attention_max_distance=128,
    dropout_rate=0.0,
    layer_norm_epsilon=1e-6,
    initializer_factor=1.0,
    feed_forward_proj="relu",
    is_encoder_decoder=False,
    use_cache=False,
    pad_token_id=0,
    eos_token_id=1,
    classifier_dropout=0.0,
)

_BACKBONE_ATTRS = dict(
    attention_head_dim=8,
    guidance_embeds=False,
    in_channels=4,
    joint_attention_dim=32,
    num_attention_heads=2,
    num_layers=1,
    num_single_layers=0,
    patch_size=1,
    pooled_projection_dim=8,
)

_VAE_ATTRS = dict(
    latent_channels=4,
    out_channels=3,
    up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
    block_out_channels=[8, 16],
    layers_per_block=1,
    norm_num_groups=2,
    act_fn="silu",
    mid_block_add_attention=True,
)


def _fake_pretrained_loader(path):
    attrs = _CLIP_ATTRS if path.endswith("text_encoder") else _T5_ATTRS

    def load_config(self):
        for k, v in attrs.items():
            setattr(self, k, v)

    return load_config


def _fake_diffusers_loader(path):
    attrs = _BACKBONE_ATTRS if path.endswith("transformer") else _VAE_ATTRS

    def load_config(self):
        for k, v in attrs.items():
            setattr(self, k, v)

    return load_config


@pytest.fixture
def patched_loaders(monkeypatch):
    monkeypatch.setattr(app, "load_pretrained_config", _fake_pretrained_loader)
    monkeypatch.setattr(app, "load_diffusers_config", _fake_diffusers_loader)
    yield


def test_create_flux_config_returns_four_configs(patched_loaders):
    clip_cfg, t5_cfg, backbone_cfg, decoder_cfg = app.create_flux_config(
        model_path="/fake/model",
        world_size=2,
        backbone_tp_degree=2,
        dtype=torch.float32,
        height=64,
        width=64,
    )
    assert clip_cfg.hidden_size == 32
    assert t5_cfg.d_model == 32
    assert backbone_cfg.in_channels == 4
    assert backbone_cfg.height == 64
    # vae_scale_factor is propagated onto the backbone config
    assert backbone_cfg.vae_scale_factor == decoder_cfg.vae_scale_factor
    # default (non-inpaint) wires transformer_in_channels onto the decoder
    assert decoder_cfg.transformer_in_channels == backbone_cfg.in_channels
    # neuron-config tp wiring
    assert clip_cfg.neuron_config.tp_degree == 1
    assert t5_cfg.neuron_config.tp_degree == 2
    assert backbone_cfg.neuron_config.tp_degree == 2


def test_create_flux_config_inpaint_branch(patched_loaders):
    _, _, _, decoder_cfg = app.create_flux_config(
        model_path="/fake/model",
        world_size=1,
        backbone_tp_degree=1,
        dtype=torch.float32,
        height=32,
        width=32,
        inpaint=True,
    )
    # inpaint path does not set transformer_in_channels
    assert not hasattr(decoder_cfg, "transformer_in_channels")


def test_application_components_and_call():
    import types

    # Build a bare instance (skip the heavy from_pretrained __init__) and drive
    # the pure components()/__call__ wiring.
    inst = app.NeuronFluxApplication.__new__(app.NeuronFluxApplication)
    inst.pipe = types.SimpleNamespace(
        text_encoder="te",
        text_encoder_2="te2",
        transformer="tr",
        vae=types.SimpleNamespace(decoder="dec"),
    )
    inst.teacache_probe = None

    specs = inst.components()
    assert [s.name for s in specs] == [
        "text_encoder",
        "text_encoder_2",
        "transformer",
        "decoder",
    ]

    # With a teacache probe mounted, an extra component spec is appended.
    inst.teacache_probe = "probe"
    spec_names = [s.name for s in inst.components()]
    assert "teacache_probe" in spec_names

    # __call__ delegates straight to the underlying pipeline.
    seen = {}

    def fake_pipe(*args, **kwargs):
        seen["args"] = (args, kwargs)
        return "image"

    inst.pipe = fake_pipe
    assert inst(7, prompt="cat") == "image"
    assert seen["args"] == ((7,), {"prompt": "cat"})


def test_create_flux_config_threads_parallel_flags(patched_loaders):
    _, _, backbone_cfg, _ = app.create_flux_config(
        model_path="/fake/model",
        world_size=2,
        backbone_tp_degree=1,
        dtype=torch.float32,
        height=64,
        width=64,
        context_parallel_enabled=True,
        cp_mode="ring",
    )
    assert backbone_cfg.context_parallel_enabled is True
    assert backbone_cfg.cp_mode == "ring"
