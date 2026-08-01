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
    inst._cache_plan = None
    inst._cache_mask = None
    inst._teacache_cadence = None
    inst._teacache_online_delta_alpha = None

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


def test_probe_free_cache_uses_composable_runtime_and_explicit_recovery(
    monkeypatch,
):
    import types

    from difflet.pipeline.cache import CacheSession, TeaCacheControllerAdapter

    class FakePipeline:
        def __init__(self):
            self.vae = types.SimpleNamespace(decoder=None)
            self.scheduler = types.SimpleNamespace()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(
            self,
            prompt=None,
            *,
            num_inference_steps=28,
            sigmas=None,
            height=None,
            width=None,
        ):
            del prompt, num_inference_steps, sigmas, height, width

    monkeypatch.setattr(app, "NeuronClipApplication", lambda **kwargs: "clip")
    monkeypatch.setattr(app, "NeuronT5Application", lambda **kwargs: "t5")
    monkeypatch.setattr(
        app, "NeuronFluxBackboneApplication", lambda **kwargs: "transformer"
    )
    monkeypatch.setattr(
        app, "NeuronVAEDecoderApplication", lambda **kwargs: "decoder"
    )

    backbone_config = types.SimpleNamespace(cfg_parallel_enabled=False)
    instance = app.NeuronFluxApplication(
        model_path="/fake/model",
        text_encoder_config=types.SimpleNamespace(),
        text_encoder2_config=types.SimpleNamespace(),
        backbone_config=backbone_config,
        decoder_config=types.SimpleNamespace(),
        pipeline_class=FakePipeline,
        teacache_cadence=1,
        cache_recovery_warmup_steps=0,
        cache_recovery_cooldown_steps=2,
        cache_recovery_max_consecutive=1,
        cache_recovery_steps=3,
        cache_require_final_anchor=True,
    )
    instance._prepare_probe_free_teacache(
        prompt="cat",
        num_inference_steps=6,
        height=768,
        width=512,
    )

    adapter = instance.pipe.teacache_controller
    assert isinstance(adapter, TeaCacheControllerAdapter)
    assert isinstance(instance.cache_session, CacheSession)
    assert adapter.session is instance.cache_session
    assert adapter.source == "teacache_cadence"
    assert adapter.num_steps == 6
    assert adapter.runner.policy.calibration.shape_label == "768x512"
    recovery = adapter.runner.recovery.config
    assert recovery.warmup_steps == 0
    assert recovery.cooldown_steps == 2
    assert recovery.max_consecutive_predictions == 1
    assert recovery.recovery_steps == 3
    assert recovery.require_final_anchor is True


def test_static_cache_configuration_installs_resolved_session_and_adapter():
    import types

    from difflet.pipeline.cache import (
        QualityRecoveryConfig,
        ResolvedCacheSession,
        TeaCacheControllerAdapter,
    )

    instance = object.__new__(app.NeuronFluxApplication)
    instance._cache_plan = None
    instance._cache_mask = (True, True, False, True)
    instance._cache_predictor_spec = {
        "type": "taylorseer",
        "order": 1,
        "coord": "index",
    }
    instance._cache_recovery_config = QualityRecoveryConfig(
        require_final_anchor=True
    )
    instance.pipe = types.SimpleNamespace(
        scheduler=types.SimpleNamespace(),
        teacache_controller=None,
    )
    instance._request_identity = lambda *args, **kwargs: (4, 512, 512)

    instance._prepare_cache_session(prompt="cat")

    assert isinstance(instance.cache_session, ResolvedCacheSession)
    assert isinstance(instance.pipe.teacache_controller, TeaCacheControllerAdapter)
    assert instance.pipe.teacache_controller.session is instance.cache_session
    assert instance.cache_session.config.anchor_mask == (True, True, False, True)


def test_request_identity_uses_pipeline_default_and_rejects_zero_steps():
    class FakePipeline:
        def __call__(self, *, num_inference_steps=17, sigmas=None, height=None, width=None):
            del num_inference_steps, sigmas, height, width

    instance = object.__new__(app.NeuronFluxApplication)
    instance.pipe = FakePipeline()
    instance.height = 768
    instance.width = 512

    assert instance._request_identity() == (17, 768, 512)
    assert instance._request_identity(sigmas=[1.0, 0.5]) == (2, 768, 512)
    with pytest.raises(ValueError, match="at least one step"):
        instance._request_identity(num_inference_steps=0)
    with pytest.raises(ValueError, match="at least one step"):
        instance._request_identity(sigmas=[])


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
