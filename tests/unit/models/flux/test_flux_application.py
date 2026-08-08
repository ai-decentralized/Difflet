"""CPU unit tests for difflet.models.flux.application.

Covers the pure parallelism math (get_flux_parallelism_config) and the
create_flux_config factory wiring. The on-disk HF/diffusers config loaders are
replaced with in-memory load hooks so no model checkpoint is required; the
NeuronFluxApplication runtime (compile/load/pipeline) is out of scope.
"""

import importlib
import os
from contextlib import nullcontext
from types import SimpleNamespace

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

    from difflet.pipeline.cache import CacheSession

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
    )
    session = instance._prepare_probe_free_teacache(
        prompt="cat",
        num_inference_steps=6,
        height=768,
        width=512,
    )

    assert isinstance(session, CacheSession)
    assert instance.pipe.teacache_controller is None
    assert session.configuration_source == "teacache_cadence"
    assert session.num_steps == 6
    assert session.runner.policy.calibration.shape_label == "768x512"
    recovery = session.runner.recovery.config
    assert recovery.warmup_steps == 5
    assert recovery.cooldown_steps == 5
    assert recovery.max_consecutive_predictions is None
    assert recovery.recovery_steps == 1
    assert recovery.require_final_anchor is True


def test_application_creates_a_fresh_session_for_each_request(monkeypatch):
    instance = object.__new__(app.NeuronFluxApplication)
    instance._qualified_cache_profile = "configured"
    instance._teacache_cadence = None
    instance._teacache_online_delta_alpha = None
    sessions = [object(), object()]
    monkeypatch.setattr(
        instance,
        "_prepare_cache_session",
        lambda *args, **kwargs: sessions.pop(0),
    )
    received = []
    instance.pipe = lambda **kwargs: received.append(kwargs["cache_session"])

    instance(prompt="first")
    instance(prompt="second")

    assert received[0] is not received[1]
    assert not hasattr(instance, "cache_session")


def test_application_accepts_explicit_session_only_without_configured_cache():
    instance = object.__new__(app.NeuronFluxApplication)
    instance._qualified_cache_profile = None
    instance._teacache_cadence = None
    instance._teacache_online_delta_alpha = None
    received = {}
    instance.pipe = lambda **kwargs: received.update(kwargs)
    session = object()

    instance(prompt="cat", cache_session=session)
    assert received["cache_session"] is session

    instance._qualified_cache_profile = "configured"
    with pytest.raises(ValueError, match="explicit cache_session"):
        instance(prompt="cat", cache_session=object())


def test_flux_pipeline_wraps_explicit_session_without_shared_state(monkeypatch):
    from difflet.pipeline.cache import (
        CacheRunner,
        CacheSession,
        PhasedStaticPolicy,
        TaylorSeerPredictor,
    )

    pipe = object.__new__(app.NeuronFluxPipeline)
    pipe.transformer = SimpleNamespace(
        config=SimpleNamespace(cfg_parallel_enabled=False),
        image_rotary_emb_cache_context=lambda: nullcontext(),
    )
    pipe.teacache_controller = None
    pipe.teacache_probe = None
    captured = []

    def fake_teacache(self, *args, cache_controller=None, **kwargs):
        del self, args, kwargs
        captured.append(cache_controller)
        return "image"

    monkeypatch.setattr(app.NeuronFluxPipeline, "_call_with_teacache", fake_teacache)
    session = CacheSession(
        CacheRunner(
            PhasedStaticPolicy((True, True, True)),
            TaylorSeerPredictor(order=1),
        ),
        num_steps=3,
        configuration_source="test",
    )

    assert pipe(prompt="cat", cache_session=session) == "image"
    assert captured[0].session is session
    assert pipe.teacache_controller is None


def test_qualified_profile_loader_is_wired_into_flux_application(monkeypatch):
    import difflet.pipeline.cache as cache_api

    calls = []

    class FakeProfile:
        generation = {"num_steps": 50, "guidance_scale": 3.5}

        def validate_runtime(self, **kwargs):
            calls.append(kwargs)

    class FakeScheduler:
        config = {}

    class FakePipeline:
        def __init__(self):
            self.vae = SimpleNamespace(decoder=None)
            self.scheduler = FakeScheduler()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(
            self,
            prompt=None,
            *,
            num_inference_steps=50,
            sigmas=None,
            height=None,
            width=None,
            guidance_scale=3.5,
            cache_session=None,
        ):
            del (
                prompt,
                num_inference_steps,
                sigmas,
                height,
                width,
                guidance_scale,
                cache_session,
            )

    monkeypatch.setattr(cache_api, "load_qualified_cache_profile", lambda *args: FakeProfile())
    monkeypatch.setattr(app, "NeuronClipApplication", lambda **kwargs: "clip")
    monkeypatch.setattr(app, "NeuronT5Application", lambda **kwargs: "t5")
    monkeypatch.setattr(app, "NeuronFluxBackboneApplication", lambda **kwargs: "transformer")
    monkeypatch.setattr(app, "NeuronVAEDecoderApplication", lambda **kwargs: "decoder")
    backbone_config = SimpleNamespace(
        cfg_parallel_enabled=False,
        neuron_config=SimpleNamespace(torch_dtype=torch.bfloat16, tp_degree=4),
    )

    instance = app.NeuronFluxApplication(
        model_path="/fake/model",
        text_encoder_config=SimpleNamespace(),
        text_encoder2_config=SimpleNamespace(),
        backbone_config=backbone_config,
        decoder_config=SimpleNamespace(),
        pipeline_class=FakePipeline,
        cache_profile_file="profile.json",
        cache_profile_qualification_file="qualification.json",
        cache_runtime_model_id="black-forest-labs/FLUX.1-dev",
        cache_runtime_model_revision="revision",
    )

    assert isinstance(instance._qualified_cache_profile, FakeProfile)
    assert calls[0]["model_revision"] == "revision"
    assert calls[0]["scheduler_class"] == "FakeScheduler"


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
