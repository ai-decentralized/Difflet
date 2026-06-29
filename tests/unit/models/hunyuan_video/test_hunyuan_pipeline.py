"""CPU coverage for HunyuanVideo pipeline helpers and orchestrator branches.

Focuses on the pure host-side helpers (bundle building, scheduler loading,
dtype/tensor coercion, teacache mod-input selection) and the orchestrator
branches reachable with lightweight fakes, complementing the existing
orchestrator/routing tests.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("DIFFLET_BACKEND", "cpu")

from difflet.models.hunyuan_video import pipeline as pl
from difflet.models.hunyuan_video.application import HunyuanVideoDiTInputBundle


# --------------------------------------------------------------------------- #
# _bundle_from_tensors
# --------------------------------------------------------------------------- #
def test_bundle_from_tensors_fills_defaults():
    latents = torch.zeros(2, 3)
    bundle = pl._bundle_from_tensors(
        latents=latents,
        encoder_hidden_states=torch.zeros(2, 4),
        encoder_attention_mask=torch.ones(2, 4),
        pooled_projections=torch.zeros(2, 5),
        guidance=None,
        guidance_scale=6.0,
        dtype=torch.float32,
    )
    assert isinstance(bundle, HunyuanVideoDiTInputBundle)
    # timestep defaults to zeros sized to the batch
    assert bundle.timestep.shape == (2,)
    # guidance defaults to guidance_scale * 1000
    assert torch.allclose(bundle.guidance, torch.full((2,), 6000.0))


def test_bundle_from_tensors_keeps_explicit_guidance():
    bundle = pl._bundle_from_tensors(
        latents=torch.zeros(1, 3),
        encoder_hidden_states=torch.zeros(1, 4),
        encoder_attention_mask=torch.ones(1, 4),
        pooled_projections=torch.zeros(1, 5),
        guidance=torch.full((1,), 1.5),
        guidance_scale=6.0,
        dtype=torch.float32,
    )
    assert torch.allclose(bundle.guidance, torch.full((1,), 1.5))


def test_bundle_from_tensors_reports_missing():
    with pytest.raises(ValueError, match="encoder_hidden_states"):
        pl._bundle_from_tensors(
            latents=torch.zeros(1, 3),
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            pooled_projections=None,
            guidance=None,
            guidance_scale=6.0,
            dtype=torch.float32,
        )


# --------------------------------------------------------------------------- #
# scheduler helpers
# --------------------------------------------------------------------------- #
def test_load_scheduler_missing_warns_when_requested():
    with tempfile.TemporaryDirectory() as path:
        with pytest.warns(RuntimeWarning, match="scheduler config is missing"):
            scheduler = pl._load_scheduler(path, warn_if_missing=True)
    assert scheduler is None


def test_load_scheduler_missing_silent():
    with tempfile.TemporaryDirectory() as path:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scheduler = pl._load_scheduler(path, warn_if_missing=False)
    assert scheduler is None


def test_load_scheduler_reads_saved_config():
    from diffusers import FlowMatchEulerDiscreteScheduler

    with tempfile.TemporaryDirectory() as path:
        FlowMatchEulerDiscreteScheduler().save_pretrained(os.path.join(path, "scheduler"))
        scheduler = pl._load_scheduler(path, warn_if_missing=True)
    assert type(scheduler).__name__ == "FlowMatchEulerDiscreteScheduler"


def test_orchestrator_timesteps_and_explicit_match_with_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler

    with tempfile.TemporaryDirectory() as path:
        FlowMatchEulerDiscreteScheduler().save_pretrained(os.path.join(path, "scheduler"))
        orch = pl.HunyuanVideoOrchestrator(model_path=path)
        timesteps = orch._timesteps(3, device=torch.device("cpu"))
        assert timesteps.shape == (3,)
        # explicit timesteps that match the scheduler-derived schedule are accepted
        matched = orch._prepare_explicit_timesteps(timesteps, device=torch.device("cpu"))
        assert torch.allclose(matched.float(), timesteps.float())
        # mismatched explicit timesteps are rejected
        with pytest.raises(ValueError, match="do not match the scheduler"):
            orch._prepare_explicit_timesteps(
                torch.tensor([1.0, 2.0, 3.0]), device=torch.device("cpu")
            )


def test_missing_scheduler_message_mentions_path():
    msg = pl._missing_scheduler_message("/models/hv")
    assert "/models/hv" in msg
    assert "scheduler_config.json" in msg


def test_teacache_shape_label():
    assert pl._teacache_shape_label(height=320, width=512, num_frames=61) == "320x512x61"


# --------------------------------------------------------------------------- #
# _teacache_mod_input
# --------------------------------------------------------------------------- #
def _bundle():
    return HunyuanVideoDiTInputBundle(
        hidden_states=torch.ones(1, 2),
        timestep=torch.zeros(1),
        encoder_hidden_states=torch.zeros(1, 2),
        encoder_attention_mask=torch.ones(1, 2),
        pooled_projections=torch.zeros(1, 2),
        guidance=torch.zeros(1),
    )


def test_teacache_mod_input_proxy_source():
    bundle = _bundle()
    out = pl._teacache_mod_input(object(), bundle, source="hidden_states_proxy")
    assert torch.equal(out, bundle.hidden_states)


def test_teacache_mod_input_uses_single_arg_hook():
    class Transformer:
        def teacache_mod_input(self, bundle):
            return bundle.hidden_states + 1.0

    out = pl._teacache_mod_input(Transformer(), _bundle(), source="block0_modulated_input")
    assert torch.allclose(out, torch.full((1, 2), 2.0))


def test_teacache_mod_input_missing_hook_raises():
    with pytest.raises(RuntimeError, match="block-0 modulated input"):
        pl._teacache_mod_input(object(), _bundle(), source="block0_modulated_input")


def test_teacache_mod_input_non_tensor_raises():
    class Transformer:
        def teacache_mod_input(self, bundle):
            return "not a tensor"

    with pytest.raises(TypeError, match="torch.Tensor"):
        pl._teacache_mod_input(Transformer(), _bundle(), source="block0_modulated_input")


# --------------------------------------------------------------------------- #
# _first_tensor
# --------------------------------------------------------------------------- #
def test_first_tensor_variants():
    t = torch.zeros(1)
    assert pl._first_tensor(t) is t
    assert pl._first_tensor({"sample": t}) is t
    assert pl._first_tensor({"other": t}) is t
    assert pl._first_tensor((t, 1)) is t
    assert pl._first_tensor([t]) is t
    assert pl._first_tensor(SimpleNamespace(sample=t)) is t


def test_first_tensor_rejects_unknown():
    with pytest.raises(TypeError, match="tensor-like"):
        pl._first_tensor(object())


# --------------------------------------------------------------------------- #
# _component_dtype / _component_config
# --------------------------------------------------------------------------- #
def test_component_dtype_direct_attribute():
    component = SimpleNamespace(dtype=torch.bfloat16)
    assert pl._component_dtype(component, torch.float32) == torch.bfloat16


def test_component_dtype_from_neuron_config():
    component = SimpleNamespace(
        config=SimpleNamespace(neuron_config=SimpleNamespace(torch_dtype=torch.float16))
    )
    assert pl._component_dtype(component, torch.float32) == torch.float16


def test_component_dtype_fallback():
    assert pl._component_dtype(object(), torch.bfloat16) == torch.bfloat16


def test_component_config():
    component = SimpleNamespace(config="cfg")
    assert pl._component_config(component) == "cfg"
    assert pl._component_config(object()) is None


# --------------------------------------------------------------------------- #
# _batch_timestep
# --------------------------------------------------------------------------- #
def test_batch_timestep_expands_scalar():
    out = pl._batch_timestep(
        torch.tensor(7.0), batch_size=3, device=torch.device("cpu"), dtype=torch.float32
    )
    assert out.shape == (3,)
    assert torch.allclose(out, torch.full((3,), 7.0))


def test_batch_timestep_keeps_batched():
    ts = torch.tensor([1.0, 2.0])
    out = pl._batch_timestep(ts, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    assert out.shape == (2,)


def test_batch_timestep_accepts_python_scalar():
    out = pl._batch_timestep(5, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    assert out.shape == (2,)
    assert out.dtype == torch.float32


# --------------------------------------------------------------------------- #
# Orchestrator branches with lightweight fakes
# --------------------------------------------------------------------------- #
class _FakeTransformer:
    def __init__(self, value=0.25):
        self.dtype = torch.float32
        self.value = value
        self.calls = []

    def __call__(self, bundle):
        self.calls.append(bundle)
        return {"sample": torch.ones_like(bundle.hidden_states) * self.value}


def _orchestrator(**kwargs):
    with tempfile.TemporaryDirectory() as path:
        return pl.HunyuanVideoOrchestrator(model_path=path, **kwargs)


def test_orchestrator_has_runtime_components():
    orch = _orchestrator()
    assert orch.has_runtime_components() is False
    orch_with = _orchestrator(transformer=_FakeTransformer())
    assert orch_with.has_runtime_components() is True


def test_orchestrator_rejects_bad_output_type():
    orch = _orchestrator(transformer=_FakeTransformer())
    with pytest.raises(ValueError, match="output_type"):
        orch(bundle=_make_full_bundle(), output_type="numpy")


def test_orchestrator_denoise_no_scheduler_fallback():
    # With no scheduler dir the scheduler is None; explicit timesteps drive the
    # plain (latents - step) fallback in _scheduler_step.
    transformer = _FakeTransformer(value=0.1)
    orch = _orchestrator(transformer=transformer)
    assert orch.scheduler is None
    out = orch(
        bundle=_make_full_bundle(),
        timesteps=torch.tensor([1000.0, 500.0]),
        return_dict=True,
    )
    assert isinstance(out, pl.HunyuanVideoPipelineOutput)
    assert out.latents.shape == (1, 4)
    assert len(transformer.calls) == 2


def test_orchestrator_returns_trajectory():
    orch = _orchestrator(transformer=_FakeTransformer())
    out = orch(
        bundle=_make_full_bundle(),
        timesteps=torch.tensor([1000.0, 500.0]),
        return_trajectory=True,
    )
    # initial latents + one per step
    assert out.trajectory is not None
    assert len(out.trajectory) == 3


def test_orchestrator_return_tuple():
    orch = _orchestrator(transformer=_FakeTransformer())
    out = orch(
        bundle=_make_full_bundle(),
        timesteps=torch.tensor([1000.0]),
        return_dict=False,
    )
    assert isinstance(out, tuple)
    assert out[0].shape == (1, 4)


def _make_full_bundle():
    return HunyuanVideoDiTInputBundle(
        hidden_states=torch.zeros(1, 4),
        timestep=torch.zeros(1),
        encoder_hidden_states=torch.zeros(1, 2),
        encoder_attention_mask=torch.ones(1, 2),
        pooled_projections=torch.zeros(1, 2),
        guidance=torch.zeros(1),
    )


def test_orchestrator_prepare_explicit_timesteps_scalar_no_scheduler():
    orch = _orchestrator()
    out = orch._prepare_explicit_timesteps(torch.tensor(3.0), device=torch.device("cpu"))
    assert out.shape == (1,)


def test_orchestrator_decode_latents_uses_vae_decode():
    class FakeVAE:
        def __init__(self):
            self.dtype = torch.float32
            self.config = SimpleNamespace(scaling_factor=2.0)
            self.seen = []

        def decode(self, latents, return_dict):
            assert return_dict is False
            self.seen.append(latents)
            return (latents * 10.0,)

    vae = FakeVAE()
    orch = _orchestrator(vae=vae)
    out = orch._decode_latents(torch.ones(1, 3))
    # scaling: latents / 2.0 then *10 in fake decode
    assert torch.allclose(out, torch.full((1, 3), 5.0))


def test_orchestrator_decode_latents_without_vae_raises():
    with tempfile.TemporaryDirectory() as path:
        orch = pl.HunyuanVideoOrchestrator(model_path=path)
        assert orch.vae is None
        with pytest.raises(ValueError, match="VAE decoder"):
            orch._decode_latents(torch.zeros(1, 16, 1, 2, 2))


def test_orchestrator_decode_latents_callable_vae():
    class CallableVAE:
        dtype = torch.float32
        config = SimpleNamespace(scaling_factor=1.0)

        def __call__(self, latents):
            return latents + 1.0

    orch = _orchestrator(vae=CallableVAE())
    out = orch._decode_latents(torch.zeros(1, 3))
    assert torch.allclose(out, torch.ones(1, 3))
