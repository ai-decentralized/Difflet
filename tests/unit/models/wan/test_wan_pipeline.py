"""Branch coverage for difflet.models.wan.pipeline helpers + orchestrator.

Complements test_wan_pipeline_orchestrator.py by hitting the module-level pure
helpers (_first_tensor / _component_dtype / _batch_timestep / _read_boundary_ratio
/ _load_scheduler) and the orchestrator branches reachable without real weights
(scheduler-backed denoise, latent validation, prompt/embeds edge cases).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from difflet.models.wan import pipeline as pl
from difflet.models.wan.pipeline import WanOrchestrator, WanPipelineOutput


class FakeTransformer:
    def __init__(self, *, bias=0.0):
        self.dtype = torch.float32
        self.bias = bias
        self.calls = []

    def __call__(self, hidden_states, timestep, encoder_hidden_states):
        self.calls.append(timestep.detach().clone())
        return torch.ones_like(hidden_states) * self.bias


# ---------------------------------------------------------------------------
# Module-level helpers


def test_first_tensor_variants():
    t = torch.zeros(2)
    assert pl._first_tensor(t) is t
    assert torch.equal(pl._first_tensor((t, 1)), t)
    assert torch.equal(pl._first_tensor([t]), t)
    assert torch.equal(pl._first_tensor(SimpleNamespace(last_hidden_state=t)), t)
    assert torch.equal(pl._first_tensor(SimpleNamespace(sample=t)), t)
    with pytest.raises(TypeError):
        pl._first_tensor(object())


def test_component_dtype_from_component_attr():
    comp = SimpleNamespace(dtype=torch.float16)
    assert pl._component_dtype(comp, torch.float32) is torch.float16


def test_component_dtype_from_neuron_config():
    comp = SimpleNamespace(
        config=SimpleNamespace(neuron_config=SimpleNamespace(torch_dtype=torch.bfloat16))
    )
    assert pl._component_dtype(comp, torch.float32) is torch.bfloat16


def test_component_dtype_fallback():
    assert pl._component_dtype(object(), torch.float32) is torch.float32
    assert pl._component_config(object()) is None


def test_batch_timestep_scalar_and_tensor():
    out = pl._batch_timestep(5, 3, torch.device("cpu"), torch.float32)
    assert out.shape == (3,)
    assert out.dtype == torch.float32
    already = torch.tensor([1.0, 2.0])
    out2 = pl._batch_timestep(already, 5, torch.device("cpu"), torch.float32)
    assert out2.shape == (2,)


def test_read_boundary_ratio_present_and_absent(tmp_path):
    assert pl._read_boundary_ratio(str(tmp_path)) is None
    (tmp_path / "model_index.json").write_text(json.dumps({"boundary_ratio": 0.875}))
    assert pl._read_boundary_ratio(str(tmp_path)) == pytest.approx(0.875)


def test_read_boundary_ratio_null_value(tmp_path):
    (tmp_path / "model_index.json").write_text(json.dumps({"boundary_ratio": None}))
    assert pl._read_boundary_ratio(str(tmp_path)) is None


def test_load_scheduler_missing_returns_none(tmp_path):
    assert pl._load_scheduler(str(tmp_path)) is None


def test_has_wan_components_helper(tmp_path):
    app = SimpleNamespace(pipeline=WanOrchestrator(model_path=str(tmp_path)))
    assert pl.has_wan_components(app) is False
    app2 = SimpleNamespace(
        pipeline=WanOrchestrator(model_path=str(tmp_path), transformer=FakeTransformer())
    )
    assert pl.has_wan_components(app2) is True
    assert pl.has_wan_components(SimpleNamespace(pipeline=None)) is False


# ---------------------------------------------------------------------------
# Orchestrator branches


def test_select_guidance_scale_static_and_boundary():
    # No boundary → base scale.
    assert WanOrchestrator._select_guidance_scale(
        torch.tensor(500.0), None, 3.0, 5.0
    ) == 3.0
    # Below boundary with guidance_scale_2 → secondary scale.
    assert WanOrchestrator._select_guidance_scale(
        torch.tensor(100.0), 500.0, 3.0, 5.0
    ) == 5.0
    # Above boundary → base scale even with guidance_scale_2.
    assert WanOrchestrator._select_guidance_scale(
        torch.tensor(900.0), 500.0, 3.0, 5.0
    ) == 3.0


def test_prepare_latents_validates_provided_shape(tmp_path):
    orch = WanOrchestrator(model_path=str(tmp_path), dtype=torch.float32)
    good = torch.zeros((1, 16, 3, 60, 104))
    assert orch.prepare_latents(batch_size=1, height=480, width=832, num_frames=9, latents=good) is good
    with pytest.raises(ValueError, match="Expected latents shape"):
        orch.prepare_latents(batch_size=1, latents=torch.zeros((1, 2, 3)))


def test_encode_prompt_passthrough_and_errors(tmp_path):
    orch = WanOrchestrator(model_path=str(tmp_path), dtype=torch.float32)
    embeds = torch.ones((1, 4, 8))
    assert orch.encode_prompt(prompt_embeds=embeds) is embeds
    assert orch.encode_prompt() is None
    with pytest.raises(ValueError, match="no Wan text_encoder"):
        orch.encode_prompt(input_ids=torch.zeros((1, 3), dtype=torch.int64))


def test_call_rejects_unknown_output_type(tmp_path):
    orch = WanOrchestrator(model_path=str(tmp_path), dtype=torch.float32)
    with pytest.raises(ValueError, match="output_type"):
        orch(prompt_embeds=torch.ones((1, 4, 8)), output_type="np")


def test_call_requires_prompt_embeds_when_transformer_active(tmp_path):
    orch = WanOrchestrator(
        model_path=str(tmp_path), dtype=torch.float32, transformer=FakeTransformer()
    )
    with pytest.raises(ValueError, match="requires prompt_embeds"):
        orch(latents=torch.zeros((1, 16, 1, 1, 1)), height=8, width=8, num_frames=1)


def test_pt_output_requires_vae(tmp_path):
    orch = WanOrchestrator(model_path=str(tmp_path), dtype=torch.float32)
    with pytest.raises(ValueError, match="VAE decoder"):
        orch(
            prompt_embeds=torch.ones((1, 4, 8)),
            latents=torch.zeros((1, 16, 1, 1, 1)),
            height=8,
            width=8,
            num_frames=1,
            output_type="pt",
        )


def test_maybe_init_teacache_without_calibration(tmp_path):
    orch = WanOrchestrator(model_path=str(tmp_path), dtype=torch.float32)
    assert orch._maybe_init_teacache() is False


def test_decode_latents_without_vae_stats(tmp_path):
    class _VAE:
        dtype = torch.float32
        config = SimpleNamespace()  # no latents_mean / latents_std

        def __call__(self, latents):
            return latents

    orch = WanOrchestrator(model_path=str(tmp_path), dtype=torch.float32, vae_decoder=_VAE())
    latents = torch.ones((1, 16, 1, 1, 1))
    out = orch._decode_latents(latents)
    # No normalization applied → input passes through unchanged.
    assert torch.allclose(out, latents)


def test_denoise_with_real_unipc_scheduler(tmp_path):
    sd = tmp_path / "scheduler"
    sd.mkdir()
    (sd / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "UniPCMultistepScheduler", "num_train_timesteps": 1000})
    )
    transformer = FakeTransformer(bias=0.1)
    orch = WanOrchestrator(
        model_path=str(tmp_path),
        dtype=torch.float32,
        transformer=transformer,
        boundary_ratio=None,
    )
    assert orch.scheduler is not None
    out = orch(
        prompt_embeds=torch.ones((1, 5, 8)),
        latents=torch.zeros((1, 16, 1, 1, 1)),
        height=8,
        width=8,
        num_frames=1,
        num_inference_steps=3,
        output_type="latent",
    )
    assert isinstance(out, WanPipelineOutput)
    # 3 scheduler timesteps → 3 transformer evaluations (guidance_scale defaults to 1).
    assert len(transformer.calls) == 3
