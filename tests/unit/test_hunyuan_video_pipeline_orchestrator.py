import json
from types import SimpleNamespace

import pytest
import torch

from nova.models.hunyuan_video.application import HunyuanVideoDiTInputBundle
from nova.models.hunyuan_video.pipeline import HunyuanVideoOrchestrator, HunyuanVideoPipelineOutput
from nova.pipeline.teacache import TeaCacheCalibration


class FakeTransformer:
    def __init__(self, *, value: float = 0.25):
        self.dtype = torch.float32
        self.value = value
        self.calls = []

    def __call__(self, bundle: HunyuanVideoDiTInputBundle):
        self.calls.append(bundle)
        return {"sample": torch.ones_like(bundle.hidden_states) * self.value}


class FakeTeaCacheTransformer(FakeTransformer):
    def __init__(self, *, value: float = 0.25):
        super().__init__(value=value)
        self.mod_input_calls = []

    def teacache_mod_input(self, bundle: HunyuanVideoDiTInputBundle):
        self.mod_input_calls.append(bundle)
        return bundle.hidden_states + 0.5


class FakeScheduler:
    def __init__(self):
        self.config = SimpleNamespace()
        self.timesteps = None
        self.steps = []

    def set_timesteps(self, *, sigmas, device):
        steps = max(len(sigmas), 1)
        self.timesteps = torch.linspace(1000.0, 500.0, steps=steps, device=device)

    def step(self, noise_pred, timestep, latents, return_dict):
        assert return_dict is False
        self.steps.append((noise_pred.detach().clone(), timestep.detach().clone()))
        return (latents - noise_pred,)


class FakeVAE:
    def __init__(self):
        self.dtype = torch.float32
        self.config = SimpleNamespace(scaling_factor=2.0)
        self.inputs = []

    def decode(self, latents, return_dict):
        assert return_dict is False
        self.inputs.append(latents.detach().clone())
        return (latents[:, :3],)


def _bundle(latents: torch.Tensor | None = None) -> HunyuanVideoDiTInputBundle:
    latents = torch.zeros((1, 16, 2, 2, 2), dtype=torch.float32) if latents is None else latents
    return HunyuanVideoDiTInputBundle(
        hidden_states=latents,
        timestep=torch.zeros([1], dtype=torch.float32),
        encoder_hidden_states=torch.ones((1, 4, 8), dtype=torch.float32),
        encoder_attention_mask=torch.ones((1, 4), dtype=torch.int64),
        pooled_projections=torch.ones((1, 5), dtype=torch.float32),
        guidance=torch.ones([1], dtype=torch.float32) * 6000.0,
    )


def test_hunyuan_orchestrator_runs_bundle_denoise_with_fallback_scheduler(tmp_path):
    transformer = FakeTransformer(value=0.25)
    with pytest.warns(RuntimeWarning, match="scheduler_config.json"):
        pipeline = HunyuanVideoOrchestrator(
            model_path=str(tmp_path),
            transformer=transformer,
            dtype=torch.float32,
        )

    output = pipeline(
        bundle=_bundle(),
        timesteps=torch.tensor([1000.0, 500.0]),
        return_trajectory=True,
    )

    assert isinstance(output, HunyuanVideoPipelineOutput)
    assert torch.allclose(output.latents, torch.full((1, 16, 2, 2, 2), -0.25))
    assert len(transformer.calls) == 2
    assert transformer.calls[0].timestep.shape == (1,)
    assert transformer.calls[0].encoder_attention_mask.dtype == torch.int64
    assert output.trajectory is not None
    assert len(output.trajectory) == 3


def test_hunyuan_orchestrator_rejects_implicit_timesteps_without_scheduler(tmp_path):
    transformer = FakeTransformer(value=0.25)
    with pytest.warns(RuntimeWarning, match="hunyuan_video_cache_dit_inputs.py"):
        pipeline = HunyuanVideoOrchestrator(
            model_path=str(tmp_path),
            transformer=transformer,
            dtype=torch.float32,
        )

    with pytest.raises(ValueError, match="FlowMatchEulerDiscreteScheduler"):
        pipeline(bundle=_bundle(), num_inference_steps=2)


def test_hunyuan_orchestrator_uses_diffusers_style_scheduler(tmp_path):
    transformer = FakeTransformer(value=0.5)
    scheduler = FakeScheduler()
    pipeline = HunyuanVideoOrchestrator(
        model_path=str(tmp_path),
        transformer=transformer,
        scheduler=scheduler,
        dtype=torch.float32,
    )

    output = pipeline(bundle=_bundle(), num_inference_steps=2)

    assert torch.allclose(output.latents, torch.full((1, 16, 2, 2, 2), -1.0))
    assert len(scheduler.steps) == 2
    assert scheduler.steps[0][1].item() == pytest.approx(1000.0)


def test_hunyuan_orchestrator_teacache_requires_calibration(tmp_path):
    transformer = FakeTransformer(value=1.0)

    with pytest.warns(RuntimeWarning, match="scheduler_config.json"):
        with pytest.raises(FileNotFoundError, match="calibrate_teacache.py"):
            HunyuanVideoOrchestrator(
                model_path=str(tmp_path),
                transformer=transformer,
                dtype=torch.float32,
                teacache_speedup=1.5,
            )


def test_hunyuan_orchestrator_teacache_skips_full_transformer_calls(tmp_path):
    transformer = FakeTeaCacheTransformer(value=1.0)
    calibration = TeaCacheCalibration(
        model="hunyuan_video",
        shape_label="320x512x61",
        num_steps=4,
        poly_coef=(0.0,),
        threshold=1.0,
        warmup_steps=1,
        cooldown_steps=0,
        target_speedup=1.5,
        fit_r2=0.95,
        n_samples=32,
        mod_input_source="block0_modulated_input",
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration.to_dict()), encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="scheduler_config.json"):
        pipeline = HunyuanVideoOrchestrator(
            model_path=str(tmp_path),
            transformer=transformer,
            dtype=torch.float32,
            teacache_speedup=1.5,
            teacache_calibration_path=str(calibration_path),
        )

    output = pipeline(
        bundle=_bundle(),
        timesteps=torch.tensor([1000.0, 900.0, 800.0, 700.0]),
        return_trajectory=True,
    )

    assert torch.allclose(output.latents, torch.full((1, 16, 2, 2, 2), -1.0))
    assert len(transformer.calls) == 2
    assert len(transformer.mod_input_calls) == 4
    assert pipeline.teacache_controller.stats()["full_steps"] == 2
    assert pipeline.teacache_controller.stats()["skipped_steps"] == 2
    assert output.trajectory is not None
    assert len(output.trajectory) == 5


def test_hunyuan_orchestrator_rejects_teacache_speedup_above_calibration(tmp_path):
    transformer = FakeTransformer(value=1.0)
    calibration = TeaCacheCalibration(
        model="hunyuan_video",
        shape_label="320x512x61",
        num_steps=4,
        poly_coef=(0.0,),
        threshold=1.0,
        target_speedup=1.3,
        mod_input_source="hidden_states_proxy",
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration.to_dict()), encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="scheduler_config.json"):
        with pytest.raises(ValueError, match="lower than requested"):
            HunyuanVideoOrchestrator(
                model_path=str(tmp_path),
                transformer=transformer,
                dtype=torch.float32,
                teacache_speedup=1.5,
                teacache_calibration_path=str(calibration_path),
            )


def test_hunyuan_orchestrator_teacache_requires_mod_input_hook(tmp_path):
    transformer = FakeTransformer(value=1.0)
    calibration = TeaCacheCalibration(
        model="hunyuan_video",
        shape_label="320x512x61",
        num_steps=4,
        poly_coef=(0.0,),
        threshold=1.0,
        target_speedup=1.5,
        mod_input_source="block0_modulated_input",
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration.to_dict()), encoding="utf-8")

    with pytest.warns(RuntimeWarning, match="scheduler_config.json"):
        pipeline = HunyuanVideoOrchestrator(
            model_path=str(tmp_path),
            transformer=transformer,
            dtype=torch.float32,
            teacache_speedup=1.5,
            teacache_calibration_path=str(calibration_path),
        )

    with pytest.raises(RuntimeError, match="teacache_mod_input"):
        pipeline(
            bundle=_bundle(),
            timesteps=torch.tensor([1000.0, 900.0, 800.0, 700.0]),
        )


def test_hunyuan_orchestrator_initializes_scheduler_for_explicit_timesteps(tmp_path):
    transformer = FakeTransformer(value=0.5)
    scheduler = FakeScheduler()
    pipeline = HunyuanVideoOrchestrator(
        model_path=str(tmp_path),
        transformer=transformer,
        scheduler=scheduler,
        dtype=torch.float32,
    )

    output = pipeline(bundle=_bundle(), timesteps=torch.tensor([1000.0, 500.0]))

    assert torch.allclose(output.latents, torch.full((1, 16, 2, 2, 2), -1.0))
    assert len(scheduler.steps) == 2
    assert scheduler.steps[1][1].item() == pytest.approx(500.0)


def test_hunyuan_orchestrator_rejects_explicit_timesteps_outside_m3_schedule(tmp_path):
    transformer = FakeTransformer(value=0.5)
    scheduler = FakeScheduler()
    pipeline = HunyuanVideoOrchestrator(
        model_path=str(tmp_path),
        transformer=transformer,
        scheduler=scheduler,
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="Explicit HunyuanVideo timesteps"):
        pipeline(bundle=_bundle(), timesteps=torch.tensor([999.0, 500.0]))


def test_hunyuan_orchestrator_builds_bundle_from_named_tensors(tmp_path):
    transformer = FakeTransformer(value=1.0)
    scheduler = FakeScheduler()
    pipeline = HunyuanVideoOrchestrator(
        model_path=str(tmp_path),
        transformer=transformer,
        scheduler=scheduler,
        dtype=torch.float32,
    )
    bundle = _bundle()

    pipeline(
        latents=bundle.hidden_states,
        timesteps=torch.tensor([1000.0]),
        encoder_hidden_states=bundle.encoder_hidden_states,
        encoder_attention_mask=bundle.encoder_attention_mask,
        pooled_projections=bundle.pooled_projections,
        guidance=bundle.guidance,
    )

    assert len(transformer.calls) == 1
    assert torch.equal(transformer.calls[0].guidance, bundle.guidance)


def test_hunyuan_orchestrator_decodes_pt_output_with_hf_vae_contract(tmp_path):
    vae = FakeVAE()
    pipeline = HunyuanVideoOrchestrator(
        model_path=str(tmp_path),
        vae=vae,
        dtype=torch.float32,
    )
    latents = torch.ones((1, 16, 2, 2, 2), dtype=torch.float32)

    output = pipeline(
        bundle=_bundle(latents),
        output_type="pt",
        return_dict=False,
    )

    assert isinstance(output, tuple)
    assert output[0].shape == (1, 3, 2, 2, 2)
    assert torch.allclose(vae.inputs[0], torch.full_like(latents, 0.5))


def test_hunyuan_orchestrator_requires_inputs_when_no_bundle(tmp_path):
    pipeline = HunyuanVideoOrchestrator(model_path=str(tmp_path), dtype=torch.float32)

    with pytest.raises(ValueError, match="Missing HunyuanVideo DiT inputs"):
        pipeline()
