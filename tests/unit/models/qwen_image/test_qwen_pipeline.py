"""CPU-only unit coverage for difflet.models.qwen_image.pipeline helpers/branches.

These exercise the pure helpers and host-side branches (scheduler fallback,
shape math, TeaCache fused-probe loop, decode error paths) that do not require
real Neuron weights or hardware.
"""

import json
import os


import numpy as np
import pytest
import torch

from difflet.models.qwen_image.application import QwenImageDiTInputBundle
from difflet.models.qwen_image import pipeline as m

torch.manual_seed(0)


@pytest.fixture(autouse=True)
def _force_cpu_backend():
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    yield
    if prev is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = prev


def _bundle(latents=None):
    latents = torch.zeros((1, 16, 64), dtype=torch.float32) if latents is None else latents
    return QwenImageDiTInputBundle(
        hidden_states=latents,
        timestep=torch.zeros([1], dtype=torch.float32),
        encoder_hidden_states=torch.ones((1, 4, 8), dtype=torch.float32),
        encoder_hidden_states_mask=torch.ones((1, 4), dtype=torch.bool),
        guidance=torch.zeros([1], dtype=torch.float32),
    )


class FakeTransformer:
    def __init__(self, value=0.5):
        self.dtype = torch.float32
        self.value = value
        self.calls = []

    def __call__(self, bundle):
        self.calls.append(bundle)
        return {"sample": torch.ones_like(bundle.hidden_states) * self.value}


class FakeScheduler:
    def __init__(self):
        self.calls = []
        self.timesteps = torch.empty(0)

    def set_timesteps(self, num_inference_steps=None, *, device=None, sigmas=None):
        self.calls.append({"num": num_inference_steps, "device": device, "sigmas": sigmas})
        if sigmas is not None:
            self.timesteps = torch.as_tensor(sigmas, dtype=torch.float32)
        else:
            self.timesteps = torch.linspace(1.0, 0.0, steps=int(num_inference_steps))

    def step(self, noise_pred, timestep, latents, return_dict=False):
        assert return_dict is False
        return (latents - noise_pred,)


def _write_calibration(tmp_path, **overrides):
    data = {
        "schema": "difflet-m9-teacache-calibration-v1",
        "model": "qwen_image",
        "shape_label": "1024x1024",
        "num_steps": 3,
        "poly_coef": [0.0],
        "threshold": 1.0,
        "warmup_steps": 0,
        "cooldown_steps": 0,
        "target_speedup": 2.0,
    }
    data.update(overrides)
    path = tmp_path / "qwen_calib.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_pack_qwen_image_latents_rejects_bad_shapes():
    with pytest.raises(ValueError, match="must have shape"):
        m.pack_qwen_image_latents(torch.zeros((1, 16, 8, 8)))
    with pytest.raises(ValueError, match="exactly one latent frame"):
        m.pack_qwen_image_latents(torch.zeros((1, 2, 16, 8, 8)))
    with pytest.raises(ValueError, match="divisible by 2"):
        m.pack_qwen_image_latents(torch.zeros((1, 1, 16, 7, 8)))


def test_first_tensor_extracts_from_dict_and_raises():
    t = torch.ones(2)
    assert m._first_tensor(t) is t
    assert m._first_tensor({"images": t}) is t
    assert torch.equal(m._first_tensor([None, t]), t)
    with pytest.raises(TypeError, match="Could not extract tensor"):
        m._first_tensor(object())


def test_component_dtype_falls_back_to_neuron_config():
    from types import SimpleNamespace

    component = SimpleNamespace(
        config=SimpleNamespace(neuron_config=SimpleNamespace(torch_dtype=torch.float16))
    )
    assert m._component_dtype(component, torch.float32) == torch.float16
    # no usable dtype anywhere -> default
    assert m._component_dtype(SimpleNamespace(config=None), torch.bfloat16) == torch.bfloat16


def test_bundle_from_tensors_fills_guidance_scale():
    bundle = m._bundle_from_tensors(
        latents=torch.zeros((2, 16, 64), dtype=torch.float32),
        encoder_hidden_states=torch.ones((2, 4, 8), dtype=torch.float32),
        encoder_hidden_states_mask=None,
        guidance=None,
        guidance_scale=3.5,
        dtype=torch.float32,
    )
    assert bundle.encoder_hidden_states_mask.shape == (2, 4)
    assert bundle.encoder_hidden_states_mask.dtype == torch.bool
    assert torch.allclose(bundle.guidance, torch.full((2,), 3.5))


# --------------------------------------------------------------------------- #
# orchestrator branches
# --------------------------------------------------------------------------- #
def test_has_runtime_components(tmp_path):
    with pytest.warns(RuntimeWarning):
        active = m.QwenImageOrchestrator(model_path=str(tmp_path), transformer=FakeTransformer())
    idle = m.QwenImageOrchestrator(model_path=str(tmp_path))
    assert active.has_runtime_components() is True
    assert idle.has_runtime_components() is False


def test_prepare_latents_packed_and_unpacked_paths(tmp_path):
    pipe = m.QwenImageOrchestrator(model_path=str(tmp_path), dtype=torch.float32, height=64, width=64)
    packed = torch.zeros((1, 16, 64), dtype=torch.float32)
    assert pipe.prepare_latents(batch_size=1, latents=packed) is packed
    with pytest.raises(ValueError, match="Expected packed latents shape"):
        pipe.prepare_latents(batch_size=1, latents=torch.zeros((1, 8, 64)))
    unpacked = torch.zeros((1, 1, 16, 8, 8), dtype=torch.float32)
    assert pipe.prepare_latents(batch_size=1, latents=unpacked).shape == (1, 16, 64)
    with pytest.raises(ValueError, match="packed 3D or unpacked 5D"):
        pipe.prepare_latents(batch_size=1, latents=torch.zeros((1, 16)))


def test_call_rejects_unknown_output_type(tmp_path):
    pipe = m.QwenImageOrchestrator(model_path=str(tmp_path), dtype=torch.float32)
    with pytest.raises(ValueError, match="output_type"):
        pipe(bundle=_bundle(), output_type="np")


def test_timesteps_fallback_linspace(tmp_path):
    pipe = m.QwenImageOrchestrator(model_path=str(tmp_path), dtype=torch.float32, scheduler=None)
    ts = pipe._timesteps(4, device=torch.device("cpu"))
    assert torch.allclose(ts, torch.tensor([1.0, 0.75, 0.5, 0.25]))
    # num_inference_steps is clamped to >= 1
    assert pipe._timesteps(0, device=torch.device("cpu")).numel() == 1


def test_timesteps_with_scheduler(tmp_path):
    scheduler = FakeScheduler()
    pipe = m.QwenImageOrchestrator(model_path=str(tmp_path), dtype=torch.float32, scheduler=scheduler)
    ts = pipe._timesteps(3, device=torch.device("cpu"))
    assert ts.numel() == 3
    assert scheduler.calls[0]["sigmas"].tolist() == pytest.approx(list(np.linspace(1.0, 1.0 / 3, 3)))


def test_call_uses_fallback_timesteps_and_scalar_timestep(tmp_path):
    transformer = FakeTransformer(value=0.25)
    with pytest.warns(RuntimeWarning):
        pipe = m.QwenImageOrchestrator(
            model_path=str(tmp_path), transformer=transformer, dtype=torch.float32
        )
    # timesteps=None -> _timesteps fallback path inside __call__
    pipe(bundle=_bundle(), num_inference_steps=2)
    assert len(transformer.calls) == 2
    transformer.calls.clear()
    # scalar 0-d timestep is promoted to shape (1,)
    pipe(bundle=_bundle(), timesteps=torch.tensor(1.0))
    assert len(transformer.calls) == 1


def test_scheduler_step_with_real_scheduler(tmp_path):
    transformer = FakeTransformer(value=1.0)
    scheduler = FakeScheduler()
    pipe = m.QwenImageOrchestrator(
        model_path=str(tmp_path), transformer=transformer, dtype=torch.float32, scheduler=scheduler
    )
    out = pipe(bundle=_bundle(), timesteps=torch.tensor([1.0]))
    # FakeScheduler.step subtracts the (==1.0) prediction from zero latents.
    assert torch.allclose(out.latents, torch.full((1, 16, 64), -1.0))


def test_decode_requires_vae(tmp_path):
    pipe = m.QwenImageOrchestrator(model_path=str(tmp_path), dtype=torch.float32, height=64, width=64)
    with pytest.raises(ValueError, match="requires an active VAE"):
        pipe(bundle=_bundle(torch.ones((1, 16, 64))), output_type="pt")


def test_decode_without_decode_method_calls_module(tmp_path):
    class CallableVAE:
        dtype = torch.float32
        config = None

        def __call__(self, latents):
            return (latents[:, :3],)

    pipe = m.QwenImageOrchestrator(
        model_path=str(tmp_path), vae=CallableVAE(), dtype=torch.float32, height=64, width=64
    )
    out = pipe(bundle=_bundle(torch.ones((1, 16, 64))), output_type="pt", return_dict=False)
    assert out[0].shape == (1, 3, 1, 8, 8)


def test_load_scheduler_reads_diffusers_config(tmp_path):
    scheduler_dir = tmp_path / "scheduler"
    scheduler_dir.mkdir()
    (scheduler_dir / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000}),
        encoding="utf-8",
    )
    scheduler = m._load_scheduler(str(tmp_path))
    assert type(scheduler).__name__ == "FlowMatchEulerDiscreteScheduler"


def test_missing_scheduler_warns_only_when_requested(tmp_path):
    # warn_if_missing False stays silent
    assert m._load_scheduler(str(tmp_path), warn_if_missing=False) is None


def test_retrieve_timesteps_without_sigmas_uses_num_steps():
    class FakeScheduler2:
        def __init__(self):
            self.kwargs = None
            self.timesteps = torch.tensor([1.0, 0.5])

        def set_timesteps(self, num_inference_steps=None, *, device=None, sigmas=None):
            self.kwargs = {"num": num_inference_steps, "sigmas": sigmas, "device": device}

    s = FakeScheduler2()
    ts, n = m._retrieve_timesteps(s, 2, "cpu")
    assert s.kwargs["num"] == 2 and s.kwargs["sigmas"] is None
    assert n == 2 and ts.numel() == 2


# --------------------------------------------------------------------------- #
# TeaCache fused-probe loop
# --------------------------------------------------------------------------- #
class FakeTeacacheTransformer(FakeTransformer):
    teacache_probe_fused = True

    def __init__(self, value=0.5):
        super().__init__(value=value)
        self.delta_calls = 0

    def teacache_delta(self, bundle):
        self.delta_calls += 1
        return torch.tensor(0.01)


def test_teacache_requires_calibration(tmp_path):
    with pytest.raises(FileNotFoundError):
        m.QwenImageOrchestrator(
            model_path=str(tmp_path),
            transformer=FakeTeacacheTransformer(),
            dtype=torch.float32,
            teacache_speedup=1.5,
        )


def test_teacache_rejects_speedup_above_calibration(tmp_path):
    calib = _write_calibration(tmp_path, target_speedup=2.0)
    with pytest.raises(ValueError, match="target speedup is lower"):
        m.QwenImageOrchestrator(
            model_path=str(tmp_path),
            transformer=FakeTeacacheTransformer(),
            dtype=torch.float32,
            teacache_speedup=3.0,
            teacache_calibration_path=calib,
        )


def test_teacache_fused_probe_skips_and_records(tmp_path):
    calib = _write_calibration(tmp_path)
    transformer = FakeTeacacheTransformer(value=0.5)
    pipe = m.QwenImageOrchestrator(
        model_path=str(tmp_path),
        transformer=transformer,
        dtype=torch.float32,
        teacache_speedup=1.5,
        teacache_calibration_path=calib,
    )
    assert pipe.teacache_controller is not None
    out = pipe(bundle=_bundle(), timesteps=torch.tensor([1.0, 0.6, 0.3]))
    # 2 full steps + 1 skipped step (constant-zero poly < threshold on the last step)
    assert transformer.delta_calls == 3
    assert len(transformer.calls) == 2
    assert pipe.teacache_controller.stats()["skipped_steps"] == 1
    assert out.latents.shape == (1, 16, 64)
