"""CPU tests for the Flux TPU application: construction contract, the geometry /
schedule helpers against diffusers' own, the device loop against the scheduler,
and the T5 broadcast encoder's call surface."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from difflet.backends.tpu.flux.config import FLUX_1_DEV_CONFIG
from difflet.models.flux import tpu_application as mod
from difflet.models.flux.entry import create_flux_application
from difflet.models.flux.tpu_application import TpuFluxApplication
from difflet.pipeline.parallel_config import DiffletParallelConfig

_SCHEDULER = {
    "base_image_seq_len": 256, "base_shift": 0.5, "max_image_seq_len": 4096, "max_shift": 1.15,
    "num_train_timesteps": 1000, "shift": 3.0, "use_dynamic_shifting": True,
}


@pytest.fixture
def snapshot(tmp_path):
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(json.dumps(FLUX_1_DEV_CONFIG))
    return tmp_path


def _app(snapshot, **kwargs):
    return create_flux_application(
        model_path=str(snapshot), parallel=DiffletParallelConfig(tp_degree=4), dtype="bf16",
        shape={"height": 1024, "width": 1024, "num_frames": None}, backend="tpu", **kwargs,
    )


def test_entry_routes_tpu_to_the_tpu_application(snapshot):
    app = _app(snapshot, teacache_cadence=2)
    assert isinstance(app, TpuFluxApplication)
    assert app.dtype is torch.bfloat16
    assert app.kwargs["teacache_cadence"] == 2
    assert app.config.tp_degree == 4 and app.config.image_seq_len == 4096
    assert app.t5 is None and app.vae is None and app.teacache is None  # load_eager builds them


def test_entry_refuses_cp_on_tpu(snapshot):
    with pytest.raises(NotImplementedError, match="tp only"):
        create_flux_application(
            model_path=str(snapshot), parallel=DiffletParallelConfig(tp_degree=2, cp_degree=2),
            dtype="bf16", shape={"height": 1024, "width": 1024}, backend="tpu",
        )


def test_entry_refuses_unknown_backend(snapshot):
    with pytest.raises(NotImplementedError, match="trainium and tpu"):
        create_flux_application(
            model_path=str(snapshot), parallel=DiffletParallelConfig(tp_degree=4), dtype="bf16",
            shape={"height": 1024, "width": 1024}, backend="cuda",
        )


def test_dit_input_contract(snapshot):
    contract = _app(snapshot).dit_input_contract()
    assert contract["hidden_states"] == {"shape": (1, 4096, 64), "dtype": torch.bfloat16}
    assert contract["encoder_hidden_states"]["shape"] == (1, 512, 4096)
    assert contract["pooled_projections"]["shape"] == (1, 768)
    assert contract["img_ids"] == {"shape": (4096, 3), "dtype": torch.float32}
    assert contract["txt_ids"]["shape"] == (512, 3)
    assert len(contract) == 7


def test_packed_latents_match_diffusers_prepare_latents():
    from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

    stub = SimpleNamespace(
        vae_scale_factor=8, _pack_latents=FluxPipeline._pack_latents,
        _prepare_latent_image_ids=FluxPipeline._prepare_latent_image_ids,
    )
    expected, expected_ids = FluxPipeline.prepare_latents(
        stub, 1, 16, 1024, 1024, torch.float32, torch.device("cpu"), torch.Generator().manual_seed(7)
    )
    packed, ids = mod.prepare_packed_latents(1024, 1024, in_channels_latent=16,
                                             generator=torch.Generator().manual_seed(7))
    assert packed.shape == (1, 4096, 64) and torch.equal(packed, expected)
    assert torch.equal(ids, expected_ids)
    # unpack is the exact inverse of the packing
    grid = mod.unpack_latents(packed, 1024, 1024)
    assert grid.shape == (1, 16, 128, 128)
    assert torch.equal(FluxPipeline._pack_latents(grid, 1, 16, 128, 128), packed)


def test_sigmas_and_mu_match_diffusers():
    import numpy as np

    sigmas, mu = mod.flux_sigmas_and_mu(_SCHEDULER, 28, 4096)
    assert np.allclose(sigmas, np.linspace(1.0, 1 / 28, 28))
    assert mu == pytest.approx(1.15)  # max_image_seq_len -> max_shift
    _, mu_small = mod.flux_sigmas_and_mu(_SCHEDULER, 28, 256)
    assert mu_small == pytest.approx(0.5)


def test_device_euler_loop_matches_scheduler_step():
    from diffusers import FlowMatchEulerDiscreteScheduler

    def velocity(x, t):
        return -0.1 * x + t.to(x.dtype)

    steps = 6
    sigmas_np, mu = mod.flux_sigmas_and_mu(_SCHEDULER, steps, 4096)
    ref = FlowMatchEulerDiscreteScheduler(**_SCHEDULER)
    ref.set_timesteps(sigmas=sigmas_np.tolist(), mu=mu, device="cpu")
    x0 = torch.randn(1, 8, 4, generator=torch.Generator().manual_seed(1))
    expected = x0.clone()
    for t in ref.timesteps:
        expected = ref.step(velocity(expected, t / 1000), t, expected, return_dict=False)[0]

    sched = FlowMatchEulerDiscreteScheduler(**_SCHEDULER)
    sched.set_timesteps(sigmas=sigmas_np.tolist(), mu=mu, device="cpu")
    s = sched.sigmas.float()
    deltas = [(s[i + 1] - s[i]).reshape(1) for i in range(steps)]
    timesteps = [(t / 1000).reshape(1) for t in sched.timesteps]
    actual = mod.device_euler_loop(velocity, x0.clone(), timesteps, deltas, controller=None,
                                   mark_step=lambda: None)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_device_euler_loop_skips_on_cadence():
    from difflet.pipeline.teacache import build_probe_free_controller

    controller = build_probe_free_controller(model="flux", shape_label="1024x1024", cadence=2)
    calls = []

    def velocity(x, t):
        calls.append(float(t))
        return torch.ones_like(x)

    steps = 20
    timesteps = [torch.tensor([1.0 - i / steps]) for i in range(steps)]
    deltas = [torch.tensor([-1.0 / steps])] * steps
    mod.device_euler_loop(velocity, torch.zeros(1, 2), timesteps, deltas, controller=controller,
                          mark_step=lambda: None)
    stats = controller.stats()
    assert stats["skipped_steps"] == 5 and stats["full_steps"] == 15  # 5/5 warmup/cooldown, cadence 2
    assert len(calls) == 15


def test_image_tensor_to_pil_matches_vae_image_processor():
    from diffusers.image_processor import VaeImageProcessor

    image = torch.rand(1, 3, 16, 24, generator=torch.Generator().manual_seed(3)) * 2 - 1
    expected = VaeImageProcessor().postprocess(image, output_type="pil")[0]
    actual = mod.image_tensor_to_pil(image)
    assert actual.size == (24, 16)
    assert list(actual.getdata()) == list(expected.getdata())


def _fake_xla(monkeypatch):
    class _XM:
        @staticmethod
        def collective_broadcast(payload, root_ordinal=0):
            return None

        @staticmethod
        def mark_step():
            return None

    fake = types.SimpleNamespace(device=lambda: "cpu", core=types.SimpleNamespace(xla_model=_XM))
    monkeypatch.setitem(sys.modules, "torch_xla", fake)
    monkeypatch.setitem(sys.modules, "torch_xla.core", fake.core)
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", _XM)


def test_t5_broadcast_encoder_returns_the_wire_dtype_and_reraises_failures(monkeypatch):
    _fake_xla(monkeypatch)
    enc = mod.TpuBroadcastT5Encoder.__new__(mod.TpuBroadcastT5Encoder)
    enc.seq_len, enc.hidden_size, enc.dtype, enc.is_encoder = 8, 6, torch.bfloat16, True
    enc.encode_local = lambda prompt: torch.ones(1, 8, 6, dtype=torch.float32) * 2
    out = enc("a fox")
    assert out.shape == (1, 8, 6) and out.dtype is torch.bfloat16 and float(out[0, 0, 0]) == 2.0

    def boom(prompt):
        raise ValueError("tokenizer exploded")

    enc.encode_local = boom
    # The encoder rank must not raise before the collective; it raises after, on every rank.
    with pytest.raises(RuntimeError, match="encoder rank"):
        enc("a fox")


def test_call_returns_latents_or_a_placeholder_without_a_vae(snapshot, monkeypatch):
    app = _app(snapshot)
    monkeypatch.setattr(app, "encode_prompt", lambda prompt: {"prompt": prompt})
    seen = {}

    def denoise(text, **kw):
        seen.update(kw)
        return torch.zeros(1, 4096, 64)

    monkeypatch.setattr(app, "denoise", denoise)
    out = app("a fox", num_inference_steps=4, guidance_scale=2.5, output_type="latent")
    assert out.latents.shape == (1, 4096, 64) and out.images is None
    assert seen["num_inference_steps"] == 4 and seen["guidance_scale"] == 2.5
    # No VAE on this replica (non-primary): a correctly sized placeholder image.
    out = app("a fox", output_type="pil")
    assert out.images[0].size == (1024, 1024)
    assert set(app.last_timings) >= {"encode_s", "denoise_s"}


def test_denoise_refuses_a_shape_other_than_the_loaded_one(snapshot):
    app = _app(snapshot)
    app.scheduler = object()
    with pytest.raises(ValueError, match="loaded for 1024x1024"):
        app.denoise({}, num_inference_steps=4, guidance_scale=3.5, generator=None, height=512, width=512)
