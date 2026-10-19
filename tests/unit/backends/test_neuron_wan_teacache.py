"""Wan TeaCache on the neuron backend (CPU, eager): the per-expert probe runs the model's own
signal, the probe-free modes skip, and a calibrated run uses the neuron probes, never the
Trainium CPU shadow."""

from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("safetensors.torch")
pytest.importorskip("accelerate")
pytest.importorskip("transformers")

from difflet.backends.neuron.core.teacache import NeuronTeaCacheProbe  # noqa: E402
from difflet.backends.neuron.ops_impl import parallel_mesh  # noqa: E402
from difflet.models.wan import neuron_application as na  # noqa: E402
from difflet.pipeline.parallel_config import DiffletParallelConfig  # noqa: E402
from difflet.pipeline.teacache import TeaCacheCalibration  # noqa: E402
from tests.unit.backends._neuron_wan_toy import (  # noqa: E402
    TINY_LATENT_SHAPE,
    TINY_SHAPE,
    TINY_TEXT_SEQ_LEN,
    TINY_TEXT_SHAPE,
    tiny_wan_pipeline_inputs,
    write_tiny_wan_model,
)


@pytest.fixture(autouse=True)
def _clean_mesh(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    parallel_mesh.destroy_parallel_mesh()
    yield
    parallel_mesh.destroy_parallel_mesh()


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_wan_tc")
    write_tiny_wan_model(path)
    return str(path)


def _app(root, **options):
    app = na.TorchNeuronWanApplication(model_path=root, parallel=DiffletParallelConfig(),
                                       dtype=torch.float32, shape=dict(TINY_SHAPE),
                                       text_seq_len=TINY_TEXT_SEQ_LEN, exec_mode="eager",
                                       device="cpu", **options)
    app.load()
    return app


def _run(app, steps):
    prompt_embeds, latents = tiny_wan_pipeline_inputs()
    return app(prompt_embeds=prompt_embeds, latents=latents, channels=TINY_LATENT_SHAPE[1],
               num_inference_steps=steps, guidance_scale=5.0, output_type="latent").latents


def test_probe_signal_is_the_models_and_ignores_the_prompt(root):
    dit = _app(root).transformer
    g = torch.Generator().manual_seed(3)
    xs = [torch.randn(TINY_LATENT_SHAPE, generator=g) for _ in range(3)]
    ts = [torch.tensor([900.0 - 300 * i]) for i in range(3)]
    # m_t does not depend on the prompt: the zero stand-in is exact
    real_text = torch.randn(TINY_TEXT_SHAPE, generator=g)
    torch.testing.assert_close(dit.teacache_mod_input(xs[0], ts[0]),
                               dit.module.teacache_mod_input(xs[0], ts[0], real_text))
    probe, prev = NeuronTeaCacheProbe(dit.teacache_mod_input), None
    for x, t in zip(xs, ts):
        m = dit.module.teacache_mod_input(x, t, real_text).float()
        p = prev if prev is not None else torch.zeros_like(m)
        expected = float((m - p).abs().mean() / (p.abs().mean() + 1e-8))
        assert float(probe.teacache_delta(x, t)) == pytest.approx(expected, rel=1e-5)
        prev = m


def test_probe_free_cadence_skips_steps(root):
    app = _app(root, teacache_cadence=2)
    out = _run(app, 20)
    ctrl = app.pipeline._teacache_controller
    assert torch.isfinite(out).all() and ctrl.skipped_steps > 0 and ctrl.probe_calls == 0


def test_calibrated_run_uses_the_neuron_probes(root, tmp_path):
    cal = TeaCacheCalibration(model="wan", shape_label="32x48x5", num_steps=20, poly_coef=(1.0, 0.0),
                              threshold=1e9, warmup_steps=1, cooldown_steps=1)
    path = tmp_path / "wan_cal.json"
    path.write_text(json.dumps(cal.to_dict()))
    app = _app(root, teacache_calibration_path=str(path))
    assert set(app.pipeline._teacache_probes) == {id(app.transformer_adapter)}
    out = _run(app, 20)
    ctrl = app.pipeline._teacache_controller
    assert torch.isfinite(out).all() and ctrl.probe_calls > 0 and ctrl.skipped_steps > 0
    assert app.pipeline._teacache_shadows == {}  # the Trainium CPU shadow was never built
