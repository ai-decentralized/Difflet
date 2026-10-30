"""HunyuanVideo 1.5 TeaCache calibration collector (CPU, fake DiT / probe / scheduler)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("DIFFLET_BACKEND", "cpu")

from difflet.models.hunyuan_video import pipeline15 as p15

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "calibrate_teacache.py"
_spec = importlib.util.spec_from_file_location("calibrate_teacache", _SCRIPT)
cal = importlib.util.module_from_spec(_spec)
sys.modules["calibrate_teacache"] = cal
_spec.loader.exec_module(cal)

H, W, F, STEPS = 32, 48, 5, 6


class _Probe:
    """Same two methods as the runtime probe (pipeline15.TeaCacheProbe)."""

    def __init__(self):
        self.calls, self.resets = [], 0

    def reset(self):
        self.resets += 1

    def teacache_delta(self, latent_input, timestep):
        self.calls.append(latent_input.clone())
        return 0.25 * len(self.calls)


class _Dit:
    cfg = 1
    image_tokens = 5
    model_config = dict(in_channels=9, out_channels=4, image_embed_dim=16)

    def __init__(self):
        self.probe, self.n = _Probe(), 0

    def load(self, skip_warmup=False):
        pass

    def begin_request(self):
        pass

    def make_teacache_probe(self):
        return self.probe

    def __call__(self, *, hidden_states, timestep, **_):
        self.n += 1
        return hidden_states[:, :4] * (1.0 + 0.1 * self.n) + 0.01 * self.n


class _Scheduler:
    def step(self, noise_pred, t, latents, return_dict=False):
        return (latents + 0.1 * noise_pred,)


@pytest.fixture
def denoiser(monkeypatch):
    monkeypatch.setattr(p15, "load_hv15_guidance", lambda *_: 2.0)
    monkeypatch.setattr(p15, "load_hv15_scheduler", lambda *_: _Scheduler())
    monkeypatch.setattr(p15, "set_hv15_timesteps", lambda _s, n: torch.linspace(1000.0, 100.0, n))
    dit = _Dit()
    d = p15.HunyuanVideo15Denoiser(dit, model_path="unused", height=H, width=W, num_frames=F, dtype=torch.float32)
    d.load()
    return d


def _embeds():
    e = (torch.ones(1, 10, 32), torch.ones(1, 10), torch.ones(1, 6, 24), torch.ones(1, 6))
    names = ("prompt_embeds", "prompt_embeds_mask", "prompt_embeds_2", "prompt_embeds_mask_2")
    return dict(zip(names, e)) | {f"negative_{n}": t for n, t in zip(names, e)}


def test_collector_uses_runtime_probe_signal(denoiser):
    spied = []
    real_step = denoiser.scheduler.step
    denoiser.scheduler.step = lambda np_, *a, **k: (spied.append(np_.clone()), real_step(np_, *a, **k))[1]
    samples = cal.collect_hv15_samples(denoiser, _embeds(), num_inference_steps=STEPS, generator=torch.Generator().manual_seed(0))
    probe = denoiser.dit.probe
    # one probe call per step, on the DiT's own make_teacache_probe() object, reset once
    assert len(probe.calls) == STEPS and probe.resets == 1
    assert probe.calls[0].shape[1] == 9  # the loop's latent_input (latents + cond + mask)
    # the first step has no previous step to compare with
    assert [s["step_index"] for s in samples] == list(range(1, STEPS))
    # x is exactly the probe's output; y is the rel-L1 of the guided noise_pred step to step
    assert [s["mod_input_diff_norm"] for s in samples] == [0.25 * (i + 1) for i in range(1, STEPS)]
    for i, s in enumerate(samples, start=1):
        want = ((spied[i] - spied[i - 1]).abs().sum() / spied[i - 1].abs().sum()).item()
        assert s["noise_pred_diff_norm"] == pytest.approx(want, rel=1e-5)


def test_calibration_model_key(tmp_path):
    pairs = {
        "schema": cal.PAIRS_SCHEMA,
        "model": "hunyuan_video_15",
        "shape_label": "320x512x61",
        "num_steps": 50,
        "mod_input_source": cal.HV15_MOD_INPUT_SOURCE,
        "samples": [
            {"mod_input_diff_norm": 0.01 * i, "noise_pred_diff_norm": 0.02 * i + 0.001 * i * i}
            for i in range(1, 30)
        ],
    }
    pairs_path, out = tmp_path / "pairs.json", tmp_path / "cal.json"
    pairs_path.write_text(json.dumps(pairs))
    sys.argv = ["calibrate_teacache.py", "--pairs-json", str(pairs_path), "--out", str(out)]
    assert cal.main() == 0
    doc = json.loads(out.read_text())
    assert doc["model"] == "hunyuan_video_15" and doc["shape_label"] == "320x512x61"
    assert doc["num_steps"] == 50 and doc["mod_input_source"] == "block0_modulated_input"
    from difflet.pipeline.teacache import load_teacache_calibration_or_raise
    from difflet.serving.options import _load_serving_teacache_calibration

    loaded = load_teacache_calibration_or_raise(out, model="hunyuan_video_15", shape_label="320x512x61")
    assert loaded.num_steps == 50 and len(loaded.poly_coef) == 4
    _load_serving_teacache_calibration(
        str(out), model="hunyuan_video_15", shape_label="320x512x61", requested_speedup=1.2
    )


def test_hv15_pairs_doc_shape(monkeypatch, denoiser):
    monkeypatch.setattr(cal, "_load_hv15_denoiser", lambda args: denoiser)
    monkeypatch.setattr(torch, "load", lambda *a, **k: _embeds())
    args = type("A", (), dict(height=320, width=512, num_frames=61, num_inference_steps=STEPS, embeds="e.pt", seed=0))()
    doc = cal._collect_hv15_pairs(args)
    assert (doc["model"], doc["shape_label"], doc["num_steps"]) == ("hunyuan_video_15", "320x512x61", STEPS)
    assert doc["schema"] == cal.PAIRS_SCHEMA and len(doc["samples"]) == STEPS - 1


def test_collector_leaves_trajectory_unchanged(denoiser):
    def run():
        denoiser.dit.n = 0
        return denoiser(**_embeds(), num_inference_steps=STEPS, generator=torch.Generator().manual_seed(0))

    plain = run()
    cal.collect_hv15_samples(denoiser, _embeds(), num_inference_steps=STEPS, generator=torch.Generator().manual_seed(0))
    assert torch.equal(plain, run())
