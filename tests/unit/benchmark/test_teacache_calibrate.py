"""benchmark.teacache_calibrate: the placeholder never skips, the threshold
search simulates exactly the controller's accumulate branch, and `fit` writes
the calibration models.resolve() names for tp4tcad at cadence 2's skip budget."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from benchmark import teacache_calibrate as tc
from benchmark.models import cadence2_skips, resolve
from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController


def _cfg(tmp_path, slug="flux_1_dev"):
    return replace(resolve(slug, "tp4tcad"),
                   teacache_calibration=str(tmp_path / f"{slug}_tp4tcad.json"))


def _drive(ctrl: TeaCacheController, signals: dict, steps: int) -> int:
    """Run the real controller over a signal series the way the pipelines do."""
    skips = 0
    for i in range(steps):
        if ctrl.should_skip(i, None, diff_norm=signals.get(i)):
            ctrl.skip_noise_pred(None)
            skips += 1
        else:
            ctrl.record_full_step(torch.full((4,), float(i)), None)
    return skips


def test_placeholder_is_a_valid_calibration_that_never_skips(tmp_path):
    cfg = _cfg(tmp_path)
    path = tc.write_placeholder(cfg)
    cal = TeaCacheCalibration.from_json(path)
    assert cal.model == "flux" and cal.shape_label == "1024x1024" and cal.num_steps == 28
    assert cal.target_speedup is None and cal.cadence == 0 and cal.online_delta_alpha == 0.0
    ctrl = TeaCacheController(cal)
    assert ctrl.needs_signal()          # the probe / host signal is computed every step
    assert _drive(ctrl, {i: 0.001 for i in range(28)}, 28) == 0
    assert ctrl.full_steps == 28 and ctrl.skipped_steps == 0
    video = _cfg(tmp_path, "wan_2_1")
    cal = TeaCacheCalibration.from_json(tc.write_placeholder(video))
    assert cal.model == "wan" and cal.shape_label == "480x832x9" and cal.num_steps == 20


def test_calibration_prompts_exclude_the_benchmark_prompt():
    prompts = tc.calibration_prompts()
    assert len(prompts) >= 3
    assert resolve("flux_1_dev", "tp4").prompt not in prompts


def test_simulate_skips_matches_the_controller_accumulate_branch():
    cal = TeaCacheCalibration(model="flux", shape_label="1024x1024", num_steps=28,
                              poly_coef=(0.0, 1.0), threshold=0.35, accumulate=True)
    signals = {i: 0.1 + 0.02 * (i % 5) for i in range(28)}
    real = _drive(TeaCacheController(cal), signals, 28)
    sim = tc._simulate_skips(np.poly1d([1.0, 0.0]), signals, steps=28, threshold=0.35)
    assert real == sim and 0 < real < 18
    # warmup / cooldown are never skipped, whatever the threshold
    assert tc._simulate_skips(np.poly1d([1.0, 0.0]), signals, steps=28, threshold=1e9) == 18


def test_fit_writes_the_tp4tcad_calibration_at_the_skip_budget(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(tc, "resolve", lambda slug, config: cfg)
    rng = np.random.default_rng(0)
    trajs = []
    for k in range(3):
        traj = []
        for i in range(1, 28):
            sig = 0.10 + 0.001 * (i % 3) + 0.002 * k
            traj.append({"step": i, "signal": sig,
                         "delta": 0.5 * sig + float(rng.normal(0, 1e-4))})
        trajs.append({"prompt_index": k, "prompt": f"p{k}", "seed": 42, "trajectory": traj})
    tc.pairs_path(cfg).write_text(json.dumps({
        "schema": "difflet-bench-teacache-pairs-v1", "model": "flux", "slug": "flux_1_dev",
        "shape_label": "1024x1024", "num_steps": 28, "signal_source": "test",
        "trajectories": trajs}))
    assert tc.fit(argparse.Namespace(model="flux_1_dev", degree=4)) == 0
    doc = json.loads(open(cfg.teacache_calibration).read())
    cal = TeaCacheCalibration.from_json(cfg.teacache_calibration)
    assert cal.accumulate and cal.model == "flux" and cal.num_steps == 28
    assert cal.target_speedup == cfg.teacache_speedup == 1.474
    assert cal.warmup_steps == 5 and cal.cooldown_steps == 5
    assert doc["target_skips"] == cadence2_skips(28) == 9
    assert abs(np.mean(doc["simulated_skips_per_prompt"]) - 9) <= 1
    assert doc["signal_pearson"] > 0.9 and 0.0 <= doc["fit_r2"] <= 1.0
    assert doc["calibration_prompts"] == ["p0", "p1", "p2"] and doc["n_samples"] == 81
    # the real controller on the calibration reproduces the simulated budget
    signals = {p["step"]: p["signal"] for p in trajs[0]["trajectory"]}
    assert _drive(TeaCacheController(cal), signals, 28) == doc["simulated_skips_per_prompt"][0]
    # and the benchmark record picks the fit up for the report
    d = cfg.teacache_dict()
    assert d["mode"] == "adaptive" and d["fit_r2"] == doc["fit_r2"] and d["accumulate"] is True
    assert d["poly_degree"] == 4 and d["threshold"] == cal.threshold


def test_fit_rejects_an_empty_pairs_file(tmp_path, monkeypatch, capsys):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(tc, "resolve", lambda slug, config: cfg)
    tc.pairs_path(cfg).write_text(json.dumps({"model": "flux", "shape_label": "1024x1024",
                                              "signal_source": "t", "trajectories": []}))
    assert tc.fit(argparse.Namespace(model="flux_1_dev", degree=4)) == 2
