import json

import pytest
import torch

from difflet.pipeline.teacache import (
    CALIBRATION_SCHEMA,
    TeaCacheCalibration,
    TeaCacheController,
    load_teacache_calibration_or_raise,
)


def _calibration(**overrides):
    data = {
        "model": "hunyuan_video",
        "shape_label": "320x512x61",
        "num_steps": 4,
        "poly_coef": (1.0, 2.0, 3.0),
        "threshold": 10.0,
        "warmup_steps": 1,
        "cooldown_steps": 1,
        "target_speedup": 1.5,
        "fit_r2": 0.95,
        "n_samples": 32,
        "mod_input_source": "block0_modulated_input",
    }
    data.update(overrides)
    return TeaCacheCalibration(**data)


def test_teacache_calibration_roundtrips_and_predicts_delta(tmp_path):
    calibration = _calibration()
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration.to_dict()), encoding="utf-8")

    loaded = TeaCacheCalibration.from_json(path)

    assert loaded.to_dict()["schema"] == CALIBRATION_SCHEMA
    assert loaded.poly_coef == (1.0, 2.0, 3.0)
    assert loaded.predict_delta(2.0) == pytest.approx(17.0)
    assert loaded.n_samples == 32
    assert loaded.mod_input_source == "block0_modulated_input"


def test_teacache_controller_protects_warmup_cooldown_and_advances_skip_state():
    calibration = _calibration(poly_coef=(0.0,), threshold=1.0)
    controller = TeaCacheController(calibration)
    mod = torch.zeros((1, 2), dtype=torch.float32)

    assert controller.should_skip(0, mod) is False
    controller.record_full_step(torch.ones((1, 2), dtype=torch.float32), mod)

    assert controller.should_skip(1, mod) is False
    controller.record_full_step(torch.ones((1, 2), dtype=torch.float32) * 3.0, mod)

    assert controller.should_skip(2, mod + 0.1) is True
    assert torch.allclose(controller.skip_noise_pred(mod + 0.1), torch.full((1, 2), 5.0))
    assert controller.should_skip(3, mod + 0.2) is False
    assert controller.stats()["full_steps"] == 2
    assert controller.stats()["skipped_steps"] == 1


def test_teacache_loader_fails_fast_without_calibration():
    with pytest.raises(FileNotFoundError, match="scripts/calibrate_teacache.py"):
        load_teacache_calibration_or_raise(
            None,
            model="hunyuan_video",
            shape_label="320x512x61",
        )


def test_teacache_loader_rejects_model_and_shape_mismatch(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_calibration(model="flux").to_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="model mismatch"):
        load_teacache_calibration_or_raise(
            path,
            model="hunyuan_video",
            shape_label="320x512x61",
        )

    path.write_text(json.dumps(_calibration(shape_label="1x1x1").to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="shape mismatch"):
        load_teacache_calibration_or_raise(
            path,
            model="hunyuan_video",
            shape_label="320x512x61",
        )


def test_should_skip_accepts_precomputed_diff_norm():
    """cclog 72: when the probe NEFF supplies a precomputed delta scalar,
    the controller must use it instead of doing its own host-side diff."""
    import torch
    from difflet.pipeline.teacache import TeaCacheCalibration, TeaCacheController

    calibration = TeaCacheCalibration(
        model="test",
        shape_label="test_shape",
        num_steps=10,
        poly_coef=(0.0, 1.0),
        threshold=0.5,
        warmup_steps=0,
        cooldown_steps=0,
    )
    controller = TeaCacheController(calibration)
    # Seed the controller with non-None prev state so should_skip can return True.
    controller.prev_mod_input = torch.zeros(3)
    controller.prev_noise_pred = torch.zeros(3)
    controller.cached_residual = torch.zeros(3)

    # Precomputed delta below threshold → skip.
    assert controller.should_skip(step_index=5, mod_input_now=torch.zeros(3), diff_norm=0.1) is True
    assert controller.last_delta_estimate == 0.1

    # Precomputed delta above threshold → don't skip.
    assert controller.should_skip(step_index=5, mod_input_now=torch.zeros(3), diff_norm=0.9) is False
    assert controller.last_delta_estimate == 0.9

    # When diff_norm=None, fall back to host-side computation.
    controller.prev_mod_input = torch.zeros(3)
    assert controller.should_skip(step_index=5, mod_input_now=torch.ones(3)) is False  # ||1-0||=sqrt(3)>0.5
    assert controller.last_delta_estimate > 0.5


def test_online_delta_mode_is_probe_free_and_skips_only_flat_steps():
    """cclog 91 generic online-delta mode: 0-per-model, data-driven, no two skips in a row."""
    cal = _calibration(num_steps=8, warmup_steps=2, cooldown_steps=1, online_delta_alpha=0.5)
    c = TeaCacheController(cal)
    assert c.needs_signal() is False  # online mode uses noise_pred, no probe

    def npred(v):
        return torch.full((1, 4), float(v))

    assert c.should_skip(0, None) is False                # warmup
    c.record_full_step(npred(1.0))                        # prev=1.0 (no delta yet)
    assert c.should_skip(1, None) is False                # warmup
    c.record_full_step(npred(3.0))                        # delta=|3-1|/1=2.0 -> baseline=2.0
    assert c.should_skip(2, None) is False                # last 2.0 !< 0.5*2.0=1.0 -> run
    c.record_full_step(npred(3.2))                        # delta=0.2/3~0.067 (flat)
    assert c.should_skip(3, None) is True                 # 0.067 < 1.0 -> SKIP
    c.skip_noise_pred()
    assert c.should_skip(4, None) is False                # just skipped -> must re-measure
    c.record_full_step(npred(3.25))                       # delta small again
    assert c.should_skip(5, None) is True                 # flat -> SKIP again
    assert c.skipped_steps == 1                            # only one skip recorded so far


def test_online_delta_calibration_roundtrips(tmp_path):
    cal = _calibration(online_delta_alpha=0.6)
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cal.to_dict()), encoding="utf-8")
    loaded = TeaCacheCalibration.from_json(path)
    assert loaded.online_delta_alpha == 0.6
    assert loaded.to_dict()["online_delta_alpha"] == 0.6


def test_teacache_gate_auto_selects_method():
    from difflet.pipeline.teacache_gate import decide_method, build_calibration

    # adaptive: signal predicts delta
    assert decide_method(0.95, 0.5) == "adaptive"
    # online: weak signal but the output trajectory is self-predictable
    assert decide_method(0.30, 0.93) == "online_delta"
    # cadence: neither
    assert decide_method(0.30, 0.10) == "cadence"

    # build_calibration wires the right controller mode
    adapt = build_calibration(model="m", shape_label="s", num_steps=50,
                              signals=[1.0, 2.0, 3.0, 4.0, 5.0], deltas=[1.1, 2.0, 3.1, 3.9, 5.2])
    assert adapt.online_delta_alpha == 0.0 and adapt.cadence == 0 and adapt.accumulate is True

    online = build_calibration(model="m", shape_label="s", num_steps=50,
                               signals=[1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0, 5.5],
                               deltas=[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    assert online.online_delta_alpha > 0 and online.cadence == 0
    assert TeaCacheController(online).needs_signal() is False  # probe-free


def test_teacache_gate_on_real_trajectories():
    """HV-1.0 -> adaptive (strong probe signal); Qwen -> online_delta (smooth output)."""
    import json
    from collections import defaultdict
    from pathlib import Path
    from difflet.pipeline.teacache_gate import build_calibration

    def load(path):
        p = Path(path)
        if not p.exists():
            pytest.skip(f"{path} not present")
        s = json.load(open(p))["samples"]
        byb = defaultdict(list)
        for x in s:
            byb[x.get("bundle", "_")].append(x)
        traj = max(byb.values(), key=len)
        traj = sorted(traj, key=lambda r: r.get("step_index", 0))
        return ([r["mod_input_diff_norm"] for r in traj], [r["noise_pred_diff_norm"] for r in traj])

    sig, dlt = load("cclogs/m9-teacache/pairs_hunyuan_video_n4_4d8s1r_50step.json")
    cal = build_calibration(model="hunyuan_video", shape_label="x", num_steps=50, signals=sig, deltas=dlt)
    assert cal.accumulate is True and cal.online_delta_alpha == 0.0, "HV should pick adaptive"

    sig, dlt = load("cclogs/m9-teacache/pairs_qwen_image_1024_50step.json")
    cal = build_calibration(model="qwen_image", shape_label="x", num_steps=50, signals=sig, deltas=dlt)
    assert cal.online_delta_alpha > 0, "Qwen should pick online_delta (weak probe, smooth output)"


def test_run_gate_end_to_end_with_fake_models():
    from difflet.pipeline.teacache_gate import run_gate

    def fake(np_vals, sig_vals):
        def init_latent():
            return torch.zeros((1, 4))
        def step_fn(i, latent):
            return torch.full((1, 4), float(np_vals[i])), torch.full((1, 4), float(sig_vals[i]))
        def advance_fn(latent, noise_pred, i):
            return latent
        return init_latent, step_fn, advance_fn

    # adaptive: signal == output → signal change perfectly predicts delta
    vals = [10, 8, 6, 4, 3, 2, 1.5, 1.2, 1.1, 1.05]
    il, sf, af = fake(vals, vals)
    cal, summ = run_gate(model="m", shape_label="s", num_steps=len(vals),
                         init_latent=il, step_fn=sf, advance_fn=af)
    assert summ["method"] == "adaptive" and cal.accumulate is True

    # online: smooth output (autocorrelated δ) but zigzag signal (uncorrelated)
    npv = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    sgv = [1, 9, 2, 8, 3, 7, 4, 6, 5, 5.5]
    il, sf, af = fake(npv, sgv)
    cal, summ = run_gate(model="m", shape_label="s", num_steps=len(npv),
                         init_latent=il, step_fn=sf, advance_fn=af)
    assert summ["method"] == "online_delta" and cal.online_delta_alpha > 0
    assert summ["delta_autocorr"] is not None and summ["delta_autocorr"] >= 0.7


def test_run_gate_zero_per_model_no_signal_picks_online():
    """No block-0 hook at all (signal=None): gate picks online_delta from output δ alone."""
    from difflet.pipeline.teacache_gate import run_gate

    npv = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    cal, summ = run_gate(
        model="m", shape_label="s", num_steps=len(npv),
        init_latent=lambda: torch.zeros((1, 4)),
        step_fn=lambda i, latent: (torch.full((1, 4), float(npv[i])), None),  # no signal
        advance_fn=lambda latent, np, i: latent,
    )
    assert summ["probe_pearson"] is None        # no signal captured
    assert summ["method"] == "online_delta" and cal.online_delta_alpha > 0


def test_build_probe_free_controller_cadence_needs_no_signal(capsys):
    from difflet.pipeline.teacache import build_probe_free_controller

    ctrl = build_probe_free_controller(model="wan", shape_label="480x832x9", cadence=2)
    assert ctrl.needs_signal() is False
    assert ctrl.calibration.cadence == 2
    assert ctrl.calibration.online_delta_alpha == 0.0
    assert ctrl.calibration.num_steps == 0  # synced at denoise time
    assert "[teacache] probe-free controller enabled" in capsys.readouterr().out


def test_build_probe_free_controller_online_delta():
    from difflet.pipeline.teacache import build_probe_free_controller

    ctrl = build_probe_free_controller(
        model="ltx_2", shape_label="256x384x121", online_delta_alpha=0.6
    )
    assert ctrl.needs_signal() is False
    assert ctrl.calibration.online_delta_alpha == pytest.approx(0.6)


def test_build_probe_free_controller_rejects_empty_and_both_modes():
    from difflet.pipeline.teacache import build_probe_free_controller

    with pytest.raises(ValueError, match="cadence > 0 or online_delta_alpha > 0"):
        build_probe_free_controller(model="wan", shape_label="s")
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_probe_free_controller(model="wan", shape_label="s", cadence=2, online_delta_alpha=0.5)


def test_sync_probe_free_num_steps_follows_request_but_not_for_adaptive():
    from difflet.pipeline.teacache import build_probe_free_controller, sync_probe_free_num_steps

    ctrl = build_probe_free_controller(model="wan", shape_label="s", cadence=2)
    sync_probe_free_num_steps(ctrl, 40)
    assert ctrl.calibration.num_steps == 40
    # Cadence 2, warmup/cooldown 5 at 40 steps: skips at steps 6, 8, ..., 34.
    ctrl.prev_noise_pred = torch.zeros(1)
    ctrl.cached_residual = torch.zeros(1)
    skips = [i for i in range(40) if ctrl.should_skip(i, None)]
    assert skips == list(range(6, 35, 2))

    adaptive = TeaCacheController(_calibration())
    sync_probe_free_num_steps(adaptive, 40)
    assert adaptive.calibration.num_steps == 4  # calibration contract untouched
