"""Extra TeaCache controller tests targeting branches not covered by
tests/unit/pipeline/test_teacache_controller.py (accumulate / cadence /
skip-run-length / cache mechanics / stats / reset)."""

import json

import pytest
import torch

from difflet.pipeline.teacache import (
    CALIBRATION_SCHEMA,
    TeaCacheCalibration,
    TeaCacheController,
)


def _cal(**kwargs):
    base = dict(
        model="m",
        shape_label="s",
        num_steps=10,
        poly_coef=(0.0, 1.0),
        threshold=0.5,
        warmup_steps=2,
        cooldown_steps=2,
    )
    base.update(kwargs)
    return TeaCacheCalibration(**base)


def _seed_cache(controller):
    """Populate prev_noise_pred + cached_residual so skips are allowed."""
    controller.record_full_step(torch.zeros(4), mod_input=torch.zeros(4))
    controller.record_full_step(torch.zeros(4), mod_input=torch.zeros(4))


def test_from_dict_rejects_bad_schema():
    with pytest.raises(ValueError):
        TeaCacheCalibration.from_dict({"schema": "nope"})


def test_from_json_roundtrip(tmp_path):
    cal = _cal(target_speedup=1.5, fit_r2=0.9, n_samples=12)
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(cal.to_dict()), encoding="utf-8")
    loaded = TeaCacheCalibration.from_json(path)
    assert loaded == cal
    assert loaded.to_dict()["schema"] == CALIBRATION_SCHEMA


def test_needs_signal_and_probe_defaults():
    c = TeaCacheController(_cal())
    assert c.needs_signal() is True
    assert c.needs_probe() is True


def test_cadence_mode_skips_by_index():
    c = TeaCacheController(_cal(cadence=2))
    assert c.needs_signal() is False
    _seed_cache(c)
    # window starts at warmup=2; pos%2 == 1 -> skip.
    assert c.should_skip(2, None) is False  # pos 0
    assert c.should_skip(3, None) is True   # pos 1
    assert c.should_skip(4, None) is False  # pos 2


def test_accumulate_mode_runs_only_when_sum_crosses_threshold():
    # poly_coef=(0,1): predict_delta(x)=x. threshold=1.0.
    c = TeaCacheController(_cal(accumulate=True, threshold=1.0, poly_coef=(0.0, 1.0)))
    _seed_cache(c)
    # accum += |x|; first 0.6 -> sum 0.6 < 1.0 -> skip
    assert c.should_skip(3, torch.zeros(4), diff_norm=0.6) is True
    # next 0.6 -> sum 1.2 >= 1.0 -> run, reset
    assert c.should_skip(4, torch.zeros(4), diff_norm=0.6) is False
    assert c._accum == 0.0


def test_skip_run_length_commits_multiple_skips_without_probe():
    c = TeaCacheController(_cal(skip_run_length=3, threshold=10.0, poly_coef=(0.0, 1.0)))
    _seed_cache(c)
    # predict_delta(0.1)=0.1 < 10 -> skip, commit run of 2 more.
    assert c.should_skip(3, torch.zeros(4), diff_norm=0.1) is True
    assert c._skip_run_remaining == 2
    assert c.needs_probe() is False
    # committed skip without re-probing
    assert c.should_skip(4, None) is True
    assert c._skip_run_remaining == 1
    assert c.should_skip(5, None) is True
    assert c._skip_run_remaining == 0


def test_should_skip_resets_state_during_warmup():
    c = TeaCacheController(_cal())
    c._skip_run_remaining = 5
    c._accum = 9.0
    assert c.should_skip(0, torch.zeros(4)) is False
    assert c._skip_run_remaining == 0
    assert c._accum == 0.0


def test_should_skip_false_during_cooldown():
    c = TeaCacheController(_cal())
    _seed_cache(c)
    # num_steps - cooldown = 8 -> step 8 is in cooldown.
    assert c.should_skip(8, torch.zeros(4)) is False


def test_should_skip_false_without_cache():
    c = TeaCacheController(_cal())
    assert c.should_skip(3, torch.zeros(4)) is False


def test_should_skip_false_when_no_prev_mod_input_and_no_diff_norm():
    c = TeaCacheController(_cal())
    # Seed cache without mod_input so prev_mod_input stays None.
    c.record_full_step(torch.zeros(4))
    c.record_full_step(torch.zeros(4))
    assert c.prev_mod_input is None
    assert c.should_skip(3, torch.zeros(4), diff_norm=None) is False


def test_host_side_diff_norm_decision():
    c = TeaCacheController(_cal(threshold=10.0, poly_coef=(0.0, 1.0)))
    _seed_cache(c)  # prev_mod_input = zeros
    # ||ones - zeros|| = 2.0; predict_delta=2.0 < 10 -> skip
    assert c.should_skip(3, torch.ones(4)) is True


def test_skip_noise_pred_uses_cached_residual():
    c = TeaCacheController(_cal())
    c.record_full_step(torch.zeros(4))
    c.record_full_step(torch.full((4,), 2.0))  # residual = 2 - 0 = 2
    out = c.skip_noise_pred(mod_input=torch.ones(4))
    # prev(2) + residual(2) = 4
    assert torch.allclose(out, torch.full((4,), 4.0))
    assert c.skipped_steps == 1
    assert c._just_skipped is True
    assert torch.allclose(c.prev_mod_input, torch.ones(4))


def test_skip_noise_pred_raises_without_cache():
    c = TeaCacheController(_cal())
    with pytest.raises(RuntimeError):
        c.skip_noise_pred()


def test_record_full_step_online_delta_tracks_baseline():
    c = TeaCacheController(_cal(online_delta_alpha=0.5))
    c.record_full_step(torch.ones(4))
    c.record_full_step(torch.full((4,), 2.0))
    assert c._last_full_delta is not None
    assert c._baseline_delta == c._last_full_delta


def test_note_probe_and_stats_and_reset():
    c = TeaCacheController(_cal())
    c.note_probe()
    c.note_probe()
    _seed_cache(c)
    stats = c.stats()
    assert stats["probe_calls"] == 2
    assert stats["full_steps"] == 2
    assert stats["cache_initialized"] is True
    c.reset()
    s2 = c.stats()
    assert s2 == {
        "full_steps": 0,
        "skipped_steps": 0,
        "probe_calls": 0,
        "last_delta_estimate": None,
        "cache_initialized": False,
    }


def test_online_delta_just_skipped_latch_forces_remeasure():
    c = TeaCacheController(_cal(online_delta_alpha=0.5))
    _seed_cache(c)
    c._just_skipped = True
    assert c.should_skip(3, None) is False
    assert c._just_skipped is False
