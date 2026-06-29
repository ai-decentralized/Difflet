"""Unit tests for difflet.pipeline.teacache_gate."""

import pytest
import torch

from difflet.pipeline.teacache_gate import (
    ADAPTIVE_PEARSON,
    DEFAULT_ALPHA,
    ONLINE_AUTOCORR,
    _rel_l1,
    build_calibration,
    decide_method,
    lag1_autocorr,
    pearson,
    run_gate,
)


def test_pearson_returns_none_for_short_input():
    assert pearson([1, 2], [1, 2]) is None


def test_pearson_perfect_correlation():
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_pearson_zero_variance_returns_none():
    assert pearson([1, 1, 1], [1, 2, 3]) is None


def test_lag1_autocorr_short_returns_none():
    assert lag1_autocorr([1, 2, 3]) is None


def test_lag1_autocorr_computes():
    val = lag1_autocorr([1.0, 2.0, 3.0, 4.0, 5.0])
    assert val == pytest.approx(1.0)


def test_decide_method_adaptive():
    assert decide_method(ADAPTIVE_PEARSON + 0.1, 0.0) == "adaptive"


def test_decide_method_online_delta():
    assert decide_method(0.1, ONLINE_AUTOCORR + 0.1) == "online_delta"


def test_decide_method_cadence_fallback():
    assert decide_method(0.1, 0.1) == "cadence"
    assert decide_method(None, None) == "cadence"


def test_build_calibration_adaptive():
    # Strongly correlated signal/delta -> adaptive (accumulate poly).
    signals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    cal = build_calibration(
        model="m", shape_label="s", num_steps=10, signals=signals, deltas=deltas
    )
    assert cal.accumulate is True
    assert cal.online_delta_alpha == 0.0
    assert cal.cadence == 0
    assert len(cal.poly_coef) >= 1


def test_build_calibration_online_delta():
    # Weak signal/delta correlation but smooth (autocorrelated) delta trajectory.
    signals = [0.5, 0.1, 0.9, 0.2, 0.7, 0.05]
    deltas = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    cal = build_calibration(
        model="m", shape_label="s", num_steps=10, signals=signals, deltas=deltas
    )
    assert cal.online_delta_alpha == pytest.approx(DEFAULT_ALPHA)
    assert cal.accumulate is False
    assert cal.cadence == 0


def test_build_calibration_cadence_fallback():
    # No signals, noisy deltas -> neither adaptive nor online -> cadence.
    deltas = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    cal = build_calibration(
        model="m", shape_label="s", num_steps=10, signals=[], deltas=deltas, cadence=3
    )
    assert cal.cadence == 3
    assert cal.accumulate is False
    assert cal.online_delta_alpha == 0.0


def test_rel_l1_with_tensors():
    cur = torch.full((4,), 2.0)
    prev = torch.ones(4)
    # |2-1|.mean() / |1|.mean() = 1.0
    assert _rel_l1(cur, prev) == pytest.approx(1.0)


def test_rel_l1_accepts_non_tensors():
    assert _rel_l1([2.0, 2.0], [1.0, 1.0]) == pytest.approx(1.0)


def test_run_gate_end_to_end_online():
    def init_latent():
        return torch.zeros(4)

    def step_fn(i, latent):
        # Smoothly growing noise_pred, weak/independent signal.
        noise_pred = torch.full((4,), float(i + 1))
        sig = torch.full((4,), float((i * 7) % 5 + 1))
        return noise_pred, sig

    def advance_fn(latent, noise_pred, i):
        return latent + 0.1

    cal, summary = run_gate(
        model="m",
        shape_label="s",
        num_steps=8,
        init_latent=init_latent,
        step_fn=step_fn,
        advance_fn=advance_fn,
    )
    assert summary["method"] in {"adaptive", "online_delta", "cadence"}
    assert summary["n_pairs"] == 7
    assert isinstance(cal.num_steps, int)


def test_run_gate_zero_signal_path_picks_non_adaptive():
    # step_fn returns sig=None -> no signals collected; picks online/cadence.
    def step_fn(i, latent):
        return torch.full((4,), float(i + 1)), None

    cal, summary = run_gate(
        model="m",
        shape_label="s",
        num_steps=8,
        init_latent=lambda: torch.zeros(4),
        step_fn=step_fn,
        advance_fn=lambda latent, np_, i: latent,
    )
    assert cal.accumulate is False  # no signal -> cannot be adaptive
    assert summary["probe_pearson"] is None
