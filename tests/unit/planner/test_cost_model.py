"""Cost model: the ratios it asserts between strategies, and how it calibrates.

The model is heuristic, so these tests pin *relationships* the code guarantees
(ulysses moves 2/cp the bytes of gather-KV; SP is communication-neutral; CFG
parallelism halves compute) rather than absolute times, which only a measurement
can establish.
"""
from __future__ import annotations

import math

import pytest

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.planner import cost_model
from difflet.planner.cost_model import (
    Calibration,
    calibrate,
    comm_bytes,
    compute_share,
    predict,
)
from difflet.planner.measurements import Measurement
from difflet.planner.model_profile import load_profile

FLUX = "black-forest-labs/FLUX.1-dev"


@pytest.fixture
def flux():
    profile = load_profile(FLUX)
    return profile, profile.sequence_lengths()


def _measurement(label: str, seconds: float, **kwargs) -> Measurement:
    defaults = dict(
        instance_type="trn2.3xlarge", model="flux", model_id=FLUX, label=label,
        height=1024, width=1024, num_frames=None, steps=28,
        step_latency_seconds=seconds, e2e_warm_seconds=None,
        compile_seconds=None, source="test",
    )
    defaults.update(kwargs)
    return Measurement(**defaults)


# ------------------------------------------------------------- compute share


def test_tensor_parallelism_reduces_compute_share():
    assert compute_share(DiffletParallelConfig(tp_degree=4)) < compute_share(
        DiffletParallelConfig(tp_degree=1)
    )


def test_speedup_is_sublinear_in_tp():
    """Halving a GEMM does not halve its time once the tiles get small."""

    one = compute_share(DiffletParallelConfig(tp_degree=1))
    four = compute_share(DiffletParallelConfig(tp_degree=4))
    assert four > one / 4


def test_cfg_parallel_halves_per_rank_compute():
    without = compute_share(DiffletParallelConfig(tp_degree=2))
    with_cfg = compute_share(DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True))
    assert with_cfg == pytest.approx(without / 2)


def test_sp_shrinks_only_the_replicated_region():
    """SP divides the replicated share by tp and touches nothing else.

    At tp=4 that is a saving of ``f_rep * (1 - 1/4)`` of the single-core total --
    a fifth of the tp=4 step, which is why SP is worth having even though it
    moves no fewer bytes.
    """

    plain = compute_share(DiffletParallelConfig(tp_degree=4))
    with_sp = compute_share(DiffletParallelConfig(tp_degree=4, sp_enabled=True))
    tax = 1.0 + cost_model.TP_EFFICIENCY_TAX * 2  # log2(4)
    expected_saving = cost_model.REPLICATED_COMPUTE_SHARE * (1 - 1 / 4) * tax
    assert plain - with_sp == pytest.approx(expected_saving)


def test_sp_is_a_no_op_at_tp1():
    """Nothing to divide the replicated region by -- feasibility rejects this too."""

    assert compute_share(DiffletParallelConfig(tp_degree=1, sp_enabled=True)) == compute_share(
        DiffletParallelConfig(tp_degree=1)
    )


def test_dp_does_not_change_per_rank_compute():
    """DP replicates whole models; it buys throughput, never single-request latency."""

    assert compute_share(DiffletParallelConfig(tp_degree=2)) == compute_share(
        DiffletParallelConfig(tp_degree=2, dp_degree=2)
    )


# ------------------------------------------------------------------- comm bytes


def test_no_collectives_at_tp1_cp1(flux):
    profile, seq = flux
    breakdown = comm_bytes(DiffletParallelConfig(tp_degree=1), profile=profile, seq=seq)
    assert breakdown.total == 0.0


def test_tensor_parallel_bytes_grow_with_tp(flux):
    profile, seq = flux
    two = comm_bytes(DiffletParallelConfig(tp_degree=2), profile=profile, seq=seq)
    four = comm_bytes(DiffletParallelConfig(tp_degree=4), profile=profile, seq=seq)
    assert four.tensor_parallel > two.tensor_parallel


def test_sp_is_communication_neutral(flux):
    """Reduce-scatter + all-gather moves exactly what the all-reduce it replaces did.

    SP's entire predicted benefit therefore rides on the compute term. If this
    ever stops holding, the comm accounting has changed and the model needs a
    fresh look rather than a tweak.
    """

    profile, seq = flux
    plain = comm_bytes(DiffletParallelConfig(tp_degree=4), profile=profile, seq=seq)
    with_sp = comm_bytes(
        DiffletParallelConfig(tp_degree=4, sp_enabled=True), profile=profile, seq=seq
    )
    assert with_sp.total == pytest.approx(plain.total)


@pytest.mark.parametrize("cp", [2, 4, 8])
def test_ulysses_moves_two_over_cp_of_gather_kv(flux, cp):
    """gather_kv sends 2(cp-1)L; ulysses sends 4(cp-1)/cp L. Ratio 2/cp.

    So they tie at cp=2 and ulysses wins beyond it -- a sharper statement than
    the design doc's "a factor of cp less", which drops the (cp-1)/cp factors
    and the tensor counts.
    """

    profile, seq = flux
    gather = comm_bytes(
        DiffletParallelConfig(tp_degree=1, cp_degree=cp), profile=profile, seq=seq
    )
    ulysses = comm_bytes(
        DiffletParallelConfig(tp_degree=1, cp_degree=cp, cp_mode="ulysses"),
        profile=profile, seq=seq,
    )
    assert ulysses.context_parallel / gather.context_parallel == pytest.approx(2.0 / cp)


def test_ring_is_charged_less_than_gather_kv_for_the_same_volume(flux):
    profile, seq = flux
    gather = comm_bytes(DiffletParallelConfig(tp_degree=1, cp_degree=4), profile=profile, seq=seq)
    ring = comm_bytes(
        DiffletParallelConfig(tp_degree=1, cp_degree=4, cp_mode="ring"), profile=profile, seq=seq
    )
    assert ring.context_parallel == pytest.approx(
        gather.context_parallel * cost_model.RING_OVERLAP_RETENTION
    )


def test_cfg_parallel_adds_a_small_gather(flux):
    profile, seq = flux
    plain = comm_bytes(DiffletParallelConfig(tp_degree=2), profile=profile, seq=seq)
    with_cfg = comm_bytes(
        DiffletParallelConfig(tp_degree=2, cfg_parallel_enabled=True), profile=profile, seq=seq
    )
    assert with_cfg.cfg_parallel > 0
    assert with_cfg.cfg_parallel < plain.tensor_parallel


# ------------------------------------------------------------------ calibration


def test_single_anchor_reproduces_its_own_measurement(flux):
    """The anchor must predict itself exactly, or the fit is not a fit."""

    profile, seq = flux
    anchor_config = DiffletParallelConfig(tp_degree=4)
    calibration = calibrate(
        (_measurement("tp4", 0.2654),),
        profile=profile, seq=seq, parallel_of=lambda label: anchor_config,
    )
    assert calibration.kind == "measured-anchor"
    prediction = predict(
        anchor_config, profile=profile, seq=seq, calibration=calibration
    )
    assert prediction.step_seconds == pytest.approx(0.2654, rel=1e-6)


def test_two_anchors_fit_the_bandwidth_too(flux):
    profile, seq = flux
    configs = {
        "tp4": DiffletParallelConfig(tp_degree=4),
        "tp2cp2": DiffletParallelConfig(tp_degree=2, cp_degree=2),
    }
    calibration = calibrate(
        (_measurement("tp4", 0.2654), _measurement("tp2cp2", 0.2100)),
        profile=profile, seq=seq, parallel_of=configs.get,
    )
    assert calibration.kind == "measured-fit"
    assert not calibration.bandwidth_is_assumed
    for label, config in configs.items():
        expected = 0.2654 if label == "tp4" else 0.2100
        prediction = predict(config, profile=profile, seq=seq, calibration=calibration)
        assert prediction.step_seconds == pytest.approx(expected, rel=1e-6)


def test_degenerate_two_anchor_fit_falls_back_rather_than_inventing_a_bandwidth(flux):
    """Two anchors of the same configuration carry one data point, not two."""

    profile, seq = flux
    config = DiffletParallelConfig(tp_degree=4)
    calibration = calibrate(
        (_measurement("tp4", 0.2654), _measurement("tp4", 0.2654, source="second")),
        profile=profile, seq=seq, parallel_of=lambda label: config,
    )
    assert calibration.kind == "measured-anchor"


def test_no_anchors_is_flagged_uncalibrated(flux):
    profile, seq = flux
    calibration = calibrate((), profile=profile, seq=seq, parallel_of=lambda label: None)
    assert calibration.kind == "uncalibrated"
    assert calibration.compute_seconds_single_core > 0
    prediction = predict(
        DiffletParallelConfig(tp_degree=4), profile=profile, seq=seq, calibration=calibration
    )
    assert prediction.evidence == "predicted-uncalibrated"


def test_anchor_never_yields_a_negative_compute_term(flux):
    """An implausibly fast measurement must not produce a negative fit."""

    profile, seq = flux
    calibration = calibrate(
        (_measurement("tp4", 1e-9),),
        profile=profile, seq=seq, parallel_of=lambda label: DiffletParallelConfig(tp_degree=4),
    )
    assert calibration.compute_seconds_single_core > 0


# ------------------------------------------------------------------- prediction


def test_a_measurement_wins_over_the_model(flux):
    profile, seq = flux
    calibration = Calibration(
        compute_seconds_single_core=1.0, bandwidth_bytes_per_second=1e11, kind="measured-anchor"
    )
    measurement = _measurement("tp4", 0.9999)
    prediction = predict(
        DiffletParallelConfig(tp_degree=4), profile=profile, seq=seq,
        calibration=calibration, measurement=measurement,
    )
    assert prediction.evidence == "measured"
    assert prediction.step_seconds == 0.9999
    assert prediction.measurement is measurement


def test_prediction_splits_compute_from_communication(flux):
    profile, seq = flux
    calibration = Calibration(
        compute_seconds_single_core=1.0, bandwidth_bytes_per_second=1e11, kind="measured-anchor"
    )
    prediction = predict(
        DiffletParallelConfig(tp_degree=2, cp_degree=2), profile=profile, seq=seq,
        calibration=calibration,
    )
    assert prediction.step_seconds == pytest.approx(
        prediction.compute_seconds + prediction.comm_seconds
    )
    assert prediction.comm_seconds > 0
    assert math.isfinite(prediction.step_seconds)
