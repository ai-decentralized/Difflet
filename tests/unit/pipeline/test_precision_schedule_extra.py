"""Extra tests for difflet.pipeline.precision_schedule targeting branches not
covered by tests/unit/pipeline/test_precision_schedule.py."""

import pytest

from difflet.pipeline.precision_schedule import (
    PRECISION_BF16,
    PRECISION_MXFP8_E4M3,
    PRECISION_MXFP8_E5M2,
    PrecisionSchedule,
    _cell_cosines,
    _normalize_precision,
    extreme_schedule,
    synthesize_two_threshold_schedule,
)


def test_normalize_precision_aliases():
    assert _normalize_precision("bfloat16") == PRECISION_BF16
    assert _normalize_precision("e4m3") == PRECISION_MXFP8_E4M3
    assert _normalize_precision("e5m2") == PRECISION_MXFP8_E5M2


def test_normalize_precision_rejects_unknown():
    with pytest.raises(ValueError):
        _normalize_precision("int4")


def test_empty_schedule_coverage_defaults():
    sched = PrecisionSchedule(model_id="m", bundle="b", tau=None, assignments={})
    assert sched.mx_coverage == 0.0
    assert sched.coverage_by_precision == {}


def test_coverage_by_precision_counts():
    sched = PrecisionSchedule(
        model_id="m",
        bundle="b",
        tau=None,
        assignments={"0:q": "e4m3", "0:k": "e5m2", "1:q": "bf16", "1:k": "bf16"},
    )
    cov = sched.coverage_by_precision
    assert cov[PRECISION_BF16] == 0.5
    assert cov[PRECISION_MXFP8_E4M3] == 0.25
    assert cov[PRECISION_MXFP8_E5M2] == 0.25
    assert sched.mx_coverage == 0.5


def test_precision_for_lookup():
    sched = PrecisionSchedule(
        model_id="m", bundle="b", tau=None, assignments={"3:proj": "e4m3"}
    )
    assert sched.precision_for(3, "proj") == PRECISION_MXFP8_E4M3


def test_from_dict_rejects_bad_schema_version():
    with pytest.raises(ValueError):
        PrecisionSchedule.from_dict({"schema_version": 999, "model_id": "m"})


def test_cell_cosines_schema2():
    row = {"block": 0, "linear": "q", "metrics": {"e4m3": {"cosine": 0.99}, "e5m2": {"cosine": 0.97}}}
    e4m3, e5m2 = _cell_cosines(row)
    assert e4m3 == pytest.approx(0.99)
    assert e5m2 == pytest.approx(0.97)


def test_cell_cosines_schema1_returns_none_for_e5m2():
    row = {"block": 0, "linear": "q", "cosine": 0.95}
    e4m3, e5m2 = _cell_cosines(row)
    assert e4m3 == pytest.approx(0.95)
    assert e5m2 is None


def test_two_threshold_assigns_all_three_levels():
    rows = [
        {"block": 0, "linear": "hi", "metrics": {"e4m3": {"cosine": 0.999}, "e5m2": {"cosine": 0.99}}},
        {"block": 0, "linear": "mid", "metrics": {"e4m3": {"cosine": 0.90}, "e5m2": {"cosine": 0.98}}},
        {"block": 0, "linear": "lo", "metrics": {"e4m3": {"cosine": 0.5}, "e5m2": {"cosine": 0.5}}},
    ]
    sched = synthesize_two_threshold_schedule(
        rows, tau_hi=0.995, tau_lo=0.95, model_id="m", bundle="b"
    )
    assert sched.precision_for(0, "hi") == PRECISION_MXFP8_E4M3
    assert sched.precision_for(0, "mid") == PRECISION_MXFP8_E5M2
    assert sched.precision_for(0, "lo") == PRECISION_BF16
    assert sched.metadata["tau_hi"] == 0.995
    assert sched.metadata["tau_lo"] == 0.95


def test_rows_dict_requires_rows_field():
    with pytest.raises(ValueError):
        extreme_schedule({"not_rows": []}, PRECISION_BF16, model_id="m", bundle="b")
