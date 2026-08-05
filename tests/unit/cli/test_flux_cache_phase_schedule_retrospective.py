from __future__ import annotations

from pathlib import Path

import pytest

from scripts.flux_cache_phase_schedule_retrospective import (
    DEFAULT_INPUTS,
    _reconcile_registered_curve,
    build_report,
    load_report,
    render_memo,
)

ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = (
    ROOT / "benchmark" / "flux_cache" / "phase-schedule-a1-retrospective.json"
)


def test_frozen_a1_retrospective_has_request_level_evidence_and_limits():
    report = load_report(REPORT_PATH)

    curve = report["extreme_pressure_rescue"]["R_by_terminal_step"]
    assert [(row["terminal_step"], row["rescued_count"]) for row in curve] == [
        (7, 12),
        (13, 12),
        (21, 12),
        (29, 8),
        (37, 1),
    ]
    assert len(report["extreme_pressure_rescue"]["per_request"]) == 12
    diagnosis = report["extreme_pressure_rescue"]["semantic_diagnosis"]
    assert diagnosis["step29_unrescued_sample_ids"] == [
        "p010-s2",
        "p019-s2",
        "p033-s2",
        "p035-s2",
    ]
    assert diagnosis["gate_use_permitted"] is False

    introduced = report["introduced_failures"]["step29_full_population"]["summary"]
    assert introduced["introduced_vqa_failure_sample_ids"] == ["p013-s2", "p039-s2"]
    assert introduced["introduced_contract_failure_count"] is None
    assert introduced["contract_status"] == "not_evaluable_missing_image_reward"

    adjudication = report["contradiction_adjudication"]
    assert adjudication["opened_stage_oil_family"]["unique_failure_request_count"] == 2
    assert adjudication["opened_stage_oil_family"][
        "terminal_intervention_target_count"
    ] == 6
    assert report["decision"]["a2_still_required"] is True
    assert report["evidence_role"]["hardware_calls"] == 0


def test_a1_registered_summary_reconciliation_preserves_small_discrepancy():
    report = load_report(REPORT_PATH)
    reconciliation = report["extreme_pressure_rescue"][
        "registered_summary_reconciliation"
    ]

    assert reconciliation["all_source_and_rescue_counts_match"] is True
    assert reconciliation["all_mean_brake_benefits_match_exactly"] is False
    step37 = reconciliation["rows"][-1]
    assert step37["terminal_step"] == 37
    assert step37["mean_brake_benefit_difference"] == pytest.approx(
        -4.069010416666782e-05
    )


def test_a1_report_rebuild_matches_snapshot_when_historical_artifacts_are_present():
    if not all(path.is_file() for path in DEFAULT_INPUTS.values()):
        pytest.skip("frozen A1 input results are not available")
    report = load_report(REPORT_PATH)
    required_paths = [
        Path(row["path"]) for row in report["required_artifact_audit"]
    ]
    if not all(path.is_file() for path in required_paths):
        pytest.skip("bound historical A1 artifacts are not available on this machine")

    assert build_report() == report
    memo = render_memo(report)
    assert "2 independent failure requests" in memo
    assert "full-contract introduced-failure count is `null`" in memo


def test_reconciliation_rejects_a_different_terminal_grid():
    with pytest.raises(ValueError, match="grids differ"):
        _reconcile_registered_curve(
            [
                {
                    "terminal_step": 7,
                    "source_failure_count": 1,
                    "rescued_count": 1,
                    "mean_brake_benefit": 0.1,
                }
            ],
            [
                {
                    "terminal_step": 13,
                    "source_failure_count": 1,
                    "rescued_count": 1,
                    "mean_brake_benefit": 0.1,
                }
            ],
        )
