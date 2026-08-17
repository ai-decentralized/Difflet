from __future__ import annotations

import argparse
import json

import pytest

from difflet.offline.cache_profile.provenance import canonical_sha256, sha256_file
from difflet.offline.cache_profile.segment_risk import (
    RESULT_SCHEMA,
    build_study,
    request_signals,
    run_study,
    segment_risks,
    separation_diagnostic,
)


CANDIDATE_ID = "flux-a12-fixed"
ANCHORS = (0, 1, 2, 3, 4, 5, 9, 15)


def _entry(
    previous: int | None,
    anchor: int,
    *,
    z: float | None = 0.5,
    exposure: float = 0.2,
    status: str = "measured",
    region: str = "middle",
) -> dict[str, object]:
    return {
        "previous_anchor_step_index": previous,
        "anchor_step_index": anchor,
        "num_steps": 50,
        "policy_region": region,
        "estimate_status": status,
        "numerically_valid": status == "measured",
        "endpoint_z": z,
        "scheduler_abs_delta_sigma": exposure,
    }


def _trace(entries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "difflet-cache-anchor-error-trace",
        "schema_revision": 1,
        "measurement": "taylor_estimate_relative_error",
        "path_semantics": "logical_post_restore",
        "physical_rollback_attempts_included": False,
        "entries": entries,
        "candidate_binding": {"candidate_id": CANDIDATE_ID},
    }


def _fixed_trace(*, z_by_segment: dict[tuple[int, int], float]) -> dict[str, object]:
    entries = [_entry(None, ANCHORS[0], z=None, exposure=0.0, status="history_not_ready")]
    for previous, anchor in zip(ANCHORS[:-1], ANCHORS[1:], strict=True):
        entries.append(
            _entry(
                previous,
                anchor,
                z=z_by_segment.get((previous, anchor), 0.4),
                exposure=0.1 * (anchor - previous - 1),
            )
        )
    return _trace(entries)


def _evaluation(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "difflet-flux-cache-natural-range-evaluation",
        "natural_range_limits": {"image_reward": 0.8, "vqa_score": 0.25},
        "rows": rows,
    }


def _row(
    sample_id: str,
    *,
    image_reward_harm: float,
    vqa_harm: float = 0.0,
) -> dict[str, object]:
    failed = []
    if image_reward_harm > 0.8:
        failed.append("image_reward")
    if vqa_harm > 0.25:
        failed.append("vqa_score")
    return {
        "candidate_id": CANDIDATE_ID,
        "sample_id": sample_id,
        "prompt_index": int(sample_id[1:4]),
        "seed": 0,
        "harms": {"image_reward": image_reward_harm, "vqa_score": vqa_harm},
        "failed_metrics": failed,
    }


def _manifest(traces: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "schema": "difflet-flux-cache-quality-input-v1",
        "hardware_measured": True,
        "comparisons": [
            {
                "candidate_id": CANDIDATE_ID,
                "sample_id": sample_id,
                "anchor_error_trace": trace,
            }
            for sample_id, trace in sorted(traces.items())
        ],
    }


def test_segment_risks_drop_the_opening_anchor_and_scale_z_by_exposure():
    trace = _trace(
        [
            _entry(None, 0, z=None, exposure=0.0, status="history_not_ready"),
            _entry(0, 1, z=0.30, exposure=0.0),
            _entry(1, 5, z=0.50, exposure=0.4),
        ]
    )

    risks = segment_risks(trace)

    assert [risk.segment_id for risk in risks] == ["0->1", "1->5"]
    assert risks[0].skipped_step_count == 0
    assert risks[0].risk == pytest.approx(0.0)
    assert risks[1].skipped_step_count == 3
    assert risks[1].risk == pytest.approx(0.2)


def test_unmeasured_segments_are_scoreless_and_fail_closed():
    trace = _trace(
        [
            _entry(None, 0, z=None, exposure=0.0, status="history_not_ready"),
            _entry(0, 5, z=0.5, exposure=0.4),
            _entry(5, 9, z=None, exposure=0.3, status="invalid_estimated_output"),
        ]
    )

    signals = request_signals(segment_risks(trace))

    assert signals["scored_segment_count"] == 1
    assert signals["unmeasured_segment_ids"] == ["5->9"]
    assert signals["fail_closed_by_numerics"] is True
    assert signals["max_endpoint_z"] == pytest.approx(0.5)
    assert signals["cumulative_risk_ledger"] == pytest.approx(0.2)
    assert signals["total_scheduler_exposure"] == pytest.approx(0.7)


def test_separation_diagnostic_reports_overlap_rather_than_a_threshold():
    overlapping = separation_diagnostic([1.0, 3.0, 2.0, 4.0], [False, False, True, True])
    disjoint = separation_diagnostic([1.0, 2.0, 3.0, 4.0], [False, False, True, True])

    assert overlapping["separable"] is False
    assert overlapping["failure_minimum"] == pytest.approx(2.0)
    assert overlapping["pass_maximum"] == pytest.approx(3.0)
    assert disjoint["separable"] is True
    assert disjoint["passes_below_failure_minimum_fraction"] == pytest.approx(1.0)
    assert "threshold" not in disjoint


def test_study_stops_when_a_scored_request_has_no_segment_trace():
    evaluation = _evaluation([_row("p000-s0", image_reward_harm=0.1), _row("p001-s0", image_reward_harm=0.1)])
    manifest = _manifest({"p000-s0": _fixed_trace(z_by_segment={})})

    study = build_study(
        evaluation,
        manifest,
        candidate_id=CANDIDATE_ID,
        split_role="development",
        minimum_positives=1,
    )

    assert study["decision"] == "stop_missing_segment_traces"
    assert study["untraced_sample_ids"] == ["p001-s0"]
    assert study["threshold_calibration_authorized"] is False


def test_study_stops_when_requests_do_not_share_one_fixed_schedule():
    evaluation = _evaluation([_row("p000-s0", image_reward_harm=0.1), _row("p001-s0", image_reward_harm=0.1)])
    shifted = _trace(
        [
            _entry(None, 0, z=None, exposure=0.0, status="history_not_ready"),
            _entry(0, 7, z=0.4, exposure=0.6),
        ]
    )
    manifest = _manifest({"p000-s0": _fixed_trace(z_by_segment={}), "p001-s0": shifted})

    study = build_study(
        evaluation,
        manifest,
        candidate_id=CANDIDATE_ID,
        split_role="development",
        minimum_positives=1,
    )

    assert study["decision"] == "stop_segment_schedule_not_fixed"
    assert study["observed_schedule_count"] == 2
    assert study["threshold_calibration_authorized"] is False


def test_study_reports_diagnostics_but_refuses_calibration_on_one_failure():
    rows = [_row(f"p{index:03d}-s0", image_reward_harm=0.1) for index in range(5)]
    rows.append(_row("p005-s0", image_reward_harm=0.95))
    traces = {
        str(row["sample_id"]): _fixed_trace(
            z_by_segment={(9, 15): 1.4 if row["failed_metrics"] else 0.3}
        )
        for row in rows
    }

    study = build_study(
        _evaluation(rows),
        _manifest(traces),
        candidate_id=CANDIDATE_ID,
        split_role="development",
        minimum_positives=6,
    )

    assert study["decision"] == "stop_insufficient_quality_positives"
    assert study["contract_failure_count"] == 1
    assert study["threshold_calibration_authorized"] is False
    diagnostics = study["diagnostics"]
    assert diagnostics["fixed_segment_ids"][-1] == "9->15"
    assert diagnostics["request_signal_separation"]["max_segment_risk"]["separable"] is True
    correlations = {
        (row["signal"], row["harm"]): row for row in diagnostics["signal_harm_correlation"]
    }
    assert correlations[("max_endpoint_z", "image_reward_harm")]["paired_count"] == len(rows)


def test_sufficient_development_positives_authorize_a_later_threshold_step():
    rows = [_row(f"p{index:03d}-s0", image_reward_harm=0.1) for index in range(6)]
    rows.extend(_row(f"p1{index:02d}-s0", image_reward_harm=0.95) for index in range(6))
    traces = {
        str(row["sample_id"]): _fixed_trace(
            z_by_segment={(9, 15): 1.4 if row["failed_metrics"] else 0.3}
        )
        for row in rows
    }

    development = build_study(
        _evaluation(rows),
        _manifest(traces),
        candidate_id=CANDIDATE_ID,
        split_role="development",
        minimum_positives=6,
    )
    holdout = build_study(
        _evaluation(rows),
        _manifest(traces),
        candidate_id=CANDIDATE_ID,
        split_role="holdout",
        minimum_positives=6,
    )

    assert development["decision"] == "paired_diagnostics_complete"
    assert development["threshold_calibration_authorized"] is True
    assert holdout["decision"] == "paired_diagnostics_complete"
    assert holdout["threshold_calibration_authorized"] is False
    summaries = {row["segment_id"]: row for row in development["diagnostics"]["segment_summaries"]}
    assert summaries["9->15"]["endpoint_z"]["contract_failure"]["median"] == pytest.approx(1.4)
    assert summaries["9->15"]["endpoint_z"]["contract_pass"]["median"] == pytest.approx(0.3)
    assert summaries["0->1"]["skipped_step_count"] == 0


def test_run_study_rejects_an_unbound_quality_manifest(tmp_path):
    manifest_path = tmp_path / "quality.json"
    manifest_path.write_text(json.dumps(_manifest({})), encoding="utf-8")
    payload = {
        **_evaluation([_row("p000-s0", image_reward_harm=0.1)]),
        "quality_manifest": {
            "path": str(manifest_path),
            "file_sha256": "0" * 64,
        },
    }
    evaluation_path = tmp_path / "natural-range.json"
    evaluation_path.write_text(
        json.dumps({**payload, "sha256": canonical_sha256(payload)}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="quality-manifest binding is invalid"):
        run_study(
            argparse.Namespace(
                natural_range_evaluation=str(evaluation_path),
                candidate_id=CANDIDATE_ID,
                split_role="development",
                minimum_quality_positives=6,
                out=None,
            )
        )


def test_run_study_writes_a_hash_bound_result(tmp_path):
    rows = [_row("p000-s0", image_reward_harm=0.1), _row("p001-s0", image_reward_harm=0.95)]
    traces = {
        str(row["sample_id"]): _fixed_trace(
            z_by_segment={(9, 15): 1.4 if row["failed_metrics"] else 0.3}
        )
        for row in rows
    }
    manifest_path = tmp_path / "quality.json"
    manifest_path.write_text(json.dumps(_manifest(traces)), encoding="utf-8")
    payload = {
        **_evaluation(rows),
        "split": "development-48",
        "bucket_id": "square-1024",
        "quality_manifest": {
            "path": str(manifest_path),
            "file_sha256": sha256_file(manifest_path),
        },
    }
    evaluation_path = tmp_path / "natural-range.json"
    evaluation_path.write_text(
        json.dumps({**payload, "sha256": canonical_sha256(payload)}),
        encoding="utf-8",
    )
    out_path = tmp_path / "segment-risk.json"

    result = run_study(
        argparse.Namespace(
            natural_range_evaluation=str(evaluation_path),
            candidate_id=CANDIDATE_ID,
            split_role="development",
            minimum_quality_positives=6,
            out=str(out_path),
        )
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["schema"] == RESULT_SCHEMA
    assert written["sha256"] == canonical_sha256(result)
    assert written["sources"]["split"] == "development-48"
    assert written["study"]["decision"] == "stop_insufficient_quality_positives"
    assert written["study"]["threshold_calibration_authorized"] is False
