#!/usr/bin/env python3
"""Audit F3.0b requirement coverage against cclog 63.

This is a requirement-level audit, not the F3.1 gate itself. It answers:

* did the upgraded profiler rerun Qwen-Image 1024 and split the required timing
  surfaces as far as this runtime exposes them?
* did neuron-profile fill the transfer and NEFF hardware-counter gaps?
* is the production gate complete, unlockable, or still partial evidence?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACCEPTED_NEFF_SOURCES = {
    "xla_metric:ExecuteReplicatedTime",
    "xla_metric:ExecuteTime",
    "neuron_profile:nrt_execute_max_worker_duration",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _all_steps_have_number(steps: list[dict[str, Any]], key: str) -> bool:
    return bool(steps) and all(_is_number(step.get(key)) for step in steps)


def _requirement(
    requirement_id: str,
    status: str,
    satisfied: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": requirement_id,
        "status": status,
        "satisfied": satisfied,
        "evidence": evidence,
    }


def _direct_xla_evidence(direct_steps: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = sorted(
        {
            name
            for step in direct_steps
            for name in step.get("xla_metric_names_available", [])
        }
    )
    counters = sorted(
        {
            name
            for step in direct_steps
            for name in step.get("xla_counter_names_available", [])
        }
    )
    parsed_keys = sorted(
        {
            key
            for step in direct_steps
            for key in step.get("xla_parsed_metric_keys", [])
        }
    )
    return {
        "available_metrics": metrics,
        "available_counters": counters,
        "parsed_key_sample": parsed_keys[:20],
    }


def _neff_sources(steps: list[dict[str, Any]]) -> list[str]:
    return sorted({str(step.get("neff_execution_time_source")) for step in steps})


def audit_requirement_coverage(
    *,
    direct_profile: dict[str, Any],
    combined_profile: dict[str, Any],
    gate_audit: dict[str, Any],
    artifact_inventory: dict[str, Any],
) -> dict[str, Any]:
    direct_summary = direct_profile.get("summary") or {}
    combined_summary = combined_profile.get("summary") or {}
    direct_steps = direct_profile.get("steps") or []
    combined_steps = combined_profile.get("steps") or []
    direct_num_steps = direct_summary.get("num_steps")
    combined_num_steps = combined_summary.get("num_steps")

    requirements: list[dict[str, Any]] = []
    requirements.append(
        _requirement(
            "qwen_image_1024_direct_profiler_rerun",
            "satisfied" if direct_num_steps else "missing",
            bool(direct_num_steps),
            {
                "schema": direct_profile.get("schema"),
                "model": direct_profile.get("model"),
                "height": direct_profile.get("height"),
                "width": direct_profile.get("width"),
                "num_steps": direct_num_steps,
            },
        )
    )
    requirements.append(
        _requirement(
            "python_scheduler_segment_recorded",
            "satisfied" if _all_steps_have_number(direct_steps, "scheduler_time_s") else "missing",
            _all_steps_have_number(direct_steps, "scheduler_time_s"),
            {
                "mean_scheduler_time_s": direct_summary.get("mean_scheduler_time_s"),
                "scheduler_tensor_bytes_recorded": _all_steps_have_number(
                    direct_steps,
                    "scheduler_tensor_bytes",
                ),
            },
        )
    )
    requirements.append(
        _requirement(
            "mark_step_xla_sync_segment_recorded",
            "satisfied" if _all_steps_have_number(direct_steps, "mark_step_time_s") else "missing",
            _all_steps_have_number(direct_steps, "mark_step_time_s"),
            {"mean_mark_step_time_s": direct_summary.get("mean_mark_step_time_s")},
        )
    )

    xla_steps = direct_summary.get("xla_metrics_steps")
    xla_attempted = bool(direct_num_steps and xla_steps == direct_num_steps)
    xla_evidence = _direct_xla_evidence(direct_steps)
    requirements.append(
        _requirement(
            "direct_torch_xla_metrics_sampled_per_step",
            "satisfied" if xla_attempted else "missing",
            xla_attempted,
            {"xla_metrics_steps": xla_steps, "num_steps": direct_num_steps, **xla_evidence},
        )
    )

    direct_transfer = (
        direct_summary.get("total_host_device_transfer_count") is not None
        and direct_summary.get("total_host_device_transfer_bytes") is not None
    )
    requirements.append(
        _requirement(
            "direct_xla_transfer_bytes_count_available",
            "satisfied" if direct_transfer else "attempted_unavailable",
            bool(direct_transfer),
            {
                "total_host_device_transfer_count": direct_summary.get(
                    "total_host_device_transfer_count"
                ),
                "total_host_device_transfer_bytes": direct_summary.get(
                    "total_host_device_transfer_bytes"
                ),
                **xla_evidence,
            },
        )
    )

    direct_hardware_timed = int(direct_summary.get("hardware_timed_steps") or 0)
    requirements.append(
        _requirement(
            "direct_xla_neff_execution_counter_available",
            "satisfied" if direct_hardware_timed else "attempted_unavailable",
            bool(direct_hardware_timed),
            {
                "hardware_timed_steps": direct_hardware_timed,
                "neff_execution_time_sources": _neff_sources(direct_steps),
                **xla_evidence,
            },
        )
    )

    combined_transfer = (
        combined_summary.get("total_host_device_transfer_count") is not None
        and combined_summary.get("total_host_device_transfer_bytes") is not None
    )
    requirements.append(
        _requirement(
            "neuron_profile_transfer_bytes_count_measured",
            "satisfied_via_neuron_profile" if combined_transfer else "missing",
            bool(combined_transfer),
            {
                "total_host_device_transfer_count": combined_summary.get(
                    "total_host_device_transfer_count"
                ),
                "total_host_device_transfer_bytes": combined_summary.get(
                    "total_host_device_transfer_bytes"
                ),
                "total_host_device_transfer_time_s": combined_summary.get(
                    "total_host_device_transfer_time_s"
                ),
            },
        )
    )

    combined_sources = set(_neff_sources(combined_steps))
    combined_hardware = (
        bool(combined_num_steps)
        and combined_summary.get("hardware_timed_steps") == combined_num_steps
        and combined_sources.issubset(ACCEPTED_NEFF_SOURCES)
    )
    requirements.append(
        _requirement(
            "neuron_profile_neff_execution_measured",
            "satisfied_via_neuron_profile" if combined_hardware else "missing",
            bool(combined_hardware),
            {
                "hardware_timed_steps": combined_summary.get("hardware_timed_steps"),
                "num_steps": combined_num_steps,
                "mean_neff_execution_time_s": combined_summary.get(
                    "mean_neff_execution_time_s"
                ),
                "neff_execution_time_sources": sorted(combined_sources),
            },
        )
    )

    recomputed_cpu_share = (
        combined_summary.get("mean_cpu_round_trip_share") is not None
        and combined_summary.get("mean_neff_execution_time_s") is not None
        and _all_steps_have_number(combined_steps, "non_neff_time_s")
    )
    requirements.append(
        _requirement(
            "cpu_round_trip_share_recomputed_from_neff",
            "satisfied" if recomputed_cpu_share else "missing",
            bool(recomputed_cpu_share),
            {
                "mean_cpu_round_trip_share": combined_summary.get(
                    "mean_cpu_round_trip_share"
                ),
                "steady_state_max_cpu_round_trip_share_after_step0": combined_summary.get(
                    "steady_state_max_cpu_round_trip_share_after_step0"
                ),
            },
        )
    )

    gate_partial = gate_audit.get("decision") == "remain_gated_partial_evidence"
    requirements.append(
        _requirement(
            "f3_1_gate_decision_recorded",
            gate_audit.get("decision") or "missing",
            bool(gate_audit.get("decision")),
            {
                "can_unlock_f3_1": gate_audit.get("can_unlock_f3_1"),
                "can_write_negative_closeout": gate_audit.get(
                    "can_write_negative_closeout"
                ),
                "measured_labels": gate_audit.get("measured_labels"),
                "missing_labels": gate_audit.get("missing_labels"),
            },
        )
    )

    qwen_profiler_ids = {
        "qwen_image_1024_direct_profiler_rerun",
        "python_scheduler_segment_recorded",
        "mark_step_xla_sync_segment_recorded",
        "direct_torch_xla_metrics_sampled_per_step",
        "neuron_profile_transfer_bytes_count_measured",
        "neuron_profile_neff_execution_measured",
        "cpu_round_trip_share_recomputed_from_neff",
    }
    qwen_profiler_satisfied = all(
        item["satisfied"] for item in requirements if item["id"] in qwen_profiler_ids
    )

    return {
        "schema": "nova-f3-0b-requirement-coverage-v1",
        "qwen_profiler_requirement_satisfied": qwen_profiler_satisfied,
        "production_gate_complete": bool(gate_audit.get("can_unlock_f3_1"))
        or bool(gate_audit.get("can_write_negative_closeout")),
        "production_gate_status": gate_audit.get("decision"),
        "production_missing_labels": gate_audit.get("missing_labels", []),
        "artifact_ready_labels": artifact_inventory.get("ready_labels", []),
        "artifact_missing_labels": artifact_inventory.get("missing_labels", []),
        "disk_available_gb": artifact_inventory.get("disk_available_gb"),
        "direct_xla_runtime_transfer_neff_available": bool(
            direct_transfer and direct_hardware_timed
        ),
        "neuron_profile_fallback_required_for_qwen": bool(
            xla_attempted and not direct_transfer and not direct_hardware_timed
        ),
        "requirements": requirements,
        "overall_status": (
            "qwen_profiler_satisfied_gate_partial_evidence"
            if qwen_profiler_satisfied and gate_partial
            else "production_gate_closed"
            if gate_audit.get("can_unlock_f3_1") or gate_audit.get("can_write_negative_closeout")
            else "incomplete"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-profile", required=True)
    parser.add_argument("--combined-profile", required=True)
    parser.add_argument("--gate-audit", required=True)
    parser.add_argument("--artifact-inventory", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = audit_requirement_coverage(
        direct_profile=_load_json(args.direct_profile),
        combined_profile=_load_json(args.combined_profile),
        gate_audit=_load_json(args.gate_audit),
        artifact_inventory=_load_json(args.artifact_inventory),
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
