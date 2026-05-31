#!/usr/bin/env python3
"""Audit F3.0b combined profile JSONs against the F3.1 gate.

The F3.1 rule is intentionally conservative:

* any measured production model with steady-state CPU/non-NEFF share >= 10%
  unlocks F3.1;
* a negative closeout is allowed only when every required production label has
  a hardware-timed combined JSON and all are below 10%.

This script keeps those two decisions separate so partial evidence cannot be
mistaken for a full negative closeout.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLD = 0.10
COMBINED_SCHEMA = "nova-f3-denoise-loop-neuron-inspect-summary-v1"
ACCEPTED_NEFF_SOURCES = {
    "xla_metric:ExecuteReplicatedTime",
    "xla_metric:ExecuteTime",
    "neuron_profile:nrt_execute_max_worker_duration",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _label_for(path: Path, labels: list[str]) -> str:
    name = path.name
    for label in labels:
        if label in name:
            return label
    return path.stem


def _steady_state_share(summary: dict[str, Any]) -> float | None:
    value = summary.get("steady_state_max_cpu_round_trip_share_after_step0")
    if value is None:
        value = summary.get("max_cpu_round_trip_share")
    return float(value) if value is not None else None


def _is_hardware_measured(data: dict[str, Any], summary: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if data.get("schema") != COMBINED_SCHEMA:
        reasons.append("schema_is_not_neuron_inspect_summary")
    num_steps = summary.get("num_steps")
    hardware_timed_steps = summary.get("hardware_timed_steps")
    if not isinstance(num_steps, int) or num_steps <= 0:
        reasons.append("num_steps_missing_or_zero")
    if hardware_timed_steps != num_steps:
        reasons.append("not_all_steps_have_neuron_profile_execution_time")
    steps = data.get("steps")
    if not isinstance(steps, list) or len(steps) != num_steps:
        reasons.append("step_records_missing_or_count_mismatch")
    else:
        for step in steps:
            if step.get("neff_execution_time_s") is None:
                reasons.append("step_neff_execution_time_missing")
                break
            if step.get("neff_execution_time_source") not in ACCEPTED_NEFF_SOURCES:
                reasons.append("step_neff_execution_source_not_hardware_counter")
                break
    if summary.get("mean_neff_execution_time_s") is None:
        reasons.append("mean_neff_execution_time_s_missing")
    if summary.get("total_host_device_transfer_bytes") is None:
        reasons.append("host_device_transfer_bytes_missing")
    if summary.get("total_host_device_transfer_count") is None:
        reasons.append("host_device_transfer_count_missing")
    return not reasons, reasons


def audit(
    paths: list[Path],
    *,
    required_labels: list[str],
    threshold: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        data = _load_json(path)
        summary = data.get("summary") or {}
        share = _steady_state_share(summary)
        label = _label_for(path, required_labels)
        hardware_measured, invalid_reasons = _is_hardware_measured(data, summary)
        rows.append(
            {
                "label": label,
                "path": str(path),
                "schema": data.get("schema"),
                "num_steps": summary.get("num_steps"),
                "hardware_measured": hardware_measured,
                "invalid_reasons": invalid_reasons,
                "hardware_timed_steps": summary.get("hardware_timed_steps"),
                "mean_neff_execution_time_s": summary.get("mean_neff_execution_time_s"),
                "steady_state_max_cpu_round_trip_share": share,
                "mean_cpu_round_trip_share": summary.get("mean_cpu_round_trip_share"),
                "total_host_device_transfer_bytes": summary.get(
                    "total_host_device_transfer_bytes"
                ),
                "total_host_device_transfer_count": summary.get(
                    "total_host_device_transfer_count"
                ),
                "passes_threshold": bool(
                    hardware_measured and share is not None and share >= threshold
                ),
            }
        )

    measured_labels = {row["label"] for row in rows if row["hardware_measured"]}
    missing_labels = [
        label for label in required_labels if label not in measured_labels
    ]
    passing_rows = [row for row in rows if row["passes_threshold"]]
    return {
        "schema": "nova-f3-0b-gate-audit-v1",
        "threshold": threshold,
        "required_labels": required_labels,
        "measured_labels": sorted(measured_labels),
        "missing_labels": missing_labels,
        "rows": rows,
        "can_unlock_f3_1": bool(passing_rows),
        "unlocking_labels": [row["label"] for row in passing_rows],
        "can_write_negative_closeout": bool(required_labels)
        and not passing_rows
        and not missing_labels,
        "decision": (
            "unlock_f3_1"
            if passing_rows
            else "negative_closeout_allowed"
            if required_labels and not missing_labels
            else "remain_gated_partial_evidence"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combined",
        action="append",
        required=True,
        help="Combined JSON path or glob. May be passed multiple times.",
    )
    parser.add_argument(
        "--required-label",
        action="append",
        default=[],
        help="Production label that must be covered before negative closeout.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.combined:
        matches = [Path(path) for path in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    missing_paths = [str(path) for path in paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"missing combined JSON paths: {missing_paths}")

    result = audit(
        paths,
        required_labels=args.required_label,
        threshold=float(args.threshold),
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
