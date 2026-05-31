#!/usr/bin/env python3
"""Audit m9 TeaCache evidence and emit a deterministic gate decision."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


SCHEMA = "nova-m9-teacache-gate-audit-v1"
CALIBRATION_SCHEMA = "nova-m9-teacache-calibration-v1"
SPEEDUP_SCHEMA = "nova-m9-teacache-speedup-curve-v1"
INTEGRATION_SCHEMA = "nova-m9-teacache-integration-v1"

MIN_FIT_R2 = 0.90
MIN_SPEEDUP = 1.50
MIN_TRAJECTORY_COSINE = 0.9999
MIN_FINAL_COSINE = 0.9995


def _glob_many(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(path) for path in glob.glob(pattern)]
        paths.extend(matches if matches else [Path(pattern)])
    return sorted(set(paths))


def _load_json(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"load_error:{exc}"]


def _label(doc: dict[str, Any] | None, path: Path) -> str | None:
    if doc is not None and doc.get("model") is not None:
        return str(doc["model"])
    name = path.name.lower()
    if "hunyuan" in name:
        return "hunyuan_video"
    if "wan" in name:
        return "wan"
    if "flux" in name:
        return "flux"
    if "qwen" in name:
        return "qwen_image"
    return None


def _calibration_row(path: Path, thresholds: dict[str, float]) -> dict[str, Any]:
    doc, invalid = _load_json(path)
    fit_r2 = doc.get("fit_r2") if doc else None
    n_samples = doc.get("n_samples") if doc else None
    mod_input_source = doc.get("mod_input_source") if doc else None
    hardware_measured = doc.get("hardware_measured") if doc else None
    if doc and doc.get("schema") != CALIBRATION_SCHEMA:
        invalid.append(f"unsupported_schema:{doc.get('schema')}")
    if fit_r2 is None:
        invalid.append("missing_fit_r2")
    if n_samples is None or int(n_samples) <= 0:
        invalid.append("missing_n_samples")
    if mod_input_source != "block0_modulated_input":
        invalid.append(f"unsupported_mod_input_source:{mod_input_source}")
    if hardware_measured is not True:
        invalid.append("not_hardware_measured")
    passes = bool(not invalid and float(fit_r2) >= thresholds["min_fit_r2"])
    return {
        "kind": "calibration",
        "label": _label(doc, path),
        "path": str(path),
        "schema": doc.get("schema") if doc else None,
        "fit_r2": fit_r2,
        "n_samples": n_samples,
        "mod_input_source": mod_input_source,
        "hardware_measured": hardware_measured,
        "passes_gate": passes,
        "invalid_reasons": invalid,
    }


def _candidate_passes(candidate: dict[str, Any], thresholds: dict[str, float]) -> bool:
    speedup = candidate.get("measured_speedup", candidate.get("target_speedup"))
    trajectory = candidate.get("trajectory_cosine")
    final = candidate.get("final_cosine")
    return bool(
        candidate.get("hardware_measured") is True
        and speedup is not None
        and float(speedup) >= thresholds["min_speedup"]
        and trajectory is not None
        and float(trajectory) >= thresholds["min_trajectory_cosine"]
        and final is not None
        and float(final) >= thresholds["min_final_cosine"]
    )


def _speedup_row(path: Path, thresholds: dict[str, float]) -> dict[str, Any]:
    doc, invalid = _load_json(path)
    candidates = doc.get("candidates") if doc else None
    if doc and doc.get("schema") != SPEEDUP_SCHEMA:
        invalid.append(f"unsupported_schema:{doc.get('schema')}")
    if not isinstance(candidates, list) or not candidates:
        invalid.append("missing_candidates")
        candidates = []
    passing = [item for item in candidates if _candidate_passes(item, thresholds)]
    return {
        "kind": "speedup_curve",
        "label": _label(doc, path),
        "path": str(path),
        "schema": doc.get("schema") if doc else None,
        "passing_targets": [item.get("target_speedup") for item in passing],
        "passes_gate": bool(not invalid and passing),
        "invalid_reasons": invalid,
    }


def _integration_row(path: Path, thresholds: dict[str, float]) -> dict[str, Any]:
    doc, invalid = _load_json(path)
    if doc and doc.get("schema") != INTEGRATION_SCHEMA:
        invalid.append(f"unsupported_schema:{doc.get('schema')}")
    wallclock_speedup = doc.get("wallclock_speedup") if doc else None
    trajectory = doc.get("trajectory_cosine") if doc else None
    final = doc.get("final_cosine") if doc else None
    default_off = doc.get("default_off") if doc else None
    hardware_measured = doc.get("hardware_measured") if doc else None
    passes = bool(
        not invalid
        and wallclock_speedup is not None
        and float(wallclock_speedup) >= thresholds["min_speedup"]
        and trajectory is not None
        and float(trajectory) >= thresholds["min_trajectory_cosine"]
        and final is not None
        and float(final) >= thresholds["min_final_cosine"]
        and default_off is True
        and hardware_measured is True
    )
    return {
        "kind": "integration",
        "label": _label(doc, path),
        "path": str(path),
        "schema": doc.get("schema") if doc else None,
        "wallclock_speedup": wallclock_speedup,
        "trajectory_cosine": trajectory,
        "final_cosine": final,
        "default_off": default_off,
        "hardware_measured": hardware_measured,
        "passes_gate": passes,
        "invalid_reasons": invalid,
    }


def audit(
    *,
    calibration_paths: list[Path],
    speedup_paths: list[Path],
    integration_paths: list[Path],
    required_labels: list[str],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or {
        "min_fit_r2": MIN_FIT_R2,
        "min_speedup": MIN_SPEEDUP,
        "min_trajectory_cosine": MIN_TRAJECTORY_COSINE,
        "min_final_cosine": MIN_FINAL_COSINE,
    }
    rows = (
        [_calibration_row(path, thresholds) for path in calibration_paths]
        + [_speedup_row(path, thresholds) for path in speedup_paths]
        + [_integration_row(path, thresholds) for path in integration_paths]
    )
    by_label: dict[str, set[str]] = {}
    for row in rows:
        if row["label"] is not None and row["passes_gate"]:
            by_label.setdefault(row["label"], set()).add(row["kind"])

    unlock_labels = sorted(
        label
        for label in required_labels
        if {"calibration", "speedup_curve"}.issubset(by_label.get(label, set()))
    )
    close_labels = sorted(
        label
        for label in required_labels
        if {"calibration", "speedup_curve", "integration"}.issubset(
            by_label.get(label, set())
        )
    )
    measured_labels = sorted({row["label"] for row in rows if row["label"] is not None})
    missing_labels = [label for label in required_labels if label not in measured_labels]
    can_unlock = bool(unlock_labels)
    can_close = bool(close_labels)
    if can_close:
        decision = "close_t1"
    elif can_unlock:
        decision = "unlock_t1"
    elif rows:
        decision = "remain_gated_partial_evidence"
    else:
        decision = "no_evidence"
    return {
        "schema": SCHEMA,
        "thresholds": dict(thresholds),
        "required_labels": required_labels,
        "measured_labels": measured_labels,
        "missing_labels": missing_labels,
        "can_unlock_t1": can_unlock,
        "can_close_t1": can_close,
        "unlocking_labels": unlock_labels,
        "closing_labels": close_labels,
        "decision": decision,
        "rows": rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", action="append", default=[])
    parser.add_argument("--speedup-curve", action="append", default=[])
    parser.add_argument("--integration", action="append", default=[])
    parser.add_argument("--required-label", action="append", default=["hunyuan_video"])
    # Policy thresholds. Defaults are the strict cclog 67 D2/D3 gates. Override
    # to TeaCache's published acceptance regime (e.g. 0.998 trajectory cosine)
    # only as an explicit, recorded decision — the chosen values are written
    # into the audit JSON's "thresholds" block.
    parser.add_argument("--min-fit-r2", type=float, default=MIN_FIT_R2)
    parser.add_argument("--min-speedup", type=float, default=MIN_SPEEDUP)
    parser.add_argument(
        "--min-trajectory-cosine", type=float, default=MIN_TRAJECTORY_COSINE
    )
    parser.add_argument("--min-final-cosine", type=float, default=MIN_FINAL_COSINE)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = audit(
        calibration_paths=_glob_many(args.calibration),
        speedup_paths=_glob_many(args.speedup_curve),
        integration_paths=_glob_many(args.integration),
        required_labels=sorted(set(args.required_label)),
        thresholds={
            "min_fit_r2": float(args.min_fit_r2),
            "min_speedup": float(args.min_speedup),
            "min_trajectory_cosine": float(args.min_trajectory_cosine),
            "min_final_cosine": float(args.min_final_cosine),
        },
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
