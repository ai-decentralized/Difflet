#!/usr/bin/env python3
"""Verify the HV-1.5 precision schedule research gate from frontier artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedules", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epsilon", type=float, default=5e-5)
    parser.add_argument("--min-coverage", type=float, default=0.50)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    schedules = json.loads(args.schedules.read_text(encoding="utf-8"))
    selective = [entry for entry in schedules["schedules"] if entry["tau"] is not None]
    coverage_candidates = [
        entry for entry in selective if entry["mx_coverage"] >= args.min_coverage
    ]
    proxy_candidates = [
        entry
        for entry in coverage_candidates
        if entry["proxy_mean_cell_cosine"] >= 1.0 - args.epsilon
    ]
    full_dit_required = bool(proxy_candidates)
    result = {
        "schema_version": 1,
        "schedules": str(args.schedules),
        "epsilon": args.epsilon,
        "min_coverage": args.min_coverage,
        "selective_count": len(selective),
        "coverage_candidate_count": len(coverage_candidates),
        "proxy_candidate_count": len(proxy_candidates),
        "full_dit_run": False,
        "full_dit_required": full_dit_required,
        "h1_pass": False,
        "status": "h1_failed_coverage_precondition"
        if not coverage_candidates
        else "h1_failed_proxy_quality_precondition",
        "frontier": [
            {
                "name": entry["name"],
                "tau": entry["tau"],
                "mx_coverage": entry["mx_coverage"],
                "proxy_mean_cell_cosine": entry["proxy_mean_cell_cosine"],
                "proxy_min_cell_cosine": entry["proxy_min_cell_cosine"],
            }
            for entry in schedules["schedules"]
        ],
    }
    _write_json(args.out, result)
    print(json.dumps({k: result[k] for k in ("status", "h1_pass")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
