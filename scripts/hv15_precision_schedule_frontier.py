#!/usr/bin/env python3
"""Build HV-1.5 precision schedules from a calibration table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import itertools  # noqa: E402

from difflet.pipeline.precision_schedule import (  # noqa: E402
    PRECISION_BF16,
    PRECISION_MXFP8_E4M3,
    PRECISION_MXFP8_E5M2,
    PrecisionSchedule,
    _cell_cosines,
    extreme_schedule,
    schedule_frontier,
    two_threshold_frontier,
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _schedule_name(schedule: PrecisionSchedule) -> str:
    meta = schedule.metadata or {}
    if "tau_hi" in meta and "tau_lo" in meta:
        return f"selective-hi{meta['tau_hi']:.4f}-lo{meta['tau_lo']:.4f}"
    if schedule.tau is None:
        cov = schedule.coverage_by_precision
        if schedule.mx_coverage == 0.0:
            return "all-bf16"
        if cov.get(PRECISION_MXFP8_E4M3, 0.0) == 1.0:
            return "all-e4m3"
        if cov.get(PRECISION_MXFP8_E5M2, 0.0) == 1.0:
            return "all-e5m2"
        return "extreme"
    return f"selective-tau-{schedule.tau:.4f}"


def _proxy_quality(
    schedule: PrecisionSchedule,
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Calibration-table proxy quality for standalone schedule ranking.

    BF16 cells contribute perfect local cosine. MX cells contribute their
    measured per-cell MX-vs-BF16 cosine. This is not a full-DiT cosine; it is
    a reproducible frontier proxy used by the calibration stage.
    """

    values = []
    for row in rows:
        key = f"{int(row['block'])}:{row['linear']}"
        precision = schedule.assignments[key]
        e4m3_cos, e5m2_cos = _cell_cosines(row)
        if precision == PRECISION_BF16:
            values.append(1.0)
        elif precision == PRECISION_MXFP8_E5M2:
            values.append(e5m2_cos if e5m2_cos is not None else e4m3_cos)
        else:
            values.append(e4m3_cos)
    return {
        "proxy_min_cell_cosine": min(values),
        "proxy_mean_cell_cosine": sum(values) / len(values),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-table", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--model-id",
        default="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    )
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--taus", nargs="+", type=float, default=[0.9990, 0.9995, 0.9997])
    parser.add_argument(
        "--two-threshold",
        action="store_true",
        help="emit the 3-level (bf16/e4m3/e5m2) two-threshold frontier",
    )
    parser.add_argument(
        "--tau-hi",
        nargs="+",
        type=float,
        default=[0.9990, 0.9995, 0.9997],
        help="E4M3 acceptance thresholds (two-threshold mode)",
    )
    parser.add_argument(
        "--tau-lo",
        nargs="+",
        type=float,
        default=[0.9980, 0.9990, 0.9995],
        help="E5M2 acceptance thresholds (two-threshold mode)",
    )
    parser.add_argument("--persist-best", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    table = json.loads(args.calibration_table.read_text(encoding="utf-8"))
    rows = list(table["rows"])
    bundle = args.bundle or str(table.get("bundle", "unknown"))
    if args.two_threshold:
        tau_pairs = list(itertools.product(args.tau_hi, args.tau_lo))
        schedules = two_threshold_frontier(
            rows,
            tau_pairs,
            model_id=args.model_id,
            bundle=bundle,
        )
    else:
        schedules = schedule_frontier(
            rows,
            args.taus,
            model_id=args.model_id,
            bundle=bundle,
        )
    entries = []
    all_bf16 = extreme_schedule(rows, PRECISION_BF16, model_id=args.model_id, bundle=bundle)
    all_mx = extreme_schedule(rows, PRECISION_MXFP8_E4M3, model_id=args.model_id, bundle=bundle)
    all_e5m2 = extreme_schedule(
        rows, PRECISION_MXFP8_E5M2, model_id=args.model_id, bundle=bundle
    )
    for schedule in schedules:
        quality = _proxy_quality(schedule, rows)
        meta = schedule.metadata or {}
        entries.append(
            {
                "name": _schedule_name(schedule),
                "tau": schedule.tau,
                "tau_hi": meta.get("tau_hi"),
                "tau_lo": meta.get("tau_lo"),
                "mx_coverage": schedule.mx_coverage,
                "coverage_by_precision": schedule.coverage_by_precision,
                **quality,
                "schedule": schedule.to_dict(),
            }
        )

    selective = [entry for entry in entries if entry["tau"] is not None]
    best = max(
        selective,
        key=lambda item: (item["proxy_mean_cell_cosine"], item["mx_coverage"]),
    )
    h1_proxy_pass = (
        best["proxy_mean_cell_cosine"] >= 1.0 - 5e-5 and best["mx_coverage"] >= 0.50
    )
    result = {
        "schema_version": 1,
        "calibration_table": str(args.calibration_table),
        "model_id": args.model_id,
        "bundle": bundle,
        "two_threshold": bool(args.two_threshold),
        "taus": args.taus,
        "tau_hi": args.tau_hi if args.two_threshold else None,
        "tau_lo": args.tau_lo if args.two_threshold else None,
        "all_bf16_proxy": _proxy_quality(all_bf16, rows),
        "all_mx_proxy": _proxy_quality(all_mx, rows),
        "all_e5m2_proxy": _proxy_quality(all_e5m2, rows),
        "best_selective_name": best["name"],
        "best_selective_mx_coverage": best["mx_coverage"],
        "h1_proxy_pass": h1_proxy_pass,
        "note": "proxy quality is calibration-table aggregate, not full-DiT cosine",
        "schedules": entries,
    }
    if args.persist_best is not None:
        PrecisionSchedule.from_dict(best["schedule"]).write_json(args.persist_best)
        result["persisted_best"] = str(args.persist_best)
    _write_json(args.out, result)
    print(json.dumps({k: result[k] for k in ("best_selective_name", "h1_proxy_pass")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
