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

from nova.pipeline.precision_schedule import (  # noqa: E402
    PRECISION_BF16,
    PRECISION_MXFP8_E4M3,
    PrecisionSchedule,
    extreme_schedule,
    schedule_frontier,
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _schedule_name(schedule: PrecisionSchedule) -> str:
    if schedule.tau is None:
        if schedule.mx_coverage == 0.0:
            return "all-bf16"
        if schedule.mx_coverage == 1.0:
            return "all-mx"
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
        values.append(1.0 if precision == PRECISION_BF16 else float(row["cosine"]))
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
    parser.add_argument("--persist-best", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    table = json.loads(args.calibration_table.read_text(encoding="utf-8"))
    rows = list(table["rows"])
    bundle = args.bundle or str(table.get("bundle", "unknown"))
    schedules = schedule_frontier(
        rows,
        args.taus,
        model_id=args.model_id,
        bundle=bundle,
    )
    entries = []
    all_bf16 = extreme_schedule(rows, PRECISION_BF16, model_id=args.model_id, bundle=bundle)
    all_mx = extreme_schedule(rows, PRECISION_MXFP8_E4M3, model_id=args.model_id, bundle=bundle)
    for schedule in schedules:
        quality = _proxy_quality(schedule, rows)
        entries.append(
            {
                "name": _schedule_name(schedule),
                "tau": schedule.tau,
                "mx_coverage": schedule.mx_coverage,
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
        "taus": args.taus,
        "all_bf16_proxy": _proxy_quality(all_bf16, rows),
        "all_mx_proxy": _proxy_quality(all_mx, rows),
        "best_selective_name": best["name"],
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
