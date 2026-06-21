#!/usr/bin/env python3
"""F2.0 gate audit. Reads per-component JSONs from cclog 65/66 and emits a
machine-readable decision on whether F2.1 is unlocked.

Decision rule (cclog 65 D3):
- any production video model VAE share >= 20% of end-to-end wall-clock
  → ``can_unlock_f2_1: true``
- all measured models below 20%, and at least one required label measured
  → ``can_write_negative_closeout: true``
- otherwise → ``remain_gated_partial_evidence``

Required labels (cclog 65 §"Local artifact audit"):
- ``hunyuan_video`` — N4 4d8s1r hybrid run
- ``wan`` — two-stage wall-clock run
- (optional) ``hunyuan_video15``, ``ltx_2`` — extend later

Accepted schema set:
- ``difflet-f2-0-component-wallclock-v1`` (from profile_component_wallclock.py)
- ``difflet-f2-0-wan-twostage-wallclock-v1`` (from profile_wan_twostage_wallclock.py)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "difflet-f2-0-gate-audit-v1"
THRESHOLD = 0.20

ACCEPTED_SCHEMAS = {
    "difflet-f2-0-component-wallclock-v1",
    "difflet-f2-0-wan-twostage-wallclock-v1",
}

LABEL_HINTS: dict[str, str] = {
    "hv_n4": "hunyuan_video",
    "hunyuan_video15": "hunyuan_video15",
    "hunyuan-video15": "hunyuan_video15",
    "hunyuan_video": "hunyuan_video",
    "hunyuan-video": "hunyuan_video",
    "wan": "wan",
    "ltx_2": "ltx_2",
    "ltx-2": "ltx_2",
}


def _infer_label(doc: dict[str, Any], path: Path) -> str | None:
    model = doc.get("model")
    if model and model in LABEL_HINTS:
        return LABEL_HINTS[model]
    schema = doc.get("schema")
    if schema == "difflet-f2-0-wan-twostage-wallclock-v1":
        return "wan"
    name = path.name.lower()
    for hint, label in LABEL_HINTS.items():
        if hint in name:
            return label
    return None


def _row_for(doc: dict[str, Any], path: Path) -> dict[str, Any]:
    label = _infer_label(doc, path)
    schema = doc.get("schema")
    invalid: list[str] = []
    vae_share: float | None = None
    if schema == "difflet-f2-0-component-wallclock-v1":
        shares = doc.get("shares") or {}
        vae_share = shares.get("vae_share")
        if vae_share is None:
            invalid.append("missing_shares.vae_share")
        vae_path = (doc.get("components") or {}).get("vae_path")
        if vae_path == "host_cpu_hf":
            # CPU VAE is an upper bound on share — useful but flag it
            invalid.append("vae_path=host_cpu_hf_upper_bound")
    elif schema == "difflet-f2-0-wan-twostage-wallclock-v1":
        shares = doc.get("shares") or {}
        vae_share = shares.get("vae_share_lower_bound")
        if vae_share is None:
            invalid.append("missing_shares.vae_share_lower_bound")
        if not doc.get("completed"):
            invalid.append("incomplete_run")
    else:
        invalid.append(f"unsupported_schema:{schema}")

    passes = bool(vae_share is not None and vae_share >= THRESHOLD)
    return {
        "label": label,
        "schema": schema,
        "path": str(path),
        "vae_share": vae_share,
        "passes_threshold": passes,
        "invalid_reasons": invalid,
        "hardware_measured": schema in ACCEPTED_SCHEMAS and not invalid,
    }


def _decide(rows: list[dict[str, Any]], required_labels: list[str]) -> dict[str, Any]:
    measured_labels = sorted({r["label"] for r in rows if r["label"] is not None})
    missing_labels = [lab for lab in required_labels if lab not in measured_labels]
    unlocking = [
        r for r in rows if r["passes_threshold"] and r["label"] is not None
    ]

    can_unlock = bool(unlocking)
    can_negative = (not can_unlock) and (not missing_labels) and bool(rows)
    if can_unlock:
        decision = "unlock_f2_1"
    elif can_negative:
        decision = "write_negative_closeout"
    elif rows:
        decision = "remain_gated_partial_evidence"
    else:
        decision = "no_evidence"
    return {
        "schema": SCHEMA,
        "threshold": THRESHOLD,
        "required_labels": required_labels,
        "measured_labels": measured_labels,
        "missing_labels": missing_labels,
        "unlocking_labels": sorted({r["label"] for r in unlocking}),
        "can_unlock_f2_1": can_unlock,
        "can_write_negative_closeout": can_negative,
        "decision": decision,
        "rows": rows,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--combined",
        action="append",
        default=[],
        help="Glob (or repeated) for input wallclock JSONs",
    )
    p.add_argument(
        "--required-label",
        action="append",
        default=["hunyuan_video", "wan"],
        help="Label expected to be measured before negative closeout is allowed",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    paths: list[Path] = []
    for pat in args.combined:
        paths.extend(Path(p) for p in glob.glob(pat))
    paths = sorted(set(paths))
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "label": None,
                    "schema": None,
                    "path": str(path),
                    "vae_share": None,
                    "passes_threshold": False,
                    "invalid_reasons": [f"load_error:{exc}"],
                    "hardware_measured": False,
                }
            )
            continue
        rows.append(_row_for(doc, path))

    decision = _decide(rows, sorted(set(args.required_label)))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
