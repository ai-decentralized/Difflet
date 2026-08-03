#!/usr/bin/env python3
"""Test cheap cache signals against frozen automatic semantic-damage labels."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_quality_contract import (  # noqa: E402
    load_contract,
    load_json,
    sha256_file,
    write_json,
)
from scripts.analyze_flux_cache_semantic_boundary import (  # noqa: E402
    BOUNDARY_ANALYSIS_SCHEMA,
    BOUNDARY_ANALYSIS_SCHEMA_REVISION,
    SIGNALS,
)
from scripts.flux_cache_protocol import canonical_sha256  # noqa: E402


SIGNAL_GATE_SCHEMA = "difflet-flux-cache-automatic-signal-gate"
SIGNAL_GATE_SCHEMA_REVISION = 1


def _validate_boundary(document: Mapping[str, Any]) -> None:
    if (
        document.get("schema") != BOUNDARY_ANALYSIS_SCHEMA
        or document.get("schema_revision") != BOUNDARY_ANALYSIS_SCHEMA_REVISION
    ):
        raise ValueError("boundary analysis schema is unsupported")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != canonical_sha256(payload):
        raise ValueError("boundary analysis sha256 does not match its contents")
    if not isinstance(document.get("rows"), list) or not document["rows"]:
        raise ValueError("boundary analysis contains no rows")


def auc(labels: Sequence[int], values: Sequence[float]) -> float | None:
    positives = [value for label, value in zip(labels, values, strict=True) if label]
    negatives = [value for label, value in zip(labels, values, strict=True) if not label]
    if not positives or not negatives:
        return None
    score = sum(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )
    return score / (len(positives) * len(negatives))


def analyze(
    contract_path: Path,
    boundary_paths: Sequence[Path],
) -> dict[str, Any]:
    if not boundary_paths:
        raise ValueError("at least one boundary analysis is required")
    contract = load_contract(contract_path)
    margins = contract["margins"]
    sources = []
    rows = []
    identities: set[tuple[str, str]] = set()
    for path in boundary_paths:
        document = load_json(path, "boundary analysis")
        _validate_boundary(document)
        sources.append({"path": str(path), "sha256": sha256_file(path)})
        for source_row in document["rows"]:
            identity = (str(source_row["candidate_id"]), str(source_row["sample_id"]))
            if identity in identities:
                raise ValueError("boundary analyses contain duplicate candidate/sample rows")
            identities.add(identity)
            image_reward_harm = float(source_row["image_reward_harm"])
            vqa_harm = float(source_row["vqa_harm"])
            if not math.isfinite(image_reward_harm) or not math.isfinite(vqa_harm):
                raise ValueError("boundary semantic harm must be finite")
            failed_metrics = []
            if image_reward_harm > float(margins["image_reward"]):
                failed_metrics.append("image_reward")
            if vqa_harm > float(margins["vqa_score"]):
                failed_metrics.append("vqa_score")
            row = {
                "candidate_id": identity[0],
                "sample_id": identity[1],
                "automatic_damage": bool(failed_metrics),
                "failed_metrics": failed_metrics,
                "image_reward_harm": image_reward_harm,
                "vqa_harm": vqa_harm,
            }
            for signal in SIGNALS:
                value = float(source_row[signal])
                if not math.isfinite(value):
                    raise ValueError("boundary signal must be finite")
                row[signal] = value
            rows.append(row)

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_id"])].append(row)
    candidate_summaries = []
    for candidate_id in sorted(grouped):
        candidate_rows = grouped[candidate_id]
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "sample_count": len(candidate_rows),
                "damage_count": sum(bool(row["automatic_damage"]) for row in candidate_rows),
            }
        )

    signal_summaries = []
    for signal in SIGNALS:
        per_candidate = []
        informative = []
        for candidate_id in sorted(grouped):
            candidate_rows = grouped[candidate_id]
            candidate_auc = auc(
                [int(row["automatic_damage"]) for row in candidate_rows],
                [float(row[signal]) for row in candidate_rows],
            )
            per_candidate.append(
                {
                    "candidate_id": candidate_id,
                    "sample_count": len(candidate_rows),
                    "damage_count": sum(
                        bool(row["automatic_damage"]) for row in candidate_rows
                    ),
                    "auc": candidate_auc,
                }
            )
            if candidate_auc is not None:
                informative.append(candidate_auc)
        signal_summaries.append(
            {
                "signal": signal,
                "pooled_auc": auc(
                    [int(row["automatic_damage"]) for row in rows],
                    [float(row[signal]) for row in rows],
                ),
                "macro_within_candidate_auc": (
                    statistics.fmean(informative) if informative else None
                ),
                "informative_candidate_count": len(informative),
                "per_candidate": per_candidate,
            }
        )

    gate = contract["signal_gate"]
    primary = next(
        summary for summary in signal_summaries if summary["signal"] == gate["primary_signal"]
    )
    primary_auc = primary["macro_within_candidate_auc"]
    passes = (
        primary_auc is not None
        and primary_auc >= float(gate["minimum_macro_within_candidate_auc"])
        and primary["informative_candidate_count"] >= int(gate["minimum_informative_candidates"])
    )
    payload = {
        "schema": SIGNAL_GATE_SCHEMA,
        "schema_revision": SIGNAL_GATE_SCHEMA_REVISION,
        "evidence_role": gate["evidence_role"],
        "contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
            "content_sha256": contract["sha256"],
        },
        "boundary_sources": sources,
        "margins": margins,
        "sample_count": len(rows),
        "damage_count": sum(bool(row["automatic_damage"]) for row in rows),
        "candidate_summaries": candidate_summaries,
        "signal_summaries": signal_summaries,
        "gate": {
            **gate,
            "observed_macro_within_candidate_auc": primary_auc,
            "observed_informative_candidate_count": primary["informative_candidate_count"],
            "passes": passes,
            "decision": "adaptive-controller-eligible" if passes else gate["failure_action"],
        },
        "rows": rows,
        "interpretation": "Pooled AUC is reported only as a diagnostic because cache strength confounds it; the gate uses macro within-candidate AUC.",
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--boundary-analysis", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"output already exists: {out}")
    try:
        document = analyze(
            Path(args.contract).expanduser().resolve(),
            [Path(path).expanduser().resolve() for path in args.boundary_analysis],
        )
        write_json(out, document)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    gate = document["gate"]
    print(
        "[automatic-signal-gate] "
        f"auc={gate['observed_macro_within_candidate_auc']} "
        f"informative={gate['observed_informative_candidate_count']} "
        f"decision={gate['decision']} -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
