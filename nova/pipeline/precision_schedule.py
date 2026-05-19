"""Precision schedule artifacts for MX calibration experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PRECISION_BF16 = "bf16"
PRECISION_MXFP8_E4M3 = "mxfp8_e4m3"
PRECISION_MXFP8_E5M2 = "mxfp8_e5m2"
# Non-BF16 levels, in descending preferred order (E4M3 first: 3-bit
# mantissa, lowest loss on well-conditioned cells; E5M2 second: wider
# exponent, the M5.4.1 remedy for high-dynamic-range cells).
_MX_PRECISIONS = (PRECISION_MXFP8_E4M3, PRECISION_MXFP8_E5M2)
# Schema stays 1: E5M2 is a new vocabulary value in `assignments`, not a
# structural change. Old two-level (bf16/e4m3) schedules deserialize
# unchanged; the dtype-general data structure absorbs E5M2 without a
# schema break (Decision D3).
SCHEMA_VERSION = 1


def _cell_key(block: int, linear: str) -> str:
    return f"{int(block)}:{linear}"


def _normalize_precision(value: str) -> str:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return PRECISION_BF16
    if normalized in {"mx", "mxfp8", "mxfp8_e4m3", "e4m3", "float8_e4m3fn_x4"}:
        return PRECISION_MXFP8_E4M3
    if normalized in {"mxfp8_e5m2", "e5m2", "float8_e5m2_x4"}:
        return PRECISION_MXFP8_E5M2
    raise ValueError(f"unsupported precision schedule dtype: {value!r}")


@dataclass(frozen=True)
class PrecisionSchedule:
    """Per-(block, linear) precision assignment."""

    model_id: str
    bundle: str
    tau: float | None
    assignments: dict[str, str]
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        normalized = {
            str(key): _normalize_precision(value)
            for key, value in self.assignments.items()
        }
        object.__setattr__(self, "assignments", normalized)

    @property
    def mx_coverage(self) -> float:
        """Fraction of cells on any non-BF16 (microscaled) precision.

        This is the H1′ coverage axis: E4M3 ∪ E5M2 (Decision D3). For a
        two-level (bf16/e4m3) schedule it is identical to the M5.4.0
        definition, so existing callers are unaffected.
        """

        if not self.assignments:
            return 0.0
        mx_count = sum(
            value in _MX_PRECISIONS for value in self.assignments.values()
        )
        return mx_count / len(self.assignments)

    @property
    def coverage_by_precision(self) -> dict[str, float]:
        """Per-precision coverage fraction (bf16 / e4m3 / e5m2)."""

        total = len(self.assignments)
        if not total:
            return {}
        counts: dict[str, int] = {}
        for value in self.assignments.values():
            counts[value] = counts.get(value, 0) + 1
        return {key: count / total for key, count in sorted(counts.items())}

    def precision_for(self, block: int, linear: str) -> str:
        return self.assignments[_cell_key(block, linear)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "bundle": self.bundle,
            "tau": self.tau,
            "mx_coverage": self.mx_coverage,
            "coverage_by_precision": self.coverage_by_precision,
            "assignments": dict(sorted(self.assignments.items())),
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PrecisionSchedule":
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported PrecisionSchedule schema_version: {version!r}")
        return cls(
            model_id=str(data["model_id"]),
            bundle=str(data["bundle"]),
            tau=data.get("tau"),
            assignments=dict(data["assignments"]),
            metadata=dict(data.get("metadata") or {}),
        )

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read_json(cls, path: str | Path) -> "PrecisionSchedule":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _rows(calibration_table: dict[str, Any] | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(calibration_table, dict):
        rows = calibration_table.get("rows")
        if rows is None:
            raise ValueError("calibration table dict must contain a 'rows' field")
        return list(rows)
    return list(calibration_table)


def synthesize_schedule(
    calibration_table: dict[str, Any] | Iterable[dict[str, Any]],
    tau: float,
    *,
    model_id: str,
    bundle: str,
    metadata: dict[str, Any] | None = None,
) -> PrecisionSchedule:
    """Assign MXFP8 to cells whose calibration cosine is at least ``tau``."""

    assignments = {}
    for row in _rows(calibration_table):
        key = _cell_key(int(row["block"]), str(row["linear"]))
        cosine = float(row["cosine"])
        assignments[key] = PRECISION_MXFP8_E4M3 if cosine >= tau else PRECISION_BF16
    return PrecisionSchedule(
        model_id=model_id,
        bundle=bundle,
        tau=float(tau),
        assignments=assignments,
        metadata=metadata,
    )


def extreme_schedule(
    calibration_table: dict[str, Any] | Iterable[dict[str, Any]],
    precision: str,
    *,
    model_id: str,
    bundle: str,
    metadata: dict[str, Any] | None = None,
) -> PrecisionSchedule:
    """Build an all-BF16 or all-MX schedule over the calibration cells."""

    normalized = _normalize_precision(precision)
    assignments = {
        _cell_key(int(row["block"]), str(row["linear"])): normalized
        for row in _rows(calibration_table)
    }
    return PrecisionSchedule(
        model_id=model_id,
        bundle=bundle,
        tau=None,
        assignments=assignments,
        metadata=metadata,
    )


def schedule_frontier(
    calibration_table: dict[str, Any] | Iterable[dict[str, Any]],
    taus: Iterable[float],
    *,
    model_id: str,
    bundle: str,
) -> list[PrecisionSchedule]:
    table_rows = _rows(calibration_table)
    schedules = [
        extreme_schedule(table_rows, PRECISION_BF16, model_id=model_id, bundle=bundle),
        extreme_schedule(table_rows, PRECISION_MXFP8_E4M3, model_id=model_id, bundle=bundle),
    ]
    schedules.extend(
        synthesize_schedule(table_rows, tau, model_id=model_id, bundle=bundle) for tau in taus
    )
    return schedules


def _cell_cosines(row: dict[str, Any]) -> tuple[float, float | None]:
    """Return ``(e4m3_cos, e5m2_cos)`` for a calibration row.

    Schema-2 rows carry ``metrics.{e4m3,e5m2}.cosine``. Schema-1 rows
    only have a top-level ``cosine`` (E4M3 implied, Decision D2); E5M2 is
    then unknown and reported as ``None`` so the two-threshold
    synthesizer degrades to single-threshold behavior on old tables.
    """

    metrics = row.get("metrics")
    if isinstance(metrics, dict) and "e4m3" in metrics:
        e4m3 = float(metrics["e4m3"]["cosine"])
        e5m2 = metrics.get("e5m2")
        return e4m3, (float(e5m2["cosine"]) if e5m2 is not None else None)
    return float(row["cosine"]), None


def synthesize_two_threshold_schedule(
    calibration_table: dict[str, Any] | Iterable[dict[str, Any]],
    tau_hi: float,
    tau_lo: float,
    *,
    model_id: str,
    bundle: str,
    metadata: dict[str, Any] | None = None,
) -> PrecisionSchedule:
    """Two-threshold lattice (Decision D3).

    For each cell: E4M3 iff ``e4m3_cos >= tau_hi``; else E5M2 iff
    ``e5m2_cos >= tau_lo``; else BF16. Pure function of the calibration
    table — no per-linear-name rules (the §anti-assumption guard).
    """

    assignments = {}
    for row in _rows(calibration_table):
        key = _cell_key(int(row["block"]), str(row["linear"]))
        e4m3_cos, e5m2_cos = _cell_cosines(row)
        if e4m3_cos >= tau_hi:
            assignments[key] = PRECISION_MXFP8_E4M3
        elif e5m2_cos is not None and e5m2_cos >= tau_lo:
            assignments[key] = PRECISION_MXFP8_E5M2
        else:
            assignments[key] = PRECISION_BF16
    meta = dict(metadata or {})
    meta.update({"tau_hi": float(tau_hi), "tau_lo": float(tau_lo)})
    return PrecisionSchedule(
        model_id=model_id,
        bundle=bundle,
        tau=float(tau_hi),
        assignments=assignments,
        metadata=meta,
    )


def two_threshold_frontier(
    calibration_table: dict[str, Any] | Iterable[dict[str, Any]],
    tau_pairs: Iterable[tuple[float, float]],
    *,
    model_id: str,
    bundle: str,
) -> list[PrecisionSchedule]:
    """Three extremes (all-BF16 / all-E4M3 / all-E5M2) + a ``(τ_hi,τ_lo)`` grid."""

    table_rows = _rows(calibration_table)
    schedules = [
        extreme_schedule(table_rows, PRECISION_BF16, model_id=model_id, bundle=bundle),
        extreme_schedule(
            table_rows, PRECISION_MXFP8_E4M3, model_id=model_id, bundle=bundle
        ),
        extreme_schedule(
            table_rows, PRECISION_MXFP8_E5M2, model_id=model_id, bundle=bundle
        ),
    ]
    schedules.extend(
        synthesize_two_threshold_schedule(
            table_rows, tau_hi, tau_lo, model_id=model_id, bundle=bundle
        )
        for tau_hi, tau_lo in tau_pairs
    )
    return schedules


__all__ = [
    "PRECISION_BF16",
    "PRECISION_MXFP8_E4M3",
    "PRECISION_MXFP8_E5M2",
    "PrecisionSchedule",
    "extreme_schedule",
    "schedule_frontier",
    "synthesize_schedule",
    "synthesize_two_threshold_schedule",
    "two_threshold_frontier",
]
