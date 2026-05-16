"""Precision schedule artifacts for MX calibration experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PRECISION_BF16 = "bf16"
PRECISION_MXFP8_E4M3 = "mxfp8_e4m3"
SCHEMA_VERSION = 1


def _cell_key(block: int, linear: str) -> str:
    return f"{int(block)}:{linear}"


def _normalize_precision(value: str) -> str:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return PRECISION_BF16
    if normalized in {"mx", "mxfp8", "mxfp8_e4m3", "float8_e4m3fn_x4"}:
        return PRECISION_MXFP8_E4M3
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
        if not self.assignments:
            return 0.0
        mx_count = sum(value == PRECISION_MXFP8_E4M3 for value in self.assignments.values())
        return mx_count / len(self.assignments)

    def precision_for(self, block: int, linear: str) -> str:
        return self.assignments[_cell_key(block, linear)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.model_id,
            "bundle": self.bundle,
            "tau": self.tau,
            "mx_coverage": self.mx_coverage,
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


__all__ = [
    "PRECISION_BF16",
    "PRECISION_MXFP8_E4M3",
    "PrecisionSchedule",
    "extreme_schedule",
    "schedule_frontier",
    "synthesize_schedule",
]
