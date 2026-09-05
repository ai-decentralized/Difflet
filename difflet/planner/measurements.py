"""Measured step latencies, used to anchor and override the cost model.

Every parallel configuration is a different compile-cache key, and an AOT
compile is expensive -- 1484s for Flux at 1024x1024, 1316s for Qwen-Image -- so
the planner cannot measure candidates on demand the way DeepSpeed's autotuner
runs 13 real training jobs. It predicts, and prefers a measurement whenever one
exists.

The store is seeded from ``benchmark/<device>/*.json`` by
``scripts/seed_planner_measurements.py`` into a packaged JSON file, so it travels
with an installed wheel rather than depending on the repo layout. Point
``DIFFLET_PLANNER_MEASUREMENTS`` at another file to use your own numbers; entries
there win over the packaged ones.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_FILENAME = "measurements.json"
SCHEMA_VERSION = 1
ENV_OVERRIDE = "DIFFLET_PLANNER_MEASUREMENTS"


@dataclass(frozen=True)
class Measurement:
    """One observed run of one configuration.

    ``label`` is the planner's ``config_label`` for the parallel config, which is
    also verify_cli's matrix key -- so a measurement, a feasibility candidate and
    an on-device test cell all name the same thing the same way.
    """

    instance_type: str
    model: str
    # The exact HF path. Wan 2.1 and 2.2 share one registry entry and one set of
    # dimensions but are different weights with different latencies, so the
    # measurement store keeps them apart even though ``model`` cannot.
    model_id: str
    label: str
    height: int | None
    width: int | None
    num_frames: int | None
    steps: int | None
    step_latency_seconds: float | None
    e2e_warm_seconds: float | None
    compile_seconds: float | None
    source: str

    def shape_key(self) -> tuple:
        return (self.height, self.width, self.num_frames)


@dataclass(frozen=True)
class MeasurementStore:
    entries: tuple[Measurement, ...]

    def for_model(self, model: str, *, instance_type: str | None = None) -> tuple[Measurement, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.model == model
            and (instance_type is None or entry.instance_type == instance_type)
        )

    def lookup(
        self,
        *,
        model: str,
        label: str,
        instance_type: str,
        shape: tuple,
        steps: int | None = None,
        model_id: str | None = None,
    ) -> Measurement | None:
        """An exact match, or ``None``.

        Deliberately strict on shape: a step latency measured at 1024x1024 says
        very little about 480x832x121, and silently reusing it would report a
        guess as ``measured``.
        """

        for entry in self.entries:
            if (
                entry.model == model
                and entry.label == label
                and entry.instance_type == instance_type
                and entry.shape_key() == shape
                and (steps is None or entry.steps is None or entry.steps == steps)
                and (model_id is None or not entry.model_id or entry.model_id == model_id)
            ):
                return entry
        return None

    def anchors(
        self, *, model: str, instance_type: str, shape: tuple, model_id: str | None = None
    ) -> tuple[Measurement, ...]:
        """Measurements usable to calibrate this (model, host, shape).

        Only entries with a step latency: end-to-end numbers fold in weight
        loading and VAE decode, which no per-step model should be fitted to.
        """

        return tuple(
            entry
            for entry in self.entries
            if entry.model == model
            and entry.instance_type == instance_type
            and entry.shape_key() == shape
            and entry.step_latency_seconds
            and (model_id is None or not entry.model_id or entry.model_id == model_id)
        )


def _packaged_path() -> Path:
    return Path(__file__).resolve().parent / "data" / DATA_FILENAME


@lru_cache(maxsize=1)
def load_store() -> MeasurementStore:
    entries: list[Measurement] = []
    for path, is_override in ((_packaged_path(), False), (_override_path(), True)):
        if path is None or not path.exists():
            continue
        entries.extend(_read(path, is_override=is_override))
    # Later entries win, so an override file shadows a packaged measurement of
    # the same cell rather than competing with it.
    deduped: dict[tuple, Measurement] = {}
    for entry in entries:
        key = (entry.instance_type, entry.model_id or entry.model, entry.label, entry.shape_key())
        deduped[key] = entry
    return MeasurementStore(tuple(deduped.values()))


def _override_path() -> Path | None:
    raw = os.environ.get(ENV_OVERRIDE, "").strip()
    return Path(raw) if raw else None


def _read(path: Path, *, is_override: bool) -> list[Measurement]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    if payload.get("schema_version") != SCHEMA_VERSION:
        return []
    rows = payload.get("measurements")
    if not isinstance(rows, list):
        return []

    out: list[Measurement] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            out.append(
                Measurement(
                    instance_type=str(row["instance_type"]),
                    model=str(row["model"]),
                    model_id=str(row.get("model_id") or ""),
                    label=str(row["label"]),
                    height=_optional_int(row.get("height")),
                    width=_optional_int(row.get("width")),
                    num_frames=_optional_int(row.get("num_frames")),
                    steps=_optional_int(row.get("steps")),
                    step_latency_seconds=_optional_float(row.get("step_latency_seconds")),
                    e2e_warm_seconds=_optional_float(row.get("e2e_warm_seconds")),
                    compile_seconds=_optional_float(row.get("compile_seconds")),
                    source=str(row.get("source") or ("override" if is_override else "packaged")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
