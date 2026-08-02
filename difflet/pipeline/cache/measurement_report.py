"""Strict serialization for request-scoped cache measurements."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from difflet.pipeline.cache.measurements import (
    AnchorMeasurement,
    LatentUpdateMeasurement,
)

CACHE_MEASUREMENT_SCHEMA = "difflet-cache-measurements"
CACHE_MEASUREMENT_SCHEMA_REVISION = 1


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _check_keys(value: Mapping[str, Any], name: str, required: set[str]) -> None:
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")


def _parse_anchor_measurement(value: Any) -> AnchorMeasurement:
    record = _require_mapping(value, "anchor measurement")
    required = {
        "step_index",
        "num_steps",
        "timestep",
        "sigma",
        "decision_reason",
        "history_size",
        "estimated_steps_since_anchor",
        "anchor_step_gap",
        "output_shape",
        "output_dtype",
        "output_norm",
        "relative_output_change",
        "relative_output_curvature",
        "estimate_status",
        "estimate_relative_error",
        "estimate_seconds",
        "measurement_seconds",
        "numerically_valid",
    }
    _check_keys(record, "anchor measurement", required)
    output_shape = record["output_shape"]
    if not isinstance(output_shape, list):
        raise ValueError("anchor measurement.output_shape must be a list")
    return AnchorMeasurement(
        step_index=record["step_index"],
        num_steps=record["num_steps"],
        timestep=record["timestep"],
        sigma=record["sigma"],
        decision_reason=record["decision_reason"],
        history_size=record["history_size"],
        estimated_steps_since_anchor=record["estimated_steps_since_anchor"],
        anchor_step_gap=record["anchor_step_gap"],
        output_shape=tuple(output_shape),
        output_dtype=record["output_dtype"],
        output_norm=record["output_norm"],
        relative_output_change=record["relative_output_change"],
        relative_output_curvature=record["relative_output_curvature"],
        estimate_status=record["estimate_status"],
        estimate_relative_error=record["estimate_relative_error"],
        estimate_seconds=record["estimate_seconds"],
        measurement_seconds=record["measurement_seconds"],
        numerically_valid=record["numerically_valid"],
    )


def _parse_latent_update(value: Any) -> LatentUpdateMeasurement:
    record = _require_mapping(value, "latent update")
    required = {
        "step_index",
        "num_steps",
        "timestep",
        "sigma",
        "decision_reason",
        "used_estimate",
        "latent_shape",
        "latent_dtype",
        "before_norm",
        "after_norm",
        "update_norm",
        "relative_update",
        "measurement_seconds",
        "numerically_valid",
    }
    _check_keys(record, "latent update", required)
    latent_shape = record["latent_shape"]
    if not isinstance(latent_shape, list):
        raise ValueError("latent update.latent_shape must be a list")
    return LatentUpdateMeasurement(
        step_index=record["step_index"],
        num_steps=record["num_steps"],
        timestep=record["timestep"],
        sigma=record["sigma"],
        decision_reason=record["decision_reason"],
        used_estimate=record["used_estimate"],
        latent_shape=tuple(latent_shape),
        latent_dtype=record["latent_dtype"],
        before_norm=record["before_norm"],
        after_norm=record["after_norm"],
        update_norm=record["update_norm"],
        relative_update=record["relative_update"],
        measurement_seconds=record["measurement_seconds"],
        numerically_valid=record["numerically_valid"],
    )


@dataclass(frozen=True)
class CacheMeasurementReport:
    """Strict, serializable runtime measurements for one cache request."""

    num_steps: int
    configuration_source: str
    anchor_measurements: tuple[AnchorMeasurement, ...]
    latent_updates: tuple[LatentUpdateMeasurement, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_steps, bool)
            or not isinstance(self.num_steps, int)
            or self.num_steps <= 0
        ):
            raise ValueError("num_steps must be a positive integer")
        if (
            not isinstance(self.configuration_source, str)
            or not self.configuration_source
        ):
            raise ValueError("configuration_source must be a non-empty string")
        measurements = tuple(self.anchor_measurements)
        previous_step = -1
        for measurement in measurements:
            if not isinstance(measurement, AnchorMeasurement):
                raise TypeError("anchor_measurements must contain AnchorMeasurement values")
            if measurement.num_steps != self.num_steps:
                raise ValueError("anchor measurement num_steps does not match report")
            if measurement.step_index <= previous_step:
                raise ValueError("anchor measurements must have increasing step indices")
            previous_step = measurement.step_index
        object.__setattr__(self, "anchor_measurements", measurements)
        latent_updates = tuple(self.latent_updates)
        previous_step = -1
        for measurement in latent_updates:
            if not isinstance(measurement, LatentUpdateMeasurement):
                raise TypeError("latent_updates must contain LatentUpdateMeasurement values")
            if measurement.num_steps != self.num_steps:
                raise ValueError("latent update num_steps does not match report")
            if measurement.step_index <= previous_step:
                raise ValueError("latent updates must have increasing step indices")
            previous_step = measurement.step_index
        object.__setattr__(self, "latent_updates", latent_updates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CACHE_MEASUREMENT_SCHEMA,
            "schema_revision": CACHE_MEASUREMENT_SCHEMA_REVISION,
            "num_steps": self.num_steps,
            "configuration_source": self.configuration_source,
            "anchor_measurements": [
                measurement.to_dict() for measurement in self.anchor_measurements
            ],
            "latent_updates": [
                measurement.to_dict() for measurement in self.latent_updates
            ],
        }

    def write_json(self, path: str | Path) -> None:
        """Atomically write finite JSON suitable for experiment artifacts."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)


def load_cache_measurements(
    source: str | Path | Mapping[str, Any],
) -> CacheMeasurementReport:
    """Load strict measurements for one request without guessing fields."""

    if isinstance(source, Mapping):
        document = source
    else:
        path = Path(source)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"cache measurements file does not exist: {path}") from error
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cache measurements is not valid JSON: {path}: {error}") from error
    document = _require_mapping(document, "cache measurements")
    _check_keys(
        document,
        "cache measurements",
        {
            "schema",
            "schema_revision",
            "num_steps",
            "configuration_source",
            "anchor_measurements",
            "latent_updates",
        },
    )
    if document["schema"] != CACHE_MEASUREMENT_SCHEMA:
        raise ValueError(
            f"cache measurements schema must be {CACHE_MEASUREMENT_SCHEMA!r}"
        )
    revision = document["schema_revision"]
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision != CACHE_MEASUREMENT_SCHEMA_REVISION
    ):
        raise ValueError(
            "cache measurements schema_revision must be "
            f"{CACHE_MEASUREMENT_SCHEMA_REVISION}"
        )
    records = document["anchor_measurements"]
    if not isinstance(records, list):
        raise ValueError("cache measurements.anchor_measurements must be a list")
    latent_updates = document["latent_updates"]
    if not isinstance(latent_updates, list):
        raise ValueError("cache measurements.latent_updates must be a list")
    return CacheMeasurementReport(
        num_steps=document["num_steps"],
        configuration_source=document["configuration_source"],
        anchor_measurements=tuple(_parse_anchor_measurement(record) for record in records),
        latent_updates=tuple(_parse_latent_update(record) for record in latent_updates),
    )


__all__ = [
    "CACHE_MEASUREMENT_SCHEMA",
    "CACHE_MEASUREMENT_SCHEMA_REVISION",
    "CacheMeasurementReport",
    "load_cache_measurements",
]
