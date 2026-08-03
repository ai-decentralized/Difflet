"""Optional spatial summaries of anchor-estimation error.

The runtime already holds both the side-effect-free estimate and the real
transformer output at a computed anchor.  This module reduces their difference
to small region-energy vectors without retaining either full tensor.  The
records are experimental observations only: they never participate in cache
decisions.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from difflet.pipeline.cache.measurements import InMemoryMeasurementSink
from difflet.pipeline.cache.types import CacheStepContext

SPATIAL_MEASUREMENT_SCHEMA = "difflet-cache-spatial-measurements"
SPATIAL_MEASUREMENT_SCHEMA_REVISION = 1


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return result


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


@dataclass(frozen=True)
class SpatialMeasurementLayout:
    """Describe a flattened token grid and its fixed measurement regions.

    ``token_height`` and ``token_width`` describe the spatial order represented
    by one tensor axis. ``region_rows`` and ``region_columns`` divide that grid
    into equal, non-overlapping rectangles. ``token_axis`` identifies the
    flattened spatial axis in the transformer output.
    """

    token_height: int
    token_width: int
    region_rows: int
    region_columns: int
    token_axis: int = -2

    def __post_init__(self) -> None:
        token_height = _positive_int(self.token_height, "token_height")
        token_width = _positive_int(self.token_width, "token_width")
        region_rows = _positive_int(self.region_rows, "region_rows")
        region_columns = _positive_int(self.region_columns, "region_columns")
        if token_height % region_rows != 0:
            raise ValueError("region_rows must divide token_height")
        if token_width % region_columns != 0:
            raise ValueError("region_columns must divide token_width")
        if isinstance(self.token_axis, bool) or not isinstance(self.token_axis, int):
            raise ValueError("token_axis must be an integer")

    @property
    def token_count(self) -> int:
        return self.token_height * self.token_width

    @property
    def region_count(self) -> int:
        return self.region_rows * self.region_columns

    def to_dict(self) -> dict[str, int]:
        return {
            "token_height": self.token_height,
            "token_width": self.token_width,
            "region_rows": self.region_rows,
            "region_columns": self.region_columns,
            "token_axis": self.token_axis,
        }


@dataclass(frozen=True)
class SpatialErrorMeasurement:
    """Squared estimate-error and reference energy for every spatial region."""

    step_index: int
    num_steps: int
    error_energy: tuple[float, ...]
    reference_energy: tuple[float, ...]
    measurement_seconds: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ValueError("step_index must be a nonnegative integer")
        num_steps = _positive_int(self.num_steps, "num_steps")
        if self.step_index >= num_steps:
            raise ValueError("step_index must identify a step in num_steps")
        error_energy = tuple(
            _nonnegative_float(value, "error_energy") for value in self.error_energy
        )
        reference_energy = tuple(
            _nonnegative_float(value, "reference_energy") for value in self.reference_energy
        )
        if not error_energy or len(error_energy) != len(reference_energy):
            raise ValueError(
                "error_energy and reference_energy must have one matching non-empty value per region"
            )
        object.__setattr__(self, "error_energy", error_energy)
        object.__setattr__(self, "reference_energy", reference_energy)
        _nonnegative_float(self.measurement_seconds, "measurement_seconds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "num_steps": self.num_steps,
            "error_energy": list(self.error_energy),
            "reference_energy": list(self.reference_energy),
            "measurement_seconds": self.measurement_seconds,
        }


def measure_spatial_error(
    *,
    context: CacheStepContext,
    estimate: Any,
    actual: Any,
    layout: SpatialMeasurementLayout,
) -> SpatialErrorMeasurement:
    """Reduce one exact anchor comparison to fixed spatial-region energies."""

    import torch

    started = time.perf_counter()
    if not isinstance(layout, SpatialMeasurementLayout):
        raise TypeError("layout must be a SpatialMeasurementLayout")
    if not torch.is_tensor(estimate) or not torch.is_tensor(actual):
        raise TypeError("spatial measurement requires tensor outputs")
    if tuple(estimate.shape) != tuple(actual.shape):
        raise ValueError("estimated and actual outputs must have identical shapes")
    if estimate.ndim < 2:
        raise ValueError("spatial measurement requires at least two tensor axes")
    token_axis = layout.token_axis
    if token_axis < 0:
        token_axis += estimate.ndim
    if not 0 <= token_axis < estimate.ndim:
        raise ValueError("token_axis is outside the output tensor rank")
    if int(estimate.shape[token_axis]) != layout.token_count:
        raise ValueError("spatial token axis does not match token_height * token_width")

    estimate_float = estimate.detach().float()
    actual_float = actual.detach().float()
    difference = estimate_float - actual_float
    difference_grid = difference.movedim(token_axis, -1).reshape(
        -1,
        layout.token_height,
        layout.token_width,
    )
    actual_grid = actual_float.movedim(token_axis, -1).reshape(
        -1,
        layout.token_height,
        layout.token_width,
    )
    region_height = layout.token_height // layout.region_rows
    region_width = layout.token_width // layout.region_columns

    def region_energy(value: Any) -> Any:
        cells = value.reshape(
            -1,
            layout.region_rows,
            region_height,
            layout.region_columns,
            region_width,
        ).permute(1, 3, 0, 2, 4)
        return cells.square().sum(dim=(2, 3, 4))

    error = region_energy(difference_grid)
    reference = region_energy(actual_grid)
    # One small device-to-host transfer captures every region. Repeated scalar
    # ``item()`` calls would introduce one synchronization per region.
    packed = torch.stack((error, reference), dim=0).detach().cpu().reshape(2, -1)
    values = packed.tolist()
    return SpatialErrorMeasurement(
        step_index=context.step_index,
        num_steps=context.num_steps,
        error_energy=tuple(float(value) for value in values[0]),
        reference_energy=tuple(float(value) for value in values[1]),
        measurement_seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class SpatialMeasurementReport:
    """Strict report containing only small spatial error summaries."""

    num_steps: int
    configuration_source: str
    layout: SpatialMeasurementLayout
    anchor_errors: tuple[SpatialErrorMeasurement, ...]

    def __post_init__(self) -> None:
        num_steps = _positive_int(self.num_steps, "num_steps")
        if not isinstance(self.configuration_source, str) or not self.configuration_source:
            raise ValueError("configuration_source must be a non-empty string")
        if not isinstance(self.layout, SpatialMeasurementLayout):
            raise TypeError("layout must be a SpatialMeasurementLayout")
        records = tuple(self.anchor_errors)
        previous_step = -1
        for record in records:
            if not isinstance(record, SpatialErrorMeasurement):
                raise TypeError("anchor_errors must contain SpatialErrorMeasurement values")
            if record.num_steps != num_steps:
                raise ValueError("spatial measurement num_steps does not match report")
            if len(record.error_energy) != self.layout.region_count:
                raise ValueError("spatial measurement region count does not match layout")
            if record.step_index <= previous_step:
                raise ValueError("spatial measurements must have increasing step indices")
            previous_step = record.step_index
        object.__setattr__(self, "anchor_errors", records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SPATIAL_MEASUREMENT_SCHEMA,
            "schema_revision": SPATIAL_MEASUREMENT_SCHEMA_REVISION,
            "num_steps": self.num_steps,
            "configuration_source": self.configuration_source,
            "layout": self.layout.to_dict(),
            "anchor_errors": [record.to_dict() for record in self.anchor_errors],
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)


class InMemorySpatialMeasurementSink(InMemoryMeasurementSink):
    """Collect ordinary request measurements plus optional spatial summaries."""

    def __init__(self, layout: SpatialMeasurementLayout) -> None:
        super().__init__()
        if not isinstance(layout, SpatialMeasurementLayout):
            raise TypeError("layout must be a SpatialMeasurementLayout")
        self.layout = layout
        self._spatial_errors: list[SpatialErrorMeasurement] = []

    def observe_anchor_tensors(
        self,
        context: CacheStepContext,
        estimate: Any,
        actual: Any,
    ) -> None:
        record = measure_spatial_error(
            context=context,
            estimate=estimate,
            actual=actual,
            layout=self.layout,
        )
        if self._spatial_errors and record.step_index <= self._spatial_errors[-1].step_index:
            raise ValueError("spatial measurements must have increasing step indices")
        self._spatial_errors.append(record)

    def clear(self) -> None:
        super().clear()
        self._spatial_errors.clear()

    def spatial_errors(self) -> tuple[SpatialErrorMeasurement, ...]:
        return tuple(self._spatial_errors)

    def build_spatial_report(
        self,
        *,
        num_steps: int,
        configuration_source: str,
    ) -> SpatialMeasurementReport:
        return SpatialMeasurementReport(
            num_steps=num_steps,
            configuration_source=configuration_source,
            layout=self.layout,
            anchor_errors=self.spatial_errors(),
        )


def load_spatial_measurements(
    source: str | Path | Mapping[str, Any],
) -> SpatialMeasurementReport:
    """Load one strict spatial report without accepting unknown fields."""

    if isinstance(source, Mapping):
        document = source
    else:
        path = Path(source)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"spatial measurements file does not exist: {path}") from error
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"spatial measurements is not valid JSON: {path}: {error}") from error
    document = _require_mapping(document, "spatial measurements")
    _check_keys(
        document,
        "spatial measurements",
        {
            "schema",
            "schema_revision",
            "num_steps",
            "configuration_source",
            "layout",
            "anchor_errors",
        },
    )
    if document["schema"] != SPATIAL_MEASUREMENT_SCHEMA:
        raise ValueError(f"spatial measurements schema must be {SPATIAL_MEASUREMENT_SCHEMA!r}")
    if (
        isinstance(document["schema_revision"], bool)
        or document["schema_revision"] != SPATIAL_MEASUREMENT_SCHEMA_REVISION
    ):
        raise ValueError(
            "spatial measurements schema_revision must be " f"{SPATIAL_MEASUREMENT_SCHEMA_REVISION}"
        )
    layout_value = _require_mapping(document["layout"], "spatial measurements.layout")
    _check_keys(
        layout_value,
        "spatial measurements.layout",
        {"token_height", "token_width", "region_rows", "region_columns", "token_axis"},
    )
    layout = SpatialMeasurementLayout(**dict(layout_value))
    records_value = document["anchor_errors"]
    if not isinstance(records_value, list):
        raise ValueError("spatial measurements.anchor_errors must be a list")
    records: list[SpatialErrorMeasurement] = []
    for value in records_value:
        record = _require_mapping(value, "spatial anchor error")
        _check_keys(
            record,
            "spatial anchor error",
            {
                "step_index",
                "num_steps",
                "error_energy",
                "reference_energy",
                "measurement_seconds",
            },
        )
        if not isinstance(record["error_energy"], list) or not isinstance(
            record["reference_energy"], list
        ):
            raise ValueError("spatial energy fields must be lists")
        records.append(
            SpatialErrorMeasurement(
                step_index=record["step_index"],
                num_steps=record["num_steps"],
                error_energy=tuple(record["error_energy"]),
                reference_energy=tuple(record["reference_energy"]),
                measurement_seconds=record["measurement_seconds"],
            )
        )
    return SpatialMeasurementReport(
        num_steps=document["num_steps"],
        configuration_source=document["configuration_source"],
        layout=layout,
        anchor_errors=tuple(records),
    )


__all__ = [
    "SPATIAL_MEASUREMENT_SCHEMA",
    "SPATIAL_MEASUREMENT_SCHEMA_REVISION",
    "InMemorySpatialMeasurementSink",
    "SpatialErrorMeasurement",
    "SpatialMeasurementLayout",
    "SpatialMeasurementReport",
    "load_spatial_measurements",
    "measure_spatial_error",
]
