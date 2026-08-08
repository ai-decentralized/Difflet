"""Read-only measurements for validating cache estimates at real anchors."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, runtime_checkable

from difflet.pipeline.cache.types import (
    CacheHistory,
    CachePredictor,
    CacheStepContext,
    RuntimeObservation,
)

if TYPE_CHECKING:
    from difflet.pipeline.cache.measurement_report import CacheMeasurementReport

AnchorEstimateStatus = Literal[
    "history_not_ready",
    "measured",
    "invalid_actual_output",
    "invalid_estimated_output",
    "prediction_error",
]

_DECISION_REASONS = {
    "policy_compute",
    "policy_skip",
    "barrier",
    "history_not_ready",
    "consecutive_skip_veto",
    "recovery_warmup",
    "recovery_cooldown",
    "recovery_final_anchor",
    "recovery_consecutive_limit",
    "recovery_requested",
    "direct_compute",
}

_RELATIVE_NORM_FLOOR = 1e-12


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, name)


def _optional_nonnegative_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative finite number or None")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite number or None")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _optional_nonnegative_float(value, name)
    if result is None:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return result


def _optional_finite_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite or None")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite or None")
    return result


def _decision_reason(value: Any, name: str) -> str:
    if value not in _DECISION_REASONS:
        raise ValueError(f"unsupported {name}: {value!r}")
    return str(value)


@dataclass(frozen=True)
class AnchorMeasurement:
    """One read-only comparison between an anchor estimate and its real output.

    All history and counters describe the state immediately before the real
    anchor is recorded.  The comparison estimate is never returned to the
    scheduler and never enters :class:`CacheHistory`.

    Norms are L2 norms computed in float32. ``relative_output_change`` is
    ``||current - previous|| / max(||previous||, 1e-12)``. Curvature compares
    the two most recent coordinate-scaled output slopes and divides their L2
    difference by the latest slope norm. ``estimate_relative_error`` is
    ``||estimate - actual|| / max(||actual||, 1e-12)``.
    """

    step_index: int
    num_steps: int
    timestep: float | None
    sigma: float | None
    decision_reason: str
    history_size: int
    estimated_steps_since_anchor: int
    anchor_step_gap: int | None
    output_shape: tuple[int, ...]
    output_dtype: str
    output_norm: float | None
    relative_output_change: float | None
    relative_output_curvature: float | None
    estimate_status: AnchorEstimateStatus
    estimate_relative_error: float | None
    estimate_seconds: float | None
    measurement_seconds: float
    numerically_valid: bool

    def __post_init__(self) -> None:
        step_index = _nonnegative_int(self.step_index, "step_index")
        num_steps = _nonnegative_int(self.num_steps, "num_steps")
        if num_steps <= 0 or step_index >= num_steps:
            raise ValueError("step_index must identify a step in num_steps")
        _decision_reason(self.decision_reason, "anchor decision reason")
        _optional_finite_float(self.timestep, "timestep")
        _optional_finite_float(self.sigma, "sigma")
        _nonnegative_int(self.history_size, "history_size")
        _nonnegative_int(
            self.estimated_steps_since_anchor,
            "estimated_steps_since_anchor",
        )
        _optional_nonnegative_int(self.anchor_step_gap, "anchor_step_gap")
        if not isinstance(self.output_shape, tuple) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.output_shape
        ):
            raise ValueError("output_shape must be a tuple of nonnegative integers")
        if not isinstance(self.output_dtype, str) or not self.output_dtype:
            raise ValueError("output_dtype must be a non-empty string")
        for value, name in (
            (self.output_norm, "output_norm"),
            (self.relative_output_change, "relative_output_change"),
            (self.relative_output_curvature, "relative_output_curvature"),
            (self.estimate_relative_error, "estimate_relative_error"),
            (self.estimate_seconds, "estimate_seconds"),
        ):
            _optional_nonnegative_float(value, name)
        _nonnegative_float(self.measurement_seconds, "measurement_seconds")
        if type(self.numerically_valid) is not bool:
            raise ValueError("numerically_valid must be a boolean")
        valid_statuses = {
            "history_not_ready",
            "measured",
            "invalid_actual_output",
            "invalid_estimated_output",
            "prediction_error",
        }
        if self.estimate_status not in valid_statuses:
            raise ValueError(f"unsupported estimate_status: {self.estimate_status!r}")
        if self.estimate_status == "measured":
            if self.estimate_relative_error is None:
                raise ValueError("measured anchor estimates require a relative error")
            if self.estimate_seconds is None:
                raise ValueError("measured anchor estimates require a duration")
        elif self.estimate_relative_error is not None:
            raise ValueError("unmeasured anchor estimates cannot have a relative error")
        if self.estimate_status in {"history_not_ready", "invalid_actual_output"}:
            if self.estimate_seconds is not None:
                raise ValueError(f"{self.estimate_status} cannot have an estimate duration")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record with explicit optional fields."""

        return {
            "step_index": self.step_index,
            "num_steps": self.num_steps,
            "timestep": self.timestep,
            "sigma": self.sigma,
            "decision_reason": self.decision_reason,
            "history_size": self.history_size,
            "estimated_steps_since_anchor": self.estimated_steps_since_anchor,
            "anchor_step_gap": self.anchor_step_gap,
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
            "output_norm": self.output_norm,
            "relative_output_change": self.relative_output_change,
            "relative_output_curvature": self.relative_output_curvature,
            "estimate_status": self.estimate_status,
            "estimate_relative_error": self.estimate_relative_error,
            "estimate_seconds": self.estimate_seconds,
            "measurement_seconds": self.measurement_seconds,
            "numerically_valid": self.numerically_valid,
        }


@dataclass(frozen=True)
class LatentUpdateMeasurement:
    """One scheduler update measured after a cache decision is completed.

    ``used_estimate`` records what the denoising loop actually supplied to the
    scheduler, rather than merely what the policy requested. Norms are L2
    norms computed in float32. ``relative_update`` is
    ``||after - before|| / max(||before||, 1e-12)``.
    """

    step_index: int
    num_steps: int
    timestep: float | None
    sigma: float | None
    decision_reason: str
    used_estimate: bool
    latent_shape: tuple[int, ...]
    latent_dtype: str
    before_norm: float | None
    after_norm: float | None
    update_norm: float | None
    relative_update: float | None
    measurement_seconds: float
    numerically_valid: bool

    def __post_init__(self) -> None:
        step_index = _nonnegative_int(self.step_index, "step_index")
        num_steps = _nonnegative_int(self.num_steps, "num_steps")
        if num_steps <= 0 or step_index >= num_steps:
            raise ValueError("step_index must identify a step in num_steps")
        _optional_finite_float(self.timestep, "timestep")
        _optional_finite_float(self.sigma, "sigma")
        _decision_reason(self.decision_reason, "latent-update decision reason")
        if type(self.used_estimate) is not bool:
            raise ValueError("used_estimate must be a boolean")
        if self.used_estimate and self.decision_reason != "policy_skip":
            raise ValueError("used estimates require a policy_skip decision")
        if not isinstance(self.latent_shape, tuple) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.latent_shape
        ):
            raise ValueError("latent_shape must be a tuple of nonnegative integers")
        if not isinstance(self.latent_dtype, str) or not self.latent_dtype:
            raise ValueError("latent_dtype must be a non-empty string")
        for value, name in (
            (self.before_norm, "before_norm"),
            (self.after_norm, "after_norm"),
            (self.update_norm, "update_norm"),
            (self.relative_update, "relative_update"),
        ):
            _optional_nonnegative_float(value, name)
        _nonnegative_float(self.measurement_seconds, "measurement_seconds")
        if type(self.numerically_valid) is not bool:
            raise ValueError("numerically_valid must be a boolean")
        norms = (
            self.before_norm,
            self.after_norm,
            self.update_norm,
            self.relative_update,
        )
        if self.numerically_valid and any(value is None for value in norms):
            raise ValueError("valid latent updates require all norm measurements")
        if not self.numerically_valid and any(value is not None for value in norms):
            raise ValueError("invalid latent updates cannot contain norm measurements")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record with explicit optional fields."""

        return {
            "step_index": self.step_index,
            "num_steps": self.num_steps,
            "timestep": self.timestep,
            "sigma": self.sigma,
            "decision_reason": self.decision_reason,
            "used_estimate": self.used_estimate,
            "latent_shape": list(self.latent_shape),
            "latent_dtype": self.latent_dtype,
            "before_norm": self.before_norm,
            "after_norm": self.after_norm,
            "update_norm": self.update_norm,
            "relative_update": self.relative_update,
            "measurement_seconds": self.measurement_seconds,
            "numerically_valid": self.numerically_valid,
        }


@runtime_checkable
class CacheMeasurementSink(Protocol):
    """Receive request-local measurements without influencing cache control.

    Implementations must accept valid records without raising and must not
    mutate policy, predictor, history, recovery, or scheduler state.
    """

    def record_anchor_measurement(self, measurement: AnchorMeasurement) -> None: ...

    def record_latent_update(self, measurement: LatentUpdateMeasurement) -> None: ...

    def clear(self) -> None: ...


class InMemoryMeasurementSink:
    """Keep request measurements in step order for experiments and tests."""

    def __init__(self) -> None:
        self._anchor_measurements: list[AnchorMeasurement] = []
        self._latent_updates: list[LatentUpdateMeasurement] = []

    def record_anchor_measurement(self, measurement: AnchorMeasurement) -> None:
        if not isinstance(measurement, AnchorMeasurement):
            raise TypeError("measurement sink requires an AnchorMeasurement")
        if (
            self._anchor_measurements
            and measurement.step_index <= self._anchor_measurements[-1].step_index
        ):
            raise ValueError("anchor measurements must have increasing step indices")
        self._anchor_measurements.append(measurement)

    def record_latent_update(self, measurement: LatentUpdateMeasurement) -> None:
        if not isinstance(measurement, LatentUpdateMeasurement):
            raise TypeError("measurement sink requires a LatentUpdateMeasurement")
        if self._latent_updates and measurement.step_index <= self._latent_updates[-1].step_index:
            raise ValueError("latent updates must have increasing step indices")
        self._latent_updates.append(measurement)

    def clear(self) -> None:
        self._anchor_measurements.clear()
        self._latent_updates.clear()

    def anchor_measurements(self) -> tuple[AnchorMeasurement, ...]:
        """Return the current request's immutable anchor records."""

        return tuple(self._anchor_measurements)

    def latent_updates(self) -> tuple[LatentUpdateMeasurement, ...]:
        """Return the current request's immutable scheduler-update records."""

        return tuple(self._latent_updates)

    def build_report(
        self,
        *,
        num_steps: int,
        configuration_source: str,
    ) -> CacheMeasurementReport:
        """Freeze the current request measurements into a strict report."""

        from difflet.pipeline.cache.measurement_report import CacheMeasurementReport

        return CacheMeasurementReport(
            num_steps=num_steps,
            configuration_source=configuration_source,
            anchor_measurements=self.anchor_measurements(),
            latent_updates=self.latent_updates(),
        )


def _torch_tensor(value: Any) -> bool:
    import torch

    return bool(torch.is_tensor(value))


def _tensor_is_finite(value: Any) -> bool:
    import torch

    return bool(torch.isfinite(value.detach()).all().item())


def _tensor_norm(value: Any) -> float:
    import torch

    flattened = value.detach().float().reshape(-1)
    return float(torch.linalg.vector_norm(flattened, ord=2).item())


def _relative_tensor_difference(left: Any, right: Any) -> float:
    difference = _tensor_norm(left.detach().float() - right.detach().float())
    denominator = max(_tensor_norm(right), _RELATIVE_NORM_FLOOR)
    return float(difference / denominator)


def _matching_tensor(left: Any, right: Any) -> bool:
    return _torch_tensor(left) and _torch_tensor(right) and tuple(left.shape) == tuple(right.shape)


def _finite_flag_and_norm(value: Any) -> tuple[bool, float]:
    """Return tensor validity and L2 norm with one device-to-host transfer."""

    import torch

    working = value.detach().float()
    summary = torch.stack(
        (
            torch.isfinite(working).all().to(dtype=torch.float32),
            torch.linalg.vector_norm(working.reshape(-1), ord=2),
        )
    )
    finite_flag, norm = summary.detach().cpu().tolist()
    return bool(finite_flag), float(norm)


def _finite_flag_and_difference_norm(estimate: Any, output: Any) -> tuple[bool, float]:
    """Summarize an estimate comparison with one device-to-host transfer."""

    import torch

    estimate_working = estimate.detach().float()
    output_working = output.detach().float()
    summary = torch.stack(
        (
            torch.isfinite(estimate_working).all().to(dtype=torch.float32),
            torch.linalg.vector_norm(
                (estimate_working - output_working).reshape(-1),
                ord=2,
            ),
        )
    )
    finite_flag, difference_norm = summary.detach().cpu().tolist()
    return bool(finite_flag), float(difference_norm)


def measure_latent_update(
    *,
    context: CacheStepContext,
    before: Any,
    after: Any,
    decision_reason: str,
    used_estimate: bool,
) -> LatentUpdateMeasurement:
    """Measure one scheduler latent update without changing either tensor."""

    started = time.perf_counter()
    if not _matching_tensor(before, after):
        raise ValueError("scheduler latent tensors must have identical shapes")
    shape = tuple(int(size) for size in before.shape)
    dtype = str(before.dtype).removeprefix("torch.")
    if after.dtype != before.dtype:
        raise ValueError("scheduler latent tensors must have identical dtypes")
    numerically_valid = _tensor_is_finite(before) and _tensor_is_finite(after)
    before_norm = None
    after_norm = None
    update_norm = None
    relative_update = None
    if numerically_valid:
        before_norm = _tensor_norm(before)
        after_norm = _tensor_norm(after)
        update_norm = _tensor_norm(after.detach().float() - before.detach().float())
        relative_update = float(update_norm / max(before_norm, _RELATIVE_NORM_FLOOR))
        numerically_valid = all(
            math.isfinite(value)
            for value in (before_norm, after_norm, update_norm, relative_update)
        )
        if not numerically_valid:
            before_norm = after_norm = update_norm = relative_update = None
    return LatentUpdateMeasurement(
        step_index=context.step_index,
        num_steps=context.num_steps,
        timestep=context.timestep,
        sigma=context.sigma,
        decision_reason=decision_reason,
        used_estimate=used_estimate,
        latent_shape=shape,
        latent_dtype=dtype,
        before_norm=before_norm,
        after_norm=after_norm,
        update_norm=update_norm,
        relative_update=relative_update,
        measurement_seconds=time.perf_counter() - started,
        numerically_valid=numerically_valid,
    )


def _relative_output_curvature(
    context: CacheStepContext,
    output: Any,
    history: CacheHistory,
    predictor: CachePredictor,
) -> float | None:
    if len(history) < 2:
        return None
    older, latest = history.tail(2)
    if not (
        _matching_tensor(older.output, latest.output) and _matching_tensor(latest.output, output)
    ):
        return None
    coordinate_kind = getattr(predictor, "coord", "index")
    if coordinate_kind not in ("index", "timestep", "sigma"):
        return None
    try:
        x0 = older.coordinate(coordinate_kind)
        x1 = latest.coordinate(coordinate_kind)
        x2 = context.coordinate(coordinate_kind)
    except ValueError:
        return None
    first_width = x1 - x0
    second_width = x2 - x1
    if first_width == 0.0 or second_width == 0.0:
        return None
    first_slope = (latest.output.detach().float() - older.output.detach().float()) / first_width
    second_slope = (output.detach().float() - latest.output.detach().float()) / second_width
    if not (_tensor_is_finite(first_slope) and _tensor_is_finite(second_slope)):
        return None
    slope_change = _tensor_norm(second_slope - first_slope)
    return float(slope_change / max(_tensor_norm(second_slope), _RELATIVE_NORM_FLOOR))


def measure_anchor_estimate(
    *,
    context: CacheStepContext,
    output: Any,
    decision_reason: str,
    predictor: CachePredictor,
    history: CacheHistory,
    observation: RuntimeObservation,
    tensor_observer: Callable[[CacheStepContext, Any, Any], None] | None = None,
) -> AnchorMeasurement:
    """Measure an anchor against a side-effect-free predictor using old history.

    ``CachePredictor.predict`` is required to be deterministic and externally
    side-effect free; request-local memoization is allowed. Configuration-related
    ``TypeError`` and ``ValueError`` failures are recorded as unavailable
    measurement. Runtime failures such as device errors and out-of-memory
    conditions are not caught.
    """

    started = time.perf_counter()
    history_size = len(history)
    latest = history.latest
    anchor_step_gap = None if latest is None else context.step_index - latest.step_index
    shape = tuple(int(size) for size in getattr(output, "shape", ()))
    dtype = str(getattr(output, "dtype", type(output).__name__)).removeprefix("torch.")
    actual_is_tensor = _torch_tensor(output)
    actual_is_finite = actual_is_tensor and _tensor_is_finite(output)
    output_norm = _tensor_norm(output) if actual_is_finite else None

    relative_output_change = None
    if (
        actual_is_finite
        and latest is not None
        and _matching_tensor(output, latest.output)
        and _tensor_is_finite(latest.output)
    ):
        relative_output_change = _relative_tensor_difference(output, latest.output)

    relative_output_curvature = None
    if actual_is_finite:
        relative_output_curvature = _relative_output_curvature(
            context,
            output,
            history,
            predictor,
        )

    estimate_status: AnchorEstimateStatus = "history_not_ready"
    estimate_relative_error = None
    estimate_seconds = None
    numerically_valid = actual_is_finite
    if not actual_is_finite:
        estimate_status = "invalid_actual_output"
    elif history.ready(predictor.required_history):
        estimate_started = time.perf_counter()
        try:
            estimate = predictor.predict(context, history)
        except (TypeError, ValueError):
            estimate_status = "prediction_error"
            numerically_valid = False
        else:
            if not _matching_tensor(estimate, output) or not _tensor_is_finite(estimate):
                estimate_status = "invalid_estimated_output"
                numerically_valid = False
            else:
                estimate_relative_error = _relative_tensor_difference(estimate, output)
                if not math.isfinite(estimate_relative_error):
                    estimate_status = "invalid_estimated_output"
                    estimate_relative_error = None
                    numerically_valid = False
                else:
                    estimate_status = "measured"
        estimate_seconds = time.perf_counter() - estimate_started
        if estimate_status == "measured" and tensor_observer is not None:
            # Optional experimental observers receive detached views only
            # after the ordinary scalar measurement is known to be valid.
            # Their output is never returned to the scheduler or controller.
            tensor_observer(context, estimate.detach(), output.detach())

    measurement_seconds = time.perf_counter() - started
    return AnchorMeasurement(
        step_index=context.step_index,
        num_steps=context.num_steps,
        timestep=context.timestep,
        sigma=context.sigma,
        decision_reason=decision_reason,
        history_size=history_size,
        estimated_steps_since_anchor=observation.consecutive_predictions,
        anchor_step_gap=anchor_step_gap,
        output_shape=shape,
        output_dtype=dtype,
        output_norm=output_norm,
        relative_output_change=relative_output_change,
        relative_output_curvature=relative_output_curvature,
        estimate_status=estimate_status,
        estimate_relative_error=estimate_relative_error,
        estimate_seconds=estimate_seconds,
        measurement_seconds=measurement_seconds,
        numerically_valid=numerically_valid,
    )


def measure_anchor_estimate_fast(
    *,
    context: CacheStepContext,
    output: Any,
    decision_reason: str,
    predictor: CachePredictor,
    history: CacheHistory,
    observation: RuntimeObservation,
) -> AnchorMeasurement:
    """Measure only the anchor error needed by an online cache policy.

    This serving path omits change and curvature diagnostics and packs each
    tensor summary into a single device-to-host transfer. The full measurement
    path remains active whenever an offline measurement sink is attached.
    """

    started = time.perf_counter()
    history_size = len(history)
    latest = history.latest
    anchor_step_gap = None if latest is None else context.step_index - latest.step_index
    shape = tuple(int(size) for size in getattr(output, "shape", ()))
    dtype = str(getattr(output, "dtype", type(output).__name__)).removeprefix("torch.")
    actual_is_tensor = _torch_tensor(output)
    output_norm = None
    actual_is_finite = False
    if actual_is_tensor:
        actual_is_finite, measured_norm = _finite_flag_and_norm(output)
        if actual_is_finite and math.isfinite(measured_norm):
            output_norm = measured_norm
        else:
            actual_is_finite = False

    estimate_status: AnchorEstimateStatus = "history_not_ready"
    estimate_relative_error = None
    estimate_seconds = None
    numerically_valid = actual_is_finite
    if not actual_is_finite:
        estimate_status = "invalid_actual_output"
    elif history.ready(predictor.required_history):
        estimate_started = time.perf_counter()
        try:
            estimate = predictor.predict(context, history)
        except (TypeError, ValueError):
            estimate_status = "prediction_error"
            numerically_valid = False
        else:
            if not _matching_tensor(estimate, output):
                estimate_status = "invalid_estimated_output"
                numerically_valid = False
            else:
                estimate_is_finite, difference_norm = _finite_flag_and_difference_norm(
                    estimate,
                    output,
                )
                if not estimate_is_finite or not math.isfinite(difference_norm):
                    estimate_status = "invalid_estimated_output"
                    numerically_valid = False
                else:
                    assert output_norm is not None
                    estimate_relative_error = float(
                        difference_norm / max(output_norm, _RELATIVE_NORM_FLOOR)
                    )
                    if math.isfinite(estimate_relative_error):
                        estimate_status = "measured"
                    else:
                        estimate_status = "invalid_estimated_output"
                        estimate_relative_error = None
                        numerically_valid = False
        estimate_seconds = time.perf_counter() - estimate_started

    return AnchorMeasurement(
        step_index=context.step_index,
        num_steps=context.num_steps,
        timestep=context.timestep,
        sigma=context.sigma,
        decision_reason=decision_reason,
        history_size=history_size,
        estimated_steps_since_anchor=observation.consecutive_predictions,
        anchor_step_gap=anchor_step_gap,
        output_shape=shape,
        output_dtype=dtype,
        output_norm=output_norm,
        relative_output_change=None,
        relative_output_curvature=None,
        estimate_status=estimate_status,
        estimate_relative_error=estimate_relative_error,
        estimate_seconds=estimate_seconds,
        measurement_seconds=time.perf_counter() - started,
        numerically_valid=numerically_valid,
    )


__all__ = [
    "AnchorMeasurement",
    "AnchorEstimateStatus",
    "CacheMeasurementSink",
    "InMemoryMeasurementSink",
    "LatentUpdateMeasurement",
    "measure_anchor_estimate",
    "measure_anchor_estimate_fast",
    "measure_latent_update",
]
