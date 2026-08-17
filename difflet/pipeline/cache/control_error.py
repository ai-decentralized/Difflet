"""Minimal Taylor anchor-error measurement for online cache control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from difflet.pipeline.cache.types import CacheHistory, CachePredictor, CacheStepContext


AnchorEstimateStatus = Literal[
    "history_not_ready",
    "measured",
    "invalid_actual_output",
    "invalid_estimated_output",
    "prediction_error",
]

ANCHOR_ERROR_TRACE_SCHEMA = "difflet-cache-anchor-error-trace"
ANCHOR_ERROR_TRACE_SCHEMA_REVISION = 1

_RELATIVE_NORM_FLOOR = 1e-12
_STATUSES = {
    "history_not_ready",
    "measured",
    "invalid_actual_output",
    "invalid_estimated_output",
    "prediction_error",
}


@dataclass(frozen=True)
class AnchorErrorMeasurement:
    """The scalar result consumed by an anchor scheduling policy."""

    step_index: int
    num_steps: int
    estimate_status: AnchorEstimateStatus
    estimate_relative_error: float | None
    numerically_valid: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or isinstance(self.num_steps, bool)
            or not isinstance(self.num_steps, int)
            or self.num_steps <= 0
            or not 0 <= self.step_index < self.num_steps
        ):
            raise ValueError("step_index must identify a step in num_steps")
        if self.estimate_status not in _STATUSES:
            raise ValueError(f"unsupported estimate_status: {self.estimate_status!r}")
        if type(self.numerically_valid) is not bool:
            raise ValueError("numerically_valid must be a boolean")
        error = self.estimate_relative_error
        if self.estimate_status == "measured":
            if isinstance(error, bool) or error is None:
                raise ValueError("measured anchor estimates require a relative error")
            value = float(error)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("estimate_relative_error must be finite and nonnegative")
            if not self.numerically_valid:
                raise ValueError("measured anchor estimates must be numerically valid")
        elif error is not None:
            raise ValueError("unmeasured anchor estimates cannot have a relative error")


@dataclass(frozen=True)
class AnchorErrorTraceEntry:
    """One endpoint measurement bound to the segment that produced it.

    This is logical-path evidence: if a speculative segment is rolled back,
    its entry must disappear with the runner snapshot.  A future physical
    transaction ledger belongs outside this structure so rollback cost is not
    accidentally erased with logical controller state.
    """

    previous_anchor_step_index: int | None
    anchor_step_index: int
    num_steps: int
    estimate_step_indices: tuple[int, ...]
    previous_timestep: float | None
    anchor_timestep: float | None
    previous_sigma: float | None
    anchor_sigma: float | None
    scheduler_signed_delta_sigma: float | None
    scheduler_abs_delta_sigma: float | None
    estimate_status: AnchorEstimateStatus
    endpoint_z: float | None
    numerically_valid: bool

    @classmethod
    def from_measurement(
        cls,
        measurement: AnchorErrorMeasurement,
        *,
        context: CacheStepContext,
        previous_anchor: Any | None,
        estimate_contexts: Sequence[CacheStepContext],
    ) -> "AnchorErrorTraceEntry":
        """Bind a scalar endpoint error to its exact logical cache segment."""

        if not isinstance(measurement, AnchorErrorMeasurement):
            raise TypeError("measurement must be an AnchorErrorMeasurement")
        if not isinstance(context, CacheStepContext):
            raise TypeError("context must be a CacheStepContext")
        if (
            context.step_index != measurement.step_index
            or context.num_steps != measurement.num_steps
        ):
            raise ValueError("measurement and anchor context coordinates differ")
        contexts = tuple(estimate_contexts)
        if any(not isinstance(context, CacheStepContext) for context in contexts):
            raise TypeError("estimate_contexts must contain CacheStepContext values")
        indices = tuple(context.step_index for context in contexts)
        if indices != tuple(sorted(set(indices))):
            raise ValueError("estimated steps must be strictly increasing")
        if indices and indices[-1] >= measurement.step_index:
            raise ValueError("estimated steps must precede the measured anchor")

        previous_context = None if previous_anchor is None else previous_anchor.context
        previous_step = None if previous_context is None else int(previous_context.step_index)
        if previous_step is not None:
            if previous_context.num_steps != measurement.num_steps:
                raise ValueError("previous anchor num_steps differs from the measurement")
            expected = tuple(range(previous_step + 1, measurement.step_index))
            if indices != expected:
                raise ValueError(
                    "estimated steps must exactly cover the segment between real anchors"
                )
        elif indices:
            raise ValueError("estimated steps cannot precede the first real anchor")

        if any(candidate.num_steps != measurement.num_steps for candidate in contexts):
            raise ValueError("estimated-step num_steps differs from the measurement")

        sigma_deltas: tuple[float, ...] | None = ()
        if contexts:
            coordinate_contexts = (*contexts, context)
            if any(candidate.sigma is None for candidate in coordinate_contexts):
                sigma_deltas = None
            else:
                sigma_deltas = tuple(
                    float(right.sigma) - float(left.sigma)
                    for left, right in zip(
                        coordinate_contexts[:-1], coordinate_contexts[1:], strict=True
                    )
                )
        signed_delta = (
            None if sigma_deltas is None else float(sum(sigma_deltas))
        )
        abs_delta = (
            None if sigma_deltas is None else float(sum(abs(value) for value in sigma_deltas))
        )

        return cls(
            previous_anchor_step_index=previous_step,
            anchor_step_index=measurement.step_index,
            num_steps=measurement.num_steps,
            estimate_step_indices=indices,
            previous_timestep=(
                None if previous_context is None else previous_context.timestep
            ),
            anchor_timestep=context.timestep,
            previous_sigma=None if previous_context is None else previous_context.sigma,
            anchor_sigma=context.sigma,
            scheduler_signed_delta_sigma=signed_delta,
            scheduler_abs_delta_sigma=abs_delta,
            estimate_status=measurement.estimate_status,
            endpoint_z=measurement.estimate_relative_error,
            numerically_valid=measurement.numerically_valid,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe record without device tensors or references."""

        anchor_gap = (
            None
            if self.previous_anchor_step_index is None
            else self.anchor_step_index - self.previous_anchor_step_index
        )
        return {
            "previous_anchor_step_index": self.previous_anchor_step_index,
            "anchor_step_index": self.anchor_step_index,
            "num_steps": self.num_steps,
            "anchor_gap": anchor_gap,
            "estimate_step_count": len(self.estimate_step_indices),
            "estimate_step_indices": list(self.estimate_step_indices),
            "previous_timestep": self.previous_timestep,
            "anchor_timestep": self.anchor_timestep,
            "previous_sigma": self.previous_sigma,
            "anchor_sigma": self.anchor_sigma,
            "scheduler_signed_delta_sigma": self.scheduler_signed_delta_sigma,
            "scheduler_abs_delta_sigma": self.scheduler_abs_delta_sigma,
            "estimate_status": self.estimate_status,
            "endpoint_z": self.endpoint_z,
            "numerically_valid": self.numerically_valid,
        }


def _torch_tensor(value: Any) -> bool:
    import torch

    return bool(torch.is_tensor(value))


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
    """Return estimate validity and difference norm in one host transfer."""

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


def measure_anchor_error(
    *,
    context: CacheStepContext,
    output: Any,
    predictor: CachePredictor,
    history: CacheHistory,
) -> AnchorErrorMeasurement:
    """Compare the Taylor estimate with a real anchor using scalar summaries."""

    actual_is_finite = False
    output_norm = None
    if _torch_tensor(output):
        actual_is_finite, measured_norm = _finite_flag_and_norm(output)
        if actual_is_finite and math.isfinite(measured_norm):
            output_norm = measured_norm
        else:
            actual_is_finite = False

    status: AnchorEstimateStatus = "history_not_ready"
    error = None
    numerically_valid = actual_is_finite
    if not actual_is_finite:
        status = "invalid_actual_output"
    elif history.ready(predictor.required_history):
        try:
            estimate = predictor.predict(context, history)
        except (TypeError, ValueError):
            status = "prediction_error"
            numerically_valid = False
        else:
            if not _matching_tensor(estimate, output):
                status = "invalid_estimated_output"
                numerically_valid = False
            else:
                estimate_is_finite, difference_norm = _finite_flag_and_difference_norm(
                    estimate,
                    output,
                )
                if not estimate_is_finite or not math.isfinite(difference_norm):
                    status = "invalid_estimated_output"
                    numerically_valid = False
                else:
                    assert output_norm is not None
                    error = float(difference_norm / max(output_norm, _RELATIVE_NORM_FLOOR))
                    if math.isfinite(error):
                        status = "measured"
                    else:
                        status = "invalid_estimated_output"
                        error = None
                        numerically_valid = False
    return AnchorErrorMeasurement(
        step_index=context.step_index,
        num_steps=context.num_steps,
        estimate_status=status,
        estimate_relative_error=error,
        numerically_valid=numerically_valid,
    )


__all__ = [
    "ANCHOR_ERROR_TRACE_SCHEMA",
    "ANCHOR_ERROR_TRACE_SCHEMA_REVISION",
    "AnchorErrorMeasurement",
    "AnchorErrorTraceEntry",
    "AnchorEstimateStatus",
    "measure_anchor_error",
]
