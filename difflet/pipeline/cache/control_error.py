"""Minimal Taylor anchor-error measurement for online cache control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from difflet.pipeline.cache.types import CacheHistory, CachePredictor, CacheStepContext


AnchorEstimateStatus = Literal[
    "history_not_ready",
    "measured",
    "invalid_actual_output",
    "invalid_estimated_output",
    "prediction_error",
]

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


__all__ = ["AnchorErrorMeasurement", "AnchorEstimateStatus", "measure_anchor_error"]
