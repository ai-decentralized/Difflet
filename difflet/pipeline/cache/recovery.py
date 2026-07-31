"""Independent online quality-recovery controls for diffusion caching.

The cache schedule and numerical predictor are research variables.  Recovery
is a safety envelope around both: it can force real transformer evaluations
without teaching a policy about predictor internals or teaching a predictor
about scheduling.

Static warmup/cooldown windows are the first implementation.  The explicit
``request_recovery`` hook is intentionally part of the same component so a
future adaptive monitor can request one or more fresh anchors without
changing Policy, Predictor, or the denoise loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from difflet.pipeline.cache.types import (
    CacheHistory,
    CacheStepContext,
    RecoveryDecision,
    RuntimeObservation,
)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return value


@dataclass(frozen=True)
class QualityRecoveryConfig:
    """Configuration for the runtime quality-recovery safety envelope.

    ``recovery_steps`` is the number of real anchors requested by an adaptive
    monitor after it calls :meth:`QualityRecoveryGuard.request_recovery`.
    ``max_consecutive_predictions`` is a quality bound independent of the
    predictor's mathematical capability bound.  The stricter bound wins.
    """

    warmup_steps: int = 0
    cooldown_steps: int = 0
    require_final_anchor: bool = False
    max_consecutive_predictions: int | None = None
    recovery_steps: int = 1

    def __post_init__(self) -> None:
        _nonnegative_int(self.warmup_steps, "warmup_steps")
        _nonnegative_int(self.cooldown_steps, "cooldown_steps")
        if type(self.require_final_anchor) is not bool:
            raise ValueError("require_final_anchor must be a boolean")
        _optional_positive_int(
            self.max_consecutive_predictions, "max_consecutive_predictions"
        )
        if (
            isinstance(self.recovery_steps, bool)
            or not isinstance(self.recovery_steps, int)
            or self.recovery_steps <= 0
        ):
            raise ValueError("recovery_steps must be a positive integer")

    def validate_num_steps(self, num_steps: int) -> None:
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps <= 0
        ):
            raise ValueError("num_steps must be a positive integer")

    def apply_to_anchor_mask(self, anchor_mask: tuple[bool, ...]) -> tuple[bool, ...]:
        """Materialize static recovery guarantees onto a schedule mask."""

        if (
            isinstance(anchor_mask, (str, bytes))
            or not hasattr(anchor_mask, "__len__")
            or any(type(item) is not bool for item in anchor_mask)
        ):
            raise ValueError("anchor_mask must be a sequence of booleans")
        effective = list(anchor_mask)
        self.validate_num_steps(len(effective))
        for index in range(min(self.warmup_steps, len(effective))):
            effective[index] = True
        if self.cooldown_steps:
            cooldown_start = max(len(effective) - self.cooldown_steps, 0)
            for index in range(cooldown_start, len(effective)):
                effective[index] = True
        if self.require_final_anchor:
            effective[-1] = True
        if self.max_consecutive_predictions is not None:
            run = 0
            for index, is_anchor in enumerate(effective):
                if is_anchor:
                    run = 0
                    continue
                run += 1
                if run > self.max_consecutive_predictions:
                    effective[index] = True
                    run = 0
        return tuple(effective)


class QualityRecoveryGuard:
    """Stateful, request-scoped quality recovery independent of scheduling."""

    def __init__(self, config: QualityRecoveryConfig | None = None) -> None:
        self.config = config or QualityRecoveryConfig()
        self._pending_steps = 0
        self._pending_reason: str | None = None
        self._reset_history_pending = False

    @property
    def pending_steps(self) -> int:
        return self._pending_steps

    def reset(self) -> None:
        """Reset request-scoped recovery state for a new generation."""

        self.reset_runtime_state()

    def reset_runtime_state(self) -> None:
        """Clear pending recovery work without altering Runner-owned metrics."""

        self._pending_steps = 0
        self._pending_reason = None
        self._reset_history_pending = False

    def request_recovery(
        self,
        reason: str,
        *,
        steps: int | None = None,
        reset_history: bool = True,
    ) -> None:
        """Request fresh anchors from an external/adaptive quality monitor.

        Requests coalesce conservatively: the larger remaining recovery window
        wins, and any request to invalidate history is retained.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("quality recovery reason must be a non-empty string")
        requested_steps = self.config.recovery_steps if steps is None else steps
        if (
            isinstance(requested_steps, bool)
            or not isinstance(requested_steps, int)
            or requested_steps <= 0
        ):
            raise ValueError("quality recovery steps must be a positive integer")
        if type(reset_history) is not bool:
            raise ValueError("reset_history must be a boolean")
        self._pending_steps = max(self._pending_steps, requested_steps)
        self._pending_reason = reason.strip()
        self._reset_history_pending = self._reset_history_pending or reset_history

    def before_step(
        self,
        context: CacheStepContext,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> RecoveryDecision:
        del history
        self.config.validate_num_steps(context.num_steps)

        if context.step_index < self.config.warmup_steps:
            return RecoveryDecision(True, "warmup")
        if (
            self.config.cooldown_steps
            and context.step_index >= context.num_steps - self.config.cooldown_steps
        ):
            return RecoveryDecision(True, "cooldown")
        if (
            self.config.require_final_anchor
            and context.step_index == context.num_steps - 1
        ):
            return RecoveryDecision(True, "final_anchor")
        if self._pending_steps:
            reset_history = self._reset_history_pending
            self._reset_history_pending = False
            return RecoveryDecision(True, "requested", reset_history=reset_history)
        maximum = self.config.max_consecutive_predictions
        if maximum is not None and observation.consecutive_predictions >= maximum:
            return RecoveryDecision(True, "consecutive_limit")
        return RecoveryDecision()

    def observe_anchor(
        self,
        context: CacheStepContext,
        output: Any,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> None:
        del context, output, history, observation
        if self._pending_steps:
            self._pending_steps -= 1
            if self._pending_steps == 0:
                self._pending_reason = None
                self._reset_history_pending = False

    def observe_prediction(
        self,
        context: CacheStepContext,
        output: Any,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> None:
        del context, output, history, observation

    def stats(self) -> dict[str, Any]:
        """Return live guard state; cumulative counts belong to CacheRunner."""

        return {
            "quality_recovery_pending_steps": int(self._pending_steps),
            "quality_recovery_reason": self._pending_reason,
        }


__all__ = ["QualityRecoveryConfig", "QualityRecoveryGuard"]
