"""Cache policies: the independent "when should we compute?" axis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from difflet.pipeline.cache.types import CacheHistory, CacheStepContext, RuntimeObservation


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _protected(context: CacheStepContext, warmup_steps: int, cooldown_steps: int) -> bool:
    return (
        context.step_index < warmup_steps
        or context.step_index >= context.num_steps - cooldown_steps
    )


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return result


@dataclass(frozen=True)
class AdaptiveAnchorConfig:
    """Bounds and signal thresholds for request-local anchor adaptation."""

    initial_anchor_interval: int
    minimum_anchor_interval: int
    maximum_anchor_interval: int
    warmup_steps: int
    cooldown_steps: int
    anchor_phase: int
    tighten_error: float
    recovery_error: float
    acceleration_error: float
    recovery_steps: int
    disable_after_recoveries: int
    stable_anchors_for_acceleration: int = 2
    allow_acceleration: bool = False
    require_final_anchor: bool = True

    def __post_init__(self) -> None:
        initial = _strict_int(
            self.initial_anchor_interval,
            "initial_anchor_interval",
            minimum=1,
        )
        minimum = _strict_int(
            self.minimum_anchor_interval,
            "minimum_anchor_interval",
            minimum=1,
        )
        maximum = _strict_int(
            self.maximum_anchor_interval,
            "maximum_anchor_interval",
            minimum=1,
        )
        if not minimum <= initial <= maximum:
            raise ValueError(
                "anchor intervals must satisfy minimum <= initial <= maximum"
            )
        _strict_int(self.warmup_steps, "warmup_steps", minimum=1)
        _strict_int(self.cooldown_steps, "cooldown_steps")
        phase = _strict_int(self.anchor_phase, "anchor_phase")
        if phase >= initial:
            raise ValueError("anchor_phase must be lower than initial_anchor_interval")
        acceleration = _finite_nonnegative(
            self.acceleration_error,
            "acceleration_error",
        )
        tighten = _finite_nonnegative(self.tighten_error, "tighten_error")
        recovery = _finite_nonnegative(self.recovery_error, "recovery_error")
        if not acceleration < tighten < recovery:
            raise ValueError(
                "signal thresholds must satisfy acceleration < tighten < recovery"
            )
        _strict_int(self.recovery_steps, "recovery_steps", minimum=1)
        _strict_int(
            self.disable_after_recoveries,
            "disable_after_recoveries",
            minimum=1,
        )
        _strict_int(
            self.stable_anchors_for_acceleration,
            "stable_anchors_for_acceleration",
            minimum=1,
        )
        if type(self.allow_acceleration) is not bool:
            raise ValueError("allow_acceleration must be a boolean")
        if type(self.require_final_anchor) is not bool:
            raise ValueError("require_final_anchor must be a boolean")


AdaptiveAnchorState = Literal["active", "recovery", "disabled"]


class AdaptiveAnchorPolicy:
    """Adjust anchor spacing from real-anchor errors within fixed hard bounds.

    With ``allow_acceleration=False`` the interval can only stay unchanged or
    shrink from its initial value.  The same implementation can later enable
    gradual acceleration after its thresholds and hard maximum are calibrated.
    Invalid comparison signals deterministically disable caching for the rest
    of the request.
    """

    def __init__(self, config: AdaptiveAnchorConfig) -> None:
        if not isinstance(config, AdaptiveAnchorConfig):
            raise TypeError("config must be an AdaptiveAnchorConfig")
        self.config = config
        self.warmup_steps = config.warmup_steps
        self.cooldown_steps = config.cooldown_steps
        self.require_final_anchor = config.require_final_anchor
        self.reset()

    def reset(self) -> None:
        self._state: AdaptiveAnchorState = "active"
        self._current_interval = self.config.initial_anchor_interval
        self._next_anchor_step: int | None = None
        self._recovery_steps_remaining = 0
        self._recovery_count = 0
        self._stable_anchor_count = 0
        self._tightening_count = 0
        self._acceleration_count = 0
        self._disabled_count = 0
        self._last_anchor_error: float | None = None

    @property
    def state(self) -> AdaptiveAnchorState:
        return self._state

    @property
    def current_anchor_interval(self) -> int:
        return self._current_interval

    def should_skip(
        self,
        context: CacheStepContext,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> bool:
        del history, observation
        if _protected(context, self.warmup_steps, self.cooldown_steps):
            return False
        if self._state != "active" or self._next_anchor_step is None:
            return False
        return context.step_index < self._next_anchor_step

    def observe_anchor_measurement(self, measurement: Any) -> None:
        """Consume the comparison produced before the real anchor is recorded."""

        step_index = int(measurement.step_index)
        num_steps = int(measurement.num_steps)
        if self._state == "disabled":
            return
        if self._state == "recovery":
            self._recovery_steps_remaining -= 1
            if self._recovery_steps_remaining <= 0:
                self._state = "active"
                self._schedule_after_anchor(step_index, num_steps)
            return

        status = measurement.estimate_status
        if status == "history_not_ready":
            self._schedule_after_anchor(step_index, num_steps)
            return
        if status != "measured" or not measurement.numerically_valid:
            self._disable()
            return
        error = float(measurement.estimate_relative_error)
        if not math.isfinite(error) or error < 0.0:
            self._disable()
            return
        self._last_anchor_error = error

        if error > self.config.recovery_error:
            self._current_interval = self.config.minimum_anchor_interval
            self._stable_anchor_count = 0
            self._recovery_count += 1
            if self._recovery_count >= self.config.disable_after_recoveries:
                self._disable()
                return
            self._state = "recovery"
            self._recovery_steps_remaining = self.config.recovery_steps
            self._next_anchor_step = None
            return

        if error > self.config.tighten_error:
            tightened = max(
                self.config.minimum_anchor_interval,
                self._current_interval // 2,
            )
            if tightened < self._current_interval:
                self._current_interval = tightened
                self._tightening_count += 1
            self._stable_anchor_count = 0
        elif (
            self.config.allow_acceleration
            and error < self.config.acceleration_error
        ):
            self._stable_anchor_count += 1
            if (
                self._stable_anchor_count
                >= self.config.stable_anchors_for_acceleration
                and self._current_interval < self.config.maximum_anchor_interval
            ):
                self._current_interval += 1
                self._acceleration_count += 1
                self._stable_anchor_count = 0
        else:
            self._stable_anchor_count = 0
        self._schedule_after_anchor(step_index, num_steps)

    def _schedule_after_anchor(self, step_index: int, num_steps: int) -> None:
        cooldown_start = num_steps - self.cooldown_steps
        if step_index < self.warmup_steps:
            if step_index == self.warmup_steps - 1:
                next_anchor = self.warmup_steps + self.config.anchor_phase
                self._next_anchor_step = (
                    next_anchor if next_anchor < cooldown_start else None
                )
            return
        if step_index >= cooldown_start:
            self._next_anchor_step = None
            return
        self._next_anchor_step = step_index + self._current_interval

    def _disable(self) -> None:
        if self._state != "disabled":
            self._disabled_count += 1
        self._state = "disabled"
        self._next_anchor_step = None
        self._recovery_steps_remaining = 0

    def stats(self) -> dict[str, Any]:
        return {
            "adaptive_state": self._state,
            "adaptive_current_anchor_interval": self._current_interval,
            "adaptive_tightenings": self._tightening_count,
            "adaptive_accelerations": self._acceleration_count,
            "adaptive_recoveries": self._recovery_count,
            "adaptive_disabled": self._disabled_count,
            "adaptive_recovery_steps_remaining": self._recovery_steps_remaining,
            "adaptive_last_anchor_error": self._last_anchor_error,
            "adaptive_acceleration_enabled": self.config.allow_acceleration,
        }


@dataclass(frozen=True)
class CadencePolicy:
    """Skip one step at the end of each fixed-size cadence window."""

    cadence: int
    warmup_steps: int = 0
    cooldown_steps: int = 0

    def __post_init__(self) -> None:
        _strict_int(self.cadence, "cadence", minimum=1)
        _strict_int(self.warmup_steps, "warmup_steps")
        _strict_int(self.cooldown_steps, "cooldown_steps")

    def should_skip(
        self,
        context: CacheStepContext,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> bool:
        del history, observation
        if _protected(context, self.warmup_steps, self.cooldown_steps):
            return False
        position = context.step_index - self.warmup_steps
        return position % self.cadence == self.cadence - 1

    def materialize_anchor_mask(self, num_steps: int) -> tuple[bool, ...]:
        return _materialize(self, num_steps)

    def reset(self) -> None:
        return None


@dataclass(frozen=True)
class PeriodicAnchorPolicy:
    """Compute periodic anchors, with optional all-compute edge windows."""

    anchor_interval: int
    anchor_phase: int = 0
    warmup_steps: int = 0
    cooldown_steps: int = 0
    require_final_anchor: bool = True

    def __post_init__(self) -> None:
        interval = _strict_int(self.anchor_interval, "anchor_interval", minimum=1)
        phase = _strict_int(self.anchor_phase, "anchor_phase")
        _strict_int(self.warmup_steps, "warmup_steps")
        _strict_int(self.cooldown_steps, "cooldown_steps")
        if phase >= interval:
            raise ValueError("anchor_phase must be lower than anchor_interval")
        if type(self.require_final_anchor) is not bool:
            raise ValueError("require_final_anchor must be a boolean")

    def should_skip(
        self,
        context: CacheStepContext,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> bool:
        del history, observation
        if _protected(context, self.warmup_steps, self.cooldown_steps):
            return False
        offset = (context.step_index - self.warmup_steps) - self.anchor_phase
        return offset % self.anchor_interval != 0

    def materialize_anchor_mask(self, num_steps: int) -> tuple[bool, ...]:
        return _materialize(self, num_steps)

    def reset(self) -> None:
        return None


class ExplicitMaskPolicy:
    """Execute an externally supplied anchor mask (True=compute, False=skip)."""

    def __init__(self, anchor_mask: Sequence[bool]) -> None:
        if isinstance(anchor_mask, (str, bytes)) or not isinstance(anchor_mask, Sequence):
            raise ValueError("anchor_mask must be a sequence of booleans")
        if not anchor_mask:
            raise ValueError("anchor_mask must not be empty")
        if any(type(item) is not bool for item in anchor_mask):
            raise ValueError("anchor_mask entries must be JSON booleans")
        self.anchor_mask = tuple(anchor_mask)

    def should_skip(
        self,
        context: CacheStepContext,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> bool:
        del history, observation
        if context.num_steps != len(self.anchor_mask):
            raise ValueError(
                f"explicit cache mask has {len(self.anchor_mask)} entries, "
                f"but runtime has {context.num_steps} steps"
            )
        return not self.anchor_mask[context.step_index]

    def materialize_anchor_mask(self, num_steps: int) -> tuple[bool, ...]:
        if int(num_steps) != len(self.anchor_mask):
            raise ValueError(
                f"explicit cache mask has {len(self.anchor_mask)} entries, "
                f"but {num_steps} were requested"
            )
        return self.anchor_mask

    def reset(self) -> None:
        return None


class TeaCachePolicy:
    """Dynamic TeaCache policy extracted from the legacy controller.

    ``calibration`` is intentionally duck-typed to avoid a dependency cycle
    with :mod:`difflet.pipeline.teacache`, where the legacy artifact dataclass
    remains public.
    """

    def __init__(self, calibration: Any) -> None:
        self.calibration = calibration
        self.last_delta_estimate: float | None = None
        self._skip_run_remaining = 0
        self._accum = 0.0
        self._last_full_delta: float | None = None
        self._baseline_delta: float | None = None
        self._just_skipped = False

    def reset(self) -> None:
        self.last_delta_estimate = None
        self._skip_run_remaining = 0
        self._accum = 0.0
        self._last_full_delta = None
        self._baseline_delta = None
        self._just_skipped = False

    def needs_signal(self) -> bool:
        return (
            int(self.calibration.cadence) <= 0
            and float(self.calibration.online_delta_alpha) <= 0.0
        )

    def needs_probe(self) -> bool:
        return self._skip_run_remaining <= 0

    def should_skip(
        self,
        context: CacheStepContext,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> bool:
        del history
        step_index = context.step_index
        if step_index < int(self.calibration.warmup_steps):
            self._skip_run_remaining = 0
            self._accum = 0.0
            return False
        if step_index >= context.num_steps - int(self.calibration.cooldown_steps):
            self._skip_run_remaining = 0
            self._accum = 0.0
            return False

        cadence = int(self.calibration.cadence)
        if cadence > 0:
            position = step_index - int(self.calibration.warmup_steps)
            return position % cadence == cadence - 1

        alpha = float(self.calibration.online_delta_alpha)
        if alpha > 0.0:
            if self._just_skipped:
                self._just_skipped = False
                return False
            if self._last_full_delta is None or self._baseline_delta is None:
                return False
            threshold = alpha * self._baseline_delta
            self.last_delta_estimate = self._last_full_delta
            return self._last_full_delta < threshold

        if self._skip_run_remaining > 0:
            self._skip_run_remaining -= 1
            return True

        if context.signal is None:
            return False
        signal = float(context.signal)
        if not math.isfinite(signal):
            raise ValueError("TeaCache policy signal must be finite")

        estimate = float(self.calibration.predict_delta(signal))
        if bool(self.calibration.accumulate):
            self._accum += abs(estimate)
            self.last_delta_estimate = self._accum
            if self._accum < float(self.calibration.threshold):
                return True
            self._accum = 0.0
            return False

        self.last_delta_estimate = estimate
        skip = estimate < float(self.calibration.threshold)
        self._skip_run_remaining = (
            max(int(self.calibration.skip_run_length) - 1, 0) if skip else 0
        )
        return skip

    def observe_anchor(
        self,
        context: CacheStepContext,
        output: Any,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> None:
        del context, history
        if float(self.calibration.online_delta_alpha) > 0.0 and observation.last_output is not None:
            current = output.detach()
            previous = observation.last_output
            denominator = previous.abs().mean().clamp_min(1e-8)
            delta = float((current - previous).abs().mean() / denominator)
            self._last_full_delta = delta
            if self._baseline_delta is None:
                self._baseline_delta = delta
        self._just_skipped = False

    def observe_prediction(
        self,
        context: CacheStepContext,
        output: Any,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> None:
        del context, output, history, observation
        self._just_skipped = True


def _materialize(policy: Any, num_steps: int) -> tuple[bool, ...]:
    steps = _strict_int(num_steps, "num_steps", minimum=1)
    history = CacheHistory(1)
    observation = RuntimeObservation()
    return tuple(
        not policy.should_skip(
            CacheStepContext(step_index=index, num_steps=steps),
            history,
            observation,
        )
        for index in range(steps)
    )


__all__ = [
    "AdaptiveAnchorConfig",
    "AdaptiveAnchorPolicy",
    "CadencePolicy",
    "ExplicitMaskPolicy",
    "PeriodicAnchorPolicy",
    "TeaCachePolicy",
]
