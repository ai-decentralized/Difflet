"""Shared types for the diffusion-cache runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

Coordinate = Literal["index", "timestep", "sigma"]


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _optional_finite_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _detach(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    return detach() if callable(detach) else value


@dataclass(frozen=True)
class CacheStepContext:
    """The scheduler identity and optional policy signal for one denoise step."""

    step_index: int
    num_steps: int
    timestep: float | None = None
    sigma: float | None = None
    is_barrier: bool = False
    signal: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        index = _strict_nonnegative_int(self.step_index, "step_index")
        steps = _strict_nonnegative_int(self.num_steps, "num_steps")
        if steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        if index >= steps:
            raise ValueError(f"step_index {index} is outside a {steps}-step trajectory")
        if type(self.is_barrier) is not bool:
            raise ValueError("is_barrier must be a boolean")
        object.__setattr__(self, "timestep", _optional_finite_float(self.timestep, "timestep"))
        object.__setattr__(self, "sigma", _optional_finite_float(self.sigma, "sigma"))
        object.__setattr__(self, "signal", _optional_finite_float(self.signal, "signal"))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")

    def coordinate(self, kind: Coordinate) -> float:
        if kind == "index":
            return float(self.step_index)
        value = self.timestep if kind == "timestep" else self.sigma
        if value is None:
            raise ValueError(
                f"cache predictor requires {kind}, but step {self.step_index} has none"
            )
        return float(value)


@dataclass(frozen=True)
class CacheAnchor:
    """A real transformer output that may be used by a predictor."""

    context: CacheStepContext
    output: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", _detach(self.output))

    @property
    def step_index(self) -> int:
        return self.context.step_index

    @property
    def timestep(self) -> float | None:
        return self.context.timestep

    @property
    def sigma(self) -> float | None:
        return self.context.sigma

    def coordinate(self, kind: Coordinate) -> float:
        return self.context.coordinate(kind)


class CacheHistory:
    """Bounded history containing real anchors only."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("history capacity must be a positive integer")
        self.capacity = capacity
        self._anchors: list[CacheAnchor] = []

    def push(self, anchor: CacheAnchor) -> None:
        if not isinstance(anchor, CacheAnchor):
            raise TypeError("CacheHistory accepts CacheAnchor values only")
        if self._anchors and anchor.step_index <= self._anchors[-1].step_index:
            raise ValueError("cache anchors must be recorded in strictly increasing step order")
        self._anchors.append(anchor)
        if len(self._anchors) > self.capacity:
            del self._anchors[: len(self._anchors) - self.capacity]

    def clear(self) -> None:
        self._anchors.clear()

    def ready(self, required: int) -> bool:
        return len(self._anchors) >= int(required)

    @property
    def anchors(self) -> tuple[CacheAnchor, ...]:
        return tuple(self._anchors)

    @property
    def size(self) -> int:
        return len(self._anchors)

    @property
    def latest(self) -> CacheAnchor | None:
        return self._anchors[-1] if self._anchors else None

    def tail(self, count: int) -> tuple[CacheAnchor, ...]:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("history tail count must be a positive integer")
        return tuple(self._anchors[-count:])

    def __len__(self) -> int:
        return len(self._anchors)

    def __iter__(self):
        return iter(self._anchors)

    def __getitem__(self, index):
        return self._anchors[index]


@dataclass
class RuntimeObservation:
    """Mutable runtime state that may include scheduler-accepted predictions."""

    last_output: Any | None = None
    last_step_index: int | None = None
    last_was_prediction: bool = False
    consecutive_predictions: int = 0
    policy_state: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.last_output = None
        self.last_step_index = None
        self.last_was_prediction = False
        self.consecutive_predictions = 0
        self.policy_state.clear()

    def record(self, context: CacheStepContext, output: Any, *, predicted: bool) -> None:
        self.last_output = _detach(output)
        self.last_step_index = context.step_index
        self.last_was_prediction = bool(predicted)
        self.consecutive_predictions = (
            self.consecutive_predictions + 1 if predicted else 0
        )


@dataclass(frozen=True)
class CacheDecision:
    """Runner decision for one step."""

    should_skip: bool
    reason: Literal[
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
    ]

    @property
    def should_compute(self) -> bool:
        return not self.should_skip

    def __bool__(self) -> bool:
        return self.should_skip


@dataclass
class CacheRunnerStats:
    full_steps: int = 0
    skipped_steps: int = 0
    policy_skip_requests: int = 0
    readiness_rejections: int = 0
    consecutive_skip_vetoes: int = 0
    barrier_resets: int = 0
    recovery_forced_steps: int = 0
    recovery_triggers: int = 0
    recovery_history_resets: int = 0
    probe_calls: int = 0

    def reset(self) -> None:
        self.full_steps = 0
        self.skipped_steps = 0
        self.policy_skip_requests = 0
        self.readiness_rejections = 0
        self.consecutive_skip_vetoes = 0
        self.barrier_resets = 0
        self.recovery_forced_steps = 0
        self.recovery_triggers = 0
        self.recovery_history_resets = 0
        self.probe_calls = 0

    def to_dict(self, *, history_size: int = 0) -> dict[str, int]:
        return {
            "full_steps": int(self.full_steps),
            "skipped_steps": int(self.skipped_steps),
            "policy_skip_requests": int(self.policy_skip_requests),
            "readiness_rejections": int(self.readiness_rejections),
            "consecutive_skip_vetoes": int(self.consecutive_skip_vetoes),
            "barrier_resets": int(self.barrier_resets),
            "recovery_forced_steps": int(self.recovery_forced_steps),
            "recovery_triggers": int(self.recovery_triggers),
            "recovery_history_resets": int(self.recovery_history_resets),
            "probe_calls": int(self.probe_calls),
            "history_size": int(history_size),
        }


@dataclass(frozen=True)
class RecoveryDecision:
    """An independent quality guard's decision before policy evaluation."""

    force_compute: bool = False
    reason: Literal[
        "none",
        "warmup",
        "cooldown",
        "final_anchor",
        "consecutive_limit",
        "requested",
    ] = "none"
    reset_history: bool = False

    def __post_init__(self) -> None:
        if type(self.force_compute) is not bool:
            raise ValueError("force_compute must be a boolean")
        if type(self.reset_history) is not bool:
            raise ValueError("reset_history must be a boolean")
        if not self.force_compute and self.reason != "none":
            raise ValueError("a non-empty recovery reason requires force_compute=true")
        if self.reset_history and not self.force_compute:
            raise ValueError("reset_history requires force_compute=true")


@runtime_checkable
class CachePolicy(Protocol):
    """Policy protocol: decide where predictions may replace real computation."""

    def should_skip(
        self,
        context: CacheStepContext,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> bool: ...

    def reset(self) -> None: ...


@runtime_checkable
class CachePredictor(Protocol):
    """Predictor protocol: estimate a missing transformer output."""

    required_history: int
    max_consecutive_predictions: int | None

    def predict(self, context: CacheStepContext, history: CacheHistory) -> Any: ...


@runtime_checkable
class CacheRecovery(Protocol):
    """Independent quality-recovery guard around a policy/predictor pair."""

    def before_step(
        self,
        context: CacheStepContext,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> RecoveryDecision: ...

    def observe_anchor(
        self,
        context: CacheStepContext,
        output: Any,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> None: ...

    def observe_prediction(
        self,
        context: CacheStepContext,
        output: Any,
        history: CacheHistory,
        observation: RuntimeObservation,
    ) -> None: ...

    def reset(self) -> None: ...


# Short aliases retained for the names used in the architecture note.
Context = CacheStepContext
StepContext = CacheStepContext
Anchor = CacheAnchor

__all__ = [
    "Anchor",
    "CacheAnchor",
    "CacheDecision",
    "CacheHistory",
    "CachePolicy",
    "CachePredictor",
    "CacheRecovery",
    "CacheRunnerStats",
    "CacheStepContext",
    "Context",
    "Coordinate",
    "RecoveryDecision",
    "RuntimeObservation",
    "StepContext",
]
