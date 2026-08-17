"""Framework-neutral, request-scoped entry point for cache execution."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from difflet.pipeline.cache.runner import CacheRunner, CacheRunnerSnapshot
from difflet.pipeline.cache.types import CacheDecision, CacheStepContext


@dataclass(frozen=True)
class CacheSessionSnapshot:
    """Opaque token for one active, request-scoped rollback checkpoint."""

    _owner_id: int
    _generation: int
    _runner: CacheRunnerSnapshot
    _contexts: tuple[tuple[int, CacheStepContext], ...]
    _timesteps: tuple[float, ...] | None
    _sigmas: tuple[float, ...] | None
    _policy_receipt: dict[str, Any] | None


class CacheSession:
    """Own all cache state for exactly one denoising request.

    A session translates host step coordinates into :class:`CacheStepContext`
    values and delegates cache decisions to a :class:`CacheRunner`.  It has no
    dependency on a model family, pipeline implementation, serving layer, or
    hardware backend.  A session must belong to exactly one request and must
    not be shared by concurrent requests.
    """

    def __init__(
        self,
        runner: CacheRunner,
        *,
        num_steps: int,
        configuration_source: str,
        barrier_steps: tuple[int, ...] = (),
        planned_anchor_steps: int | None = None,
        planned_estimate_steps: int | None = None,
    ) -> None:
        if not isinstance(runner, CacheRunner):
            raise TypeError("CacheSession requires a CacheRunner")
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        if not isinstance(configuration_source, str) or not configuration_source:
            raise ValueError("configuration_source must be a non-empty string")
        normalized_barriers = tuple(barrier_steps)
        if len(set(normalized_barriers)) != len(normalized_barriers):
            raise ValueError("barrier_steps must not contain duplicates")
        for step_index in normalized_barriers:
            if (
                isinstance(step_index, bool)
                or not isinstance(step_index, int)
                or not 0 <= step_index < num_steps
            ):
                raise ValueError("barrier_steps must contain valid denoising step indices")
        for value, name in (
            (planned_anchor_steps, "planned_anchor_steps"),
            (planned_estimate_steps, "planned_estimate_steps"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if (
            planned_anchor_steps is not None
            and planned_estimate_steps is not None
            and planned_anchor_steps + planned_estimate_steps != num_steps
        ):
            raise ValueError("planned anchor and estimate steps must cover the request")
        self.runner = runner
        self.num_steps = num_steps
        self.configuration_source = configuration_source
        self._barrier_steps = frozenset(normalized_barriers)
        self._planned_anchor_steps = planned_anchor_steps
        self._planned_estimate_steps = planned_estimate_steps
        self._contexts: dict[int, CacheStepContext] = {}
        self._timesteps: tuple[float, ...] | None = None
        self._sigmas: tuple[float, ...] | None = None
        self._active_step_index: int | None = None
        self._policy_receipt: dict[str, Any] | None = None
        self._snapshot_generation = 0
        self._active_snapshot: CacheSessionSnapshot | None = None

    def bind_policy_receipt(self, receipt: Mapping[str, Any]) -> None:
        """Bind one immutable executable/policy receipt to this request."""

        if self._active_snapshot is not None:
            raise RuntimeError("policy receipt cannot change while a rollback snapshot is active")
        if not isinstance(receipt, Mapping) or not receipt:
            raise ValueError("policy receipt must be a non-empty mapping")
        document = deepcopy(dict(receipt))
        if self._policy_receipt is not None and self._policy_receipt != document:
            raise RuntimeError("cache session is already bound to another policy receipt")
        self._policy_receipt = document

    @property
    def active_step_index(self) -> int | None:
        """Return the step whose decision still needs to be completed."""

        return self._active_step_index

    @property
    def policy_delta_estimate(self) -> float | None:
        """Return an optional scalar exposed by a signal-driven policy."""

        value = getattr(self.runner.policy, "last_delta_estimate", None)
        return None if value is None else float(value)

    @property
    def has_active_snapshot(self) -> bool:
        """Return whether this request currently owns an unconsumed checkpoint."""

        return self._active_snapshot is not None

    def snapshot(self) -> CacheSessionSnapshot:
        """Create the request's single rollback checkpoint.

        Checkpoints are accepted only between completed steps.  A checkpoint
        must be consumed by :meth:`restore` or :meth:`commit_snapshot` before a
        new one can be created, which bounds retained device state to one
        controller snapshot per request.
        """

        self._require_quiescent("snapshot")
        if self._active_snapshot is not None:
            raise RuntimeError("cache session already has an active rollback snapshot")
        self._snapshot_generation += 1
        snapshot = CacheSessionSnapshot(
            _owner_id=id(self),
            _generation=self._snapshot_generation,
            _runner=self.runner.snapshot(),
            _contexts=tuple(
                (step_index, deepcopy(context))
                for step_index, context in sorted(self._contexts.items())
            ),
            _timesteps=deepcopy(self._timesteps),
            _sigmas=deepcopy(self._sigmas),
            _policy_receipt=deepcopy(self._policy_receipt),
        )
        self._active_snapshot = snapshot
        return snapshot

    def restore(self, snapshot: CacheSessionSnapshot) -> None:
        """Consume ``snapshot`` and reopen the trajectory at its boundary."""

        self._validate_active_snapshot(snapshot)
        self._require_quiescent("restore")
        self.runner.restore(snapshot._runner)
        self._contexts = {
            step_index: deepcopy(context) for step_index, context in snapshot._contexts
        }
        self._timesteps = deepcopy(snapshot._timesteps)
        self._sigmas = deepcopy(snapshot._sigmas)
        self._policy_receipt = deepcopy(snapshot._policy_receipt)
        self._active_step_index = None
        self._active_snapshot = None

    def commit_snapshot(self, snapshot: CacheSessionSnapshot) -> None:
        """Consume a checkpoint while retaining all work completed after it."""

        self._validate_active_snapshot(snapshot)
        self._require_quiescent("commit snapshot")
        self._active_snapshot = None

    def _validate_active_snapshot(self, snapshot: CacheSessionSnapshot) -> None:
        if not isinstance(snapshot, CacheSessionSnapshot):
            raise TypeError("session snapshot must be a CacheSessionSnapshot")
        if snapshot._owner_id != id(self):
            raise ValueError("session snapshot belongs to another CacheSession")
        if self._active_snapshot is not snapshot:
            raise RuntimeError("session snapshot is not the active rollback checkpoint")

    def _require_quiescent(self, operation: str) -> None:
        if self._active_step_index is not None:
            raise RuntimeError(
                f"cannot {operation} cache session while step "
                f"{self._active_step_index} awaits output"
            )

    def clear_request_state(self) -> None:
        """Clear all trajectory state before the session is reused in tests.

        Production hosts should create a fresh session for every request.  The
        explicit reset operation remains available for existing pipeline
        lifecycle hooks and deterministic unit tests.  Resetting also clears
        any recovery request submitted before the denoising loop starts; hosts
        that use the legacy adapter must therefore submit external recovery
        requests after its loop-entry ``reset()`` call.
        """

        self.runner.reset()
        self._contexts.clear()
        self._timesteps = None
        self._sigmas = None
        self._active_step_index = None
        self._active_snapshot = None

    @staticmethod
    def _finite_coordinates(
        values: Any,
        name: str,
        num_steps: int,
    ) -> tuple[float, ...] | None:
        if values is None:
            return None
        result: list[float] = []
        for index, value in enumerate(values):
            if index >= num_steps:
                break
            item = getattr(value, "item", None)
            coordinate = float(item() if callable(item) else value)
            if not math.isfinite(coordinate):
                raise ValueError(f"cache {name}[{index}] must be finite")
            result.append(coordinate)
        if len(result) != num_steps:
            raise ValueError(f"cache session requires {num_steps} {name}, got {len(result)}")
        return tuple(result)

    def bind_schedule_coordinates(self, timesteps: Any, sigmas: Any = None) -> None:
        """Bind the scheduler coordinates for this request's denoising steps."""

        if self._active_step_index is not None:
            raise RuntimeError("schedule coordinates cannot change while a step awaits output")
        if self._active_snapshot is not None:
            raise RuntimeError(
                "schedule coordinates cannot change while a rollback snapshot is active"
            )
        self._timesteps = self._finite_coordinates(
            timesteps,
            "timesteps",
            self.num_steps,
        )
        # Diffusers schedulers commonly expose N+1 sigmas.  Only the N denoise
        # coordinates are relevant to transformer-output estimation.
        self._sigmas = self._finite_coordinates(
            sigmas,
            "sigmas",
            self.num_steps,
        )
        self._contexts.clear()

    def policy_requires_signal(self) -> bool:
        """Return whether the configured scheduling policy consumes a signal."""

        hook = getattr(self.runner.policy, "needs_signal", None)
        return bool(hook()) if callable(hook) else False

    def policy_requires_probe(self) -> bool:
        """Return whether the policy requires an additional model-side probe."""

        hook = getattr(self.runner.policy, "needs_probe", None)
        return bool(hook()) if callable(hook) else False

    def _context_for_step(
        self,
        step_index: int,
        *,
        signal: float | None = None,
    ) -> CacheStepContext:
        step_index = self._validate_step_index(step_index)
        context = self._contexts.get(step_index)
        if context is None:
            context = CacheStepContext(
                step_index=step_index,
                num_steps=self.num_steps,
                timestep=(None if self._timesteps is None else self._timesteps[step_index]),
                sigma=None if self._sigmas is None else self._sigmas[step_index],
                is_barrier=step_index in self._barrier_steps,
                signal=signal,
            )
            self._contexts[step_index] = context
        elif signal is not None and context.signal != signal:
            raise ValueError(f"cache step {step_index} was rebound with a different signal")
        return context

    def decide_step(
        self,
        step_index: int,
        *,
        signal: float | None = None,
    ) -> CacheDecision:
        """Return the single cache decision for one denoising step."""

        context = self._context_for_step(step_index, signal=signal)
        decision = self.runner.decide(context)
        self._active_step_index = context.step_index
        return decision

    def estimate_output(self, step_index: int) -> Any:
        """Complete an accepted estimate decision for ``step_index``."""

        context = self._require_active_context(step_index)
        output = self.runner.predict(context)
        self._active_step_index = None
        return output

    def record_anchor(self, step_index: int, output: Any) -> None:
        """Complete a compute decision with the real output for ``step_index``."""

        context = self._require_active_context(step_index)
        self.runner.record_anchor(context, output)
        self._active_step_index = None

    def _require_active_context(self, step_index: int) -> CacheStepContext:
        step_index = self._validate_step_index(step_index)
        if self._active_step_index is None:
            raise RuntimeError("no cache step is awaiting an output")
        if step_index != self._active_step_index:
            raise RuntimeError(
                f"cache step {self._active_step_index} is awaiting an output, "
                f"not step {step_index}"
            )
        context = self._contexts.get(step_index)
        if context is None:
            raise RuntimeError(f"cache step {step_index} has no bound context")
        return context

    def _validate_step_index(self, step_index: int) -> int:
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or not 0 <= step_index < self.num_steps
        ):
            raise ValueError(f"step_index must be an integer in [0, {self.num_steps})")
        return step_index

    def record_probe_call(self) -> None:
        """Record one host-side policy probe invocation."""

        self.runner.note_probe()

    def request_recovery(
        self,
        reason: str,
        *,
        steps: int | None = None,
        reset_history: bool = True,
    ) -> None:
        """Request a bounded real-compute recovery window for this session."""

        self.runner.request_quality_recovery(
            reason,
            steps=steps,
            reset_history=reset_history,
        )

    def statistics(self) -> dict[str, Any]:
        """Return cache execution and recovery counters for this request."""

        statistics: dict[str, Any] = dict(self.runner.stats())
        recovery_statistics = getattr(self.runner.recovery, "stats", None)
        if callable(recovery_statistics):
            statistics.update(recovery_statistics())
        policy_statistics = getattr(self.runner.policy, "stats", None)
        if callable(policy_statistics):
            statistics.update(policy_statistics())
        if self._planned_anchor_steps is not None:
            statistics["planned_anchor_steps"] = int(self._planned_anchor_steps)
        if self._planned_estimate_steps is not None:
            # Preserve the published statistics key even though the public API
            # consistently calls the operation an estimate rather than a skip.
            statistics["planned_skip_steps"] = int(self._planned_estimate_steps)
        statistics["source"] = self.configuration_source
        if self._policy_receipt is not None:
            statistics["policy_receipt"] = deepcopy(self._policy_receipt)
        return statistics

    def anchor_error_trace(self) -> dict[str, Any]:
        """Return per-segment endpoint measurements for this logical request path."""

        self._require_quiescent("read anchor-error trace")
        return self.runner.anchor_error_trace()


__all__ = ["CacheSession", "CacheSessionSnapshot"]
