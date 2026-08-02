"""Framework-neutral, request-scoped entry point for cache execution."""

from __future__ import annotations

import math
from typing import Any

from difflet.pipeline.cache.runner import CacheRunner
from difflet.pipeline.cache.spec import ResolvedCacheConfig
from difflet.pipeline.cache.measurements import (
    CacheMeasurementSink,
    measure_latent_update,
)
from difflet.pipeline.cache.types import CacheDecision, CacheStepContext


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
        self._decisions: dict[int, CacheDecision] = {}
        self._completed_with_estimate: dict[int, bool] = {}
        self._measured_latent_steps: set[int] = set()

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
    def measurements_enabled(self) -> bool:
        """Return whether this request has an attached measurement sink."""

        return self.runner.measurement_sink is not None

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
        self._decisions.clear()
        self._completed_with_estimate.clear()
        self._measured_latent_steps.clear()

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
        self._decisions[context.step_index] = decision
        return decision

    def estimate_output(self, step_index: int) -> Any:
        """Complete an accepted estimate decision for ``step_index``."""

        context = self._require_active_context(step_index)
        output = self.runner.predict(context)
        self._completed_with_estimate[context.step_index] = True
        self._active_step_index = None
        return output

    def record_anchor(self, step_index: int, output: Any) -> None:
        """Complete a compute decision with the real output for ``step_index``."""

        context = self._require_active_context(step_index)
        self.runner.record_anchor(context, output)
        self._completed_with_estimate[context.step_index] = False
        self._active_step_index = None

    def record_latent_update(self, step_index: int, before: Any, after: Any) -> None:
        """Measure the scheduler update after this cache step has completed."""

        sink = self.runner.measurement_sink
        if sink is None:
            return
        step_index = self._validate_step_index(step_index)
        decision = self._decisions.get(step_index)
        if decision is None or step_index not in self._completed_with_estimate:
            raise RuntimeError(
                f"cache step {step_index} must complete before its latent update is measured"
            )
        if step_index in self._measured_latent_steps:
            raise RuntimeError(f"cache step {step_index} latent update was measured twice")
        context = self._contexts.get(step_index)
        if context is None:
            raise RuntimeError(f"cache step {step_index} has no bound context")
        measurement = measure_latent_update(
            context=context,
            before=before,
            after=after,
            decision_reason=decision.reason,
            used_estimate=self._completed_with_estimate[step_index],
        )
        sink.record_latent_update(measurement)
        self._measured_latent_steps.add(step_index)

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
        if self._planned_anchor_steps is not None:
            statistics["planned_anchor_steps"] = int(self._planned_anchor_steps)
        if self._planned_estimate_steps is not None:
            # Preserve the published statistics key even though the public API
            # consistently calls the operation an estimate rather than a skip.
            statistics["planned_skip_steps"] = int(self._planned_estimate_steps)
        statistics["source"] = self.configuration_source
        return statistics


class ResolvedCacheSession(CacheSession):
    """Create a request session from a fully validated static cache config."""

    def __init__(
        self,
        config: ResolvedCacheConfig,
        *,
        measurement_sink: CacheMeasurementSink | None = None,
    ) -> None:
        if not isinstance(config, ResolvedCacheConfig):
            raise TypeError("ResolvedCacheSession requires a ResolvedCacheConfig")
        self.config = config
        super().__init__(
            config.build_runner(measurement_sink=measurement_sink),
            num_steps=config.num_steps,
            configuration_source=config.source,
            barrier_steps=config.barrier_steps,
            planned_anchor_steps=config.planned_anchor_steps,
            planned_estimate_steps=config.planned_skip_steps,
        )


__all__ = ["CacheSession", "ResolvedCacheSession"]
