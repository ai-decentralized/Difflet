"""Production-loop adapter: drive a CacheRunner behind the TeaCacheController
interface consumed by the denoise loops (``should_skip`` / ``skip_noise_pred``
/ ``record_full_step``), so a resolved cache plan runs through the existing
pipeline code without touching the loop itself.
"""

from __future__ import annotations

import math
from typing import Any

from difflet.pipeline.cache.runner import CacheRunner
from difflet.pipeline.cache.spec import ResolvedCacheConfig
from difflet.pipeline.cache.types import CacheStepContext


class CacheRuntimeController:
    """Drive a CacheRunner through the production denoise-loop interface.

    Static plan/mask schedules need no probe. Dynamic policies may opt into a
    signal by implementing ``needs_signal`` and ``needs_probe``. Predictor
    coordinates may use index, timestep, or sigma after :meth:`bind_schedule`.
    """

    def __init__(
        self,
        runner: CacheRunner,
        *,
        num_steps: int,
        source: str,
        barrier_steps: tuple[int, ...] = (),
        planned_anchor_steps: int | None = None,
        planned_skip_steps: int | None = None,
    ) -> None:
        if not isinstance(runner, CacheRunner):
            raise TypeError("CacheRuntimeController requires a CacheRunner")
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps <= 0
        ):
            raise ValueError("num_steps must be a positive integer")
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string")
        self.num_steps = num_steps
        self.source = source
        self._barrier_steps = frozenset(barrier_steps)
        self._planned_anchor_steps = planned_anchor_steps
        self._planned_skip_steps = planned_skip_steps
        self.runner = runner
        self._contexts: dict[int, CacheStepContext] = {}
        self._timesteps: tuple[float, ...] | None = None
        self._sigmas: tuple[float, ...] | None = None
        self.last_delta_estimate: float | None = None

    def reset(self) -> None:
        self.runner.reset()
        self._contexts.clear()
        self._timesteps = None
        self._sigmas = None
        self.last_delta_estimate = None

    @staticmethod
    def _coordinates(values: Any, name: str, num_steps: int) -> tuple[float, ...] | None:
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
            raise ValueError(
                f"cache controller requires {num_steps} {name}, got {len(result)}"
            )
        return tuple(result)

    def bind_schedule(self, timesteps: Any, sigmas: Any = None) -> None:
        """Bind request-specific scheduler coordinates after timestep retrieval."""

        self._timesteps = self._coordinates(timesteps, "timesteps", self.num_steps)
        # Diffusers schedulers commonly expose N+1 sigmas. Only the N denoise
        # coordinates are relevant to transformer-output prediction.
        self._sigmas = self._coordinates(sigmas, "sigmas", self.num_steps)
        self._contexts.clear()

    def needs_signal(self) -> bool:
        hook = getattr(self.runner.policy, "needs_signal", None)
        return bool(hook()) if callable(hook) else False

    def needs_probe(self) -> bool:
        hook = getattr(self.runner.policy, "needs_probe", None)
        return bool(hook()) if callable(hook) else False

    def _context(
        self,
        step_index: int,
        *,
        signal: float | None = None,
    ) -> CacheStepContext:
        step_index = int(step_index)
        context = self._contexts.get(step_index)
        if context is None:
            context = CacheStepContext(
                step_index=step_index,
                num_steps=self.num_steps,
                timestep=(
                    None if self._timesteps is None else self._timesteps[step_index]
                ),
                sigma=None if self._sigmas is None else self._sigmas[step_index],
                is_barrier=step_index in self._barrier_steps,
                signal=signal,
            )
            self._contexts[step_index] = context
        elif signal is not None and context.signal != signal:
            raise ValueError(
                f"cache step {step_index} was rebound with a different signal"
            )
        return context

    def should_skip(
        self,
        step_index: int,
        mod_input_now: Any = None,
        *,
        diff_norm: float | None = None,
    ) -> bool:
        del mod_input_now
        decision = self.runner.decide(
            self._context(step_index, signal=diff_norm)
        )
        self.last_delta_estimate = getattr(
            self.runner.policy, "last_delta_estimate", None
        )
        return decision.should_skip

    def skip_noise_pred(self, mod_input: Any = None) -> Any:
        del mod_input
        return self.runner.predict()

    def record_full_step(self, noise_pred: Any, mod_input: Any = None) -> None:
        del mod_input
        self.runner.record_full_step(noise_pred)

    def note_probe(self) -> None:
        self.runner.note_probe()

    def request_quality_recovery(
        self,
        reason: str,
        *,
        steps: int | None = None,
        reset_history: bool = True,
    ) -> None:
        self.runner.request_quality_recovery(
            reason,
            steps=steps,
            reset_history=reset_history,
        )

    def stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = dict(self.runner.stats())
        recovery_stats = getattr(self.runner.recovery, "stats", None)
        if callable(recovery_stats):
            stats.update(recovery_stats())
        if self._planned_anchor_steps is not None:
            stats["planned_anchor_steps"] = int(self._planned_anchor_steps)
        if self._planned_skip_steps is not None:
            stats["planned_skip_steps"] = int(self._planned_skip_steps)
        stats["source"] = self.source
        return stats


class CachePlanController(CacheRuntimeController):
    """Execute a strictly resolved plan/mask in a production denoise loop."""

    def __init__(self, config: ResolvedCacheConfig) -> None:
        if not isinstance(config, ResolvedCacheConfig):
            raise TypeError("CachePlanController requires a ResolvedCacheConfig")
        self.config = config
        super().__init__(
            config.build_runner(),
            num_steps=config.num_steps,
            source=config.source,
            barrier_steps=config.barrier_steps,
            planned_anchor_steps=config.planned_anchor_steps,
            planned_skip_steps=config.planned_skip_steps,
        )


__all__ = ["CachePlanController", "CacheRuntimeController"]
