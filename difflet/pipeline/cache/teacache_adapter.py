"""Adapter for denoising loops that expose a TeaCache controller interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from difflet.pipeline.cache.runner import CacheRunner
from difflet.pipeline.cache.session import CacheSession, CacheSessionSnapshot


@dataclass(frozen=True)
class TeaCacheControllerSnapshot:
    """Opaque adapter checkpoint paired with its underlying session token."""

    _owner_id: int
    _session: CacheSessionSnapshot
    _last_delta_estimate: float | None


class TeaCacheControllerAdapter:
    """Expose a :class:`CacheSession` through the existing TeaCache loop API.

    The adapter contains no scheduling, estimation, recovery, or history
    logic. Its only responsibility is translating legacy method names and
    arguments into the framework-neutral session contract.
    """

    def __init__(self, session: CacheSession) -> None:
        if not isinstance(session, CacheSession):
            raise TypeError("TeaCacheControllerAdapter requires a CacheSession")
        self.session = session
        self.last_delta_estimate: float | None = None

    @property
    def runner(self) -> CacheRunner:
        """Expose the runner while existing diagnostics migrate to sessions."""

        return self.session.runner

    @property
    def num_steps(self) -> int:
        return self.session.num_steps

    @property
    def source(self) -> str:
        return self.session.configuration_source

    def reset(self) -> None:
        """Clear the session at legacy denoising-loop entry.

        This compatibility hook also discards recovery work requested before
        the loop starts.  New hosts should create one session per request and
        should not need to reset it before its first step.
        """

        self.session.clear_request_state()
        self.last_delta_estimate = None

    def bind_schedule(self, timesteps: Any, sigmas: Any = None) -> None:
        self.session.bind_schedule_coordinates(timesteps, sigmas)

    def snapshot(self) -> TeaCacheControllerSnapshot:
        """Capture controller state at a completed denoising-step boundary."""

        return TeaCacheControllerSnapshot(
            _owner_id=id(self),
            _session=self.session.snapshot(),
            _last_delta_estimate=self.last_delta_estimate,
        )

    def restore(self, snapshot: TeaCacheControllerSnapshot) -> None:
        """Restore and consume one controller checkpoint."""

        self._validate_snapshot(snapshot)
        self.session.restore(snapshot._session)
        self.last_delta_estimate = snapshot._last_delta_estimate

    def commit_snapshot(self, snapshot: TeaCacheControllerSnapshot) -> None:
        """Commit work since a checkpoint and consume its retained state."""

        self._validate_snapshot(snapshot)
        self.session.commit_snapshot(snapshot._session)

    def _validate_snapshot(self, snapshot: TeaCacheControllerSnapshot) -> None:
        if not isinstance(snapshot, TeaCacheControllerSnapshot):
            raise TypeError("controller snapshot must be a TeaCacheControllerSnapshot")
        if snapshot._owner_id != id(self):
            raise ValueError("controller snapshot belongs to another adapter")

    def needs_signal(self) -> bool:
        return self.session.policy_requires_signal()

    def needs_probe(self) -> bool:
        return self.session.policy_requires_probe()

    def should_skip(
        self,
        step_index: int,
        mod_input_now: Any = None,
        *,
        diff_norm: float | None = None,
    ) -> bool:
        del mod_input_now
        decision = self.session.decide_step(step_index, signal=diff_norm)
        self.last_delta_estimate = self.session.policy_delta_estimate
        return decision.should_skip

    def skip_noise_pred(self, mod_input: Any = None) -> Any:
        del mod_input
        step_index = self.session.active_step_index
        if step_index is None:
            raise RuntimeError("skip_noise_pred() requires a preceding accepted skip decision")
        return self.session.estimate_output(step_index)

    def record_full_step(self, noise_pred: Any, mod_input: Any = None) -> None:
        del mod_input
        step_index = self.session.active_step_index
        if step_index is None:
            raise RuntimeError("record_full_step() requires a preceding compute decision")
        self.session.record_anchor(step_index, noise_pred)

    def note_probe(self) -> None:
        self.session.record_probe_call()

    def request_quality_recovery(
        self,
        reason: str,
        *,
        steps: int | None = None,
        reset_history: bool = True,
    ) -> None:
        self.session.request_recovery(
            reason,
            steps=steps,
            reset_history=reset_history,
        )

    def stats(self) -> dict[str, Any]:
        return self.session.statistics()

    def anchor_error_trace(self) -> dict[str, Any]:
        """Expose request-scoped segment evidence to offline collectors."""

        return self.session.anchor_error_trace()


__all__ = ["TeaCacheControllerAdapter", "TeaCacheControllerSnapshot"]
