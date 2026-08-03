"""Runtime composition and safety checks for cache policies and predictors."""

from __future__ import annotations

from typing import Any, Callable

from difflet.pipeline.cache.measurements import (
    CacheMeasurementSink,
    measure_anchor_estimate,
)
from difflet.pipeline.cache.types import (
    CacheAnchor,
    CacheDecision,
    CacheHistory,
    CachePolicy,
    CachePredictor,
    CacheRecovery,
    CacheRunnerStats,
    CacheStepContext,
    RecoveryDecision,
    RuntimeObservation,
)


class CacheRunner:
    """Execute a policy/predictor pair without contaminating true history."""

    def __init__(
        self,
        policy: CachePolicy,
        predictor: CachePredictor,
        *,
        recovery: CacheRecovery | None = None,
        history_capacity: int | None = None,
        measurement_sink: CacheMeasurementSink | None = None,
    ) -> None:
        if not isinstance(policy, CachePolicy):
            raise TypeError("policy must implement CachePolicy")
        if not isinstance(predictor, CachePredictor):
            raise TypeError("predictor must implement CachePredictor")
        if recovery is None:
            from difflet.pipeline.cache.recovery import (
                QualityRecoveryConfig,
                QualityRecoveryGuard,
            )

            calibration = getattr(policy, "calibration", None)
            recovery = QualityRecoveryGuard(
                QualityRecoveryConfig(
                    warmup_steps=int(
                        getattr(
                            policy,
                            "warmup_steps",
                            getattr(calibration, "warmup_steps", 0),
                        )
                    ),
                    cooldown_steps=int(
                        getattr(
                            policy,
                            "cooldown_steps",
                            getattr(calibration, "cooldown_steps", 0),
                        )
                    ),
                    require_final_anchor=bool(getattr(policy, "require_final_anchor", False)),
                )
            )
        if not isinstance(recovery, CacheRecovery):
            raise TypeError("recovery must implement CacheRecovery")
        required = int(predictor.required_history)
        if required <= 0:
            raise ValueError("predictor.required_history must be positive")
        if history_capacity is not None and (
            isinstance(history_capacity, bool)
            or not isinstance(history_capacity, int)
            or history_capacity <= 0
        ):
            raise ValueError("history_capacity must be a positive integer")
        capacity = history_capacity if history_capacity is not None else required
        if capacity < required:
            raise ValueError(
                f"history_capacity={capacity} is lower than predictor requirement {required}"
            )
        if measurement_sink is not None and not isinstance(measurement_sink, CacheMeasurementSink):
            raise TypeError("measurement_sink must implement CacheMeasurementSink")
        _validate_static_pair(policy, predictor, recovery)
        self.policy = policy
        self.predictor = predictor
        self.recovery = recovery
        self.history = CacheHistory(capacity)
        self.observation = RuntimeObservation()
        self.counters = CacheRunnerStats()
        self.measurement_sink = measurement_sink
        self._pending_context: CacheStepContext | None = None
        self._compute_context: CacheStepContext | None = None
        self._last_context: CacheStepContext | None = None
        self._pending_decision_reason: str | None = None
        self._compute_decision_reason: str | None = None

    @property
    def ready(self) -> bool:
        return self.history.ready(self.predictor.required_history)

    @property
    def pending_context(self) -> CacheStepContext | None:
        return self._pending_context

    def reset(self, *, reset_policy: bool = True) -> None:
        self.history.clear()
        self.observation.reset()
        self.counters.reset()
        self._pending_context = None
        self._compute_context = None
        self._last_context = None
        self._pending_decision_reason = None
        self._compute_decision_reason = None
        if reset_policy:
            self.policy.reset()
        self.recovery.reset()
        if self.measurement_sink is not None:
            self.measurement_sink.clear()

    def reset_history(
        self,
        *,
        reset_policy: bool = True,
        reset_recovery: bool = True,
    ) -> None:
        """Invalidate trajectory state while retaining aggregate counters."""

        self.history.clear()
        self.observation.reset()
        self._pending_context = None
        self._compute_context = None
        self._last_context = None
        self._pending_decision_reason = None
        self._compute_decision_reason = None
        if reset_policy:
            self.policy.reset()
        if reset_recovery:
            reset_runtime_state = getattr(self.recovery, "reset_runtime_state", None)
            if callable(reset_runtime_state):
                reset_runtime_state()
            else:
                # Third-party recovery implementations only have the protocol's
                # full reset hook. Built-in guards preserve cumulative metrics.
                self.recovery.reset()

    def decide(self, context: CacheStepContext) -> CacheDecision:
        if self._pending_context is not None or self._compute_context is not None:
            raise RuntimeError(
                "the previous cache decision has not been completed with "
                "predict() or record_anchor()"
            )
        self._validate_step_order(context)
        if context.is_barrier:
            # A semantic barrier invalidates trajectory-dependent history, but
            # it must not discard an independent quality-recovery request.
            # The real barrier output counts as one requested fresh anchor.
            self.reset_history(reset_policy=True, reset_recovery=False)
            self.counters.barrier_resets += 1
            self._last_context = context
            self._compute_context = context
            self._compute_decision_reason = "barrier"
            return CacheDecision(False, "barrier")

        recovery = self.recovery.before_step(context, self.history, self.observation)
        if not isinstance(recovery, RecoveryDecision):
            raise TypeError("recovery.before_step() must return a RecoveryDecision")
        if recovery.force_compute:
            if recovery.reset_history:
                # Keep the active recovery request while invalidating every
                # trajectory-dependent policy/predictor state.
                self.reset_history(reset_policy=True, reset_recovery=False)
                self.counters.recovery_history_resets += 1
            self.counters.recovery_forced_steps += 1
            self._last_context = context
            self._compute_context = context
            recovery_reasons = {
                "warmup": "recovery_warmup",
                "cooldown": "recovery_cooldown",
                "final_anchor": "recovery_final_anchor",
                "consecutive_limit": "recovery_consecutive_limit",
                "requested": "recovery_requested",
            }
            decision_reason = recovery_reasons.get(recovery.reason)
            if decision_reason is None:
                raise ValueError(f"unsupported force-compute recovery reason: {recovery.reason!r}")
            self._compute_decision_reason = decision_reason
            return CacheDecision(False, decision_reason)

        requested = bool(self.policy.should_skip(context, self.history, self.observation))
        if not requested:
            self._last_context = context
            self._compute_context = context
            self._compute_decision_reason = "policy_compute"
            return CacheDecision(False, "policy_compute")

        self.counters.policy_skip_requests += 1
        if not self.ready:
            self.counters.readiness_rejections += 1
            self._last_context = context
            self._compute_context = context
            self._compute_decision_reason = "history_not_ready"
            return CacheDecision(False, "history_not_ready")

        maximum = self.predictor.max_consecutive_predictions
        if maximum is not None and self.observation.consecutive_predictions >= int(maximum):
            self.counters.consecutive_skip_vetoes += 1
            self._last_context = context
            self._compute_context = context
            self._compute_decision_reason = "consecutive_skip_veto"
            return CacheDecision(False, "consecutive_skip_veto")

        self._pending_context = context
        self._last_context = context
        self._pending_decision_reason = "policy_skip"
        return CacheDecision(True, "policy_skip")

    def should_skip(self, context: CacheStepContext) -> bool:
        return self.decide(context).should_skip

    def predict(self, context: CacheStepContext | None = None) -> Any:
        if self._pending_context is None:
            raise RuntimeError("predict() requires a preceding accepted skip decision")
        context = context or self._pending_context
        if context != self._pending_context:
            raise RuntimeError("prediction context does not match the pending skip decision")
        output = self.predictor.predict(context, self.history)
        hook = getattr(self.policy, "observe_prediction", None)
        if callable(hook):
            hook(context, output, self.history, self.observation)
        self.recovery.observe_prediction(context, output, self.history, self.observation)
        self.observation.record(context, output, predicted=True)
        self.counters.skipped_steps += 1
        self._pending_context = None
        self._pending_decision_reason = None
        return output

    def record_anchor(self, context: CacheStepContext, output: Any) -> None:
        if self._compute_context is not None and context != self._compute_context:
            raise RuntimeError("anchor context does not match the pending compute decision")
        if self._pending_context is not None and context != self._pending_context:
            raise RuntimeError("anchor context does not match the pending skip decision")
        decision_reason = (
            self._pending_decision_reason or self._compute_decision_reason or "direct_compute"
        )
        measurement = None
        measurement_hook = getattr(
            self.policy,
            "observe_anchor_measurement",
            None,
        )
        tensor_observer = getattr(
            self.measurement_sink,
            "observe_anchor_tensors",
            None,
        )
        if not callable(tensor_observer):
            tensor_observer = None
        if self.measurement_sink is not None or callable(measurement_hook):
            measurement = measure_anchor_estimate(
                context=context,
                output=output,
                decision_reason=decision_reason,
                predictor=self.predictor,
                history=self.history,
                observation=self.observation,
                tensor_observer=tensor_observer,
            )
        if measurement is not None and callable(measurement_hook):
            measurement_hook(measurement)
        if self._pending_context is not None:
            # A caller may elect to compute after seeing a skip decision. That
            # is safe, but the stale pending decision must not leak.
            self._pending_context = None
            self._pending_decision_reason = None
        self._compute_context = None
        self._compute_decision_reason = None
        hook = getattr(self.policy, "observe_anchor", None)
        if callable(hook):
            hook(context, output, self.history, self.observation)
        self.history.push(CacheAnchor(context=context, output=output))
        self.recovery.observe_anchor(context, output, self.history, self.observation)
        self.observation.record(context, output, predicted=False)
        self.counters.full_steps += 1
        self._last_context = context
        if measurement is not None:
            if self.measurement_sink is not None:
                self.measurement_sink.record_anchor_measurement(measurement)

    def record_full_step(self, output: Any, context: CacheStepContext | None = None) -> None:
        context = context or self._last_context
        if context is None:
            raise RuntimeError("record_full_step() requires a step context")
        self.record_anchor(context, output)

    def run_step(
        self,
        context: CacheStepContext,
        compute: Callable[[], Any],
    ) -> tuple[Any, bool]:
        decision = self.decide(context)
        if decision.should_skip:
            return self.predict(context), True
        output = compute()
        self.record_anchor(context, output)
        return output, False

    def note_probe(self) -> None:
        self.counters.probe_calls += 1

    def request_quality_recovery(
        self,
        reason: str,
        *,
        steps: int | None = None,
        reset_history: bool = True,
    ) -> None:
        request = getattr(self.recovery, "request_recovery", None)
        if not callable(request):
            raise RuntimeError(
                "the configured recovery component does not accept adaptive requests"
            )
        request(reason, steps=steps, reset_history=reset_history)
        self.counters.recovery_triggers += 1

    def stats(self) -> dict[str, int]:
        return self.counters.to_dict(history_size=len(self.history))

    def _validate_step_order(self, context: CacheStepContext) -> None:
        if self._last_context is None:
            return
        if context.step_index < self._last_context.step_index:
            raise ValueError(
                "cache step indices moved backwards; call reset() before a new trajectory"
            )
        if context.step_index == self._last_context.step_index:
            raise ValueError(f"cache step {context.step_index} was decided more than once")


def _validate_static_pair(
    policy: CachePolicy,
    predictor: CachePredictor,
    recovery: CacheRecovery,
) -> None:
    maximum = predictor.max_consecutive_predictions
    if maximum is None:
        return
    recovery_config = getattr(recovery, "config", None)
    recovery_maximum = getattr(recovery_config, "max_consecutive_predictions", None)
    if recovery_maximum is not None and int(recovery_maximum) <= int(maximum):
        # The independent recovery envelope inserts a real anchor before the
        # predictor's mathematical limit can be exceeded.
        return
    from difflet.pipeline.cache.policies import (
        CadencePolicy,
        ExplicitMaskPolicy,
        PeriodicAnchorPolicy,
        TeaCachePolicy,
    )

    if isinstance(policy, CadencePolicy) and policy.cadence == 1:
        raise ValueError(
            "cadence=1 requests unbounded consecutive skips, but the predictor "
            f"supports at most {maximum}"
        )
    if isinstance(policy, PeriodicAnchorPolicy) and policy.anchor_interval - 1 > maximum:
        raise ValueError(
            f"periodic policy can request {policy.anchor_interval - 1} consecutive skips, "
            f"but the predictor supports at most {maximum}"
        )
    if isinstance(policy, ExplicitMaskPolicy):
        run = _longest_false_run(policy.anchor_mask)
        if run > maximum:
            raise ValueError(
                f"explicit mask requests {run} consecutive skips, "
                f"but the predictor supports at most {maximum}"
            )
    if isinstance(policy, TeaCachePolicy) and int(policy.calibration.skip_run_length) > maximum:
        raise ValueError(
            f"TeaCache skip_run_length={policy.calibration.skip_run_length} exceeds "
            f"the predictor limit {maximum}"
        )


def _longest_false_run(mask) -> int:
    longest = current = 0
    for anchor in mask:
        if anchor:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


__all__ = ["CacheRunner"]
