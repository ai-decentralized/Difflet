"""Runtime composition and safety checks for cache policies and predictors."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from difflet.pipeline.cache.control_error import (
    ANCHOR_ERROR_TRACE_SCHEMA,
    ANCHOR_ERROR_TRACE_SCHEMA_REVISION,
    AnchorErrorTraceEntry,
    measure_anchor_error,
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


@dataclass(frozen=True)
class _ComponentSnapshot:
    uses_hook: bool
    state: Any


@dataclass(frozen=True)
class CacheRunnerSnapshot:
    """Opaque, owner-bound state captured at a completed step boundary."""

    _owner_id: int
    _history: tuple[CacheAnchor, ...]
    _observation: RuntimeObservation
    _counters: CacheRunnerStats
    _last_context: CacheStepContext | None
    _segment_estimate_contexts: tuple[CacheStepContext, ...]
    _anchor_error_trace: tuple[AnchorErrorTraceEntry, ...]
    _policy: _ComponentSnapshot
    _recovery: _ComponentSnapshot


class CacheRunner:
    """Execute a policy/predictor pair without contaminating true history."""

    def __init__(
        self,
        policy: CachePolicy,
        predictor: CachePredictor,
        *,
        recovery: CacheRecovery | None = None,
        history_capacity: int | None = None,
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
        _validate_static_pair(policy, predictor, recovery)
        self.policy = policy
        self.predictor = predictor
        self.recovery = recovery
        self.history = CacheHistory(capacity)
        self.observation = RuntimeObservation()
        self.counters = CacheRunnerStats()
        self._pending_context: CacheStepContext | None = None
        self._compute_context: CacheStepContext | None = None
        self._last_context: CacheStepContext | None = None
        self._segment_estimate_contexts: list[CacheStepContext] = []
        self._anchor_error_trace: list[AnchorErrorTraceEntry] = []

    @property
    def ready(self) -> bool:
        return self.history.ready(self.predictor.required_history)

    @property
    def pending_context(self) -> CacheStepContext | None:
        return self._pending_context

    def reset(self, *, reset_policy: bool = True) -> None:
        self.history.clear()
        self._reset_predictor_cache()
        self.observation.reset()
        self.counters.reset()
        self._pending_context = None
        self._compute_context = None
        self._last_context = None
        self._segment_estimate_contexts.clear()
        self._anchor_error_trace.clear()
        if reset_policy:
            self.policy.reset()
        self.recovery.reset()

    def snapshot(self) -> CacheRunnerSnapshot:
        """Capture all request-derived runner state at a quiescent boundary."""

        self._require_quiescent("snapshot")
        return CacheRunnerSnapshot(
            _owner_id=id(self),
            _history=tuple(_clone_anchor(anchor) for anchor in self.history),
            _observation=_clone_observation(self.observation),
            _counters=deepcopy(self.counters),
            _last_context=deepcopy(self._last_context),
            _segment_estimate_contexts=tuple(
                deepcopy(context) for context in self._segment_estimate_contexts
            ),
            _anchor_error_trace=tuple(deepcopy(self._anchor_error_trace)),
            _policy=_snapshot_component(self.policy),
            _recovery=_snapshot_component(self.recovery),
        )

    def restore(self, snapshot: CacheRunnerSnapshot) -> None:
        """Restore one explicit snapshot without weakening normal step ordering."""

        if not isinstance(snapshot, CacheRunnerSnapshot):
            raise TypeError("runner snapshot must be a CacheRunnerSnapshot")
        if snapshot._owner_id != id(self):
            raise ValueError("runner snapshot belongs to another CacheRunner")
        self._require_quiescent("restore")
        history = CacheHistory(self.history.capacity)
        for anchor in snapshot._history:
            history.push(_clone_anchor(anchor))
        self.history = history
        self.observation = _clone_observation(snapshot._observation)
        self.counters = deepcopy(snapshot._counters)
        self._pending_context = None
        self._compute_context = None
        self._last_context = deepcopy(snapshot._last_context)
        self._segment_estimate_contexts = [
            deepcopy(context) for context in snapshot._segment_estimate_contexts
        ]
        self._anchor_error_trace = list(deepcopy(snapshot._anchor_error_trace))
        _restore_component(self.policy, snapshot._policy)
        _restore_component(self.recovery, snapshot._recovery)
        # Predictor memoization is derived solely from restored real anchors.
        # Recompute lazily so a cached tensor can never alias pre-rollback state.
        self._reset_predictor_cache()

    def reset_history(
        self,
        *,
        reset_policy: bool = True,
        reset_recovery: bool = True,
    ) -> None:
        """Invalidate trajectory state while retaining aggregate counters."""

        self.history.clear()
        self._reset_predictor_cache()
        self.observation.reset()
        self._pending_context = None
        self._compute_context = None
        self._last_context = None
        self._segment_estimate_contexts.clear()
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
            return CacheDecision(False, decision_reason)

        requested = bool(self.policy.should_skip(context, self.history, self.observation))
        if not requested:
            self._last_context = context
            self._compute_context = context
            return CacheDecision(False, "policy_compute")

        self.counters.policy_skip_requests += 1
        if not self.ready:
            self.counters.readiness_rejections += 1
            self._last_context = context
            self._compute_context = context
            return CacheDecision(False, "history_not_ready")

        maximum = self.predictor.max_consecutive_predictions
        if maximum is not None and self.observation.consecutive_predictions >= int(maximum):
            self.counters.consecutive_skip_vetoes += 1
            self._last_context = context
            self._compute_context = context
            return CacheDecision(False, "consecutive_skip_veto")

        self._pending_context = context
        self._last_context = context
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
        self._segment_estimate_contexts.append(deepcopy(context))
        self._pending_context = None
        return output

    def record_anchor(self, context: CacheStepContext, output: Any) -> None:
        if self._compute_context is not None and context != self._compute_context:
            raise RuntimeError("anchor context does not match the pending compute decision")
        if self._pending_context is not None and context != self._pending_context:
            raise RuntimeError("anchor context does not match the pending skip decision")
        measurement = None
        measurement_hook = getattr(
            self.policy,
            "observe_anchor_measurement",
            None,
        )
        if callable(measurement_hook):
            measurement = measure_anchor_error(
                context=context,
                output=output,
                predictor=self.predictor,
                history=self.history,
            )
        trace_entry = None
        if measurement is not None:
            trace_entry = AnchorErrorTraceEntry.from_measurement(
                measurement,
                context=context,
                previous_anchor=self.history.latest,
                estimate_contexts=self._segment_estimate_contexts,
            )
        if measurement is not None and callable(measurement_hook):
            measurement_hook(measurement)
        if trace_entry is not None:
            self._anchor_error_trace.append(trace_entry)
        if self._pending_context is not None:
            # A caller may elect to compute after seeing a skip decision. That
            # is safe, but the stale pending decision must not leak.
            self._pending_context = None
        self._compute_context = None
        hook = getattr(self.policy, "observe_anchor", None)
        if callable(hook):
            hook(context, output, self.history, self.observation)
        self.history.push(CacheAnchor(context=context, output=output))
        # Coefficients derived from the pre-anchor history are now stale. Drop
        # them immediately so a run of real steps does not retain old device
        # tensors until the next prediction.
        self._reset_predictor_cache()
        self.recovery.observe_anchor(context, output, self.history, self.observation)
        self.observation.record(context, output, predicted=False)
        self.counters.full_steps += 1
        self._last_context = context
        self._segment_estimate_contexts.clear()

    def _reset_predictor_cache(self) -> None:
        reset_predictor_cache = getattr(self.predictor, "reset_cache", None)
        if callable(reset_predictor_cache):
            reset_predictor_cache()

    def _require_quiescent(self, operation: str) -> None:
        if self._pending_context is not None or self._compute_context is not None:
            raise RuntimeError(
                f"cannot {operation} cache runner while a step decision awaits completion"
            )

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

    def anchor_error_trace(self) -> dict[str, Any]:
        """Return the completed request's rollback-aware logical trace."""

        self._require_quiescent("read anchor-error trace")
        return {
            "schema": ANCHOR_ERROR_TRACE_SCHEMA,
            "schema_revision": ANCHOR_ERROR_TRACE_SCHEMA_REVISION,
            "measurement": "endpoint_transformer_output_relative_l2",
            "path_semantics": "logical_post_restore_path",
            "physical_rollback_attempts_included": False,
            "entries": [entry.to_dict() for entry in self._anchor_error_trace],
        }

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
        PhasedStaticPolicy,
        StaticPlusBrakePolicy,
        TeaCachePolicy,
    )

    if isinstance(policy, (PhasedStaticPolicy, StaticPlusBrakePolicy)):
        run = _longest_false_run(policy.anchor_mask)
        if run > maximum:
            raise ValueError(
                f"static anchor mask requests {run} consecutive skips, "
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


def _clone_runtime_value(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().clone()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {
            _clone_runtime_value(key): _clone_runtime_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_runtime_value(item) for item in value)
    if isinstance(value, set):
        return {_clone_runtime_value(item) for item in value}
    return deepcopy(value)


def _clone_anchor(anchor: CacheAnchor) -> CacheAnchor:
    return CacheAnchor(
        context=deepcopy(anchor.context),
        output=_clone_runtime_value(anchor.output),
    )


def _clone_observation(observation: RuntimeObservation) -> RuntimeObservation:
    return RuntimeObservation(
        last_output=_clone_runtime_value(observation.last_output),
        last_step_index=observation.last_step_index,
        last_was_prediction=observation.last_was_prediction,
        consecutive_predictions=observation.consecutive_predictions,
        policy_state=_clone_runtime_value(observation.policy_state),
    )


def _snapshot_component(component: Any) -> _ComponentSnapshot:
    hook = getattr(component, "snapshot_state", None)
    if callable(hook):
        if not callable(getattr(component, "restore_state", None)):
            raise TypeError(
                f"{type(component).__name__} snapshot_state() requires restore_state()"
            )
        return _ComponentSnapshot(True, _clone_runtime_value(hook()))
    try:
        attributes = vars(component)
    except TypeError as error:
        raise TypeError(
            f"{type(component).__name__} must expose snapshot_state() for rollback"
        ) from error
    # The protocol does not prescribe how third-party components name mutable
    # request state.  Capture every instance attribute instead of relying on a
    # private-name convention that could silently omit a public counter.
    state = {key: _clone_runtime_value(value) for key, value in attributes.items()}
    return _ComponentSnapshot(False, state)


def _restore_component(component: Any, snapshot: _ComponentSnapshot) -> None:
    if snapshot.uses_hook:
        hook = getattr(component, "restore_state", None)
        if not callable(hook):
            raise TypeError(
                f"{type(component).__name__} snapshot_state() requires restore_state()"
            )
        hook(_clone_runtime_value(snapshot.state))
        return
    if not isinstance(snapshot.state, Mapping):
        raise TypeError("fallback component snapshot must contain a mapping")
    attributes = vars(component)
    for key in tuple(attributes):
        if key not in snapshot.state:
            del attributes[key]
    for key, value in snapshot.state.items():
        attributes[key] = _clone_runtime_value(value)


__all__ = ["CacheRunner", "CacheRunnerSnapshot"]
