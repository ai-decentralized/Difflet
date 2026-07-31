"""Versioned cache schedule artifacts: masks, plans, and strict resolution.

Two JSON formats exist on purpose:

* ``difflet-cache-mask-v1`` is a bare per-step anchor list. It carries no
  model identity or predictor configuration and is the step-precise entry
  point for SCM / external-schedule ablations.
* ``difflet-cache-plan-v1`` is a full experiment identity: compatibility
  (model, shape, steps, scheduler), the policy parameters, the predictor
  parameters, and a frozen per-step mask. Parameters explain the schedule;
  the frozen mask pins the exact behaviour of the calibrated run. Both must
  agree or resolution fails.

Parsing is fail closed: unknown fields, missing fields, or a wrong schema
string raise instead of being silently ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from difflet.pipeline.cache.policies import ExplicitMaskPolicy, PeriodicAnchorPolicy
from difflet.pipeline.cache.predictors import LegacyResidualPredictor, TaylorSeerPredictor
from difflet.pipeline.cache.recovery import (
    QualityRecoveryConfig,
    QualityRecoveryGuard,
)
from difflet.pipeline.cache.runner import CacheRunner
from difflet.pipeline.cache.types import CachePolicy, CachePredictor

CACHE_MASK_SCHEMA = "difflet-cache-mask-v1"
CACHE_PLAN_SCHEMA = "difflet-cache-plan-v1"

_POLICY_TYPES = ("periodic_anchor",)
_PREDICTOR_TYPES = ("legacy_residual", "taylorseer")


class CacheSpecError(ValueError):
    """Raised when a cache mask/plan is malformed, unsafe, or incompatible."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CacheSpecError(f"{name} must be a JSON object")
    return value


def _check_keys(
    data: Mapping[str, Any],
    name: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    keys = set(data)
    if any(not isinstance(key, str) for key in keys):
        raise CacheSpecError(f"{name} field names must be strings")
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise CacheSpecError(f"{name} is missing required fields: {sorted(missing)}")
    if unknown:
        raise CacheSpecError(f"{name} has unknown fields: {sorted(unknown)}")


def _strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CacheSpecError(f"{name} must be a positive integer")
    return value


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CacheSpecError(f"{name} must be a nonnegative integer")
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise CacheSpecError(f"{name} must be a JSON boolean")
    return value


def _strict_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CacheSpecError(f"{name} must be a non-empty string")
    return value


def _strict_mask(value: Any, name: str, *, num_steps: int) -> tuple[bool, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CacheSpecError(f"{name} must be a list of booleans")
    if any(type(item) is not bool for item in value):
        raise CacheSpecError(f"{name} entries must be JSON booleans")
    mask = tuple(value)
    if len(mask) != num_steps:
        raise CacheSpecError(
            f"{name} has {len(mask)} entries, but num_steps is {num_steps}"
        )
    return mask


def _load_json(source: str | Path | Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as error:
        raise CacheSpecError(f"{name} file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise CacheSpecError(f"{name} file is not valid JSON: {path}: {error}") from error
    except UnicodeError as error:
        raise CacheSpecError(f"{name} file is not valid UTF-8: {path}: {error}") from error
    except OSError as error:
        raise CacheSpecError(f"{name} file could not be read: {path}: {error}") from error
    return _require_mapping(data, name)


def _longest_false_run(mask: Sequence[bool]) -> int:
    longest = current = 0
    for anchor in mask:
        if anchor:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


@dataclass(frozen=True)
class CacheCompatibility:
    """The experiment identity a plan is valid for. All fields must match."""

    model: str
    shape_label: str
    num_steps: int
    scheduler_class: str

    def __post_init__(self) -> None:
        _strict_str(self.model, "compatibility.model")
        _strict_str(self.shape_label, "compatibility.shape_label")
        _strict_positive_int(self.num_steps, "compatibility.num_steps")
        _strict_str(self.scheduler_class, "compatibility.scheduler_class")

    def validate_runtime(
        self,
        *,
        model: str,
        shape_label: str,
        num_steps: int,
        scheduler_class: str,
    ) -> None:
        expected = (self.model, self.shape_label, self.num_steps, self.scheduler_class)
        actual = (
            _strict_str(model, "runtime.model"),
            _strict_str(shape_label, "runtime.shape_label"),
            _strict_positive_int(num_steps, "runtime.num_steps"),
            _strict_str(scheduler_class, "runtime.scheduler_class"),
        )
        if expected != actual:
            raise CacheSpecError(
                "cache plan compatibility does not match the runtime: "
                f"plan is for model={self.model!r} shape={self.shape_label!r} "
                f"steps={self.num_steps} scheduler={self.scheduler_class!r}, "
                f"runtime is model={actual[0]!r} shape={actual[1]!r} "
                f"steps={actual[2]} scheduler={actual[3]!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "shape_label": self.shape_label,
            "num_steps": int(self.num_steps),
            "scheduler_class": self.scheduler_class,
        }


@dataclass(frozen=True)
class CacheMask:
    """A bare per-step anchor mask with no model identity attached."""

    num_steps: int
    anchor_mask: tuple[bool, ...]
    description: str | None = None

    def __post_init__(self) -> None:
        steps = _strict_positive_int(self.num_steps, "num_steps")
        object.__setattr__(
            self, "anchor_mask", _strict_mask(self.anchor_mask, "anchor_mask", num_steps=steps)
        )
        if self.description is not None:
            _strict_str(self.description, "description")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema": CACHE_MASK_SCHEMA,
            "num_steps": int(self.num_steps),
            "anchor_mask": list(self.anchor_mask),
        }
        if self.description is not None:
            data["description"] = self.description
        return data


@dataclass(frozen=True)
class CachePlan:
    """A reproducible cache experiment: identity + policy + predictor + mask."""

    compatibility: CacheCompatibility
    policy: Mapping[str, Any]
    predictor: Mapping[str, Any]
    frozen_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.compatibility, CacheCompatibility):
            raise CacheSpecError(
                "compatibility must be a CacheCompatibility instance"
            )
        policy = dict(_require_mapping(self.policy, "policy"))
        predictor = dict(_require_mapping(self.predictor, "predictor"))
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "predictor", predictor)
        object.__setattr__(
            self,
            "frozen_mask",
            _strict_mask(self.frozen_mask, "frozen_mask", num_steps=self.compatibility.num_steps),
        )
        # Validate eagerly so a malformed plan cannot exist in memory.
        build_policy(policy)
        build_predictor(predictor)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema": CACHE_PLAN_SCHEMA,
            "compatibility": self.compatibility.to_dict(),
            "policy": dict(self.policy),
            "predictor": dict(self.predictor),
            "frozen_mask": list(self.frozen_mask),
        }
        return data

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ResolvedCacheConfig:
    """A validated, executable policy/predictor pair plus its anchor mask."""

    policy: CachePolicy
    predictor: CachePredictor
    recovery: QualityRecoveryConfig
    anchor_mask: tuple[bool, ...]
    num_steps: int
    source: str
    plan: CachePlan | None = None
    barrier_steps: tuple[int, ...] = ()

    @property
    def planned_anchor_steps(self) -> int:
        return sum(1 for anchor in self.anchor_mask if anchor)

    @property
    def planned_skip_steps(self) -> int:
        return self.num_steps - self.planned_anchor_steps

    def build_runner(self, *, history_capacity: int | None = None) -> CacheRunner:
        return CacheRunner(
            self.policy,
            self.predictor,
            recovery=QualityRecoveryGuard(self.recovery),
            history_capacity=history_capacity,
        )


def build_policy(spec: Mapping[str, Any]) -> PeriodicAnchorPolicy:
    """Build a schedule policy from a plan's ``policy`` object (fail closed)."""

    spec = _require_mapping(spec, "policy")
    kind = _strict_str(spec.get("type", ""), "policy.type")
    if kind not in _POLICY_TYPES:
        raise CacheSpecError(
            f"policy.type {kind!r} is not supported by {CACHE_PLAN_SCHEMA}; "
            f"supported types: {list(_POLICY_TYPES)}"
        )
    _check_keys(
        spec,
        "policy",
        required={
            "type",
            "anchor_interval",
            "anchor_phase",
            "warmup_steps",
            "cooldown_steps",
            "require_final_anchor",
        },
    )
    try:
        return PeriodicAnchorPolicy(
            anchor_interval=_strict_positive_int(spec["anchor_interval"], "policy.anchor_interval"),
            anchor_phase=_strict_nonnegative_int(spec["anchor_phase"], "policy.anchor_phase"),
            warmup_steps=_strict_nonnegative_int(spec["warmup_steps"], "policy.warmup_steps"),
            cooldown_steps=_strict_nonnegative_int(
                spec["cooldown_steps"], "policy.cooldown_steps"
            ),
            require_final_anchor=_strict_bool(
                spec["require_final_anchor"], "policy.require_final_anchor"
            ),
        )
    except ValueError as error:
        raise CacheSpecError(f"policy is invalid: {error}") from error


def build_predictor(spec: Mapping[str, Any]) -> CachePredictor:
    """Build a predictor from a plan's ``predictor`` object (fail closed)."""

    spec = _require_mapping(spec, "predictor")
    kind = _strict_str(spec.get("type", ""), "predictor.type")
    if kind not in _PREDICTOR_TYPES:
        raise CacheSpecError(
            f"predictor.type {kind!r} is not supported; "
            f"supported types: {list(_PREDICTOR_TYPES)}"
        )
    try:
        if kind == "legacy_residual":
            _check_keys(spec, "predictor", required={"type", "coord"})
            return LegacyResidualPredictor(coord=_strict_str(spec["coord"], "predictor.coord"))
        _check_keys(spec, "predictor", required={"type", "order", "coord"})
        order = _strict_positive_int(spec["order"], "predictor.order")
        if order not in (1, 2):
            raise CacheSpecError("predictor.order must be 1 or 2")
        return TaylorSeerPredictor(
            order=order,
            coord=_strict_str(spec["coord"], "predictor.coord"),
        )
    except ValueError as error:
        raise CacheSpecError(f"predictor is invalid: {error}") from error


def load_cache_mask(source: str | Path | Mapping[str, Any]) -> CacheMask:
    """Parse a ``difflet-cache-mask-v1`` document. Unknown fields are errors."""

    data = _load_json(source, "cache mask")
    _check_keys(
        data,
        "cache mask",
        required={"schema", "num_steps", "anchor_mask"},
        optional={"description"},
    )
    schema = _strict_str(data["schema"], "schema")
    if schema != CACHE_MASK_SCHEMA:
        raise CacheSpecError(
            f"cache mask schema must be {CACHE_MASK_SCHEMA!r}, got {schema!r}"
        )
    return CacheMask(
        num_steps=_strict_positive_int(data["num_steps"], "num_steps"),
        anchor_mask=data["anchor_mask"],
        description=data.get("description"),
    )


def load_cache_plan(source: str | Path | Mapping[str, Any]) -> CachePlan:
    """Parse a ``difflet-cache-plan-v1`` document. Unknown fields are errors."""

    data = _load_json(source, "cache plan")
    _check_keys(
        data,
        "cache plan",
        required={"schema", "compatibility", "policy", "predictor", "frozen_mask"},
    )
    schema = _strict_str(data["schema"], "schema")
    if schema != CACHE_PLAN_SCHEMA:
        raise CacheSpecError(
            f"cache plan schema must be {CACHE_PLAN_SCHEMA!r}, got {schema!r}"
        )
    compat = _require_mapping(data["compatibility"], "compatibility")
    _check_keys(
        compat,
        "compatibility",
        required={"model", "shape_label", "num_steps", "scheduler_class"},
    )
    compatibility = CacheCompatibility(
        model=_strict_str(compat["model"], "compatibility.model"),
        shape_label=_strict_str(compat["shape_label"], "compatibility.shape_label"),
        num_steps=_strict_positive_int(compat["num_steps"], "compatibility.num_steps"),
        scheduler_class=_strict_str(compat["scheduler_class"], "compatibility.scheduler_class"),
    )
    frozen = data["frozen_mask"]
    if isinstance(frozen, (str, bytes)) or not isinstance(frozen, Sequence):
        raise CacheSpecError("frozen_mask must be a list of booleans")
    return CachePlan(
        compatibility=compatibility,
        policy=_require_mapping(data["policy"], "policy"),
        predictor=_require_mapping(data["predictor"], "predictor"),
        frozen_mask=tuple(frozen),
    )


def validate_schedule_safety(
    mask: Sequence[bool],
    predictor: CachePredictor,
    *,
    require_final_anchor: bool = False,
    barrier_steps: Sequence[int] = (),
) -> None:
    """Reject schedules the predictor cannot execute soundly.

    * Enough real anchors must exist before the first planned prediction
      (``required_history``, i.e. ``order + 1`` for TaylorSeer, 2 for legacy).
    * The longest planned consecutive prediction run must not exceed
      ``max_consecutive_predictions`` (1 for the legacy residual predictor).
    * With ``require_final_anchor`` the last denoise step must be an anchor.
    """

    if isinstance(mask, (str, bytes)) or not isinstance(mask, Sequence):
        raise CacheSpecError("anchor mask must be a list of booleans")
    mask = _strict_mask(mask, "anchor mask", num_steps=len(mask))
    if not mask:
        raise CacheSpecError("anchor mask must not be empty")
    if type(require_final_anchor) is not bool:
        raise CacheSpecError("require_final_anchor must be a boolean")
    if not isinstance(predictor, CachePredictor):
        raise CacheSpecError("predictor must implement CachePredictor")
    required = int(predictor.required_history)
    normalized_barriers: list[int] = []
    for value in barrier_steps:
        if isinstance(value, bool) or not isinstance(value, int):
            raise CacheSpecError("barrier_steps must contain integers")
        if value < 0 or value >= len(mask):
            raise CacheSpecError(
                f"barrier step {value} is outside a {len(mask)}-step trajectory"
            )
        if not mask[value]:
            raise CacheSpecError(f"barrier step {value} must be a real anchor")
        normalized_barriers.append(value)
    if len(set(normalized_barriers)) != len(normalized_barriers):
        raise CacheSpecError("barrier_steps must not contain duplicates")

    segment_starts = [0]
    segment_starts.extend(index for index in sorted(normalized_barriers) if index > 0)
    segment_ends = [*segment_starts[1:], len(mask)]
    for segment_start, segment_end in zip(segment_starts, segment_ends):
        segment = mask[segment_start:segment_end]
        try:
            relative_first_skip = segment.index(False)
        except ValueError:
            continue
        first_skip = segment_start + relative_first_skip
        anchors_before = sum(
            1 for anchor in mask[segment_start:first_skip] if anchor
        )
        if anchors_before < required:
            boundary = (
                "trajectory start"
                if segment_start == 0
                else f"barrier step {segment_start}"
            )
            raise CacheSpecError(
                f"the first planned prediction after {boundary} is at step "
                f"{first_skip}, but only {anchors_before} real anchors precede "
                f"it; the predictor requires {required}"
            )
    maximum = predictor.max_consecutive_predictions
    if maximum is not None:
        run = _longest_false_run(mask)
        if run > int(maximum):
            raise CacheSpecError(
                f"schedule requests {run} consecutive predictions, but the "
                f"predictor supports at most {int(maximum)}"
            )
    if require_final_anchor and not mask[-1]:
        raise CacheSpecError(
            "require_final_anchor=true, but the final denoise step is not an anchor"
        )


def resolve_cache_plan(
    plan: CachePlan,
    *,
    model: str,
    shape_label: str,
    num_steps: int,
    scheduler_class: str,
) -> ResolvedCacheConfig:
    """Validate a plan against the runtime identity and current code semantics.

    Invariants enforced here (the architecture note, section 07):

    1. fields were already strictly parsed by :func:`load_cache_plan`;
    2. the experiment identity must match the runtime exactly;
    3. enough real anchors must precede the first prediction;
    4. with ``require_final_anchor`` the final step must be an anchor;
    5. the mask regenerated from policy parameters must equal ``frozen_mask``
       bit for bit, catching semantic drift in the policy code;
    6. entry exclusivity is enforced by the callers (application / CLI), which
       reject a plan combined with any other cache configuration source.
    """

    if not isinstance(plan, CachePlan):
        raise CacheSpecError("plan must be a CachePlan loaded by load_cache_plan()")
    plan.compatibility.validate_runtime(
        model=model,
        shape_label=shape_label,
        num_steps=num_steps,
        scheduler_class=scheduler_class,
    )
    policy = build_policy(plan.policy)
    predictor = build_predictor(plan.predictor)
    recovery = QualityRecoveryConfig(
        warmup_steps=policy.warmup_steps,
        cooldown_steps=policy.cooldown_steps,
        require_final_anchor=policy.require_final_anchor,
    )
    try:
        recovery.validate_num_steps(plan.compatibility.num_steps)
    except ValueError as error:
        raise CacheSpecError(
            f"quality recovery configuration is invalid: {error}"
        ) from error
    regenerated = recovery.apply_to_anchor_mask(
        policy.materialize_anchor_mask(plan.compatibility.num_steps)
    )
    if regenerated != plan.frozen_mask:
        drift = [
            index
            for index, (new, frozen) in enumerate(zip(regenerated, plan.frozen_mask))
            if new != frozen
        ]
        raise CacheSpecError(
            "policy parameters no longer regenerate the frozen mask; the policy "
            f"semantics drifted at steps {drift[:8]}{'...' if len(drift) > 8 else ''}. "
            "Recalibrate or pin the older code."
        )
    validate_schedule_safety(
        plan.frozen_mask,
        predictor,
        require_final_anchor=policy.require_final_anchor,
    )
    return ResolvedCacheConfig(
        policy=policy,
        predictor=predictor,
        recovery=recovery,
        anchor_mask=plan.frozen_mask,
        num_steps=plan.compatibility.num_steps,
        source="plan",
        plan=plan,
    )


def resolve_cache_config(
    *,
    num_steps: int,
    policy: CachePolicy | None = None,
    mask: CacheMask | Sequence[bool] | None = None,
    predictor: CachePredictor | Mapping[str, Any] | None = None,
    recovery: QualityRecoveryConfig | None = None,
    require_final_anchor: bool = False,
    barrier_steps: Sequence[int] = (),
) -> ResolvedCacheConfig:
    """Resolve a policy-or-mask plus a predictor into an executable config.

    Exactly one of ``policy`` and ``mask`` must be given (entry exclusivity).
    ``predictor`` accepts either a built predictor or a plan-style spec object.
    """

    steps = _strict_positive_int(num_steps, "num_steps")
    if (policy is None) == (mask is None):
        raise CacheSpecError("exactly one of policy and mask must be provided")
    if predictor is None:
        raise CacheSpecError("a predictor (instance or spec object) is required")
    if isinstance(predictor, Mapping):
        predictor = build_predictor(predictor)
    if not isinstance(predictor, CachePredictor):
        raise CacheSpecError("predictor must implement CachePredictor")
    if policy is not None and not isinstance(policy, CachePolicy):
        raise CacheSpecError("policy must implement CachePolicy")
    if type(require_final_anchor) is not bool:
        raise CacheSpecError("require_final_anchor must be a boolean")

    if mask is not None:
        if isinstance(mask, CacheMask):
            if mask.num_steps != steps:
                raise CacheSpecError(
                    f"cache mask is for {mask.num_steps} steps, but runtime has {steps}"
                )
            anchor_mask = mask.anchor_mask
        else:
            anchor_mask = _strict_mask(mask, "anchor_mask", num_steps=steps)
        source = "mask"
        try:
            resolved_policy: CachePolicy = ExplicitMaskPolicy(anchor_mask)
        except ValueError as error:
            raise CacheSpecError(f"anchor mask is invalid: {error}") from error
    else:
        assert policy is not None
        materialize = getattr(policy, "materialize_anchor_mask", None)
        if not callable(materialize):
            raise CacheSpecError(
                "policy cannot be materialized to a finite mask; dynamic policies "
                "are resolved at runtime, not through resolve_cache_config()"
            )
        anchor_mask = tuple(materialize(steps))
        resolved_policy = policy
        source = "params"
        require_final_anchor = bool(
            getattr(policy, "require_final_anchor", require_final_anchor)
        )

    if recovery is None:
        recovery = QualityRecoveryConfig(
            warmup_steps=int(getattr(resolved_policy, "warmup_steps", 0)),
            cooldown_steps=int(getattr(resolved_policy, "cooldown_steps", 0)),
            require_final_anchor=require_final_anchor,
        )
    if not isinstance(recovery, QualityRecoveryConfig):
        raise CacheSpecError("recovery must be a QualityRecoveryConfig")
    try:
        recovery.validate_num_steps(steps)
        anchor_mask = recovery.apply_to_anchor_mask(tuple(anchor_mask))
    except ValueError as error:
        raise CacheSpecError(
            f"quality recovery configuration is invalid: {error}"
        ) from error
    validate_schedule_safety(
        anchor_mask,
        predictor,
        require_final_anchor=recovery.require_final_anchor,
        barrier_steps=barrier_steps,
    )
    return ResolvedCacheConfig(
        policy=resolved_policy,
        predictor=predictor,
        recovery=recovery,
        anchor_mask=anchor_mask,
        num_steps=steps,
        source=source,
        plan=None,
        barrier_steps=tuple(barrier_steps),
    )


__all__ = [
    "CACHE_MASK_SCHEMA",
    "CACHE_PLAN_SCHEMA",
    "CacheCompatibility",
    "CacheMask",
    "CachePlan",
    "CacheSpecError",
    "ResolvedCacheConfig",
    "build_policy",
    "build_predictor",
    "load_cache_mask",
    "load_cache_plan",
    "resolve_cache_config",
    "resolve_cache_plan",
    "validate_schedule_safety",
]
