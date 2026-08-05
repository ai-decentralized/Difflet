"""Strict phase-aware cache candidate schema and runtime adapter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from scripts.flux_cache_protocol import canonical_sha256


PHASED_CANDIDATE_SCHEMA = "difflet-flux-cache-phased-candidate"
PHASED_CANDIDATE_SCHEMA_REVISION = 1
ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class PhasedCandidateArm:
    """One frozen static schedule, optionally combined with a bounded brake."""

    source_path: Path
    candidate_id: str
    policy: Mapping[str, Any]
    order: int
    coord: str
    horizon_ref: Mapping[str, str]
    quality_contract_ref: Mapping[str, str]
    content_sha256: str
    file_sha256: str

    def __post_init__(self) -> None:
        from difflet.pipeline.cache import TaylorSeerPredictor

        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("phased candidate_id must be a non-empty string")
        if self.policy.get("type") not in {"phased_static", "phased_static_plus_brake"}:
            raise ValueError("phased candidate policy type is unsupported")
        TaylorSeerPredictor(order=self.order, coord=self.coord)
        frozen_policy = dict(self.policy)
        frozen_policy["static_anchor_steps"] = tuple(
            frozen_policy["static_anchor_steps"]
        )
        if "plastic_window" in frozen_policy:
            frozen_policy["plastic_window"] = tuple(frozen_policy["plastic_window"])
        object.__setattr__(self, "policy", MappingProxyType(frozen_policy))
        object.__setattr__(self, "horizon_ref", MappingProxyType(dict(self.horizon_ref)))
        object.__setattr__(
            self,
            "quality_contract_ref",
            MappingProxyType(dict(self.quality_contract_ref)),
        )

    def policy_spec(self) -> dict[str, Any]:
        result = dict(self.policy)
        result["static_anchor_steps"] = list(result["static_anchor_steps"])
        if "plastic_window" in result:
            result["plastic_window"] = list(result["plastic_window"])
        return result

    def predictor_spec(self) -> dict[str, Any]:
        return {"type": "taylorseer", "order": self.order, "coord": self.coord}

    def build_pipeline_adapter(self, num_steps: int, *, measurement_sink: Any = None):
        from difflet.pipeline.cache import (
            CacheRunner,
            CacheSession,
            PhasedStaticPolicy,
            QualityRecoveryConfig,
            QualityRecoveryGuard,
            StaticPlusBrakeConfig,
            StaticPlusBrakePolicy,
            TaylorSeerPredictor,
            TeaCacheControllerAdapter,
        )

        expected_steps = int(self.policy["num_steps"])
        if num_steps != expected_steps:
            raise ValueError(
                f"phased candidate requires {expected_steps} steps, got {num_steps}"
            )
        anchor_steps = set(self.policy["static_anchor_steps"])
        anchor_mask = tuple(index in anchor_steps for index in range(num_steps))
        kind = self.policy["type"]
        if kind == "phased_static":
            policy = PhasedStaticPolicy(
                anchor_mask,
                invalid_measurement_fail_closed=self.policy[
                    "invalid_measurement_fail_closed"
                ],
            )
        else:
            window_start, window_end = self.policy["plastic_window"]
            policy = StaticPlusBrakePolicy(
                StaticPlusBrakeConfig(
                    anchor_mask=anchor_mask,
                    plastic_window_start=window_start,
                    plastic_window_end=window_end,
                    dynamic_budget=self.policy["dynamic_budget"],
                    tighten_error=self.policy["tighten_error"],
                    recovery_error=self.policy["recovery_error"],
                    recovery_steps=self.policy["recovery_steps"],
                    disable_after_recoveries=self.policy[
                        "disable_after_recoveries"
                    ],
                    tighten_rule=self.policy["tighten_rule"],
                    allow_acceleration=self.policy["allow_acceleration"],
                    invalid_measurement_fail_closed=self.policy[
                        "invalid_measurement_fail_closed"
                    ],
                )
            )
        recovery = QualityRecoveryGuard(
            QualityRecoveryConfig(
                warmup_steps=self.policy["warmup_steps"],
                cooldown_steps=self.policy["cooldown_steps"],
                require_final_anchor=self.policy["require_final_anchor"],
            )
        )
        runner = CacheRunner(
            policy,
            TaylorSeerPredictor(order=self.order, coord=self.coord),
            recovery=recovery,
            measurement_sink=measurement_sink,
        )
        session = CacheSession(
            runner,
            num_steps=num_steps,
            configuration_source=kind.replace("_", "-"),
            planned_anchor_steps=(sum(anchor_mask) if kind == "phased_static" else None),
            planned_estimate_steps=(
                num_steps - sum(anchor_mask) if kind == "phased_static" else None
            ),
        )
        return TeaCacheControllerAdapter(session)


def _validate_frozen_reference(
    value: Any,
    *,
    name: str,
    candidate_path: Path,
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must contain exactly path and sha256")
    path_value = value["path"]
    digest = value["sha256"]
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{name}.path must be a non-empty string")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name}.sha256 must be a lowercase SHA-256 digest")
    referenced_path = Path(path_value).expanduser()
    if not referenced_path.is_absolute():
        rooted = ROOT / referenced_path
        referenced_path = rooted if rooted.exists() else candidate_path.parent / referenced_path
    referenced_path = referenced_path.resolve()
    if not referenced_path.is_file():
        raise ValueError(f"{name} file does not exist: {referenced_path}")
    if _sha256_file(referenced_path) != digest:
        raise ValueError(f"{name} sha256 does not match {referenced_path}")
    return {"path": path_value, "sha256": digest}


def _validate_phased_policy(policy: Any) -> dict[str, Any]:
    from difflet.pipeline.cache import StaticPlusBrakeConfig

    if not isinstance(policy, dict):
        raise ValueError("phased candidate policy must be a JSON object")
    common_fields = {
        "type",
        "num_steps",
        "static_anchor_steps",
        "warmup_steps",
        "cooldown_steps",
        "require_final_anchor",
        "dynamic_budget",
        "invalid_measurement_fail_closed",
    }
    brake_fields = {
        "plastic_window",
        "tighten_error",
        "recovery_error",
        "recovery_steps",
        "disable_after_recoveries",
        "tighten_rule",
        "allow_acceleration",
    }
    kind = policy.get("type")
    expected_fields = (
        common_fields if kind == "phased_static" else common_fields | brake_fields
    )
    if kind not in {"phased_static", "phased_static_plus_brake"}:
        raise ValueError("phased candidate policy type is unsupported")
    if set(policy) != expected_fields:
        raise ValueError("phased candidate policy fields do not match the protocol")

    num_steps = _strict_positive_int(policy["num_steps"], "policy.num_steps")
    warmup_steps = _strict_positive_int(
        policy["warmup_steps"], "policy.warmup_steps"
    )
    cooldown_steps = policy["cooldown_steps"]
    if (
        isinstance(cooldown_steps, bool)
        or not isinstance(cooldown_steps, int)
        or cooldown_steps < 0
    ):
        raise ValueError("policy.cooldown_steps must be a nonnegative integer")
    if warmup_steps + cooldown_steps >= num_steps:
        raise ValueError("policy warmup and cooldown must leave a cacheable step")
    if type(policy["require_final_anchor"]) is not bool:
        raise ValueError("policy.require_final_anchor must be boolean")
    if policy["require_final_anchor"] is not True:
        raise ValueError("phased candidates must require a final anchor")
    if type(policy["invalid_measurement_fail_closed"]) is not bool:
        raise ValueError("policy.invalid_measurement_fail_closed must be boolean")
    if policy["invalid_measurement_fail_closed"] is not True:
        raise ValueError("phased candidates must fail closed")

    anchor_steps = policy["static_anchor_steps"]
    if not isinstance(anchor_steps, list) or not anchor_steps:
        raise ValueError("policy.static_anchor_steps must be a non-empty list")
    if any(
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step < num_steps
        for step in anchor_steps
    ):
        raise ValueError("policy.static_anchor_steps contains an invalid step")
    if anchor_steps != sorted(set(anchor_steps)):
        raise ValueError("policy.static_anchor_steps must be strictly increasing")
    required_steps = set(range(warmup_steps))
    required_steps.update(range(num_steps - cooldown_steps, num_steps))
    required_steps.add(num_steps - 1)
    if not required_steps <= set(anchor_steps):
        raise ValueError("static anchors do not cover warmup, cooldown, and final step")

    dynamic_budget = policy["dynamic_budget"]
    if isinstance(dynamic_budget, bool) or not isinstance(dynamic_budget, int):
        raise ValueError("policy.dynamic_budget must be an integer")
    if kind == "phased_static":
        if dynamic_budget != 0:
            raise ValueError("phased_static dynamic_budget must be zero")
        return dict(policy)

    if dynamic_budget <= 0:
        raise ValueError("phased_static_plus_brake dynamic_budget must be positive")
    window = policy["plastic_window"]
    if (
        not isinstance(window, list)
        or len(window) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in window)
    ):
        raise ValueError("policy.plastic_window must contain two integer steps")
    if window[0] < warmup_steps or window[1] >= num_steps - cooldown_steps:
        raise ValueError("policy.plastic_window must exclude warmup and cooldown")
    anchor_set = set(anchor_steps)
    mask = tuple(index in anchor_set for index in range(num_steps))
    StaticPlusBrakeConfig(
        anchor_mask=mask,
        plastic_window_start=window[0],
        plastic_window_end=window[1],
        dynamic_budget=dynamic_budget,
        tighten_error=policy["tighten_error"],
        recovery_error=policy["recovery_error"],
        recovery_steps=policy["recovery_steps"],
        disable_after_recoveries=policy["disable_after_recoveries"],
        tighten_rule=policy["tighten_rule"],
        allow_acceleration=policy["allow_acceleration"],
        invalid_measurement_fail_closed=policy["invalid_measurement_fail_closed"],
    )
    return dict(policy)


def load_phased_candidate(path: Path) -> PhasedCandidateArm:
    """Load a strict phase-aware static or static-plus-brake candidate."""

    path = Path(path).expanduser().resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read phased candidate {path}: {error}") from error
    expected = {
        "schema",
        "schema_revision",
        "candidate_id",
        "policy",
        "predictor",
        "horizon_ref",
        "quality_contract_ref",
        "sha256",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError("phased candidate fields do not match the protocol")
    if (
        document["schema"] != PHASED_CANDIDATE_SCHEMA
        or document["schema_revision"] != PHASED_CANDIDATE_SCHEMA_REVISION
    ):
        raise ValueError("phased candidate schema is unsupported")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if document["sha256"] != canonical_sha256(payload):
        raise ValueError("phased candidate sha256 does not match its contents")
    candidate_id = document["candidate_id"]
    if (
        not isinstance(candidate_id, str)
        or not candidate_id
        or candidate_id != candidate_id.strip()
    ):
        raise ValueError("phased candidate_id must be a non-empty trimmed string")
    policy = _validate_phased_policy(document["policy"])
    predictor = document["predictor"]
    if not isinstance(predictor, dict) or set(predictor) != {"type", "order", "coord"}:
        raise ValueError("phased candidate predictor fields do not match the protocol")
    if predictor["type"] != "taylorseer":
        raise ValueError("phased candidate predictor type is unsupported")
    horizon_ref = _validate_frozen_reference(
        document["horizon_ref"],
        name="horizon_ref",
        candidate_path=path,
    )
    quality_contract_ref = _validate_frozen_reference(
        document["quality_contract_ref"],
        name="quality_contract_ref",
        candidate_path=path,
    )
    return PhasedCandidateArm(
        source_path=path,
        candidate_id=candidate_id,
        policy=policy,
        order=predictor["order"],
        coord=predictor["coord"],
        horizon_ref=horizon_ref,
        quality_contract_ref=quality_contract_ref,
        content_sha256=document["sha256"],
        file_sha256=_sha256_file(path),
    )


def load_phased_candidates(paths: Sequence[str | Path]) -> tuple[PhasedCandidateArm, ...]:
    """Load an exact non-empty candidate set with one predictor coordinate."""

    if isinstance(paths, (str, Path)):
        paths = (paths,)
    arms = tuple(load_phased_candidate(Path(path)) for path in paths)
    if not arms:
        raise ValueError("phased candidate selection is empty")
    candidate_ids = [arm.candidate_id for arm in arms]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("phased candidate selection contains duplicate identifiers")
    if len({arm.coord for arm in arms}) != 1:
        raise ValueError("all phased candidates must use one shared coordinate")
    return arms


__all__ = [
    "PHASED_CANDIDATE_SCHEMA",
    "PHASED_CANDIDATE_SCHEMA_REVISION",
    "PhasedCandidateArm",
    "load_phased_candidate",
    "load_phased_candidates",
]
