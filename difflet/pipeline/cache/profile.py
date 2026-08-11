"""Qualified, immutable cache profiles for request-time execution."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from difflet.pipeline.cache.policies import (
    PhasedStaticPolicy,
    StaticPlusBrakeConfig,
    StaticPlusBrakePolicy,
)
from difflet.pipeline.cache.predictors import CalibratedLinearPredictor, TaylorSeerPredictor
from difflet.pipeline.cache.recovery import QualityRecoveryConfig, QualityRecoveryGuard
from difflet.pipeline.cache.runner import CacheRunner
from difflet.pipeline.cache.session import CacheSession

PHASED_CANDIDATE_SCHEMA = "difflet-flux-cache-phased-candidate"
PHASED_CANDIDATE_SCHEMA_REVISION = 1
PROFILE_QUALIFICATION_SCHEMA = "difflet-flux-cache-profile-qualification"
PROFILE_QUALIFICATION_SCHEMA_REVISION = 1
QUALITY_CONTRACT_SCHEMA = "difflet-flux-cache-multires-quality-contract"
QUALITY_CONTRACT_SCHEMA_REVISION = 1
ROOT = Path(__file__).resolve().parents[3]


class CacheProfileError(ValueError):
    """Raised when a profile is malformed, unqualified, or runtime-incompatible."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scheduler_config_sha256(scheduler: Any) -> str:
    """Hash a Diffusers scheduler config using the qualification normalization."""

    config = getattr(scheduler, "config", None)
    if config is None:
        raise CacheProfileError("runtime scheduler does not expose a reproducible config")
    try:
        normalized = json.loads(json.dumps(dict(config), default=str, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise CacheProfileError("runtime scheduler config is not JSON-normalizable") from error
    default_values = normalized.get("_use_default_values")
    if default_values is not None:
        if (
            not isinstance(default_values, list)
            or any(not isinstance(value, str) for value in default_values)
            or len(default_values) != len(set(default_values))
        ):
            raise CacheProfileError("scheduler _use_default_values is invalid")
        normalized["_use_default_values"] = sorted(default_values)
    return canonical_sha256(normalized)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CacheProfileError(f"{name} file does not exist: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CacheProfileError(f"could not read {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CacheProfileError(f"{name} must contain a JSON object")
    return value


def _strict_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CacheProfileError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CacheProfileError(f"{name} must be a positive integer")
    return value


def _strict_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CacheProfileError(f"{name} must be a non-empty trimmed string")
    return value


def _validate_content_hash(document: Mapping[str, Any], name: str) -> str:
    digest = _strict_digest(document.get("sha256"), f"{name}.sha256")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if canonical_sha256(payload) != digest:
        raise CacheProfileError(f"{name} sha256 does not match its contents")
    return digest


def _resolve_reference(candidate_path: Path, value: Any, name: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise CacheProfileError(f"{name} must contain exactly path and sha256")
    path_value = _strict_string(value["path"], f"{name}.path")
    digest = _strict_digest(value["sha256"], f"{name}.sha256")
    unresolved = Path(path_value).expanduser()
    candidates = (
        (unresolved,)
        if unresolved.is_absolute()
        else (candidate_path.parent / unresolved, ROOT / unresolved)
    )
    path = next((item.resolve() for item in candidates if item.is_file()), None)
    if path is None:
        raise CacheProfileError(f"{name} file does not exist relative to {candidate_path}")
    if sha256_file(path) != digest:
        raise CacheProfileError(f"{name} sha256 does not match {path}")
    return path, digest


def _validate_policy(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CacheProfileError("cache profile policy must be a JSON object")
    common = {
        "type",
        "num_steps",
        "static_anchor_steps",
        "warmup_steps",
        "cooldown_steps",
        "require_final_anchor",
        "dynamic_budget",
        "invalid_measurement_fail_closed",
    }
    brake = {
        "plastic_window",
        "tighten_error",
        "recovery_error",
        "recovery_steps",
        "disable_after_recoveries",
        "tighten_rule",
        "allow_acceleration",
    }
    kind = value.get("type")
    if kind not in {"phased_static", "phased_static_plus_brake"}:
        raise CacheProfileError("cache profile policy type is unsupported")
    if set(value) != (common if kind == "phased_static" else common | brake):
        raise CacheProfileError("cache profile policy fields do not match the schema")
    policy = dict(value)
    num_steps = _strict_positive_int(policy["num_steps"], "policy.num_steps")
    warmup = _strict_positive_int(policy["warmup_steps"], "policy.warmup_steps")
    cooldown = policy["cooldown_steps"]
    if isinstance(cooldown, bool) or not isinstance(cooldown, int) or cooldown < 0:
        raise CacheProfileError("policy.cooldown_steps must be a nonnegative integer")
    if warmup + cooldown >= num_steps:
        raise CacheProfileError("policy warmup and cooldown leave no cacheable step")
    if policy["require_final_anchor"] is not True:
        raise CacheProfileError("qualified profiles must require a final anchor")
    if policy["invalid_measurement_fail_closed"] is not True:
        raise CacheProfileError("qualified profiles must fail closed on invalid measurements")
    anchors = policy["static_anchor_steps"]
    if (
        not isinstance(anchors, list)
        or not anchors
        or anchors != sorted(set(anchors))
        or any(isinstance(step, bool) or not isinstance(step, int) for step in anchors)
        or any(not 0 <= step < num_steps for step in anchors)
    ):
        raise CacheProfileError(
            "policy.static_anchor_steps must be valid and strictly increasing"
        )
    required = set(range(warmup))
    required.update(range(num_steps - cooldown, num_steps))
    required.add(num_steps - 1)
    if not required <= set(anchors):
        raise CacheProfileError("static anchors do not cover warmup, cooldown, and final step")
    dynamic_budget = policy["dynamic_budget"]
    if isinstance(dynamic_budget, bool) or not isinstance(dynamic_budget, int):
        raise CacheProfileError("policy.dynamic_budget must be an integer")
    anchor_mask = tuple(index in set(anchors) for index in range(num_steps))
    if kind == "phased_static":
        if dynamic_budget != 0:
            raise CacheProfileError("phased_static dynamic_budget must be zero")
    else:
        window = policy["plastic_window"]
        if (
            dynamic_budget <= 0
            or not isinstance(window, list)
            or len(window) != 2
            or any(isinstance(step, bool) or not isinstance(step, int) for step in window)
        ):
            raise CacheProfileError(
                "bounded-brake budget or policy.plastic_window is invalid"
            )
        if window[0] < warmup or window[1] >= num_steps - cooldown:
            raise CacheProfileError("policy.plastic_window must exclude warmup and cooldown")
        StaticPlusBrakeConfig(
            anchor_mask=anchor_mask,
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
    policy["static_anchor_steps"] = tuple(anchors)
    if "plastic_window" in policy:
        policy["plastic_window"] = tuple(policy["plastic_window"])
    return MappingProxyType(policy)


def _validate_predictor(value: Any, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the predictor block and return normalized construction fields."""

    if not isinstance(value, Mapping):
        raise CacheProfileError("cache profile predictor must be a JSON object")
    kind = value.get("type")
    if kind == "taylorseer":
        if set(value) != {"type", "order", "coord"}:
            raise CacheProfileError("cache profile predictor fields do not match the schema")
        TaylorSeerPredictor(order=value["order"], coord=value["coord"])
        return {"order": value["order"], "coord": value["coord"], "weights": None}
    if kind != "calibrated_linear":
        raise CacheProfileError("cache profile predictor type is unsupported")
    if set(value) != {"type", "coord", "weights"} or value["coord"] != "index":
        raise CacheProfileError("calibrated predictor fields do not match the schema")
    if policy["type"] != "phased_static":
        raise CacheProfileError("calibrated predictors require a phased_static policy")
    weights_doc = value["weights"]
    if not isinstance(weights_doc, Mapping):
        raise CacheProfileError("calibrated predictor weights must be an object")
    anchors = set(policy["static_anchor_steps"])
    skipped = {step for step in range(policy["num_steps"]) if step not in anchors}
    try:
        table = {
            int(step): tuple((anchor, weight) for anchor, weight in entry)
            for step, entry in weights_doc.items()
        }
    except (TypeError, ValueError) as error:
        raise CacheProfileError("calibrated predictor weights are malformed") from error
    if set(table) != skipped or any(
        anchor not in anchors for entry in table.values() for anchor, _ in entry
    ):
        raise CacheProfileError(
            "calibrated weights must cover exactly the skipped steps using static anchors"
        )
    predictor = CalibratedLinearPredictor(weights=table)
    return {"order": 1, "coord": "index", "weights": predictor.weights}


@dataclass(frozen=True)
class PhasedCandidateArm:
    """A strict static or static-plus-brake runtime profile."""

    source_path: Path
    candidate_id: str
    policy: Mapping[str, Any]
    order: int
    coord: str
    horizon_ref: Mapping[str, str]
    quality_contract_ref: Mapping[str, str]
    content_sha256: str
    file_sha256: str
    weights: Mapping[int, tuple[tuple[int, float], ...]] | None = None

    def policy_spec(self) -> dict[str, Any]:
        value = dict(self.policy)
        value["static_anchor_steps"] = list(value["static_anchor_steps"])
        if "plastic_window" in value:
            value["plastic_window"] = list(value["plastic_window"])
        return value

    def predictor_spec(self) -> dict[str, Any]:
        if self.weights is None:
            return {"type": "taylorseer", "order": self.order, "coord": self.coord}
        return {
            "type": "calibrated_linear",
            "coord": self.coord,
            "weights": {
                str(step): [[anchor, weight] for anchor, weight in entry]
                for step, entry in self.weights.items()
            },
        }

    def build_session(self, num_steps: int) -> CacheSession:
        expected_steps = int(self.policy["num_steps"])
        if num_steps != expected_steps:
            raise CacheProfileError(
                f"cache profile requires {expected_steps} steps, got {num_steps}"
            )
        anchors = set(self.policy["static_anchor_steps"])
        mask = tuple(index in anchors for index in range(num_steps))
        kind = self.policy["type"]
        if kind == "phased_static":
            policy = PhasedStaticPolicy(
                mask,
                invalid_measurement_fail_closed=self.policy[
                    "invalid_measurement_fail_closed"
                ],
            )
        else:
            window_start, window_end = self.policy["plastic_window"]
            policy = StaticPlusBrakePolicy(
                StaticPlusBrakeConfig(
                    anchor_mask=mask,
                    plastic_window_start=window_start,
                    plastic_window_end=window_end,
                    dynamic_budget=self.policy["dynamic_budget"],
                    tighten_error=self.policy["tighten_error"],
                    recovery_error=self.policy["recovery_error"],
                    recovery_steps=self.policy["recovery_steps"],
                    disable_after_recoveries=self.policy["disable_after_recoveries"],
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
        predictor = (
            TaylorSeerPredictor(order=self.order, coord=self.coord)
            if self.weights is None
            else CalibratedLinearPredictor(weights=self.weights)
        )
        return CacheSession(
            CacheRunner(
                policy,
                predictor,
                recovery=recovery,
            ),
            num_steps=num_steps,
            configuration_source=kind.replace("_", "-"),
            planned_anchor_steps=(sum(mask) if kind == "phased_static" else None),
            planned_estimate_steps=(num_steps - sum(mask) if kind == "phased_static" else None),
        )

    def build_pipeline_adapter(self, num_steps: int):
        from difflet.pipeline.cache.teacache_adapter import TeaCacheControllerAdapter

        return TeaCacheControllerAdapter(
            self.build_session(num_steps)
        )


def load_phased_candidate(path: str | Path) -> PhasedCandidateArm:
    path = Path(path).expanduser().resolve()
    document = _load_json(path, "cache profile")
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
    if set(document) != expected:
        raise CacheProfileError("cache profile fields do not match the schema")
    if (
        document["schema"] != PHASED_CANDIDATE_SCHEMA
        or document["schema_revision"] != PHASED_CANDIDATE_SCHEMA_REVISION
    ):
        raise CacheProfileError("cache profile schema is unsupported")
    content_sha256 = _validate_content_hash(document, "cache profile")
    candidate_id = _strict_string(document["candidate_id"], "candidate_id")
    policy = _validate_policy(document["policy"])
    predictor = _validate_predictor(document["predictor"], policy)
    horizon_path, horizon_digest = _resolve_reference(
        path, document["horizon_ref"], "horizon_ref"
    )
    contract_path, contract_digest = _resolve_reference(
        path, document["quality_contract_ref"], "quality_contract_ref"
    )
    return PhasedCandidateArm(
        source_path=path,
        candidate_id=candidate_id,
        policy=policy,
        order=predictor["order"],
        coord=predictor["coord"],
        weights=predictor["weights"],
        horizon_ref=MappingProxyType(
            {"path": str(horizon_path), "sha256": horizon_digest}
        ),
        quality_contract_ref=MappingProxyType(
            {"path": str(contract_path), "sha256": contract_digest}
        ),
        content_sha256=content_sha256,
        file_sha256=sha256_file(path),
    )


def load_phased_candidates(paths: Any) -> tuple[PhasedCandidateArm, ...]:
    if isinstance(paths, (str, Path)):
        paths = (paths,)
    arms = tuple(load_phased_candidate(path) for path in paths)
    if not arms:
        raise CacheProfileError("cache profile selection is empty")
    identifiers = [arm.candidate_id for arm in arms]
    if len(identifiers) != len(set(identifiers)):
        raise CacheProfileError("cache profile selection has duplicate identifiers")
    if len({arm.coord for arm in arms}) != 1:
        raise CacheProfileError("cache profiles must use one predictor coordinate")
    return arms


@dataclass(frozen=True)
class QualifiedCacheProfile:
    """A profile whose qualification and exact runtime identity were verified."""

    candidate: PhasedCandidateArm
    qualification_path: Path
    qualification_sha256: str
    build_id: str
    measured_speedup: float
    minimum_speedup: float
    generation: Mapping[str, Any]
    resolution: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation", MappingProxyType(dict(self.generation)))
        object.__setattr__(self, "resolution", MappingProxyType(dict(self.resolution)))

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    def validate_runtime(
        self,
        *,
        model_id: str,
        model_revision: str,
        height: int,
        width: int,
        num_steps: int,
        scheduler_class: str,
        scheduler_config_sha256: str,
        dtype: str,
        guidance_scale: float,
        tp_degree: int,
    ) -> None:
        expected = {
            "model_id": self.generation["model_id"],
            "model_revision": self.generation["model_revision"],
            "height": self.resolution["height"],
            "width": self.resolution["width"],
            "num_steps": self.generation["num_steps"],
            "scheduler_class": self.generation["scheduler_class"],
            "scheduler_config_sha256": self.generation["scheduler_config_sha256"],
            "dtype": self.generation["dtype"],
            "guidance_scale": float(self.generation["guidance_scale"]),
            "tp_degree": self.generation["tp_degree"],
        }
        actual = {
            "model_id": model_id,
            "model_revision": model_revision,
            "height": height,
            "width": width,
            "num_steps": num_steps,
            "scheduler_class": scheduler_class,
            "scheduler_config_sha256": scheduler_config_sha256,
            "dtype": dtype,
            "guidance_scale": float(guidance_scale),
            "tp_degree": tp_degree,
        }
        mismatches = {
            key: {"qualified": expected[key], "runtime": actual[key]}
            for key in expected
            if expected[key] != actual[key]
        }
        if mismatches:
            raise CacheProfileError(
                "qualified cache profile does not match the runtime: "
                + json.dumps(mismatches, sort_keys=True)
            )

    def build_session(self, num_steps: int) -> CacheSession:
        return self.candidate.build_session(num_steps)


def _validate_file_binding(value: Any, name: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise CacheProfileError(f"{name} binding must be an object")
    expected = {"path", "file_sha256", "content_sha256"}
    if set(value) != expected:
        raise CacheProfileError(f"{name} binding fields are invalid")
    path = Path(_strict_string(value["path"], f"{name}.path")).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != _strict_digest(
        value["file_sha256"], f"{name}.file_sha256"
    ):
        raise CacheProfileError(f"{name} file binding is invalid")
    document = _load_json(path, name)
    if document.get("sha256") != _strict_digest(
        value["content_sha256"], f"{name}.content_sha256"
    ):
        raise CacheProfileError(f"{name} content binding is invalid")
    _validate_content_hash(document, name)
    return path, document


def load_qualified_cache_profile(
    profile_path: str | Path,
    qualification_path: str | Path | None = None,
) -> QualifiedCacheProfile:
    """Load a profile only if its frozen qualification and runtime identity agree."""

    candidate = load_phased_candidate(profile_path)
    qualification_path = (
        Path(qualification_path).expanduser().resolve()
        if qualification_path is not None
        else candidate.source_path.with_name("profile-qualification.json")
    )
    qualification = _load_json(qualification_path, "profile qualification")
    expected = {
        "schema",
        "schema_revision",
        "build_id",
        "completed_at",
        "status",
        "build_spec",
        "profile",
        "decision",
        "evidence",
        "sha256",
    }
    if set(qualification) != expected:
        raise CacheProfileError("profile qualification fields do not match the schema")
    if (
        qualification["schema"] != PROFILE_QUALIFICATION_SCHEMA
        or qualification["schema_revision"] != PROFILE_QUALIFICATION_SCHEMA_REVISION
        or qualification["status"] != "qualified"
    ):
        raise CacheProfileError("profile qualification is unsupported or not qualified")
    qualification_sha256 = _validate_content_hash(
        qualification, "profile qualification"
    )
    profile_binding = qualification["profile"]
    if not isinstance(profile_binding, Mapping) or set(profile_binding) != {
        "path",
        "file_sha256",
        "content_sha256",
        "candidate_id",
    }:
        raise CacheProfileError("qualified profile binding fields are invalid")
    if (
        profile_binding["file_sha256"] != candidate.file_sha256
        or profile_binding["content_sha256"] != candidate.content_sha256
        or profile_binding["candidate_id"] != candidate.candidate_id
    ):
        raise CacheProfileError("qualification binds a different cache profile")
    decision = qualification["decision"]
    if not isinstance(decision, Mapping) or set(decision) != {
        "quality_passed",
        "failure_count",
        "measured_speedup",
        "minimum_speedup",
        "speed_passed",
    }:
        raise CacheProfileError("profile qualification decision fields are invalid")
    measured = float(decision["measured_speedup"])
    minimum = float(decision["minimum_speedup"])
    if (
        decision["quality_passed"] is not True
        or decision["speed_passed"] is not True
        or decision["failure_count"] != 0
        or not math.isfinite(measured)
        or not math.isfinite(minimum)
        or measured < minimum
    ):
        raise CacheProfileError("profile qualification decision did not pass")
    _, build_spec = _validate_file_binding(qualification["build_spec"], "build spec")
    quality_contract = build_spec.get("quality_contract")
    if not isinstance(quality_contract, Mapping) or set(quality_contract) != {
        "contract",
        "bucket_id",
    }:
        raise CacheProfileError("build spec quality contract fields are invalid")
    bucket_id = _strict_string(quality_contract["bucket_id"], "quality bucket_id")
    contract_path = Path(candidate.quality_contract_ref["path"])
    contract = _load_json(contract_path, "quality contract")
    if (
        contract.get("schema") != QUALITY_CONTRACT_SCHEMA
        or contract.get("schema_revision") != QUALITY_CONTRACT_SCHEMA_REVISION
    ):
        raise CacheProfileError("quality contract schema is unsupported")
    _validate_content_hash(contract, "quality contract")
    controlled = contract.get("controlled_generation")
    if not isinstance(controlled, Mapping):
        raise CacheProfileError("quality contract controlled_generation is invalid")
    required_generation = {
        "model_id",
        "model_revision",
        "tp_degree",
        "num_steps",
        "guidance_scale",
        "dtype",
        "scheduler_class",
        "scheduler_config_sha256",
    }
    if set(controlled) != required_generation:
        raise CacheProfileError("quality contract generation fields are invalid")
    rows = contract.get("resolution_contracts")
    matches = [
        row
        for row in rows if isinstance(row, Mapping) and row.get("bucket_id") == bucket_id
    ] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise CacheProfileError("quality contract has no unique qualified resolution")
    if int(candidate.policy["num_steps"]) != int(controlled["num_steps"]):
        raise CacheProfileError("profile step count differs from its quality contract")
    return QualifiedCacheProfile(
        candidate=candidate,
        qualification_path=qualification_path,
        qualification_sha256=qualification_sha256,
        build_id=_strict_string(qualification["build_id"], "qualification.build_id"),
        measured_speedup=measured,
        minimum_speedup=minimum,
        generation=controlled,
        resolution=matches[0],
    )


__all__ = [
    "CacheProfileError",
    "PHASED_CANDIDATE_SCHEMA",
    "PHASED_CANDIDATE_SCHEMA_REVISION",
    "PROFILE_QUALIFICATION_SCHEMA",
    "PROFILE_QUALIFICATION_SCHEMA_REVISION",
    "PhasedCandidateArm",
    "QualifiedCacheProfile",
    "canonical_sha256",
    "load_phased_candidate",
    "load_phased_candidates",
    "load_qualified_cache_profile",
    "scheduler_config_sha256",
    "sha256_file",
]
