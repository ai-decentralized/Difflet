#!/usr/bin/env python3
"""Close the offline FLUX cache-gate development loop from frozen artifacts.

This command is deliberately stricter than ``analyze_flux_cache_online_gate``:
it validates the preregistration and methodology, runs prompt-grouped OOF model
development, selects thresholds only from target-policy OOF scores, evaluates
the frozen stopping rules, and exports a frozen gate only when every
development gate passes.  A frozen gate is still *not* a serving
qualification; it must survive a prospective threshold confirmation and an
interventional quality/speed holdout.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_flux_cache_online_gate import (
    FEATURE_GROUPS,
    IMAGE_REWARD_MAX_HARM,
    VQA_SCORE_MAX_HARM,
    _load_json,
    _sha256_file,
    _write_json,
    analyze,
    threshold_at_full_recall,
)


METHODOLOGY_SCHEMA = "difflet-flux-cache-offline-gate-methodology"
METHODOLOGY_SCHEMA_REVISION = 1
DEVELOPMENT_REGISTRATION_SCHEMA = (
    "difflet-flux-cache-online-signal-training-audit-registration"
)
FROZEN_GATE_SCHEMA = "difflet-flux-cache-online-gate-frozen"
FROZEN_GATE_SCHEMA_REVISION = 1
CLOSURE_SCHEMA = "difflet-flux-cache-offline-gate-closure"
CLOSURE_SCHEMA_REVISION = 1
PROFILE_REPLAY_SCHEMA = "difflet-flux-cache-profile-selection-replay"
PROFILE_REPLAY_SCHEMA_REVISION = 1
STAGE_GATE_STUDY_SCHEMA = "difflet-flux-cache-stage-gate-study-result"


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_probability(value: Any, name: str, *, include_zero: bool = True) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a probability")
    result = float(value)
    lower_ok = result >= 0.0 if include_zero else result > 0.0
    if not math.isfinite(result) or not lower_ok or result > 1.0:
        raise ValueError(f"{name} must be in {'[0, 1]' if include_zero else '(0, 1]'}")
    return result


def load_methodology(path: Path) -> dict[str, Any]:
    document = _load_json(path, "offline gate methodology")
    if (
        document.get("schema") != METHODOLOGY_SCHEMA
        or document.get("schema_revision") != METHODOLOGY_SCHEMA_REVISION
    ):
        raise ValueError("unsupported offline gate methodology schema")

    development = document.get("development")
    if not isinstance(development, dict):
        raise ValueError("methodology.development must be an object")
    _require_int(development.get("minimum_total_failures"), "minimum_total_failures", minimum=1)
    _require_int(development.get("minimum_target_failures"), "minimum_target_failures", minimum=1)
    _require_probability(
        development.get("minimum_pooled_oof_roc_auc"),
        "minimum_pooled_oof_roc_auc",
    )
    _require_probability(
        development.get("minimum_target_oof_roc_auc"),
        "minimum_target_oof_roc_auc",
    )

    brake = document.get("brake_threshold")
    if not isinstance(brake, dict):
        raise ValueError("methodology.brake_threshold must be an object")
    if brake.get("selection") != "minimum_positive_target_oof_score":
        raise ValueError("unsupported brake threshold selection")
    if _require_probability(brake.get("required_failure_recall"), "required_failure_recall") != 1.0:
        raise ValueError("revision 1 requires full failure recall")
    _require_probability(
        brake.get("maximum_passing_false_brake_rate"),
        "maximum_passing_false_brake_rate",
    )

    oil = document.get("oil_threshold")
    if not isinstance(oil, dict):
        raise ValueError("methodology.oil_threshold must be an object")
    if oil.get("selection") != "largest_qualified_target_oof_score":
        raise ValueError("unsupported oil threshold selection")
    _require_probability(oil.get("confidence"), "oil confidence", include_zero=False)
    _require_probability(
        oil.get("maximum_failure_rate_upper_bound"),
        "maximum_failure_rate_upper_bound",
    )
    _require_bool(oil.get("require_zero_observed_failures"), "require_zero_observed_failures")
    if oil.get("independent_unit") != "prompt_index":
        raise ValueError("revision 1 requires prompt_index as the independent oil unit")
    _require_int(
        oil.get("minimum_independent_group_count"),
        "minimum_independent_group_count",
        minimum=1,
    )

    contract = document.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("methodology.contract must be an object")
    if float(contract.get("image_reward_max_harm")) != IMAGE_REWARD_MAX_HARM:
        raise ValueError("methodology ImageReward contract does not match the analyzer")
    if float(contract.get("vqa_score_max_harm")) != VQA_SCORE_MAX_HARM:
        raise ValueError("methodology VQAScore contract does not match the analyzer")
    if not _require_bool(
        contract.get("weighted_metric_average_forbidden"),
        "methodology weighted_metric_average_forbidden",
    ):
        raise ValueError("methodology must forbid metric averaging")

    profile = document.get("profile_selection")
    if not isinstance(profile, dict):
        raise ValueError("methodology.profile_selection must be an object")
    if profile.get("independent_unit") != "prompt_index":
        raise ValueError("profile selection must use prompt_index as its independent unit")
    _require_probability(profile.get("confidence"), "profile confidence", include_zero=False)
    _require_probability(
        profile.get("maximum_failure_rate_upper_bound"),
        "profile maximum_failure_rate_upper_bound",
    )
    _require_int(
        profile.get("minimum_independent_group_count"),
        "profile minimum_independent_group_count",
        minimum=1,
    )

    features = document.get("features")
    if not isinstance(features, dict) or features.get("groups") != {
        name: list(values) for name, values in FEATURE_GROUPS.items()
    }:
        raise ValueError("methodology feature groups do not match the implementation")
    return document


def _check_file_hash(path: Path, expected: str, name: str) -> None:
    if not path.is_file():
        raise ValueError(f"registered {name} does not exist: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"registered {name} sha256 mismatch: expected {expected}, got {actual}")


def validate_development_inputs(
    registration_path: Path,
    methodology_path: Path,
    quality_path: Path,
    semantic_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate frozen identities before any labels are used for model fitting."""

    registration = _load_json(registration_path, "development registration")
    methodology = load_methodology(methodology_path)
    quality = _load_json(quality_path, "quality manifest")
    semantic = _load_json(semantic_path, "semantic report")

    if registration.get("schema") != DEVELOPMENT_REGISTRATION_SCHEMA:
        raise ValueError("unsupported development registration schema")
    if registration.get("schema_revision") != 1:
        raise ValueError("unsupported development registration revision")
    if not _require_bool(
        registration.get("frozen_before_collection"), "frozen_before_collection"
    ):
        raise ValueError("development registration was not frozen before collection")
    role = registration.get("evidence_role")
    if not isinstance(role, dict) or role.get("kind") != "failure-enriched-model-development":
        raise ValueError("registration is not failure-enriched model-development evidence")
    if _require_bool(role.get("serving_claim"), "evidence_role.serving_claim"):
        raise ValueError("development evidence cannot make a serving claim")
    if _require_bool(role.get("confirmation_claim"), "evidence_role.confirmation_claim"):
        raise ValueError("development evidence cannot make a confirmation claim")

    prompt_set = registration.get("prompt_set")
    if not isinstance(prompt_set, dict):
        raise ValueError("registration.prompt_set must be an object")
    prompt_path = _resolve_repo_path(str(prompt_set.get("path")))
    _check_file_hash(prompt_path, str(prompt_set.get("file_sha256")), "prompt set")
    prompt_document = _load_json(prompt_path, "registered prompt set")
    prompts = prompt_document.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != int(prompt_set.get("prompt_count", -1)):
        raise ValueError("registered prompt count mismatch")

    strata = registration.get("candidate_strata")
    if not isinstance(strata, list) or not strata:
        raise ValueError("registration.candidate_strata must be non-empty")
    candidate_ids: list[str] = []
    target_ids: list[str] = []
    for index, stratum in enumerate(strata):
        if not isinstance(stratum, dict):
            raise ValueError("candidate stratum must be an object")
        candidate_path = _resolve_repo_path(str(stratum.get("path")))
        _check_file_hash(
            candidate_path,
            str(stratum.get("file_sha256")),
            f"candidate stratum {index}",
        )
        candidate = _load_json(candidate_path, f"candidate stratum {index}")
        candidate_id = str(stratum.get("candidate_id"))
        if candidate.get("candidate_id") != candidate_id:
            raise ValueError("registered candidate id does not match candidate artifact")
        candidate_ids.append(candidate_id)
        if str(stratum.get("role")) == "target oil policy":
            target_ids.append(candidate_id)
    if len(target_ids) != 1:
        raise ValueError("registration must identify exactly one target oil policy")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("registration contains duplicate candidate ids")

    matrix = registration.get("sample_matrix")
    if not isinstance(matrix, dict):
        raise ValueError("registration.sample_matrix must be an object")
    expected_comparisons = _require_int(
        matrix.get("candidate_comparison_count"),
        "candidate_comparison_count",
        minimum=1,
    )
    comparisons = quality.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != expected_comparisons:
        raise ValueError("quality comparison count does not match registration")
    observed_candidates = {str(row.get("candidate_id")) for row in comparisons}
    if observed_candidates != set(candidate_ids):
        raise ValueError("quality candidate ids do not match registration")
    if int(quality.get("prompt_count", -1)) != len(prompts):
        raise ValueError("quality prompt count does not match registration")
    expected_seeds = matrix.get("seeds")
    if (
        not isinstance(expected_seeds, list)
        or not expected_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in expected_seeds)
        or len(set(expected_seeds)) != len(expected_seeds)
    ):
        raise ValueError("registration.sample_matrix.seeds must contain unique integers")
    expected_matrix = {
        (candidate_id, prompt_index, seed)
        for candidate_id in candidate_ids
        for prompt_index in range(len(prompts))
        for seed in expected_seeds
    }
    observed_matrix: set[tuple[str, int, int]] = set()
    for comparison in comparisons:
        prompt_index = int(comparison.get("prompt_index", -1))
        if prompt_index < 0 or prompt_index >= len(prompts):
            raise ValueError("quality comparison prompt_index is outside the prompt set")
        if comparison.get("prompt") != prompts[prompt_index]:
            raise ValueError("quality comparison prompt text does not match the prompt set")
        key = (
            str(comparison.get("candidate_id")),
            prompt_index,
            int(comparison.get("seed", -1)),
        )
        if key in observed_matrix:
            raise ValueError("quality comparison matrix contains a duplicate cell")
        observed_matrix.add(key)
    if observed_matrix != expected_matrix:
        raise ValueError("quality comparisons do not form the registered Cartesian matrix")

    registered_features = registration.get("online_features", {}).get("feature_names")
    if registered_features != list(FEATURE_GROUPS["combined"]):
        raise ValueError("registered feature names do not match the implementation")
    contract = registration.get("quality_contract")
    if not isinstance(contract, dict):
        raise ValueError("registration.quality_contract must be an object")
    if float(contract.get("image_reward_max_harm")) != IMAGE_REWARD_MAX_HARM:
        raise ValueError("registered ImageReward contract does not match the analyzer")
    if float(contract.get("vqa_score_max_harm")) != VQA_SCORE_MAX_HARM:
        raise ValueError("registered VQAScore contract does not match the analyzer")
    if not _require_bool(
        contract.get("weighted_metric_average_forbidden"),
        "weighted_metric_average_forbidden",
    ):
        raise ValueError("metric averaging must remain forbidden")

    if not semantic.get("complete"):
        raise ValueError("semantic report is incomplete")
    semantic_sources = semantic.get("sources")
    if not isinstance(semantic_sources, list) or not any(
        str(source.get("sha256")) == _sha256_file(quality_path)
        for source in semantic_sources
        if isinstance(source, dict)
    ):
        raise ValueError("semantic report does not bind to the supplied quality manifest")

    source = quality.get("protocol", {}).get("source", {})
    if source.get("git_dirty") is not False:
        raise ValueError("quality collection did not record a clean execution worktree")

    integrity = {
        "registration_sha256": _sha256_file(registration_path),
        "methodology_sha256": _sha256_file(methodology_path),
        "quality_manifest_sha256": _sha256_file(quality_path),
        "semantic_report_sha256": _sha256_file(semantic_path),
        "prompt_set_sha256": _sha256_file(prompt_path),
        "target_candidate_id": target_ids[0],
        "candidate_ids": sorted(candidate_ids),
        "comparison_count": len(comparisons),
        "execution_git_commit": source.get("git_commit"),
        "execution_git_dirty": source.get("git_dirty"),
        # The current discovery registration predates methodology-v1.  A
        # negative/reject replay is valid, but no positive gate may claim the
        # methodology was preregistered unless the registration binds its hash.
        "methodology_preregistered": bool(
            isinstance(registration.get("methodology"), dict)
            and registration["methodology"].get("sha256")
            == _sha256_file(methodology_path)
        ),
    }
    return registration, methodology, integrity


def one_sided_binomial_upper_bound(
    failures: int,
    sample_count: int,
    *,
    confidence: float,
) -> float:
    """Clopper-Pearson one-sided upper confidence bound."""

    failures = _require_int(failures, "failures")
    sample_count = _require_int(sample_count, "sample_count", minimum=1)
    confidence = _require_probability(confidence, "confidence", include_zero=False)
    if failures > sample_count:
        raise ValueError("failures cannot exceed sample_count")
    if failures == sample_count:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(confidence, failures + 1, sample_count - failures))


def replay_profile_selection(
    stage_result: Mapping[str, Any],
    methodology: Mapping[str, Any],
    candidate_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Recompute the legacy stage-gate profile choice without upgrading its claim."""

    if (
        stage_result.get("schema") != STAGE_GATE_STUDY_SCHEMA
        or stage_result.get("schema_revision") != 1
    ):
        raise ValueError("unsupported stage-gate study result")
    profiles = stage_result.get("candidates")
    cumulative = stage_result.get("cumulative_evidence")
    second_audit = stage_result.get("experiments", {}).get("second_fresh_audit")
    if not isinstance(profiles, dict) or not isinstance(cumulative, dict):
        raise ValueError("stage-gate result is missing candidate or cumulative evidence")
    if not isinstance(second_audit, dict):
        raise ValueError("stage-gate result is missing comparable speed evidence")
    if set(candidate_paths) != set(profiles):
        raise ValueError("candidate paths must exactly cover the stage-gate profiles")

    from scripts.collect_flux_cache_ab import load_adaptive_candidate

    policy = methodology["profile_selection"]
    confidence = float(policy["confidence"])
    maximum_upper = float(policy["maximum_failure_rate_upper_bound"])
    minimum_groups = int(policy["minimum_independent_group_count"])
    rows: list[dict[str, Any]] = []
    for profile_name in sorted(profiles):
        recorded = profiles[profile_name]
        evidence = cumulative.get(profile_name)
        speed_evidence = second_audit.get(profile_name)
        if not all(isinstance(value, dict) for value in (recorded, evidence, speed_evidence)):
            raise ValueError(f"stage-gate evidence is incomplete for {profile_name}")
        candidate_path = candidate_paths[profile_name].expanduser().resolve()
        if _sha256_file(candidate_path) != recorded.get("candidate_sha256"):
            raise ValueError(f"candidate file sha256 mismatch for {profile_name}")
        candidate = load_adaptive_candidate(candidate_path)
        if candidate.candidate_id != recorded.get("candidate_id"):
            raise ValueError(f"candidate id mismatch for {profile_name}")
        candidate_policy = candidate.policy_spec()
        for field in (
            "warmup_steps",
            "initial_anchor_interval",
            "maximum_anchor_interval",
            "tighten_error",
            "allow_acceleration",
        ):
            if field in recorded and recorded[field] != candidate_policy[field]:
                raise ValueError(f"candidate policy mismatch for {profile_name}.{field}")

        failures = _require_int(
            evidence.get("automatic_failures"),
            f"{profile_name}.automatic_failures",
        )
        sample_count = _require_int(
            evidence.get("sample_count"),
            f"{profile_name}.sample_count",
            minimum=1,
        )
        upper = one_sided_binomial_upper_bound(
            failures,
            sample_count,
            confidence=confidence,
        )
        recorded_upper = float(
            evidence.get("one_sided_95_percent_failure_rate_upper_bound")
        )
        if not math.isclose(upper, recorded_upper, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"recorded failure-rate bound disagrees for {profile_name}")
        speedup = float(speed_evidence.get("measured_speedup"))
        if not math.isfinite(speedup) or speedup <= 0.0:
            raise ValueError(f"measured speedup is invalid for {profile_name}")
        eligible = bool(sample_count >= minimum_groups and upper <= maximum_upper)
        rows.append(
            {
                "profile_name": profile_name,
                "candidate_id": candidate.candidate_id,
                "candidate_path": str(candidate_path),
                "candidate_file_sha256": _sha256_file(candidate_path),
                "candidate_content_sha256": _load_json(
                    candidate_path, f"{profile_name} candidate"
                )["sha256"],
                "policy": candidate_policy,
                "predictor": candidate.predictor_spec(),
                "independent_group_count": sample_count,
                "failure_count": failures,
                "failure_rate_upper_bound": upper,
                "measured_speedup": speedup,
                "quality_eligible": eligible,
            }
        )
    eligible = [row for row in rows if row["quality_eligible"]]
    selected = (
        min(
            eligible,
            key=lambda row: (-float(row["measured_speedup"]), str(row["candidate_id"])),
        )
        if eligible
        else None
    )
    return {
        "schema": PROFILE_REPLAY_SCHEMA,
        "schema_revision": PROFILE_REPLAY_SCHEMA_REVISION,
        "status": (
            "legacy_profile_selected_not_methodology_preregistered"
            if selected is not None
            else "no_quality_eligible_profile"
        ),
        "selection_rule": (
            "quality upper-bound constraint first, then maximum measured speedup, "
            "then candidate_id"
        ),
        "profiles": rows,
        "selected_profile": selected,
        "methodology_preregistered": False,
        "export_serving_profile": False,
        "serving_claim": False,
        "qualification_note": (
            "This deterministic replay preserves the historical conservative choice. "
            "The evidence predates methodology-v1, so a new registered confirmation "
            "is required before a serving profile can be exported."
        ),
    }


def run_profile_replay(args: argparse.Namespace) -> dict[str, Any]:
    methodology_path = args.methodology.expanduser().resolve()
    methodology = load_methodology(methodology_path)
    stage_path = args.stage_result.expanduser().resolve()
    stage_result = _load_json(stage_path, "stage-gate result")
    candidate_paths: dict[str, Path] = {}
    for binding in args.candidate:
        name, separator, value = binding.partition("=")
        if not separator or not name or not value or name in candidate_paths:
            raise ValueError("--candidate must be a unique PROFILE_NAME=PATH binding")
        candidate_paths[name] = Path(value)
    replay = replay_profile_selection(stage_result, methodology, candidate_paths)
    replay["sources"] = {
        "methodology": {
            "path": str(methodology_path),
            "sha256": _sha256_file(methodology_path),
        },
        "stage_result": {
            "path": str(stage_path),
            "sha256": _sha256_file(stage_path),
        },
    }
    _write_json(args.out.expanduser().resolve(), replay)
    return replay


def select_oil_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    group_ids: Sequence[int],
    *,
    brake_threshold: float,
    confidence: float,
    maximum_failure_rate_upper_bound: float,
    minimum_independent_group_count: int,
    require_zero_observed_failures: bool,
) -> dict[str, Any]:
    """Choose the largest low-risk OOF cutoff meeting the frozen oil rules."""

    if len(labels) != len(scores) or len(labels) != len(group_ids) or not labels:
        raise ValueError("labels, scores, and group_ids must be non-empty and equal length")
    if not any(bool(label) for label in labels):
        raise ValueError("oil threshold selection requires at least one failure")
    candidates = sorted(
        {float(score) for score in scores if float(score) < float(brake_threshold)}
    )
    evaluated: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []
    for threshold in candidates:
        indices = [index for index, score in enumerate(scores) if float(score) <= threshold]
        failures = sum(bool(labels[index]) for index in indices)
        sample_count = len(indices)
        eligible_groups = sorted({int(group_ids[index]) for index in indices})
        failing_groups = sum(
            any(
                bool(labels[index])
                for index in indices
                if int(group_ids[index]) == group_id
            )
            for group_id in eligible_groups
        )
        independent_group_count = len(eligible_groups)
        upper = one_sided_binomial_upper_bound(
            failing_groups,
            independent_group_count,
            confidence=confidence,
        )
        row = {
            "threshold": threshold,
            "sample_count": sample_count,
            "failure_count": failures,
            "failure_rate": failures / sample_count,
            "independent_group_count": independent_group_count,
            "failing_group_count": failing_groups,
            "independent_group_failure_rate": (
                failing_groups / independent_group_count
            ),
            "independent_group_failure_rate_upper_bound": upper,
            "coverage": sample_count / len(labels),
            "group_coverage": independent_group_count / len(set(group_ids)),
        }
        row["qualified"] = bool(
            independent_group_count >= minimum_independent_group_count
            and upper <= maximum_failure_rate_upper_bound
            and (not require_zero_observed_failures or failing_groups == 0)
        )
        evaluated.append(row)
        if row["qualified"]:
            qualified.append(row)
    selected = max(qualified, key=lambda row: float(row["threshold"])) if qualified else None
    best_available = (
        min(
            evaluated,
            key=lambda row: (
                float(row["independent_group_failure_rate_upper_bound"]),
                -int(row["independent_group_count"]),
                -int(row["sample_count"]),
                -float(row["threshold"]),
            ),
        )
        if evaluated
        else None
    )
    return {
        "qualified": selected is not None,
        "selection_rule": "largest score <= threshold satisfying every oil constraint",
        "selected": selected,
        "best_available": best_available,
        "candidate_threshold_count": len(evaluated),
        "constraints": {
            "strictly_below_brake_threshold": float(brake_threshold),
            "confidence": confidence,
            "maximum_failure_rate_upper_bound": maximum_failure_rate_upper_bound,
            "independent_unit": "prompt_index",
            "minimum_independent_group_count": minimum_independent_group_count,
            "require_zero_observed_failures": require_zero_observed_failures,
        },
    }


def evaluate_development(
    development: Mapping[str, Any],
    methodology: Mapping[str, Any],
    *,
    target_candidate_id: str,
    methodology_preregistered: bool,
) -> dict[str, Any]:
    selected_name = str(development["selection"]["selected_model"])
    selected_model = development["models"][selected_name]
    target_metrics = selected_model["by_candidate"].get(target_candidate_id)
    if target_metrics is None:
        raise ValueError("selected model has no target-policy OOF metrics")
    target_rows = [
        row for row in development["rows"]
        if str(row["candidate_id"]) == target_candidate_id
    ]
    labels = [int(bool(row["failed"])) for row in target_rows]
    scores = [float(row["oof_probabilities"][selected_name]) for row in target_rows]
    group_ids = [int(row["prompt_index"]) for row in target_rows]
    brake = threshold_at_full_recall(labels, scores)

    oil_policy = methodology["oil_threshold"]
    oil = select_oil_threshold(
        labels,
        scores,
        group_ids,
        brake_threshold=float(brake["threshold"]),
        confidence=float(oil_policy["confidence"]),
        maximum_failure_rate_upper_bound=float(
            oil_policy["maximum_failure_rate_upper_bound"]
        ),
        minimum_independent_group_count=int(
            oil_policy["minimum_independent_group_count"]
        ),
        require_zero_observed_failures=bool(
            oil_policy["require_zero_observed_failures"]
        ),
    )

    policy = methodology["development"]
    brake_policy = methodology["brake_threshold"]
    pooled_auc = selected_model["metrics"]["roc_auc"]
    target_auc = target_metrics["roc_auc"]
    checks = {
        "minimum_total_failures": int(development["failure_count"])
        >= int(policy["minimum_total_failures"]),
        "minimum_target_failures": int(
            development["failure_count_by_candidate"].get(target_candidate_id, 0)
        )
        >= int(policy["minimum_target_failures"]),
        "minimum_pooled_oof_roc_auc": pooled_auc is not None
        and float(pooled_auc) >= float(policy["minimum_pooled_oof_roc_auc"]),
        "minimum_target_oof_roc_auc": target_auc is not None
        and float(target_auc) >= float(policy["minimum_target_oof_roc_auc"]),
        "full_failure_recall": float(brake["failure_recall"])
        >= float(brake_policy["required_failure_recall"]),
        "maximum_passing_false_brake_rate": brake["passing_false_brake_rate"]
        is not None
        and float(brake["passing_false_brake_rate"])
        <= float(brake_policy["maximum_passing_false_brake_rate"]),
        "qualified_oil_all_clear_zone": bool(oil["qualified"]),
        "methodology_preregistered": bool(methodology_preregistered),
    }
    counts_ok = checks["minimum_total_failures"] and checks["minimum_target_failures"]
    ranking_ok = (
        checks["minimum_pooled_oof_roc_auc"]
        and checks["minimum_target_oof_roc_auc"]
        and checks["full_failure_recall"]
        and checks["maximum_passing_false_brake_rate"]
    )
    exportable = counts_ok and ranking_ok and checks["qualified_oil_all_clear_zone"]
    # Methodology provenance is a hard export requirement. It is intentionally
    # separate from scientific rejection: the current study can validly reject
    # a feature family even though methodology-v1 was codified afterwards.
    exportable = exportable and checks["methodology_preregistered"]
    if not counts_ok:
        status = "insufficient_failures"
    elif not ranking_ok:
        status = "reject_feature_family"
    elif not checks["qualified_oil_all_clear_zone"]:
        status = "brake_only_development"
    elif not checks["methodology_preregistered"]:
        status = "replay_only_methodology_not_preregistered"
    else:
        status = "freeze_for_prospective_confirmation"
    return {
        "status": status,
        "export_frozen_gate": exportable,
        "selected_model": selected_name,
        "target_candidate_id": target_candidate_id,
        "pooled_oof": selected_model["metrics"],
        "target_oof": target_metrics,
        "brake_threshold": {
            **brake,
            "comparison": "risk_score >= threshold",
            "source": "target-policy grouped OOF predictions",
        },
        "oil_threshold": {
            **oil,
            "comparison": "risk_score <= threshold",
            "source": "target-policy grouped OOF predictions",
        },
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "serving_claim": False,
        "next_stage": (
            "freeze coefficients and thresholds, then run a prospectively registered "
            "threshold confirmation followed by an interventional quality/speed holdout"
            if exportable
            else "do not run a confirmation holdout or deploy a threshold"
        ),
    }


def build_frozen_gate(
    development: Mapping[str, Any],
    decision: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    if not decision.get("export_frozen_gate"):
        raise ValueError("development decision does not permit frozen-gate export")
    selected = development["models"][decision["selected_model"]]
    oil_selected = decision["oil_threshold"]["selected"]
    if not isinstance(oil_selected, dict):
        raise ValueError("qualified development decision has no oil threshold")
    return {
        "schema": FROZEN_GATE_SCHEMA,
        "schema_revision": FROZEN_GATE_SCHEMA_REVISION,
        "state": "awaiting_prospective_confirmation",
        "target_candidate_id": decision["target_candidate_id"],
        "feature_names": selected["feature_names"],
        "model": selected["final_fit"],
        "thresholds": {
            "brake": {
                "comparison": "risk_score >= threshold",
                "threshold": decision["brake_threshold"]["threshold"],
            },
            "oil": {
                "comparison": "risk_score <= threshold",
                "threshold": oil_selected["threshold"],
            },
        },
        "provenance": dict(integrity),
        "serving_claim": False,
        "required_next_stage": (
            "prospective threshold confirmation and interventional quality/speed holdout"
        ),
    }


def run_development(args: argparse.Namespace) -> dict[str, Any]:
    registration_path = args.registration.expanduser().resolve()
    methodology_path = args.methodology.expanduser().resolve()
    quality_path = args.quality_input.expanduser().resolve()
    semantic_path = args.semantic_report.expanduser().resolve()
    _registration, methodology, integrity = validate_development_inputs(
        registration_path,
        methodology_path,
        quality_path,
        semantic_path,
    )
    development = analyze(quality_path, semantic_path)
    decision = evaluate_development(
        development,
        methodology,
        target_candidate_id=str(integrity["target_candidate_id"]),
        methodology_preregistered=bool(integrity["methodology_preregistered"]),
    )
    closure = {
        "schema": CLOSURE_SCHEMA,
        "schema_revision": CLOSURE_SCHEMA_REVISION,
        "phase": "development",
        "integrity": integrity,
        "development_artifact": {
            "path": str(args.development_out.expanduser().resolve()),
        },
        "decision": decision,
        "qualification": {
            "serving_claim": False,
            "development_only": True,
            "note": (
                "A positive development decision only freezes a candidate gate; "
                "it never qualifies serving by itself."
            ),
        },
    }
    _write_json(args.development_out.expanduser().resolve(), development)
    closure["development_artifact"]["sha256"] = _sha256_file(
        args.development_out.expanduser().resolve()
    )
    _write_json(args.decision_out.expanduser().resolve(), closure)

    if args.gate_out is not None:
        gate_path = args.gate_out.expanduser().resolve()
        if decision["export_frozen_gate"]:
            _write_json(gate_path, build_frozen_gate(development, decision, integrity))
        elif gate_path.exists():
            raise RuntimeError(
                f"development rejected the gate but stale --gate-out already exists: {gate_path}"
            )
    return closure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    develop = subparsers.add_parser(
        "develop",
        help="validate frozen discovery artifacts, fit grouped OOF models, and apply stopping rules",
    )
    develop.add_argument("--registration", type=Path, required=True)
    develop.add_argument("--methodology", type=Path, required=True)
    develop.add_argument("--quality-input", type=Path, required=True)
    develop.add_argument("--semantic-report", type=Path, required=True)
    develop.add_argument("--development-out", type=Path, required=True)
    develop.add_argument("--decision-out", type=Path, required=True)
    develop.add_argument(
        "--gate-out",
        type=Path,
        help="written only if all development and methodology-provenance gates pass",
    )
    replay = subparsers.add_parser(
        "profile-replay",
        help="recompute the legacy offline adaptive-profile choice without upgrading its claim",
    )
    replay.add_argument("--methodology", type=Path, required=True)
    replay.add_argument("--stage-result", type=Path, required=True)
    replay.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="PROFILE_NAME=PATH; repeat once for every profile in the study result",
    )
    replay.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "profile-replay":
        replay = run_profile_replay(args)
        print(json.dumps(replay, indent=2, sort_keys=True))
        print(f"profile_replay={args.out}")
        return 0
    if args.command != "develop":
        raise AssertionError(f"unhandled command: {args.command}")
    closure = run_development(args)
    print(json.dumps(closure["decision"], indent=2, sort_keys=True))
    print(f"development={args.development_out}")
    print(f"decision={args.decision_out}")
    if args.gate_out is not None:
        print(
            f"gate={args.gate_out if closure['decision']['export_frozen_gate'] else 'NOT_EXPORTED'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
