"""Derive frozen FLUX cache schedules from label-free full-DiT trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.offline.cache_profile import schedule as schedule_math  # noqa: E402
from difflet.offline.cache_profile import provenance  # noqa: E402
from scripts.flux_cache_phased_candidate import (  # noqa: E402
    PHASED_CANDIDATE_SCHEMA,
    PHASED_CANDIDATE_SCHEMA_REVISION,
    load_phased_candidate,
)
from scripts.flux_cache_protocol import canonical_sha256  # noqa: E402

REGISTRATION_SCHEMA = "difflet-flux-cache-schedule-derivation-registration"
REGISTRATION_SCHEMA_REVISION = 3
RESULT_SCHEMA = "difflet-flux-cache-schedule-derivation-result"
RESULT_SCHEMA_REVISION = 2
CANDIDATE_SET_SCHEMA = "difflet-flux-cache-derived-candidate-set"
CANDIDATE_SET_SCHEMA_REVISION = 2

_IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    Path(schedule_math.__file__).resolve(),
    Path(provenance.__file__).resolve(),
    ROOT / "scripts" / "flux_cache_phased_candidate.py",
    ROOT / "scripts" / "flux_cache_protocol.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {name} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return document


def _repo_path(value: str, name: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{name} must be repository-relative")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"{name} escapes the repository")
    return resolved


# Backward-compatible private aliases keep the CLI import surface stable while
# the implementation lives in the offline domain package.
_nearest_rank = schedule_math.nearest_rank
_finite_positive = schedule_math.finite_positive
_fit_affine_step_cost = schedule_math.fit_affine_step_cost
_derive_hardware_budget = schedule_math.derive_hardware_budget
_scheduler_sigmas = schedule_math.scheduler_sigmas
_relative_prediction_error = schedule_math.relative_prediction_error
_gap_cap = schedule_math.gap_cap
_optimize_mask = schedule_math.optimize_mask
_materialized_path_segments = schedule_math.materialized_path_segments


def _hardware_budget_payload(
    speed_manifest_path: Path,
    generation: Mapping[str, Any],
    *,
    target_speedup: float,
    dynamic_step_reserve: int,
    warmup_steps: int,
    cooldown_steps: int,
) -> dict[str, Any]:
    manifest = _load_json(speed_manifest_path, "hardware speed manifest")
    if manifest.get("schema") != "speedup-candidates-v1":
        raise ValueError("hardware speed manifest schema is unsupported")
    if manifest.get("hardware_measured") is not True:
        raise ValueError("hardware speed manifest must contain measured hardware timings")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("generation") != dict(generation):
        raise ValueError("hardware speed manifest generation identity differs")
    hardware = protocol.get("hardware")
    if not isinstance(hardware, dict):
        raise ValueError("hardware speed manifest has no hardware identity")

    baseline = manifest.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("hardware speed manifest has no baseline timing")
    baseline_latency = _finite_positive(
        baseline.get("total_s"),
        "hardware baseline total_s",
    )
    sample_count = manifest.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("hardware speed manifest sample_count must be positive")

    points: list[tuple[int, float]] = []
    bindings: list[dict[str, Any]] = []
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("hardware speed manifest candidates must be a list")
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("hardware speed manifest contains a malformed candidate")
        policy = candidate.get("policy")
        if not isinstance(policy, dict) or policy.get("type") != "phased_static":
            continue
        if policy.get("dynamic_budget") != 0:
            raise ValueError("static hardware calibration candidate has a dynamic budget")
        anchor_steps = policy.get("static_anchor_steps")
        if not isinstance(anchor_steps, list) or not anchor_steps:
            raise ValueError("static hardware calibration candidate has no anchor steps")
        anchor_count = len(anchor_steps)
        total_s = _finite_positive(candidate.get("total_s"), "static candidate total_s")
        points.append((anchor_count, total_s))
        bindings.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "real_steps": anchor_count,
                "aggregate_latency_s": total_s,
            }
        )

    cost_model = _fit_affine_step_cost(points)
    budget = _derive_hardware_budget(
        num_steps=int(generation["num_steps"]),
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
        baseline_latency_s=baseline_latency,
        target_speedup=target_speedup,
        intercept_s=float(cost_model["intercept_s"]),
        incremental_real_step_s=float(cost_model["incremental_real_step_s"]),
        dynamic_step_reserve=dynamic_step_reserve,
    )
    return {
        **budget,
        "latency_statistic": "aggregate_wall_time_over_shared_requests",
        "sample_count": sample_count,
        "baseline_aggregate_latency_s": baseline_latency,
        "hardware": dict(hardware),
        "source": {
            "path": str(speed_manifest_path.resolve()),
            "file_sha256": sha256_file(speed_manifest_path),
            "schema": manifest["schema"],
            "static_profiles": bindings,
        },
        "cost_model": cost_model,
    }


def _registration_payload(
    quality_path: Path,
    methodology_path: Path,
    hardware_speed_manifest_path: Path,
    *,
    study_id: str,
    created_at: str,
    target_speedup: float,
) -> dict[str, Any]:
    quality = _load_json(quality_path, "A2 quality input")
    protocol = quality.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("A2 quality input has no experiment protocol")
    generation = protocol.get("generation")
    if not isinstance(generation, dict) or int(generation.get("num_steps", -1)) != 50:
        raise ValueError("A2 generation identity is unsupported")
    if generation.get("dtype") != "bfloat16" or int(quality.get("prompt_count", -1)) != 48:
        raise ValueError("A2 trajectory matrix is unsupported")
    comparisons = quality.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 96:
        raise ValueError("A2 quality input must contain 96 candidate comparisons")

    rows: dict[str, dict[str, str]] = {}
    for comparison in comparisons:
        sample_id = comparison.get("sample_id")
        baseline = comparison.get("baseline")
        if not isinstance(sample_id, str) or not isinstance(baseline, dict):
            raise ValueError("A2 comparison is malformed")
        relative = baseline.get("trajectory")
        if not isinstance(relative, str):
            raise ValueError("A2 baseline trajectory is missing")
        trajectory_path = (quality_path.parent / relative).resolve()
        if not trajectory_path.is_file():
            raise ValueError(f"A2 baseline trajectory does not exist: {trajectory_path}")
        row = {
            "sample_id": sample_id,
            "path": str(trajectory_path),
            "file_sha256": sha256_file(trajectory_path),
        }
        previous = rows.setdefault(sample_id, row)
        if previous != row:
            raise ValueError("candidate comparisons disagree on the shared baseline")
    if len(rows) != 48:
        raise ValueError("A2 quality input must bind 48 unique baseline trajectories")

    warmup_steps = 6
    cooldown_steps = 1
    dynamic_step_reserve = 2
    hardware_budget = _hardware_budget_payload(
        hardware_speed_manifest_path,
        generation,
        target_speedup=target_speedup,
        dynamic_step_reserve=dynamic_step_reserve,
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
    )

    methodology_relative = methodology_path.resolve().relative_to(ROOT).as_posix()
    return {
        "schema": REGISTRATION_SCHEMA,
        "schema_revision": REGISTRATION_SCHEMA_REVISION,
        "study_id": study_id,
        "created_at": created_at,
        "status": "registered_not_derived",
        "evidence_role": {
            "stage": "label_free_schedule_candidate_generation",
            "semantic_labels_permitted": False,
            "serving_claim_permitted": False,
        },
        "source": {
            "quality_input_path": str(quality_path.resolve()),
            "quality_input_file_sha256": sha256_file(quality_path),
            "quality_input_schema": quality.get("schema"),
            "protocol_sha256": protocol.get("sha256"),
            "trajectory_count": len(rows),
            "trajectories": [rows[key] for key in sorted(rows)],
        },
        "controlled_generation": generation,
        "hardware_budget": hardware_budget,
        "cache_semantics": {
            "prediction_target": "transformer_noise_prediction",
            "predictor": {"type": "taylorseer", "order": 1, "coord": "index"},
            "trajectory_reconstruction": "v_t=(x_after_t-x_after_t_minus_1)/(sigma_t_plus_1-sigma_t) for t>=1",
        },
        "optimizer": {
            "objective": "sum_of_segment_prompt_q95_scheduler_weighted_relative_l2_error",
            "segment_formula": "sum_skipped(abs(delta_sigma_t)*l2(vhat_t-v_t)/max(l2(v_t),norm_floor))",
            "prompt_aggregation": "nearest_rank",
            "prompt_quantile": 0.95,
            "working_dtype": "float32",
            "norm_floor": 1e-8,
            "warmup_steps": warmup_steps,
            "cooldown_steps": cooldown_steps,
            "require_final_anchor": True,
            "anchor_budget_source": "hardware_budget.static_anchor_budget",
            "phase_boundary_step": 21,
            "middle_max_anchor_gap": 6,
            "tail_max_anchor_gap": 10,
            "tie_break": "lexicographically_smallest_anchor_sequence",
        },
        "bounded_brake": {
            "plastic_window": [6, 29],
            "dynamic_budget": dynamic_step_reserve,
            "tighten_quantile": 0.95,
            "recovery_quantile": 0.99,
            "quantile_method": "nearest_rank_over_label_free_predicted_anchor_errors",
            "tighten_rule": "bisect_next_static_gap",
            "recovery_steps": 2,
            "disable_after_recoveries": 2,
            "allow_acceleration": False,
            "invalid_measurement_fail_closed": True,
        },
        "quality_contract_ref": {
            "path": methodology_relative,
            "file_sha256": sha256_file(methodology_path),
        },
        "implementation": provenance.implementation_bundle(
            _IMPLEMENTATION_PATHS,
            root=ROOT,
        ),
        "parameters_frozen_before_derivation": True,
    }


def register(args: argparse.Namespace) -> None:
    quality_path = Path(args.quality_input).expanduser().resolve()
    methodology_path = _repo_path(args.methodology, "methodology")
    hardware_speed_manifest_path = Path(args.hardware_speed_manifest).expanduser().resolve()
    payload = _registration_payload(
        quality_path,
        methodology_path,
        hardware_speed_manifest_path,
        study_id=args.study_id,
        created_at=args.created_at,
        target_speedup=args.target_speedup,
    )
    _write_json(
        Path(args.out).expanduser().resolve(), {**payload, "sha256": canonical_sha256(payload)}
    )


def load_registration(path: Path) -> dict[str, Any]:
    document = _load_json(path, "schedule derivation registration")
    if (
        document.get("schema") != REGISTRATION_SCHEMA
        or document.get("schema_revision") != REGISTRATION_SCHEMA_REVISION
    ):
        raise ValueError("schedule derivation registration schema is unsupported")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if canonical_sha256(payload) != document.get("sha256"):
        raise ValueError("schedule derivation registration content hash is invalid")
    if document.get("status") != "registered_not_derived":
        raise ValueError("schedule derivation registration status is invalid")
    if document.get("parameters_frozen_before_derivation") is not True:
        raise ValueError("schedule derivation parameters were not frozen")
    source = document["source"]
    quality_path = Path(source["quality_input_path"]).resolve()
    if sha256_file(quality_path) != source["quality_input_file_sha256"]:
        raise ValueError("registered A2 quality input hash differs")
    for row in source["trajectories"]:
        trajectory_path = Path(row["path"]).resolve()
        if sha256_file(trajectory_path) != row["file_sha256"]:
            raise ValueError(f"registered trajectory hash differs for {row['sample_id']}")
    provenance.validate_implementation_bundle(
        document["implementation"],
        root=ROOT,
        required_paths=_IMPLEMENTATION_PATHS,
    )
    contract = document["quality_contract_ref"]
    contract_path = _repo_path(contract["path"], "quality contract")
    if sha256_file(contract_path) != contract["file_sha256"]:
        raise ValueError("registered quality contract hash differs")
    optimizer = document["optimizer"]
    if optimizer.get("anchor_budget_source") != "hardware_budget.static_anchor_budget":
        raise ValueError("registered anchor budget source is unsupported")
    if "anchor_budgets" in optimizer:
        raise ValueError("registered optimizer must not contain manual anchor budgets")
    hardware_budget = document["hardware_budget"]
    hardware_source = hardware_budget["source"]
    speed_manifest_path = Path(hardware_source["path"]).resolve()
    if sha256_file(speed_manifest_path) != hardware_source["file_sha256"]:
        raise ValueError("registered hardware speed manifest hash differs")
    recomputed_budget = _hardware_budget_payload(
        speed_manifest_path,
        document["controlled_generation"],
        target_speedup=float(hardware_budget["target_speedup"]),
        dynamic_step_reserve=int(hardware_budget["dynamic_step_reserve"]),
        warmup_steps=int(optimizer["warmup_steps"]),
        cooldown_steps=int(optimizer["cooldown_steps"]),
    )
    if recomputed_budget != hardware_budget:
        raise ValueError("registered hardware-derived anchor budget differs")
    if int(document["bounded_brake"]["dynamic_budget"]) != int(
        hardware_budget["dynamic_step_reserve"]
    ):
        raise ValueError("brake budget differs from the hardware dynamic-step reserve")
    return document


def _trajectory_gram(path: Path, deltas: Sequence[float]) -> Any:
    """Compatibility wrapper for the offline schedule implementation."""

    return schedule_math.trajectory_gram(path, deltas)


def derive(args: argparse.Namespace) -> None:
    registration_path = Path(args.registration).expanduser().resolve()
    registration = load_registration(registration_path)
    generation = registration["controlled_generation"]
    sigmas = _scheduler_sigmas(generation)
    deltas = tuple(sigmas[index + 1] - sigmas[index] for index in range(len(sigmas) - 1))
    optimizer = registration["optimizer"]
    norm_floor = float(optimizer["norm_floor"])
    prompt_quantile = float(optimizer["prompt_quantile"])
    num_steps = int(generation["num_steps"])

    prompt_segment_costs: dict[tuple[int, int, int], list[float]] = {}
    prompt_anchor_errors: dict[tuple[int, int, int], list[float]] = {}
    maximum_gap = max(
        int(optimizer["middle_max_anchor_gap"]),
        int(optimizer["tail_max_anchor_gap"]),
    )
    for row in registration["source"]["trajectories"]:
        gram = _trajectory_gram(Path(row["path"]), deltas)
        for a in range(1, num_steps - 2):
            for b in range(a + 1, min(num_steps - 1, a + maximum_gap) + 1):
                cap = _gap_cap(
                    b,
                    boundary=int(optimizer["phase_boundary_step"]),
                    middle_cap=int(optimizer["middle_max_anchor_gap"]),
                    tail_cap=int(optimizer["tail_max_anchor_gap"]),
                )
                for c in range(b + 1, min(num_steps - 1, b + cap) + 1):
                    key = (a, b, c)
                    cost = 0.0
                    for target in range(b + 1, c):
                        cost += abs(deltas[target]) * _relative_prediction_error(
                            gram,
                            a,
                            b,
                            target,
                            norm_floor=norm_floor,
                        )
                    prompt_segment_costs.setdefault(key, []).append(cost)
                    prompt_anchor_errors.setdefault(key, []).append(
                        _relative_prediction_error(
                            gram,
                            a,
                            b,
                            c,
                            norm_floor=norm_floor,
                        )
                    )

    expected_prompts = int(registration["source"]["trajectory_count"])
    segment_costs: dict[tuple[int, int, int], float] = {}
    for key, values in prompt_segment_costs.items():
        if len(values) != expected_prompts:
            raise RuntimeError("segment cost matrix is incomplete")
        segment_costs[key] = _nearest_rank(values, prompt_quantile)

    schedules = []
    brake = registration["bounded_brake"]
    target_speedup = float(registration["hardware_budget"]["target_speedup"])
    budget = int(registration["hardware_budget"]["static_anchor_budget"])
    for budget in (budget,):
        anchors, objective = _optimize_mask(
            segment_costs,
            num_steps=num_steps,
            warmup_steps=int(optimizer["warmup_steps"]),
            anchor_budget=int(budget),
            phase_boundary=int(optimizer["phase_boundary_step"]),
            middle_gap_cap=int(optimizer["middle_max_anchor_gap"]),
            tail_gap_cap=int(optimizer["tail_max_anchor_gap"]),
        )
        path_segments = _materialized_path_segments(
            anchors,
            segment_costs,
            warmup_steps=int(optimizer["warmup_steps"]),
        )
        threshold_values: list[float] = []
        for index in range(int(optimizer["warmup_steps"]), len(anchors)):
            a, b, c = anchors[index - 2 : index + 1]
            key = (a, b, c)
            if brake["plastic_window"][0] <= c <= brake["plastic_window"][1]:
                threshold_values.extend(prompt_anchor_errors[key])
        tighten = _nearest_rank(threshold_values, float(brake["tighten_quantile"]))
        recovery = _nearest_rank(threshold_values, float(brake["recovery_quantile"]))
        if not tighten < recovery:
            raise ValueError(f"derived brake thresholds are not ordered for budget {budget}")
        schedules.append(
            {
                "anchor_budget": int(budget),
                "target_speedup": target_speedup,
                "total_real_step_budget": int(
                    registration["hardware_budget"]["total_real_step_budget"]
                ),
                "static_anchor_steps": list(anchors),
                "objective_cost": objective,
                "segments": path_segments,
                "brake_thresholds": {
                    "observation_count": len(threshold_values),
                    "tighten_error": tighten,
                    "recovery_error": recovery,
                },
            }
        )

    payload = {
        "schema": RESULT_SCHEMA,
        "schema_revision": RESULT_SCHEMA_REVISION,
        "study_id": registration["study_id"],
        "status": "derived_without_semantic_labels",
        "registration": {
            "path": str(registration_path),
            "file_sha256": sha256_file(registration_path),
            "content_sha256": registration["sha256"],
        },
        "source_trajectory_count": expected_prompts,
        "hardware_budget": registration["hardware_budget"],
        "sigma_schedule": list(sigmas),
        "sigma_schedule_sha256": canonical_sha256(list(sigmas)),
        "schedules": schedules,
        "semantic_labels_read": False,
    }
    _write_json(
        Path(args.out).expanduser().resolve(), {**payload, "sha256": canonical_sha256(payload)}
    )


def _candidate_document(
    *,
    schedule: Mapping[str, Any],
    combined: bool,
    registration: Mapping[str, Any],
    derivation_path: Path,
    derivation: Mapping[str, Any],
) -> dict[str, Any]:
    optimizer = registration["optimizer"]
    brake = registration["bounded_brake"]
    budget = int(schedule["anchor_budget"])
    policy: dict[str, Any] = {
        "type": "phased_static_plus_brake" if combined else "phased_static",
        "num_steps": int(registration["controlled_generation"]["num_steps"]),
        "static_anchor_steps": list(schedule["static_anchor_steps"]),
        "warmup_steps": int(optimizer["warmup_steps"]),
        "cooldown_steps": int(optimizer["cooldown_steps"]),
        "require_final_anchor": bool(optimizer["require_final_anchor"]),
        "dynamic_budget": int(brake["dynamic_budget"]) if combined else 0,
        "invalid_measurement_fail_closed": True,
    }
    if combined:
        policy.update(
            {
                "plastic_window": list(brake["plastic_window"]),
                "tighten_error": float(schedule["brake_thresholds"]["tighten_error"]),
                "recovery_error": float(schedule["brake_thresholds"]["recovery_error"]),
                "recovery_steps": int(brake["recovery_steps"]),
                "disable_after_recoveries": int(brake["disable_after_recoveries"]),
                "tighten_rule": brake["tighten_rule"],
                "allow_acceleration": False,
            }
        )
    target_token = format(float(schedule["target_speedup"]), ".6g").replace(".", "p")
    candidate_id = (
        f"target-static-brake-s{target_token}-a{budget}-b{brake['dynamic_budget']}-o1-index"
        if combined
        else f"target-static-s{target_token}-a{budget}-o1-index"
    )
    payload = {
        "schema": PHASED_CANDIDATE_SCHEMA,
        "schema_revision": PHASED_CANDIDATE_SCHEMA_REVISION,
        "candidate_id": candidate_id,
        "policy": policy,
        "predictor": {"type": "taylorseer", "order": 1, "coord": "index"},
        "horizon_ref": {
            "path": derivation_path.resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256_file(derivation_path),
        },
        "quality_contract_ref": {
            "path": registration["quality_contract_ref"]["path"],
            "sha256": registration["quality_contract_ref"]["file_sha256"],
        },
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def materialize(args: argparse.Namespace) -> None:
    registration_path = Path(args.registration).expanduser().resolve()
    derivation_path = Path(args.derivation).expanduser().resolve()
    registration = load_registration(registration_path)
    derivation = _load_json(derivation_path, "schedule derivation result")
    payload = {key: value for key, value in derivation.items() if key != "sha256"}
    if (
        derivation.get("schema") != RESULT_SCHEMA
        or derivation.get("schema_revision") != RESULT_SCHEMA_REVISION
        or canonical_sha256(payload) != derivation.get("sha256")
        or derivation.get("registration", {}).get("content_sha256") != registration["sha256"]
        or derivation.get("semantic_labels_read") is not False
    ):
        raise ValueError("schedule derivation result does not match the registration")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bindings = []
    for schedule in derivation["schedules"]:
        for combined in (False, True):
            document = _candidate_document(
                schedule=schedule,
                combined=combined,
                registration=registration,
                derivation_path=derivation_path,
                derivation=derivation,
            )
            candidate_path = out_dir / f"{document['candidate_id']}.json"
            _write_json(candidate_path, document)
            arm = load_phased_candidate(candidate_path)
            bindings.append(
                {
                    "path": candidate_path.relative_to(ROOT).as_posix(),
                    "file_sha256": arm.file_sha256,
                    "content_sha256": arm.content_sha256,
                    "candidate_id": arm.candidate_id,
                    "anchor_budget": int(schedule["anchor_budget"]),
                    "target_speedup": float(schedule["target_speedup"]),
                    "total_real_step_budget": int(schedule["total_real_step_budget"]),
                    "family": "combined" if combined else "static",
                }
            )
    manifest_payload = {
        "schema": CANDIDATE_SET_SCHEMA,
        "schema_revision": CANDIDATE_SET_SCHEMA_REVISION,
        "study_id": registration["study_id"],
        "registration": {
            "path": registration_path.relative_to(ROOT).as_posix(),
            "file_sha256": sha256_file(registration_path),
            "content_sha256": registration["sha256"],
        },
        "derivation": {
            "path": derivation_path.relative_to(ROOT).as_posix(),
            "file_sha256": sha256_file(derivation_path),
            "content_sha256": derivation["sha256"],
        },
        "candidates": bindings,
    }
    manifest_path = out_dir / "derived-schedule-candidate-set.json"
    _write_json(
        manifest_path,
        {**manifest_payload, "sha256": canonical_sha256(manifest_payload)},
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--quality-input", required=True)
    register_parser.add_argument("--methodology", required=True)
    register_parser.add_argument("--hardware-speed-manifest", required=True)
    register_parser.add_argument("--target-speedup", required=True, type=float)
    register_parser.add_argument("--study-id", required=True)
    register_parser.add_argument("--created-at", required=True)
    register_parser.add_argument("--out", required=True)
    register_parser.set_defaults(func=register)

    derive_parser = subparsers.add_parser("derive")
    derive_parser.add_argument("--registration", required=True)
    derive_parser.add_argument("--out", required=True)
    derive_parser.set_defaults(func=derive)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--registration", required=True)
    materialize_parser.add_argument("--derivation", required=True)
    materialize_parser.add_argument("--out-dir", required=True)
    materialize_parser.set_defaults(func=materialize)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        args.func(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
