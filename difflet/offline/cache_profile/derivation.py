"""Derive frozen FLUX cache schedules from label-free full-DiT trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.offline.cache_profile import schedule as schedule_math  # noqa: E402
from difflet.offline.cache_profile import provenance  # noqa: E402
from difflet.pipeline.cache import profile as runtime_profile  # noqa: E402
from difflet.pipeline.cache.profile import (  # noqa: E402
    PHASED_CANDIDATE_SCHEMA,
    PHASED_CANDIDATE_SCHEMA_REVISION,
    load_phased_candidate,
)
from scripts.flux_cache_protocol import canonical_sha256  # noqa: E402

REGISTRATION_SCHEMA = "difflet-flux-cache-schedule-derivation-registration"
REGISTRATION_SCHEMA_REVISION = 5
RESULT_SCHEMA = "difflet-flux-cache-schedule-derivation-result"
RESULT_SCHEMA_REVISION = 4
CANDIDATE_SET_SCHEMA = "difflet-flux-cache-derived-candidate-set"
CANDIDATE_SET_SCHEMA_REVISION = 4

EMPIRICAL_SEARCH_FLOOR_RATIO = 0.20
EMPIRICAL_SEARCH_FLOOR_FORMULA = "ceil(0.20 * total_steps)"
EMPIRICAL_SEARCH_FLOOR_REFERENCE = {
    "title": "Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache",
    "venue": "CVPR 2026",
    "evidence": "FLUX.1-dev evaluates 9 and 13 full-compute key-timestep schedules over 50 denoising steps",
    "url": "https://openaccess.thecvf.com/content/CVPR2026/html/Cui_Denoising_as_Path_Planning_Training-Free_Acceleration_of_Diffusion_Models_with_CVPR_2026_paper.html",
}

_IMPLEMENTATION_PATHS = (
    Path(__file__).resolve(),
    Path(schedule_math.__file__).resolve(),
    Path(provenance.__file__).resolve(),
    Path(runtime_profile.__file__).resolve(),
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
_scheduler_sigmas = schedule_math.scheduler_sigmas
_relative_prediction_error = schedule_math.relative_prediction_error
_gap_cap = schedule_math.gap_cap
_optimize_budget_frontier = schedule_math.optimize_budget_frontier
_materialized_path_segments = schedule_math.materialized_path_segments


def empirical_search_floor(num_steps: int) -> int:
    """Return the pre-registered, literature-informed ladder search floor."""

    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    return int(math.ceil(EMPIRICAL_SEARCH_FLOOR_RATIO * num_steps))


def _registration_payload(
    trajectory_input_path: Path,
    quality_contract_path: Path,
    *,
    study_id: str,
    created_at: str,
) -> dict[str, Any]:
    trajectory_input = _load_json(trajectory_input_path, "trajectory input")
    protocol = trajectory_input.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("trajectory input has no experiment protocol")
    generation = protocol.get("generation")
    if not isinstance(generation, dict) or int(generation.get("num_steps", -1)) != 50:
        raise ValueError("trajectory input generation identity is unsupported")
    if (
        generation.get("dtype") != "bfloat16"
        or int(trajectory_input.get("prompt_count", -1)) != 48
    ):
        raise ValueError("trajectory input matrix is unsupported")
    rows: dict[str, dict[str, str]] = {}
    if trajectory_input.get("schema") == "difflet-flux-cache-trajectory-input-v1":
        trajectories = trajectory_input.get("trajectories")
        if (
            trajectory_input.get("semantic_labels_collected") is not False
            or not isinstance(trajectories, list)
            or int(trajectory_input.get("trajectory_count", -1)) != len(trajectories)
        ):
            raise ValueError("trajectory input manifest is malformed")
        for source in trajectories:
            if not isinstance(source, dict):
                raise ValueError("trajectory input row is malformed")
            sample_id = source.get("sample_id")
            relative = source.get("path")
            expected_hash = source.get("file_sha256")
            if not isinstance(sample_id, str) or not isinstance(relative, str):
                raise ValueError("trajectory input row identity is malformed")
            trajectory_path = (trajectory_input_path.parent / relative).resolve()
            if (
                not trajectory_path.is_file()
                or not isinstance(expected_hash, str)
                or sha256_file(trajectory_path) != expected_hash
            ):
                raise ValueError(f"trajectory input file binding is invalid: {sample_id}")
            if sample_id in rows:
                raise ValueError("trajectory input contains duplicate sample ids")
            rows[sample_id] = {
                "sample_id": sample_id,
                "path": str(trajectory_path),
                "file_sha256": expected_hash,
            }
    else:
        comparisons = trajectory_input.get("comparisons")
        if not isinstance(comparisons, list) or not comparisons:
            raise ValueError(
                "legacy trajectory source must contain candidate comparisons"
            )
        for comparison in comparisons:
            sample_id = comparison.get("sample_id")
            baseline = comparison.get("baseline")
            if not isinstance(sample_id, str) or not isinstance(baseline, dict):
                raise ValueError("A2 comparison is malformed")
            relative = baseline.get("trajectory")
            if not isinstance(relative, str):
                raise ValueError("A2 baseline trajectory is missing")
            trajectory_path = (trajectory_input_path.parent / relative).resolve()
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
        raise ValueError("trajectory input must bind 48 unique baseline trajectories")

    # Order-1 TaylorSeer requires two real history anchors.  The current
    # trajectory artifact starts after denoising step zero, so the label-free
    # objective first becomes observable with the third real anchor.  Anchor
    # budget selection is independent: expensive qualification starts at the
    # pre-registered literature-informed 20% search floor below.
    warmup_steps = 3
    cooldown_steps = 0
    search_floor = empirical_search_floor(int(generation["num_steps"]))

    quality_contract_relative = quality_contract_path.resolve().relative_to(ROOT).as_posix()
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
            "trajectory_input_path": str(trajectory_input_path.resolve()),
            "trajectory_input_file_sha256": sha256_file(trajectory_input_path),
            "trajectory_input_schema": trajectory_input.get("schema"),
            "protocol_sha256": protocol.get("sha256"),
            "trajectory_count": len(rows),
            "trajectories": [rows[key] for key in sorted(rows)],
        },
        "controlled_generation": generation,
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
            "anchor_budget_source": "literature_informed_empirical_search_floor",
            "budget_selection": "ascending_first_closed_loop_quality_pass",
            "search_floor_role": "candidate_domain_floor_not_quality_guarantee",
            "search_floor_formula": EMPIRICAL_SEARCH_FLOOR_FORMULA,
            "search_floor_ratio": EMPIRICAL_SEARCH_FLOOR_RATIO,
            "search_floor_anchor_budget": search_floor,
            "search_floor_reference": dict(EMPIRICAL_SEARCH_FLOOR_REFERENCE),
            "predictor_history_floor": 2,
            "trajectory_observation_floor": warmup_steps,
            "closed_loop_mechanism_floor": 4,
            "phase_boundary_step": 21,
            "middle_max_anchor_gap": 6,
            "tail_max_anchor_gap": 10,
            "tie_break": "lexicographically_smallest_anchor_sequence",
        },
        "quality_contract_ref": {
            "path": quality_contract_relative,
            "file_sha256": sha256_file(quality_contract_path),
        },
        "implementation": provenance.implementation_bundle(
            _IMPLEMENTATION_PATHS,
            root=ROOT,
        ),
        "parameters_frozen_before_derivation": True,
    }


def register(args: argparse.Namespace) -> None:
    trajectory_input_path = Path(args.trajectory_input).expanduser().resolve()
    quality_contract_path = _repo_path(args.quality_contract, "quality contract")
    payload = _registration_payload(
        trajectory_input_path,
        quality_contract_path,
        study_id=args.study_id,
        created_at=args.created_at,
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
    trajectory_input_path = Path(source["trajectory_input_path"]).resolve()
    if sha256_file(trajectory_input_path) != source["trajectory_input_file_sha256"]:
        raise ValueError("registered trajectory input hash differs")
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
    if (
        optimizer.get("anchor_budget_source")
        != "literature_informed_empirical_search_floor"
    ):
        raise ValueError("registered anchor budget source is unsupported")
    if optimizer.get("budget_selection") != "ascending_first_closed_loop_quality_pass":
        raise ValueError("registered budget selection rule is unsupported")
    if "anchor_budgets" in optimizer:
        raise ValueError("registered optimizer must not contain manual anchor budgets")
    expected_search_floor = empirical_search_floor(
        int(document["controlled_generation"]["num_steps"])
    )
    if (
        optimizer.get("search_floor_role")
        != "candidate_domain_floor_not_quality_guarantee"
        or optimizer.get("search_floor_formula") != EMPIRICAL_SEARCH_FLOOR_FORMULA
        or optimizer.get("search_floor_ratio") != EMPIRICAL_SEARCH_FLOOR_RATIO
        or optimizer.get("search_floor_anchor_budget") != expected_search_floor
        or optimizer.get("search_floor_reference") != EMPIRICAL_SEARCH_FLOOR_REFERENCE
        or optimizer.get("predictor_history_floor") != 2
        or optimizer.get("trajectory_observation_floor") != 3
        or optimizer.get("closed_loop_mechanism_floor") != 4
    ):
        raise ValueError("registered empirical search floor is invalid")
    if "hardware_budget" in document:
        raise ValueError("schedule registration must not contain a hardware-derived budget")
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

    expected_prompts = int(registration["source"]["trajectory_count"])
    segment_costs: dict[tuple[int, int, int], float] = {}
    for key, values in prompt_segment_costs.items():
        if len(values) != expected_prompts:
            raise RuntimeError("segment cost matrix is incomplete")
        segment_costs[key] = _nearest_rank(values, prompt_quantile)

    schedules = []
    frontier = _optimize_budget_frontier(
        segment_costs,
        num_steps=num_steps,
        warmup_steps=int(optimizer["warmup_steps"]),
        minimum_anchor_budget=int(optimizer["search_floor_anchor_budget"]),
        phase_boundary=int(optimizer["phase_boundary_step"]),
        middle_gap_cap=int(optimizer["middle_max_anchor_gap"]),
        tail_gap_cap=int(optimizer["tail_max_anchor_gap"]),
    )
    for budget, anchors, objective in frontier:
        path_segments = _materialized_path_segments(
            anchors,
            segment_costs,
            warmup_steps=int(optimizer["warmup_steps"]),
        )
        schedules.append(
            {
                "anchor_budget": int(budget),
                "static_anchor_steps": list(anchors),
                "objective_cost": objective,
                "segments": path_segments,
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
        "budget_frontier": {
            "candidate_domain": "budgets_at_or_above_empirical_search_floor",
            "qualification_order": "ascending_anchor_budget",
            "quality_selection_rule": "first_quality_pass",
            "speed_is_selection_input": False,
            "search_floor_formula": optimizer["search_floor_formula"],
            "search_floor_ratio": optimizer["search_floor_ratio"],
            "search_floor_anchor_budget": optimizer["search_floor_anchor_budget"],
            "search_floor_is_quality_guarantee": False,
            "feasible_anchor_budgets": [row["anchor_budget"] for row in schedules],
        },
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
    registration: Mapping[str, Any],
    derivation_path: Path,
    derivation: Mapping[str, Any],
) -> dict[str, Any]:
    optimizer = registration["optimizer"]
    budget = int(schedule["anchor_budget"])
    policy: dict[str, Any] = {
        "type": "phased_static",
        "num_steps": int(registration["controlled_generation"]["num_steps"]),
        "static_anchor_steps": list(schedule["static_anchor_steps"]),
        "warmup_steps": int(optimizer["warmup_steps"]),
        "cooldown_steps": int(optimizer["cooldown_steps"]),
        "require_final_anchor": bool(optimizer["require_final_anchor"]),
        "dynamic_budget": 0,
        "invalid_measurement_fail_closed": True,
    }
    candidate_id = f"quality-static-a{budget}-o1-index"
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
        document = _candidate_document(
            schedule=schedule,
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
                "family": "static_frontier",
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
    register_parser.add_argument("--trajectory-input", required=True)
    register_parser.add_argument("--quality-contract", required=True)
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
