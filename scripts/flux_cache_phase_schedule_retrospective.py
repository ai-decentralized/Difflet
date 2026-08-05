#!/usr/bin/env python3
"""Build the zero-hardware FLUX phase-schedule evidence retrospective."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.flux_cache_protocol import canonical_sha256  # noqa: E402

REPORT_SCHEMA = "difflet-flux-cache-phase-schedule-retrospective"
REPORT_SCHEMA_REVISION = 1
VQA_HARM_MARGIN = 0.25

DEFAULT_INPUTS = {
    "causal_label_result": ROOT
    / "benchmark/flux_cache/terminal-brake-causal-label-result.json",
    "step21_futility_result": ROOT
    / "benchmark/flux_cache/terminal-brake-step21-futility-result.json",
    "brake_intervention_pilot_result": ROOT
    / "benchmark/flux_cache/brake-intervention-pilot-result.json",
    "terminal_brake_followup_result": ROOT
    / "benchmark/flux_cache/terminal-brake-followup-result.json",
}

EXPECTED_SCHEMAS = {
    "causal_label_result": "difflet-flux-cache-terminal-brake-causal-label-result",
    "step21_futility_result": "difflet-flux-cache-terminal-brake-step21-futility-result",
    "brake_intervention_pilot_result": "difflet-flux-cache-brake-intervention-pilot-result",
    "terminal_brake_followup_result": "difflet-flux-cache-terminal-brake-followup-result",
}

REQUIRED_ARTIFACTS = {
    "causal_label_result": (
        "source_semantic",
        "timing_run",
        "timing_semantic",
        "completion_run",
        "completion_semantic",
    ),
    "step21_futility_result": (
        "source_quality",
        "source_semantic",
        "timing_run",
        "timing_semantic",
    ),
    "brake_intervention_pilot_result": (
        "run_result",
        "quality_input",
        "semantic_scores",
        "analysis",
    ),
    "terminal_brake_followup_result": (
        "run",
        "quality_input",
        "semantic_scores",
        "analysis",
    ),
}

# This taxonomy was assigned after outcomes were opened.  It is deliberately
# stored next to the code, checked against exact prompt text, and forbidden from
# gate or profile-selection use.
POSTHOC_SEMANTIC_TAGS = {
    "p010-s2": (
        "a round table set for six guests with exactly six plates, six glasses, "
        "and six folded napkins in alternating blue and white",
        ("counting", "attribute_binding", "alternation"),
    ),
    "p019-s2": (
        "four chefs piping four different frosting patterns onto four cakes "
        "arranged from smallest to largest",
        ("counting", "attribute_binding", "ordering"),
    ),
    "p020-s2": (
        "three astronauts tethered by thin white lines to a spacecraft, with one "
        "astronaut above it and two below it",
        ("counting", "spatial_relation", "relation_binding"),
    ),
    "p028-s2": (
        "five paper boats floating in a row, numbered one to five, with boat three "
        "passing under a tiny arched bridge",
        ("counting", "ordering", "spatial_relation"),
    ),
    "p029-s2": (
        "four planets aligned diagonally behind a silver satellite whose two thin "
        "solar panels extend horizontally",
        ("counting", "spatial_relation", "orientation"),
    ),
    "p030-s2": (
        "seven colored pencils threaded through separate loops of a white ribbon "
        "in rainbow order",
        ("counting", "ordering", "relation_binding"),
    ),
    "p031-s2": (
        "three horses inside a wooden fence and five sheep outside it, with a red "
        "gate positioned between the groups",
        ("counting", "spatial_relation", "group_binding"),
    ),
    "p033-s2": (
        "four acrobats forming a human pyramid with three people on the bottom and "
        "one person on top, all holding thin ribbons",
        ("counting", "spatial_relation", "group_binding"),
    ),
    "p035-s2": (
        "two firefighters aiming crossed water hoses above three orange traffic "
        "cones in front of a brick building",
        ("counting", "spatial_relation", "crossing_relation"),
    ),
    "p037-s2": (
        "three foxes walking behind a fallen log while four rabbits sit in front of "
        "it among small white flowers",
        ("counting", "spatial_relation", "group_binding"),
    ),
    "p038-s2": (
        "six chess pieces arranged left to right as king, queen, bishop, knight, "
        "rook, and pawn on separate black squares",
        ("counting", "ordering", "identity_binding"),
    ),
    "p045-s2": (
        "three fishing boats towing separate nets while four seabirds fly in a line "
        "between the boats and the horizon",
        ("counting", "spatial_relation", "relation_binding"),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {name} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return document


def _load_hashed_result(path: Path, name: str) -> dict[str, Any]:
    document = _load_json(path, name)
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != canonical_sha256(payload):
        raise ValueError(f"{name} content sha256 does not match")
    if document.get("schema") != EXPECTED_SCHEMAS[name]:
        raise ValueError(f"{name} schema is unsupported")
    return document


def _resolve_binding_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _verified_artifact(
    source_name: str,
    role: str,
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = result.get("artifacts", {}).get(role)
    if not isinstance(binding, dict) or set(binding) < {"path", "file_sha256"}:
        raise ValueError(f"{source_name} artifact {role} binding is incomplete")
    path = _resolve_binding_path(str(binding["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"{source_name} artifact {role} does not exist: {path}")
    observed = sha256_file(path)
    if observed != binding["file_sha256"]:
        raise ValueError(
            f"{source_name} artifact {role} sha256 mismatch: "
            f"expected {binding['file_sha256']}, got {observed}"
        )
    audit = {
        "input": source_name,
        "role": role,
        "path": str(path),
        "file_sha256": observed,
        "required_for_retrospective": True,
        "verified": True,
    }
    return audit, _load_json(path, f"{source_name} artifact {role}")


def _input_binding(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    try:
        relative = str(path.resolve().relative_to(ROOT))
    except ValueError:
        relative = str(path.resolve())
    return {
        "path": relative,
        "file_sha256": sha256_file(path),
        "content_sha256": document["sha256"],
        "schema": document["schema"],
        "schema_revision": document["schema_revision"],
    }


def _comparison_index(
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in comparisons:
        key = (str(row["candidate_id"]), str(row["sample_id"]))
        if key in result:
            raise ValueError(f"duplicate semantic comparison {key}")
        result[key] = row
    return result


def _metric_names(semantic: Mapping[str, Any]) -> list[str]:
    rows = semantic.get("comparisons")
    if not isinstance(rows, list) or not rows:
        raise ValueError("semantic report has no comparisons")
    names = set(rows[0]["baseline_scores"]) | set(rows[0]["candidate_scores"])
    for row in rows:
        if set(row["baseline_scores"]) | set(row["candidate_scores"]) != names:
            raise ValueError("semantic comparisons have inconsistent metric coverage")
    return sorted(str(name) for name in names)


def _stress_rescue_matrix(
    timing_run: Mapping[str, Any], timing_semantic: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    samples = timing_run.get("samples")
    comparisons = timing_semantic.get("comparisons")
    if not isinstance(samples, list) or len(samples) != 12:
        raise ValueError("timing run must contain the 12 registered source failures")
    if not isinstance(comparisons, list):
        raise ValueError("timing semantic report has no comparisons")
    if _metric_names(timing_semantic) != ["vqa_score"]:
        raise ValueError("timing semantic report must be VQAScore-only")
    index = _comparison_index(comparisons)
    observed_ids = {str(row["sample_id"]) for row in samples}
    if observed_ids != set(POSTHOC_SEMANTIC_TAGS):
        raise ValueError("post-hoc semantic taxonomy does not match timing requests")

    per_request: list[dict[str, Any]] = []
    expected_steps = [7, 13, 21, 29, 37]
    for sample in sorted(samples, key=lambda row: str(row["sample_id"])):
        sample_id = str(sample["sample_id"])
        expected_prompt, tags = POSTHOC_SEMANTIC_TAGS[sample_id]
        if sample["prompt"] != expected_prompt:
            raise ValueError(f"post-hoc semantic taxonomy prompt mismatch for {sample_id}")
        runs = sample.get("terminal_runs")
        if not isinstance(runs, list):
            raise ValueError(f"timing run has no terminal branches for {sample_id}")
        steps = [int(run["terminal_step"]) for run in runs]
        if steps != expected_steps:
            raise ValueError(f"timing grid differs for {sample_id}")
        outcomes: list[dict[str, Any]] = []
        for run in runs:
            step = int(run["terminal_step"])
            if run.get("prefix_matches_continue_cache") is not True:
                raise ValueError(f"prefix identity failed for {sample_id} at step {step}")
            comparison = index[(f"terminal-step-{step:02d}", sample_id)]
            baseline = float(comparison["baseline_scores"]["vqa_score"])
            terminal = float(comparison["candidate_scores"]["vqa_score"])
            terminal_harm = baseline - terminal
            source_harm = float(sample["source_vqa_harm"])
            outcomes.append(
                {
                    "terminal_step": step,
                    "prefix_matches_continue_cache": True,
                    "baseline_vqa": baseline,
                    "continue_cache_vqa_harm": source_harm,
                    "terminal_vqa": terminal,
                    "terminal_vqa_harm": terminal_harm,
                    "brake_benefit": source_harm - terminal_harm,
                    "rescued": terminal_harm <= VQA_HARM_MARGIN,
                }
            )
        rescued_steps = [row["terminal_step"] for row in outcomes if row["rescued"]]
        per_request.append(
            {
                "sample_id": sample_id,
                "seed": int(sample["seed"]),
                "prompt": sample["prompt"],
                "semantic_tags": list(tags),
                "continue_cache_vqa_harm": float(sample["source_vqa_harm"]),
                "outcomes": outcomes,
                "latest_tested_rescued_step": max(rescued_steps) if rescued_steps else None,
            }
        )

    curve: list[dict[str, Any]] = []
    for step in expected_steps:
        step_rows = [
            next(row for row in request["outcomes"] if row["terminal_step"] == step)
            for request in per_request
        ]
        rescued = sum(bool(row["rescued"]) for row in step_rows)
        curve.append(
            {
                "terminal_step": step,
                "source_failure_count": len(step_rows),
                "rescued_count": rescued,
                "R": rescued / len(step_rows),
                "mean_brake_benefit": sum(row["brake_benefit"] for row in step_rows)
                / len(step_rows),
            }
        )

    step29_unrescued = [
        request
        for request in per_request
        if not next(
            row["rescued"]
            for row in request["outcomes"]
            if row["terminal_step"] == 29
        )
    ]
    shared_tags = sorted(
        set.intersection(*(set(request["semantic_tags"]) for request in step29_unrescued))
    )
    all_tags = sorted(
        {tag for request in per_request for tag in request["semantic_tags"]}
    )
    diagnosis = {
        "taxonomy_origin": "posthoc_manual_prompt_text_multilabel",
        "gate_use_permitted": False,
        "profile_selection_use_permitted": False,
        "sample_size_warning": "Four step-29 non-rescues support description only.",
        "step29_unrescued_sample_ids": [row["sample_id"] for row in step29_unrescued],
        "step29_unrescued_shared_tags": shared_tags,
        "tag_counts": [
            {
                "tag": tag,
                "all_source_failures": sum(
                    tag in request["semantic_tags"] for request in per_request
                ),
                "step29_unrescued": sum(
                    tag in request["semantic_tags"] for request in step29_unrescued
                ),
            }
            for tag in all_tags
        ],
    }
    return per_request, curve, diagnosis


def _step29_introduced_table(
    completion_run: Mapping[str, Any], completion_semantic: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = completion_run.get("samples")
    comparisons = completion_semantic.get("comparisons")
    if not isinstance(samples, list) or len(samples) != 36:
        raise ValueError("step-29 completion must contain 36 continue-cache passes")
    if not isinstance(comparisons, list) or len(comparisons) != 36:
        raise ValueError("step-29 semantic report must contain 36 comparisons")
    if _metric_names(completion_semantic) != ["vqa_score"]:
        raise ValueError("step-29 completion must remain VQAScore-only")
    index = _comparison_index(comparisons)
    table: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda row: str(row["sample_id"])):
        sample_id = str(sample["sample_id"])
        if sample.get("prefix_matches_continue_cache") is not True:
            raise ValueError(f"step-29 prefix identity failed for {sample_id}")
        comparison = index[("terminal-step-29", sample_id)]
        baseline = float(comparison["baseline_scores"]["vqa_score"])
        terminal = float(comparison["candidate_scores"]["vqa_score"])
        terminal_harm = baseline - terminal
        source_harm = float(sample["source_vqa_harm"])
        source_failed = source_harm > VQA_HARM_MARGIN
        if source_failed:
            raise ValueError(f"step-29 completion source is not a VQA pass: {sample_id}")
        table.append(
            {
                "sample_id": sample_id,
                "seed": int(sample["seed"]),
                "prompt": sample["prompt"],
                "continue_cache_vqa_harm": source_harm,
                "terminal_vqa_harm": terminal_harm,
                "brake_benefit": source_harm - terminal_harm,
                "introduced_vqa_failure": terminal_harm > VQA_HARM_MARGIN,
                "introduced_contract_failure": None,
                "contract_status": "not_evaluable_missing_image_reward",
            }
        )
    introduced = [row["sample_id"] for row in table if row["introduced_vqa_failure"]]
    return table, {
        "terminal_step": 29,
        "full_population_request_count": 48,
        "source_failure_request_count": 12,
        "source_pass_control_count": len(table),
        "introduced_vqa_failure_count": len(introduced),
        "introduced_vqa_failure_sample_ids": introduced,
        "introduced_contract_failure_count": None,
        "metric_coverage": ["vqa_score"],
        "contract_status": "not_evaluable_missing_image_reward",
    }


def _reconcile_registered_curve(
    recomputed: Sequence[Mapping[str, Any]],
    registered: Any,
) -> dict[str, Any]:
    if not isinstance(registered, list):
        raise ValueError("causal-label result has no registered rescue curve")
    registered_by_step = {int(row["terminal_step"]): row for row in registered}
    if set(registered_by_step) != {int(row["terminal_step"]) for row in recomputed}:
        raise ValueError("registered and recomputed rescue grids differ")
    rows = []
    for row in recomputed:
        step = int(row["terminal_step"])
        source = registered_by_step[step]
        registered_mean = float(source["mean_brake_benefit"])
        recomputed_mean = float(row["mean_brake_benefit"])
        rows.append(
            {
                "terminal_step": step,
                "registered_source_failure_count": int(source["source_failure_count"]),
                "recomputed_source_failure_count": int(row["source_failure_count"]),
                "registered_rescued_count": int(source["rescued_count"]),
                "recomputed_rescued_count": int(row["rescued_count"]),
                "registered_mean_brake_benefit": registered_mean,
                "recomputed_mean_brake_benefit": recomputed_mean,
                "mean_brake_benefit_difference": recomputed_mean - registered_mean,
            }
        )
    return {
        "basis": "recomputed_from_bound_per_request_semantic_artifacts",
        "all_source_and_rescue_counts_match": all(
            row["registered_source_failure_count"]
            == row["recomputed_source_failure_count"]
            and row["registered_rescued_count"] == row["recomputed_rescued_count"]
            for row in rows
        ),
        "all_mean_brake_benefits_match_exactly": all(
            row["mean_brake_benefit_difference"] == 0.0 for row in rows
        ),
        "rows": rows,
        "interpretation": (
            "Request-level recomputation is reported without replacing or silently "
            "rounding the registered historical summary."
        ),
    }


def _historical_control_table(
    pilot_analysis: Mapping[str, Any], followup_analysis: Mapping[str, Any]
) -> dict[str, Any]:
    pilot_rows = pilot_analysis.get("rows")
    followup_rows = followup_analysis.get("rows")
    if not isinstance(pilot_rows, list) or not isinstance(followup_rows, list):
        raise ValueError("historical intervention analyses have no rows")
    pilot_controls = []
    for row in pilot_rows:
        if row["prior_label"] != "target-candidate-pass-control":
            continue
        quality = row["quality"]
        pilot_controls.append(
            {
                "action": row["action"],
                "sample_id": row["sample_id"],
                "phase": row["phase"],
                "target_step": int(row["target_step"]),
                "continue_failed": bool(quality["continue_failed"]),
                "action_failed": bool(quality["action_failed"]),
                "introduced_contract_failure": (
                    not bool(quality["continue_failed"])
                    and bool(quality["action_failed"])
                ),
                "outcome": quality["outcome"],
            }
        )
    followup_controls = []
    for row in followup_rows:
        if row["prior_label"] != "target-candidate-pass-control":
            continue
        quality = row["quality"]
        followup_controls.append(
            {
                "action": "terminal_brake",
                "sample_id": row["sample_id"],
                "phase": row["phase"],
                "target_step": int(row["target_step"]),
                "continue_failed": bool(quality["continue_failed"]),
                "action_failed": bool(quality["terminal_failed"]),
                "introduced_contract_failure": (
                    not bool(quality["continue_failed"])
                    and bool(quality["terminal_failed"])
                ),
                "outcome": quality["outcome"],
            }
        )
    if len(pilot_controls) != 6 or len(followup_controls) != 3:
        raise ValueError("historical pass-control target counts differ from evidence")
    return {
        "metric_coverage": ["image_reward", "vqa_score"],
        "evidence_role": "descriptive_small_n_controls",
        "pilot_brake_and_recovery": pilot_controls,
        "terminal_followup": followup_controls,
        "introduced_contract_failure_count": sum(
            row["introduced_contract_failure"]
            for row in [*pilot_controls, *followup_controls]
        ),
    }


def _contradiction_adjudication(
    pressure_curve: Sequence[Mapping[str, Any]],
    followup_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    failure_rows = [
        row
        for row in followup_analysis["rows"]
        if row["prior_label"] == "target-candidate-vqa-failure"
    ]
    unique_requests = sorted({str(row["sample_id"]) for row in failure_rows})
    step21 = next(row for row in pressure_curve if row["terminal_step"] == 21)
    return {
        "opened_stage_oil_family": {
            "candidate_id": "adaptive-stage-oil-p30-e1p40-w6-i8-k12-o1-index",
            "unique_failure_request_count": len(unique_requests),
            "unique_failure_sample_ids": unique_requests,
            "terminal_intervention_target_count": len(failure_rows),
            "terminal_target_steps": sorted(
                {int(row["target_step"]) for row in failure_rows}
            ),
            "rescued_target_count": sum(
                row["quality"]["outcome"] == "rescue" for row in failure_rows
            ),
        },
        "extreme_pressure_family": {
            "candidate_id": "adaptive-vqa-stress-w6-i32-m16-k40-o1-index",
            "unique_failure_request_count": int(step21["source_failure_count"]),
            "terminal_step": 21,
            "rescued_request_count": int(step21["rescued_count"]),
        },
        "confounds": [
            "profile family and anchor distribution differ",
            "prompt and seed populations differ",
            "two unique requests with six phase targets are not six independent failures",
            "the stage-oil terminal grid begins at step 15 while the pressure grid includes earlier interventions",
        ],
        "allowed_conclusion": (
            "Observed rescue horizon depends on the profile family, failure mechanism, "
            "and sampled request distribution; neither historical curve is a deployable "
            "or model-wide horizon."
        ),
        "forbidden_conclusions": [
            "step 21 is a universal last fully rescuable point",
            "all non-pressure failures are irrecoverable by step 15",
            "the profile family alone caused the difference",
            "historical curves may substitute for the registered near-frontier A2 probe",
        ],
    }


def build_report(
    input_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    paths = dict(DEFAULT_INPUTS if input_paths is None else input_paths)
    if set(paths) != set(DEFAULT_INPUTS):
        raise ValueError("A1 requires exactly the four frozen input roles")
    inputs = {
        name: _load_hashed_result(Path(paths[name]).resolve(), name)
        for name in DEFAULT_INPUTS
    }
    input_bindings = {
        name: _input_binding(Path(paths[name]).resolve(), inputs[name])
        for name in DEFAULT_INPUTS
    }

    artifact_audit: list[dict[str, Any]] = []
    artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    for source_name, roles in REQUIRED_ARTIFACTS.items():
        for role in roles:
            audit, document = _verified_artifact(source_name, role, inputs[source_name])
            artifact_audit.append(audit)
            artifacts[(source_name, role)] = document

    timing_run = artifacts[("causal_label_result", "timing_run")]
    timing_semantic = artifacts[("causal_label_result", "timing_semantic")]
    completion_run = artifacts[("causal_label_result", "completion_run")]
    completion_semantic = artifacts[("causal_label_result", "completion_semantic")]
    pilot_analysis = artifacts[("brake_intervention_pilot_result", "analysis")]
    followup_analysis = artifacts[("terminal_brake_followup_result", "analysis")]

    per_request, curve, diagnosis = _stress_rescue_matrix(
        timing_run, timing_semantic
    )
    step29_table, step29_summary = _step29_introduced_table(
        completion_run, completion_semantic
    )
    historical_controls = _historical_control_table(
        pilot_analysis, followup_analysis
    )
    futility = inputs["step21_futility_result"]

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_revision": REPORT_SCHEMA_REVISION,
        "study_id": "flux-cache-phase-schedule-a1-historical-retrospective",
        "status": "complete_historical_descriptive",
        "evidence_role": {
            "hardware_calls": 0,
            "opened_historical_data": True,
            "serving_claim_permitted": False,
            "profile_selection_permitted": False,
            "a2_or_a3_parameter_changes_permitted": False,
        },
        "inputs": input_bindings,
        "required_artifact_audit": artifact_audit,
        "quality_rule": {
            "metric": "vqa_score",
            "harm": "baseline_vqa_minus_candidate_vqa",
            "failure": "harm_strictly_greater_than_margin",
            "margin": VQA_HARM_MARGIN,
        },
        "extreme_pressure_rescue": {
            "candidate_id": "adaptive-vqa-stress-w6-i32-m16-k40-o1-index",
            "metric_coverage": ["vqa_score"],
            "request_count": len(per_request),
            "per_request": per_request,
            "R_by_terminal_step": curve,
            "registered_summary_reconciliation": _reconcile_registered_curve(
                curve,
                inputs["causal_label_result"]["failure_rescue_timing"],
            ),
            "semantic_diagnosis": diagnosis,
        },
        "introduced_failures": {
            "step29_full_population": {
                "summary": step29_summary,
                "source_pass_controls": step29_table,
            },
            "opened_stage_oil_controls": historical_controls,
        },
        "step21_signal_futility": {
            "source_failure_count": futility["causal_reduction"][
                "source_failure_count"
            ],
            "source_failure_rescued_count": futility["causal_reduction"][
                "source_failure_rescued_count"
            ],
            "best_opened_scalar_auc": futility["bounded_opened_screen"][
                "best_fixed_scalar"
            ]["roc_auc"],
            "full_recall_false_brake_count": futility["bounded_opened_screen"][
                "best_fixed_scalar"
            ]["full_recall_false_brake_count"],
            "full_recall_passing_count": futility["bounded_opened_screen"][
                "best_fixed_scalar"
            ]["full_recall_passing_count"],
            "gate_passed": futility["hardware_futility_gate"]["gate_passed"],
            "interpretation": (
                "A1 does not reopen the rejected historical online-signal route."
            ),
        },
        "contradiction_adjudication": _contradiction_adjudication(
            curve, followup_analysis
        ),
        "decision": {
            "a1_complete": True,
            "a2_still_required": True,
            "historical_t_full_or_t_dead_may_be_used_for_mask_generation": False,
            "reason": (
                "The retrospective standardizes historical evidence but the two "
                "curves use different profile families and request populations."
            ),
        },
        "implementation": {
            "path": "scripts/flux_cache_phase_schedule_retrospective.py",
            "file_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    report["sha256"] = canonical_sha256(report)
    return report


def load_report(path: Path) -> dict[str, Any]:
    report = _load_json(path, "phase-schedule A1 retrospective")
    payload = {key: value for key, value in report.items() if key != "sha256"}
    if report.get("sha256") != canonical_sha256(payload):
        raise ValueError("phase-schedule A1 retrospective sha256 does not match")
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("schema_revision") != REPORT_SCHEMA_REVISION
    ):
        raise ValueError("phase-schedule A1 retrospective schema is unsupported")
    return report


def render_memo(report: Mapping[str, Any]) -> str:
    curve = report["extreme_pressure_rescue"]["R_by_terminal_step"]
    diagnosis = report["extreme_pressure_rescue"]["semantic_diagnosis"]
    introduced = report["introduced_failures"]["step29_full_population"]["summary"]
    adjudication = report["contradiction_adjudication"]
    reconciliation = report["extreme_pressure_rescue"][
        "registered_summary_reconciliation"
    ]
    curve_rows = "\n".join(
        f"| {row['terminal_step']} | {row['rescued_count']}/{row['source_failure_count']} "
        f"| {row['R']:.4f} | {row['mean_brake_benefit']:.6f} |"
        for row in curve
    )
    unrescued = ", ".join(diagnosis["step29_unrescued_sample_ids"])
    shared_tags = ", ".join(diagnosis["step29_unrescued_shared_tags"])
    introduced_ids = ", ".join(introduced["introduced_vqa_failure_sample_ids"])
    stage = adjudication["opened_stage_oil_family"]
    pressure = adjudication["extreme_pressure_family"]
    return f"""# FLUX cache phase-schedule A1 historical retrospective

Status: complete. Machine-readable result SHA-256: `{report['sha256']}`.

## Evidence boundary

- This retrospective reads existing JSON only and makes zero hardware calls. Every required artifact passed its file-hash check.
- The historical outcomes were already opened. This report may standardize descriptive evidence only; it may not select a profile, change A2/A3, or support a serving claim.
- The pressure timing and step-29 completion datasets contain VQAScore only. An introduced failure in the latter is a VQA failure, not a full ImageReward-plus-VQA contract failure.
- Semantic tags are post-hoc manual multilabel annotations. They describe four step-29 non-rescues and are forbidden from every gate.

## Per-request rescue curve for the pressure profile

Candidate: `adaptive-vqa-stress-w6-i32-m16-k40-o1-index`; 12 independent source failures.

| terminal step | rescued | R(t) | mean brake benefit |
|---:|---:|---:|---:|
{curve_rows}

Step-29 non-rescues: {unrescued}. Their shared post-hoc tag is: {shared_tags}. This says only that all four cases in this failure-enriched queue contain counting constraints. The sample is too small, and the queue is already compositionally enriched, so it does not establish a general semantic-category effect.

The per-request recomputation exactly matches all historical source and rescue counts. Mean brake benefit matches exactly at steps 7, 13, 21, and 29. At step 37 the per-request recomputation is `{reconciliation['rows'][-1]['recomputed_mean_brake_benefit']:.12f}`, while the historical result records `{reconciliation['rows'][-1]['registered_mean_brake_benefit']:.12f}`, a difference of `{reconciliation['rows'][-1]['mean_brake_benefit_difference']:.12f}`. This report preserves both values, uses the bound per-request semantic artifact as the recomputation source, and does not silently rewrite the historical file.

## Introduced failure

Among the 36 step-29 continue-cache VQA pass controls, terminal braking introduced {introduced['introduced_vqa_failure_count']} VQA failures: {introduced_ids}. Because ImageReward was not rescored, the full-contract introduced-failure count is `null`.

The light brake and recovery controls in the historical stage-oil pilot, plus the three terminal-followup controls, have both contract metrics and contain no observed introduced contract failure. Their sample size supports description only, not a safety claim.

## Contradiction adjudication

- The stage-oil family contains {stage['unique_failure_request_count']} independent failure requests repeated across {stage['terminal_intervention_target_count']} phase targets, with zero rescued targets. Calling these "six failures" is not an independence-correct description.
- The extreme-pressure family rescues {pressure['rescued_request_count']} of {pressure['unique_failure_request_count']} independent failure requests at step {pressure['terminal_step']}.
- Profile, prompt/seed population, and intervention grid all change together. The difference cannot be attributed to one factor.

The only allowed conclusion is: {adjudication['allowed_conclusion']}

Therefore A1 supplies no `t_full` or `t_dead` for mask generation. The registered near-frontier A2 probe remains required with its existing futility rules.
"""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark/flux_cache/phase-schedule-a1-retrospective.json",
    )
    parser.add_argument(
        "--memo",
        type=Path,
        default=ROOT / "docs/flux-cache-phase-schedule-a1-retrospective.md",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report()
        _write_text(
            args.output.expanduser().resolve(),
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        _write_text(args.memo.expanduser().resolve(), render_memo(report))
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"[phase-schedule-a1] result={args.output} sha256={report['sha256']}")
    print(f"[phase-schedule-a1] memo={args.memo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
