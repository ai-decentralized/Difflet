#!/usr/bin/env python3
"""Select, collect, and analyze registered phase-schedule interventions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.pipeline.cache import (  # noqa: E402
    AdaptiveAnchorPolicy,
    CacheRunner,
    CacheSession,
    QualityRecoveryConfig,
    QualityRecoveryGuard,
    TaylorSeerPredictor,
    TeaCacheControllerAdapter,
)
from scripts.automatic_quality_contract import (  # noqa: E402
    load_semantic_report,
    semantic_source,
    validate_generation_identity,
)
from scripts.flux_cache_brake_intervention import (  # noqa: E402
    _git_identity,
    tensor_sha256,
)
from scripts.flux_cache_phase_schedule_registration import (  # noqa: E402
    load_registration,
    sha256_file,
)
from scripts.flux_cache_profile_confirmation import (  # noqa: E402
    _validate_metric_identity,
)
from scripts.flux_cache_protocol import canonical_sha256, load_prompt_suite  # noqa: E402

SELECTION_SCHEMA = "difflet-flux-cache-phase-schedule-selection"
SELECTION_SCHEMA_REVISION = 1
RUN_SCHEMA = "difflet-flux-cache-phase-schedule-intervention-run"
RUN_SCHEMA_REVISION = 1
TERMINAL_ANALYSIS_SCHEMA = "difflet-flux-cache-phase-schedule-terminal-analysis"
TERMINAL_ANALYSIS_SCHEMA_REVISION = 1
HORIZON_SCHEMA = "difflet-flux-cache-phase-schedule-horizon"
HORIZON_SCHEMA_REVISION = 1
HARDWARE_ACK = "I am running the offline FLUX phase schedule interventions"


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {name} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return document


def _write_json(path: Path, document: Mapping[str, Any], *, add_digest: bool) -> None:
    payload = dict(document)
    if add_digest:
        if "sha256" in payload:
            raise ValueError("hashed JSON must not already contain sha256")
        payload["sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_hashed(path: Path, *, schema: str, revision: int, name: str) -> dict[str, Any]:
    document = _load_json(path, name)
    digest = document.get("sha256")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if digest != canonical_sha256(payload):
        raise ValueError(f"{name} sha256 does not match its contents")
    if document.get("schema") != schema or document.get("schema_revision") != revision:
        raise ValueError(f"{name} schema is unsupported")
    return document


def _registration_binding(path: Path, registration: Mapping[str, Any]) -> dict[str, str]:
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "content_sha256": str(registration["sha256"]),
    }


def _source_binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "file_sha256": sha256_file(path)}


def _quality_failed(delta: Mapping[str, Any], margins: Mapping[str, Any]) -> bool:
    return any(-float(delta[metric]) > float(margins[metric]) for metric in margins)


def _vqa_failed(delta: Mapping[str, Any], margin: float) -> bool:
    return -float(delta["vqa_score"]) > margin


def _comparison_index(
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in comparisons:
        key = (str(row["candidate_id"]), str(row["sample_id"]))
        if key in result:
            raise ValueError(f"duplicate semantic comparison: {key}")
        result[key] = dict(row)
    return result


def _quality_comparison_index(
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in comparisons:
        key = (str(row["candidate_id"]), str(row["sample_id"]))
        if key in result:
            raise ValueError(f"duplicate quality comparison: {key}")
        result[key] = dict(row)
    return result


def _validate_source_evidence(
    registration: Mapping[str, Any], semantic_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    semantic = load_semantic_report(semantic_path)
    quality, selection, quality_path = semantic_source(semantic)
    validate_generation_identity(quality, registration["controlled_generation"])
    if quality.get("protocol", {}).get("timing", {}).get("pipeline_warmup_enabled") is not True:
        raise ValueError("source collection did not use the registered pipeline warmup")
    _validate_metric_identity(semantic["metrics"], registration["semantic_metrics"])
    prompt_binding = registration["prompt_suite"]
    if (
        selection.get("split") != prompt_binding["split"]
        or selection.get("sha256") != prompt_binding["split_sha256"]
    ):
        raise ValueError("source semantic report uses a different prompt split")
    expected_candidates = [row["candidate_id"] for row in registration["source_profiles"]]
    observed_candidates = [row["candidate_id"] for row in quality.get("candidates", [])]
    if observed_candidates != expected_candidates:
        raise ValueError("source quality candidate order differs from registration")
    comparisons = semantic.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("source semantic report has no comparisons")
    expected_count = registration["source_collection"]["candidate_request_count"]
    if len(comparisons) != expected_count:
        raise ValueError("source semantic comparison count differs from registration")
    return semantic, quality, quality_path


def _prompt_categories(registration: Mapping[str, Any]) -> dict[int, str]:
    binding = registration["prompt_suite"]
    suite_path = (ROOT / binding["path"]).resolve()
    selection = load_prompt_suite(suite_path, binding["split"])
    rows = selection.descriptor["prompts"]
    return {index: str(row["category"]) for index, row in enumerate(rows)}


def build_selection(
    registration_path: Path,
    source_semantic_path: Path,
) -> dict[str, Any]:
    registration = load_registration(registration_path)
    semantic, quality, quality_path = _validate_source_evidence(
        registration, source_semantic_path
    )
    margins = registration["quality_contract"]["margins"]
    vqa_margin = float(margins["vqa_score"])
    profile_ids = [row["candidate_id"] for row in registration["source_profiles"]]
    profile_order = {candidate_id: index for index, candidate_id in enumerate(profile_ids)}
    categories = _prompt_categories(registration)
    semantic_rows = _comparison_index(semantic["comparisons"])
    quality_rows = _quality_comparison_index(quality["comparisons"])

    failures: list[dict[str, Any]] = []
    passing: list[dict[str, Any]] = []
    for (candidate_id, sample_id), row in semantic_rows.items():
        if candidate_id not in profile_order:
            raise ValueError(f"unregistered source candidate in semantic report: {candidate_id}")
        quality_row = quality_rows[(candidate_id, sample_id)]
        prompt_index = int(row["prompt_index"])
        record = {
            "evaluation_id": f"profile-{profile_order[candidate_id]}::{sample_id}",
            "candidate_id": candidate_id,
            "registered_profile_order": profile_order[candidate_id],
            "sample_id": sample_id,
            "prompt_index": prompt_index,
            "semantic_category": categories[prompt_index],
            "seed": int(row["seed"]),
            "prompt": str(row["prompt"]),
            "baseline": dict(quality_row["baseline"]),
            "continue_cache": dict(quality_row["candidate"]),
            "source_candidate_minus_baseline": dict(row["candidate_minus_baseline"]),
        }
        if _vqa_failed(row["candidate_minus_baseline"], vqa_margin):
            failures.append(record)
        elif not _quality_failed(row["candidate_minus_baseline"], margins):
            passing.append(record)

    failures.sort(key=lambda row: (row["sample_id"], row["registered_profile_order"]))
    minimum = int(registration["source_selection"]["minimum_source_failures"])
    maximum = int(registration["source_selection"]["maximum_source_failures"])
    selected_failures = failures[:maximum]
    status = "ready_for_terminal_horizon"
    reason = None
    selected_controls: list[dict[str, Any]] = []
    if len(failures) < minimum:
        status = "insufficient_source_failures"
        reason = f"observed {len(failures)} VQA source failures; registration requires {minimum}"
    else:
        excluded_sample_ids = {row["sample_id"] for row in selected_failures}
        available = [row for row in passing if row["sample_id"] not in excluded_sample_ids]
        for failure in selected_failures:
            matches = [
                row
                for row in available
                if row["candidate_id"] == failure["candidate_id"]
                and row["semantic_category"] == failure["semantic_category"]
            ]
            matches.sort(key=lambda row: row["sample_id"])
            if not matches:
                status = "insufficient_matched_controls"
                reason = (
                    "no unused exact profile/category control for "
                    f"{failure['evaluation_id']}"
                )
                selected_controls = []
                break
            chosen = matches[0]
            selected_controls.append(chosen)
            available = [
                row for row in available if row["sample_id"] != chosen["sample_id"]
            ]

    return {
        "schema": SELECTION_SCHEMA,
        "schema_revision": SELECTION_SCHEMA_REVISION,
        "study_id": registration["study_id"],
        "status": status,
        "reason": reason,
        "registration": _registration_binding(registration_path, registration),
        "source_quality": _source_binding(quality_path),
        "source_semantic": _source_binding(source_semantic_path),
        "source_failure_count": len(failures),
        "source_pass_count": len(passing),
        "selected_failures": selected_failures,
        "selected_controls": selected_controls,
        "selection_labels_opened": True,
        "serving_claim": False,
    }


def load_selection(path: Path, registration_path: Path) -> dict[str, Any]:
    selection = _load_hashed(
        path,
        schema=SELECTION_SCHEMA,
        revision=SELECTION_SCHEMA_REVISION,
        name="phase-schedule selection",
    )
    registration = load_registration(registration_path)
    binding = selection["registration"]
    if (
        binding["file_sha256"] != sha256_file(registration_path)
        or binding["content_sha256"] != registration["sha256"]
    ):
        raise ValueError("selection is not bound to this registration")
    if selection["study_id"] != registration["study_id"]:
        raise ValueError("selection study id differs from registration")
    for name in ("source_quality", "source_semantic"):
        source = selection[name]
        source_path = Path(source["path"]).resolve()
        if not source_path.is_file() or sha256_file(source_path) != source["file_sha256"]:
            raise ValueError(f"selection {name} binding is invalid")
    if selection["status"] == "ready_for_terminal_horizon":
        expected = registration["source_selection"]["maximum_source_failures"]
        if (
            len(selection["selected_failures"]) != expected
            or len(selection["selected_controls"]) != expected
        ):
            raise ValueError("ready selection does not contain six failures and controls")
    return selection


class RegisteredTerminalPolicy(AdaptiveAnchorPolicy):
    """Follow the source profile prefix, then permanently disable caching."""

    def __init__(self, config: Any, *, terminal_step: int) -> None:
        self.terminal_step = int(terminal_step)
        if self.terminal_step < int(config.warmup_steps):
            raise ValueError("terminal step precedes source-profile warmup")
        super().__init__(config)

    def reset(self) -> None:
        super().reset()
        self._terminal_applied = False

    def should_skip(self, context: Any, history: Any, observation: Any) -> bool:
        if context.step_index >= self.terminal_step and not self._terminal_applied:
            self._disable()
            self._terminal_applied = True
        return super().should_skip(context, history, observation)

    def observe_anchor(
        self, context: Any, output: Any, history: Any, observation: Any
    ) -> None:
        del output, history, observation
        if context.step_index >= self.terminal_step and not self._terminal_applied:
            # An independent recovery guard may force the terminal step before
            # the policy is queried. That step is already real; disable from
            # the immediately following step to preserve terminal semantics.
            self._disable()
            self._terminal_applied = True

    def validate_complete(self) -> None:
        if not self._terminal_applied:
            raise RuntimeError("registered terminal intervention was never applied")

    def stats(self) -> dict[str, Any]:
        result = super().stats()
        result.update(
            {
                "registered_terminal_step": self.terminal_step,
                "registered_terminal_applied": self._terminal_applied,
            }
        )
        return result


class RegisteredRepairBurstPolicy(AdaptiveAnchorPolicy):
    """Force a registered consecutive-real-step burst, then resume the source policy."""

    def __init__(self, config: Any, *, start_step: int, real_steps: int) -> None:
        self.start_step = int(start_step)
        self.real_steps = int(real_steps)
        if self.start_step < int(config.warmup_steps) or self.real_steps < 1:
            raise ValueError("repair burst must start after warmup and contain real steps")
        super().__init__(config)

    def reset(self) -> None:
        super().reset()
        self._burst_applied_steps: set[int] = set()

    def should_skip(self, context: Any, history: Any, observation: Any) -> bool:
        if self.start_step <= context.step_index < self.start_step + self.real_steps:
            return False
        return super().should_skip(context, history, observation)

    def observe_anchor(
        self, context: Any, output: Any, history: Any, observation: Any
    ) -> None:
        del output, history, observation
        if self.start_step <= context.step_index < self.start_step + self.real_steps:
            self._burst_applied_steps.add(int(context.step_index))

    def validate_complete(self) -> None:
        expected = set(range(self.start_step, self.start_step + self.real_steps))
        if self._burst_applied_steps != expected:
            raise RuntimeError("registered repair burst was not fully applied")

    def stats(self) -> dict[str, Any]:
        result = super().stats()
        result.update(
            {
                "registered_repair_start_step": self.start_step,
                "registered_repair_real_steps": self.real_steps,
                "registered_repair_applied_steps": sorted(self._burst_applied_steps),
            }
        )
        return result


def _build_adapter(arm: Any, policy: AdaptiveAnchorPolicy, *, num_steps: int):
    recovery = QualityRecoveryGuard(
        QualityRecoveryConfig(
            warmup_steps=arm.config.warmup_steps,
            cooldown_steps=arm.config.cooldown_steps,
            require_final_anchor=arm.config.require_final_anchor,
        )
    )
    runner = CacheRunner(
        policy,
        TaylorSeerPredictor(order=arm.order, coord=arm.coord),
        recovery=recovery,
    )
    return TeaCacheControllerAdapter(
        CacheSession(
            runner,
            num_steps=num_steps,
            configuration_source="offline-phase-schedule-intervention",
        )
    )


def _resolve_source_artifact(source_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (source_root / path).resolve()
    if not resolved.is_file():
        raise ValueError(f"source artifact does not exist: {resolved}")
    return resolved


def _unit_id(row: Mapping[str, Any]) -> str:
    return str(row["evaluation_id"]).replace("::", "-")


def _intervention_id(kind: str, source_candidate: str, *values: int) -> str:
    suffix = "-".join(f"{value:02d}" for value in values)
    return f"{source_candidate}--{kind}-{suffix}"


def _load_pipeline(registration: Mapping[str, Any], args: argparse.Namespace):
    from scripts.collect_flux_cache_ab import _load_pipeline as load_pipeline

    generation = registration["controlled_generation"]
    return load_pipeline(
        SimpleNamespace(
            model_id=generation["model_id"],
            model_revision=generation["model_revision"],
            tp_degree=generation["tp_degree"],
            dtype=generation["dtype"],
            compile_cache_dir=args.compile_cache_dir,
            height=generation["height"],
            width=generation["width"],
            force_compile=False,
            skip_warmup=bool(args.skip_warmup),
            collect_online_signals=False,
        )
    )


def _collect_interventions(
    *,
    registration_path: Path,
    selection_path: Path,
    terminal_analysis_path: Path | None,
    output_root: Path,
    mode: str,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    if not args.allow_hardware or args.foreground_ack != HARDWARE_ACK:
        raise RuntimeError(
            f"hardware execution requires --allow-hardware and --foreground-ack {HARDWARE_ACK!r}"
        )
    if args.skip_warmup:
        raise ValueError("registered phase-schedule interventions require pipeline warmup")
    registration = load_registration(registration_path)
    selection = load_selection(selection_path, registration_path)
    if selection["status"] != "ready_for_terminal_horizon":
        raise ValueError("selection status does not permit intervention collection")
    terminal_analysis = None
    if mode == "repair":
        if terminal_analysis_path is None:
            raise ValueError("repair collection requires terminal analysis")
        terminal_analysis = _load_hashed(
            terminal_analysis_path,
            schema=TERMINAL_ANALYSIS_SCHEMA,
            revision=TERMINAL_ANALYSIS_SCHEMA_REVISION,
            name="terminal analysis",
        )
        if terminal_analysis["status"] != "usable":
            raise ValueError("terminal analysis status does not permit repair-depth collection")
        if terminal_analysis["selection"]["file_sha256"] != sha256_file(selection_path):
            raise ValueError("terminal analysis is not bound to the registered selection")
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")
    output_root.mkdir(parents=True)

    from scripts.collect_flux_cache_ab import load_adaptive_candidate
    from scripts.flux_cache_causal_repair import _run_image_only

    source_quality_path = Path(selection["source_quality"]["path"]).resolve()
    source_quality = _load_json(source_quality_path, "source quality manifest")
    source_root = source_quality_path.parent
    source_rows = _quality_comparison_index(source_quality["comparisons"])
    profile_bindings = {
        row["candidate_id"]: row for row in registration["source_profiles"]
    }
    arm_cache = {
        candidate_id: load_adaptive_candidate((ROOT / binding["path"]).resolve())
        for candidate_id, binding in profile_bindings.items()
    }
    generation = registration["controlled_generation"]
    num_steps = int(generation["num_steps"])
    pipe = _load_pipeline(registration, args)
    flux_pipeline = pipe.app.pipe
    flux_pipeline._tc_record = False
    started = time.time()
    comparisons: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    units = [
        *(dict(row, source_role="failure") for row in selection["selected_failures"]),
        *(dict(row, source_role="control") for row in selection["selected_controls"]),
    ]
    for unit in units:
        candidate_id = unit["candidate_id"]
        source = source_rows[(candidate_id, unit["sample_id"])]
        baseline_path = _resolve_source_artifact(source_root, source["baseline"]["image"])
        trajectory_path = _resolve_source_artifact(
            source_root, source["candidate"]["trajectory"]
        )
        import torch

        source_trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=True)
        arm = arm_cache[candidate_id]
        if mode == "terminal":
            grid = [(int(step),) for step in registration["terminal_horizon"]["steps"]]
        else:
            grid = [
                (int(start), int(real_steps))
                for start in registration["repair_depth"]["start_steps"]
                for real_steps in registration["repair_depth"]["consecutive_real_steps"]
            ]
        for values in grid:
            if mode == "terminal":
                policy: AdaptiveAnchorPolicy = RegisteredTerminalPolicy(
                    arm.config, terminal_step=values[0]
                )
                intervention_id = _intervention_id("terminal", candidate_id, *values)
                label = f"terminal-{values[0]:02d}"
                prefix_step = values[0] - 1
            else:
                policy = RegisteredRepairBurstPolicy(
                    arm.config, start_step=values[0], real_steps=values[1]
                )
                intervention_id = _intervention_id("repair", candidate_id, *values)
                label = f"repair-{values[0]:02d}-{values[1]:02d}"
                prefix_step = values[0] - 1
            adapter = _build_adapter(arm, policy, num_steps=num_steps)
            flux_pipeline.teacache_controller = adapter
            destination = output_root / "artifacts" / _unit_id(unit) / f"{label}.png"
            run = _run_image_only(
                pipe,
                flux_pipeline,
                prompt=unit["prompt"],
                seed=int(unit["seed"]),
                num_steps=num_steps,
                height=int(generation["height"]),
                width=int(generation["width"]),
                guidance_scale=float(generation["guidance_scale"]),
                image_path=destination,
            )
            policy.validate_complete()
            actual_prefix_hash = tensor_sha256(flux_pipeline._tc_last_trajectory[prefix_step])
            source_prefix_hash = tensor_sha256(source_trajectory[prefix_step])
            if actual_prefix_hash != source_prefix_hash:
                raise RuntimeError(
                    f"intervention prefix diverged for {unit['evaluation_id']}/{label}"
                )
            run.update(
                {
                    "evaluation_id": unit["evaluation_id"],
                    "source_role": unit["source_role"],
                    "source_candidate_id": candidate_id,
                    "intervention_id": intervention_id,
                    "intervention_values": list(values),
                    "runner_stats": adapter.stats(),
                    "prefix_step_index": prefix_step,
                    "prefix_latent_sha256": actual_prefix_hash,
                    "prefix_matches_continue_cache": True,
                    "image": str(destination.relative_to(output_root)),
                }
            )
            runs.append(run)
            comparisons.append(
                {
                    "sample_id": _unit_id(unit),
                    "prompt_index": int(unit["prompt_index"]),
                    "seed": int(unit["seed"]),
                    "prompt": unit["prompt"],
                    "candidate_id": intervention_id,
                    "baseline": {"image": str(baseline_path)},
                    "candidate": {"image": str(destination.relative_to(output_root))},
                }
            )
            print(
                f"[phase-schedule-{mode}] {unit['evaluation_id']} {label} "
                f"full={adapter.stats()['full_steps']} skip={adapter.stats()['skipped_steps']}",
                flush=True,
            )

    split = f"phase_schedule_{mode}_interventions"
    quality_output = {
        "schema": "difflet-flux-cache-phase-schedule-quality-input",
        "schema_revision": 1,
        "protocol": {
            "prompt_selection": {"split": split},
            "study_id": registration["study_id"],
            "registration_sha256": registration["sha256"],
            "serving_claim": False,
        },
        "comparisons": comparisons,
    }
    quality_name = registration["registered_outputs"][
        "terminal_quality_input" if mode == "terminal" else "repair_quality_input"
    ]
    quality_path = output_root / quality_name
    _write_json(quality_path, quality_output, add_digest=False)
    run_document = {
        "schema": RUN_SCHEMA,
        "schema_revision": RUN_SCHEMA_REVISION,
        "study_id": registration["study_id"],
        "mode": mode,
        "serving_claim": False,
        "registration": _registration_binding(registration_path, registration),
        "selection": {
            "path": str(selection_path),
            "file_sha256": sha256_file(selection_path),
            "content_sha256": selection["sha256"],
        },
        "terminal_analysis": (
            None
            if terminal_analysis_path is None
            else {
                "path": str(terminal_analysis_path),
                "file_sha256": sha256_file(terminal_analysis_path),
                "content_sha256": terminal_analysis["sha256"],
            }
        ),
        "execution": {
            **_git_identity(),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
            "pipeline_warmup_skipped": bool(args.skip_warmup),
        },
        "quality_input": {
            "path": str(quality_path.relative_to(output_root)),
            "file_sha256": sha256_file(quality_path),
            "expected_unique_images": len(units) + len(comparisons),
        },
        "runs": runs,
    }
    run_path = output_root / f"{mode}-intervention-run.json"
    _write_json(run_path, run_document, add_digest=True)
    return run_path, quality_path


def _validate_intervention_semantic(
    semantic_path: Path,
    registration: Mapping[str, Any],
    expected_split: str,
) -> dict[str, Any]:
    semantic = load_semantic_report(semantic_path)
    _, selection, _ = semantic_source(semantic)
    if selection.get("split") != expected_split:
        raise ValueError("intervention semantic split differs from the protocol")
    _validate_metric_identity(semantic["metrics"], registration["semantic_metrics"])
    return semantic


def _intervention_semantic_index(
    semantic: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in semantic["comparisons"]:
        key = (str(row["sample_id"]), str(row["candidate_id"]))
        if key in result:
            raise ValueError(f"duplicate intervention semantic comparison: {key}")
        result[key] = dict(row)
    return result


def evaluate_terminal(
    registration_path: Path,
    selection_path: Path,
    semantic_path: Path,
) -> dict[str, Any]:
    registration = load_registration(registration_path)
    selection = load_selection(selection_path, registration_path)
    if selection["status"] != "ready_for_terminal_horizon":
        raise ValueError("selection status does not permit terminal evaluation")
    semantic = _validate_intervention_semantic(
        semantic_path, registration, "phase_schedule_terminal_interventions"
    )
    rows = _intervention_semantic_index(semantic)
    margins = registration["quality_contract"]["margins"]
    steps = registration["terminal_horizon"]["steps"]
    curves = []
    for step in steps:
        rescued = 0
        introduced = 0
        failure_rows = []
        control_rows = []
        for role, units in (
            ("failure", selection["selected_failures"]),
            ("control", selection["selected_controls"]),
        ):
            for unit in units:
                candidate_id = _intervention_id(
                    "terminal", unit["candidate_id"], int(step)
                )
                row = rows[(_unit_id(unit), candidate_id)]
                delta = row["candidate_minus_baseline"]
                failed_vqa = _vqa_failed(delta, float(margins["vqa_score"]))
                failed_contract = _quality_failed(delta, margins)
                item = {
                    "evaluation_id": unit["evaluation_id"],
                    "candidate_minus_baseline": dict(delta),
                    "vqa_failed": failed_vqa,
                    "contract_failed": failed_contract,
                }
                if role == "failure":
                    rescued += int(not failed_vqa)
                    failure_rows.append(item)
                else:
                    introduced += int(failed_contract)
                    control_rows.append(item)
        source_count = len(selection["selected_failures"])
        control_count = len(selection["selected_controls"])
        curves.append(
            {
                "terminal_step": int(step),
                "rescued_source_failures": rescued,
                "source_failure_count": source_count,
                "R": rescued / source_count,
                "introduced_control_failures": introduced,
                "control_count": control_count,
                "I": introduced / control_count,
                "failure_rows": failure_rows,
                "control_rows": control_rows,
            }
        )
    full_steps = [row["terminal_step"] for row in curves if row["R"] == 1.0 and row["I"] == 0.0]
    t_full = max(full_steps) if full_steps else None
    t_dead = None
    for index, row in enumerate(curves):
        if row["R"] <= 0.2 and all(later["R"] <= 0.2 for later in curves[index:]):
            t_dead = row["terminal_step"]
            break
    status = "usable"
    reason = None
    if t_full is None:
        status = "no_observed_full_rescue_window"
        reason = "no tested terminal step had R=1 and I=0"
    elif t_dead is None:
        status = "no_observed_tail_relaxation_window"
        reason = "no tested terminal step met the registered dead-horizon rule"
    elif t_dead <= t_full:
        status = "invalid_horizon_order"
        reason = "t_dead_observed is not after t_full_observed"
    return {
        "schema": TERMINAL_ANALYSIS_SCHEMA,
        "schema_revision": TERMINAL_ANALYSIS_SCHEMA_REVISION,
        "study_id": registration["study_id"],
        "status": status,
        "reason": reason,
        "serving_claim": False,
        "registration": _registration_binding(registration_path, registration),
        "selection": {
            "path": str(selection_path),
            "file_sha256": sha256_file(selection_path),
            "content_sha256": selection["sha256"],
        },
        "semantic_report": _source_binding(semantic_path),
        "terminal_curve": curves,
        "t_full_observed": t_full,
        "t_dead_observed": t_dead,
    }


def evaluate_repair(
    registration_path: Path,
    selection_path: Path,
    terminal_analysis_path: Path,
    semantic_path: Path,
) -> dict[str, Any]:
    registration = load_registration(registration_path)
    selection = load_selection(selection_path, registration_path)
    terminal = _load_hashed(
        terminal_analysis_path,
        schema=TERMINAL_ANALYSIS_SCHEMA,
        revision=TERMINAL_ANALYSIS_SCHEMA_REVISION,
        name="terminal analysis",
    )
    if terminal["status"] != "usable":
        raise ValueError("terminal horizon is not usable for repair-depth evaluation")
    if terminal["selection"]["file_sha256"] != sha256_file(selection_path):
        raise ValueError("terminal analysis and selection differ")
    semantic = _validate_intervention_semantic(
        semantic_path, registration, "phase_schedule_repair_interventions"
    )
    rows = _intervention_semantic_index(semantic)
    margins = registration["quality_contract"]["margins"]
    curves = []
    for start in registration["repair_depth"]["start_steps"]:
        for real_steps in registration["repair_depth"]["consecutive_real_steps"]:
            rescued = 0
            introduced = 0
            failure_rows = []
            control_rows = []
            for role, units in (
                ("failure", selection["selected_failures"]),
                ("control", selection["selected_controls"]),
            ):
                for unit in units:
                    candidate_id = _intervention_id(
                        "repair", unit["candidate_id"], int(start), int(real_steps)
                    )
                    row = rows[(_unit_id(unit), candidate_id)]
                    delta = row["candidate_minus_baseline"]
                    failed_vqa = _vqa_failed(delta, float(margins["vqa_score"]))
                    failed_contract = _quality_failed(delta, margins)
                    item = {
                        "evaluation_id": unit["evaluation_id"],
                        "candidate_minus_baseline": dict(delta),
                        "vqa_failed": failed_vqa,
                        "contract_failed": failed_contract,
                    }
                    if role == "failure":
                        rescued += int(not failed_vqa)
                        failure_rows.append(item)
                    else:
                        introduced += int(failed_contract)
                        control_rows.append(item)
            source_count = len(selection["selected_failures"])
            control_count = len(selection["selected_controls"])
            curves.append(
                {
                    "start_step": int(start),
                    "consecutive_real_steps": int(real_steps),
                    "rescued_source_failures": rescued,
                    "source_failure_count": source_count,
                    "D_rescue": rescued / source_count,
                    "introduced_control_failures": introduced,
                    "control_count": control_count,
                    "D_introduce": introduced / control_count,
                    "failure_rows": failure_rows,
                    "control_rows": control_rows,
                }
            )
    return {
        "schema": HORIZON_SCHEMA,
        "schema_revision": HORIZON_SCHEMA_REVISION,
        "study_id": registration["study_id"],
        "status": "usable_for_candidate_generation",
        "serving_claim": False,
        "registration": _registration_binding(registration_path, registration),
        "selection": {
            "path": str(selection_path),
            "file_sha256": sha256_file(selection_path),
            "content_sha256": selection["sha256"],
        },
        "terminal_analysis": {
            "path": str(terminal_analysis_path),
            "file_sha256": sha256_file(terminal_analysis_path),
            "content_sha256": terminal["sha256"],
        },
        "repair_semantic_report": _source_binding(semantic_path),
        "t_full_observed": terminal["t_full_observed"],
        "t_dead_observed": terminal["t_dead_observed"],
        "terminal_curve": terminal["terminal_curve"],
        "repair_depth_curve": curves,
        "limitations": [
            "development-only; no serving or quality qualification",
            "horizon is bound to the registered model, scheduler, steps, resolution, profiles, and prompt distribution",
            "six matched controls provide descriptive introduced-failure evidence only",
            "VQA defines source failures; ImageReward remains an independent endpoint diagnostic",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--registration", required=True)
    select.add_argument("--source-semantic", required=True)
    select.add_argument("--out", required=True)

    for command in ("collect-terminal", "collect-repair"):
        collect = subparsers.add_parser(command)
        collect.add_argument("--registration", required=True)
        collect.add_argument("--selection", required=True)
        collect.add_argument("--terminal-analysis")
        collect.add_argument("--out-dir", required=True)
        collect.add_argument("--compile-cache-dir")
        collect.add_argument("--skip-warmup", action="store_true")
        collect.add_argument("--allow-hardware", action="store_true")
        collect.add_argument("--foreground-ack")

    terminal = subparsers.add_parser("evaluate-terminal")
    terminal.add_argument("--registration", required=True)
    terminal.add_argument("--selection", required=True)
    terminal.add_argument("--semantic-report", required=True)
    terminal.add_argument("--out", required=True)

    repair = subparsers.add_parser("evaluate-repair")
    repair.add_argument("--registration", required=True)
    repair.add_argument("--selection", required=True)
    repair.add_argument("--terminal-analysis", required=True)
    repair.add_argument("--semantic-report", required=True)
    repair.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registration_path = Path(args.registration).expanduser().resolve()
        if args.command == "select":
            document = build_selection(
                registration_path,
                Path(args.source_semantic).expanduser().resolve(),
            )
            _write_json(Path(args.out).expanduser().resolve(), document, add_digest=True)
        elif args.command in {"collect-terminal", "collect-repair"}:
            mode = "terminal" if args.command == "collect-terminal" else "repair"
            run_path, quality_path = _collect_interventions(
                registration_path=registration_path,
                selection_path=Path(args.selection).expanduser().resolve(),
                terminal_analysis_path=(
                    None
                    if args.terminal_analysis is None
                    else Path(args.terminal_analysis).expanduser().resolve()
                ),
                output_root=Path(args.out_dir).expanduser().resolve(),
                mode=mode,
                args=args,
            )
            print(f"[phase-schedule-{mode}] run={run_path}")
            print(f"[phase-schedule-{mode}] quality={quality_path}")
        elif args.command == "evaluate-terminal":
            document = evaluate_terminal(
                registration_path,
                Path(args.selection).expanduser().resolve(),
                Path(args.semantic_report).expanduser().resolve(),
            )
            _write_json(Path(args.out).expanduser().resolve(), document, add_digest=True)
        else:
            document = evaluate_repair(
                registration_path,
                Path(args.selection).expanduser().resolve(),
                Path(args.terminal_analysis).expanduser().resolve(),
                Path(args.semantic_report).expanduser().resolve(),
            )
            _write_json(Path(args.out).expanduser().resolve(), document, add_digest=True)
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
