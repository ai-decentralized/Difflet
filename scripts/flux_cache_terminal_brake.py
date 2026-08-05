#!/usr/bin/env python3
"""Measure the endpoint upper bound of disabling FLUX cache after a real anchor.

The runner reuses targets from the frozen paired-brake pilot.  It executes only
one new arm per target: the prefix follows the original adaptive cache policy,
then cache prediction is permanently disabled at the selected real anchor.
This is an opened-data offline causal diagnostic, not a serving qualification.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.pipeline.cache import InMemoryMeasurementSink  # noqa: E402
from scripts.flux_cache_brake_intervention import (  # noqa: E402
    BrakeInterventionPolicy,
    RUN_SCHEMA as SOURCE_RUN_SCHEMA,
    RUN_SCHEMA_REVISION as SOURCE_RUN_SCHEMA_REVISION,
    TargetAnchorFingerprintHook,
    _build_adapter,
    _git_identity,
    _load_hashed_json,
    _load_protocol as _load_source_protocol,
    _quality_failed,
    _relative_image_run,
    _scalar_signals,
    _spearman,
    _write_json,
    _write_measurements,
    sha256_file,
)

PROTOCOL_SCHEMA = "difflet-flux-cache-terminal-brake-followup"
PROTOCOL_SCHEMA_REVISION = 1
RUN_SCHEMA = "difflet-flux-cache-terminal-brake-run"
RUN_SCHEMA_REVISION = 1
ANALYSIS_SCHEMA = "difflet-flux-cache-terminal-brake-analysis"
ANALYSIS_SCHEMA_REVISION = 1
QUALITY_SCHEMA = "difflet-flux-cache-terminal-brake-quality-input"
QUALITY_SCHEMA_REVISION = 1
HARDWARE_ACK = "I am running the offline FLUX terminal brake followup"


def _rooted(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _load_hashed_json(
        path,
        schema=PROTOCOL_SCHEMA,
        revision=PROTOCOL_SCHEMA_REVISION,
    )
    if protocol.get("serving_claim") is not False:
        raise ValueError("terminal-brake protocol must disable serving claims")
    if protocol.get("offline_intervention_only") is not True:
        raise ValueError("terminal-brake protocol must be offline-only")
    if protocol.get("action") != "terminal":
        raise ValueError("terminal-brake protocol action must be terminal")
    if int(protocol.get("expected_terminal_images", -1)) != 9:
        raise ValueError("terminal-brake protocol must register nine target images")
    if int(protocol.get("expected_quality_images", -1)) != 23:
        raise ValueError("terminal-brake protocol must register 23 quality images")
    for name in ("source_protocol", "source_run"):
        source = protocol.get(name)
        if not isinstance(source, dict):
            raise ValueError(f"terminal-brake {name} binding must be an object")
        source_path = _rooted(source["path"])
        if sha256_file(source_path) != source.get("file_sha256"):
            raise ValueError(f"terminal-brake {name} file hash does not match")
    return protocol


def _source_image(source_root: Path, record: Mapping[str, Any]) -> Path:
    path = (source_root / str(record["image"])).resolve()
    if not path.is_file():
        raise ValueError(f"source image does not exist: {path}")
    expected = record.get("image_sha256")
    if expected is not None and sha256_file(path) != expected:
        raise ValueError(f"source image hash does not match: {path}")
    return path


def _candidate_id(phase: str, action: str) -> str:
    return f"anchor-{phase}-{action}"


def run_hardware(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.allow_hardware or args.foreground_ack != HARDWARE_ACK:
        raise RuntimeError(
            f"hardware execution requires --allow-hardware and --foreground-ack {HARDWARE_ACK!r}"
        )
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = _load_protocol(protocol_path)
    source_protocol_path = _rooted(protocol["source_protocol"]["path"])
    source_protocol = _load_source_protocol(source_protocol_path)
    if source_protocol["sha256"] != protocol["source_protocol"]["content_sha256"]:
        raise ValueError("source protocol content hash does not match")
    source_run_path = _rooted(protocol["source_run"]["path"])
    source_run = _load_hashed_json(
        source_run_path,
        schema=SOURCE_RUN_SCHEMA,
        revision=SOURCE_RUN_SCHEMA_REVISION,
    )
    if source_run["sha256"] != protocol["source_run"]["content_sha256"]:
        raise ValueError("source run content hash does not match")
    if source_run["protocol"]["content_sha256"] != source_protocol["sha256"]:
        raise ValueError("source run is not bound to the registered source protocol")

    output_root = Path(args.out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    source_root = source_run_path.parent
    generation = source_protocol["generation_identity"]
    num_steps = int(generation["num_steps"])

    from scripts.collect_flux_cache_ab import _load_pipeline, load_adaptive_candidate
    from scripts.flux_cache_causal_repair import _run_image_only

    arm = load_adaptive_candidate(_rooted(source_protocol["candidate"]["path"]))
    if arm.candidate_id != source_protocol["candidate"]["candidate_id"]:
        raise ValueError("source candidate_id does not match the candidate file")
    pipe_args = SimpleNamespace(
        model_id=generation["model_id"],
        model_revision=generation["model_revision"],
        tp_degree=int(generation["tp_degree"]),
        dtype=generation["dtype"],
        compile_cache_dir=args.compile_cache_dir,
        height=int(generation["height"]),
        width=int(generation["width"]),
        force_compile=False,
        skip_warmup=bool(args.skip_warmup),
        collect_online_signals=False,
    )
    pipe = _load_pipeline(pipe_args)
    flux_pipeline = pipe.app.pipe
    started = time.time()
    sample_results = []
    comparisons = []
    terminal_count = 0

    for sample in source_run["samples"]:
        sample_id = sample["sample_id"]
        prompt = sample["prompt"]
        prompt_index = int(sample["prompt_index"])
        seed = int(sample["seed"])
        baseline_path = _source_image(source_root, sample["baseline"])
        target_results = []
        for target in sample["targets"]:
            phase = target["phase"]
            target_step = int(target["target_step"])
            continue_record = target["branches"]["continue"]
            continue_path = _source_image(source_root, continue_record)
            sink = InMemoryMeasurementSink()
            policy = BrakeInterventionPolicy(
                arm.config,
                action="terminal",
                target_step=target_step,
            )
            adapter = _build_adapter(
                arm,
                num_steps=num_steps,
                policy=policy,
                measurement_sink=sink,
            )
            hook = TargetAnchorFingerprintHook(
                target_step,
                history_supplier=lambda adapter=adapter: adapter.runner.history.anchors,
            )
            flux_pipeline.teacache_controller = adapter
            flux_pipeline._tc_record = False
            flux_pipeline._tc_output_dynamics_record = False
            flux_pipeline._cache_counterfactual_hook = hook
            destination = output_root / "artifacts" / sample_id / f"{phase}-terminal.png"
            try:
                terminal_run = _run_image_only(
                    pipe,
                    flux_pipeline,
                    prompt=prompt,
                    seed=seed,
                    num_steps=num_steps,
                    height=int(generation["height"]),
                    width=int(generation["width"]),
                    guidance_scale=float(generation["guidance_scale"]),
                    image_path=destination,
                )
            finally:
                del flux_pipeline._cache_counterfactual_hook
            policy.validate_complete()
            fingerprint = hook.validate_complete()
            if fingerprint != target["target_fingerprint"]:
                raise RuntimeError(
                    f"terminal branch diverged before {sample_id}/{phase} target anchor"
                )
            event = policy.target_event
            if event is None:
                raise RuntimeError("terminal policy did not expose its target event")
            reference_event = target["reference_event"]
            if float(event["estimate_relative_error"]) != float(
                reference_event["estimate_relative_error"]
            ) or int(event["pre_interval"]) != int(reference_event["pre_interval"]):
                raise RuntimeError("terminal target measurement differs from source reference")
            terminal_run = _relative_image_run(terminal_run, output_root)
            terminal_run.update(
                {
                    "candidate_id": _candidate_id(phase, "terminal"),
                    "runner_stats": adapter.stats(),
                    "measurements": _write_measurements(
                        sink,
                        num_steps=num_steps,
                        destination=destination.with_suffix(".measurements.json"),
                        output_root=output_root,
                    ),
                    "target_event": event,
                    "target_fingerprint": fingerprint,
                }
            )
            target_results.append(
                {
                    "phase": phase,
                    "target_step": target_step,
                    "signals": target["signals"],
                    "source_continue": continue_record,
                    "terminal": terminal_run,
                }
            )
            common = {
                "sample_id": sample_id,
                "prompt_index": prompt_index,
                "seed": seed,
                "prompt": prompt,
                "baseline": {"image": str(baseline_path)},
            }
            comparisons.extend(
                (
                    {
                        **common,
                        "candidate_id": _candidate_id(phase, "continue"),
                        "candidate": {"image": str(continue_path)},
                    },
                    {
                        **common,
                        "candidate_id": _candidate_id(phase, "terminal"),
                        "candidate": {
                            "image": str((output_root / terminal_run["image"]).resolve())
                        },
                    },
                )
            )
            terminal_count += 1
            print(
                f"[terminal-brake] {sample_id}/{phase} step={target_step} "
                f"full={terminal_run['runner_stats']['full_steps']} "
                f"skip={terminal_run['runner_stats']['skipped_steps']}",
                flush=True,
            )
        sample_results.append(
            {
                "sample_id": sample_id,
                "prompt_index": prompt_index,
                "seed": seed,
                "prompt": prompt,
                "prior_label": sample.get("prior_label"),
                "baseline": sample["baseline"],
                "targets": target_results,
            }
        )

    if terminal_count != int(protocol["expected_terminal_images"]):
        raise RuntimeError("terminal target count does not match the protocol")
    quality = {
        "schema": QUALITY_SCHEMA,
        "schema_revision": QUALITY_SCHEMA_REVISION,
        "protocol": {
            "prompt_selection": {"split": protocol["quality_split"]},
            "study_id": protocol["study_id"],
            "protocol_path": str(protocol_path),
            "protocol_sha256": protocol["sha256"],
            "offline_intervention_only": True,
            "serving_speed_claim": False,
        },
        "comparisons": comparisons,
    }
    quality_path = output_root / "quality-input.json"
    _write_json(quality_path, quality, add_digest=False)
    run_document = {
        "schema": RUN_SCHEMA,
        "schema_revision": RUN_SCHEMA_REVISION,
        "study_id": protocol["study_id"],
        "protocol": {
            "path": str(protocol_path),
            "file_sha256": sha256_file(protocol_path),
            "content_sha256": protocol["sha256"],
        },
        "source_run": {
            "path": str(source_run_path),
            "file_sha256": sha256_file(source_run_path),
            "content_sha256": source_run["sha256"],
        },
        "execution": {
            **_git_identity(),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
            "pipeline_warmup_skipped": bool(args.skip_warmup),
        },
        "offline_intervention_only": True,
        "serving_claim": False,
        "quality_input": {
            "path": str(quality_path.relative_to(output_root)),
            "sha256": sha256_file(quality_path),
            "expected_unique_images": protocol["expected_quality_images"],
        },
        "samples": sample_results,
    }
    run_path = output_root / "terminal-brake-run.json"
    _write_json(run_path, run_document, add_digest=True)
    return run_path, quality_path


def analyze(args: argparse.Namespace) -> Path:
    protocol = _load_protocol(Path(args.protocol).expanduser().resolve())
    run = _load_hashed_json(
        Path(args.run_result).expanduser().resolve(),
        schema=RUN_SCHEMA,
        revision=RUN_SCHEMA_REVISION,
    )
    semantic_path = Path(args.semantic_scores).expanduser().resolve()
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    if semantic.get("complete") is not True:
        raise ValueError("semantic score report is incomplete")
    semantic_rows = {
        (row["sample_id"], row["candidate_id"]): row for row in semantic.get("comparisons", ())
    }
    margins = {
        "image_reward": float(protocol["quality_contract"]["image_reward_max_harm"]),
        "vqa_score": float(protocol["quality_contract"]["vqa_score_max_harm"]),
    }
    rows = []
    for sample in run["samples"]:
        sample_id = sample["sample_id"]
        for target in sample["targets"]:
            phase = target["phase"]
            continue_semantic = semantic_rows[(sample_id, _candidate_id(phase, "continue"))]
            terminal_semantic = semantic_rows[(sample_id, _candidate_id(phase, "terminal"))]
            baseline_scores = continue_semantic["baseline_scores"]
            continue_scores = continue_semantic["candidate_scores"]
            terminal_scores = terminal_semantic["candidate_scores"]
            continue_failed, continue_harm = _quality_failed(
                baseline=baseline_scores,
                candidate=continue_scores,
                margins=margins,
            )
            terminal_failed, terminal_harm = _quality_failed(
                baseline=baseline_scores,
                candidate=terminal_scores,
                margins=margins,
            )
            if continue_failed and not terminal_failed:
                outcome = "rescue"
            elif continue_failed and terminal_failed:
                outcome = "both_fail"
            elif not continue_failed and terminal_failed:
                outcome = "introduced_failure"
            else:
                outcome = "both_pass"
            gains = {
                metric: float(terminal_scores[metric]) - float(continue_scores[metric])
                for metric in ("image_reward", "vqa_score")
            }
            recoverable_fraction = {}
            for metric in ("image_reward", "vqa_score"):
                gap = float(baseline_scores[metric]) - float(continue_scores[metric])
                recoverable_fraction[metric] = gains[metric] / gap if gap > 0.0 else None
            continue_stats = target["source_continue"]["runner_stats"]
            terminal_stats = target["terminal"]["runner_stats"]
            rows.append(
                {
                    "sample_id": sample_id,
                    "prompt_index": sample["prompt_index"],
                    "seed": sample["seed"],
                    "prior_label": sample.get("prior_label"),
                    "phase": phase,
                    "target_step": target["target_step"],
                    "signals": target["signals"],
                    "quality": {
                        "continue_failed": continue_failed,
                        "terminal_failed": terminal_failed,
                        "outcome": outcome,
                        "continue_harm": continue_harm,
                        "terminal_harm": terminal_harm,
                        "terminal_minus_continue_gain": gains,
                        "recovered_fraction_of_continue_gap": recoverable_fraction,
                    },
                    "cost": {
                        "continue_full_steps": int(continue_stats["full_steps"]),
                        "terminal_full_steps": int(terminal_stats["full_steps"]),
                        "extra_full_steps": int(terminal_stats["full_steps"])
                        - int(continue_stats["full_steps"]),
                        "continue_skipped_steps": int(continue_stats["skipped_steps"]),
                        "terminal_skipped_steps": int(terminal_stats["skipped_steps"]),
                    },
                }
            )

    scalar_names = sorted(set.intersection(*(set(_scalar_signals(row["signals"])) for row in rows)))
    correlations = {
        signal: {
            metric: _spearman(
                [_scalar_signals(row["signals"])[signal] for row in rows],
                [row["quality"]["terminal_minus_continue_gain"][metric] for row in rows],
            )
            for metric in ("image_reward", "vqa_score")
        }
        for signal in scalar_names
    }
    known_rows = [row for row in rows if str(row.get("prior_label", "")).endswith("-failure")]
    control_rows = [
        row for row in rows if str(row.get("prior_label", "")).endswith("-pass-control")
    ]
    rescued = [row for row in known_rows if row["quality"]["outcome"] == "rescue"]
    introduced = [row for row in control_rows if row["quality"]["outcome"] == "introduced_failure"]
    if rescued and not introduced:
        status = "actuator_viable"
        interpretation = protocol["decision_rule"]["interpretation_if_viable"]
    elif rescued:
        status = "actuator_viable_with_control_harm"
        interpretation = (
            "Terminal brake rescued a known failure but also introduced a control failure."
        )
    else:
        status = "terminal_brake_no_rescue"
        interpretation = protocol["decision_rule"]["interpretation_if_no_rescue"]
    by_phase = {}
    for phase in ("early", "middle", "late"):
        phase_rows = [row for row in known_rows if row["phase"] == phase]
        if phase_rows:
            by_phase[phase] = {
                "target_count": len(phase_rows),
                "rescue_count": sum(row["quality"]["outcome"] == "rescue" for row in phase_rows),
                "mean_terminal_gain": {
                    metric: statistics.fmean(
                        row["quality"]["terminal_minus_continue_gain"][metric] for row in phase_rows
                    )
                    for metric in ("image_reward", "vqa_score")
                },
            }
    document = {
        "schema": ANALYSIS_SCHEMA,
        "schema_revision": ANALYSIS_SCHEMA_REVISION,
        "study_id": protocol["study_id"],
        "evidence_role": protocol["evidence_role"],
        "serving_claim": False,
        "inputs": {
            "protocol_sha256": protocol["sha256"],
            "run_result_sha256": run["sha256"],
            "semantic_scores_path": str(semantic_path),
            "semantic_scores_file_sha256": sha256_file(semantic_path),
        },
        "quality_contract": margins,
        "summary": {
            "target_count": len(rows),
            "known_failure_target_count": len(known_rows),
            "known_failure_rescue_count": len(rescued),
            "pass_control_target_count": len(control_rows),
            "introduced_control_failure_count": len(introduced),
            "mean_extra_full_steps": statistics.fmean(
                row["cost"]["extra_full_steps"] for row in rows
            ),
            "mean_terminal_gain": {
                metric: statistics.fmean(
                    row["quality"]["terminal_minus_continue_gain"][metric] for row in rows
                )
                for metric in ("image_reward", "vqa_score")
            },
            "known_failures_by_phase": by_phase,
        },
        "decision": {
            "status": status,
            "interpretation": interpretation,
            "rescued_targets": [
                {"sample_id": row["sample_id"], "phase": row["phase"]} for row in rescued
            ],
            "introduced_control_failures": [
                {"sample_id": row["sample_id"], "phase": row["phase"]} for row in introduced
            ],
        },
        "signal_to_terminal_gain_spearman": correlations,
        "rows": rows,
        "interpretation_limits": [
            "All prompts, endpoint labels, and target anchors were already opened.",
            "Terminal brake is an actuation upper bound, not a serving speed claim.",
            "VQAScore is a model-based endpoint evaluator and its scalar distance is not calibrated human loss.",
            "A positive result can only register a new signal hypothesis for prospective confirmation.",
        ],
    }
    destination = Path(args.out).expanduser().resolve()
    _write_json(destination, document, add_digest=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run terminal-brake arms on Trainium")
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("--compile-cache-dir")
    run_parser.add_argument("--skip-warmup", action="store_true")
    run_parser.add_argument("--allow-hardware", action="store_true")
    run_parser.add_argument("--foreground-ack")
    analyze_parser = subparsers.add_parser("analyze", help="join endpoint scores")
    analyze_parser.add_argument("--protocol", required=True)
    analyze_parser.add_argument("--run-result", required=True)
    analyze_parser.add_argument("--semantic-scores", required=True)
    analyze_parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        run_path, quality_path = run_hardware(args)
        print(f"[terminal-brake] run={run_path}", flush=True)
        print(f"[terminal-brake] quality_input={quality_path}", flush=True)
    else:
        destination = analyze(args)
        print(f"[terminal-brake] analysis={destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
