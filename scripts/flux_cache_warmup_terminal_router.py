#!/usr/bin/env python3
"""Test a warmup VQA risk signal with a true pre-skip terminal brake.

The opened-data signal is already available after step 5.  For registered
triggered requests this runner follows the original full-compute warmup and
permanently disables cache at step 6, before the first eligible cache skip.
This is a development-only causal upper-bound test, not serving evidence.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.pipeline.cache import (  # noqa: E402
    AdaptiveAnchorPolicy,
    InMemoryMeasurementSink,
)
from scripts.flux_cache_brake_intervention import (  # noqa: E402
    _build_adapter,
    _git_identity,
    _quality_failed,
    _write_json,
    sha256_file,
)
from scripts.flux_cache_causal_repair import _run_image_only  # noqa: E402
from scripts.flux_cache_warmup_vqa_router import (  # noqa: E402
    DISCOVERY_SCHEMA,
    DISCOVERY_SCHEMA_REVISION,
    _load_hashed,
    _load_json,
    _load_protocol as _load_router_protocol,
    _rooted,
    _semantic_image_index,
    _source_semantic_index,
    _speed_index,
)

PROTOCOL_SCHEMA = "difflet-flux-cache-warmup-terminal-router-pilot"
PROTOCOL_SCHEMA_REVISION = 1
RUN_SCHEMA = "difflet-flux-cache-warmup-terminal-router-run"
RUN_SCHEMA_REVISION = 1
ANALYSIS_SCHEMA = "difflet-flux-cache-warmup-terminal-router-analysis"
ANALYSIS_SCHEMA_REVISION = 1
QUALITY_SCHEMA = "difflet-flux-cache-warmup-terminal-router-quality-input"
QUALITY_SCHEMA_REVISION = 1
HARDWARE_ACK = "I am running the offline FLUX warmup terminal router pilot"


class WarmupTerminalPolicy(AdaptiveAnchorPolicy):
    """Permanently stop requesting cache predictions at a registered step."""

    def __init__(self, config: Any, *, terminal_step: int) -> None:
        if isinstance(terminal_step, bool) or not isinstance(terminal_step, int):
            raise TypeError("terminal_step must be an integer")
        if terminal_step < config.warmup_steps:
            raise ValueError("terminal_step must not precede protected warmup")
        self.terminal_step = terminal_step
        super().__init__(config)

    def reset(self) -> None:
        super().reset()
        self._terminal_applied = False

    def should_skip(self, context: Any, history: Any, observation: Any) -> bool:
        if context.step_index >= self.terminal_step and not self._terminal_applied:
            self._disable()
            self._terminal_applied = True
        return super().should_skip(context, history, observation)

    def validate_complete(self) -> None:
        if not self._terminal_applied:
            raise RuntimeError("warmup terminal brake was never applied")

    def stats(self) -> dict[str, Any]:
        result = super().stats()
        result.update(
            {
                "warmup_terminal_step": self.terminal_step,
                "warmup_terminal_applied": self._terminal_applied,
            }
        )
        return result


def _load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _load_hashed(path, PROTOCOL_SCHEMA, PROTOCOL_SCHEMA_REVISION)
    if protocol.get("serving_claim") is not False:
        raise ValueError("warmup terminal protocol must disable serving claims")
    if protocol.get("action") != "disable_cache_before_first_skip":
        raise ValueError("unsupported warmup terminal action")
    if int(protocol["terminal_step"]) != 6:
        raise ValueError("warmup terminal pilot must act at step 6")
    for name in ("router_protocol", "discovery", "safe_route_analysis", "prefix_profile"):
        binding = protocol[name]
        binding_path = _rooted(binding["path"])
        if sha256_file(binding_path) != binding["file_sha256"]:
            raise ValueError(f"registered {name} file hash does not match")
    router_path = _rooted(protocol["router_protocol"]["path"])
    router = _load_router_protocol(router_path)
    if router["sha256"] != protocol["router_protocol"]["content_sha256"]:
        raise ValueError("router protocol content hash does not match")
    discovery_path = _rooted(protocol["discovery"]["path"])
    discovery = _load_hashed(
        discovery_path,
        DISCOVERY_SCHEMA,
        DISCOVERY_SCHEMA_REVISION,
    )
    if discovery["sha256"] != protocol["discovery"]["content_sha256"]:
        raise ValueError("discovery content hash does not match")
    safe_analysis = _load_json(_rooted(protocol["safe_route_analysis"]["path"]))
    if safe_analysis.get("sha256") != protocol["safe_route_analysis"]["content_sha256"]:
        raise ValueError("safe route analysis content hash does not match")
    if safe_analysis.get("decision", {}).get("status") != protocol["safe_route_analysis"][
        "required_status"
    ]:
        raise ValueError("safe route analysis did not register the required rejection")
    if int(router["profiles"]["safe"]["candidate_id"].split("-w", 1)[1].split("-", 1)[0]) != 6:
        raise ValueError("router profile warmup does not end before terminal step")
    samples = protocol["samples"]
    if len(samples) != int(protocol["expected_terminal_images"]):
        raise ValueError("registered terminal image count does not match")
    if int(protocol["expected_quality_images"]) != 2 * len(samples):
        raise ValueError("registered quality image count does not match")
    return protocol, router, discovery


def run_hardware(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.allow_hardware or args.foreground_ack != HARDWARE_ACK:
        raise RuntimeError(
            f"hardware execution requires --allow-hardware and --foreground-ack {HARDWARE_ACK!r}"
        )
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol, router, discovery = _load_protocol(protocol_path)
    output_root = Path(args.out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    from scripts.collect_flux_cache_ab import _load_pipeline, load_adaptive_candidate

    generation = router["generation_identity"]
    profile = protocol["prefix_profile"]
    arm = load_adaptive_candidate(_rooted(profile["path"]))
    if arm.candidate_id != profile["candidate_id"]:
        raise ValueError("prefix profile candidate_id does not match")
    if int(arm.config.warmup_steps) != int(protocol["terminal_step"]):
        raise ValueError("terminal step must equal the prefix profile warmup length")
    prompts = _load_json(_rooted(router["prompt_source"]["path"]))["prompts"]
    source_semantic = _load_json(_rooted(router["source"]["semantic_scores"]["path"]))
    source_images = _semantic_image_index(source_semantic)
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
    runs = []
    comparisons = []
    for sample in protocol["samples"]:
        sample_id = sample["sample_id"]
        prompt_index = int(sample["prompt_index"])
        seed = int(sample["seed"])
        prompt = prompts[prompt_index]
        baseline = source_images[f"inline:baseline:{sample_id}"]
        baseline_path = Path(baseline["image_path"]).resolve()
        if sha256_file(baseline_path) != baseline["image_sha256"]:
            raise ValueError(f"baseline image hash does not match for {sample_id}")
        policy = WarmupTerminalPolicy(
            arm.config,
            terminal_step=int(protocol["terminal_step"]),
        )
        adapter = _build_adapter(
            arm,
            num_steps=int(generation["num_steps"]),
            policy=policy,
            measurement_sink=InMemoryMeasurementSink(),
        )
        flux_pipeline.teacache_controller = adapter
        destination = output_root / "artifacts" / f"{sample_id}.terminal.png"
        run = _run_image_only(
            pipe,
            flux_pipeline,
            prompt=prompt,
            seed=seed,
            num_steps=int(generation["num_steps"]),
            height=int(generation["height"]),
            width=int(generation["width"]),
            guidance_scale=float(generation["guidance_scale"]),
            image_path=destination,
        )
        policy.validate_complete()
        stats = adapter.stats()
        if stats["skipped_steps"] != 0 or stats["full_steps"] != int(generation["num_steps"]):
            raise RuntimeError("warmup terminal branch did not execute full-DiT suffix")
        run.update(
            {
                "sample_id": sample_id,
                "prompt_index": prompt_index,
                "seed": seed,
                "prompt": prompt,
                "image": str(destination.relative_to(output_root)),
                "image_sha256": sha256_file(destination),
                "baseline_image_sha256": baseline["image_sha256"],
                "byte_identical_to_baseline": sha256_file(destination)
                == baseline["image_sha256"],
                "runner_stats": stats,
            }
        )
        runs.append(run)
        comparisons.append(
            {
                "sample_id": sample_id,
                "prompt_index": prompt_index,
                "seed": seed,
                "prompt": prompt,
                "candidate_id": "warmup-terminal-route",
                "baseline": {"image": str(baseline_path)},
                "candidate": {"image": str(destination)},
            }
        )
        print(
            f"[warmup-terminal-router] {sample_id} full={stats['full_steps']} "
            f"skip={stats['skipped_steps']} identical={run['byte_identical_to_baseline']}",
            flush=True,
        )
    quality = {
        "schema": QUALITY_SCHEMA,
        "schema_revision": QUALITY_SCHEMA_REVISION,
        "protocol": {
            "prompt_selection": {"split": "warmup_terminal_route_development"},
            "study_id": protocol["study_id"],
            "protocol_sha256": protocol["sha256"],
            "serving_claim": False,
        },
        "comparisons": comparisons,
    }
    quality_path = output_root / "quality-input.json"
    _write_json(quality_path, quality, add_digest=False)
    run_document = {
        "schema": RUN_SCHEMA,
        "schema_revision": RUN_SCHEMA_REVISION,
        "study_id": protocol["study_id"],
        "serving_claim": False,
        "protocol": {
            "path": str(protocol_path),
            "file_sha256": sha256_file(protocol_path),
            "content_sha256": protocol["sha256"],
        },
        "discovery_content_sha256": discovery["sha256"],
        "execution": {
            **_git_identity(),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
            "pipeline_warmup_skipped": bool(args.skip_warmup),
        },
        "quality_input": {
            "path": str(quality_path.relative_to(output_root)),
            "file_sha256": sha256_file(quality_path),
            "expected_unique_images": int(protocol["expected_quality_images"]),
        },
        "terminal_runs": runs,
    }
    run_path = output_root / "warmup-terminal-router-run.json"
    _write_json(run_path, run_document, add_digest=True)
    return run_path, quality_path


def analyze(args: argparse.Namespace) -> Path:
    protocol, router, discovery = _load_protocol(Path(args.protocol).expanduser().resolve())
    run = _load_hashed(
        Path(args.run_result).expanduser().resolve(), RUN_SCHEMA, RUN_SCHEMA_REVISION
    )
    semantic_path = Path(args.semantic_scores).expanduser().resolve()
    semantic = _load_json(semantic_path)
    if semantic.get("complete") is not True:
        raise ValueError("terminal semantic report is incomplete")
    terminal_rows = {row["sample_id"]: row for row in semantic["comparisons"]}
    terminal_runs = {row["sample_id"]: row for row in run["terminal_runs"]}
    source_semantic = _load_json(_rooted(router["source"]["semantic_scores"]["path"]))
    source_rows = _source_semantic_index(source_semantic)
    speed = _load_json(_rooted(router["source"]["speed_manifest"]["path"]))
    speed_rows = _speed_index(speed)
    margins = {
        "image_reward": float(router["quality_contract"]["image_reward_max_harm"]),
        "vqa_score": float(router["quality_contract"]["vqa_score_max_harm"]),
    }
    rows = []
    for signal_row in discovery["rows"]:
        key = (signal_row["candidate_id"], signal_row["sample_id"])
        original = source_rows[key]
        original_failed, original_harm = _quality_failed(
            baseline=original["baseline_scores"],
            candidate=original["candidate_scores"],
            margins=margins,
        )
        if signal_row["triggered"]:
            routed = terminal_rows[signal_row["sample_id"]]
            routed_scores = routed["candidate_scores"]
            baseline_scores = routed["baseline_scores"]
            elapsed = float(terminal_runs[signal_row["sample_id"]]["elapsed_s"])
            route = "terminal"
        else:
            routed_scores = original["candidate_scores"]
            baseline_scores = original["baseline_scores"]
            elapsed = speed_rows[key]
            route = "oil"
        routed_failed, routed_harm = _quality_failed(
            baseline=baseline_scores,
            candidate=routed_scores,
            margins=margins,
        )
        rows.append(
            {
                **signal_row,
                "route": route,
                "original_failed": original_failed,
                "routed_failed": routed_failed,
                "original_harm": original_harm,
                "routed_harm": routed_harm,
                "elapsed_s_diagnostic": elapsed,
            }
        )
    summaries = {}
    for candidate_id in sorted({row["candidate_id"] for row in rows}):
        selected = [row for row in rows if row["candidate_id"] == candidate_id]
        original_time = sum(speed_rows[(candidate_id, row["sample_id"])] for row in selected)
        routed_time = sum(row["elapsed_s_diagnostic"] for row in selected)
        summaries[candidate_id] = {
            "sample_count": len(selected),
            "triggered_count": sum(row["triggered"] for row in selected),
            "original_vqa_failure_count": sum(row["vqa_failed"] for row in selected),
            "routed_vqa_failure_count": sum(
                row["routed_harm"]["vqa_score"] > margins["vqa_score"] for row in selected
            ),
            "original_contract_failure_count": sum(row["original_failed"] for row in selected),
            "routed_contract_failure_count": sum(row["routed_failed"] for row in selected),
            "original_total_s": original_time,
            "routed_total_s_diagnostic": routed_time,
            "routing_cost_increase_fraction_diagnostic": routed_time / original_time - 1.0,
        }
    failures = [row for row in rows if row["vqa_failed"]]
    unrescued = [
        row for row in failures if row["routed_harm"]["vqa_score"] > margins["vqa_score"]
    ]
    introduced = [row for row in rows if not row["original_failed"] and row["routed_failed"]]
    mapping_found = not unrescued and not introduced and all(row["triggered"] for row in failures)
    document = {
        "schema": ANALYSIS_SCHEMA,
        "schema_revision": ANALYSIS_SCHEMA_REVISION,
        "study_id": protocol["study_id"],
        "serving_claim": False,
        "inputs": {
            "protocol_content_sha256": protocol["sha256"],
            "discovery_content_sha256": discovery["sha256"],
            "run_content_sha256": run["sha256"],
            "semantic_scores_path": str(semantic_path),
            "semantic_scores_file_sha256": sha256_file(semantic_path),
        },
        "signal": router["online_signal"],
        "offline_label": router["offline_label"],
        "profile_summaries": summaries,
        "decision": {
            "status": (
                "development_terminal_mapping_found"
                if mapping_found
                else "development_terminal_mapping_rejected"
            ),
            "all_opened_vqa_failures_rescued": not unrescued,
            "unrescued_vqa_failures": [
                [row["candidate_id"], row["sample_id"]] for row in unrescued
            ],
            "introduced_contract_failures": [
                [row["candidate_id"], row["sample_id"]] for row in introduced
            ],
            "serving_qualified": False,
            "next": "Freeze this exact router and test it on a new prompt-group holdout.",
        },
        "terminal_runs": [
            {
                "sample_id": row["sample_id"],
                "elapsed_s": row["elapsed_s"],
                "byte_identical_to_baseline": row["byte_identical_to_baseline"],
                "full_steps": row["runner_stats"]["full_steps"],
                "skipped_steps": row["runner_stats"]["skipped_steps"],
            }
            for row in run["terminal_runs"]
        ],
        "rows": rows,
        "interpretation_limits": [
            "Signal, direction, window, and threshold were selected on opened development data.",
            "Only two independent positive prompt groups are present.",
            "Terminal routing is the full-compute quality upper bound, not a speed-optimal brake.",
            "Timing omits shallow-probe overhead and is diagnostic.",
        ],
    }
    destination = Path(args.out).expanduser().resolve()
    _write_json(destination, document, add_digest=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("--compile-cache-dir")
    run_parser.add_argument("--skip-warmup", action="store_true")
    run_parser.add_argument("--allow-hardware", action="store_true")
    run_parser.add_argument("--foreground-ack")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--protocol", required=True)
    analyze_parser.add_argument("--run-result", required=True)
    analyze_parser.add_argument("--semantic-scores", required=True)
    analyze_parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        run_path, quality_path = run_hardware(args)
        print(f"[warmup-terminal-router] run={run_path}", flush=True)
        print(f"[warmup-terminal-router] quality_input={quality_path}", flush=True)
    else:
        destination = analyze(args)
        print(f"[warmup-terminal-router] analysis={destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
