#!/usr/bin/env python3
"""Discover and test a warmup-only online router for offline FLUX VQAScore loss.

The signal is computed from shallow 4x4 modulation maps during full-compute
warmup, before the first cache skip.  The hardware arm generates only the safe
profile images needed by triggered requests; analysis splices those paired
scores into the already-complete oil-profile matrix.  All evidence is opened
development data and cannot qualify serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_flux_cache_online_gate import grouped_oof_logistic  # noqa: E402
from scripts.evaluate_flux_cache_online_signal import extract_features  # noqa: E402
from scripts.flux_cache_brake_intervention import (  # noqa: E402
    _git_identity,
    _quality_failed,
    _write_json,
    canonical_sha256,
    sha256_file,
)

PROTOCOL_SCHEMA = "difflet-flux-cache-warmup-vqa-router-pilot"
PROTOCOL_SCHEMA_REVISION = 1
DISCOVERY_SCHEMA = "difflet-flux-cache-warmup-vqa-signal-discovery"
DISCOVERY_SCHEMA_REVISION = 1
RUN_SCHEMA = "difflet-flux-cache-warmup-vqa-router-run"
RUN_SCHEMA_REVISION = 1
ANALYSIS_SCHEMA = "difflet-flux-cache-warmup-vqa-router-analysis"
ANALYSIS_SCHEMA_REVISION = 1
QUALITY_SCHEMA = "difflet-flux-cache-warmup-vqa-router-quality-input"
QUALITY_SCHEMA_REVISION = 1
HARDWARE_ACK = "I am running the offline FLUX warmup VQA router pilot"


def _rooted(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return document


def _load_hashed(path: Path, schema: str, revision: int) -> dict[str, Any]:
    document = _load_json(path)
    digest = document.get("sha256")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if digest != canonical_sha256(payload):
        raise ValueError(f"JSON artifact digest does not match: {path}")
    if document.get("schema") != schema or document.get("schema_revision") != revision:
        raise ValueError(f"JSON artifact schema is unsupported: {path}")
    return document


def _signal_files(protocol: Mapping[str, Any]) -> list[Path]:
    root = Path(protocol["source"]["artifact_root"]).resolve()
    return sorted(root.glob("artifacts/*/*.online-signal.json"))


def _signal_dataset_digest(protocol: Mapping[str, Any]) -> str:
    root = Path(protocol["source"]["artifact_root"]).resolve()
    rows = [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
        for path in _signal_files(protocol)
    ]
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _load_hashed(path, PROTOCOL_SCHEMA, PROTOCOL_SCHEMA_REVISION)
    if protocol.get("serving_claim") is not False:
        raise ValueError("warmup VQA router protocol must disable serving claims")
    for binding in (
        protocol["source"]["gate_development"],
        protocol["source"]["semantic_scores"],
        protocol["source"]["speed_manifest"],
        protocol["prompt_source"],
        protocol["profiles"]["safe"],
        *protocol["profiles"]["oil"],
    ):
        path_value = _rooted(binding["path"])
        if sha256_file(path_value) != binding["file_sha256"]:
            raise ValueError(f"registered file hash does not match: {path_value}")
    files = _signal_files(protocol)
    if len(files) != int(protocol["source"]["online_signal_file_count"]):
        raise ValueError("online signal file count does not match")
    if _signal_dataset_digest(protocol) != protocol["source"]["online_signal_dataset_sha256"]:
        raise ValueError("online signal dataset digest does not match")
    signal = protocol["online_signal"]
    if signal["window"] != [2, 5] or signal["risk_direction"] != "lower_is_riskier":
        raise ValueError("warmup signal definition is unsupported")
    if signal.get("available_before_first_cache_skip") is not True:
        raise ValueError("warmup signal must be available before the first cache skip")
    return protocol


def _feature_for_row(protocol: Mapping[str, Any], candidate_id: str, sample_id: str) -> float:
    root = Path(protocol["source"]["artifact_root"]).resolve()
    path = root / "artifacts" / candidate_id / f"{sample_id}.online-signal.json"
    raw = _load_json(path)
    first, last = (int(value) for value in protocol["online_signal"]["window"])
    features = extract_features(
        raw["spatial_records"],
        first_eligible_step=first,
        last_eligible_step=last,
    )
    return float(features["max_acceleration_cv"])


def _discovery_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    gate = _load_json(_rooted(protocol["source"]["gate_development"]["path"]))
    threshold = float(protocol["online_signal"]["development_threshold"])
    rows = []
    for row in gate["rows"]:
        value = _feature_for_row(protocol, row["candidate_id"], row["sample_id"])
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "sample_id": row["sample_id"],
                "prompt_index": int(row["prompt_index"]),
                "seed": int(row["seed"]),
                "vqa_score_harm": float(row["vqa_score_harm"]),
                "vqa_failed": float(row["vqa_score_harm"])
                > float(protocol["offline_label"]["paired_harm_threshold"]),
                "signal": value,
                "triggered": value <= threshold,
            }
        )
    return rows


def discover(args: argparse.Namespace) -> Path:
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = _load_protocol(protocol_path)
    rows = _discovery_rows(protocol)
    model_rows = [
        {
            **row,
            "failed": row["vqa_failed"],
            "features": {"warmup_acceleration_cv": row["signal"]},
        }
        for row in rows
    ]
    model = grouped_oof_logistic(model_rows, ("warmup_acceleration_cv",))
    failures = [row for row in rows if row["vqa_failed"]]
    passes = [row for row in rows if not row["vqa_failed"]]
    missed = [row for row in failures if not row["triggered"]]
    false_routes = [row for row in passes if row["triggered"]]
    expected_triggers = {
        (row["candidate_id"], row["sample_id"]) for row in protocol["triggered_comparisons"]
    }
    observed_triggers = {
        (row["candidate_id"], row["sample_id"]) for row in rows if row["triggered"]
    }
    if observed_triggers != expected_triggers:
        raise RuntimeError("registered and reproduced warmup triggers differ")
    document = {
        "schema": DISCOVERY_SCHEMA,
        "schema_revision": DISCOVERY_SCHEMA_REVISION,
        "study_id": protocol["study_id"],
        "evidence_role": protocol["evidence_role"],
        "serving_claim": False,
        "inputs": {
            "protocol_path": str(protocol_path),
            "protocol_file_sha256": sha256_file(protocol_path),
            "protocol_content_sha256": protocol["sha256"],
            "online_signal_dataset_sha256": _signal_dataset_digest(protocol),
        },
        "signal": protocol["online_signal"],
        "label": protocol["offline_label"],
        "sample_count": len(rows),
        "vqa_failure_comparison_count": len(failures),
        "vqa_failure_prompt_group_count": len({row["prompt_index"] for row in failures}),
        "grouped_oof": model,
        "direct_threshold": {
            "failure_recall": 1.0 - len(missed) / len(failures),
            "passing_request_false_route_rate": len(false_routes) / len(passes),
            "passing_request_false_route_count": len(false_routes),
            "passing_request_count": len(passes),
            "false_route_prompt_groups": sorted({row["prompt_index"] for row in false_routes}),
            "triggered_comparison_count": len(observed_triggers),
        },
        "decision": {
            "status": "candidate_found_development_only",
            "reason": "A metric-specific warmup signal ranks all opened VQAScore failures before the first cache skip, but only two independent positive prompt groups exist and the window was selected on opened data.",
            "run_safe_route_pilot": True,
            "serving_qualified": False,
        },
        "rows": rows,
    }
    destination = Path(args.out).expanduser().resolve()
    _write_json(destination, document, add_digest=True)
    return destination


def _semantic_image_index(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["image_id"]): row for row in document["images"]}


def run_hardware(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.allow_hardware or args.foreground_ack != HARDWARE_ACK:
        raise RuntimeError(
            f"hardware execution requires --allow-hardware and --foreground-ack {HARDWARE_ACK!r}"
        )
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = _load_protocol(protocol_path)
    discovery = _load_hashed(
        Path(args.discovery).expanduser().resolve(),
        DISCOVERY_SCHEMA,
        DISCOVERY_SCHEMA_REVISION,
    )
    if discovery["inputs"]["protocol_content_sha256"] != protocol["sha256"]:
        raise ValueError("discovery is not bound to the router protocol")
    output_root = Path(args.out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    from scripts.collect_flux_cache_ab import _load_pipeline, load_adaptive_candidate
    from scripts.flux_cache_causal_repair import _run_image_only

    prompts_document = _load_json(_rooted(protocol["prompt_source"]["path"]))
    prompts = prompts_document["prompts"]
    safe = load_adaptive_candidate(_rooted(protocol["profiles"]["safe"]["path"]))
    if safe.candidate_id != protocol["profiles"]["safe"]["candidate_id"]:
        raise ValueError("safe candidate_id does not match")
    generation = protocol["generation_identity"]
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
    source_semantic = _load_json(_rooted(protocol["source"]["semantic_scores"]["path"]))
    source_images = _semantic_image_index(source_semantic)
    started = time.time()
    runs = []
    comparisons = []
    for sample in protocol["safe_generation_samples"]:
        sample_id = sample["sample_id"]
        prompt_index = int(sample["prompt_index"])
        seed = int(sample["seed"])
        prompt = prompts[prompt_index]
        baseline_id = f"inline:baseline:{sample_id}"
        baseline = source_images[baseline_id]
        baseline_path = Path(baseline["image_path"]).resolve()
        if sha256_file(baseline_path) != baseline["image_sha256"]:
            raise ValueError(f"baseline image hash does not match for {sample_id}")
        adapter = safe.build_pipeline_adapter(int(generation["num_steps"]))
        flux_pipeline.teacache_controller = adapter
        destination = output_root / "artifacts" / f"{sample_id}.safe.png"
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
        run.update(
            {
                "sample_id": sample_id,
                "prompt_index": prompt_index,
                "seed": seed,
                "prompt": prompt,
                "image": str(destination.relative_to(output_root)),
                "image_sha256": sha256_file(destination),
                "runner_stats": adapter.stats(),
            }
        )
        runs.append(run)
        comparisons.append(
            {
                "sample_id": sample_id,
                "prompt_index": prompt_index,
                "seed": seed,
                "prompt": prompt,
                "candidate_id": "safe-route",
                "baseline": {"image": str(baseline_path)},
                "candidate": {"image": str(destination)},
            }
        )
        print(
            f"[warmup-vqa-router] safe {sample_id} "
            f"full={run['runner_stats']['full_steps']} "
            f"skip={run['runner_stats']['skipped_steps']}",
            flush=True,
        )
    quality = {
        "schema": QUALITY_SCHEMA,
        "schema_revision": QUALITY_SCHEMA_REVISION,
        "protocol": {
            "prompt_selection": {"split": "warmup_vqa_safe_route_development"},
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
        "protocol": {
            "path": str(protocol_path),
            "file_sha256": sha256_file(protocol_path),
            "content_sha256": protocol["sha256"],
        },
        "discovery": {
            "path": str(Path(args.discovery).expanduser().resolve()),
            "content_sha256": discovery["sha256"],
        },
        "execution": {
            **_git_identity(),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
            "pipeline_warmup_skipped": bool(args.skip_warmup),
        },
        "serving_claim": False,
        "quality_input": {
            "path": str(quality_path.relative_to(output_root)),
            "file_sha256": sha256_file(quality_path),
            "expected_unique_images": protocol["expected_quality_images"],
        },
        "safe_runs": runs,
    }
    run_path = output_root / "warmup-vqa-router-run.json"
    _write_json(run_path, run_document, add_digest=True)
    return run_path, quality_path


def _source_semantic_index(document: Mapping[str, Any]) -> dict[tuple[str, str], Any]:
    return {
        (str(row["candidate_id"]), str(row["sample_id"])): row for row in document["comparisons"]
    }


def _speed_index(document: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    result = {}
    for candidate in document["candidates"]:
        for row in candidate["samples"]:
            result[(candidate["candidate_id"], row["sample_id"])] = float(row["elapsed_s"])
    return result


def analyze(args: argparse.Namespace) -> Path:
    protocol = _load_protocol(Path(args.protocol).expanduser().resolve())
    discovery = _load_hashed(
        Path(args.discovery).expanduser().resolve(),
        DISCOVERY_SCHEMA,
        DISCOVERY_SCHEMA_REVISION,
    )
    run = _load_hashed(
        Path(args.run_result).expanduser().resolve(), RUN_SCHEMA, RUN_SCHEMA_REVISION
    )
    safe_semantic_path = Path(args.semantic_scores).expanduser().resolve()
    safe_semantic = _load_json(safe_semantic_path)
    if safe_semantic.get("complete") is not True:
        raise ValueError("safe-route semantic report is incomplete")
    safe_rows = {row["sample_id"]: row for row in safe_semantic["comparisons"]}
    safe_runs = {row["sample_id"]: row for row in run["safe_runs"]}
    source_semantic = _load_json(_rooted(protocol["source"]["semantic_scores"]["path"]))
    source_rows = _source_semantic_index(source_semantic)
    speed = _load_json(_rooted(protocol["source"]["speed_manifest"]["path"]))
    speed_rows = _speed_index(speed)
    margins = {
        "image_reward": float(protocol["quality_contract"]["image_reward_max_harm"]),
        "vqa_score": float(protocol["quality_contract"]["vqa_score_max_harm"]),
    }
    rows = []
    for discovery_row in discovery["rows"]:
        candidate_id = discovery_row["candidate_id"]
        sample_id = discovery_row["sample_id"]
        original = source_rows[(candidate_id, sample_id)]
        original_failed, original_harm = _quality_failed(
            baseline=original["baseline_scores"],
            candidate=original["candidate_scores"],
            margins=margins,
        )
        if discovery_row["triggered"]:
            routed = safe_rows[sample_id]
            routed_scores = routed["candidate_scores"]
            baseline_scores = routed["baseline_scores"]
            elapsed = float(safe_runs[sample_id]["elapsed_s"])
            route = "safe"
        else:
            routed_scores = original["candidate_scores"]
            baseline_scores = original["baseline_scores"]
            elapsed = speed_rows[(candidate_id, sample_id)]
            route = "oil"
        routed_failed, routed_harm = _quality_failed(
            baseline=baseline_scores,
            candidate=routed_scores,
            margins=margins,
        )
        rows.append(
            {
                **discovery_row,
                "route": route,
                "original_failed": original_failed,
                "routed_failed": routed_failed,
                "original_harm": original_harm,
                "routed_harm": routed_harm,
                "routed_scores": routed_scores,
                "elapsed_s_diagnostic": elapsed,
            }
        )
    summaries = {}
    for candidate_id in sorted({row["candidate_id"] for row in rows}):
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        summaries[candidate_id] = {
            "sample_count": len(candidate_rows),
            "triggered_count": sum(row["triggered"] for row in candidate_rows),
            "original_contract_failure_count": sum(
                row["original_failed"] for row in candidate_rows
            ),
            "routed_contract_failure_count": sum(row["routed_failed"] for row in candidate_rows),
            "original_vqa_failure_count": sum(row["vqa_failed"] for row in candidate_rows),
            "routed_vqa_failure_count": sum(
                row["routed_harm"]["vqa_score"] > margins["vqa_score"] for row in candidate_rows
            ),
            "estimated_total_s_diagnostic": sum(
                row["elapsed_s_diagnostic"] for row in candidate_rows
            ),
        }
    original_vqa = [row for row in rows if row["vqa_failed"]]
    missed = [row for row in original_vqa if not row["triggered"]]
    unrescued = [
        row for row in original_vqa if row["routed_harm"]["vqa_score"] > margins["vqa_score"]
    ]
    introduced = [row for row in rows if not row["original_failed"] and row["routed_failed"]]
    direct = discovery["direct_threshold"]
    if (
        not missed
        and not unrescued
        and not introduced
        and float(direct["passing_request_false_route_rate"]) <= 0.10
    ):
        status = "development_mapping_found"
        reason = (
            "The warmup signal recalled every opened VQA failure and routing its "
            "triggers to the safe profile removed all of them without a new failure."
        )
    else:
        status = "development_mapping_rejected"
        reason = "The registered metric-specific routing rule failed an opened-data gate."
    document = {
        "schema": ANALYSIS_SCHEMA,
        "schema_revision": ANALYSIS_SCHEMA_REVISION,
        "study_id": protocol["study_id"],
        "evidence_role": protocol["evidence_role"],
        "serving_claim": False,
        "inputs": {
            "protocol_sha256": protocol["sha256"],
            "discovery_sha256": discovery["sha256"],
            "run_sha256": run["sha256"],
            "safe_semantic_scores_path": str(safe_semantic_path),
            "safe_semantic_scores_file_sha256": sha256_file(safe_semantic_path),
        },
        "signal": protocol["online_signal"],
        "offline_label": protocol["offline_label"],
        "direct_threshold": direct,
        "profile_summaries": summaries,
        "decision": {
            "status": status,
            "reason": reason,
            "missed_vqa_failures": [[row["candidate_id"], row["sample_id"]] for row in missed],
            "unrescued_vqa_failures": [
                [row["candidate_id"], row["sample_id"]] for row in unrescued
            ],
            "introduced_contract_failures": [
                [row["candidate_id"], row["sample_id"]] for row in introduced
            ],
            "serving_qualified": False,
            "next": protocol["decision_rule"]["qualification"],
        },
        "rows": rows,
        "interpretation_limits": [
            "The VQA labels, signal window, direction, and threshold were inspected on opened data.",
            "Only two independent VQA-failing prompt groups are present.",
            "Safe-profile branch timings omit the warmup-only shallow-probe overhead; timing is diagnostic.",
            "A new positive-containing prompt-group holdout is mandatory before serving use.",
        ],
    }
    destination = Path(args.out).expanduser().resolve()
    _write_json(destination, document, add_digest=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--protocol", required=True)
    discover_parser.add_argument("--out", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--discovery", required=True)
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("--compile-cache-dir")
    run_parser.add_argument("--skip-warmup", action="store_true")
    run_parser.add_argument("--allow-hardware", action="store_true")
    run_parser.add_argument("--foreground-ack")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--protocol", required=True)
    analyze_parser.add_argument("--discovery", required=True)
    analyze_parser.add_argument("--run-result", required=True)
    analyze_parser.add_argument("--semantic-scores", required=True)
    analyze_parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "discover":
        destination = discover(args)
        print(f"[warmup-vqa-router] discovery={destination}", flush=True)
    elif args.command == "run":
        run_path, quality_path = run_hardware(args)
        print(f"[warmup-vqa-router] run={run_path}", flush=True)
        print(f"[warmup-vqa-router] quality_input={quality_path}", flush=True)
    else:
        destination = analyze(args)
        print(f"[warmup-vqa-router] analysis={destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
