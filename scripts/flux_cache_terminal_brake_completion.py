#!/usr/bin/env python3
"""Complete one terminal-brake time point on registered non-failure samples."""

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

from scripts.flux_cache_brake_intervention import (  # noqa: E402
    _git_identity,
    _write_json,
    canonical_sha256,
    sha256_file,
    tensor_sha256,
)
from scripts.flux_cache_terminal_brake_sweep import (  # noqa: E402
    TerminalStepPolicy,
    _build_adapter,
    _comparison_index,
    _load_json,
    _resolve_artifact,
    _rooted,
)

PROTOCOL_SCHEMA = "difflet-flux-cache-terminal-brake-completion"
PROTOCOL_SCHEMA_REVISION = 1
RUN_SCHEMA = "difflet-flux-cache-terminal-brake-completion-run"
RUN_SCHEMA_REVISION = 1
QUALITY_SCHEMA = "difflet-flux-cache-terminal-brake-completion-quality-input"
QUALITY_SCHEMA_REVISION = 1
HARDWARE_ACK = "I am completing the offline FLUX terminal brake time point"


def _load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _load_json(path)
    digest = protocol.get("sha256")
    payload = {key: value for key, value in protocol.items() if key != "sha256"}
    if digest != canonical_sha256(payload):
        raise ValueError("terminal-brake completion protocol digest does not match")
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("schema_revision") != PROTOCOL_SCHEMA_REVISION
    ):
        raise ValueError("terminal-brake completion protocol schema is unsupported")
    if protocol.get("serving_claim") is not False:
        raise ValueError("terminal-brake completion must disable serving claims")
    if protocol.get("offline_intervention_only") is not True:
        raise ValueError("terminal-brake completion must be offline-only")
    for name in (
        "source_quality",
        "source_semantic",
        "candidate",
        "timing_protocol",
        "timing_run",
        "timing_semantic",
        "collector",
        "terminal_policy_implementation",
    ):
        binding = protocol[name]
        if sha256_file(_rooted(binding["path"])) != binding["file_sha256"]:
            raise ValueError(f"registered {name} file hash does not match")
    quality = _load_json(_rooted(protocol["source_quality"]["path"]))
    semantic = _load_json(_rooted(protocol["source_semantic"]["path"]))
    if semantic.get("complete") is not True:
        raise ValueError("source semantic report is incomplete")
    sample_ids = protocol["sample_ids"]
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("protocol sample_ids must be unique and non-empty")
    if int(protocol["expected_unique_quality_images"]) != 2 * len(sample_ids):
        raise ValueError("registered unique quality image count is invalid")
    terminal_step = protocol["terminal_step"]
    if isinstance(terminal_step, bool) or not isinstance(terminal_step, int):
        raise ValueError("terminal_step must be an integer")
    return protocol, quality, semantic


def run_hardware(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.allow_hardware or args.foreground_ack != HARDWARE_ACK:
        raise RuntimeError(
            f"hardware execution requires --allow-hardware and --foreground-ack {HARDWARE_ACK!r}"
        )
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol, quality, semantic = _load_protocol(protocol_path)
    output_root = Path(args.out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    from scripts.collect_flux_cache_ab import _load_pipeline, load_adaptive_candidate
    from scripts.flux_cache_causal_repair import _run_image_only

    generation = protocol["generation_identity"]
    num_steps = int(generation["num_steps"])
    terminal_step = int(protocol["terminal_step"])
    arm = load_adaptive_candidate(_rooted(protocol["candidate"]["path"]))
    if arm.candidate_id != protocol["candidate"]["candidate_id"]:
        raise ValueError("registered candidate_id does not match")
    pipe = _load_pipeline(
        SimpleNamespace(
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
    )
    flux_pipeline = pipe.app.pipe
    source_quality_path = _rooted(protocol["source_quality"]["path"])
    source_root = source_quality_path.parent
    quality_rows = _comparison_index(quality["comparisons"])
    semantic_rows = _comparison_index(semantic["comparisons"])
    started = time.time()
    comparisons = []
    samples = []
    for sample_id in protocol["sample_ids"]:
        source = quality_rows[sample_id]
        scores = semantic_rows[sample_id]
        source_harm = -float(scores["candidate_minus_baseline"]["vqa_score"])
        threshold = float(protocol["source_selection_rule"]["maximum_vqa_harm"])
        if source_harm > threshold:
            raise ValueError(f"registered non-failure is a source failure: {sample_id}")
        baseline_path = _resolve_artifact(source_root, source["baseline"]["image"])
        cache_path = _resolve_artifact(source_root, source["candidate"]["image"])
        trajectory_path = _resolve_artifact(source_root, source["candidate"]["trajectory"])
        if not baseline_path.is_file() or not cache_path.is_file() or not trajectory_path.is_file():
            raise ValueError(f"source artifacts are missing for {sample_id}")
        import torch

        source_trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=True)
        policy = TerminalStepPolicy(arm.config, terminal_step=terminal_step)
        adapter = _build_adapter(arm, num_steps=num_steps, policy=policy)
        flux_pipeline.teacache_controller = adapter
        flux_pipeline._tc_record = False
        flux_pipeline._tc_output_dynamics_record = False
        destination = output_root / "artifacts" / f"{sample_id}.terminal-step-29.png"
        run = _run_image_only(
            pipe,
            flux_pipeline,
            prompt=str(source["prompt"]),
            seed=int(source["seed"]),
            num_steps=num_steps,
            height=int(generation["height"]),
            width=int(generation["width"]),
            guidance_scale=float(generation["guidance_scale"]),
            image_path=destination,
        )
        policy.validate_complete()
        prefix_index = terminal_step - 1
        actual_prefix_hash = tensor_sha256(flux_pipeline._tc_last_trajectory[prefix_index])
        source_prefix_hash = tensor_sha256(source_trajectory[prefix_index])
        if actual_prefix_hash != source_prefix_hash:
            raise RuntimeError(f"terminal branch diverged before {sample_id}/step-{terminal_step}")
        stats = adapter.stats()
        run.update(
            {
                "sample_id": sample_id,
                "prompt_index": int(source["prompt_index"]),
                "seed": int(source["seed"]),
                "prompt": str(source["prompt"]),
                "source_vqa_harm": source_harm,
                "terminal_step": terminal_step,
                "image": str(destination.relative_to(output_root)),
                "runner_stats": stats,
                "prefix_step_index": prefix_index,
                "prefix_latent_sha256": actual_prefix_hash,
                "prefix_matches_continue_cache": True,
            }
        )
        samples.append(run)
        comparisons.append(
            {
                "sample_id": sample_id,
                "prompt_index": int(source["prompt_index"]),
                "seed": int(source["seed"]),
                "prompt": str(source["prompt"]),
                "candidate_id": "terminal-step-29",
                "baseline": {"image": str(baseline_path)},
                "candidate": {"image": str(destination)},
            }
        )
        print(
            f"[terminal-brake-completion] {sample_id} "
            f"full={stats['full_steps']} skip={stats['skipped_steps']}",
            flush=True,
        )
    quality_output = {
        "schema": QUALITY_SCHEMA,
        "schema_revision": QUALITY_SCHEMA_REVISION,
        "protocol": {
            "prompt_selection": {"split": protocol["quality_split"]},
            "study_id": protocol["study_id"],
            "protocol_sha256": protocol["sha256"],
            "serving_claim": False,
            "offline_intervention_only": True,
        },
        "comparisons": comparisons,
    }
    quality_output_path = output_root / "quality-input.json"
    _write_json(quality_output_path, quality_output, add_digest=False)
    run_document = {
        "schema": RUN_SCHEMA,
        "schema_revision": RUN_SCHEMA_REVISION,
        "study_id": protocol["study_id"],
        "serving_claim": False,
        "offline_intervention_only": True,
        "protocol": {
            "path": str(protocol_path),
            "file_sha256": sha256_file(protocol_path),
            "content_sha256": protocol["sha256"],
        },
        "execution": {
            **_git_identity(),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
            "pipeline_warmup_skipped": bool(args.skip_warmup),
        },
        "quality_input": {
            "path": str(quality_output_path.relative_to(output_root)),
            "file_sha256": sha256_file(quality_output_path),
            "expected_unique_images": int(protocol["expected_unique_quality_images"]),
        },
        "samples": samples,
    }
    run_path = output_root / "terminal-brake-completion-run.json"
    _write_json(run_path, run_document, add_digest=True)
    return run_path, quality_output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--compile-cache-dir")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--foreground-ack")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_path, quality_path = run_hardware(args)
    print(f"[terminal-brake-completion] run={run_path}", flush=True)
    print(f"[terminal-brake-completion] quality_input={quality_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
