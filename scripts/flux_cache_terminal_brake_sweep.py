#!/usr/bin/env python3
"""Collect a registered terminal-brake timing sweep for FLUX cache failures."""

from __future__ import annotations

import argparse
import json
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
    CacheRunner,
    CacheSession,
    QualityRecoveryConfig,
    QualityRecoveryGuard,
    TaylorSeerPredictor,
    TeaCacheControllerAdapter,
)
from scripts.flux_cache_brake_intervention import (  # noqa: E402
    _git_identity,
    _write_json,
    canonical_sha256,
    sha256_file,
    tensor_sha256,
)

PROTOCOL_SCHEMA = "difflet-flux-cache-terminal-brake-sweep"
PROTOCOL_SCHEMA_REVISION = 1
RUN_SCHEMA = "difflet-flux-cache-terminal-brake-sweep-run"
RUN_SCHEMA_REVISION = 1
QUALITY_SCHEMA = "difflet-flux-cache-terminal-brake-sweep-quality-input"
QUALITY_SCHEMA_REVISION = 1
HARDWARE_ACK = "I am running the offline FLUX terminal brake timing sweep"


class TerminalStepPolicy(AdaptiveAnchorPolicy):
    """Follow the configured cache prefix, then permanently compute full steps."""

    def __init__(self, config: Any, *, terminal_step: int) -> None:
        if isinstance(terminal_step, bool) or not isinstance(terminal_step, int):
            raise TypeError("terminal_step must be an integer")
        if terminal_step < int(config.warmup_steps):
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
            raise RuntimeError("terminal brake was never applied")

    def stats(self) -> dict[str, Any]:
        result = super().stats()
        result.update(
            {
                "terminal_step": self.terminal_step,
                "terminal_applied": self._terminal_applied,
            }
        )
        return result


def _rooted(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return document


def _load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _load_json(path)
    digest = protocol.get("sha256")
    payload = {key: value for key, value in protocol.items() if key != "sha256"}
    if digest != canonical_sha256(payload):
        raise ValueError("terminal-brake sweep protocol digest does not match")
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("schema_revision") != PROTOCOL_SCHEMA_REVISION
    ):
        raise ValueError("terminal-brake sweep protocol schema is unsupported")
    if protocol.get("serving_claim") is not False:
        raise ValueError("terminal-brake sweep must disable serving claims")
    if protocol.get("offline_intervention_only") is not True:
        raise ValueError("terminal-brake sweep must be offline-only")
    for name in ("source_quality", "source_semantic", "candidate", "collector"):
        binding = protocol[name]
        binding_path = _rooted(binding["path"])
        if sha256_file(binding_path) != binding["file_sha256"]:
            raise ValueError(f"registered {name} file hash does not match")
    quality = _load_json(_rooted(protocol["source_quality"]["path"]))
    semantic = _load_json(_rooted(protocol["source_semantic"]["path"]))
    if semantic.get("complete") is not True:
        raise ValueError("source semantic report is incomplete")
    sample_ids = protocol["sample_ids"]
    terminal_steps = protocol["terminal_steps"]
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("protocol sample_ids must be unique and non-empty")
    if (
        not isinstance(terminal_steps, list)
        or not terminal_steps
        or len(set(terminal_steps)) != len(terminal_steps)
        or terminal_steps != sorted(terminal_steps)
    ):
        raise ValueError("protocol terminal_steps must be unique and sorted")
    expected = len(sample_ids) * (len(terminal_steps) + 1)
    if int(protocol["expected_unique_quality_images"]) != expected:
        raise ValueError("registered unique quality image count is invalid")
    return protocol, quality, semantic


def _comparison_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in result:
            raise ValueError(f"duplicate source comparison for {sample_id}")
        result[sample_id] = row
    return result


def _resolve_artifact(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _build_adapter(arm: Any, *, num_steps: int, policy: TerminalStepPolicy):
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
            configuration_source="offline-terminal-brake-sweep",
        )
    )


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
    quality_path = _rooted(protocol["source_quality"]["path"])
    source_root = quality_path.parent
    quality_rows = _comparison_index(quality["comparisons"])
    semantic_rows = _comparison_index(semantic["comparisons"])
    started = time.time()
    comparisons = []
    samples = []
    for sample_id in protocol["sample_ids"]:
        source = quality_rows[sample_id]
        scores = semantic_rows[sample_id]
        harm = -float(scores["candidate_minus_baseline"]["vqa_score"])
        if harm <= float(protocol["source_failure_rule"]["vqa_harm_threshold"]):
            raise ValueError(f"registered source failure is no longer a failure: {sample_id}")
        baseline_path = _resolve_artifact(source_root, source["baseline"]["image"])
        cache_path = _resolve_artifact(source_root, source["candidate"]["image"])
        trajectory_path = _resolve_artifact(source_root, source["candidate"]["trajectory"])
        if not baseline_path.is_file() or not cache_path.is_file() or not trajectory_path.is_file():
            raise ValueError(f"source artifacts are missing for {sample_id}")
        import torch

        source_trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=True)
        terminal_runs = []
        for terminal_step in protocol["terminal_steps"]:
            terminal_step = int(terminal_step)
            policy = TerminalStepPolicy(arm.config, terminal_step=terminal_step)
            adapter = _build_adapter(arm, num_steps=num_steps, policy=policy)
            flux_pipeline.teacache_controller = adapter
            flux_pipeline._tc_record = False
            flux_pipeline._tc_output_dynamics_record = False
            destination = (
                output_root
                / "artifacts"
                / sample_id
                / f"terminal-step-{terminal_step:02d}.png"
            )
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
            actual_prefix_hash = tensor_sha256(
                flux_pipeline._tc_last_trajectory[prefix_index]
            )
            source_prefix_hash = tensor_sha256(source_trajectory[prefix_index])
            if actual_prefix_hash != source_prefix_hash:
                raise RuntimeError(
                    f"terminal branch diverged before {sample_id}/step-{terminal_step}"
                )
            stats = adapter.stats()
            run.update(
                {
                    "terminal_step": terminal_step,
                    "image": str(destination.relative_to(output_root)),
                    "runner_stats": stats,
                    "prefix_step_index": prefix_index,
                    "prefix_latent_sha256": actual_prefix_hash,
                    "prefix_matches_continue_cache": True,
                }
            )
            terminal_runs.append(run)
            comparisons.append(
                {
                    "sample_id": sample_id,
                    "prompt_index": int(source["prompt_index"]),
                    "seed": int(source["seed"]),
                    "prompt": str(source["prompt"]),
                    "candidate_id": f"terminal-step-{terminal_step:02d}",
                    "baseline": {"image": str(baseline_path)},
                    "candidate": {"image": str(destination)},
                }
            )
            print(
                f"[terminal-brake-sweep] {sample_id} step={terminal_step} "
                f"full={stats['full_steps']} skip={stats['skipped_steps']}",
                flush=True,
            )
        samples.append(
            {
                "sample_id": sample_id,
                "prompt_index": int(source["prompt_index"]),
                "seed": int(source["seed"]),
                "prompt": str(source["prompt"]),
                "source_vqa_harm": harm,
                "source_baseline_image": str(baseline_path),
                "source_cache_image": str(cache_path),
                "terminal_runs": terminal_runs,
            }
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
    run_path = output_root / "terminal-brake-sweep-run.json"
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
    print(f"[terminal-brake-sweep] run={run_path}", flush=True)
    print(f"[terminal-brake-sweep] quality_input={quality_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
