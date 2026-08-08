"""Collect one frozen FLUX cache profile against a full-compute baseline.

This module is intentionally narrower than the historical experiment collector:
it accepts exactly one already-frozen profile, records only the decoded images
needed by the semantic gate, and emits the two manifests consumed by profile
qualification. Candidate sweeps, trajectory dumps, spatial probes, and learned
online-signal collection belong to archived research code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from difflet.pipeline.cache.profile import PhasedCandidateArm
from scripts.flux_cache_protocol import (
    DEFAULT_PROMPT_SUITE_PATH,
    PromptSelection,
    build_experiment_protocol,
    inline_prompt_selection,
    load_prompt_suite,
)

QUALITY_INPUT_SCHEMA = "quality-input-v2"
SPEEDUP_CANDIDATES_SCHEMA = "speedup-candidates-v1"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_LABEL = "flux"
DEFAULT_SEEDS = (0, 1)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def select_prompts(args: argparse.Namespace) -> PromptSelection:
    """Resolve either one frozen prompt split or explicit development prompts."""

    prompt_suite = getattr(args, "prompt_suite", None)
    prompts_json = getattr(args, "prompts_json", None)
    inline = getattr(args, "prompt", None)
    prompt_split = getattr(args, "prompt_split", "legacy_parity")
    custom_prompts = bool(prompts_json or inline)
    if custom_prompts:
        if prompt_suite:
            raise ValueError("--prompt-suite cannot be combined with custom prompts")
        if prompt_split != "legacy_parity":
            raise ValueError("--prompt-split cannot be combined with custom prompts")
        if prompts_json and inline:
            raise ValueError("--prompts-json and --prompt are mutually exclusive")
        if prompts_json:
            path = Path(prompts_json).expanduser().resolve()
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(f"could not read prompts JSON {path}: {error}") from error
            values = document.get("prompts") if isinstance(document, dict) else document
            if isinstance(document, dict) and set(document) != {"prompts"}:
                raise ValueError("prompts JSON object must contain only the prompts field")
            source = f"json:{path}"
        else:
            values = inline
            source = "command-line"
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("prompts must be a non-empty list")
        prompts = tuple(values)
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must contain non-empty strings")
        return inline_prompt_selection(tuple(prompt.strip() for prompt in prompts), source)

    path = Path(prompt_suite or DEFAULT_PROMPT_SUITE_PATH).expanduser().resolve()
    return load_prompt_suite(path, prompt_split)


def sample_matrix(
    prompts: Sequence[str],
    seeds: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must not contain duplicates")
    matrix: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seeds must be nonnegative integers")
            matrix.append(
                {
                    "sample_id": f"p{prompt_index:03d}-s{seed}",
                    "prompt_index": int(prompt_index),
                    "prompt": prompt,
                    "seed": int(seed),
                }
            )
    return tuple(matrix)


def prepare_output_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}; use a new directory")
    path.mkdir(parents=True, exist_ok=True)


def extract_image(result: Any):
    images = getattr(result, "images", None)
    if images is None and isinstance(result, (tuple, list)) and result:
        images = result[0]
    if isinstance(images, (tuple, list)):
        images = images[0] if images else None
    if images is None or not callable(getattr(images, "save", None)):
        raise RuntimeError("FLUX pipeline did not return a decoded PIL image")
    return images


def run_image_sample(
    pipe: Any,
    *,
    sample: Mapping[str, Any],
    num_steps: int,
    height: int,
    width: int,
    guidance_scale: float,
    artifact_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run one request and persist only the image used by the quality gate."""

    import torch

    generator = torch.Generator().manual_seed(int(sample["seed"]))
    started = time.perf_counter()
    result = pipe(
        prompt=sample["prompt"],
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        output_type="pil",
        generator=generator,
    )
    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("pipeline returned an invalid wall-clock duration")
    image = extract_image(result)
    image_path = artifact_dir / f"{sample['sample_id']}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    return {
        "sample_id": sample["sample_id"],
        "elapsed_s": float(elapsed),
        "artifacts": {
            "image": image_path.relative_to(output_root).as_posix(),
            "image_sha256": sha256_file(image_path),
        },
    }


def aggregate_runner_stats(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    aggregate: dict[str, int] = {}
    for row in rows:
        stats = row.get("runner_stats", {})
        for key, value in stats.items():
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            aggregate[key] = aggregate.get(key, 0) + value
    return aggregate


def build_manifests(
    *,
    identity: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    arm: PhasedCandidateArm,
    baseline_runs: Sequence[Mapping[str, Any]],
    candidate_runs: Sequence[Mapping[str, Any]],
    started_at: str,
    completed_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact two manifests consumed by qualification."""

    num_steps = strict_positive_int(identity.get("num_steps"), "identity.num_steps")
    sample_by_id = {row["sample_id"]: row for row in samples}
    baseline_by_id = {row["sample_id"]: row for row in baseline_runs}
    candidate_by_id = {row["sample_id"]: row for row in candidate_runs}
    expected_ids = set(sample_by_id)
    if len(sample_by_id) != len(samples):
        raise ValueError("sample matrix contains duplicate sample identifiers")
    if len(baseline_by_id) != len(baseline_runs):
        raise ValueError("baseline records contain duplicate sample identifiers")
    if len(candidate_by_id) != len(candidate_runs):
        raise ValueError("candidate records contain duplicate sample identifiers")
    if set(baseline_by_id) != expected_ids or set(candidate_by_id) != expected_ids:
        raise ValueError("collected records do not match the prompt/seed sample matrix")

    def total_duration(rows: Sequence[Mapping[str, Any]], name: str) -> float:
        total = 0.0
        for row in rows:
            elapsed = float(row["elapsed_s"])
            if not math.isfinite(elapsed) or elapsed <= 0.0:
                raise ValueError(f"{name} durations must be positive and finite")
            total += elapsed
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(f"{name} total duration must be positive and finite")
        return total

    baseline_total = total_duration(baseline_runs, "baseline")
    candidate_total = total_duration(candidate_runs, "candidate")
    for row in candidate_runs:
        stats = row.get("runner_stats")
        if not isinstance(stats, dict):
            raise ValueError("candidate record is missing runner statistics")
        for key in ("full_steps", "skipped_steps", "consecutive_skip_vetoes"):
            value = stats.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"candidate runner stat {key!r} is invalid")
        if stats["full_steps"] + stats["skipped_steps"] != num_steps:
            raise ValueError("candidate runner steps do not sum to num_steps")

    comparisons = []
    for sample_id, sample in sample_by_id.items():
        comparisons.append(
            {
                **dict(sample),
                "candidate_id": arm.candidate_id,
                "baseline": dict(baseline_by_id[sample_id]["artifacts"]),
                "candidate": dict(candidate_by_id[sample_id]["artifacts"]),
            }
        )
    common = {
        **dict(identity),
        "hardware_measured": True,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    candidate_definition = {
        "candidate_id": arm.candidate_id,
        "policy": arm.policy_spec(),
        "predictor": arm.predictor_spec(),
    }
    quality = {
        "schema": QUALITY_INPUT_SCHEMA,
        **common,
        "candidates": [candidate_definition],
        "comparisons": comparisons,
    }
    speed = {
        "schema": SPEEDUP_CANDIDATES_SCHEMA,
        **common,
        "baseline": {
            "total_s": baseline_total,
            "samples": [
                {"sample_id": row["sample_id"], "elapsed_s": float(row["elapsed_s"])}
                for row in baseline_runs
            ],
        },
        "candidates": [
            {
                **candidate_definition,
                "total_s": candidate_total,
                "measured_speedup": baseline_total / candidate_total,
                "hardware_measured": True,
                "runner_stats": aggregate_runner_stats(candidate_runs),
                "samples": [
                    {
                        "sample_id": row["sample_id"],
                        "elapsed_s": float(row["elapsed_s"]),
                        "runner_stats": dict(row["runner_stats"]),
                    }
                    for row in candidate_runs
                ],
            }
        ],
    }
    return quality, speed


def load_pipeline(args: argparse.Namespace):
    import torch

    from difflet import DiffletParallelConfig, DiffletPipeline

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    return DiffletPipeline.from_pretrained(
        args.model_id,
        model_type="flux",
        revision=args.model_revision,
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
        dtype=dtype,
        compile_cache_dir=args.compile_cache_dir,
        height=args.height,
        width=args.width,
        force_compile=bool(args.force_compile),
        skip_warmup=bool(args.skip_warmup),
    )


def build_baseline_adapter(num_steps: int):
    from difflet.pipeline.cache import (
        CacheRunner,
        CacheSession,
        PhasedStaticPolicy,
        TaylorSeerPredictor,
        TeaCacheControllerAdapter,
    )

    session = CacheSession(
        CacheRunner(
            PhasedStaticPolicy((True,) * num_steps),
            TaylorSeerPredictor(order=1),
        ),
        num_steps=num_steps,
        configuration_source="full-compute-baseline",
        planned_anchor_steps=num_steps,
        planned_estimate_steps=0,
    )
    return TeaCacheControllerAdapter(session)


def collect_confirmation(
    args: argparse.Namespace,
    arm: PhasedCandidateArm,
) -> tuple[Path, Path]:
    """Run a paired baseline/candidate confirmation for exactly one profile."""

    if args.model_id != MODEL_ID:
        raise ValueError(f"this collector is FLUX-only; expected --model-id {MODEL_ID!r}")
    if not args.model_revision:
        raise ValueError("confirmation requires an exact --model-revision")
    num_steps = strict_positive_int(args.num_steps, "num_steps")
    height = strict_positive_int(args.height, "height")
    width = strict_positive_int(args.width, "width")
    strict_positive_int(args.tp_degree, "tp_degree")
    guidance_scale = float(args.guidance_scale)
    if not math.isfinite(guidance_scale):
        raise ValueError("guidance_scale must be finite")
    arm.build_pipeline_adapter(num_steps)
    prompt_selection = select_prompts(args)
    seeds = tuple(DEFAULT_SEEDS if args.seed is None else args.seed)
    samples = sample_matrix(prompt_selection.prompts, seeds)

    output_root = Path(args.out_dir).expanduser().resolve()
    prepare_output_directory(output_root)
    started_at = utc_now()
    pipe = load_pipeline(args)
    flux_pipeline = pipe.app.pipe
    flux_pipeline._tc_record = False
    flux_pipeline._tc_output_dynamics_record = False
    protocol = build_experiment_protocol(
        pipe=pipe,
        scheduler=flux_pipeline.scheduler,
        prompt_selection=prompt_selection,
        seeds=seeds,
        num_steps=num_steps,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        dtype=args.dtype,
        tp_degree=args.tp_degree,
        requested_model_revision=args.model_revision,
        cache_coordinate=arm.coord,
        pipeline_warmup_enabled=not bool(args.skip_warmup),
    )
    print(
        "[flux-cache-confirmation] "
        f"protocol={protocol['sha256']} candidate={arm.candidate_id} "
        f"samples={len(samples)}",
        flush=True,
    )

    baseline_adapter = build_baseline_adapter(num_steps)
    flux_pipeline.teacache_controller = baseline_adapter
    baseline_runs = []
    for sample in samples:
        run = run_image_sample(
            pipe,
            sample=sample,
            num_steps=num_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            artifact_dir=output_root / "artifacts" / "baseline",
            output_root=output_root,
        )
        stats = baseline_adapter.stats()
        if stats["full_steps"] != num_steps or stats["skipped_steps"] != 0:
            raise RuntimeError("baseline adapter did not execute every denoise step")
        baseline_runs.append(run)
        print(
            f"[flux-cache-confirmation] baseline {sample['sample_id']} " f"{run['elapsed_s']:.3f}s",
            flush=True,
        )

    candidate_adapter = arm.build_pipeline_adapter(num_steps)
    flux_pipeline.teacache_controller = candidate_adapter
    candidate_runs = []
    for sample in samples:
        run = run_image_sample(
            pipe,
            sample=sample,
            num_steps=num_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            artifact_dir=output_root / "artifacts" / arm.candidate_id,
            output_root=output_root,
        )
        run["runner_stats"] = candidate_adapter.stats()
        candidate_runs.append(run)
        print(
            f"[flux-cache-confirmation] {arm.candidate_id} {sample['sample_id']} "
            f"{run['elapsed_s']:.3f}s skip={run['runner_stats']['skipped_steps']}",
            flush=True,
        )

    identity = {
        "model": MODEL_LABEL,
        "model_id": args.model_id,
        "shape_label": f"{height}x{width}",
        "num_steps": num_steps,
        "scheduler_class": type(flux_pipeline.scheduler).__name__,
        "guidance_scale": guidance_scale,
        "prompt_count": len(prompt_selection.prompts),
        "seed_count": len(seeds),
        "sample_count": len(samples),
        "protocol": protocol,
    }
    quality, speed = build_manifests(
        identity=identity,
        samples=samples,
        arm=arm,
        baseline_runs=baseline_runs,
        candidate_runs=candidate_runs,
        started_at=started_at,
        completed_at=utc_now(),
    )
    quality_path = output_root / "quality-input-v2.json"
    speed_path = output_root / "speedup-candidates-v1.json"
    write_json(quality_path, quality)
    write_json(speed_path, speed)
    print(f"[flux-cache-confirmation] quality manifest: {quality_path}", flush=True)
    print(f"[flux-cache-confirmation] speed manifest: {speed_path}", flush=True)
    return quality_path, speed_path


def parse_confirmation_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--prompt-suite", required=True)
    parser.add_argument("--prompt-split", required=True)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--compile-cache-dir", default=None)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args(argv)
