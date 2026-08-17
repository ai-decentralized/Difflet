"""Collect frozen FLUX cache evidence against a full-compute baseline.

This module is intentionally narrower than the historical experiment collector:
each confirmation call accepts one frozen rung of an ascending static-budget
ladder, records only the decoded images needed by the semantic gate, and emits
the two manifests consumed by profile qualification. Arbitrary candidate
sweeps, spatial probes, and learned online-signal collection belong to archived
research code.

The calibration entry point is deliberately separate from confirmation. It
accepts no cache candidate and records only the full-compute trajectories
needed by deterministic schedule derivation. Candidate budgets are enumerated
later from structural feasibility; calibration neither derives a budget from
speed nor semantically scores candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from difflet.pipeline.cache import (
    ANCHOR_ERROR_TRACE_SCHEMA,
    ANCHOR_ERROR_TRACE_SCHEMA_REVISION,
)
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
TRAJECTORY_INPUT_SCHEMA = "difflet-flux-cache-trajectory-input-v1"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_LABEL = "flux"
DEFAULT_SEEDS = (0, 1)
ROOT = Path(__file__).resolve().parents[3]


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
    save_trajectory: bool = False,
) -> dict[str, Any]:
    """Run one request and persist its image plus optional baseline trajectory."""

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
    artifacts = {
        "image": image_path.relative_to(output_root).as_posix(),
        "image_sha256": sha256_file(image_path),
    }
    if save_trajectory:
        trajectory_values = getattr(pipe.app.pipe, "_tc_last_trajectory", None)
        if not isinstance(trajectory_values, list) or len(trajectory_values) != num_steps:
            raise RuntimeError(
                "FLUX pipeline did not expose the complete baseline trajectory: "
                f"expected {num_steps} tensors"
            )
        trajectory = torch.stack([value.detach().to("cpu") for value in trajectory_values])
        for name, tensor in (("trajectory", trajectory), ("final_latent", trajectory[-1])):
            path = artifact_dir / f"{sample['sample_id']}.{name.replace('_', '-')}.pt"
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(tensor, temporary)
            os.replace(temporary, path)
            artifacts[name] = path.relative_to(output_root).as_posix()
            artifacts[f"{name}_sha256"] = sha256_file(path)
    return {
        "sample_id": sample["sample_id"],
        "elapsed_s": float(elapsed),
        "artifacts": artifacts,
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


def _registration_phase_boundary(arm: PhasedCandidateArm) -> tuple[int | None, str]:
    """Resolve the registered middle/tail boundary without trusting an unbound path."""

    horizon_path = Path(str(arm.horizon_ref.get("path", ""))).expanduser()
    try:
        horizon = json.loads(horizon_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "unavailable"
    optimizer = horizon.get("optimizer") if isinstance(horizon, dict) else None
    if isinstance(optimizer, Mapping):
        value = optimizer.get("phase_boundary_step")
        if isinstance(value, int) and not isinstance(value, bool):
            return value, str(horizon_path.resolve())

    registration = horizon.get("registration") if isinstance(horizon, dict) else None
    if not isinstance(registration, Mapping):
        return None, "unavailable"
    candidates: list[Path] = []
    path_value = registration.get("path")
    if isinstance(path_value, str):
        candidates.append(Path(path_value).expanduser())
    content_digest = registration.get("content_sha256")
    if isinstance(content_digest, str) and len(content_digest) == 64:
        for path in (ROOT / "benchmark" / "flux_cache").glob("**/*.json"):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(document, dict) and document.get("sha256") == content_digest:
                candidates.append(path)
                break

    expected_file_digest = registration.get("file_sha256")
    for path in candidates:
        if not path.is_file():
            continue
        if (
            isinstance(expected_file_digest, str)
            and len(expected_file_digest) == 64
            and sha256_file(path) != expected_file_digest
        ):
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        optimizer = document.get("optimizer") if isinstance(document, dict) else None
        value = optimizer.get("phase_boundary_step") if isinstance(optimizer, Mapping) else None
        if isinstance(value, int) and not isinstance(value, bool):
            return value, str(path.resolve())
    return None, "unavailable"


def bind_anchor_error_trace(
    trace: Mapping[str, Any],
    arm: PhasedCandidateArm,
) -> dict[str, Any]:
    """Bind a runtime trace to the exact candidate and its scheduling phases."""

    if not isinstance(trace, Mapping):
        raise ValueError("anchor-error trace must be a mapping")
    expected = {
        "schema",
        "schema_revision",
        "measurement",
        "path_semantics",
        "physical_rollback_attempts_included",
        "entries",
    }
    if set(trace) != expected:
        raise ValueError("anchor-error trace fields do not match the schema")
    if (
        trace.get("schema") != ANCHOR_ERROR_TRACE_SCHEMA
        or trace.get("schema_revision") != ANCHOR_ERROR_TRACE_SCHEMA_REVISION
        or trace.get("physical_rollback_attempts_included") is not False
    ):
        raise ValueError("anchor-error trace schema or path semantics are unsupported")
    entries = trace.get("entries")
    if not isinstance(entries, list):
        raise ValueError("anchor-error trace entries must be a list")

    num_steps = int(arm.policy["num_steps"])
    warmup_steps = int(arm.policy["warmup_steps"])
    cooldown_steps = int(arm.policy["cooldown_steps"])
    phase_boundary, phase_source = _registration_phase_boundary(arm)
    bound_entries: list[dict[str, Any]] = []
    previous_measured_step = -1
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("anchor-error trace entry must be a mapping")
        row = deepcopy(dict(entry))
        anchor_step = row.get("anchor_step_index")
        previous_anchor = row.get("previous_anchor_step_index")
        if (
            isinstance(anchor_step, bool)
            or not isinstance(anchor_step, int)
            or not 0 <= anchor_step < num_steps
            or anchor_step <= previous_measured_step
            or row.get("num_steps") != num_steps
        ):
            raise ValueError("anchor-error trace step coordinates are invalid")
        if previous_anchor is not None and (
            isinstance(previous_anchor, bool)
            or not isinstance(previous_anchor, int)
            or not 0 <= previous_anchor < anchor_step
        ):
            raise ValueError("anchor-error trace previous anchor is invalid")
        estimate_steps = row.get("estimate_step_indices")
        expected_estimates = (
            [] if previous_anchor is None else list(range(previous_anchor + 1, anchor_step))
        )
        if estimate_steps != expected_estimates:
            raise ValueError("anchor-error trace does not cover its exact segment")
        z = row.get("endpoint_z")
        if z is not None and (
            isinstance(z, bool) or not math.isfinite(float(z)) or float(z) < 0.0
        ):
            raise ValueError("anchor-error trace endpoint z is invalid")

        if previous_anchor is None:
            region = "bootstrap"
        elif anchor_step < warmup_steps:
            region = "warmup"
        elif cooldown_steps and anchor_step >= num_steps - cooldown_steps:
            region = "cooldown"
        elif phase_boundary is None:
            region = "cacheable_unresolved_phase"
        elif previous_anchor < phase_boundary:
            region = "middle"
        else:
            region = "tail"
        row["policy_region"] = region
        bound_entries.append(row)
        previous_measured_step = anchor_step

    result = deepcopy(dict(trace))
    result["candidate_binding"] = {
        "candidate_id": arm.candidate_id,
        "candidate_content_sha256": arm.content_sha256,
        "candidate_file_sha256": arm.file_sha256,
        "horizon_ref": dict(arm.horizon_ref),
    }
    result["phase_binding"] = {
        "warmup_steps": warmup_steps,
        "cooldown_steps": cooldown_steps,
        "derivation_phase_boundary_step": phase_boundary,
        "derivation_phase_boundary_source": phase_source,
        "gap_phase_uses_previous_anchor": True,
    }
    result["entries"] = bound_entries
    return result


def build_manifests(
    *,
    identity: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    arm: PhasedCandidateArm | Sequence[PhasedCandidateArm],
    baseline_runs: Sequence[Mapping[str, Any]],
    candidate_runs: (Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Mapping[str, Any]]]),
    started_at: str,
    completed_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build paired quality and speed manifests for one or more frozen arms."""

    num_steps = strict_positive_int(identity.get("num_steps"), "identity.num_steps")
    arms = (arm,) if isinstance(arm, PhasedCandidateArm) else tuple(arm)
    if not arms or len({item.candidate_id for item in arms}) != len(arms):
        raise ValueError("candidate profiles must be nonempty and unique")
    runs_by_candidate = (
        candidate_runs
        if isinstance(candidate_runs, Mapping)
        else {arms[0].candidate_id: candidate_runs}
    )
    sample_by_id = {row["sample_id"]: row for row in samples}
    baseline_by_id = {row["sample_id"]: row for row in baseline_runs}
    expected_ids = set(sample_by_id)
    if (
        len(sample_by_id) != len(samples)
        or len(baseline_by_id) != len(baseline_runs)
        or set(baseline_by_id) != expected_ids
    ):
        raise ValueError("baseline records do not match the prompt/seed sample matrix")

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
    comparisons, definitions, speed_rows = [], [], []
    trace_presence: list[bool] = []
    for item in arms:
        rows = tuple(runs_by_candidate.get(item.candidate_id, ()))
        rows_by_id = {row["sample_id"]: row for row in rows}
        if len(rows_by_id) != len(rows) or set(rows_by_id) != expected_ids:
            raise ValueError("candidate records do not match the prompt/seed sample matrix")
        for row in rows:
            stats = row.get("runner_stats")
            if not isinstance(stats, dict):
                raise ValueError("candidate record is missing runner statistics")
            if any(
                isinstance(stats.get(key), bool)
                or not isinstance(stats.get(key), int)
                or stats[key] < 0
                for key in ("full_steps", "skipped_steps", "consecutive_skip_vetoes")
            ):
                raise ValueError("candidate runner statistics are invalid")
            if stats["full_steps"] + stats["skipped_steps"] != num_steps:
                raise ValueError("candidate runner steps do not sum to num_steps")
            trace = row.get("anchor_error_trace")
            trace_presence.append(trace is not None)
            if trace is not None:
                if not isinstance(trace, Mapping):
                    raise ValueError("candidate anchor-error trace is invalid")
                entries = trace.get("entries")
                binding = trace.get("candidate_binding")
                if (
                    not isinstance(entries, list)
                    or len(entries) != stats["full_steps"]
                    or not isinstance(binding, Mapping)
                    or binding.get("candidate_id") != item.candidate_id
                    or binding.get("candidate_content_sha256") != item.content_sha256
                ):
                    raise ValueError("candidate anchor-error trace binding is invalid")
        definition = {
            "candidate_id": item.candidate_id,
            "policy": item.policy_spec(),
            "predictor": item.predictor_spec(),
        }
        definitions.append(definition)
        for sample_id, sample in sample_by_id.items():
            comparison = {
                **dict(sample),
                "candidate_id": item.candidate_id,
                "baseline": dict(baseline_by_id[sample_id]["artifacts"]),
                "candidate": dict(rows_by_id[sample_id]["artifacts"]),
            }
            trace = rows_by_id[sample_id].get("anchor_error_trace")
            if trace is not None:
                comparison["anchor_error_trace"] = deepcopy(trace)
            comparisons.append(comparison)
        candidate_total = total_duration(rows, item.candidate_id)
        speed_rows.append(
            {
                **definition,
                "total_s": candidate_total,
                "measured_speedup": baseline_total / candidate_total,
                "hardware_measured": True,
                "runner_stats": aggregate_runner_stats(rows),
                "samples": [
                    {
                        "sample_id": row["sample_id"],
                        "elapsed_s": float(row["elapsed_s"]),
                        "runner_stats": dict(row["runner_stats"]),
                        **(
                            {
                                "anchor_error_measurement_count": len(
                                    row["anchor_error_trace"]["entries"]
                                )
                            }
                            if row.get("anchor_error_trace") is not None
                            else {}
                        ),
                    }
                    for row in rows
                ],
            }
        )
    if trace_presence and any(trace_presence) and not all(trace_presence):
        raise ValueError("candidate anchor-error traces must be collected for every sample")
    common = {
        **dict(identity),
        "hardware_measured": True,
        "anchor_error_traces_collected": bool(trace_presence and all(trace_presence)),
        "started_at": started_at,
        "completed_at": completed_at,
    }
    quality = {
        "schema": QUALITY_INPUT_SCHEMA,
        **common,
        "candidates": definitions,
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
        "candidates": speed_rows,
    }
    return quality, speed


def load_reusable_baseline(
    quality_path: Path,
    speed_path: Path,
    *,
    protocol: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    """Load one hash-bound baseline without repeating full-compute generation."""

    quality_path = Path(quality_path).expanduser().resolve()
    speed_path = Path(speed_path).expanduser().resolve()
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    speed = json.loads(speed_path.read_text(encoding="utf-8"))
    if (
        not isinstance(quality, dict)
        or quality.get("schema") != QUALITY_INPUT_SCHEMA
        or quality.get("hardware_measured") is not True
        or quality.get("protocol") != protocol
    ):
        raise ValueError("reusable baseline quality manifest is incompatible")
    if (
        not isinstance(speed, dict)
        or speed.get("schema") != SPEEDUP_CANDIDATES_SCHEMA
        or speed.get("hardware_measured") is not True
        or speed.get("protocol") != protocol
    ):
        raise ValueError("reusable baseline speed manifest is incompatible")

    expected = {str(row["sample_id"]): dict(row) for row in samples}
    if len(expected) != len(samples):
        raise ValueError("reusable baseline sample matrix contains duplicates")
    artifacts_by_id: dict[str, dict[str, Any]] = {}
    comparisons = quality.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("reusable baseline quality manifest has no comparisons")
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("reusable baseline comparison is malformed")
        sample_id = comparison.get("sample_id")
        sample = expected.get(str(sample_id))
        if sample is None or any(
            comparison.get(key) != sample[key]
            for key in ("prompt_index", "prompt", "seed")
        ):
            raise ValueError("reusable baseline sample identity differs")
        artifacts = comparison.get("baseline")
        if not isinstance(artifacts, dict):
            raise ValueError("reusable baseline artifacts are malformed")
        previous = artifacts_by_id.setdefault(str(sample_id), dict(artifacts))
        if previous != artifacts:
            raise ValueError("reusable baseline artifacts disagree across candidates")
    if set(artifacts_by_id) != set(expected):
        raise ValueError("reusable baseline artifacts do not cover the sample matrix")

    baseline = speed.get("baseline")
    timing_rows = baseline.get("samples") if isinstance(baseline, dict) else None
    if not isinstance(timing_rows, list):
        raise ValueError("reusable baseline timing manifest is malformed")
    elapsed_by_id: dict[str, float] = {}
    for row in timing_rows:
        if not isinstance(row, dict) or str(row.get("sample_id")) not in expected:
            raise ValueError("reusable baseline timing row is malformed")
        sample_id = str(row["sample_id"])
        elapsed = float(row.get("elapsed_s"))
        if sample_id in elapsed_by_id or not math.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError("reusable baseline timing row is invalid")
        elapsed_by_id[sample_id] = elapsed
    if set(elapsed_by_id) != set(expected):
        raise ValueError("reusable baseline timings do not cover the sample matrix")
    baseline_total = float(baseline.get("total_s"))
    if not math.isfinite(baseline_total) or not math.isclose(
        baseline_total,
        sum(elapsed_by_id.values()),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("reusable baseline total duration is invalid")

    runs = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        artifacts = artifacts_by_id[sample_id]
        relative = artifacts.get("image")
        digest = artifacts.get("image_sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("reusable baseline image path is invalid")
        image_path = (quality_path.parent / relative).resolve()
        if (
            not image_path.is_file()
            or not isinstance(digest, str)
            or sha256_file(image_path) != digest
        ):
            raise ValueError("reusable baseline image binding is invalid")
        runs.append(
            {
                "sample_id": sample_id,
                "elapsed_s": elapsed_by_id[sample_id],
                "artifacts": {
                    "image": Path(os.path.relpath(image_path, output_root)).as_posix(),
                    "image_sha256": digest,
                },
            }
        )
    return runs


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


def _collect_profiles(
    args: argparse.Namespace,
    arms: Sequence[PhasedCandidateArm],
    *,
    label: str,
    save_baseline_trajectories: bool,
) -> tuple[Path, Path] | Path:
    if args.model_id != MODEL_ID:
        raise ValueError(f"this collector is FLUX-only; expected --model-id {MODEL_ID!r}")
    if not args.model_revision:
        raise ValueError(f"{label} requires an exact --model-revision")
    num_steps = strict_positive_int(args.num_steps, "num_steps")
    height = strict_positive_int(args.height, "height")
    width = strict_positive_int(args.width, "width")
    strict_positive_int(args.tp_degree, "tp_degree")
    guidance_scale = float(args.guidance_scale)
    if not math.isfinite(guidance_scale):
        raise ValueError("guidance_scale must be finite")
    for arm in arms:
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
        cache_coordinate=arms[0].coord if arms else "index",
        pipeline_warmup_enabled=not bool(args.skip_warmup),
    )
    print(
        f"[flux-cache-{label}] protocol={protocol['sha256']} "
        f"profiles={','.join(arm.candidate_id for arm in arms) if arms else 'none'} "
        f"samples={len(samples)}",
        flush=True,
    )

    reuse_quality = getattr(args, "baseline_quality_manifest", None)
    reuse_speed = getattr(args, "baseline_speed_manifest", None)
    if bool(reuse_quality) != bool(reuse_speed):
        raise ValueError("baseline reuse requires both quality and speed manifests")
    if reuse_quality and not arms:
        raise ValueError("baseline-only calibration cannot reuse confirmation evidence")
    if reuse_quality:
        baseline_runs = load_reusable_baseline(
            Path(reuse_quality),
            Path(reuse_speed),
            protocol=protocol,
            samples=samples,
            output_root=output_root,
        )
        print(
            f"[flux-cache-{label}] reused baseline from {reuse_quality}",
            flush=True,
        )
    else:
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
                save_trajectory=save_baseline_trajectories,
            )
            stats = baseline_adapter.stats()
            if stats["full_steps"] != num_steps or stats["skipped_steps"] != 0:
                raise RuntimeError("baseline adapter did not execute every denoise step")
            baseline_runs.append(run)
            print(
                f"[flux-cache-{label}] baseline {sample['sample_id']} "
                f"{run['elapsed_s']:.3f}s",
                flush=True,
            )

    if not arms:
        trajectories = []
        by_sample = {row["sample_id"]: row for row in baseline_runs}
        for sample in samples:
            artifacts = by_sample[sample["sample_id"]]["artifacts"]
            trajectories.append(
                {
                    **dict(sample),
                    "path": artifacts["trajectory"],
                    "file_sha256": artifacts["trajectory_sha256"],
                }
            )
        manifest = {
            "schema": TRAJECTORY_INPUT_SCHEMA,
            "protocol": protocol,
            "prompt_count": len(prompt_selection.prompts),
            "seed_count": len(seeds),
            "trajectory_count": len(trajectories),
            "trajectories": trajectories,
            "semantic_labels_collected": False,
            "hardware_measured": True,
            "started_at": started_at,
            "completed_at": utc_now(),
        }
        path = output_root / "trajectory-input-v1.json"
        write_json(path, manifest)
        print(f"[flux-cache-{label}] trajectory manifest: {path}", flush=True)
        return path

    candidate_runs: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        adapter = arm.build_pipeline_adapter(num_steps)
        flux_pipeline.teacache_controller = adapter
        rows = []
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
            run["runner_stats"] = adapter.stats()
            run["anchor_error_trace"] = bind_anchor_error_trace(
                adapter.anchor_error_trace(),
                arm,
            )
            rows.append(run)
            print(
                f"[flux-cache-{label}] {arm.candidate_id} {sample['sample_id']} "
                f"{run['elapsed_s']:.3f}s skip={run['runner_stats']['skipped_steps']}",
                flush=True,
            )
        candidate_runs[arm.candidate_id] = rows

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
        arm=arms,
        baseline_runs=baseline_runs,
        candidate_runs=candidate_runs,
        started_at=started_at,
        completed_at=utc_now(),
    )
    if reuse_quality:
        baseline_source = {
            "quality_manifest": {
                "path": str(Path(reuse_quality).expanduser().resolve()),
                "file_sha256": sha256_file(Path(reuse_quality).expanduser().resolve()),
            },
            "speed_manifest": {
                "path": str(Path(reuse_speed).expanduser().resolve()),
                "file_sha256": sha256_file(Path(reuse_speed).expanduser().resolve()),
            },
        }
        quality["baseline_source"] = baseline_source
        speed["baseline_source"] = baseline_source
    quality_path = output_root / "quality-input-v2.json"
    speed_path = output_root / "speedup-candidates-v1.json"
    write_json(quality_path, quality)
    write_json(speed_path, speed)
    print(f"[flux-cache-{label}] quality manifest: {quality_path}", flush=True)
    print(f"[flux-cache-{label}] speed manifest: {speed_path}", flush=True)
    return quality_path, speed_path


def collect_confirmation(
    args: argparse.Namespace,
    arm: PhasedCandidateArm | Sequence[PhasedCandidateArm],
) -> tuple[Path, Path]:
    """Run confirmation for one or more explicitly supplied static rungs."""

    arms = (arm,) if isinstance(arm, PhasedCandidateArm) else tuple(arm)
    return _collect_profiles(
        args,
        arms,
        label="confirmation",
        save_baseline_trajectories=False,
    )


def collect_calibration(
    args: argparse.Namespace,
) -> Path:
    """Collect only full-compute trajectories for quality-led derivation."""

    return _collect_profiles(
        args,
        (),
        label="calibration",
        save_baseline_trajectories=True,
    )


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
    parser.add_argument("--baseline-quality-manifest", default=None)
    parser.add_argument("--baseline-speed-manifest", default=None)
    return parser.parse_args(argv)
