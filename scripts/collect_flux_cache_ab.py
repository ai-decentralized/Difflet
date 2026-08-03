#!/usr/bin/env python3
"""Collect reproducible FLUX baseline/cache A/B artifacts on Trainium.

The collector deliberately does no quality scoring. It runs every candidate
against the same prompt/seed matrix, saves the tensors and decoded image needed
by the offline evaluator, and records real wall-clock timings plus CacheRunner
statistics. The two output manifests are:

* ``quality-input-v2.json``: portable, relative paths for per-sample artifacts;
* ``speedup-candidates-v1.json``: timings and runner counters for each arm.

Hardware execution is gated by an explicit acknowledgement because a default
12-arm sweep is expensive and occupies the foreground Neuron runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

QUALITY_INPUT_SCHEMA = "quality-input-v2"
SPEEDUP_CANDIDATES_SCHEMA = "speedup-candidates-v1"
FOREGROUND_ACK = "I am running FLUX cache A/B in the foreground"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_LABEL = "flux"
DEFAULT_PROMPTS = (
    "a red fox sitting in a snowy forest at dawn, sharp detail",
    "a bustling night market with neon signs and steam, cinematic",
)
DEFAULT_SEEDS = (0, 1)
DEFAULT_WARMUP_STEPS = (10, 12, 14)
DEFAULT_ANCHOR_INTERVALS = (4, 5)
DEFAULT_ORDERS = (1, 2)
CANDIDATE_LADDER_SCHEMA = "difflet-flux-cache-candidate-ladder"
CANDIDATE_LADDER_SCHEMA_REVISION = 1
ADAPTIVE_CANDIDATE_SCHEMA = "difflet-flux-cache-adaptive-candidate"
ADAPTIVE_CANDIDATE_SCHEMA_REVISION = 1
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.flux_cache_protocol import (  # noqa: E402
    DEFAULT_PROMPT_SUITE_PATH,
    PromptSelection,
    build_experiment_protocol,
    canonical_sha256,
    inline_prompt_selection,
    load_prompt_suite,
)


@dataclass(frozen=True)
class CandidateArm:
    """One controlled PeriodicAnchorPolicy × TaylorSeer experiment arm."""

    warmup_steps: int
    anchor_interval: int
    order: int
    anchor_phase: int = 1
    cooldown_steps: int = 1
    require_final_anchor: bool = True
    coord: str = "index"

    def __post_init__(self) -> None:
        from difflet.pipeline.cache import PeriodicAnchorPolicy, TaylorSeerPredictor

        PeriodicAnchorPolicy(
            anchor_interval=self.anchor_interval,
            anchor_phase=self.anchor_phase,
            warmup_steps=self.warmup_steps,
            cooldown_steps=self.cooldown_steps,
            require_final_anchor=self.require_final_anchor,
        )
        TaylorSeerPredictor(order=self.order, coord=self.coord)

    @property
    def candidate_id(self) -> str:
        final = "f1" if self.require_final_anchor else "f0"
        return (
            f"periodic-w{self.warmup_steps}-i{self.anchor_interval}"
            f"-p{self.anchor_phase}-c{self.cooldown_steps}"
            f"-o{self.order}-{self.coord}-{final}"
        )

    def policy_spec(self) -> dict[str, Any]:
        return {
            "type": "periodic_anchor",
            "anchor_interval": int(self.anchor_interval),
            "anchor_phase": int(self.anchor_phase),
            "warmup_steps": int(self.warmup_steps),
            "cooldown_steps": int(self.cooldown_steps),
            "require_final_anchor": bool(self.require_final_anchor),
        }

    def predictor_spec(self) -> dict[str, Any]:
        return {
            "type": "taylorseer",
            "order": int(self.order),
            "coord": self.coord,
        }

    def build_pipeline_adapter(self, num_steps: int, *, measurement_sink: Any = None):
        """Build the TeaCache-loop adapter for this experimental arm."""

        from difflet.pipeline.cache import (
            PeriodicAnchorPolicy,
            ResolvedCacheSession,
            TaylorSeerPredictor,
            TeaCacheControllerAdapter,
            resolve_cache_config,
        )

        if self.warmup_steps + self.cooldown_steps >= num_steps:
            raise ValueError(
                "cache-plan-v1 candidates require warmup_steps + " "cooldown_steps < num_steps"
            )
        policy = PeriodicAnchorPolicy(
            anchor_interval=self.anchor_interval,
            anchor_phase=self.anchor_phase,
            warmup_steps=self.warmup_steps,
            cooldown_steps=self.cooldown_steps,
            require_final_anchor=self.require_final_anchor,
        )
        resolved = resolve_cache_config(
            num_steps=num_steps,
            policy=policy,
            predictor=TaylorSeerPredictor(order=self.order, coord=self.coord),
            require_final_anchor=self.require_final_anchor,
        )
        return TeaCacheControllerAdapter(
            ResolvedCacheSession(resolved, measurement_sink=measurement_sink)
        )


@dataclass(frozen=True)
class AdaptiveCandidateArm:
    """One bounded adaptive-anchor arm using the same TaylorSeer predictor."""

    candidate_id: str
    config: Any
    order: int
    coord: str

    def __post_init__(self) -> None:
        from difflet.pipeline.cache import AdaptiveAnchorConfig, TaylorSeerPredictor

        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("adaptive candidate_id must be a non-empty string")
        if not isinstance(self.config, AdaptiveAnchorConfig):
            raise TypeError("adaptive config must be an AdaptiveAnchorConfig")
        TaylorSeerPredictor(order=self.order, coord=self.coord)

    def policy_spec(self) -> dict[str, Any]:
        return {
            "type": "adaptive_anchor",
            "initial_anchor_interval": self.config.initial_anchor_interval,
            "minimum_anchor_interval": self.config.minimum_anchor_interval,
            "maximum_anchor_interval": self.config.maximum_anchor_interval,
            "warmup_steps": self.config.warmup_steps,
            "cooldown_steps": self.config.cooldown_steps,
            "anchor_phase": self.config.anchor_phase,
            "tighten_error": self.config.tighten_error,
            "recovery_error": self.config.recovery_error,
            "acceleration_error": self.config.acceleration_error,
            "recovery_steps": self.config.recovery_steps,
            "disable_after_recoveries": self.config.disable_after_recoveries,
            "stable_anchors_for_acceleration": (self.config.stable_anchors_for_acceleration),
            "acceleration_start_progress": self.config.acceleration_start_progress,
            "allow_acceleration": self.config.allow_acceleration,
            "require_final_anchor": self.config.require_final_anchor,
        }

    def predictor_spec(self) -> dict[str, Any]:
        return {"type": "taylorseer", "order": self.order, "coord": self.coord}

    def build_pipeline_adapter(self, num_steps: int, *, measurement_sink: Any = None):
        from difflet.pipeline.cache import (
            AdaptiveAnchorPolicy,
            CacheRunner,
            CacheSession,
            QualityRecoveryConfig,
            QualityRecoveryGuard,
            TaylorSeerPredictor,
            TeaCacheControllerAdapter,
        )

        if self.config.warmup_steps + self.config.cooldown_steps >= num_steps:
            raise ValueError(
                "adaptive candidates require warmup_steps + cooldown_steps < num_steps"
            )
        recovery = QualityRecoveryGuard(
            QualityRecoveryConfig(
                warmup_steps=self.config.warmup_steps,
                cooldown_steps=self.config.cooldown_steps,
                require_final_anchor=self.config.require_final_anchor,
            )
        )
        runner = CacheRunner(
            AdaptiveAnchorPolicy(self.config),
            TaylorSeerPredictor(order=self.order, coord=self.coord),
            recovery=recovery,
            measurement_sink=measurement_sink,
        )
        session = CacheSession(
            runner,
            num_steps=num_steps,
            configuration_source="adaptive-anchor",
        )
        return TeaCacheControllerAdapter(session)


@dataclass(frozen=True)
class CandidateLadder:
    """A verified ordered set of explicitly paired experiment arms."""

    source_path: Path
    ladder_id: str
    labels: tuple[str, ...]
    arms: tuple[CandidateArm, ...]
    content_sha256: str
    file_sha256: str


def build_candidate_arms(
    *,
    warmup_steps: Sequence[int] = DEFAULT_WARMUP_STEPS,
    anchor_intervals: Sequence[int] = DEFAULT_ANCHOR_INTERVALS,
    orders: Sequence[int] = DEFAULT_ORDERS,
    anchor_phase: int = 1,
    cooldown_steps: int = 1,
    require_final_anchor: bool = True,
    coord: str = "index",
) -> tuple[CandidateArm, ...]:
    """Build the deterministic Cartesian sweep (12 arms by default)."""

    arms = tuple(
        CandidateArm(
            warmup_steps=warmup,
            anchor_interval=interval,
            order=order,
            anchor_phase=anchor_phase,
            cooldown_steps=cooldown_steps,
            require_final_anchor=require_final_anchor,
            coord=coord,
        )
        for warmup in warmup_steps
        for interval in anchor_intervals
        for order in orders
    )
    if not arms:
        raise ValueError("candidate sweep must contain at least one arm")
    candidate_ids = [arm.candidate_id for arm in arms]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate sweep produced duplicate candidate identifiers")
    return arms


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _strict_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_candidate_ladder(path: Path) -> CandidateLadder:
    """Load one strict, digest-bearing list of explicitly paired candidates."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read candidate ladder {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("candidate ladder must contain a JSON object")
    expected_keys = {
        "schema",
        "schema_revision",
        "ladder_id",
        "candidates",
        "sha256",
    }
    if set(document) != expected_keys:
        raise ValueError(
            "candidate ladder fields do not match the protocol: "
            f"expected {sorted(expected_keys)}, got {sorted(document)}"
        )
    if document["schema"] != CANDIDATE_LADDER_SCHEMA:
        raise ValueError(f"candidate ladder schema must be {CANDIDATE_LADDER_SCHEMA!r}")
    if document["schema_revision"] != CANDIDATE_LADDER_SCHEMA_REVISION:
        raise ValueError(
            "candidate ladder schema_revision must be " f"{CANDIDATE_LADDER_SCHEMA_REVISION}"
        )
    ladder_id = document["ladder_id"]
    if not isinstance(ladder_id, str) or not ladder_id or ladder_id != ladder_id.strip():
        raise ValueError("candidate ladder ladder_id must be a non-empty trimmed string")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if document["sha256"] != canonical_sha256(payload):
        raise ValueError("candidate ladder sha256 does not match its contents")
    rows = document["candidates"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate ladder candidates must be a non-empty list")
    candidate_keys = {
        "label",
        "warmup_steps",
        "anchor_interval",
        "order",
        "anchor_phase",
        "cooldown_steps",
        "require_final_anchor",
        "coord",
    }
    labels: list[str] = []
    arms: list[CandidateArm] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != candidate_keys:
            raise ValueError(
                f"candidate ladder candidates[{index}] fields do not match the protocol"
            )
        label = row["label"]
        if not isinstance(label, str) or not label or label != label.strip():
            raise ValueError(f"candidate ladder candidates[{index}].label is invalid")
        if label in labels:
            raise ValueError("candidate ladder contains duplicate labels")
        labels.append(label)
        for field in (
            "warmup_steps",
            "anchor_interval",
            "order",
            "anchor_phase",
            "cooldown_steps",
        ):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"candidate ladder candidates[{index}].{field} " "must be a nonnegative integer"
                )
        if row["warmup_steps"] <= 0 or row["anchor_interval"] <= 0 or row["order"] <= 0:
            raise ValueError(
                f"candidate ladder candidates[{index}] warmup, interval, and order "
                "must be positive"
            )
        if not isinstance(row["require_final_anchor"], bool):
            raise ValueError(
                f"candidate ladder candidates[{index}].require_final_anchor must be boolean"
            )
        if row["coord"] not in {"index", "sigma", "timestep"}:
            raise ValueError(f"candidate ladder candidates[{index}].coord is unsupported")
        arms.append(
            CandidateArm(
                warmup_steps=row["warmup_steps"],
                anchor_interval=row["anchor_interval"],
                order=row["order"],
                anchor_phase=row["anchor_phase"],
                cooldown_steps=row["cooldown_steps"],
                require_final_anchor=row["require_final_anchor"],
                coord=row["coord"],
            )
        )
    candidate_ids = [arm.candidate_id for arm in arms]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate ladder produced duplicate candidate identifiers")
    coordinates = {arm.coord for arm in arms}
    if len(coordinates) != 1:
        raise ValueError("candidate ladder must use one shared coordinate")
    return CandidateLadder(
        source_path=path,
        ladder_id=ladder_id,
        labels=tuple(labels),
        arms=tuple(arms),
        content_sha256=document["sha256"],
        file_sha256=_sha256_file(path),
    )


def load_adaptive_candidate(path: Path) -> AdaptiveCandidateArm:
    """Load one strict adaptive candidate without changing ladder semantics."""

    from difflet.pipeline.cache import AdaptiveAnchorConfig

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read adaptive candidate {path}: {error}") from error
    expected = {
        "schema",
        "schema_revision",
        "candidate_id",
        "policy",
        "predictor",
        "sha256",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError("adaptive candidate fields do not match the protocol")
    if (
        document["schema"] != ADAPTIVE_CANDIDATE_SCHEMA
        or document["schema_revision"] != ADAPTIVE_CANDIDATE_SCHEMA_REVISION
    ):
        raise ValueError("adaptive candidate schema is unsupported")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if document["sha256"] != canonical_sha256(payload):
        raise ValueError("adaptive candidate sha256 does not match its contents")
    policy = document["policy"]
    required_policy_fields = {
        "type",
        "initial_anchor_interval",
        "minimum_anchor_interval",
        "maximum_anchor_interval",
        "warmup_steps",
        "cooldown_steps",
        "anchor_phase",
        "tighten_error",
        "recovery_error",
        "acceleration_error",
        "recovery_steps",
        "disable_after_recoveries",
        "stable_anchors_for_acceleration",
        "allow_acceleration",
        "require_final_anchor",
    }
    optional_policy_fields = {"acceleration_start_progress"}
    if (
        not isinstance(policy, dict)
        or not required_policy_fields <= set(policy)
        or set(policy) - required_policy_fields - optional_policy_fields
    ):
        raise ValueError("adaptive candidate policy fields do not match the protocol")
    if policy["type"] != "adaptive_anchor":
        raise ValueError("adaptive candidate policy type is unsupported")
    predictor = document["predictor"]
    if not isinstance(predictor, dict) or set(predictor) != {"type", "order", "coord"}:
        raise ValueError("adaptive candidate predictor fields do not match the protocol")
    if predictor["type"] != "taylorseer":
        raise ValueError("adaptive candidate predictor type is unsupported")
    config = AdaptiveAnchorConfig(**{key: value for key, value in policy.items() if key != "type"})
    return AdaptiveCandidateArm(
        candidate_id=document["candidate_id"],
        config=config,
        order=predictor["order"],
        coord=predictor["coord"],
    )


def select_candidate_arms(
    args: argparse.Namespace,
) -> tuple[Any, ...]:
    """Resolve either one explicit ladder or the legacy Cartesian sweep."""

    candidate_ladder = getattr(args, "candidate_ladder", None)
    adaptive_only = bool(getattr(args, "adaptive_only", False))
    sweep_values = (
        getattr(args, "warmup_steps", None),
        getattr(args, "anchor_intervals", None),
        getattr(args, "orders", None),
        getattr(args, "coord", None),
    )
    if adaptive_only:
        if candidate_ladder is not None or any(value is not None for value in sweep_values):
            raise ValueError(
                "--adaptive-only cannot be combined with a static ladder, "
                "sweep, or coordinate flags"
            )
        arms: tuple[Any, ...] = ()
    elif candidate_ladder is not None:
        if any(value is not None for value in sweep_values):
            raise ValueError("--candidate-ladder cannot be combined with sweep or coordinate flags")
        ladder = load_candidate_ladder(Path(candidate_ladder).expanduser().resolve())
        arms: tuple[Any, ...] = ladder.arms
    else:
        warmup_steps = tuple(sweep_values[0] or DEFAULT_WARMUP_STEPS)
        anchor_intervals = tuple(sweep_values[1] or DEFAULT_ANCHOR_INTERVALS)
        orders = tuple(sweep_values[2] or DEFAULT_ORDERS)
        coord = sweep_values[3] or "index"
        arms = build_candidate_arms(
            warmup_steps=warmup_steps,
            anchor_intervals=anchor_intervals,
            orders=orders,
            anchor_phase=getattr(args, "anchor_phase", 1),
            cooldown_steps=getattr(args, "cooldown_steps", 1),
            require_final_anchor=True,
            coord=coord,
        )

    adaptive_paths = getattr(args, "adaptive_candidate", None) or ()
    if isinstance(adaptive_paths, (str, Path)):
        adaptive_paths = (adaptive_paths,)
    for adaptive_path in adaptive_paths:
        arms = (
            *arms,
            load_adaptive_candidate(Path(adaptive_path).expanduser().resolve()),
        )
    if not arms:
        raise ValueError("candidate selection is empty")
    candidate_ids = [arm.candidate_id for arm in arms]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate selection contains duplicate identifiers")
    if len({arm.coord for arm in arms}) != 1:
        raise ValueError("all candidates must use one shared coordinate")
    return arms


def _load_prompts(path: Path | None, inline: Sequence[str] | None) -> tuple[str, ...]:
    if path is not None and inline:
        raise ValueError("--prompts-json and --prompt are mutually exclusive")
    if path is None:
        prompts = tuple(inline or DEFAULT_PROMPTS)
    else:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read prompts JSON {path}: {error}") from error
        prompts_value = document.get("prompts") if isinstance(document, dict) else document
        if isinstance(document, dict) and set(document) != {"prompts"}:
            raise ValueError("prompts JSON object must contain only the prompts field")
        if not isinstance(prompts_value, list):
            raise ValueError("prompts JSON must be a list or an object with a prompts list")
        prompts = tuple(prompts_value)
    if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must be a non-empty list of non-empty strings")
    return tuple(prompt.strip() for prompt in prompts)


def _select_prompts(args: argparse.Namespace) -> PromptSelection:
    custom_prompts = bool(args.prompts_json or args.prompt)
    if custom_prompts:
        if args.prompt_split != "legacy_parity":
            raise ValueError("--prompt-split cannot be combined with custom prompts")
        path = Path(args.prompts_json) if args.prompts_json else None
        prompts = _load_prompts(path, args.prompt)
        source = f"json:{path.resolve()}" if path is not None else "command-line"
        return inline_prompt_selection(prompts, source)
    suite_path = Path(args.prompt_suite or DEFAULT_PROMPT_SUITE_PATH)
    return load_prompt_suite(suite_path.expanduser().resolve(), args.prompt_split)


def _sample_matrix(
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


def _prepare_output_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {path}; use a new directory "
            "to avoid mixing experiment runs"
        )
    path.mkdir(parents=True, exist_ok=True)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_tensor(path: Path, tensor: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor.detach().cpu(), temporary)
    temporary.replace(path)


def _extract_image(result: Any):
    images = getattr(result, "images", None)
    if images is None and isinstance(result, (tuple, list)) and result:
        images = result[0]
    if isinstance(images, (tuple, list)):
        images = images[0] if images else None
    if images is None or not callable(getattr(images, "save", None)):
        raise RuntimeError("FLUX pipeline did not return a decoded PIL image")
    return images


def _run_sample(
    pipe: Any,
    flux_pipeline: Any,
    *,
    sample: dict[str, Any],
    num_steps: int,
    height: int,
    width: int,
    guidance_scale: float,
    artifact_dir: Path,
    output_root: Path,
    measurement_sink: Any = None,
    configuration_source: str | None = None,
) -> dict[str, Any]:
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

    trajectory_values = getattr(flux_pipeline, "_tc_last_trajectory", None)
    if not isinstance(trajectory_values, list) or len(trajectory_values) != num_steps:
        raise RuntimeError(
            "FLUX pipeline did not expose the complete cache trajectory: "
            f"expected {num_steps} tensors"
        )
    trajectory = torch.stack(
        [value.detach().cpu() for value in trajectory_values],
        dim=0,
    )
    final_latent = trajectory[-1]
    image = _extract_image(result)

    trajectory_path = artifact_dir / f"{sample['sample_id']}.trajectory.pt"
    final_path = artifact_dir / f"{sample['sample_id']}.final-latent.pt"
    image_path = artifact_dir / f"{sample['sample_id']}.png"
    _save_tensor(trajectory_path, trajectory)
    _save_tensor(final_path, final_latent)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    artifacts = {
        "trajectory": _relative(trajectory_path, output_root),
        "final_latent": _relative(final_path, output_root),
        "image": _relative(image_path, output_root),
    }
    if measurement_sink is not None:
        if not configuration_source:
            raise RuntimeError("runtime measurements require a configuration source")
        report = measurement_sink.build_report(
            num_steps=num_steps,
            configuration_source=configuration_source,
        )
        latent_updates = report.latent_updates
        if tuple(record.step_index for record in latent_updates) != tuple(range(num_steps)):
            raise RuntimeError("FLUX runtime measurements do not cover every denoising step")
        actual_anchor_count = sum(not record.used_estimate for record in latent_updates)
        if len(report.anchor_measurements) != actual_anchor_count:
            raise RuntimeError("FLUX anchor measurements disagree with the executed cache actions")
        measurement_path = artifact_dir / f"{sample['sample_id']}.cache-measurements.json"
        report.write_json(measurement_path)
        artifacts["cache_measurements"] = _relative(measurement_path, output_root)
        artifacts["cache_measurements_sha256"] = _sha256_file(measurement_path)
        spatial_builder = getattr(measurement_sink, "build_spatial_report", None)
        if callable(spatial_builder):
            spatial_report = spatial_builder(
                num_steps=num_steps,
                configuration_source=configuration_source,
            )
            expected_spatial_steps = tuple(
                record.step_index
                for record in report.anchor_measurements
                if record.estimate_status == "measured"
            )
            if (
                tuple(record.step_index for record in spatial_report.anchor_errors)
                != expected_spatial_steps
            ):
                raise RuntimeError(
                    "FLUX spatial measurements disagree with measured anchor estimates"
                )
            spatial_path = artifact_dir / f"{sample['sample_id']}.spatial-measurements.json"
            spatial_report.write_json(spatial_path)
            artifacts["spatial_measurements"] = _relative(spatial_path, output_root)
            artifacts["spatial_measurements_sha256"] = _sha256_file(spatial_path)
    return {
        "sample_id": sample["sample_id"],
        "elapsed_s": float(elapsed),
        "artifacts": artifacts,
    }


def _aggregate_runner_stats(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    aggregate: dict[str, int] = {}
    for row in rows:
        stats = row.get("runner_stats", {})
        for key, value in stats.items():
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            aggregate[key] = aggregate.get(key, 0) + int(value)
    return aggregate


def build_manifests(
    *,
    identity: dict[str, Any],
    samples: Sequence[dict[str, Any]],
    arms: Sequence[CandidateArm],
    baseline_runs: Sequence[dict[str, Any]],
    candidate_runs: dict[str, Sequence[dict[str, Any]]],
    started_at: str,
    completed_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build both versioned documents from already-collected run records."""

    num_steps = _strict_positive_int(identity.get("num_steps"), "identity.num_steps")
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    baseline_by_id = {run["sample_id"]: run for run in baseline_runs}
    if len(sample_by_id) != len(samples):
        raise ValueError("sample matrix contains duplicate sample identifiers")
    if len(baseline_by_id) != len(baseline_runs):
        raise ValueError("baseline records contain duplicate sample identifiers")
    if set(sample_by_id) != set(baseline_by_id):
        raise ValueError("baseline records do not match the prompt/seed sample matrix")
    for run in baseline_runs:
        elapsed = float(run["elapsed_s"])
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError("baseline sample durations must be positive and finite")
    baseline_total = sum(float(run["elapsed_s"]) for run in baseline_runs)
    if not math.isfinite(baseline_total) or baseline_total <= 0.0:
        raise ValueError("baseline total duration must be positive and finite")

    candidate_definitions = [
        {
            "candidate_id": arm.candidate_id,
            "policy": arm.policy_spec(),
            "predictor": arm.predictor_spec(),
        }
        for arm in arms
    ]
    comparisons: list[dict[str, Any]] = []
    speedup_rows: list[dict[str, Any]] = []
    for arm in arms:
        runs = list(candidate_runs.get(arm.candidate_id, ()))
        runs_by_id = {run["sample_id"]: run for run in runs}
        if len(runs_by_id) != len(runs):
            raise ValueError(f"candidate {arm.candidate_id!r} has duplicate sample records")
        if set(runs_by_id) != set(sample_by_id):
            raise ValueError(
                f"candidate {arm.candidate_id!r} records do not match the sample matrix"
            )
        for run in runs:
            elapsed = float(run["elapsed_s"])
            if not math.isfinite(elapsed) or elapsed <= 0.0:
                raise ValueError(
                    f"candidate {arm.candidate_id!r} sample durations must "
                    "be positive and finite"
                )
            stats = run.get("runner_stats")
            if not isinstance(stats, dict):
                raise ValueError(f"candidate {arm.candidate_id!r} is missing runner statistics")
            for key in (
                "full_steps",
                "skipped_steps",
                "consecutive_skip_vetoes",
            ):
                value = stats.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"candidate {arm.candidate_id!r} runner stat {key!r} "
                        "must be a nonnegative integer"
                    )
            if stats["full_steps"] + stats["skipped_steps"] != num_steps:
                raise ValueError(
                    f"candidate {arm.candidate_id!r} runner steps do not "
                    f"sum to num_steps={num_steps}"
                )
        total = sum(float(run["elapsed_s"]) for run in runs)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(f"candidate {arm.candidate_id!r} duration must be positive and finite")
        for sample_id, sample in sample_by_id.items():
            baseline = baseline_by_id[sample_id]
            candidate = runs_by_id[sample_id]
            comparisons.append(
                {
                    **sample,
                    "candidate_id": arm.candidate_id,
                    "baseline": dict(baseline["artifacts"]),
                    "candidate": dict(candidate["artifacts"]),
                }
            )
        speedup_rows.append(
            {
                "candidate_id": arm.candidate_id,
                "policy": arm.policy_spec(),
                "predictor": arm.predictor_spec(),
                "total_s": float(total),
                "measured_speedup": float(baseline_total / total),
                "hardware_measured": True,
                "runner_stats": _aggregate_runner_stats(runs),
                "samples": [
                    {
                        "sample_id": run["sample_id"],
                        "elapsed_s": float(run["elapsed_s"]),
                        "runner_stats": dict(run.get("runner_stats", {})),
                    }
                    for run in runs
                ],
            }
        )

    common = {
        **identity,
        "hardware_measured": True,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    quality_input = {
        "schema": QUALITY_INPUT_SCHEMA,
        **common,
        "candidates": candidate_definitions,
        "comparisons": comparisons,
    }
    speedup_candidates = {
        "schema": SPEEDUP_CANDIDATES_SCHEMA,
        **common,
        "baseline": {
            "total_s": float(baseline_total),
            "samples": [
                {
                    "sample_id": run["sample_id"],
                    "elapsed_s": float(run["elapsed_s"]),
                }
                for run in baseline_runs
            ],
        },
        "candidates": speedup_rows,
    }
    return quality_input, speedup_candidates


def _load_pipeline(args: argparse.Namespace):
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


def _build_baseline_adapter(num_steps: int, *, measurement_sink: Any = None):
    from difflet.pipeline.cache import (
        ResolvedCacheSession,
        TaylorSeerPredictor,
        TeaCacheControllerAdapter,
        resolve_cache_config,
    )

    resolved = resolve_cache_config(
        num_steps=num_steps,
        mask=(True,) * num_steps,
        predictor=TaylorSeerPredictor(order=1),
        require_final_anchor=True,
    )
    return TeaCacheControllerAdapter(
        ResolvedCacheSession(resolved, measurement_sink=measurement_sink)
    )


def collect(args: argparse.Namespace) -> tuple[Path, Path]:
    """Run the foreground hardware sweep and write the two manifests."""

    if not args.allow_hardware:
        raise RuntimeError("FLUX A/B collection requires --allow-hardware")
    if args.foreground_ack != FOREGROUND_ACK:
        raise RuntimeError(f"--foreground-ack must equal {FOREGROUND_ACK!r}")
    if args.model_id != MODEL_ID:
        raise ValueError(f"this collector is FLUX-only; expected --model-id {MODEL_ID!r}")

    num_steps = _strict_positive_int(args.num_steps, "num_steps")
    height = _strict_positive_int(args.height, "height")
    width = _strict_positive_int(args.width, "width")
    _strict_positive_int(args.tp_degree, "tp_degree")
    guidance_scale = float(args.guidance_scale)
    if not math.isfinite(guidance_scale):
        raise ValueError("guidance_scale must be finite")
    prompt_selection = _select_prompts(args)
    prompts = prompt_selection.prompts
    seeds = tuple(DEFAULT_SEEDS if args.seed is None else args.seed)
    samples = _sample_matrix(prompts, seeds)
    arms = select_candidate_arms(args)
    for arm in arms:
        arm.build_pipeline_adapter(num_steps)

    output_root = Path(args.out_dir).expanduser().resolve()
    _prepare_output_directory(output_root)
    started_at = _utc_now()
    pipe = _load_pipeline(args)
    flux_pipeline = pipe.app.pipe
    flux_pipeline._tc_record = False
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
        cache_coordinate=arms[0].coord,
        pipeline_warmup_enabled=not bool(args.skip_warmup),
    )
    print(
        "[flux-cache-ab] protocol "
        f"{protocol['sha256']} prompts={prompt_selection.descriptor['split']} "
        f"model={protocol['model']['resolved_revision']}",
        flush=True,
    )
    collect_measurements = bool(getattr(args, "collect_cache_measurements", False))
    collect_spatial_measurements = bool(getattr(args, "collect_spatial_measurements", False))
    if collect_spatial_measurements and not collect_measurements:
        raise ValueError("--collect-spatial-measurements requires --collect-cache-measurements")
    spatial_layout = None
    if collect_spatial_measurements:
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError("FLUX spatial measurements require height and width divisible by 16")
        from difflet.pipeline.cache import SpatialMeasurementLayout

        spatial_layout = SpatialMeasurementLayout(
            token_height=height // 16,
            token_width=width // 16,
            region_rows=_strict_positive_int(
                args.spatial_region_rows,
                "spatial_region_rows",
            ),
            region_columns=_strict_positive_int(
                args.spatial_region_columns,
                "spatial_region_columns",
            ),
            token_axis=-2,
        )
    if collect_measurements:
        print(
            "[flux-cache-ab] runtime measurements enabled; reported wall-clock "
            "times include measurement overhead and are not clean serving timings",
            flush=True,
        )
    if collect_spatial_measurements:
        print(
            "[flux-cache-ab] spatial anchor measurements enabled; these "
            "read-only region summaries do not affect cache decisions",
            flush=True,
        )

    def make_measurement_sink(*, include_spatial: bool):
        if not collect_measurements:
            return None
        if include_spatial and spatial_layout is not None:
            from difflet.pipeline.cache import InMemorySpatialMeasurementSink

            return InMemorySpatialMeasurementSink(spatial_layout)
        from difflet.pipeline.cache import InMemoryMeasurementSink

        return InMemoryMeasurementSink()

    baseline_runs: list[dict[str, Any]] = []
    baseline_sink = make_measurement_sink(include_spatial=False)
    baseline_adapter = _build_baseline_adapter(
        num_steps,
        measurement_sink=baseline_sink,
    )
    flux_pipeline.teacache_controller = baseline_adapter
    for sample in samples:
        run = _run_sample(
            pipe,
            flux_pipeline,
            sample=sample,
            num_steps=num_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            artifact_dir=output_root / "artifacts" / "baseline",
            output_root=output_root,
            measurement_sink=baseline_sink,
            configuration_source=baseline_adapter.source,
        )
        baseline_stats = baseline_adapter.stats()
        if baseline_stats["full_steps"] != num_steps or baseline_stats["skipped_steps"] != 0:
            raise RuntimeError("baseline adapter did not execute every denoise step")
        baseline_runs.append(run)
        print(
            f"[flux-cache-ab] baseline {sample['sample_id']} " f"{run['elapsed_s']:.3f}s",
            flush=True,
        )

    candidate_runs: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        candidate_sink = make_measurement_sink(include_spatial=True)
        adapter = arm.build_pipeline_adapter(
            num_steps,
            measurement_sink=candidate_sink,
        )
        flux_pipeline.teacache_controller = adapter
        rows: list[dict[str, Any]] = []
        for sample in samples:
            run = _run_sample(
                pipe,
                flux_pipeline,
                sample=sample,
                num_steps=num_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                artifact_dir=output_root / "artifacts" / arm.candidate_id,
                output_root=output_root,
                measurement_sink=candidate_sink,
                configuration_source=adapter.source,
            )
            run["runner_stats"] = adapter.stats()
            rows.append(run)
            print(
                f"[flux-cache-ab] {arm.candidate_id} {sample['sample_id']} "
                f"{run['elapsed_s']:.3f}s skip={run['runner_stats']['skipped_steps']}",
                flush=True,
            )
        candidate_runs[arm.candidate_id] = rows

    scheduler_class = type(flux_pipeline.scheduler).__name__
    identity = {
        "model": MODEL_LABEL,
        "model_id": args.model_id,
        "shape_label": f"{height}x{width}",
        "num_steps": num_steps,
        "scheduler_class": scheduler_class,
        "guidance_scale": guidance_scale,
        "prompt_count": len(prompts),
        "seed_count": len(seeds),
        "sample_count": len(samples),
        "protocol": protocol,
    }
    quality, speedup = build_manifests(
        identity=identity,
        samples=samples,
        arms=arms,
        baseline_runs=baseline_runs,
        candidate_runs=candidate_runs,
        started_at=started_at,
        completed_at=_utc_now(),
    )
    quality_path = output_root / "quality-input-v2.json"
    speedup_path = output_root / "speedup-candidates-v1.json"
    _write_json(quality_path, quality)
    _write_json(speedup_path, speedup)
    print(f"[flux-cache-ab] quality manifest: {quality_path}", flush=True)
    print(f"[flux-cache-ab] speedup manifest: {speedup_path}", flush=True)
    return quality_path, speedup_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=None)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt-suite", default=None)
    prompt_group.add_argument("--prompts-json", default=None)
    prompt_group.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--prompt-split", default="legacy_parity")
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--candidate-ladder", default=None)
    parser.add_argument(
        "--adaptive-candidate",
        action="append",
        default=None,
        help=(
            "append one strict adaptive candidate to the static candidate set; "
            "repeat the flag to compare several adaptive settings in one paired run"
        ),
    )
    parser.add_argument(
        "--adaptive-only",
        action="store_true",
        help=(
            "collect the baseline and explicitly supplied adaptive candidates "
            "without adding a static ladder or Cartesian sweep"
        ),
    )
    parser.add_argument("--warmup-steps", type=int, nargs="+", default=None)
    parser.add_argument("--anchor-intervals", type=int, nargs="+", default=None)
    parser.add_argument("--orders", type=int, nargs="+", default=None)
    parser.add_argument("--anchor-phase", type=int, default=1)
    parser.add_argument("--cooldown-steps", type=int, default=1)
    parser.add_argument("--coord", choices=["index", "timestep", "sigma"], default=None)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--compile-cache-dir", default=None)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--collect-cache-measurements",
        action="store_true",
        help=(
            "write per-request cache measurement reports; this adds tensor "
            "measurement overhead to the recorded wall-clock times"
        ),
    )
    parser.add_argument(
        "--collect-spatial-measurements",
        action="store_true",
        help=(
            "write exact per-region anchor-error energies alongside ordinary "
            "cache measurements; this never changes cache decisions"
        ),
    )
    parser.add_argument("--spatial-region-rows", type=int, default=8)
    parser.add_argument("--spatial-region-columns", type=int, default=8)
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--foreground-ack", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        collect(args)
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
