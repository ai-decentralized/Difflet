#!/usr/bin/env python3
"""Offline-only causal interventions for the FLUX cache pilot.

This module deliberately does not implement a runtime cache controller.  A
``ShadowCaptureHook`` pays for full-DiT outputs at naturally skipped steps while
the unmodified cached prediction still advances the trajectory.  A later
``ReplayRepairHook`` verifies that it reached the identical pre-intervention
state and applies exactly one full-step or spatial-region repair from that
trace.  The expensive truth path is a teacher used to discover causal labels;
it is forbidden in serving and cannot support a speed claim.
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
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

TRACE_SCHEMA = "difflet-flux-cache-shadow-trace"
TRACE_SCHEMA_REVISION = 1
RESULT_SCHEMA = "difflet-flux-cache-causal-repair-result"
RESULT_SCHEMA_REVISION = 1
HARDWARE_ACK = "I am running the offline FLUX causal repair pilot"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    """Hash tensor dtype, shape, and exact host bytes."""

    import torch

    if not torch.is_tensor(tensor):
        raise TypeError("tensor_sha256 requires a tensor")
    host = tensor.detach().to("cpu").contiguous()
    header = json.dumps(
        {"dtype": str(host.dtype), "shape": list(host.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = host.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


@dataclass(frozen=True)
class CausalRepairLayout:
    """A flattened image-token grid divided into equal repair regions."""

    token_height: int
    token_width: int
    region_rows: int = 4
    region_columns: int = 4
    token_axis: int = -2

    def __post_init__(self) -> None:
        for name in ("token_height", "token_width", "region_rows", "region_columns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.token_height % self.region_rows:
            raise ValueError("region_rows must divide token_height")
        if self.token_width % self.region_columns:
            raise ValueError("region_columns must divide token_width")
        if isinstance(self.token_axis, bool) or not isinstance(self.token_axis, int):
            raise ValueError("token_axis must be an integer")

    @property
    def token_count(self) -> int:
        return self.token_height * self.token_width

    @property
    def region_count(self) -> int:
        return self.region_rows * self.region_columns

    def to_dict(self) -> dict[str, int]:
        return {
            "token_height": self.token_height,
            "token_width": self.token_width,
            "region_rows": self.region_rows,
            "region_columns": self.region_columns,
            "token_axis": self.token_axis,
        }


def splice_true_region(
    predicted: Any,
    actual: Any,
    *,
    layout: CausalRepairLayout,
    region_index: int,
) -> Any:
    """Return ``predicted`` with exactly one flattened spatial region repaired."""

    import torch

    if not torch.is_tensor(predicted) or not torch.is_tensor(actual):
        raise TypeError("causal repair requires tensor predictions")
    if predicted.shape != actual.shape:
        raise ValueError("predicted and actual outputs must have identical shapes")
    if predicted.dtype != actual.dtype or predicted.device != actual.device:
        raise ValueError("predicted and actual outputs must share dtype and device")
    if not isinstance(layout, CausalRepairLayout):
        raise TypeError("layout must be a CausalRepairLayout")
    if (
        isinstance(region_index, bool)
        or not isinstance(region_index, int)
        or not 0 <= region_index < layout.region_count
    ):
        raise ValueError("region_index is outside the repair layout")

    axis = layout.token_axis if layout.token_axis >= 0 else predicted.ndim + layout.token_axis
    if not 0 <= axis < predicted.ndim:
        raise ValueError("token_axis is outside the output tensor rank")
    if int(predicted.shape[axis]) != layout.token_count:
        raise ValueError("output token axis does not match the repair layout")

    row = region_index // layout.region_columns
    column = region_index % layout.region_columns
    region_height = layout.token_height // layout.region_rows
    region_width = layout.token_width // layout.region_columns
    row_start = row * region_height
    column_start = column * region_width

    repaired = predicted.clone()
    predicted_grid = repaired.movedim(axis, -1).reshape(
        *repaired.movedim(axis, -1).shape[:-1],
        layout.token_height,
        layout.token_width,
    )
    actual_grid = actual.movedim(axis, -1).reshape_as(predicted_grid)
    predicted_grid[
        ...,
        row_start : row_start + region_height,
        column_start : column_start + region_width,
    ] = actual_grid[
        ...,
        row_start : row_start + region_height,
        column_start : column_start + region_width,
    ]
    return repaired


def _validate_actual(predicted: Any, actual: Any) -> None:
    import torch

    if not torch.is_tensor(actual):
        raise TypeError("shadow full-DiT callback must return a tensor")
    if predicted.shape != actual.shape:
        raise ValueError("shadow true output shape differs from the cached prediction")
    if predicted.dtype != actual.dtype or predicted.device != actual.device:
        raise ValueError("shadow true output must share prediction dtype and device")
    if not bool(torch.isfinite(actual).all().detach().cpu().item()):
        raise ValueError("shadow true output contains non-finite values")


class ShadowCaptureHook:
    """Capture one exact teacher output for every naturally skipped step."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self._records)

    def __call__(
        self,
        *,
        step_index: int,
        timestep: Any,
        latents: Any,
        predicted: Any,
        used_cache_prediction: bool,
        compute_actual: Callable[[], Any],
    ) -> Any:
        del timestep
        if not used_cache_prediction:
            return predicted
        if any(record["step_index"] == step_index for record in self._records):
            raise RuntimeError(f"shadow step {step_index} was captured more than once")
        actual = compute_actual()
        _validate_actual(predicted, actual)

        from safetensors.torch import save_file

        tensors = {
            # Clone even when the source is already a contiguous CPU tensor:
            # safetensors rejects distinct keys that alias the same storage.
            "actual": actual.detach().to("cpu").contiguous().clone(),
            "predicted": predicted.detach().to("cpu").contiguous().clone(),
            "pre_step_latents": latents.detach().to("cpu").contiguous().clone(),
        }
        destination = self.output_dir / f"step-{step_index:03d}.safetensors"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        save_file(tensors, str(temporary), metadata={"format": TRACE_SCHEMA})
        temporary.replace(destination)
        self._records.append(
            {
                "step_index": int(step_index),
                "artifact": destination.name,
                "artifact_sha256": _sha256_file(destination),
                "pre_step_latents_sha256": tensor_sha256(tensors["pre_step_latents"]),
                "predicted_sha256": tensor_sha256(tensors["predicted"]),
                "actual_sha256": tensor_sha256(tensors["actual"]),
                "shape": list(tensors["actual"].shape),
                "dtype": str(tensors["actual"].dtype),
            }
        )
        return predicted

    def write_manifest(self, *, identity: Mapping[str, Any]) -> Path:
        if not self._records:
            raise RuntimeError("cannot write an empty shadow trace")
        records = sorted(self._records, key=lambda record: int(record["step_index"]))
        if len({int(record["step_index"]) for record in records}) != len(records):
            raise RuntimeError("shadow trace contains duplicate steps")
        payload = {
            "schema": TRACE_SCHEMA,
            "schema_revision": TRACE_SCHEMA_REVISION,
            "offline_teacher_only": True,
            "serving_speed_claim": False,
            "identity": dict(identity),
            "steps": records,
        }
        document = {**payload, "sha256": _canonical_sha256(payload)}
        destination = self.output_dir / "shadow-trace.json"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination


def load_shadow_trace(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read shadow trace {source}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("shadow trace must be a JSON object")
    expected = {
        "schema",
        "schema_revision",
        "offline_teacher_only",
        "serving_speed_claim",
        "identity",
        "steps",
        "sha256",
    }
    if set(document) != expected:
        raise ValueError("shadow trace fields do not match the protocol")
    if (
        document["schema"] != TRACE_SCHEMA
        or document["schema_revision"] != TRACE_SCHEMA_REVISION
        or document["offline_teacher_only"] is not True
        or document["serving_speed_claim"] is not False
    ):
        raise ValueError("shadow trace identity is invalid")
    digest = document.pop("sha256")
    if not isinstance(digest, str) or digest != _canonical_sha256(document):
        raise ValueError("shadow trace digest does not match its content")
    document["sha256"] = digest
    if not isinstance(document["identity"], dict):
        raise ValueError("shadow trace identity must be an object")
    if not isinstance(document["steps"], list) or not document["steps"]:
        raise ValueError("shadow trace must contain at least one step")
    previous = -1
    for record in document["steps"]:
        if not isinstance(record, dict) or set(record) != {
            "step_index",
            "artifact",
            "artifact_sha256",
            "pre_step_latents_sha256",
            "predicted_sha256",
            "actual_sha256",
            "shape",
            "dtype",
        }:
            raise ValueError("shadow trace step fields do not match the protocol")
        step = record["step_index"]
        if isinstance(step, bool) or not isinstance(step, int) or step <= previous:
            raise ValueError("shadow trace steps must be increasing nonnegative integers")
        previous = step
        artifact = source.parent / record["artifact"]
        if not artifact.is_file() or _sha256_file(artifact) != record["artifact_sha256"]:
            raise ValueError(f"shadow trace artifact is missing or corrupt: {artifact}")
    return document


class ReplayRepairHook:
    """Apply one registered full-step or single-region offline intervention."""

    def __init__(
        self,
        trace_path: str | Path,
        *,
        target_step: int,
        layout: CausalRepairLayout | None = None,
        region_index: int | None = None,
    ) -> None:
        if isinstance(target_step, bool) or not isinstance(target_step, int) or target_step < 0:
            raise ValueError("target_step must be a nonnegative integer")
        if (layout is None) != (region_index is None):
            raise ValueError("layout and region_index must either both be set or both be absent")
        self.trace_path = Path(trace_path).expanduser().resolve()
        self.document = load_shadow_trace(self.trace_path)
        matching = [
            record for record in self.document["steps"] if record["step_index"] == target_step
        ]
        if len(matching) != 1:
            raise ValueError("target_step is not present exactly once in the shadow trace")
        self.target_step = target_step
        self.layout = layout
        self.region_index = region_index
        self.record = matching[0]
        self.applied = False

    def __call__(
        self,
        *,
        step_index: int,
        timestep: Any,
        latents: Any,
        predicted: Any,
        used_cache_prediction: bool,
        compute_actual: Callable[[], Any],
    ) -> Any:
        del timestep, compute_actual
        if step_index != self.target_step:
            return predicted
        if self.applied:
            raise RuntimeError("causal repair was applied more than once")
        if not used_cache_prediction:
            raise RuntimeError("causal repair target is no longer a skipped step")
        if tensor_sha256(latents) != self.record["pre_step_latents_sha256"]:
            raise RuntimeError("causal replay did not reach the registered pre-step latent")
        if tensor_sha256(predicted) != self.record["predicted_sha256"]:
            raise RuntimeError("causal replay prediction differs before intervention")

        from safetensors.torch import load_file

        artifact = self.trace_path.parent / self.record["artifact"]
        actual = load_file(str(artifact), device="cpu")["actual"].to(
            device=predicted.device,
            dtype=predicted.dtype,
        )
        _validate_actual(predicted, actual)
        if tensor_sha256(actual) != self.record["actual_sha256"]:
            raise RuntimeError("shadow actual tensor differs from the trace manifest")
        self.applied = True
        if self.layout is None:
            return actual
        assert self.region_index is not None
        return splice_true_region(
            predicted,
            actual,
            layout=self.layout,
            region_index=self.region_index,
        )

    def validate_complete(self) -> None:
        if not self.applied:
            raise RuntimeError("causal repair target was never applied")


def repair_gain(*, cached_vqa: float, repaired_vqa: float) -> float:
    cached = float(cached_vqa)
    repaired = float(repaired_vqa)
    if not math.isfinite(cached) or not math.isfinite(repaired):
        raise ValueError("VQAScore values must be finite")
    return repaired - cached


def _write_result(path: Path, payload: Mapping[str, Any]) -> None:
    document = {**dict(payload), "sha256": _canonical_sha256(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_image_only(
    pipe: Any,
    flux_pipeline: Any,
    *,
    prompt: str,
    seed: int,
    num_steps: int,
    height: int,
    width: int,
    guidance_scale: float,
    image_path: Path,
) -> dict[str, Any]:
    """Run one deterministic intervention without retaining its full trajectory."""

    import torch

    from scripts.collect_flux_cache_ab import _extract_image

    generator = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    result = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        output_type="pil",
        generator=generator,
    )
    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("pipeline returned an invalid wall-clock duration")
    trajectory = getattr(flux_pipeline, "_tc_last_trajectory", None)
    if not isinstance(trajectory, list) or len(trajectory) != num_steps:
        raise RuntimeError("causal sweep did not expose the complete cache trajectory")
    image = _extract_image(result)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    return {
        "elapsed_s": float(elapsed),
        "image": str(image_path),
        "image_sha256": _sha256_file(image_path),
    }


def _experiment_identity(
    args: argparse.Namespace,
    *,
    prompt: str,
    candidate_id: str,
) -> dict[str, Any]:
    identity = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "candidate_id": candidate_id,
        "prompt_split": args.prompt_split,
        "prompt_index": args.prompt_index,
        "prompt": prompt,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "dtype": args.dtype,
        "tp_degree": args.tp_degree,
    }
    if args.prompts_json is not None:
        identity["prompts_json_sha256"] = _sha256_file(
            Path(args.prompts_json).expanduser().resolve()
        )
    return identity


def _requested_prompts(args: argparse.Namespace) -> tuple[str, ...]:
    if args.prompts_json is not None:
        from scripts.collect_flux_cache_ab import _load_prompts

        return _load_prompts(Path(args.prompts_json).expanduser().resolve(), None)

    from scripts.flux_cache_protocol import load_prompt_suite

    selection = load_prompt_suite(Path(args.prompt_suite), args.prompt_split)
    return tuple(selection.prompts)


def _validate_replay_identity(trace: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    actual = trace.get("identity")
    if actual != dict(expected):
        raise ValueError("shadow trace generation identity differs from the requested replay")


def _run_temporal_sweep(args: argparse.Namespace) -> Path:
    """Generate the full baseline and every single-step full repair in one load."""

    from scripts.collect_flux_cache_ab import (
        _build_baseline_adapter,
        _load_pipeline,
        load_adaptive_candidate,
    )
    if args.trace is None:
        raise ValueError("temporal-sweep mode requires --trace")
    if args.target_step is not None or args.region_index is not None:
        raise ValueError("temporal-sweep mode does not accept repair targets")

    prompts = _requested_prompts(args)
    if not 0 <= args.prompt_index < len(prompts):
        raise ValueError("prompt_index is outside the selected prompt split")
    prompt = prompts[args.prompt_index]
    arm = load_adaptive_candidate(Path(args.candidate).expanduser().resolve())
    identity = _experiment_identity(args, prompt=prompt, candidate_id=arm.candidate_id)
    trace_path = Path(args.trace).expanduser().resolve()
    trace = load_shadow_trace(trace_path)
    _validate_replay_identity(trace, identity)

    sample_id = f"p{args.prompt_index:03d}-s{args.seed}"
    if args.cached_image is None:
        cached_image = trace_path.parent.parent / "artifacts" / f"{sample_id}.png"
    else:
        cached_image = Path(args.cached_image).expanduser().resolve()
    if not cached_image.is_file():
        raise ValueError(f"cached image does not exist: {cached_image}")

    output_root = Path(args.out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    artifact_dir = output_root / "artifacts"
    pipe_args = SimpleNamespace(
        model_id=args.model_id,
        model_revision=args.model_revision,
        tp_degree=args.tp_degree,
        dtype=args.dtype,
        compile_cache_dir=args.compile_cache_dir,
        height=args.height,
        width=args.width,
        force_compile=False,
        skip_warmup=args.skip_warmup,
    )
    pipe = _load_pipeline(pipe_args)
    flux_pipeline = pipe.app.pipe
    flux_pipeline._tc_record = False

    baseline_adapter = _build_baseline_adapter(args.num_steps)
    flux_pipeline.teacache_controller = baseline_adapter
    baseline_run = _run_image_only(
        pipe,
        flux_pipeline,
        prompt=prompt,
        seed=args.seed,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        image_path=artifact_dir / f"{sample_id}-full-dit.png",
    )
    baseline_run["runner_stats"] = baseline_adapter.stats()

    repair_runs = []
    for record in trace["steps"]:
        target_step = int(record["step_index"])
        hook = ReplayRepairHook(trace_path, target_step=target_step)
        adapter = arm.build_pipeline_adapter(args.num_steps)
        flux_pipeline.teacache_controller = adapter
        flux_pipeline._cache_counterfactual_hook = hook
        try:
            run = _run_image_only(
                pipe,
                flux_pipeline,
                prompt=prompt,
                seed=args.seed,
                num_steps=args.num_steps,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                image_path=artifact_dir / f"{sample_id}-step-{target_step:03d}-full.png",
            )
        finally:
            del flux_pipeline._cache_counterfactual_hook
        hook.validate_complete()
        run["target_step"] = target_step
        run["runner_stats"] = adapter.stats()
        repair_runs.append(run)

    comparisons = [
        {
            "sample_id": sample_id,
            "prompt_index": args.prompt_index,
            "seed": args.seed,
            "prompt": prompt,
            "candidate_id": "full-dit-baseline",
            "baseline": {"image": str(cached_image)},
            "candidate": {"image": baseline_run["image"]},
        },
        *[
            {
                "sample_id": sample_id,
                "prompt_index": args.prompt_index,
                "seed": args.seed,
                "prompt": prompt,
                "candidate_id": f"full-repair-step-{run['target_step']:03d}",
                "baseline": {"image": str(cached_image)},
                "candidate": {"image": run["image"]},
            }
            for run in repair_runs
        ],
    ]
    quality_input = {
        "schema": "difflet-flux-cache-causal-repair-quality-input",
        "schema_revision": 1,
        "protocol": {
            "prompt_selection": {
                "split": f"causal_temporal_p{args.prompt_index:03d}_s{args.seed}"
            },
            "offline_teacher_only": True,
            "serving_speed_claim": False,
            "identity": identity,
            "shadow_trace": str(trace_path),
        },
        "comparisons": comparisons,
    }
    quality_path = output_root / "temporal-quality-input.json"
    temporary = quality_path.with_suffix(quality_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(quality_input, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(quality_path)

    payload = {
        "schema": RESULT_SCHEMA,
        "schema_revision": RESULT_SCHEMA_REVISION,
        "offline_teacher_only": True,
        "serving_speed_claim": False,
        "identity": identity,
        "action": {
            "mode": "temporal-sweep",
            "shadow_trace": str(trace_path),
            "cached_image": str(cached_image),
            "quality_input": str(quality_path),
            "repair_count": len(repair_runs),
        },
        "run": {"full_dit_baseline": baseline_run, "repairs": repair_runs},
    }
    result_path = output_root / "causal-repair-result.json"
    _write_result(result_path, payload)
    return result_path


def _run_spatial_sweep(args: argparse.Namespace) -> Path:
    """Generate one full-step repair and all 4x4 single-region repairs."""

    from scripts.collect_flux_cache_ab import _load_pipeline, load_adaptive_candidate
    if args.trace is None:
        raise ValueError("spatial-sweep mode requires --trace")
    if args.target_step is None:
        raise ValueError("spatial-sweep mode requires --target-step")
    if args.region_index is not None:
        raise ValueError("spatial-sweep mode generates every region; omit --region-index")
    if args.height % 16 or args.width % 16:
        raise ValueError("FLUX causal repair requires height and width divisible by 16")

    prompts = _requested_prompts(args)
    if not 0 <= args.prompt_index < len(prompts):
        raise ValueError("prompt_index is outside the selected prompt split")
    prompt = prompts[args.prompt_index]
    arm = load_adaptive_candidate(Path(args.candidate).expanduser().resolve())
    identity = _experiment_identity(args, prompt=prompt, candidate_id=arm.candidate_id)
    trace_path = Path(args.trace).expanduser().resolve()
    trace = load_shadow_trace(trace_path)
    _validate_replay_identity(trace, identity)

    target_step = int(args.target_step)
    if target_step not in {int(record["step_index"]) for record in trace["steps"]}:
        raise ValueError("target_step is not present in the shadow trace")
    layout = CausalRepairLayout(
        token_height=args.height // 16,
        token_width=args.width // 16,
        region_rows=4,
        region_columns=4,
    )
    sample_id = f"p{args.prompt_index:03d}-s{args.seed}"
    if args.cached_image is None:
        cached_image = trace_path.parent.parent / "artifacts" / f"{sample_id}.png"
    else:
        cached_image = Path(args.cached_image).expanduser().resolve()
    if not cached_image.is_file():
        raise ValueError(f"cached image does not exist: {cached_image}")

    output_root = Path(args.out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    artifact_dir = output_root / "artifacts"
    pipe_args = SimpleNamespace(
        model_id=args.model_id,
        model_revision=args.model_revision,
        tp_degree=args.tp_degree,
        dtype=args.dtype,
        compile_cache_dir=args.compile_cache_dir,
        height=args.height,
        width=args.width,
        force_compile=False,
        skip_warmup=args.skip_warmup,
    )
    pipe = _load_pipeline(pipe_args)
    flux_pipeline = pipe.app.pipe
    flux_pipeline._tc_record = False

    intervention_runs = []
    for region_index in (None, *range(layout.region_count)):
        hook = ReplayRepairHook(
            trace_path,
            target_step=target_step,
            layout=None if region_index is None else layout,
            region_index=region_index,
        )
        adapter = arm.build_pipeline_adapter(args.num_steps)
        flux_pipeline.teacache_controller = adapter
        flux_pipeline._cache_counterfactual_hook = hook
        label = "full" if region_index is None else f"region-{region_index:02d}"
        try:
            run = _run_image_only(
                pipe,
                flux_pipeline,
                prompt=prompt,
                seed=args.seed,
                num_steps=args.num_steps,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                image_path=artifact_dir / f"{sample_id}-step-{target_step:03d}-{label}.png",
            )
        finally:
            del flux_pipeline._cache_counterfactual_hook
        hook.validate_complete()
        run["target_step"] = target_step
        run["region_index"] = region_index
        run["runner_stats"] = adapter.stats()
        intervention_runs.append(run)

    comparisons = [
        {
            "sample_id": sample_id,
            "prompt_index": args.prompt_index,
            "seed": args.seed,
            "prompt": prompt,
            "candidate_id": (
                f"full-repair-step-{target_step:03d}"
                if run["region_index"] is None
                else f"region-{run['region_index']:02d}-repair-step-{target_step:03d}"
            ),
            "baseline": {"image": str(cached_image)},
            "candidate": {"image": run["image"]},
        }
        for run in intervention_runs
    ]
    quality_input = {
        "schema": "difflet-flux-cache-causal-repair-quality-input",
        "schema_revision": 1,
        "protocol": {
            "prompt_selection": {
                "split": (
                    f"causal_spatial_p{args.prompt_index:03d}_s{args.seed}"
                    f"_step{target_step:03d}"
                )
            },
            "offline_teacher_only": True,
            "serving_speed_claim": False,
            "identity": identity,
            "shadow_trace": str(trace_path),
            "target_step": target_step,
            "layout": layout.to_dict(),
        },
        "comparisons": comparisons,
    }
    quality_path = output_root / "spatial-quality-input.json"
    temporary = quality_path.with_suffix(quality_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(quality_input, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(quality_path)

    payload = {
        "schema": RESULT_SCHEMA,
        "schema_revision": RESULT_SCHEMA_REVISION,
        "offline_teacher_only": True,
        "serving_speed_claim": False,
        "identity": identity,
        "action": {
            "mode": "spatial-sweep",
            "shadow_trace": str(trace_path),
            "cached_image": str(cached_image),
            "quality_input": str(quality_path),
            "target_step": target_step,
            "layout": layout.to_dict(),
            "intervention_count": len(intervention_runs),
        },
        "run": {"interventions": intervention_runs},
    }
    result_path = output_root / "causal-repair-result.json"
    _write_result(result_path, payload)
    return result_path


def run_hardware(args: argparse.Namespace) -> Path:
    """Run one capture or one registered intervention on Trainium."""

    if not args.allow_hardware or args.foreground_ack != HARDWARE_ACK:
        raise RuntimeError(
            f"hardware execution requires --allow-hardware and --foreground-ack {HARDWARE_ACK!r}"
        )
    if args.mode == "temporal-sweep":
        return _run_temporal_sweep(args)
    if args.mode == "spatial-sweep":
        return _run_spatial_sweep(args)
    if args.mode == "capture" and args.trace is not None:
        raise ValueError("capture mode does not accept --trace")
    if args.mode == "repair" and args.trace is None:
        raise ValueError("repair mode requires --trace")
    if args.mode == "capture" and (args.target_step is not None or args.region_index is not None):
        raise ValueError("capture mode does not accept repair targets")
    if args.mode == "repair" and args.target_step is None:
        raise ValueError("repair mode requires --target-step")

    from scripts.collect_flux_cache_ab import (
        _load_pipeline,
        _run_sample,
        load_adaptive_candidate,
    )
    prompts = _requested_prompts(args)
    if not 0 <= args.prompt_index < len(prompts):
        raise ValueError("prompt_index is outside the selected prompt split")
    prompt = prompts[args.prompt_index]
    arm = load_adaptive_candidate(Path(args.candidate).expanduser().resolve())
    identity = _experiment_identity(args, prompt=prompt, candidate_id=arm.candidate_id)

    output_root = Path(args.out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    if args.mode == "capture":
        hook: Any = ShadowCaptureHook(output_root / "shadow")
    else:
        trace = load_shadow_trace(args.trace)
        _validate_replay_identity(trace, identity)
        layout = None
        if args.region_index is not None:
            if args.height % 16 or args.width % 16:
                raise ValueError("FLUX causal repair requires height and width divisible by 16")
            layout = CausalRepairLayout(
                token_height=args.height // 16,
                token_width=args.width // 16,
                region_rows=4,
                region_columns=4,
            )
        hook = ReplayRepairHook(
            args.trace,
            target_step=args.target_step,
            layout=layout,
            region_index=args.region_index,
        )

    pipe_args = SimpleNamespace(
        model_id=args.model_id,
        model_revision=args.model_revision,
        tp_degree=args.tp_degree,
        dtype=args.dtype,
        compile_cache_dir=args.compile_cache_dir,
        height=args.height,
        width=args.width,
        force_compile=False,
        skip_warmup=args.skip_warmup,
    )
    pipe = _load_pipeline(pipe_args)
    flux_pipeline = pipe.app.pipe
    flux_pipeline._tc_record = False
    adapter = arm.build_pipeline_adapter(args.num_steps)
    flux_pipeline.teacache_controller = adapter
    flux_pipeline._cache_counterfactual_hook = hook
    sample = {
        "sample_id": f"p{args.prompt_index:03d}-s{args.seed}",
        "prompt_index": args.prompt_index,
        "seed": args.seed,
        "prompt": prompt,
    }
    try:
        run = _run_sample(
            pipe,
            flux_pipeline,
            sample=sample,
            num_steps=args.num_steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            artifact_dir=output_root / "artifacts",
            output_root=output_root,
            configuration_source=adapter.source,
        )
    finally:
        del flux_pipeline._cache_counterfactual_hook
    stats = adapter.stats()
    run["runner_stats"] = stats

    if args.mode == "capture":
        if len(hook.records) != stats["skipped_steps"]:
            raise RuntimeError("shadow trace count differs from executed cached steps")
        trace_path = hook.write_manifest(identity=identity)
        action = {
            "mode": "capture",
            "shadow_trace": str(trace_path.relative_to(output_root)),
            "captured_steps": [record["step_index"] for record in hook.records],
        }
    else:
        hook.validate_complete()
        action = {
            "mode": "repair",
            "shadow_trace": str(Path(args.trace).expanduser().resolve()),
            "target_step": args.target_step,
            "region_index": args.region_index,
            "repair_scope": "full_step" if args.region_index is None else "single_region_4x4",
        }

    payload = {
        "schema": RESULT_SCHEMA,
        "schema_revision": RESULT_SCHEMA_REVISION,
        "offline_teacher_only": True,
        "serving_speed_claim": False,
        "identity": identity,
        "action": action,
        "run": run,
    }
    result_path = output_root / "causal-repair-result.json"
    _write_result(result_path, payload)
    return result_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("capture", "repair", "temporal-sweep", "spatial-sweep"),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--candidate",
        default=str(ROOT / "benchmark/flux_cache/adaptive-oil-e1p40-k12-candidate.json"),
    )
    parser.add_argument(
        "--prompt-suite",
        default=str(ROOT / "benchmark/flux_cache/prompt-suite-v1.json"),
    )
    parser.add_argument("--prompts-json")
    parser.add_argument("--prompt-split", default="adaptive_profile_holdout")
    parser.add_argument("--prompt-index", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace")
    parser.add_argument("--cached-image")
    parser.add_argument("--target-step", type=int)
    parser.add_argument("--region-index", type=int)
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument(
        "--model-revision",
        default="3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
    )
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--compile-cache-dir")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--foreground-ack")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_hardware(args)
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    print(f"[flux-cache-causal-repair] result={result}", flush=True)
    return 0


__all__ = [
    "CausalRepairLayout",
    "ReplayRepairHook",
    "ShadowCaptureHook",
    "load_shadow_trace",
    "repair_gain",
    "splice_true_region",
    "tensor_sha256",
]


if __name__ == "__main__":
    raise SystemExit(main())
