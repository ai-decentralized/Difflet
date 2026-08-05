#!/usr/bin/env python3
"""Paired anchor-level braking interventions for the FLUX cache controller.

The hardware runner replays the same prompt and seed to one naturally computed
anchor and then applies exactly one of three actions:

* ``continue`` ignores the target anchor measurement once and preserves the
  pre-anchor interval;
* ``brake`` forces the minimum configured anchor interval; and
* ``recovery`` forces the configured consecutive-real-step recovery window and
  then continues at the minimum interval.

This is an offline causal experiment.  It never runs a shadow Transformer at a
skipped step and may not be used for a serving speed or quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
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
    InMemoryMeasurementSink,
    QualityRecoveryConfig,
    QualityRecoveryGuard,
    TaylorSeerPredictor,
    TeaCacheControllerAdapter,
)

PROTOCOL_SCHEMA = "difflet-flux-cache-brake-intervention-pilot"
PROTOCOL_SCHEMA_REVISION = 1
RUN_SCHEMA = "difflet-flux-cache-brake-intervention-run"
RUN_SCHEMA_REVISION = 1
ANALYSIS_SCHEMA = "difflet-flux-cache-brake-intervention-analysis"
ANALYSIS_SCHEMA_REVISION = 1
QUALITY_SCHEMA = "difflet-flux-cache-brake-intervention-quality-input"
QUALITY_SCHEMA_REVISION = 1
HARDWARE_ACK = "I am running the offline FLUX brake intervention pilot"
ACTIONS = ("continue", "brake", "recovery")
TERMINAL_ACTION = "terminal"


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    """Hash tensor dtype, shape, and exact bytes without retaining the tensor."""

    import torch

    if not torch.is_tensor(tensor):
        raise TypeError("tensor_sha256 requires a tensor")
    host = tensor.detach().to("cpu").contiguous()
    header = json.dumps(
        {"dtype": str(host.dtype), "shape": list(host.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + host.view(torch.uint8).numpy().tobytes()).hexdigest()


def _write_json(path: Path, document: Mapping[str, Any], *, add_digest: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(document)
    if add_digest:
        if "sha256" in payload:
            raise ValueError("hashed JSON payload must not already contain sha256")
        payload["sha256"] = canonical_sha256(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_hashed_json(path: Path, *, schema: str, revision: int) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON artifact {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    digest = document.get("sha256")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if not isinstance(digest, str) or digest != canonical_sha256(payload):
        raise ValueError(f"JSON artifact digest does not match its content: {path}")
    if document.get("schema") != schema or document.get("schema_revision") != revision:
        raise ValueError(f"JSON artifact schema is unsupported: {path}")
    return document


def _rooted(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


class BrakeInterventionPolicy(AdaptiveAnchorPolicy):
    """Instrument the normal adaptive policy and intervene at one real anchor."""

    def __init__(
        self,
        config: Any,
        *,
        action: str = "natural",
        target_step: int | None = None,
    ) -> None:
        if action not in ("natural", *ACTIONS, TERMINAL_ACTION):
            raise ValueError(f"unsupported intervention action: {action!r}")
        if action == "natural" and target_step is not None:
            raise ValueError("natural policy does not accept a target step")
        if action != "natural" and (
            isinstance(target_step, bool) or not isinstance(target_step, int) or target_step < 0
        ):
            raise ValueError("intervention policy requires a nonnegative target step")
        self.action = action
        self.target_step = target_step
        super().__init__(config)

    def reset(self) -> None:
        super().reset()
        self._anchor_events: list[dict[str, Any]] = []
        self._intervention_applied = False
        self._target_event: dict[str, Any] | None = None

    @property
    def anchor_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(event) for event in self._anchor_events)

    @property
    def target_event(self) -> dict[str, Any] | None:
        return None if self._target_event is None else dict(self._target_event)

    def observe_anchor_measurement(self, measurement: Any) -> None:
        step_index = int(measurement.step_index)
        pre_state = self._state
        pre_interval = self._current_interval
        pre_recovery_count = self._recovery_count

        if self.action == "natural" or step_index != self.target_step:
            super().observe_anchor_measurement(measurement)
            applied_action = "natural"
        else:
            if self._intervention_applied:
                raise RuntimeError("anchor intervention was applied more than once")
            if pre_state != "active":
                raise RuntimeError("anchor intervention target was not in ACTIVE state")
            if (
                measurement.estimate_status != "measured"
                or not measurement.numerically_valid
                or measurement.estimate_relative_error is None
            ):
                raise RuntimeError("anchor intervention requires a valid measured error")
            error = float(measurement.estimate_relative_error)
            if not math.isfinite(error) or error < 0.0:
                raise RuntimeError("anchor intervention received an invalid measured error")
            self._last_anchor_error = error
            self._stable_anchor_count = 0
            self._current_interval = (
                pre_interval if self.action == "continue" else self.config.minimum_anchor_interval
            )
            if self.action == TERMINAL_ACTION:
                self._disable()
            elif self.action == "recovery":
                self._state = "recovery"
                self._recovery_steps_remaining = self.config.recovery_steps
                self._next_anchor_step = None
            else:
                self._state = "active"
                self._recovery_steps_remaining = 0
                self._schedule_after_anchor(step_index, int(measurement.num_steps))
            self._intervention_applied = True
            applied_action = self.action

        event = {
            **measurement.to_dict(),
            "pre_state": pre_state,
            "pre_interval": int(pre_interval),
            "pre_recovery_count": int(pre_recovery_count),
            "post_state": self._state,
            "post_interval": int(self._current_interval),
            "post_recovery_steps_remaining": int(self._recovery_steps_remaining),
            "post_next_anchor_step": self._next_anchor_step,
            "applied_action": applied_action,
        }
        self._anchor_events.append(event)
        if applied_action in (*ACTIONS, TERMINAL_ACTION):
            self._target_event = dict(event)

    def validate_complete(self) -> None:
        if self.action != "natural" and not self._intervention_applied:
            raise RuntimeError("registered anchor intervention was never applied")

    def stats(self) -> dict[str, Any]:
        result = super().stats()
        result.update(
            {
                "intervention_action": self.action,
                "intervention_target_step": self.target_step,
                "intervention_applied": self._intervention_applied,
            }
        )
        return result


class TargetAnchorFingerprintHook:
    """Verify that every intervention arm reaches the identical target state."""

    def __init__(self, target_step: int, *, history_supplier: Any = None) -> None:
        if isinstance(target_step, bool) or not isinstance(target_step, int) or target_step < 0:
            raise ValueError("target_step must be a nonnegative integer")
        self.target_step = target_step
        if history_supplier is not None and not callable(history_supplier):
            raise TypeError("history_supplier must be callable")
        self.history_supplier = history_supplier
        self.fingerprint: dict[str, Any] | None = None

    def __call__(
        self,
        *,
        step_index: int,
        timestep: Any,
        latents: Any,
        predicted: Any,
        used_cache_prediction: bool,
        compute_actual: Any,
    ) -> Any:
        del compute_actual
        if step_index != self.target_step:
            return predicted
        if self.fingerprint is not None:
            raise RuntimeError("target anchor fingerprint was captured more than once")
        if used_cache_prediction:
            raise RuntimeError("intervention target is not a real anchor in this replay")
        item = getattr(timestep, "item", None)
        coordinate = float(item() if callable(item) else timestep)
        history = None
        if self.history_supplier is not None:
            supplied = self.history_supplier()
            history = [
                {
                    "step_index": int(anchor.step_index),
                    "output_sha256": tensor_sha256(anchor.output),
                }
                for anchor in supplied
            ]
            if not history or history[-1]["step_index"] != step_index:
                raise RuntimeError("target anchor was not present in supplied cache history")
        self.fingerprint = {
            "step_index": int(step_index),
            "timestep": coordinate,
            "pre_step_latents_sha256": tensor_sha256(latents),
            "actual_output_sha256": tensor_sha256(predicted),
            "cache_history": history,
        }
        return predicted

    def validate_complete(self) -> dict[str, Any]:
        if self.fingerprint is None:
            raise RuntimeError("target anchor fingerprint was never captured")
        return dict(self.fingerprint)


def select_target_anchor(
    events: Sequence[Mapping[str, Any]],
    *,
    target_progress: float,
    num_steps: int,
    minimum_remaining_steps: int,
    minimum_interval: int | None = None,
) -> dict[str, Any]:
    """Choose the closest ACTIVE, measured anchor using only pre-action data."""

    progress = float(target_progress)
    if not 0.0 < progress < 1.0:
        raise ValueError("target_progress must be strictly between zero and one")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 1:
        raise ValueError("num_steps must be an integer greater than one")
    if (
        isinstance(minimum_remaining_steps, bool)
        or not isinstance(minimum_remaining_steps, int)
        or minimum_remaining_steps < 1
    ):
        raise ValueError("minimum_remaining_steps must be a positive integer")
    maximum_step = num_steps - 1 - minimum_remaining_steps
    if minimum_interval is not None and (
        isinstance(minimum_interval, bool)
        or not isinstance(minimum_interval, int)
        or minimum_interval < 1
    ):
        raise ValueError("minimum_interval must be a positive integer or None")
    eligible = [
        dict(event)
        for event in events
        if event.get("pre_state") == "active"
        and event.get("estimate_status") == "measured"
        and event.get("numerically_valid") is True
        and event.get("estimate_relative_error") is not None
        and int(event["step_index"]) <= maximum_step
        and (
            minimum_interval is None
            or int(event.get("pre_interval", minimum_interval)) > minimum_interval
        )
    ]
    if not eligible:
        raise RuntimeError("reference trajectory has no eligible intervention anchor")
    target_coordinate = progress * (num_steps - 1)
    return min(
        eligible,
        key=lambda event: (
            abs(int(event["step_index"]) - target_coordinate),
            int(event["step_index"]),
        ),
    )


def _coefficient_of_variation(values: Sequence[float]) -> float:
    numbers = [float(value) for value in values]
    mean = statistics.fmean(numbers)
    return 0.0 if abs(mean) <= 1e-12 else statistics.pstdev(numbers) / abs(mean)


def _target_signals(
    *,
    event: Mapping[str, Any],
    target_step: int,
    spatial_records: Sequence[Sequence[Any]],
    output_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze serving-cheap target-time features before the intervention."""

    result: dict[str, Any] = {
        "anchor_error": float(event["estimate_relative_error"]),
        "relative_output_change": event.get("relative_output_change"),
        "relative_output_curvature": event.get("relative_output_curvature"),
        "anchor_step_gap": event.get("anchor_step_gap"),
        "pre_interval": int(event["pre_interval"]),
        "stage_progress": target_step / max(int(event["num_steps"]) - 1, 1),
    }
    spatial_by_step = {
        int(record[0]): [float(value) for value in record[1]] for record in spatial_records
    }
    current = spatial_by_step.get(target_step)
    if current is not None:
        ordered = sorted(current, reverse=True)
        result.update(
            {
                "spatial_level": current,
                "spatial_max_region_level": ordered[0],
                "spatial_top2_mean_level": statistics.fmean(ordered[:2]),
                "spatial_level_cv": _coefficient_of_variation(current),
            }
        )
        previous = spatial_by_step.get(target_step - 1)
        older = spatial_by_step.get(target_step - 2)
        if previous is not None and older is not None:
            acceleration = [
                max(now - 2.0 * prior + old, 0.0)
                for now, prior, old in zip(current, previous, older)
            ]
            ordered_acceleration = sorted(acceleration, reverse=True)
            result.update(
                {
                    "spatial_positive_acceleration": acceleration,
                    "spatial_max_region_acceleration": ordered_acceleration[0],
                    "spatial_top2_mean_acceleration": statistics.fmean(ordered_acceleration[:2]),
                    "spatial_acceleration_cv": _coefficient_of_variation(acceleration),
                }
            )
    output = next(
        (record for record in output_records if int(record["step_index"]) == target_step),
        None,
    )
    if output is not None:
        packed = {
            name: [float(value) for value in output[name]]
            for name in ("relative_l1", "velocity_turn", "acceleration_ratio")
        }
        result["output_dynamics"] = packed
        for name, values in packed.items():
            ordered = sorted(values, reverse=True)
            result[f"output_max_region_{name}"] = ordered[0]
            result[f"output_top2_mean_{name}"] = statistics.fmean(ordered[:2])
    return result


def _load_protocol(path: Path) -> dict[str, Any]:
    document = _load_hashed_json(
        path,
        schema=PROTOCOL_SCHEMA,
        revision=PROTOCOL_SCHEMA_REVISION,
    )
    if document.get("serving_claim") is not False:
        raise ValueError("brake intervention protocol must disable serving claims")
    if document.get("offline_intervention_only") is not True:
        raise ValueError("brake intervention protocol must be offline-only")
    if tuple(document.get("actions", ())) != ACTIONS:
        raise ValueError(f"brake intervention actions must be {ACTIONS!r}")
    candidate = document.get("candidate")
    prompts = document.get("prompt_source")
    if not isinstance(candidate, dict) or not isinstance(prompts, dict):
        raise ValueError("protocol candidate and prompt_source must be objects")
    candidate_path = _rooted(candidate["path"])
    prompt_path = _rooted(prompts["path"])
    if sha256_file(candidate_path) != candidate.get("file_sha256"):
        raise ValueError("protocol candidate file hash does not match")
    if sha256_file(prompt_path) != prompts.get("file_sha256"):
        raise ValueError("protocol prompt file hash does not match")
    phases = document.get("target_phases")
    if not isinstance(phases, dict) or not phases:
        raise ValueError("protocol target_phases must be a non-empty object")
    for name, value in phases.items():
        if not isinstance(name, str) or not name or not 0.0 < float(value) < 1.0:
            raise ValueError("protocol target phase is invalid")
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("protocol samples must be a non-empty list")
    sample_ids: set[str] = set()
    target_count = 0
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("protocol sample must be an object")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError("protocol sample_id must be unique and non-empty")
        sample_ids.add(sample_id)
        requested_phases = sample.get("phases")
        if not isinstance(requested_phases, list) or not requested_phases:
            raise ValueError("protocol sample phases must be a non-empty list")
        if len(set(requested_phases)) != len(requested_phases) or any(
            phase not in phases for phase in requested_phases
        ):
            raise ValueError("protocol sample phases are duplicated or unknown")
        target_count += len(requested_phases)
    expected = len(samples) * 2 + target_count * len(ACTIONS)
    if int(document.get("expected_unique_images", -1)) != expected:
        raise ValueError(
            f"protocol expected_unique_images must be {expected}, got "
            f"{document.get('expected_unique_images')!r}"
        )
    return document


def _load_prompts(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    path = _rooted(protocol["prompt_source"]["path"])
    document = json.loads(path.read_text(encoding="utf-8"))
    values = document.get("prompts") if isinstance(document, dict) else None
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise ValueError("prompt source must contain a non-empty prompts list")
    return tuple(value.strip() for value in values)


def _build_adapter(
    arm: Any,
    *,
    num_steps: int,
    policy: BrakeInterventionPolicy,
    measurement_sink: InMemoryMeasurementSink,
) -> TeaCacheControllerAdapter:
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
        measurement_sink=measurement_sink,
    )
    session = CacheSession(
        runner,
        num_steps=num_steps,
        configuration_source="offline-brake-intervention",
    )
    return TeaCacheControllerAdapter(session)


def _relative_image_run(run: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    result = dict(run)
    result["image"] = str(Path(result["image"]).resolve().relative_to(output_root))
    return result


def _write_measurements(
    sink: InMemoryMeasurementSink,
    *,
    num_steps: int,
    destination: Path,
    output_root: Path,
) -> str:
    report = sink.build_report(
        num_steps=num_steps,
        configuration_source="offline-brake-intervention",
    )
    report.write_json(destination)
    return str(destination.resolve().relative_to(output_root))


def _candidate_id(phase: str, action: str) -> str:
    return f"anchor-{phase}-{action}"


def _branch_order(protocol_sha256: str, sample_id: str, phase: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            ACTIONS,
            key=lambda action: hashlib.sha256(
                f"{protocol_sha256}:{sample_id}:{phase}:{action}".encode("utf-8")
            ).hexdigest(),
        )
    )


def _git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "worktree_dirty": dirty}


def run_hardware(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.allow_hardware or args.foreground_ack != HARDWARE_ACK:
        raise RuntimeError(
            f"hardware execution requires --allow-hardware and --foreground-ack {HARDWARE_ACK!r}"
        )
    protocol_path = Path(args.protocol).expanduser().resolve()
    protocol = _load_protocol(protocol_path)
    prompts = _load_prompts(protocol)
    generation = protocol["generation_identity"]
    num_steps = int(generation["num_steps"])
    output_root = Path(args.out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    from scripts.collect_flux_cache_ab import (  # imported only for hardware execution
        _build_baseline_adapter,
        _load_pipeline,
        load_adaptive_candidate,
    )
    from scripts.flux_cache_causal_repair import _run_image_only

    arm = load_adaptive_candidate(_rooted(protocol["candidate"]["path"]))
    if arm.candidate_id != protocol["candidate"]["candidate_id"]:
        raise ValueError("protocol candidate_id does not match the candidate file")
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
        collect_online_signals=True,
    )
    pipe = _load_pipeline(pipe_args)
    flux_pipeline = pipe.app.pipe
    started = time.time()
    sample_results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for sample in protocol["samples"]:
        sample_id = sample["sample_id"]
        prompt_index = int(sample["prompt_index"])
        seed = int(sample["seed"])
        if not 0 <= prompt_index < len(prompts):
            raise ValueError(f"prompt index is outside source for {sample_id}")
        prompt = prompts[prompt_index]
        sample_root = output_root / "artifacts" / sample_id

        reference_sink = InMemoryMeasurementSink()
        reference_policy = BrakeInterventionPolicy(arm.config)
        reference_adapter = _build_adapter(
            arm,
            num_steps=num_steps,
            policy=reference_policy,
            measurement_sink=reference_sink,
        )
        flux_pipeline.teacache_controller = reference_adapter
        flux_pipeline._tc_record = True
        flux_pipeline._tc_output_dynamics_record = True
        reference_run = _run_image_only(
            pipe,
            flux_pipeline,
            prompt=prompt,
            seed=seed,
            num_steps=num_steps,
            height=int(generation["height"]),
            width=int(generation["width"]),
            guidance_scale=float(generation["guidance_scale"]),
            image_path=sample_root / "natural.png",
        )
        reference_run = _relative_image_run(reference_run, output_root)
        reference_run["runner_stats"] = reference_adapter.stats()
        reference_run["measurements"] = _write_measurements(
            reference_sink,
            num_steps=num_steps,
            destination=sample_root / "natural.measurements.json",
            output_root=output_root,
        )
        spatial_records = list(flux_pipeline._tc_spatial_deltas)
        output_records = list(flux_pipeline._tc_output_dynamics)
        selected_targets: dict[str, dict[str, Any]] = {}
        for phase in sample["phases"]:
            event = select_target_anchor(
                reference_policy.anchor_events,
                target_progress=float(protocol["target_phases"][phase]),
                num_steps=num_steps,
                minimum_remaining_steps=int(protocol["minimum_remaining_steps"]),
                minimum_interval=int(arm.config.minimum_anchor_interval),
            )
            target_step = int(event["step_index"])
            selected_targets[phase] = {
                "phase": phase,
                "target_progress": float(protocol["target_phases"][phase]),
                "target_step": target_step,
                "reference_event": event,
                "signals": _target_signals(
                    event=event,
                    target_step=target_step,
                    spatial_records=spatial_records,
                    output_records=output_records,
                ),
            }

        flux_pipeline._tc_record = False
        flux_pipeline._tc_output_dynamics_record = False
        baseline_adapter = _build_baseline_adapter(num_steps)
        flux_pipeline.teacache_controller = baseline_adapter
        baseline_run = _run_image_only(
            pipe,
            flux_pipeline,
            prompt=prompt,
            seed=seed,
            num_steps=num_steps,
            height=int(generation["height"]),
            width=int(generation["width"]),
            guidance_scale=float(generation["guidance_scale"]),
            image_path=sample_root / "full-dit.png",
        )
        baseline_run = _relative_image_run(baseline_run, output_root)
        baseline_run["runner_stats"] = baseline_adapter.stats()

        target_results: list[dict[str, Any]] = []
        for phase in sample["phases"]:
            selected = selected_targets[phase]
            target_step = int(selected["target_step"])
            branch_runs: dict[str, Any] = {}
            expected_fingerprint: dict[str, Any] | None = None
            action_order = _branch_order(protocol["sha256"], sample_id, phase)
            for action in action_order:
                sink = InMemoryMeasurementSink()
                policy = BrakeInterventionPolicy(
                    arm.config,
                    action=action,
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
                try:
                    run = _run_image_only(
                        pipe,
                        flux_pipeline,
                        prompt=prompt,
                        seed=seed,
                        num_steps=num_steps,
                        height=int(generation["height"]),
                        width=int(generation["width"]),
                        guidance_scale=float(generation["guidance_scale"]),
                        image_path=sample_root / f"{phase}-{action}.png",
                    )
                finally:
                    del flux_pipeline._cache_counterfactual_hook
                policy.validate_complete()
                fingerprint = hook.validate_complete()
                if expected_fingerprint is None:
                    expected_fingerprint = fingerprint
                elif fingerprint != expected_fingerprint:
                    raise RuntimeError(
                        f"paired branches diverged before {sample_id}/{phase} target anchor"
                    )
                event = policy.target_event
                if event is None:
                    raise RuntimeError("intervention policy did not expose its target event")
                reference_event = selected["reference_event"]
                if float(event["estimate_relative_error"]) != float(
                    reference_event["estimate_relative_error"]
                ) or int(event["pre_interval"]) != int(reference_event["pre_interval"]):
                    raise RuntimeError("paired branch target measurement differs from reference")
                run = _relative_image_run(run, output_root)
                run.update(
                    {
                        "candidate_id": _candidate_id(phase, action),
                        "runner_stats": adapter.stats(),
                        "measurements": _write_measurements(
                            sink,
                            num_steps=num_steps,
                            destination=sample_root / f"{phase}-{action}.measurements.json",
                            output_root=output_root,
                        ),
                        "target_event": event,
                        "target_fingerprint": fingerprint,
                    }
                )
                branch_runs[action] = run
            target_results.append(
                {
                    **selected,
                    "action_order": list(action_order),
                    "target_fingerprint": expected_fingerprint,
                    "branches": branch_runs,
                }
            )

        sample_result = {
            "sample_id": sample_id,
            "prompt_index": prompt_index,
            "seed": seed,
            "prompt": prompt,
            "prior_label": sample.get("prior_label"),
            "reference": reference_run,
            "baseline": baseline_run,
            "targets": target_results,
        }
        sample_results.append(sample_result)
        base_comparison = {
            "sample_id": sample_id,
            "prompt_index": prompt_index,
            "seed": seed,
            "prompt": prompt,
            "baseline": {"image": baseline_run["image"]},
        }
        comparisons.append(
            {
                **base_comparison,
                "candidate_id": "natural-profile",
                "candidate": {"image": reference_run["image"]},
            }
        )
        for target in target_results:
            for action in ACTIONS:
                run = target["branches"][action]
                comparisons.append(
                    {
                        **base_comparison,
                        "candidate_id": run["candidate_id"],
                        "candidate": {"image": run["image"]},
                    }
                )
        print(
            f"[brake-intervention] {sample_id} targets="
            f"{[(target['phase'], target['target_step']) for target in target_results]}",
            flush=True,
        )

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
        "execution": {
            **_git_identity(),
            "started_unix_s": started,
            "completed_unix_s": time.time(),
            "pipeline_warmup_skipped": bool(args.skip_warmup),
        },
        "offline_intervention_only": True,
        "serving_speed_claim": False,
        "quality_input": {
            "path": str(quality_path.relative_to(output_root)),
            "sha256": sha256_file(quality_path),
            "expected_unique_images": protocol["expected_unique_images"],
        },
        "samples": sample_results,
    }
    result_path = output_root / "brake-intervention-run.json"
    _write_json(result_path, run_document, add_digest=True)
    return result_path, quality_path


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: float(values[index]))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[position]]):
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_energy = sum((x - left_mean) ** 2 for x in left)
    right_energy = sum((y - right_mean) ** 2 for y in right)
    if left_energy <= 0.0 or right_energy <= 0.0:
        return None
    return numerator / math.sqrt(left_energy * right_energy)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _quality_failed(
    *,
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    margins: Mapping[str, float],
) -> tuple[bool, dict[str, float]]:
    harms = {
        metric: float(baseline[metric]) - float(candidate[metric])
        for metric in ("image_reward", "vqa_score")
    }
    failed = any(harms[metric] > float(margins[metric]) for metric in harms)
    return failed, harms


def _scalar_signals(signals: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for name, value in signals.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            result[name] = number
    return result


def evaluate_pilot_decision(
    *,
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    correlations: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the registered pilot rule without consulting endpoint images."""

    known_failure_rows = [
        row for row in rows if str(row.get("prior_label", "")).endswith("-failure")
    ]
    pass_control_rows = [
        row for row in rows if str(row.get("prior_label", "")).endswith("-pass-control")
    ]
    known_failure_targets = {
        (str(row["sample_id"]), str(row["phase"])) for row in known_failure_rows
    }
    pass_control_targets = {(str(row["sample_id"]), str(row["phase"])) for row in pass_control_rows}
    rescued_targets = {
        (str(row["sample_id"]), str(row["phase"]))
        for row in known_failure_rows
        if row["quality"]["outcome"] == "rescue"
    }
    introduced_control_targets = {
        (str(row["sample_id"]), str(row["phase"]))
        for row in pass_control_rows
        if row["quality"]["outcome"] == "introduced_failure"
    }

    margins = protocol["quality_contract"]
    margin_by_metric = {
        "image_reward": float(margins["image_reward_max_harm"]),
        "vqa_score": float(margins["vqa_score_max_harm"]),
    }
    failure_metrics = sorted(
        {
            metric
            for row in known_failure_rows
            for metric, margin in margin_by_metric.items()
            if float(row["quality"]["continue_harm"][metric]) > margin
        }
    )
    positive_associations = []
    rescued_actions = {
        str(row["action"]) for row in known_failure_rows if row["quality"]["outcome"] == "rescue"
    }
    for action in sorted(rescued_actions):
        for signal, metric_values in sorted(correlations.get(action, {}).items()):
            for metric in failure_metrics:
                coefficient = metric_values.get(metric)
                if coefficient is not None and float(coefficient) > 0.0:
                    positive_associations.append(
                        {
                            "action": action,
                            "metric": metric,
                            "signal": signal,
                            "spearman": float(coefficient),
                        }
                    )

    if not rescued_targets:
        status = "negative_pilot"
        reason = "No known failing target was rescued by either registered action."
        prescribed_next_step = protocol["decision_rule"]["next_if_negative"]
    elif introduced_control_targets or not positive_associations:
        status = "inconclusive_pilot"
        reason = (
            "At least one failure was rescued, but the complete registered positive "
            "rule was not met."
        )
        prescribed_next_step = (
            "Do not change the serving controller; register a follow-up causal pilot."
        )
    else:
        status = "positive_pilot"
        reason = "The complete registered positive pilot rule was met."
        prescribed_next_step = protocol["decision_rule"]["next_if_positive"]

    return {
        "status": status,
        "reason": reason,
        "registered_rule": protocol["decision_rule"][
            "negative_pilot" if status == "negative_pilot" else "positive_pilot"
        ],
        "prescribed_next_step": prescribed_next_step,
        "known_failure_target_count": len(known_failure_targets),
        "known_failure_rescued_target_count": len(rescued_targets),
        "pass_control_target_count": len(pass_control_targets),
        "introduced_control_failure_target_count": len(introduced_control_targets),
        "failure_metrics": failure_metrics,
        "positive_signal_associations_for_rescued_actions": positive_associations,
    }


def analyze(args: argparse.Namespace) -> Path:
    protocol = _load_protocol(Path(args.protocol).expanduser().resolve())
    run = _load_hashed_json(
        Path(args.run_result).expanduser().resolve(),
        schema=RUN_SCHEMA,
        revision=RUN_SCHEMA_REVISION,
    )
    semantics_path = Path(args.semantic_scores).expanduser().resolve()
    semantics = json.loads(semantics_path.read_text(encoding="utf-8"))
    if semantics.get("complete") is not True:
        raise ValueError("semantic score report is incomplete")
    semantic_rows = {
        (row["sample_id"], row["candidate_id"]): row for row in semantics.get("comparisons", ())
    }
    margins = {
        "image_reward": float(protocol["quality_contract"]["image_reward_max_harm"]),
        "vqa_score": float(protocol["quality_contract"]["vqa_score_max_harm"]),
    }
    rows: list[dict[str, Any]] = []
    for sample in run["samples"]:
        sample_id = sample["sample_id"]
        for target in sample["targets"]:
            phase = target["phase"]
            continue_row = semantic_rows[(sample_id, _candidate_id(phase, "continue"))]
            baseline_scores = continue_row["baseline_scores"]
            continue_scores = continue_row["candidate_scores"]
            continue_failed, continue_harms = _quality_failed(
                baseline=baseline_scores,
                candidate=continue_scores,
                margins=margins,
            )
            for action in ("brake", "recovery"):
                semantic = semantic_rows[(sample_id, _candidate_id(phase, action))]
                action_scores = semantic["candidate_scores"]
                action_failed, action_harms = _quality_failed(
                    baseline=baseline_scores,
                    candidate=action_scores,
                    margins=margins,
                )
                gains = {
                    metric: float(action_scores[metric]) - float(continue_scores[metric])
                    for metric in ("image_reward", "vqa_score")
                }
                if continue_failed and not action_failed:
                    outcome = "rescue"
                elif continue_failed and action_failed:
                    outcome = "both_fail"
                elif not continue_failed and action_failed:
                    outcome = "introduced_failure"
                else:
                    outcome = "both_pass"
                continue_stats = target["branches"]["continue"]["runner_stats"]
                action_stats = target["branches"][action]["runner_stats"]
                rows.append(
                    {
                        "sample_id": sample_id,
                        "prompt_index": sample["prompt_index"],
                        "seed": sample["seed"],
                        "prior_label": sample.get("prior_label"),
                        "phase": phase,
                        "target_step": target["target_step"],
                        "action": action,
                        "signals": target["signals"],
                        "quality": {
                            "continue_failed": continue_failed,
                            "action_failed": action_failed,
                            "outcome": outcome,
                            "continue_harm": continue_harms,
                            "action_harm": action_harms,
                            "action_minus_continue_gain": gains,
                        },
                        "cost": {
                            "extra_full_steps": int(action_stats["full_steps"])
                            - int(continue_stats["full_steps"]),
                            "removed_skipped_steps": int(continue_stats["skipped_steps"])
                            - int(action_stats["skipped_steps"]),
                            "elapsed_s_delta_diagnostic_only": float(
                                target["branches"][action]["elapsed_s"]
                            )
                            - float(target["branches"]["continue"]["elapsed_s"]),
                        },
                    }
                )

    correlations: dict[str, Any] = {}
    for action in ("brake", "recovery"):
        action_rows = [row for row in rows if row["action"] == action]
        signal_names = sorted(
            set.intersection(*(set(_scalar_signals(row["signals"])) for row in action_rows))
        )
        correlations[action] = {}
        for signal_name in signal_names:
            signal_values = [_scalar_signals(row["signals"])[signal_name] for row in action_rows]
            correlations[action][signal_name] = {
                metric: _spearman(
                    signal_values,
                    [
                        float(row["quality"]["action_minus_continue_gain"][metric])
                        for row in action_rows
                    ],
                )
                for metric in ("image_reward", "vqa_score")
            }

    summary: dict[str, Any] = {}
    for action in ("brake", "recovery"):
        action_rows = [row for row in rows if row["action"] == action]
        summary[action] = {
            "target_count": len(action_rows),
            "outcomes": {
                name: sum(row["quality"]["outcome"] == name for row in action_rows)
                for name in ("rescue", "both_fail", "introduced_failure", "both_pass")
            },
            "mean_gain": {
                metric: statistics.fmean(
                    row["quality"]["action_minus_continue_gain"][metric] for row in action_rows
                )
                for metric in ("image_reward", "vqa_score")
            },
            "mean_extra_full_steps": statistics.fmean(
                row["cost"]["extra_full_steps"] for row in action_rows
            ),
        }
    document = {
        "schema": ANALYSIS_SCHEMA,
        "schema_revision": ANALYSIS_SCHEMA_REVISION,
        "study_id": protocol["study_id"],
        "evidence_role": "exploratory-causal-action-mapping-only",
        "serving_claim": False,
        "inputs": {
            "protocol_sha256": protocol["sha256"],
            "run_result_sha256": run["sha256"],
            "semantic_scores_path": str(semantics_path),
            "semantic_scores_file_sha256": sha256_file(semantics_path),
        },
        "quality_contract": margins,
        "summary": summary,
        "decision": evaluate_pilot_decision(
            protocol=protocol,
            rows=rows,
            correlations=correlations,
        ),
        "signal_to_treatment_gain_spearman": correlations,
        "rows": rows,
        "interpretation_limits": [
            "Development prompts were selected from opened failure-enriched data.",
            "Anchor error and intervention phase are confounded in this small pilot; their effects cannot be separated.",
            "Elapsed-time deltas are diagnostic only; full-step deltas are the primary action cost.",
            "A positive result selects signals for a new preregistered intervention holdout; it cannot qualify serving.",
        ],
    }
    destination = Path(args.out).expanduser().resolve()
    _write_json(destination, document, add_digest=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run paired interventions on Trainium")
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("--compile-cache-dir")
    run_parser.add_argument("--skip-warmup", action="store_true")
    run_parser.add_argument("--allow-hardware", action="store_true")
    run_parser.add_argument("--foreground-ack")
    analyze_parser = subparsers.add_parser("analyze", help="join semantic scores to actions")
    analyze_parser.add_argument("--protocol", required=True)
    analyze_parser.add_argument("--run-result", required=True)
    analyze_parser.add_argument("--semantic-scores", required=True)
    analyze_parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            result, quality = run_hardware(args)
            print(f"[brake-intervention] run={result}", flush=True)
            print(f"[brake-intervention] quality_input={quality}", flush=True)
        else:
            result = analyze(args)
            print(f"[brake-intervention] analysis={result}", flush=True)
    except (FileExistsError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    return 0


__all__ = [
    "ACTIONS",
    "BrakeInterventionPolicy",
    "TargetAnchorFingerprintHook",
    "analyze",
    "canonical_sha256",
    "select_target_anchor",
]


if __name__ == "__main__":
    raise SystemExit(main())
