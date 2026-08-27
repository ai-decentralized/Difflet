"""Configuration objects for `difflet serve`."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.registry import ModelEntry
from difflet.serving.errors import invalid_extra_body
from difflet.serving.types import OutputModality, ServingPlacement, ServingProfile

MIN_WORKER_HEARTBEAT_INTERVAL_SECONDS = 5.0
MAX_WORKER_HEARTBEAT_INTERVAL_SECONDS = 120.0


def validate_worker_heartbeat_interval(value: float) -> float:
    interval = float(value)
    if (
        not math.isfinite(interval)
        or interval < MIN_WORKER_HEARTBEAT_INTERVAL_SECONDS
        or interval > MAX_WORKER_HEARTBEAT_INTERVAL_SECONDS
    ):
        raise ValueError(
            "worker heartbeat interval must be finite and between "
            f"{MIN_WORKER_HEARTBEAT_INTERVAL_SECONDS:g} and "
            f"{MAX_WORKER_HEARTBEAT_INTERVAL_SECONDS:g} seconds inclusive"
        )
    return interval


class DownloadPolicy(str, Enum):
    AUTO = "auto"
    NEVER = "never"


class CompilePolicy(str, Enum):
    AUTO = "auto"
    NEVER = "never"
    FORCE = "force"


@dataclass(frozen=True)
class ServeOptions:
    model_id: str
    revision: str | None = None
    host: str = "0.0.0.0"
    port: int = 8091
    api_key: str | None = field(default=None, repr=False)
    tp_degree: int | None = None
    cp_degree: int | None = None
    cp_mode: str | None = None
    cfg_parallel: bool | None = None
    sp_enabled: bool | None = None
    height: int | None = None
    width: int | None = None
    num_frames: int | None = None
    # CSV shape set ("320x512x61,320x512x33"): serve K shapes from one
    # bucketed artifact / one resident worker (HunyuanVideo video serving).
    shapes: str | None = None
    cache_dir: str | None = None
    host_vae: bool = False
    clip_placement: ServingPlacement | None = None
    teacache_cadence: int | None = None
    teacache_online_delta: float | None = None
    teacache_speedup: float | None = None
    teacache_calibration: str | None = None
    download_policy: DownloadPolicy = DownloadPolicy.AUTO
    compile_policy: CompilePolicy = CompilePolicy.AUTO
    max_running_requests: int = 1
    max_queued_requests: int = 8
    # None preserves modality-specific defaults: 30 seconds for image serving
    # and 24 hours for the shared video FIFO.
    queue_timeout: float | None = None
    request_timeout: float = 300.0
    artifact_store_timeout: float = 60.0
    worker_cancel_timeout: float = 10.0
    worker_restart_timeout: float = 900.0
    worker_heartbeat_interval: float = 30.0
    artifact_ttl_seconds: int = 3600
    validation_workers: int = 4
    validation_max_waiting: int = 32
    validation_timeout: float = 30.0
    video_retention_seconds: int = 25 * 60 * 60
    video_max_jobs: int = 4096
    video_sweep_interval_seconds: float = 5 * 60.0

    def __post_init__(self) -> None:
        validate_worker_heartbeat_interval(self.worker_heartbeat_interval)
        if self.api_key is not None and (
            not isinstance(self.api_key, str)
            or not self.api_key
            or any(character.isspace() for character in self.api_key)
        ):
            raise ValueError("API key must be non-empty and cannot contain whitespace")
        if self.queue_timeout is not None and (
            not math.isfinite(self.queue_timeout) or self.queue_timeout <= 0
        ):
            raise ValueError("queue timeout must be positive")
        if self.clip_placement not in {None, "host", "neuron"}:
            raise ValueError("clip placement must be 'host' or 'neuron'")
        if (
            isinstance(self.validation_workers, bool)
            or not isinstance(self.validation_workers, int)
            or self.validation_workers <= 0
            or isinstance(self.validation_max_waiting, bool)
            or not isinstance(self.validation_max_waiting, int)
            or self.validation_max_waiting < 0
        ):
            raise ValueError("validation worker/waiting limits are invalid")
        if not math.isfinite(self.validation_timeout) or self.validation_timeout <= 0:
            raise ValueError("validation timeout must be positive")
        if (
            isinstance(self.video_retention_seconds, bool)
            or not isinstance(self.video_retention_seconds, int)
            or self.video_retention_seconds <= 0
            or isinstance(self.video_max_jobs, bool)
            or not isinstance(self.video_max_jobs, int)
            or self.video_max_jobs <= 0
        ):
            raise ValueError("video retention and job limits must be positive")
        if (
            not math.isfinite(self.video_sweep_interval_seconds)
            or self.video_sweep_interval_seconds <= 0
        ):
            raise ValueError("video sweep interval must be positive")

    def effective_queue_timeout(self, output_modality: OutputModality) -> float:
        if self.queue_timeout is not None:
            return float(self.queue_timeout)
        return 24.0 * 60.0 * 60.0 if output_modality == "video" else 30.0


# (model_type, output_modality) pairs whose serving adapters route a compiled
# shape SET on one resident worker. HunyuanVideo 1.5 and LTX-2 are excluded:
# their applications do not accept K>1 bucket sets.
_MULTI_SHAPE_SERVING_MODELS: frozenset[tuple[str, str]] = frozenset(
    {
        ("hunyuan_video", "video"),
        ("wan", "video"),
        ("flux", "image"),
        ("qwen_image", "image"),
    }
)


def build_serving_profile(
    *,
    model_id: str,
    model_type: str,
    entry: ModelEntry,
    output_modality: OutputModality,
    output_mime_type: str,
    default_fps: int | None,
    default_host_vae: bool,
    revision: str | None,
    cache_dir: str | None,
    tp_degree: int | None,
    cp_degree: int | None,
    cp_mode: str | None,
    cfg_parallel: bool | None,
    sp_enabled: bool | None,
    height: int | None,
    width: int | None,
    num_frames: int | None,
    shapes: str | None = None,
    host_vae: bool,
    clip_placement: ServingPlacement | None,
    teacache_cadence: int | None,
    teacache_online_delta: float | None,
    teacache_speedup: float | None,
    teacache_calibration: str | None,
) -> ServingProfile:
    """Resolve registry defaults plus `difflet serve` overrides."""

    if output_modality == "image" and num_frames is not None:
        raise invalid_extra_body(
            "--num-frames is reserved for future video serving; Qwen/Flux image serving "
            "requires num_frames to be omitted."
        )
    if output_modality == "image" and host_vae:
        raise invalid_extra_body("Qwen/Flux image serving does not support --host-vae.")
    if output_modality == "image" and clip_placement is not None:
        raise invalid_extra_body(
            "Qwen/Flux image serving does not support CLIP placement overrides."
        )
    if clip_placement is not None and model_type != "hunyuan_video":
        raise invalid_extra_body(
            f"{model_id} does not support --clip-placement; it is HunyuanVideo-only."
        )
    if cfg_parallel:
        raise invalid_extra_body(
            f"{model_id} serving does not expose the true-CFG request path required "
            "by --cfg-parallel."
        )
    if sp_enabled and model_type not in {"flux", "wan", "hunyuan_video"}:
        raise invalid_extra_body(f"{model_id} does not support --sp serving.")
    if teacache_cadence is not None or teacache_online_delta is not None:
        raise invalid_extra_body(
            "resident serving does not yet implement --teacache-cadence or "
            "--teacache-online-delta; use adaptive --teacache-speedup with a "
            "calibration file."
        )
    if output_modality == "video" and teacache_speedup is not None:
        raise invalid_extra_body("resident video serving does not yet expose adaptive TeaCache.")
    canonical_shapes = None
    if shapes:
        if (model_type, output_modality) not in _MULTI_SHAPE_SERVING_MODELS:
            raise invalid_extra_body(
                "--shapes (multi-shape bucketed serving) is supported for "
                "HunyuanVideo/Wan video serving and Flux/Qwen-Image image serving only."
            )
        if teacache_speedup is not None:
            raise invalid_extra_body(
                "multi-shape serving does not support TeaCache: calibrations are frozen "
                "per shape; drop --shapes or --teacache-speedup."
            )
        from difflet.backends.trainium.core.bucketing import canonicalize_shapes
        from difflet.cli.orchestrators.base import parse_shapes_arg

        try:
            canonical_shapes = canonicalize_shapes(parse_shapes_arg(shapes))
        except ValueError as exc:
            raise invalid_extra_body(f"invalid --shapes: {exc}") from exc
        if output_modality == "video" and canonical_shapes[0][2] is None:
            raise invalid_extra_body(
                "invalid --shapes: video serving takes HxWxF entries (e.g. 480x832x9)."
            )
        if output_modality == "image" and canonical_shapes[0][2] is not None:
            raise invalid_extra_body(
                "invalid --shapes: image serving takes HxW entries (e.g. 1024x1024)."
            )
        # The profile's single h/w/f is pinned to the largest (priority) shape.
        height, width, num_frames = canonical_shapes[0]
    shape = entry.resolve_shape(
        height=height,
        width=width,
        num_frames=num_frames if output_modality == "video" else None,
    )
    resolved_height = shape.get("height")
    resolved_width = shape.get("width")
    if resolved_height is None or resolved_width is None:
        raise ValueError(f"model {model_type!r} must define default height and width")

    frozen_calibration = None
    if teacache_speedup is not None:
        if not math.isfinite(teacache_speedup) or teacache_speedup <= 0:
            raise invalid_extra_body("--teacache-speedup must be a finite positive number.")
        if not teacache_calibration:
            raise invalid_extra_body("--teacache-speedup requires --teacache-calibration PATH.")
        try:
            frozen_calibration = _load_serving_teacache_calibration(
                teacache_calibration,
                model=model_type,
                shape_label=f"{int(resolved_height)}x{int(resolved_width)}",
                requested_speedup=teacache_speedup,
            )
        except (OSError, TypeError, ValueError, OverflowError, KeyError) as exc:
            raise invalid_extra_body(f"invalid TeaCache calibration: {exc}") from exc

    default_parallel = entry.default_parallel
    resolved_cfg_parallel = (
        default_parallel.cfg_parallel_enabled if cfg_parallel is None else cfg_parallel
    )
    resolved_sp = default_parallel.sp_enabled if sp_enabled is None else sp_enabled
    try:
        parallel = DiffletParallelConfig(
            tp_degree=tp_degree if tp_degree is not None else default_parallel.tp_degree,
            cp_degree=cp_degree if cp_degree is not None else default_parallel.cp_degree,
            cp_mode=cp_mode if cp_mode is not None else default_parallel.cp_mode,
            cfg_parallel_enabled=resolved_cfg_parallel,
            sp_enabled=resolved_sp,
            dp_degree=default_parallel.dp_degree,
        )
    except ValueError as exc:
        raise invalid_extra_body(f"invalid serving parallel profile: {exc}") from exc
    if model_type == "qwen_image" and parallel.cp_degree != 1:
        raise invalid_extra_body("Qwen-Image P0 serving requires cp_degree=1.")

    resolved_num_frames = shape.get("num_frames")
    if output_modality == "video":
        if resolved_num_frames is None or int(resolved_num_frames) <= 0:
            raise ValueError(f"video model {model_type!r} must define positive num_frames")
        if default_fps is None or default_fps <= 0:
            raise ValueError(f"video model {model_type!r} must define a positive FPS")
    else:
        resolved_num_frames = None

    resolved_clip_placement: ServingPlacement | None = None
    if model_type == "hunyuan_video":
        resolved_clip_placement = clip_placement or "host"

    return ServingProfile(
        model_id=model_id,
        model_type=model_type,
        height=int(resolved_height),
        width=int(resolved_width),
        num_frames=int(resolved_num_frames) if resolved_num_frames is not None else None,
        parallel=parallel,
        cache_dir=cache_dir,
        revision=revision,
        output_modality=output_modality,
        output_mime_type=output_mime_type,
        teacache_speedup=teacache_speedup,
        teacache_calibration=None,
        teacache_calibration_data=frozen_calibration,
        output_fps=default_fps if output_modality == "video" else None,
        host_vae=(default_host_vae or host_vae) if output_modality == "video" else False,
        clip_placement=resolved_clip_placement,
        shapes=canonical_shapes,
    )


def _load_serving_teacache_calibration(
    path: str,
    *,
    model: str,
    shape_label: str,
    requested_speedup: float,
):
    from difflet.pipeline.teacache import CALIBRATION_SCHEMA, TeaCacheCalibration

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("TeaCache calibration must be a JSON object")
    if raw.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"unsupported TeaCache calibration schema: {raw.get('schema')!r}")
    if type(raw.get("model")) is not str or raw["model"] != model:
        raise ValueError(f"TeaCache calibration model must be {model!r}")
    if type(raw.get("shape_label")) is not str or raw["shape_label"] != shape_label:
        raise ValueError(f"TeaCache calibration shape must be {shape_label!r}")

    num_steps = _strict_int(raw.get("num_steps"), "num_steps", minimum=1)
    poly_coef = raw.get("poly_coef")
    if not isinstance(poly_coef, list) or not poly_coef:
        raise ValueError("poly_coef must be a nonempty JSON array")
    for value in poly_coef:
        _strict_finite_number(value, "poly_coef entry")
    _strict_finite_number(raw.get("threshold"), "threshold", minimum=0.0)
    warmup_steps = _strict_int(raw.get("warmup_steps", 5), "warmup_steps", minimum=0)
    cooldown_steps = _strict_int(raw.get("cooldown_steps", 5), "cooldown_steps", minimum=0)
    if warmup_steps + cooldown_steps >= num_steps:
        raise ValueError("warmup_steps + cooldown_steps must be lower than num_steps")
    _strict_int(raw.get("skip_run_length", 1), "skip_run_length", minimum=1)
    if type(raw.get("accumulate", False)) is not bool:
        raise ValueError("accumulate must be a JSON boolean")
    if _strict_int(raw.get("cadence", 0), "cadence", minimum=0) != 0:
        raise ValueError("resident serving supports adaptive TeaCache only; cadence must be 0")
    if (
        _strict_finite_number(
            raw.get("online_delta_alpha", 0.0),
            "online_delta_alpha",
            minimum=0.0,
        )
        != 0.0
    ):
        raise ValueError(
            "resident serving supports adaptive TeaCache only; online_delta_alpha must be 0"
        )
    if raw.get("mod_input_source", "block0_modulated_input") != "block0_modulated_input":
        raise ValueError("unsupported TeaCache mod_input_source")

    target_speedup = raw.get("target_speedup")
    if target_speedup is not None:
        target_speedup = _strict_finite_number(
            target_speedup,
            "target_speedup",
            minimum=0.0,
            minimum_inclusive=False,
        )
        if requested_speedup > target_speedup + 1e-6:
            raise ValueError("calibration target speedup is lower than requested speedup")
    if raw.get("fit_r2") is not None:
        _strict_finite_number(raw["fit_r2"], "fit_r2")
    if raw.get("n_samples") is not None:
        _strict_int(raw["n_samples"], "n_samples", minimum=1)

    return TeaCacheCalibration.from_dict(raw)


def _strict_int(value: Any, name: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _strict_finite_number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite JSON number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite JSON number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    if minimum is not None:
        valid = value >= minimum if minimum_inclusive else value > minimum
        if not valid:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{name} must be {operator} {minimum}")
    return value
