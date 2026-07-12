"""Configuration objects for `difflet serve`."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.registry import ModelEntry
from difflet.serving.errors import invalid_extra_body
from difflet.serving.types import ServingProfile


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
    tp_degree: int | None = None
    cp_degree: int | None = None
    cp_mode: str | None = None
    cfg_parallel: bool | None = None
    sp_enabled: bool | None = None
    height: int | None = None
    width: int | None = None
    num_frames: int | None = None
    cache_dir: str | None = None
    host_vae: bool = False
    teacache_cadence: int | None = None
    teacache_online_delta: float | None = None
    teacache_speedup: float | None = None
    teacache_calibration: str | None = None
    download_policy: DownloadPolicy = DownloadPolicy.AUTO
    compile_policy: CompilePolicy = CompilePolicy.AUTO
    max_running_requests: int = 1
    max_queued_requests: int = 8
    queue_timeout: float = 30.0
    request_timeout: float = 300.0
    artifact_store_timeout: float = 60.0
    worker_cancel_timeout: float = 10.0
    worker_restart_timeout: float = 900.0
    worker_heartbeat_interval: float = 30.0
    artifact_store: str = "r2"
    artifact_ttl_seconds: int = 3600


def build_serving_profile(
    *,
    model_id: str,
    model_type: str,
    entry: ModelEntry,
    output_mime_type: str,
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
    host_vae: bool,
    teacache_cadence: int | None,
    teacache_online_delta: float | None,
    teacache_speedup: float | None,
    teacache_calibration: str | None,
) -> ServingProfile:
    """Resolve registry defaults plus `difflet serve` overrides."""

    if num_frames is not None:
        raise invalid_extra_body(
            "--num-frames is reserved for future video serving; Qwen/Flux image serving "
            "requires num_frames to be omitted."
        )
    if host_vae:
        raise invalid_extra_body("Qwen/Flux image serving does not support --host-vae.")
    if cfg_parallel:
        raise invalid_extra_body(
            f"{model_id} serving does not expose the true-CFG request path required "
            "by --cfg-parallel."
        )
    if sp_enabled and model_type != "flux":
        raise invalid_extra_body(f"{model_id} does not support --sp serving.")
    if teacache_cadence is not None or teacache_online_delta is not None:
        raise invalid_extra_body(
            "resident serving does not yet implement --teacache-cadence or "
            "--teacache-online-delta; use adaptive --teacache-speedup with a "
            "calibration file."
        )
    shape = entry.resolve_shape(height=height, width=width, num_frames=None)
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

    return ServingProfile(
        model_id=model_id,
        model_type=model_type,
        height=int(resolved_height),
        width=int(resolved_width),
        num_frames=None,
        parallel=parallel,
        cache_dir=cache_dir,
        revision=revision,
        output_modality="image",
        output_mime_type=output_mime_type,
        teacache_speedup=teacache_speedup,
        teacache_calibration=None,
        teacache_calibration_data=frozen_calibration,
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
