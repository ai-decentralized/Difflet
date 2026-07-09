"""Configuration objects for `difflet serve`."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
    height: int | None = None
    width: int | None = None
    num_frames: int | None = None
    cache_dir: str | None = None
    download_policy: DownloadPolicy = DownloadPolicy.AUTO
    compile_policy: CompilePolicy = CompilePolicy.AUTO
    max_running_requests: int = 1
    max_queued_requests: int = 8
    queue_timeout: float = 30.0
    request_timeout: float = 300.0
    artifact_store_timeout: float = 60.0
    worker_cancel_timeout: float = 10.0
    worker_restart_timeout: float = 900.0
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
    height: int | None,
    width: int | None,
    num_frames: int | None,
) -> ServingProfile:
    """Resolve registry defaults plus `difflet serve` overrides."""

    if num_frames is not None:
        raise invalid_extra_body(
            "--num-frames is reserved for future video serving; Qwen/Flux image serving "
            "requires num_frames to be omitted."
        )

    shape = entry.resolve_shape(height=height, width=width, num_frames=None)
    resolved_height = shape.get("height")
    resolved_width = shape.get("width")
    if resolved_height is None or resolved_width is None:
        raise ValueError(f"model {model_type!r} must define default height and width")

    default_parallel = entry.default_parallel
    parallel = DiffletParallelConfig(
        tp_degree=tp_degree or default_parallel.tp_degree,
        cp_degree=cp_degree if cp_degree is not None else default_parallel.cp_degree,
        cp_mode=cp_mode or default_parallel.cp_mode,
        cfg_parallel_enabled=False,
        sp_enabled=False,
        dp_degree=default_parallel.dp_degree,
    )
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
    )

