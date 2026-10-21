"""Serving-specific model registry overlay."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from difflet.common.registry.base import ServingModelMetadata
from difflet.registry import ModelEntry, resolve_model
from difflet.serving.options import ServeOptions, build_serving_profile
from difflet.serving.types import ServingProfile

Factory = Callable[..., Any]

_SERVING_METADATA: dict[str, ServingModelMetadata] = {}
_BUILTINS_LOADED = False


@dataclass(frozen=True)
class ResolvedServingModel:
    model_id: str
    base_entry: ModelEntry
    metadata: ServingModelMetadata
    profile: ServingProfile


def register_serving_model(metadata: ServingModelMetadata) -> ServingModelMetadata:
    _SERVING_METADATA[metadata.model_type] = metadata
    return metadata


def resolve_serving_model(options: ServeOptions) -> ResolvedServingModel:
    _ensure_builtin_serving_models_registered()
    entry = resolve_model(options.model_id)
    try:
        metadata = _SERVING_METADATA[entry.name]
    except KeyError as exc:
        raise ValueError(f"model type {entry.name!r} is not enabled for serving") from exc
    normalized = options.model_id.rstrip("/")
    if normalized not in metadata.checkpoint_ids:
        allowed = ", ".join(metadata.checkpoint_ids)
        raise ValueError(
            f"checkpoint {options.model_id!r} is not enabled for P0 serving; "
            f"allowed for {entry.name}: {allowed}"
        )
    from difflet.platforms import current_platform

    tp_degree = options.tp_degree
    platform = current_platform()
    on_gpu = platform.capabilities.flow == "gpu"
    if on_gpu:
        # The GPU flow's refusals are DiffletServingErrors, so `difflet serve` exits with the
        # message.
        from difflet.serving.errors import feature_not_supported

        metadata = _gpu_serving_metadata(entry.name, options.model_id, platform)
        if normalized not in metadata.checkpoint_ids:
            allowed = ", ".join(metadata.checkpoint_ids)
            raise feature_not_supported(
                f"checkpoint {options.model_id!r} is not served on the "
                f"{platform.display_name} backend yet; served for {entry.name}: {allowed}"
            )
        # Before the GPU count: a video model cannot use replicas on any box.
        _refuse_video_replicas(int(options.replicas), metadata)
        if options.teacache_calibration and options.teacache_speedup is None:
            # The profile keeps no calibration without a speedup, so the file would be ignored
            # silently; the GPU CLI refuses it by name the same way.
            raise feature_not_supported(
                "--teacache-calibration without --teacache-speedup is ignored by resident "
                f"serving; on the {platform.display_name} backend TeaCache is a `difflet run` "
                "option (Qwen-Image's fixed cadence) and not served — drop the flag"
            )
    _refuse_unsupported_replicas(int(options.replicas), platform)
    if on_gpu and tp_degree is None:
        # The registry default (tp=4) is a Trainium topology; the GPU CLI uses
        # `args.tp_degree or 1`.
        tp_degree = platform.default_tp_degree()
    profile = build_serving_profile(
        model_id=options.model_id,
        model_type=metadata.model_type,
        entry=entry,
        output_modality=metadata.output_modality,
        output_mime_type=metadata.output_mime_type,
        default_fps=metadata.default_fps,
        default_host_vae=metadata.default_host_vae,
        revision=options.revision,
        cache_dir=options.cache_dir,
        tp_degree=tp_degree,
        cp_degree=options.cp_degree,
        cp_mode=options.cp_mode,
        cfg_parallel=options.cfg_parallel,
        sp_enabled=options.sp_enabled,
        height=options.height,
        width=options.width,
        num_frames=options.num_frames,
        shapes=getattr(options, "shapes", None),
        host_vae=options.host_vae,
        clip_placement=options.clip_placement,
        teacache_cadence=options.teacache_cadence,
        teacache_online_delta=options.teacache_online_delta,
        teacache_speedup=options.teacache_speedup,
        teacache_calibration=options.teacache_calibration,
        torch_compile=bool(getattr(options, "torch_compile", False)),
    )
    return ResolvedServingModel(
        model_id=options.model_id,
        base_entry=entry,
        metadata=metadata,
        profile=profile,
    )


def _visible_gpu_count(platform) -> int:
    return int(platform.device_count() or 0)


def _refuse_unsupported_replicas(dp_degree: int, platform) -> None:
    """``--dp-degree N > 1`` is the GPU flow's one-worker-per-GPU replica set, refused
    before any snapshot resolution elsewhere, and on a GPU platform when N exceeds the visible
    GPUs."""
    if dp_degree <= 1:
        return
    from difflet.serving.errors import feature_not_supported

    if platform.capabilities.flow != "gpu":
        raise feature_not_supported(
            f"--dp-degree {dp_degree}: data-parallel serving replicas (one resident worker "
            f"per GPU) exist on GPU backends only; the {platform.name} "
            "backend serves one resident worker"
        )
    count = _visible_gpu_count(platform)
    if dp_degree > count:
        raise feature_not_supported(
            f"--dp-degree {dp_degree} exceeds the {count} visible GPU(s) "
            f"({platform.visible_devices_var}); one resident replica needs one GPU"
        )


def _refuse_video_replicas(dp_degree: int, metadata: ServingModelMetadata) -> None:
    """``--dp-degree N > 1`` for a video model on CUDA: the video service
    dispatches one video job at a time, so every replica past the first would only hold a GPU."""
    if dp_degree <= 1 or metadata.output_modality != "video":
        return
    from difflet.serving.errors import feature_not_supported

    raise feature_not_supported(
        f"--dp-degree {dp_degree}: {metadata.model_type} serves video, and the video service "
        "runs one video job at a time, so a second resident replica would sit idle; serve "
        "with one replica (a concurrent video dispatcher is a follow-up)"
    )


def _gpu_serving_metadata(entry_name: str, model_id: str, platform) -> ServingModelMetadata:
    from difflet.models.spec import gpu_model_spec, gpu_model_specs

    spec = gpu_model_spec(entry_name)
    if spec is None:
        from difflet.serving.errors import feature_not_supported

        served = ", ".join(c for s in gpu_model_specs() for c in s.served_checkpoint_ids)
        raise feature_not_supported(
            f"{model_id} cannot be served on the {platform.display_name} backend: serving there "
            f"exists for {served}; set DIFFLET_BACKEND=trainium for the staged Neuron path"
        )
    from difflet.backends.gpu.serving import gpu_serving_metadata

    return gpu_serving_metadata(spec)


def _ensure_builtin_serving_models_registered() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from difflet.common.registry import flux, hunyuan_video, ltx_2, qwen_image, wan

    register_serving_model(flux.serving_metadata())
    register_serving_model(hunyuan_video.serving_metadata())
    register_serving_model(ltx_2.serving_metadata())
    register_serving_model(qwen_image.serving_metadata())
    register_serving_model(wan.serving_metadata())


def load_factory(path: str) -> Factory:
    module_name, sep, attr = path.partition(":")
    if sep != ":":
        raise ValueError(f"invalid factory reference {path!r}")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    if not callable(factory):
        raise TypeError(f"factory {path!r} is not callable")
    return factory


def load_artifact_preparer_factory(metadata: ServingModelMetadata) -> Factory:
    return load_factory(metadata.artifact_preparer_factory)


def load_request_validator_factory(metadata: ServingModelMetadata) -> Factory | None:
    if metadata.request_validator_factory is None:
        return None
    return load_factory(metadata.request_validator_factory)
