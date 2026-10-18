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

# Models served on the CUDA backend (one process per GPU, no Neuron stages),
# keyed by registry entry name. Selected by backend name the way
# ``difflet.cli.main._get_orchestrator`` selects the CLI orchestrator; on every other
# backend the Trainium metadata above is used unchanged.
_CUDA_SERVING_OVERLAYS: dict[str, str] = {
    "qwen_image": "difflet.common.registry.qwen_image:cuda_serving_metadata",
}


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
    from difflet.backends.registry import current_backend

    tp_degree = options.tp_degree
    _refuse_unsupported_replicas(int(options.replicas), current_backend())
    if current_backend() == "cuda":
        metadata = _cuda_serving_metadata(entry.name, options.model_id)
        if tp_degree is None:
            # Release 1 is one card: the registry default (tp=4) is a Trainium topology, and
            # the CUDA CLI uses `args.tp_degree or 1`.
            tp_degree = 1
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


def _visible_gpu_count() -> int:
    import torch

    return int(torch.cuda.device_count())


def _refuse_unsupported_replicas(dp_degree: int, backend: str) -> None:
    """``--dp-degree N > 1`` is CUDA's one-worker-per-GPU replica set, refused
    before any snapshot resolution elsewhere, and on CUDA when N exceeds the visible GPUs."""
    if dp_degree <= 1:
        return
    from difflet.serving.errors import feature_not_supported

    if backend != "cuda":
        raise feature_not_supported(
            f"--dp-degree {dp_degree}: data-parallel serving replicas (one resident worker "
            f"per GPU) exist on the CUDA backend only; the {backend} "
            "backend serves one resident worker"
        )
    count = _visible_gpu_count()
    if dp_degree > count:
        raise feature_not_supported(
            f"--dp-degree {dp_degree} exceeds the {count} visible GPU(s) "
            "(CUDA_VISIBLE_DEVICES); one resident replica needs one GPU"
        )


def _cuda_serving_metadata(entry_name: str, model_id: str) -> ServingModelMetadata:
    path = _CUDA_SERVING_OVERLAYS.get(entry_name)
    if path is None:
        raise ValueError(
            f"{model_id} cannot be served on the CUDA backend: serving on CUDA exists for "
            "Qwen-Image only (Wan and Flux generate on CUDA through `difflet run`; serving "
            "them is not supported on the CUDA backend yet); set DIFFLET_BACKEND=trainium for the staged "
            "Neuron path"
        )
    return load_factory(path)()


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
