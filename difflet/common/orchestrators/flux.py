"""Shared Flux serving/CLI helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from difflet.pipeline.compile_cache import CacheSpec
from difflet.serving.types import (
    ArtifactPublishTarget,
    CompileArtifactIdentity,
    DiffletCompileSpec,
    ResolvedModelSource,
    ServingProfile,
)

HF_MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_TYPE = "flux"
MAX_SEQUENCE_LENGTH = 512


def build_pipeline(
    model_id: str,
    profile: ServingProfile,
    *,
    load: bool,
    force_compile: bool = False,
    skip_compile: bool = False,
    model_path_override: str | None = None,
    resolved_source_id: str | None = None,
    compiled_path_override: str | None = None,
):
    from difflet.pipeline.difflet_pipeline import DiffletPipeline

    return DiffletPipeline.from_pretrained(
        model_id,
        model_type=MODEL_TYPE,
        parallel=profile.parallel,
        dtype=_torch_bfloat16(),
        height=profile.height,
        width=profile.width,
        compile_cache_dir=profile.cache_dir,
        revision=profile.revision,
        local_files_only=True,
        force_compile=force_compile,
        skip_compile=skip_compile,
        load=load,
        teacache_speedup=profile.teacache_speedup,
        teacache_calibration=profile.teacache_calibration_data,
        teacache_calibration_path=profile.teacache_calibration,
        model_path_override=model_path_override,
        resolved_source_id=resolved_source_id,
        compiled_path_override=compiled_path_override,
    )


def pipeline_artifacts_ready(pipe) -> bool:
    from difflet.pipeline.compile_cache import has_valid_manifest

    if not has_valid_manifest(pipe.compiled_path, pipe.cache_spec):
        return False
    checker = getattr(pipe.app, "has_compiled_artifacts", None)
    if checker is None:
        return True
    return bool(checker(str(pipe.compiled_path)))


def build_compile_plan(
    source: ResolvedModelSource,
    profile: ServingProfile,
) -> tuple[DiffletCompileSpec, ...]:
    application_kwargs = {
        "teacache_probe_enabled": profile.teacache_speedup is not None,
    }
    cache_spec = CacheSpec(
        model_id=source.model_id,
        model_path=source.pinned_model_path,
        model_name=MODEL_TYPE,
        parallel=profile.parallel,
        dtype=_torch_bfloat16(),
        height=profile.height,
        width=profile.width,
        num_frames=profile.num_frames,
        revision=source.resolved_source_id,
        application_kwargs=application_kwargs or None,
    )
    identity = CompileArtifactIdentity.from_cache_inputs(
        {"component_id": "pipeline", "cache_inputs": cache_spec.cache_inputs()}
    )
    return (
        DiffletCompileSpec(
            artifact_id="pipeline",
            component_id="pipeline",
            identity=identity,
        ),
    )


def compile_serving_artifact(
    source: ResolvedModelSource,
    profile: ServingProfile,
    spec: DiffletCompileSpec,
    target: ArtifactPublishTarget,
) -> None:
    _validate_spec_target(spec, target)
    with _serving_compile_environment(profile.world_size):
        pipe = build_pipeline(
            source.model_id,
            profile,
            load=False,
            force_compile=True,
            model_path_override=source.pinned_model_path,
            resolved_source_id=source.resolved_source_id,
            compiled_path_override=str(target.staging_path),
        )
    if not pipeline_artifacts_ready(pipe):
        raise ValueError(f"Flux compile produced incomplete payload at {target.staging_path}")


def validate_compiled_artifact(
    source: ResolvedModelSource,
    profile: ServingProfile,
    spec: DiffletCompileSpec,
    artifact_root: Path,
) -> None:
    if spec.component_id != "pipeline":
        raise ValueError(f"unknown Flux component {spec.component_id!r}")
    pipe = build_pipeline(
        source.model_id,
        profile,
        load=False,
        skip_compile=True,
        model_path_override=source.pinned_model_path,
        resolved_source_id=source.resolved_source_id,
        compiled_path_override=str(artifact_root),
    )
    if not pipeline_artifacts_ready(pipe):
        raise ValueError(f"Flux artifact payload is incomplete at {artifact_root}")


def _validate_spec_target(
    spec: DiffletCompileSpec,
    target: ArtifactPublishTarget,
) -> None:
    if spec.component_id != "pipeline":
        raise ValueError(f"unknown Flux component {spec.component_id!r}")
    if spec.artifact_id != target.artifact_id or spec.identity != target.identity:
        raise ValueError("Flux compile target does not match compile spec")


@contextmanager
def _serving_compile_environment(world_size: int):
    names = (
        "NEURON_RT_VISIBLE_CORES",
        "NEURON_RT_NUM_CORES",
        "NEURON_RT_VIRTUAL_CORE_SIZE",
        "NEURON_LOGICAL_NC_CONFIG",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
    )
    original = {name: os.environ.get(name) for name in names}
    try:
        os.environ["NEURON_RT_VISIBLE_CORES"] = ",".join(str(index) for index in range(world_size))
        os.environ["NEURON_RT_NUM_CORES"] = str(world_size)
        os.environ.pop("NEURON_RT_VIRTUAL_CORE_SIZE", None)
        os.environ.pop("NEURON_LOGICAL_NC_CONFIG", None)
        os.environ.update(
            {"WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}
        )
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _torch_bfloat16():
    import torch

    return torch.bfloat16
