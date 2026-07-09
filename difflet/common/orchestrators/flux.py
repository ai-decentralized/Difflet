"""Shared Flux serving/CLI helpers."""

from __future__ import annotations

from difflet.serving.types import ServingProfile

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
    )


def pipeline_artifacts_ready(pipe) -> bool:
    from difflet.pipeline.compile_cache import has_valid_manifest

    if not has_valid_manifest(pipe.compiled_path, pipe.cache_spec):
        return False
    checker = getattr(pipe.app, "has_compiled_artifacts", None)
    if checker is None:
        return True
    return bool(checker(str(pipe.compiled_path)))


def _torch_bfloat16():
    import torch

    return torch.bfloat16
