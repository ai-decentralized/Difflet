"""Serving adapter protocols."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, Sequence

from difflet.common.neuron_cores import resolve_available_neuron_core_ids
from difflet.serving.errors import invalid_extra_body
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.types import (
    DiffletGenerateRequest,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    ServingProfile,
)


def resolve_hf_model_source(
    model_id: str,
    *,
    revision: str | None,
    download_policy: DownloadPolicy,
    allow_patterns: Sequence[str] | None = None,
) -> ResolvedModelSource:
    """Resolve one commit-addressed HF snapshot for resident serving."""

    if Path(model_id).expanduser().exists():
        raise ValueError("P0 resident serving requires a Hugging Face model ID")

    from difflet.pipeline.path_resolver import resolve_model_path

    raw_path = Path(
        resolve_model_path(
            model_id,
            revision=revision,
            local_files_only=download_policy == DownloadPolicy.NEVER,
            allow_patterns=allow_patterns,
        )
    ).expanduser()
    parts = raw_path.parts
    try:
        snapshot_index = len(parts) - 1 - tuple(reversed(parts)).index("snapshots")
        resolved_source_id = parts[snapshot_index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"resolved Hugging Face path is not commit-addressed: {raw_path}") from exc
    if not re.fullmatch(r"[0-9a-f]{40,64}", resolved_source_id):
        raise ValueError(
            f"resolved Hugging Face snapshot has invalid commit ID: {resolved_source_id!r}"
        )
    pinned_path = raw_path.resolve()
    if not pinned_path.is_dir():
        raise ValueError(f"resolved Hugging Face snapshot does not exist: {pinned_path}")
    return ResolvedModelSource(
        source_kind="hf_snapshot",
        model_id=model_id,
        requested_revision=revision,
        pinned_model_path=str(pinned_path),
        resolved_source_id=resolved_source_id,
    )


class ServingArtifactPreparer(Protocol):
    model_id: str
    model_type: str

    def prepare_runtime(
        self,
        profile: ServingProfile,
        *,
        download_policy: DownloadPolicy,
        compile_policy: CompilePolicy,
    ) -> ResolvedRuntimeBundle: ...


class ServingRequestValidator(Protocol):
    def validate(self, request: DiffletGenerateRequest) -> None: ...


class NoopServingRequestValidator:
    def __init__(self, runtime: ResolvedRuntimeBundle) -> None:
        self.runtime = runtime

    def validate(self, request: DiffletGenerateRequest) -> None:
        return None


def validate_guidance_scale(
    request: DiffletGenerateRequest,
    *,
    maximum: float,
) -> None:
    guidance = request.guidance_scale
    if guidance < 0 or guidance > maximum:
        raise invalid_extra_body(f"guidance_scale must satisfy 0 <= value <= {maximum:g}")


def request_uses_teacache(profile: ServingProfile, num_inference_steps: int) -> bool:
    calibration = profile.teacache_calibration_data
    return bool(
        profile.teacache_speedup is not None
        and calibration is not None
        and int(num_inference_steps) == int(calibration.num_steps)
    )
