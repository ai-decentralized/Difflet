"""Serving adapter protocols."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Protocol, Sequence

from difflet.serving.errors import invalid_extra_body
from difflet.serving.options import CompilePolicy, DownloadPolicy
from difflet.serving.types import (
    DiffletGenerateRequest,
    ResolvedModelSource,
    ResolvedRuntimeBundle,
    ServingProfile,
)

DEFAULT_NEURON_CORE_IDS: tuple[int, ...] = (0, 1, 2, 3)


def resolve_available_neuron_core_ids(*, required_num_cores: int) -> tuple[int, ...]:
    """Resolve inherited Neuron core visibility, defaulting to the four-core host."""

    raw = os.environ.get("NEURON_RT_VISIBLE_CORES")
    if raw is None or not raw.strip():
        core_ids = DEFAULT_NEURON_CORE_IDS
    else:
        resolved: list[int] = []
        for item in raw.split(","):
            token = item.strip()
            if not re.fullmatch(r"\d+(?:-\d+)?", token):
                raise ValueError(
                    "NEURON_RT_VISIBLE_CORES must contain comma-separated core IDs or ranges"
                )
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError("NEURON_RT_VISIBLE_CORES ranges must be ascending")
                resolved.extend(range(start, end + 1))
            else:
                resolved.append(int(token))
        core_ids = tuple(resolved)

    if len(core_ids) != len(set(core_ids)):
        raise ValueError("NEURON_RT_VISIBLE_CORES must not contain duplicate core IDs")
    if len(core_ids) < required_num_cores:
        raise ValueError(
            "NEURON_RT_VISIBLE_CORES does not provide enough cores: "
            f"requires {required_num_cores}, has {len(core_ids)}"
        )
    return core_ids


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
