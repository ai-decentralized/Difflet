"""Shared data contracts for Difflet serving."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar

from difflet.pipeline.parallel_config import DiffletParallelConfig

if TYPE_CHECKING:
    from difflet.pipeline.teacache import TeaCacheCalibration

OutputModality = Literal["image", "video"]
StageRole = Literal["prompt_encoder", "denoiser", "decoder", "pipeline"]
StageKind = Literal["extracted", "opaque_pipeline"]
StagePlacement = Literal["host", "neuron", "hybrid"]


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    kind: StageKind
    role: StageRole
    output_keys: tuple[str, ...] = ()
    final_output: bool = False
    runner_factory: str | None = None

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise ValueError("stage_id must not be empty")
        if self.kind == "extracted" and not self.runner_factory:
            raise ValueError(f"extracted stage {self.stage_id!r} requires runner_factory")
        if self.kind == "opaque_pipeline" and self.runner_factory is not None:
            raise ValueError(
                f"opaque pipeline stage {self.stage_id!r} must not define runner_factory"
            )


@dataclass(frozen=True)
class PipelineDefinition:
    model_type: str
    stages: tuple[StageDefinition, ...]

    def __post_init__(self) -> None:
        if not self.model_type:
            raise ValueError("model_type must not be empty")
        if not self.stages:
            raise ValueError("pipeline must define at least one stage")
        stage_ids = tuple(stage.stage_id for stage in self.stages)
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("pipeline stage IDs must be unique")
        final_indexes = [index for index, stage in enumerate(self.stages) if stage.final_output]
        if final_indexes != [len(self.stages) - 1]:
            raise ValueError("pipeline must have exactly one final stage in the last position")


@dataclass(frozen=True)
class DistributedProcessEnvironment:
    world_size: int
    local_world_size: int
    rank: int
    local_rank: int


@dataclass(frozen=True)
class RuntimeEnvironment:
    available_core_ids: tuple[int, ...]
    num_cores_override: int | None
    virtual_core_size_override: int | None
    logical_nc_config_override: int | None
    inherited_distributed: DistributedProcessEnvironment
    child_distributed: DistributedProcessEnvironment


@dataclass(frozen=True)
class WorkerAllocationSpec:
    allocation_id: str
    requested_num_cores: int
    effective_num_cores: int
    world_size: int
    requested_virtual_core_size: int | None = None
    effective_virtual_core_size: int | None = None
    requested_logical_nc_config: int | None = None
    effective_logical_nc_config: int | None = None


@dataclass(frozen=True)
class ParallelTopology:
    tp_degree: int
    cp_degree: int
    world_size: int


@dataclass(frozen=True)
class StageRuntimeSpec:
    stage_id: str
    allocation_id: str | None
    topology: ParallelTopology | None
    artifact_id: str | None
    placement: StagePlacement = "neuron"

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise ValueError("runtime stage_id must not be empty")
        if self.placement == "host":
            if any(
                value is not None for value in (self.allocation_id, self.topology, self.artifact_id)
            ):
                raise ValueError(
                    f"host stage {self.stage_id!r} must not claim a Neuron "
                    "allocation, topology, or compiled artifact"
                )
            return
        if self.placement not in {"neuron", "hybrid"}:
            raise ValueError(f"unsupported stage placement {self.placement!r}")
        if self.allocation_id is None or self.topology is None or self.artifact_id is None:
            raise ValueError(
                f"{self.placement} stage {self.stage_id!r} requires allocation, "
                "topology, and compiled artifact"
            )


@dataclass(frozen=True)
class RuntimePlan:
    mode: Literal["cli_staged", "resident"]
    profile_identity: str
    environment: RuntimeEnvironment
    allocations: tuple[WorkerAllocationSpec, ...]
    stages: tuple[StageRuntimeSpec, ...]

    def validate_against(self, pipeline: PipelineDefinition) -> None:
        pipeline_ids = tuple(stage.stage_id for stage in pipeline.stages)
        runtime_ids = tuple(stage.stage_id for stage in self.stages)
        if runtime_ids != pipeline_ids:
            raise ValueError(
                f"runtime stage IDs/order {runtime_ids!r} do not match pipeline {pipeline_ids!r}"
            )
        allocation_ids = {allocation.allocation_id for allocation in self.allocations}
        if len(allocation_ids) != len(self.allocations):
            raise ValueError("runtime allocation IDs must be unique")
        for stage in self.stages:
            if stage.allocation_id is not None and stage.allocation_id not in allocation_ids:
                raise ValueError(
                    f"stage {stage.stage_id!r} references unknown allocation "
                    f"{stage.allocation_id!r}"
                )


@dataclass(frozen=True)
class CompileArtifactIdentity:
    schema_version: int
    canonical_cache_inputs_json: bytes
    digest: str

    @classmethod
    def from_cache_inputs(
        cls,
        cache_inputs: dict[str, Any],
        *,
        schema_version: int = 1,
    ) -> "CompileArtifactIdentity":
        canonical = json.dumps(
            cache_inputs,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return cls(
            schema_version=schema_version,
            canonical_cache_inputs_json=canonical,
            digest=hashlib.sha256(canonical).hexdigest(),
        )


@dataclass(frozen=True)
class ArtifactBinding:
    artifact_id: str
    path: Path
    manifest_path: Path
    identity: CompileArtifactIdentity
    generation_id: str
    content_digest: str


@dataclass(frozen=True)
class ArtifactPublishTarget:
    artifact_id: str
    identity: CompileArtifactIdentity
    identity_root: Path
    staging_path: Path


@dataclass(frozen=True)
class ArtifactSet:
    bindings: tuple[ArtifactBinding, ...]

    def __post_init__(self) -> None:
        artifact_ids = tuple(binding.artifact_id for binding in self.bindings)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("artifact binding IDs must be unique")

    def require(self, artifact_id: str) -> ArtifactBinding:
        matches = [binding for binding in self.bindings if binding.artifact_id == artifact_id]
        if len(matches) != 1:
            raise ValueError(
                f"artifact {artifact_id!r} must resolve exactly once; found {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True)
class ResolvedModelSource:
    source_kind: Literal["hf_snapshot"]
    model_id: str
    requested_revision: str | None
    pinned_model_path: str
    resolved_source_id: str


@dataclass(frozen=True)
class ServingProfile:
    """The single loaded model/profile identity for a serving process."""

    model_id: str
    model_type: str
    height: int
    width: int
    num_frames: int | None
    parallel: DiffletParallelConfig
    cache_dir: str | None = None
    revision: str | None = None
    dtype: str = "bfloat16"
    output_modality: OutputModality = "image"
    output_mime_type: str = "image/png"
    teacache_speedup: float | None = None
    teacache_calibration: str | None = None
    teacache_calibration_data: TeaCacheCalibration | None = None
    output_fps: int | None = None
    host_vae: bool = False

    @property
    def world_size(self) -> int:
        return self.parallel.world_size

    def shape_dict(self) -> dict[str, int | None]:
        return {"height": self.height, "width": self.width, "num_frames": self.num_frames}


@dataclass(frozen=True)
class DiffletGenerateRequest:
    request_id: str
    model: str
    prompt: str
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    output_format: str = "png"
    video: "VideoGenerateOptions | None" = None


@dataclass(frozen=True)
class DiffletGenerateOutput:
    data: bytes
    mime_type: str
    output_format: str = "png"


@dataclass(frozen=True, slots=True)
class FileOutputTarget:
    """Parent-owned worker output target; never populated from a client filename."""

    staging_path: str
    mime_type: str = "video/mp4"
    output_format: str = "mp4"


@dataclass(frozen=True, slots=True)
class VideoGenerateOptions:
    """Immutable video-only request fields carried through resident-worker IPC."""

    num_frames: int
    fps: int
    output_target: FileOutputTarget | None = None
    negative_prompt: str | None = None
    guidance_scale_2: float | None = None
    boundary_ratio: float | None = None
    flow_shift: float | None = None
    true_cfg_scale: float | None = None
    requested_seconds: str | None = None
    user: str | None = None


@dataclass(frozen=True, slots=True)
class FileBackedGenerateOutput:
    """Small worker result descriptor for media already written to shared storage."""

    path: str
    mime_type: str
    output_format: str
    size_bytes: int
    width: int
    height: int
    num_frames: int
    fps: float
    duration_s: float


GenerateOutput: TypeAlias = DiffletGenerateOutput | FileBackedGenerateOutput


class StagePayload(ABC):
    """Nominal base for logical values passed between serving stages."""

    __slots__ = ()


InputPayloadT = TypeVar("InputPayloadT", bound=StagePayload)
OutputPayloadT = TypeVar("OutputPayloadT", bound=StagePayload)


@dataclass(frozen=True, slots=True)
class QwenInitialPayload(StagePayload):
    pass


@dataclass(frozen=True, slots=True)
class QwenTextPayload(StagePayload):
    encoder_hidden_states: Any
    encoder_hidden_states_mask: Any


@dataclass(frozen=True, slots=True)
class QwenLatentPayload(StagePayload):
    packed_latents: Any


@dataclass(frozen=True, slots=True)
class QwenFinalPayload(StagePayload):
    output: DiffletGenerateOutput


@dataclass(frozen=True, slots=True)
class FluxInitialPayload(StagePayload):
    pass


@dataclass(frozen=True, slots=True)
class FluxFinalPayload(StagePayload):
    output: DiffletGenerateOutput


@dataclass(frozen=True, slots=True)
class StageInvocation(Generic[InputPayloadT]):
    request: DiffletGenerateRequest
    stage: StageDefinition
    input: InputPayloadT
    context: "WorkerRequestContext"


@dataclass(frozen=True, slots=True)
class StageExecutionMetadata:
    started_monotonic: float
    finished_monotonic: float


@dataclass(frozen=True, slots=True)
class StageExecutionResult(Generic[OutputPayloadT]):
    output: OutputPayloadT
    metadata: StageExecutionMetadata


@dataclass(frozen=True)
class DiffletCompileSpec:
    artifact_id: str
    component_id: str
    identity: CompileArtifactIdentity
    required: bool = True


@dataclass(frozen=True)
class ResolvedRuntimeBundle:
    profile: ServingProfile
    source: ResolvedModelSource
    pipeline_definition: PipelineDefinition
    runtime_plan: RuntimePlan
    compile_specs: tuple[DiffletCompileSpec, ...]
    artifacts: ArtifactSet
    adapter_config: Any = None

    def __post_init__(self) -> None:
        if self.profile.model_type != self.pipeline_definition.model_type:
            raise ValueError("profile and pipeline model_type must match")
        if self.profile.model_id != self.source.model_id:
            raise ValueError("profile and source model_id must match")
        self.runtime_plan.validate_against(self.pipeline_definition)
        spec_ids = tuple(spec.artifact_id for spec in self.compile_specs)
        if len(spec_ids) != len(set(spec_ids)):
            raise ValueError("compile spec artifact IDs must be unique")
        binding_ids = tuple(binding.artifact_id for binding in self.artifacts.bindings)
        required_ids = tuple(spec.artifact_id for spec in self.compile_specs if spec.required)
        if set(binding_ids) != set(required_ids):
            raise ValueError("artifact bindings must exactly match required compile specs")
        by_spec = {spec.artifact_id: spec for spec in self.compile_specs}
        for binding in self.artifacts.bindings:
            if binding.identity != by_spec[binding.artifact_id].identity:
                raise ValueError(
                    f"binding identity does not match compile spec {binding.artifact_id!r}"
                )
        for stage in self.runtime_plan.stages:
            if stage.artifact_id is not None and stage.artifact_id not in by_spec:
                raise ValueError(
                    f"runtime stage {stage.stage_id!r} references unknown artifact "
                    f"{stage.artifact_id!r}"
                )

    def require_compile_spec(self, artifact_id: str) -> DiffletCompileSpec:
        matches = [spec for spec in self.compile_specs if spec.artifact_id == artifact_id]
        if len(matches) != 1:
            raise ValueError(
                f"compile spec {artifact_id!r} must resolve exactly once; found {len(matches)}"
            )
        return matches[0]


@dataclass
class CancellationSignal:
    """Cooperative cancellation flag checked at model-safe points."""

    _event: asyncio.Event = field(default_factory=asyncio.Event)
    external_event: Any | None = None

    def cancel(self) -> None:
        self._event.set()
        if self.external_event is not None:
            self.external_event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or (
            self.external_event is not None and self.external_event.is_set()
        )

    def throw_if_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError()


@dataclass(frozen=True)
class WorkerRequestContext:
    request_id: str
    deadline_monotonic: float
    cancellation: CancellationSignal
    stage_callback: Any | None = None

    @classmethod
    def with_timeout(
        cls,
        request_id: str,
        timeout_s: float,
        *,
        cancellation: CancellationSignal | None = None,
        stage_callback: Any | None = None,
    ) -> "WorkerRequestContext":
        return cls(
            request_id=request_id,
            deadline_monotonic=time.monotonic() + float(timeout_s),
            cancellation=cancellation or CancellationSignal(),
            stage_callback=stage_callback,
        )

    def report_stage(self, stage_id: str) -> None:
        if self.stage_callback is not None:
            self.stage_callback(stage_id)
