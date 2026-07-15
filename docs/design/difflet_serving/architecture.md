# Difflet Serving Architecture

This document captures the serving code organization agreed for the first
text-to-image serving implementation.

This file, together with `engine.md` and `chat_completions_contract.md`, is the
authoritative P0 serving design. The older
`docs/plans/2026-07-06-difflet-serving-engine.md` keeps broader future design
notes, including rotating, subprocess, and multi-plan ideas, but those are not
part of the P0 implementation scope unless repeated here.

[`implementation_changes.md`](implementation_changes.md) lists the exact delta
from the current branch, including obsolete code to delete; it does not define a
competing runtime contract.

## Scope

MVP serving supports:

- Qwen-Image through a resident worker that owns its stage engine, adapter, and
  process-local runners.
- Flux through a resident worker that owns its loaded `DiffletPipeline`.
- One active model/profile per server process.
- One active serving profile in the resident worker lifecycle.
- OpenAI-style `/v1/chat/completions` as the public API.
- Base64 data URLs by default, with S3-backed artifact URLs when a complete S3
  configuration is present.

MVP does not include subprocess serving fallback, rotating resident fallback,
multi-profile loading, local file serving, or video serving.

Qwen-Image is a P0 target only when the selected `ServingProfile` passes a real
shared-worker co-load and smoke test: prompt encoder, denoiser, and decoder must
load into the same resident Trainium worker process and run a startup smoke
without falling back to per-stage processes or file handoff. If this gate fails,
startup must fail clearly; P0 must not silently downgrade to subprocess or
rotating behavior.

The serving process is "single-active-profile by design": a process resolves and
validates exactly one `ServingProfile` at startup. Request-time overrides for
`tp_degree`, `cp_degree`, `cp_mode`, `cfg_parallel`, or `sp_enabled` are not
supported. The active profile can be changed only by restarting the serving worker
(process teardown + load + smoke), not by changing one request field.


## Code Ownership

| Layer | Owns | Must not own |
|---|---|---|
| `difflet/registry.py` | Existing `ModelEntry`, model matching, aliases/HF paths, download patterns, default shape/parallel config, backend support, pipeline factory | Serving lifecycle or topology policy |
| `difflet/common/registry/<model>.py` | Shared family metadata and stable stage roles derived from the base registry | Duplicate detectors, aliases, or base defaults |
| `difflet/common/orchestrators/<model>.py` | Shared model constants, CLI-compatible builders, and model payload validation helpers still used by adapters | Serving cache-path selection, publication, FastAPI, or worker IPC |
| `difflet/cli/orchestrators/<model>.py` | CLI parsing, file workflow, subprocess traversal, and compatibility wrappers | Reusable model runtime logic |
| `difflet/serving/model_registry.py` | Serving checkpoint allowlist, topology type, output modality, artifact policy, validator/artifact-preparer/stage-adapter factories | Broad model-ID matching or duplicate defaults |
| `difflet/serving/artifact_manager.py` | Canonical identity lookup, lock, staging, payload inventory, manifest, immutable publication, and direct binding | Model compile topology or worker loading |
| `difflet/serving/orchestrators/<model>.py` | Model adapter config/compile spec/compile/validation, runner construction/load, concrete payload validation, initial/final conversion, smoke/reset/shutdown | Ordered traversal, process creation, queues, generic publication, or HTTP serialization |
| `difflet/serving/engines/resident_worker.py` | Process/IPC lifecycle, bounded admission, request records, timeout/abort, recovery, readiness, heartbeat, shutdown, and worker-side generic ordered stage traversal through the selected executor | Model-specific payload interpretation, artifact selection, or P0 stage scheduling/placement |
| `difflet/serving/openai/` | Public request parsing, validation/error mapping, artifact upload, and OpenAI-compatible response shape | Model loading or worker lifecycle |

P0 touches the following hierarchy. `common/` remains the shared CLI/serving layer;
it is not folded into `serving/`:

```text
difflet/
  registry.py                                      [existing, unchanged]
  common/
    registry/{base,qwen_image,flux}.py             [modified]
    orchestrators/{qwen_image,flux}.py             [modified]
  cli/
    main.py                                        [modified serve routing]
    serve.py                                       [serve command implementation]
    runner.py                                      [existing, unchanged]
    stage.py                                       [existing, unchanged]
    orchestrators/qwen_image.py                    [existing CLI behavior unchanged]
  pipeline/
    compile_cache.py                               [modified]
    difflet_pipeline.py                            [modified]
  models/
    qwen_image/{application,pipeline}.py           [modified]
    flux/{application,pipeline}.py                 [modified]
  serving/
    types.py                                       [modified]
    errors.py                                      [modified]
    options.py                                     [modified]
    model_registry.py                              [modified]
    factory.py                                     [modified]
    artifact_manager.py                            [new]
    artifact_store.py                              [modified]
    orchestrators/{base,qwen_image,flux}.py        [modified]
    engines/{__init__,resident_worker}.py          [modified]
    openai/{__init__,api_server,serving_chat}.py                [modified]
```

The base registry remains `difflet/registry.py`; do not replace it with a package.
Common and serving registries wrap `resolve_model(...)` and apply only their own
metadata/policy. CLI migration may be incremental, but new shared model logic must
not be duplicated in serving adapters.

### Pipeline, runtime, and artifact contracts

Serving uses immutable definitions for model structure and a separately resolved
runtime plan for one deployment profile. The staged CLI remains unchanged and is
used only for topology/artifact equivalence tests in this increment.

```python
StageRole = Literal["prompt_encoder", "denoiser", "decoder", "pipeline"]
StageKind = Literal["extracted", "opaque_pipeline"]


@dataclass(frozen=True)
class PipelineDefinition:
    model_type: str
    stages: tuple[StageDefinition, ...]


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    kind: StageKind
    role: StageRole
    output_keys: tuple[str, ...]
    final_output: bool
    runner_factory: str | None


@dataclass(frozen=True)
class RuntimePlan:
    mode: Literal["cli_staged", "resident"]
    profile_identity: str
    environment: RuntimeEnvironment
    allocations: tuple[WorkerAllocationSpec, ...]
    stages: tuple[StageRuntimeSpec, ...]


@dataclass(frozen=True)
class RuntimeEnvironment:
    available_core_ids: tuple[int, ...]
    num_cores_override: int | None
    virtual_core_size_override: int | None
    logical_nc_config_override: int | None
    inherited_distributed: DistributedProcessEnvironment
    child_distributed: DistributedProcessEnvironment


@dataclass(frozen=True)
class DistributedProcessEnvironment:
    world_size: int
    local_world_size: int
    rank: int
    local_rank: int


@dataclass(frozen=True)
class WorkerAllocationSpec:
    allocation_id: str
    requested_num_cores: int
    effective_num_cores: int
    world_size: int
    requested_virtual_core_size: int | None
    effective_virtual_core_size: int | None
    requested_logical_nc_config: int | None
    effective_logical_nc_config: int | None


@dataclass(frozen=True)
class StageRuntimeSpec:
    stage_id: str
    allocation_id: str
    topology: ParallelTopology
    artifact_id: str


@dataclass(frozen=True)
class ParallelTopology:
    tp_degree: int
    cp_degree: int
    world_size: int


@dataclass(frozen=True)
class CompileArtifactIdentity:
    schema_version: int
    canonical_cache_inputs_json: bytes
    digest: str


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


@dataclass(frozen=True)
class DiffletCompileSpec:
    artifact_id: str
    component_id: str
    identity: CompileArtifactIdentity
    required: bool = True
```

`PipelineDefinition.stages` tuple order is the sole v1 execution-order authority;
there is no duplicate dependency graph. An `extracted` stage requires non-null
`runner_factory` and is executed through a `StageRunner`. An `opaque_pipeline`
stage requires `runner_factory=None`, meaning common metadata does not provide an
extracted-stage runner factory. The selected serving adapter constructs its opaque
runtime runner, but the generic `StagePipelineEngine` and selected executor still
own traversal and execution.
`StageDefinition` does not resolve resources or artifacts: `RuntimePlan` is the
resource authority, while adapter compile specs plus the common artifact manager
are the artifact authority. Every stage kind still has exactly one
`StageRuntimeSpec` that binds its worker allocation and artifact. Flux therefore
has one opaque external stage/runtime spec while its internal CLIP/T5/DiT/VAE
topology remains owned by
`MultiComponentApplication.components()`.

For an extracted stage, `StageRuntimeSpec.topology` is the runner's homogeneous
TP/CP/world topology. For an opaque stage, it is only the normalized external
application/backbone profile used for allocation and artifact identity; it does not
claim that every internal component has that TP. The Flux adapter's frozen config,
compile spec, and component admission retain and validate the heterogeneous
component topology.

Runtime-plan validation requires:

1. Stage IDs/order/kinds match the pipeline definition; `runner_factory` nullability
   matches each kind; and every stage resolves one allocation and artifact ID.
2. Stage world fits its effective allocation and visible physical cores. Effective
   virtual-core/logical-NC settings match the compiled artifact.
3. P0 child launcher environment is explicitly OS-process world/local-world 1 and
   rank/local-rank 0 before model-owned NxD setup. This differs from the NxD
   application world recorded in `WorkerAllocationSpec`. Before model/Neuron
   imports, the child also applies the single resident allocation as exact
   `NEURON_RT_VISIBLE_CORES`, matching `NEURON_RT_NUM_CORES`, and effective
   virtual-core/logical-NC settings; it clears inherited optional settings when the
   plan leaves them unset.
4. Resident admission counts each concurrently live allocation once. Sequential
   CLI peak is the maximum allocation, not the sum.
5. Artifact identity includes pinned source ID, component/stage, topology, fixed
   shape/dtype, compile features, logical-core mode, and toolchain ABI. Allocation
   labels are not compile inputs.
6. No generic HBM estimator is required in P0. Unsupported profiles fail adapter
   admission, and readiness requires a bounded real load/smoke.

`CompileArtifactIdentity` extends existing `compile_cache.CacheSpec` canonical
inputs rather than creating a second cache. `DiffletCompileSpec` is serving-only;
the staged CLI does not consume it. It contains no selected path. The selected
model adapter exhaustively validates `component_id` and payload contents, while the
common artifact manager alone chooses identity/staging/final paths.

Qwen topology is profile-relative. Let `t` be model TP, `c` CP, and `w=t*c` for the
CLI generate stage. Resident P0 requires `c=1`, so shared worker world `w_r=t`.

| Stage | Staged CLI | Resident serving |
|---|---|---|
| `text` | TP=`t`, W=`t`, `t*c`-core subprocess allocation | TP=`t`, W=`w_r`, shared `w_r` cores |
| `generate` | TP=`t`, CP=`c`, W=`t*c`, `t*c` cores | TP=`t`, CP=1, W=`w_r`, shared allocation |
| `vae` | TP=1, W=1, 1 core | TP=`w_r`, W=`w_r`, shared allocation |

The measured Trn2 profile resolves resident TP4/W4 for all three applications.
This is a one-process compatibility profile, not a mathematical VAE requirement.
CLI stages remain sequential and keep their existing TP/world/artifact behavior.

Qwen serving compile children use a strict compile-only transport:

```python
@dataclass(frozen=True)
class StageCompileInvocation:
    model_id: str
    requested_revision: str | None
    pinned_model_path: str
    resolved_source_id: str
    stage_id: str
    runtime: StageRuntimeSpec
    publish_target: ArtifactPublishTarget
    compile_options: CompileOptions


@dataclass(frozen=True)
class QwenTextCompileOptions:
    schema_version: Literal[1]
    stage_id: Literal["text"]
    dtype: Literal["bfloat16"]
    batch_size: Literal[1]
    encoder_seq_len: int
    capture_modules: tuple[str, ...]


@dataclass(frozen=True)
class QwenGenerateCompileOptions:
    schema_version: Literal[1]
    stage_id: Literal["generate"]
    dtype: Literal["bfloat16"]
    height: int
    width: int
    num_frames: None
    text_seq_len: int
    cp_mode: str
    teacache_fused: bool


@dataclass(frozen=True)
class QwenVaeCompileOptions:
    schema_version: Literal[1]
    stage_id: Literal["vae"]
    dtype: Literal["bfloat16"]
    height: int
    width: int
    num_frames: Literal[1]


CompileOptions = (
    QwenTextCompileOptions
    | QwenGenerateCompileOptions
    | QwenVaeCompileOptions
)
```

The parent constructs it from the pinned source, path-free compile spec, and
parent-owned publish target. The child never resolves source/revision, selects a
cache path, or publishes. Qwen compile options are a versioned stage-discriminated
union that rejects unknown/missing fields, source/stage/runtime mismatch, and the
wrong option discriminant. It contains fixed shape/dtype/sequence/CP facts.
TeaCache contributes only
probe inclusion (`teacache_fused` in the current Qwen application and normalized
`teacache_probe_enabled` in artifact identity); calibration/controller values are
runtime-only.

Serving stage execution is engine-owned, sequential, process-local, and typed at
the model boundary. Generic engine code treats payload values as opaque:

```python
class StagePayload(ABC):
    """Nominal marker for logical values passed between adjacent stages."""

    __slots__ = ()


InputPayloadT = TypeVar("InputPayloadT", bound=StagePayload)
OutputPayloadT = TypeVar("OutputPayloadT", bound=StagePayload)


class StageExecutionContext(Protocol):
    @property
    def request_id(self) -> str: ...
    @property
    def deadline_monotonic(self) -> float: ...
    def throw_if_aborted(self) -> None: ...
    def report_progress(self, completed: int, total: int) -> None: ...


@dataclass(frozen=True)
class StageLoadContext:
    runtime: "ResolvedRuntimeBundle"
    stage: StageRuntimeSpec
    artifact: ArtifactBinding


@dataclass(frozen=True, slots=True)
class QwenInitialPayload(StagePayload):
    pass


@dataclass(frozen=True, slots=True)
class QwenTextPayload(StagePayload):
    encoder_hidden_states: torch.Tensor
    encoder_hidden_states_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class QwenLatentPayload(StagePayload):
    packed_latents: torch.Tensor


@dataclass(frozen=True, slots=True)
class QwenFinalPayload(StagePayload):
    output: DiffletGenerateOutput


@dataclass(frozen=True, slots=True)
class FluxInitialPayload(StagePayload):
    pass


@dataclass(frozen=True, slots=True)
class FluxFinalPayload(StagePayload):
    output: DiffletGenerateOutput


QwenStagePayload = (
    QwenInitialPayload | QwenTextPayload | QwenLatentPayload | QwenFinalPayload
)


@dataclass(frozen=True, slots=True)
class StageInvocation(Generic[InputPayloadT]):
    request: DiffletGenerateRequest
    stage: StageDefinition
    input: InputPayloadT
    context: StageExecutionContext


@dataclass(frozen=True)
class StageExecutionMetadata:
    started_monotonic: float
    finished_monotonic: float


@dataclass(frozen=True, slots=True)
class StageExecutionResult(Generic[OutputPayloadT]):
    output: OutputPayloadT
    metadata: StageExecutionMetadata


class StageCompiler(Protocol):
    def compile(self, invocation: StageCompileInvocation) -> None: ...


class StageRunner(Protocol[InputPayloadT, OutputPayloadT]):
    async def execute(
        self, invocation: StageInvocation[InputPayloadT]
    ) -> StageExecutionResult[OutputPayloadT]: ...
    async def shutdown(self) -> None: ...


class ErasedStageRunner(Protocol):
    @property
    def input_type(self) -> type[StagePayload]: ...
    @property
    def output_type(self) -> type[StagePayload]: ...

    async def execute(
        self, invocation: StageInvocation[StagePayload]
    ) -> StageExecutionResult[StagePayload]: ...
    async def shutdown(self) -> None: ...


class StageExecutor(Protocol):
    async def execute(
        self, invocation: StageInvocation[StagePayload]
    ) -> StageExecutionResult[StagePayload]: ...
    async def shutdown(self) -> None: ...
```

Concrete runners retain exact adjacent-stage typing even though the heterogeneous
executor registry is erased:

```python
class QwenTextStageRunner(StageRunner[QwenInitialPayload, QwenTextPayload]): ...
class QwenGenerateStageRunner(StageRunner[QwenTextPayload, QwenLatentPayload]): ...
class QwenVaeStageRunner(StageRunner[QwenLatentPayload, QwenFinalPayload]): ...
class FluxPipelineRunner(StageRunner[FluxInitialPayload, FluxFinalPayload]): ...
```

The heterogeneous registry contains `ErasedStageRunner` wrappers rather than
erasing generic parameters directly on `StageRunner`. A
`ValidatedStageRunner[InputPayloadT, OutputPayloadT]`
wraps each concrete runner, checks its declared exact input type before model work,
checks its exact output type afterward, and exposes only the nominal
`StagePayload -> StagePayload` interface. Input/output type metadata is stored in
private fields and exposed through read-only properties. Validation retains
`isinstance` for generic narrowing and additionally requires
`type(value) is expected_type`, so subclasses are rejected. `Any` is not part of
the engine or registry contract.
The wrapper implements async `shutdown()` by delegating exactly once to its inner
runner, preserving executor ownership of concrete runner cleanup.

Qwen extracts `text`, `generate`, and `vae` serving runners. The generic
worker-side engine owns ordered traversal, cancellation checkpoints, stage logs,
and executor invocation. The Qwen adapter owns runner construction, concrete
payload validation, initial payload construction, final output conversion,
request-state reset, smoke inputs/validation, and runner loading. Loaded runner
ownership transfers to the executor, whose shutdown is the sole runner-shutdown
path. The parent engine additionally
owns processes, queues, request records, timeout/abort, recovery, heartbeat, and
shutdown. Flux remains one opaque `pipeline` runner in this increment.

`StageLoadContext` is built only by the Qwen serving adapter. Its stage is the exact
entry from `runtime.runtime_plan`, `stage.artifact_id` equals
`artifact.artifact_id`, and `artifact` is the sole matching binding in
`runtime.artifacts`. Runners load only that binding. In P0, tensor payload objects
are process-local and never serialized through CLI files or worker IPC. The text
runner uses the request prompt, so its initial payload is empty; all inter-stage
payloads are typed above. A stage output is the next stage input without an
engine-owned conversion.

The generic stage engine uses the pipeline definition directly rather than another
hard-coded stage list:

```python
async def generate(request, context):
    payload = adapter.initial_payload(request)
    stages = runtime.pipeline_definition.stages
    for index, stage_def in enumerate(stages):
        context.throw_if_aborted()
        invocation = StageInvocation(request, stage_def, payload, context)
        result = await executor.execute(invocation)
        context.throw_if_aborted()
        if stage_def.final_output:
            require(index == len(stages) - 1)
            return adapter.finalize(result.output)
        payload = result.output
    raise InvalidPipelineDefinition("missing final output stage")
```

P0 constructs `InProcessStageExecutor(runners)` once during worker load and calls
each runner directly in list order. There is no P0 stage queue, DAG scheduler,
parallel stage execution, core placement decision, or inter-stage transport. A
future executor may encode local payloads as private IPC or shared-memory wire
handles and hydrate them before runner invocation. Handles never appear in the
logical `StageInvocation`/`StageExecutionResult` API. Future implementations must
preserve context separation and the invariant that the logical output of one stage
is the logical input of its successor.

Bundle construction and worker load verify that `pipeline_definition.model_type`
matches the selected model, its exact stage IDs/order match `runtime_plan.stages`,
and it has exactly one final stage in the last position. The Qwen loop additionally
requires every stage kind to be `extracted`, emits stage start/terminal activity,
and validates concrete payload combinations inside the selected runner/adapter. The
pre-stage check prevents new work after abort. The post-stage check catches abort
that arrived while an uninterruptible Neuron graph was running and prevents the
next stage or final response. Long-running Python denoising loops additionally call
`throw_if_aborted()` before every timestep; abort does not preempt a graph already
in flight. Any abort/error exits traversal and follows the engine-owned terminal
reset contract. Flux exposes one external `pipeline` stage: its pipeline runner checks
before/after the call and uses the existing step callback for internal iteration.

### `difflet/serving/orchestrators`

Serving stage adapters bridge generic engine contracts to common model builders.

P0 splits cold startup work from worker runtime work:

- Parent-side artifact preparation completes before the worker starts.
- Parent-side request validators check model-specific request limits before
  worker admission.
- Worker-owned stage adapters load runtime handles and supply smoke inputs/output
  validation; `StagePipelineEngine` runs smoke and generation through the same
  ordered traversal.

The parent FastAPI process resolves the artifact-preparer/stage-adapter factories,
runs download/compile/artifact checks through `prepare_runtime()`, receives one
immutable `ResolvedRuntimeBundle`, then starts the resident worker and sends
`LOAD_RUNTIME_BUNDLE`. The worker constructs the stage adapter and stage engine,
loads only the pinned model source/artifacts from that bundle, and runs smoke.

The parent-side artifact preparer owns:

- model path resolution and optional download.
- compile plan construction.
- compiled artifact checks and optional compile.
- startup progress logs for cold operations.

The lifecycle order is common across models, but the implementation of each
method is model-specific. For example, Flux can build a `CacheSpec` and validate
pipeline cache manifests, while Qwen-Image must enumerate staged encoder,
denoiser, and VAE artifacts. Keep these operations in the parent-side
`ServingArtifactPreparer`; do not duplicate download, compile, or artifact
selection inside the worker stage adapter. Worker manifest checking is
a final integrity assertion against the already selected bundle, not a second
resolution/compile policy.

```python
@dataclass(frozen=True)
class ResolvedModelSource:
    source_kind: Literal["hf_snapshot"]
    model_id: str
    requested_revision: str | None
    pinned_model_path: str
    resolved_source_id: str


TeaCacheDisabledReason = Literal[
    "not_requested",
    "incomplete_request",
    "unreadable_calibration",
    "malformed_calibration",
    "unsupported_calibration",
    "profile_mismatch",
]

TeaCacheRequestFallbackReason = Literal["step_mismatch"]


@dataclass(frozen=True)
class TeaCacheRequestMode:
    use_teacache: bool
    fallback_reason: TeaCacheRequestFallbackReason | None

    def __post_init__(self) -> None:
        if self.use_teacache and self.fallback_reason is not None:
            raise ValueError("enabled TeaCache request cannot have a fallback reason")


@dataclass(frozen=True)
class QwenTeaCacheCalibration:
    num_steps: int
    poly_coef: tuple[float, ...]
    threshold: float
    warmup_steps: int
    cooldown_steps: int
    target_speedup: float | None
    mod_input_source: str
    skip_run_length: int
    accumulate: bool
    cadence: int
    online_delta_alpha: float


@dataclass(frozen=True)
class QwenServingRuntimeConfig:
    schema_version: Literal[1]
    model_type: Literal["qwen_image"]
    model_id: str
    profile_identity: str
    tp_degree: int
    cp_degree: int
    world_size: int
    height: int
    width: int
    teacache_enabled: bool
    teacache_disabled_reason: TeaCacheDisabledReason | None
    teacache_speedup: float | None
    teacache_calibration: QwenTeaCacheCalibration | None
    requires_teacache_probe: bool


@dataclass(frozen=True)
class FluxTeaCacheCalibration:
    num_steps: int
    poly_coef: tuple[float, ...]
    threshold: float
    warmup_steps: int
    cooldown_steps: int
    target_speedup: float | None
    mod_input_source: str
    skip_run_length: int
    accumulate: bool
    cadence: int
    online_delta_alpha: float


@dataclass(frozen=True)
class FluxServingRuntimeConfig:
    schema_version: Literal[1]
    model_type: Literal["flux"]
    model_id: str
    profile_identity: str
    tp_degree: int
    cp_degree: int
    world_size: int
    height: int
    width: int
    teacache_enabled: bool
    teacache_disabled_reason: TeaCacheDisabledReason | None
    teacache_speedup: float | None
    teacache_calibration: FluxTeaCacheCalibration | None
    requires_teacache_probe: bool


AdapterRuntimeConfig = QwenServingRuntimeConfig | FluxServingRuntimeConfig


@dataclass(frozen=True)
class ResolvedRuntimeBundle:
    profile: ServingProfile
    source: ResolvedModelSource
    pipeline_definition: PipelineDefinition
    runtime_plan: RuntimePlan
    compile_specs: tuple[DiffletCompileSpec, ...]
    artifacts: ArtifactSet
    adapter_config: AdapterRuntimeConfig


class ServingModelAdapter(Protocol):
    model_id: str
    model_type: str

    def resolve_adapter_config(
        self,
        *,
        options: ServeOptions,
        source: ResolvedModelSource,
        profile: ServingProfile,
    ) -> AdapterRuntimeConfig: ...

    def build_compile_plan(
        self,
        *,
        source: ResolvedModelSource,
        profile: ServingProfile,
        config: AdapterRuntimeConfig,
    ) -> tuple[DiffletCompileSpec, ...]: ...

    def compile(
        self,
        *,
        source: ResolvedModelSource,
        profile: ServingProfile,
        config: AdapterRuntimeConfig,
        spec: DiffletCompileSpec,
        target: ArtifactPublishTarget,
    ) -> None: ...

    def validate_compiled_artifact(
        self,
        spec: DiffletCompileSpec,
        artifact_root: Path,
    ) -> None: ...


class ServingArtifactPreparer(Protocol):
    def prepare_runtime(
        self,
        options: ServeOptions,
        profile: ServingProfile,
        download_policy: DownloadPolicy,
        compile_policy: CompilePolicy,
    ) -> ResolvedRuntimeBundle: ...
```

`prepare_runtime(...)` is the only public artifact-preparation lifecycle operation.
The stack selects the model adapter and passes the existing immutable `ServeOptions`
through unchanged. Generic code may transport those raw options but must not pair,
reject, or interpret `teacache_*` values. The preparer pins the model source, asks
the adapter to resolve one frozen config, and passes that exact source/profile/config
instance to `build_compile_plan(...)`. The common artifact manager performs cache
lookup, locking, staging, manifest writing, and immutable publication; on a miss it
calls the adapter's `compile(...)`. Qwen uses `StageCompileInvocation`; Flux invokes
its existing opaque pipeline compiler. The compile hook receives the same pinned
source, core profile, and frozen config used to build its spec; it cannot re-resolve
options or model source. Serving worker stage adapters do not compile. The resulting
bundle is serializable and contains no credentials or calibration path.

This extends the existing `difflet.serving.types.DiffletCompileSpec`; it does not
introduce a second serving compile-plan type. Migration removes its old mutable
`artifact_path`; only the common artifact manager chooses a cache path and creates
`ArtifactPublishTarget`. The already selected model adapter dispatches an exhaustive
component validator. `CompileArtifactIdentity.schema_version` versions the payload
contract, so no string validator registry is needed.

`ResolvedRuntimeBundle.compile_specs` stores the exact immutable specs returned by
the adapter and used by the common manager for lookup/compile/publication. It is the
sole compile-plan authority after preflight. The parent does not discard it after
binding, and workers do not derive a replacement from profile/config.

`ResolvedModelSource` is the sole model-source authority, `PipelineDefinition` is
the frozen stage-definition/order authority, and `ArtifactSet` is the sole
artifact-binding authority. `RuntimePlan` stages contain only artifact IDs.
`AdapterRuntimeConfig` is a frozen, serializable Qwen-or-Flux-specific type created
and consumed by that adapter. The separate Qwen and Flux calibration value objects
are intentionally not a generic calibration schema even though their P0 fields
currently overlap. The generic bundle/engine treats their payload as opaque except
for checking the discriminant, model ID, and profile identity. The complete profile
identity hashes the normalized generic model/source/topology/shape fields plus the
canonical frozen runtime values that can change output; the disabled reason and
calibration path are excluded.
Bundle construction validates before serialization, and worker load validates
again:

- `profile.model_id`, `source.model_id`, and plan profile identity agree;
- `source_kind` is `hf_snapshot`, `resolved_source_id` is a commit SHA, and the
  pinned path is that commit-addressed snapshot; P0 resident serving rejects every
  caller-provided local model directory before bundle creation;
- artifact IDs are unique and every stage artifact reference resolves exactly once;
- compile-spec artifact IDs are unique; every binding matches exactly one frozen
  spec by `artifact_id`, and binding/spec identities are byte-for-byte equal;
- every required frozen spec has exactly one binding, and every runtime stage
  artifact ID resolves through that same spec/binding pair;
- every artifact manifest's canonical source input equals
  `source.resolved_source_id`;
- every binding points directly to one immutable published generation, and its
  generation ID/content digest match that generation's manifest;
- generic profile inputs agree on model/source, TP, CP, world, width, and height;
- the selected adapter verifies its config repeats those resolved values and that
  `profile_identity` recomputes from the generic profile plus its frozen runtime
  values.

Artifact validation follows one possible lifecycle order:

- **Reuse:** common generation-manifest/inventory validation, then adapter
  payload-manifest validation, then create the binding.
- **Cache miss:** validate source/config/spec/invocation inputs before compile;
  after compile, validate produced payload and adapter manifest, then inventory,
  write the generation manifest, and publish/bind.
- **Initial/replacement load:** retrieve the exact frozen spec from the bundle,
  repeat common generation/inventory validation, then adapter payload-manifest
  validation before loading model applications. Workers never call
  `build_compile_plan()`.

At each adapter validation point, canonical compile inputs including
`teacache_probe_enabled` must match the compile spec and frozen config.

`ResolvedModelSource` pins model version, not a second byte-level source inventory.
The Hugging Face cache is a trusted output of the download layer. Download/preflight
resolves a branch/tag to one commit SHA; the serving lifecycle never checks whether
that revision later moved and never auto-downloads or switches source. Updating a
model requires a new download/preflight plus service restart. Missing local snapshot
files fail startup/load. Protection against operator mutation, symlink retargeting,
or external cache GC is outside P0 serving and belongs to download/storage
operations.

TeaCache is enabled only by an explicit `--teacache-speedup`. Without that flag,
serving does not read a calibration pathname even if one was supplied. When the
flag is present, a calibration pathname is required and startup reads it exactly
once, validates its schema/model/shape and requested target, then stores the frozen
`TeaCacheCalibration` value in the resolved runtime profile. Missing, unreadable,
malformed, mismatched, or insufficient calibration fails startup with
`invalid_extra_body`; it does not silently enable TeaCache or defer failure to a
worker. Initial and replacement workers consume the frozen value without reopening
the source path. Existing CLI pathname behavior is unchanged.

Qwen and Flux P0 reject cadence and online-delta serving modes. The frozen
calibration and speedup are runtime controller inputs; the compile identity records
whether the optional TeaCache probe is required, not the source pathname. Logs must
not copy calibration file contents or paths.

Enabled configs require a finite positive speedup and a non-null frozen calibration
value. Disabled configs carry neither effective speedup nor calibration. Each
worker builds fresh process-local controller state from the frozen value. Adapter compile specs add only
`teacache_probe_enabled: bool` to the existing canonical compile inputs. Speedup,
thresholds, coefficients, and calibration metadata are runtime-only and do not
split the main/probe NEFF identity. P0 keeps the probe as an optional component of
the generate/pipeline artifact, so baseline and probe-enabled profiles may use two
disk generations but one worker never loads two copies of the main model. Splitting
the probe into an independently published artifact is a later optimization.

The resolved profile is authoritative for model, width, and height. To enable
adaptive TeaCache, startup requires:

- positive integer `num_steps` and a nonempty JSON array of finite numeric
  coefficients;
- finite nonnegative numeric threshold plus nonnegative integer warmup/cooldown
  values with
  `warmup + cooldown < num_steps`;
- integer `skip_run_length >= 1` and a finite positive requested speedup;
- a null or finite positive calibration target speedup that is not lower than the
  requested speedup;
- a strict boolean `accumulate` and a `mod_input_source` accepted by that model's
  adapter;
- P0 adaptive mode values integer `cadence == 0` and finite numeric
  `online_delta_alpha == 0`.

Missing fields, invalid values, unsupported mode combinations, or profile mismatch
fail startup before artifact preparation and worker spawn.

Qwen and Flux application construction add one serving-compatible input while
preserving the existing CLI pathname API:

```python
teacache_calibration: TeaCacheCalibration | None = None
teacache_calibration_path: str | None = None
```

The inputs are mutually exclusive. Existing CLI callers continue passing only the
pathname. In a worker, the selected serving adapter reconstructs a process-local
frozen `TeaCacheCalibration` from its typed runtime config and passes the object;
the Flux application creates a fresh mutable `TeaCacheController` from it, while
the Qwen application forwards it to `QwenImageOrchestrator`, which does the same.
The object input is added through both Qwen construction layers. Neither the
controller nor the source pathname crosses the process boundary. Initial and
replacement workers therefore produce the same controller after the original
calibration file is changed or deleted.

Artifact preparation uses this deterministic per-identity layout:

```text
<cache_root>/serving/<model_type>/<identity_digest>/
  .publish.lock
  staging/<uuid>.tmp/
  generations/g0000000000000001/
```

The identity root, `staging`, and `generations` must be real directories beneath
the configured cache root on one filesystem. The manager rejects symlinks and
realpath escape. It acquires `.publish.lock` with a bounded timeout and performs
lookup, cleanup, generation allocation, compile, validation, and publication while
holding that lock. It rescans `generations/` under the lock and never uses a mutable
`latest` index. Valid generation names are fixed-width monotonic integers. A new
generation uses `max(existing IDs) + 1`; a collision increments and retries.

Candidate reuse is deterministic: validate finalized generations from newest ID to
oldest and bind the first valid one. `NEVER` binds that candidate or fails. `AUTO`
binds it or compiles a new generation. `FORCE` always compiles a new generation.
Concurrent `AUTO`/`FORCE` calls serialize on the same lock, so a later `AUTO` sees a
generation published by an earlier call. Invalid finalized generations are skipped
without moving or deleting them; finalized cleanup is offline only.

Compilation/import occurs only inside the manager-created unique `staging_path`.
The adapter may write only below that realpath and cannot choose its destination.
Abandoned staging directories from a crash may be removed or quarantined under the
same lock. The common publisher reserves
`difflet_generation_manifest.json` as the generation manifest; adapters may not use
that name. Existing adapter manifests such as Flux `manifest.json` remain ordinary
payload files, are included in the canonical inventory/content digest, and are
validated by the selected adapter. The inventory recursively hashes all regular
payload files by normalized relative path, size, and SHA-256 while excluding only
`difflet_generation_manifest.json`; other file types are rejected. Before rename,
the adapter validates its payload/manifest, then the publisher computes inventory,
writes and fsyncs the generation manifest, and fsyncs payload files, nested
directories, and the staging root. It atomically renames staging to the allocated
generation path on the same filesystem, then fsyncs `generations/` and the identity
root. A crash before rename leaves only removable staging. A crash after rename
leaves a finalized candidate that the next scan validates and either reuses or
skips. Runtime bundles pin that exact path and digest. Parent reuse and every worker
pre-load recompute the payload digest and compare it with the binding and generation
manifest, then invoke adapter validation; stored digest equality alone is not
trusted. `FORCE` never overwrites a path an existing worker may still load.

Request validators are also model-specific, but they run in the parent/FastAPI
process:

```python
class ServingRequestValidator(Protocol):
    def validate(self, request: DiffletGenerateRequest) -> None: ...


class ServingRequestValidatorFactory(Protocol):
    def __call__(
        self,
        runtime: ResolvedRuntimeBundle,
    ) -> ServingRequestValidator: ...
```

The protocol lives in `difflet/serving/orchestrators/base.py`. Concrete
validators live next to the serving adapter, for example
`FluxServingRequestValidator` in `difflet/serving/orchestrators/flux.py` and
`QwenImageServingRequestValidator` in
`difflet/serving/orchestrators/qwen_image.py`.

Use this hook for checks that must happen before worker admission, such as
prompt tokenizer/bucket limits. It must not load Trainium runtime objects or
hold Neuron cores.

The factory receives the already frozen bundle after `prepare_runtime()`. Concrete
validators store that bundle and load tokenizers/config only from
`runtime.source.pinned_model_path`; they never call `resolve_model_path()` or use
the requested mutable revision. Request validation uses `runtime.profile`, so the
validator cannot be constructed against a different source/profile generation.

Adaptive TeaCache calibration is valid only for its frozen `num_steps`. A request
with a different `num_inference_steps` is not rejected: the worker executes that
request with TeaCache disabled and emits an allowlisted request-level fallback
reason. Matching requests use the resident controller. This decision is made by the
model adapter/denoising runner before its loop and does not mutate the frozen
bundle or artifact selection.

The model adapter creates one `TeaCacheRequestMode` per call. Qwen passes it to the
generate runner/pipeline and Flux passes it to the pipeline call. Each denoising
loop snapshots local references before its first step:

```python
controller = self.teacache_controller if request_mode.use_teacache else None
probe = self.teacache_probe if request_mode.use_teacache else None
```

The loop uses only those local references. A baseline request never invokes the
probe and never clears/rebinds the resident controller. The mode is request-local,
is not stored on the pipeline, and cannot affect the next matching request.
Defensive pre-request reset and every terminal-clean reset clear controller state
regardless of whether that request used TeaCache.

Worker-owned stage adapters own:

- active `ServingProfile`
- runner construction and pipeline/stage handles only until successful ownership
  transfer, including exhaustive partial-failure cleanup
- smoke request construction and output validation
- adapter-specific state that was never transferred into a loaded runner
- future serving-only profile switching/recovery behavior

On successful `create_loaded_runners()` return, runner-held pipeline/stage handles
transfer atomically and exclusively to the executor lifecycle. The adapter must
drop its ownership references; its later shutdown must not close, reset, or
otherwise mutate transferred handles. Executor shutdown reaches those handles only
through each `ValidatedStageRunner -> inner runner` delegation, exactly once.

Worker stage adapters consume the already resolved model path/profile/artifact
identity prepared during startup. They may compute child paths below
`runtime.source.pinned_model_path`, but must not resolve the requested revision again, download
weights, or run AOT compile. Before loading Trainium handles they re-read each
bound manifest and require exact equality with the bundle identity. A missing,
changed, or stale manifest fails load/smoke and readiness; the worker never repairs
or recompiles it.

The model adapter exposes lifecycle and runner construction. The generic
worker-side stage engine exposes the generation method called by worker IPC:

```python
class ServingStageAdapter(Protocol):
    model_id: str
    model_type: str
    active_profile: ServingProfile

    async def create_loaded_runners(
        self, runtime: ResolvedRuntimeBundle
    ) -> Mapping[str, ErasedStageRunner]: ...
    def initial_payload(self, request: DiffletGenerateRequest) -> StagePayload: ...
    def finalize(self, payload: StagePayload) -> DiffletGenerateOutput: ...
    def smoke_request(self) -> DiffletGenerateRequest: ...
    def validate_smoke_output(self, output: DiffletGenerateOutput) -> None: ...
    def reset_request_state(self, outcome: str) -> None: ...
    async def shutdown(self) -> None: ...
```

Stage traversal is internal to the generic `StagePipelineEngine`, not the model
adapter. The engine passes opaque payloads between adjacent runners and delegates
only initial/final conversion and concrete payload validation to the adapter.
The adapter transfers loaded runners to the selected executor. The executor is the
only owner that shuts those runners down; adapter shutdown occurs afterward and
clears only adapter-owned state.

`create_loaded_runners()` transfers ownership atomically only after it returns the
complete validated mapping. If construction or load fails partway, the adapter must
attempt `await runner.shutdown()` in reverse creation order for every runner it
already created. An individual cleanup failure is retained but does not stop
cleanup of earlier runners; after all attempts, the adapter raises an aggregate
error. No partial mapping is published and no executor is constructed from it.

The returned mapping has a mandatory ordering contract: insertion order is runner
creation order and exactly matches `PipelineDefinition.stages`. The executor must
preserve that order and shut runners down in strict reverse insertion order. An
unordered mapping implementation is invalid even if it contains the correct stage
IDs.

Executor shutdown follows the same exhaustive rule: it attempts every runner in
strict reverse insertion order, retains individual failures, and reports an
aggregate error only after all runners have been attempted. Worker shutdown always
attempts async adapter-state cleanup after executor cleanup, even when executor
cleanup failed. Failures from both phases are aggregated and make the worker
process-unsafe.

This cleanup contract applies to every exit after adapter construction, not only
orderly shutdown after a request. Adapter load, partial runner construction,
readiness smoke, startup cancellation, replacement, and request-time retirement all
enter one idempotent cleanup path. That path awaits executor cleanup only when
runner ownership transferred and an executor exists, then always awaits adapter
cleanup. Startup never publishes readiness until this ownership state is
unambiguous.

`active_profile` is assigned from `runtime.profile`; model/tokenizer paths are
assigned from `runtime.source.pinned_model_path`. The same bundle object is serialized to
the initial child and every replacement child. Recovery never reruns preflight or
model resolution, so a mutable branch cannot move between initial and replacement
load. Parent request-validator construction also consumes the pinned bundle rather
than independently resolving the requested revision.

`WorkerRequestContext` is created by the worker runtime, not by the HTTP layer.
It carries the request deadline and one cancellation signal:

```python
@dataclass
class WorkerRequestContext:
    request_id: str
    deadline_monotonic: float
    abort_event: threading.Event
```

The request-specific thread-safe event is the only cancellation hook model code
should see. The worker's control listener validates the active request ID under a
lock before setting it. It is checked at safe points; it is not a model-specific
control channel and cannot preempt an in-flight Neuron graph.

`WorkerRequestContext` implements the common `StageExecutionContext` contract used
by extracted Qwen runners: `throw_if_aborted()` checks `abort_event`, and
`report_progress()` emits best-effort non-blocking status. The stage engine
passes the same context into every serving runner, so the runner API does not depend
on serving globals, thread-locals, or model-specific cancellation side channels.

For Qwen:

```text
generate(request, context)
  -> for stage in pipeline_definition.stages
       -> context.throw_if_aborted()
       -> in_process_executor.execute(invocation)
       -> context.throw_if_aborted()
       -> return final output when the last stage is final
       -> otherwise pass result.output unchanged to the next stage
     (`generate` runner also checks before each Python timestep)
  -> DiffletGenerateOutput(bytes, mime="image/png")
```

For Flux:

```text
StagePipelineEngine.generate(request, context)
  -> generic loop over the one-entry PipelineDefinition
  -> InProcessStageExecutor.execute(invocation)
  -> FluxPipelineRunner.execute(invocation)
       -> context.throw_if_aborted()
       -> pipe(..., callback_on_step_end=check cancellation)
       -> context.throw_if_aborted()
       -> StageExecutionResult(FluxFinalPayload(...))
  -> adapter.finalize(payload)
  -> DiffletGenerateOutput(bytes, mime="image/png")
```

The parent serving engine should send `RUN_REQUEST` to the worker during request
execution. The worker runtime should create `WorkerRequestContext`, own the
cancellation signal, and call only `stage_engine.generate(request, context)`.
Caller timeout/disconnect sends request-keyed `ABORT_REQUEST`; the engine-owned
request record remains active until one worker terminal or process isolation. A
clean abort clears request-local tensors while keeping loaded models in HBM.
Before each request and before every terminal-clean READY transition, the worker
calls the model-owned `reset_request_state(outcome)` hook. This resets scheduler,
TeaCache, callbacks, interrupt flags, and request-derived tensors without unloading
weights or NEFFs. A model/profile that cannot prove a complete reset is marked
process-unsafe and replaced instead of reused.

On success, `generate` must first materialize the response as parent-owned immutable
bytes with no references to model/request buffers. Only then may the worker call
`reset_request_state("completed")` and emit `REQUEST_COMPLETED`. Abort/error paths
discard partial output before reset.

Between output materialization and reset, the worker freezes `completed` by moving
the request to `finalizing` under its request-state lock. An abort received after
that transition is stale: the request remains successful, reset continues with the
completed outcome, and a successful reset returns the same worker to READY without
reloading its HBM-resident models.

### `difflet/serving/engines`

Serving engines own service runtime behavior, not model internals.

Detailed engine behavior is defined in
[Difflet Serving Engine](./engine.md).

They own:

- queueing
- `max_running_requests`
- request timeout
- worker health
- worker IPC
- cancellation
- shutdown
- readiness state

They should not contain Qwen-specific stage names like `text`, `generate`, or
`vae`.

### `difflet/serving/openai`

The OpenAI layer owns HTTP protocol behavior:

- request/response Pydantic models.
- `/v1/chat/completions`.
- prompt extraction.
- `extra_body` parsing and validation.
- model/profile mismatch errors.
- conversion of `DiffletGenerateOutput` into chat content parts.
- deployment-owned selection between inline data URLs and `ArtifactStore`
  upload/URL generation.

It should not know the model stage count.

## Startup Flow

```text
parse serve args
resolve base ModelEntry through difflet.registry
resolve serving metadata through difflet.serving.model_registry
build one ServingProfile from registry defaults plus serve-flag overrides
select artifact-preparer and serving stage-adapter factories
artifact preparer: resolve/download model weights
artifact preparer: build compile plan
common artifact manager: check or compile/publish artifacts
artifact preparer: freeze ResolvedRuntimeBundle with pinned source and bindings
construct parent-side request validator from the pinned bundle
create engine with the opaque runtime bundle
create FastAPI app with lifespan startup
lifespan startup: start resident worker
construct serving stage adapter, loaded runners, executor, and stage engine inside worker
verify bound manifests and load pinned pipeline/stage apps inside worker
run serving smoke through worker
bind HTTP
serve ready traffic
```

Download, compile, load, and smoke must emit progress logs before and after
long-running steps so startup does not appear stuck.
In ASGI deployments, the app object can be constructed before the worker is
started; the important invariant is that lifespan startup finishes preflight,
worker load, and smoke before readiness becomes true and before serving ready
traffic.

Serving must not hard-code one global parallel default. The `serve` subcommand
accepts the complete set of existing CLI options that can affect compilation,
loading, or the resident runtime profile: parallelism, fixed shape, cache and
force-compile behavior, host/device decode selection, and TeaCache mode. The
serving parser preserves omission state with `None` defaults where registry
defaults must remain distinguishable from explicit operator overrides. This
includes compile/runtime booleans such as `cfg_parallel` and `sp_enabled`; an
absent flag must not be normalized to `False` before model defaults are resolved.
Their serve-parser form uses a tri-state destination (`bool | None`), with explicit
enable/disable spellings where a model default may be true.

Registry defaults are the only generic model-profile default authority. Resolution
applies explicit operator values over the selected `ModelEntry` and produces a
`ServingProfile` containing model/source selection, topology, shape, dtype, and
output modality. Artifact preparers and resident stage adapters consume those core
values without secondary `or 4` or `or 1024` fallbacks. Raw model-specific startup
values, including `teacache_*`, remain in `ServeOptions` until the selected adapter
resolves its frozen config. Operational defaults such as host, port, queue limits,
and timeouts also remain `ServeOptions` concerns and do not enter artifact identity.

The current Flux registry default is TP8 while the measured Trn2 deployment used
an explicit TP4 profile. Omitting `--tp-degree` therefore means TP8, not "reuse the
last compiled profile". Admission must reject insufficient cores or artifact
mismatch before worker load. Startup logging records the non-secret resolved model,
topology, shape, dtype, and feature switches so the effective default is visible.

Accepting a CLI-compatible startup option does not imply that every serving model
supports every value. Model semantics are adapter-owned: the adapter must thread a
supported value into its frozen config and compile spec, explicitly reject an
unsupported requested mode, or apply a documented optional-feature fallback.
Fallback is not silent ignore: the frozen config records disabled state and the
startup log emits an allowlisted reason. Generic profile construction must not
reject a model-specific option before the selected adapter sees it.

Per-generation CLI options remain request fields rather than process-startup
fields. `--prompt`, `--steps`, `--guidance-scale`, `--seed`, and output naming
map to `/v1/chat/completions` messages and `extra_body`; `--output`,
`--work-dir`, and `--keep-work-dir` are CLI file-workflow concerns replaced by
the serving `ArtifactStore` and in-memory stage handoff.

`--num-frames` is reserved for future video adapters. For Qwen/Flux P0 image
serving, a non-null startup `num_frames` override must be rejected during
startup profile validation rather than baked into `ServingProfile`.

Current startup capability matrix:

| Startup option | Flux | Qwen-Image |
| --- | --- | --- |
| `--tp-degree`, `--cp-degree`, `--cp-mode`, `--height`, `--width` | Profile override | Profile override; Qwen still requires CP=1 |
| `--sp` | Supported and included in compile identity | Rejected by the same model capability rule as CLI |
| `--cfg-parallel` | Parsed, then rejected because the exposed Flux path is guidance-distilled and has no true-CFG request contract | Rejected |
| `--teacache-speedup` + `--teacache-calibration` | Resolved into a fixed resident Flux adapter config | Resolved into a fixed resident denoiser adapter config |
| `--teacache-cadence`, `--teacache-online-delta` | Transported in `ServeOptions`, then rejected by the Flux P0 adapter | Transported in `ServeOptions`, then rejected by the Qwen P0 adapter |
| `--num-frames`, `--host-vae` | Rejected for the current image adapter | Rejected for the current image adapter |
| `--cache-dir`, `--force`, `--revision` | Supported | Supported |

For adaptive TeaCache, "supported" means the adapter resolves effective enablement.
Calibration metadata that the selected adapter cannot use produces an allowlisted
disabled reason and baseline inference rather than a generic startup failure.

Serving also has an operational-only `--worker-heartbeat-interval` option
(positive seconds, default `30`). It does not affect artifact identity or model
output. The worker starts heartbeat reporting before profile load so compile/load
and smoke delays remain observable.

The option is an engine lifecycle setting, not part of `ServingProfile`. Its
required propagation path is:

```text
_add_serve_flags argparse option
  -> validate_serve_args(value > 0)
  -> options_from_args
  -> ServeOptions.worker_heartbeat_interval
  -> build_serving_stack
  -> ResidentWorkerConfig.worker_heartbeat_interval
  -> _ResidentWorkerProcess / _worker_main
```

Every layer above requires a focused propagation test. No layer may silently use
the default after an operator supplied an explicit value.

P0 uses one model-level serving profile that can be overridden at startup with
`difflet serve --tp-degree`, `--cp-degree`, `--height`, `--width`, and related
flags. Stage-specific differences belong in serving stage metadata and
placement/core calculations. For Qwen, let `t` be the resolved model TP. P0
requires `cp_degree=1`, so the shared worker world is `w_r=t`; text and denoiser
use TP=`t`/W=`w_r`, while the resident decoder compatibility artifact uses
TP=`w_r`/W=`w_r`. The measured Trn2 TP4 profile therefore resolves all three to
TP4/W4, but TP4 is not a fixed serving constant. The staged CLI retains a
separate TP1/W1 decoder artifact because each CLI stage runs in its own process.
Do not expose separate per-stage TP/CP flags in the first serving milestone; add
an explicit stage-profile override only if a future model requires genuinely
different stage parallel configs.

The complete propagation path is
`serve parser -> ServeOptions -> selected model adapter -> frozen adapter config +
core ServingProfile -> artifact preflight -> resident construction`. Generic
`validate_serve_args` may enforce parsing types and universal operational bounds,
but it does not pair or interpret model-specific fields. Tests must prove that every
parsed compile/load/runtime option reaches the selected adapter and is implemented,
explicitly rejected there, or represented by a documented disabled fallback; silent
ignore is not compatible behavior. Stage/runtime execution consumes the frozen
bundle and does not maintain another model-option allowlist.

Lifecycle output uses structured Python `logging`, not ad hoc `print()` calls.
The authoritative startup/stage event and worker-heartbeat contract is in
`engine.md`; stage identity, runner, and compile contracts are defined earlier in
this document.

## Request Flow

```text
FastAPI /v1/chat/completions
  -> parse OpenAI envelope
  -> validate prompt, extra_body, profile, and modality
  -> engine.generate(request)
  -> worker IPC RUN_REQUEST
  -> worker runtime creates WorkerRequestContext
  -> worker-owned StagePipelineEngine.generate(request, context)
  -> sequential InProcessStageExecutor calls model-owned runners
  -> image bytes
  -> if S3 configured: await ArtifactStore.put_bytes(...) -> get_url(ref)
  -> otherwise: encode data:image/...;base64,...
  -> chat response with image_url.url
```

## MVP Model Mapping

| Model | Base registry | Serving stage adapter | Engine | Notes |
| --- | --- | --- | --- | --- |
| Qwen-Image | `qwen_image` | `serving/orchestrators/qwen_image.py` | `ResidentWorkerServingEngine` | Three `extracted` stages `text -> generate -> vae`; roles/display labels `prompt_encoder -> denoiser -> decoder`. |
| Flux | `flux` | `serving/orchestrators/flux.py` | `ResidentWorkerServingEngine` | One `opaque_pipeline` stage and worker-owned loaded `DiffletPipeline`. |

Each server process loads exactly one model/profile.

For Flux, let `t` and `c` be resolved backbone TP/CP and let the existing model
configuration derive application world `w`. Internal components intentionally use
different TP degrees but one application world:

```text
pipeline(CLIP TP=1/W=w, T5 TP=w/W=w, transformer TP=t/W=w, VAE TP=1/W=w)
```

The measured Trn2 TP4/CP1 profile produced CLIP TP1/W4, T5 TP4/W4, transformer
TP4/W4, and VAE TP1/W4. This is evidence, not a hard-coded default. Before compile
or load, the Flux adapter constructs with `load=False`, inspects actual component
configs/load priority, rejects unsupported DP/CFG/world/allocation combinations,
validates every required component including an optional TeaCache probe, and emits
sanitized component topology diagnostics. The generic `RuntimePlan` continues to
expose only external stage `pipeline`.


## Adding A Model

Use existing layers; do not add a second registry, artifact resolver, or request
schema.

### Required Integration

1. Base registry (`difflet/registry.py`): resolve model ID/type and local/download
   path without serving policy.
2. Common metadata (`difflet/common/registry`): defaults, capability flags,
   compile-affecting fields, and stable stage IDs.
3. Common orchestrator (`difflet/common/orchestrators`): provide shared model
   constants/builders and payload validation helpers without selecting serving
   cache paths.
4. Serving artifact manager/adapter: pin source, build path-free specs, compile into
   manager-owned staging, validate payloads, publish immutable generations, and
   freeze the runtime bundle.
5. Serving registry (`difflet/serving/model_registry.py`): map model type to
   artifact preparer, stage adapter, validator, and supported profile.
6. Serving stage adapter (`difflet/serving/orchestrators`): implement
   async `create_loaded_runners(runtime)`, `initial_payload(request)`,
   `finalize(payload)`, `smoke_request()`, `validate_smoke_output(output)`,
   `reset_request_state(outcome)`, and async adapter-state `shutdown()`. The generic
   `StagePipelineEngine` owns `generate()` and ordered traversal.
7. OpenAI adapter: add request/output mapping only when the existing contract does
   not already cover the modality.
8. Tests: profile/default resolution, artifact identity/integrity, initial and
   replacement load, smoke/readiness, repeated requests, cancellation/reset, and
   public request/error behavior.

### Pipeline Models

Pipeline models retain their model-owned component graph. Serving injects pinned
source and artifact paths without changing CLI defaults:

```python
DiffletPipeline.precompile(
    model_id=runtime.source.model_id,
    model_path_override=runtime.source.pinned_model_path,
    resolved_source_id=runtime.source.resolved_source_id,
    compiled_path_override=publish_target.staging_path,
)
DiffletPipeline.from_pretrained(
    model_id=runtime.source.model_id,
    model_path_override=runtime.source.pinned_model_path,
    resolved_source_id=runtime.source.resolved_source_id,
    compiled_path_override=artifact_binding.path,
    skip_compile=True,
)
```

The common pipeline API accepts exactly two modes:

| Mode | Required values |
|---|---|
| Legacy CLI | logical `model_id`; no override fields |
| Bound serving | logical `model_id` plus `model_path_override`, `resolved_source_id`, and `compiled_path_override` |

Every partial override tuple is rejected. Bound compile requires an unpublished
parent staging path. Bound load requires a finalized binding plus
`skip_compile=True`. Source override bypasses model/revision resolution while
preserving logical identity; artifact override bypasses cache selection. Worker
load validates but never mutates the finalized generation.
TeaCache and other optional model-specific values come only from the frozen
`runtime.adapter_config`; generic pipeline loading does not interpret them.

### Staged Models

Register one `PipelineDefinition`, one profile-specific `RuntimePlan`, typed
`StageCompileInvocation` transport, process-local serving runners, and one
model-aware `ServingStageAdapter`. Runner execution accepts
`StageExecutionContext`; the generic `StagePipelineEngine` performs traversal via
`InProcessStageExecutor`. Current CLI execution remains unchanged.

### Fail-Closed Checklist

- Every parsed startup option is implemented, rejected by the selected adapter, or
  represented by its documented disabled fallback; none is silently ignored.
- Parent and worker agree on pinned source, profile, plan, artifact IDs, manifests,
  content digests, effective cores/LNC, and world/rank environment.
- Resident model sources are commit-addressed HF snapshots; caller-provided local
  directories are rejected in P0.
- Initial and replacement workers consume the same `ResolvedRuntimeBundle`.
- READY follows bounded real end-to-end smoke, never metadata-only checks.
- Request reset is complete before worker reuse; unsafe state triggers replacement.
- Logs/status are sanitized and cannot block execution or consume terminal replies.
- Public request, response, error, and CLI behavior remain compatible unless the
  model addition explicitly extends the documented API.
