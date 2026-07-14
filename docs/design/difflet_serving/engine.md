# Difflet Serving Engine

This document defines the serving engine boundary for the MVP implementation.

The engine is the serving runtime layer. It owns request execution safety,
worker lifecycle, and generic ordered stage traversal. It does not own
model-specific stage implementations or interpret model-specific payloads.

This file, together with `architecture.md` and
`chat_completions_contract.md`, is the authoritative P0 serving design. The
older `docs/plans/2026-07-06-difflet-serving-engine.md` keeps broader future
design notes, including rotating, subprocess, and multi-plan ideas, but those
are not part of the P0 implementation scope unless repeated here.

## MVP Scope

MVP engines support:

- Qwen-Image through one resident Trainium worker process.
- Flux through the same resident Trainium worker-process pattern.
- One active model/profile per server process.
- `max_running_requests=1`.
- Bounded queueing.
- Request timeout.
- Worker health and readiness.
- Structured startup, stage lifecycle, and request progress logs.
- Periodic worker activity heartbeat from process start through shutdown.
- Graceful shutdown.
- Request-keyed cooperative abort with HBM-resident model reuse.
- Generic `PipelineDefinition.stages` traversal inside the resident worker.
- Sequential, process-local stage execution through `InProcessStageExecutor`.

MVP engines do not implement:

- multi-profile scheduling.
- co-batching.
- dynamic stage placement.
- rotating resident scheduling.
- subprocess fallback.
- cross-host distributed execution.
- generic stage DAG optimization.
- per-stage worker processes or inter-stage IPC/shared-memory transport.

Those are future architecture hooks, not P0 behavior.

### Serving profile lifecycle (P0)

- P0 resolves and validates one immutable `ResolvedRuntimeBundle` at startup and
  starts exactly one resident worker for that bundle.
- The pinned source, artifact bindings, runtime plan, adapter config, and active
  profile inside the bundle are fixed for the engine lifetime, including
  replacement workers.
- Additional profiles may be precompiled and stored on disk, but they are not loaded
  into the active worker unless the process restarts.
- Request-time `extra_body` fields are not a profile switch mechanism; startup
  profile fields (`tp_degree`, `cp_degree`, `cp_mode`, `cfg_parallel`,
  `sp_enabled`) are rejected by request validation.
- To avoid duplicate worker copies while changing profile, a deployment must either
  run another pod/instance with the new profile (for zero-downtime transition) or
  restart the current process and accept a startup/smoke cold window.

## Runtime Boundary

```text
FastAPI route
  -> DiffletServingEngine.generate(request)
  -> queue / timeout / health / worker lifecycle
  -> worker IPC
  -> worker runtime creates WorkerRequestContext
  -> worker-owned StagePipelineEngine.generate(request, context)
  -> for stage in PipelineDefinition.stages
  -> InProcessStageExecutor.execute(stage invocation)
  -> model-owned StageRunner inside the same worker process
```

The parent-side resident engine and worker-side stage engine must not hard-code
model-specific stage names such as Qwen `text/generate/vae`. Together they see
only:

- an opaque `ResolvedRuntimeBundle` carrying one `ServingProfile` and pinned load inputs
- a model adapter factory that supplies initial/final payload conversion and
  stage runners
- `DiffletGenerateRequest`
- `DiffletGenerateOutput`
- opaque `StagePayload` values passed unchanged between adjacent stages

## Engine Protocol

```python
class DiffletServingEngine(Protocol):
    runtime: ResolvedRuntimeBundle

    async def start(self) -> None: ...
    async def generate(self, request: DiffletGenerateRequest) -> DiffletGenerateOutput: ...
    async def shutdown(self) -> None: ...

    @property
    def profile(self) -> ServingProfile: ...  # always runtime.profile

    @property
    def ready(self) -> bool: ...

    @property
    def healthy(self) -> bool: ...
```

The engine owns:

- queue admission
- `max_running_requests`
- `max_queued_requests`
- `queue_timeout`
- `request_timeout`
- draining state
- worker process lifecycle
- worker health state
- readiness state
- cancellation and timeout cleanup
- request-level logs/metrics
- ordered traversal of `PipelineDefinition.stages`
- cancellation/deadline checkpoints before and after each stage
- generic stage lifecycle logs
- selection of the P0 `InProcessStageExecutor`

The engine does not own:

- model id detection
- tokenizer/template logic
- compiled path naming
- model download
- AOT compile implementation
- Qwen payload interpretation or stage-to-stage type branching
- Flux pipeline invocation details
- stage placement, per-stage core scheduling, IPC, or shared-memory policy in P0
- R2 upload
- OpenAI response formatting

## Generic Stage Execution Contract

`PipelineDefinition.stages` tuple order is the sole P0 execution-order authority.
The worker-side engine executes that list sequentially; P0 has no dependency graph,
ready queue, placement algorithm, or stage overlap.

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


class ServingStageAdapter(Protocol):
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

The engine treats `StagePayload` as an opaque nominal logical value. Every concrete
model payload must inherit it; arbitrary Python objects are invalid stage values.
Concrete adjacent runners own payload schema and validation. Payloads contain model data only;
request/deadline/cancellation state stays in `StageExecutionContext`. This
separation permits a future executor to encode a logical value as a private wire
handle and hydrate it before invoking a runner. `StageInvocation.input` and
`StageExecutionResult.output` always expose logical payloads; IPC/shared-memory
handles never escape the executor/transport implementation.

`ValidatedStageRunner` is the only generic-erasure adapter. It exposes a nominal
base-type interface while retaining the concrete types needed by the inner runner:

```python
class ValidatedStageRunner(Generic[InputPayloadT, OutputPayloadT]):
    def __init__(
        self,
        inner: StageRunner[InputPayloadT, OutputPayloadT],
        input_type: type[InputPayloadT],
        output_type: type[OutputPayloadT],
    ) -> None:
        self.inner = inner
        self._input_type = input_type
        self._output_type = output_type

    @property
    def input_type(self) -> type[StagePayload]:
        return self._input_type

    @property
    def output_type(self) -> type[StagePayload]:
        return self._output_type

    async def execute(
        self, invocation: StageInvocation[StagePayload]
    ) -> StageExecutionResult[StagePayload]:
        if (
            not isinstance(invocation.input, self._input_type)
            or type(invocation.input) is not self._input_type
        ):
            raise StagePayloadTypeError(...)
        typed_invocation = StageInvocation(
            request=invocation.request,
            stage=invocation.stage,
            input=invocation.input,
            context=invocation.context,
        )
        result = await self.inner.execute(typed_invocation)
        if (
            not isinstance(result.output, self._output_type)
            or type(result.output) is not self._output_type
        ):
            raise StagePayloadTypeError(...)
        return StageExecutionResult(output=result.output, metadata=result.metadata)

    async def shutdown(self) -> None:
        await self.inner.shutdown()
```

P0 provides only:

```python
class InProcessStageExecutor:
    def __init__(self, runners: Mapping[str, ErasedStageRunner]) -> None:
        # Required: insertion order is runner creation/pipeline order.
        self.runners = runners

    async def execute(
        self, invocation: StageInvocation[StagePayload]
    ) -> StageExecutionResult[StagePayload]:
        if not isinstance(invocation.input, StagePayload):
            raise StagePayloadTypeError(...)
        result = await self.runners[invocation.stage.stage_id].execute(invocation)
        if not isinstance(result.output, StagePayload):
            raise StagePayloadTypeError(...)
        return result

    async def shutdown(self):
        errors = []
        for runner in reversed(tuple(self.runners.values())):
            try:
                await runner.shutdown()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise StageShutdownError(errors)
```

The execution backend is selected once when the worker is built. P0 accepts only
`stage_executor="in_process"`; the main loop must not contain scattered
`single_process` conditionals. A future worker-pool executor may introduce IPC,
shared memory, host buffers, ownership, and cleanup behind the same executor
boundary, but those mechanisms are explicitly deferred.

The heterogeneous runner mapping uses `ErasedStageRunner`, not `Any`. Each concrete
runner still declares exact generic input/output payloads; its validated wrapper
checks the exact input before model work and the exact output afterward. The
executor additionally fail-closes on the nominal base type around dispatch.
Adapter `finalize()` must validate the model's exact final payload type before
constructing `DiffletGenerateOutput`; it uses the same combined `isinstance` plus
`type(value) is ExpectedFinalPayload` rule and rejects subclasses.
`ValidatedStageRunner.shutdown()` delegates exactly once to its inner runner, so
the executor's sole-owner shutdown path reaches every concrete runner.

The adapter constructs and loads runners, then transfers their runtime ownership
to the executor. `StageExecutor.shutdown()` is the only runner-shutdown path. It
must attempt every runner in strict reverse order even after an individual failure,
then raise an aggregate shutdown error after all attempts. The adapter's own
`shutdown()` clears only adapter-owned state and must not close the runners again.
Compile-time stage operations use a separate compile contract and are not methods
on the loaded runtime `StageRunner`.

Ownership transfer is atomic on successful `create_loaded_runners()` return. The
adapter must attempt `await runner.shutdown()` in reverse creation order for every
runner created during a partial construction/load failure, retaining failures and
continuing through all earlier runners before re-raising an aggregate error; a
partially populated mapping is never handed to the executor.

The returned mapping must preserve insertion order, with insertion order equal to
runner creation order and `PipelineDefinition.stages`. `InProcessStageExecutor`
preserves that order and its shutdown strictly reverses it. Returning an unordered
mapping is a contract violation.

Readiness smoke uses the same generic stage loop: the adapter supplies a bounded,
deterministic request and validates the final model output, while
`StagePipelineEngine` performs traversal and executor calls. The adapter must not
own a second private traversal path for smoke.

## P0 Engine Type

### `ResidentWorkerServingEngine`

Used by Qwen-Image and Flux.

```text
ResidentWorkerServingEngine
  -> owns queue/timeout/health/draining
  -> starts one Trainium worker process for the resolved runtime bundle
  -> sends LOAD_RUNTIME_BUNDLE / SMOKE / RUN_REQUEST over IPC
  -> receives final image bytes plus metadata
```

The worker process owns one `StagePipelineEngine`, one model adapter, and the
process-local runners selected by that adapter. The same generic loop executes a
single opaque stage such as Flux or multiple extracted stages such as Qwen.

Initial startup uses a provisional-worker lease. Under the transition lock,
`start()` records a new worker instance, startup epoch, queues, and one idempotent
cleanup future, but does not publish readiness. It starts the status consumer
before waiting for `LOAD_RUNTIME_BUNDLE -> SMOKE -> READY`. Publishing `READY`
requires the same instance/epoch, an open engine, and successful load/smoke.

Any exception, malformed/error reply, timeout, task cancellation, or shutdown
before that publication closes the provisional lease before `start()` returns:
terminate the child if alive, join it with bounded escalation, stop the status
consumer/heartbeat when present, drain and close all per-spawn queues, and clear
the provisional instance. Startup and shutdown share the cleanup future, so a race
cannot double-close queues or leave process ownership ambiguous. Shutdown advances
the engine epoch and may perform the cleanup; a cancelled startup then awaits that
same cleanup instead of publishing readiness. No queued request record can be
dispatched to a provisional worker.

Flux parent preflight:

```text
FluxServingArtifactPreparer
  prepare_runtime(options, profile, policies)
    pin model source and resolve frozen Flux adapter config
    adapter.build_compile_plan(source, profile, config)
      inspect MultiComponentApplication component metadata with load=False
      verify component TP/world/load order and worker allocation
      return DiffletCompileSpec without an output/cache path
    common artifact manager
      select identity, lock, and create ArtifactPublishTarget
      call adapter.compile(source, profile, config, spec, target) on cache miss
      validate payload, write manifest, publish, and bind immutable generation
    return ResolvedRuntimeBundle
```

The branch-local `ensure_artifacts()` and `compile_plan(profile)` public contracts
are removed in the runtime-bundle migration. Engine startup receives only the
finished bundle; it does not call either legacy helper or participate in compile
planning/publication.

Flux compile/load passes `model_id`, `model_path_override`, and
`resolved_source_id` from the same bundle, so it bypasses mutable model/revision
resolution while preserving logical model identity. Worker load also passes
`compiled_path_override=binding.path` with `skip_compile=True`; this bypasses
`cache_path(...)` and mutable index/identity-root selection. Worker validation and
`app.load()` consume exactly the same bound generation path. A worker never repairs
or recompiles that path; corruption fails startup and parent preflight must publish
a different generation for a future bundle. Existing Flux CLI callers omit the
overrides and retain current behavior.

Flux worker layout:

```text
FluxServingStageAdapter.create_loaded_runners(runtime)
  spec = runtime.require_compile_spec(flux_artifact_id)
  binding = runtime.artifacts.require(flux_artifact_id)
  verify spec/binding identity and Flux payload manifests
  application_kwargs = flux_adapter.build_application_kwargs(
      runtime.adapter_config
  )
  # kwargs contain a reconstructed TeaCacheCalibration object, never its path
  pipe = DiffletPipeline.from_pretrained(
      model_id=runtime.source.model_id,
      model_path_override=runtime.source.pinned_model_path,
      resolved_source_id=runtime.source.resolved_source_id,
      compiled_path_override=binding.path,
      application_kwargs=application_kwargs,
      skip_compile=True,
  )
  return insertion_ordered({"pipeline": FluxPipelineRunner(pipe)})

FluxPipelineRunner.execute(invocation)
    context = invocation.context
    context.throw_if_aborted()
    output = pipe(...)
    context.throw_if_aborted()
    return StageExecutionResult(FluxFinalPayload(output), metadata)
```

Flux smoke supplies a deterministic request through the adapter and runs this same
generic ordered `StagePipelineEngine` loop over the one-entry pipeline definition.
There is no Flux-specific stage-engine branch and no adapter-owned `generate()`
entry point.

Flux P0 smoke is a bounded deterministic end-to-end inference using the active
profile shape, a fixed prompt/seed, and a startup timeout. Baseline profiles use a
low step count. When adaptive TeaCache is enabled, smoke uses the frozen
calibration `num_steps` so readiness exercises the actual controller/probe path.
It must validate a non-empty image with the expected dimensions before readiness.
The added startup cost is accepted because object-only readiness cannot prove that
the heterogeneous component topology executes.

Qwen worker layout:

```text
StagePipelineEngine
  active_runtime
  adapter = QwenServingStageAdapter
  executor = InProcessStageExecutor(runners)

  load(runtime)
    runners = await adapter.create_loaded_runners(runtime)
    executor = InProcessStageExecutor(runners)
    retrieve exact frozen spec/binding pair for each Qwen stage
    verify each identity and Qwen payload manifest without rebuilding plan
    reconstruct optional TeaCacheCalibration from runtime.adapter_config
    bind the object through the serving-only constructor input, never its path
    load all stage apps into the worker process

  generate(request, context)
    value = adapter.initial_payload(request)
    for stage_def in active_runtime.pipeline_definition.stages
      context.throw_if_aborted()
      invocation = StageInvocation(request, stage_def, value, context)
      result = await executor.execute(invocation)
      context.throw_if_aborted()
      if stage_def.final_output
        require stage_def is last
        return adapter.finalize(result.output)
      value = result.output
    fail missing final output stage
```

The engine passes every non-final `StageExecutionResult.output` unchanged as the
next `StageInvocation.input`. Adjacent Qwen runners therefore share one logical
payload contract: `text` returns `QwenTextPayload`, `generate` consumes it and
returns `QwenLatentPayload`, and `vae` consumes that payload and returns
`QwenFinalPayload`. The generic engine contains no `isinstance` routing over Qwen
types.

P0 execution is deliberately sequential and process-local. `InProcessStageExecutor`
calls the selected runner directly and the payload is an in-memory Python object,
so no tensor is serialized or copied merely to cross a stage boundary. The
`StageExecutor` and payload-carrier contracts are the future replacement seams for
worker-pool scheduling and IPC/shared-memory transport; P0 does not implement or
select those transports.

Qwen is enabled only after this shared-worker layout passes startup load and
smoke for the selected `ServingProfile`. The smoke must prove that all Qwen
stage handles co-load in the single resident worker process and can execute a
minimal request without per-stage subprocesses, rotating load/unload, or file
handoff. If this gate fails, startup fails and `/ready` never becomes healthy.
As with Flux, an enabled adaptive TeaCache profile uses calibration `num_steps` in
smoke; only baseline smoke may use a reduced step count.

The parent FastAPI process should not load Trainium model objects directly.
Keeping Flux and Qwen behind the same worker-process boundary keeps
process-level Neuron core/env ownership clean and avoids one-off runtime paths.

## Request Admission

P0 uses serialized model execution:

```text
max_running_requests = 1
max_queued_requests = 8
queue_timeout = configurable, default 30s
request_timeout = configurable, default 300s
artifact_store_timeout = configurable, default 60s
worker_cancel_timeout = configurable, default 10s
worker_restart_timeout = configurable, default 900s
worker_heartbeat_interval = configurable, default 30s
```

`worker_heartbeat_interval` must be a finite positive number. It is parsed by
`difflet serve`, stored in `ServeOptions`, copied into `ResidentWorkerConfig`, and
passed to the child process. It is operational metadata and is excluded from
`ServingProfile` and compile-cache identity.

Expected behavior:

| Condition | Response |
| --- | --- |
| queue full | `429 queue_full` |
| queue wait exceeds timeout | `429 queue_timeout` |
| request exceeds timeout | `504 request_timeout` |
| worker recovering after timeout | `503 engine_recovering` |
| worker dead | `503 engine_unavailable` |
| server draining | `503 engine_draining` |

`request_timeout` is an external wall-clock deadline. It starts when the HTTP
handler hands the normalized request to the engine and includes queue wait plus
worker execution. The engine should stamp `received_at_monotonic` and compute
`deadline_monotonic` in its admission path; model adapters should not own this
timer.

When R2 is configured, artifact upload and URL generation happen after the engine
returns bytes to the OpenAI handler, so they are bounded by
`artifact_store_timeout`, not by the engine `request_timeout`. With no required
R2 variables, the handler returns a Base64 data URL and does not use an artifact
backend. A partial R2 configuration returns `artifact_store_unavailable` during
startup. Unexpected SDK/backend failures retain full details only in protected
logs and return fixed `internal_error`; a hung R2 upload or presign must not hold
the HTTP request indefinitely.

## Generate Flow

P0 uses engine-owned request records rather than caller-owned tickets. The engine
owns every record from admission through terminal cleanup; an HTTP coroutine only
waits on the record's result future and may detach at timeout or disconnect.

The parent engine dispatches one `RUN_REQUEST` and does not iterate model stages or
interpret stage IDs. The worker creates one `WorkerRequestContext` and calls
`stage_engine.generate(request, context)`. The generic worker-side engine iterates
the frozen `PipelineDefinition`, reports the current stage through the
context/status queue, and checks abort immediately before and after every runner.
Qwen therefore executes three extracted stages; Flux executes its one opaque
`pipeline` runner through the same loop. Long Python loops add inner safe-point
checks. An in-flight Neuron graph is not preempted. Abort/error returns through the
engine's normal terminal reset and recovery rules.

```python
@dataclass
class RequestRecord:
    request_id: str
    request: DiffletGenerateRequest
    worker_instance_id: str | None
    received_at_monotonic: float
    queue_deadline_monotonic: float
    deadline_monotonic: float
    state: Literal["queued", "running", "aborting", "terminal"]
    delivery_state: Literal["pending", "delivered", "detached", "shutdown_error"]
    result_future: asyncio.Future[DiffletGenerateOutput]
```

The parent transition lock owns `closed`, `draining`, `recovering`, the bounded
queue, the single active record, current worker identity, and `engine_epoch`.
Admission creates and queues a record atomically after checking state and capacity.
It sets `queue_deadline_monotonic = received + queue_timeout` and
`deadline_monotonic = received + request_timeout`. P0 dispatches at most one record
because `max_running_requests=1`; the scheduler dispatches `record.request` and does
not depend on a caller-owned payload.

While queued, the scheduler/timer expires a record at the earlier deadline. It
checks the overall request deadline first, so equal deadlines resolve as
`504 request_timeout`; otherwise queue deadline resolves as `429 queue_timeout`.
Either queued expiry removes the record and completes its future without sending an
abort. After dispatch only the overall request deadline applies and timeout follows
the running-request abort/recovery contract.

```python
async def generate(self, request: DiffletGenerateRequest) -> DiffletGenerateOutput:
    record = await self.submit_or_raise(request)
    remaining = record.deadline_monotonic - monotonic()
    if remaining <= 0:
        await self.abort_or_remove(record.request_id, reason="request_timeout")
        raise RequestTimeout()

    try:
        return await asyncio.wait_for(
            asyncio.shield(record.result_future),
            timeout=remaining,
        )
    except asyncio.TimeoutError:
        await self.abort_or_remove(record.request_id, reason="request_timeout")
        raise RequestTimeout()
    except asyncio.CancelledError:
        await self.abort_or_remove(record.request_id, reason="caller_cancelled")
        raise
```

There is no caller/recovery ownership transfer and no caller-side slot release.
The engine scheduler removes a queued request on queue/request timeout. For a
running request, `abort_or_remove` marks the record `aborting`, detaches the
caller, and sends one request-keyed `ABORT_REQUEST`. Repeated abort calls are
idempotent. The operation is cancellation-masked after it is scheduled: a repeated
caller cancellation cannot interrupt the engine's locked state change or queued
control send. Failure to send abort is process-unsafe. The record continues to
occupy the execution slot until the engine's
single terminal dispatcher consumes exactly one matching worker terminal.

Worker terminals are:

```text
REQUEST_COMPLETED(request_id, output, recovery_disposition)
REQUEST_ABORTED(request_id, recovery_disposition)
REQUEST_ERROR(request_id, recovery_disposition)
```

`recovery_disposition` is an independent worker-reuse result:
`terminal_clean|process_unsafe`. Normal `REQUEST_COMPLETED` and
`REQUEST_ABORTED` are terminal-clean. Completion that
races a timeout/abort is accepted as the one terminal; its output is discarded when
the caller detached, request-local state is cleared, and the same loaded worker
returns to READY. An explicitly allowlisted request error may also be
`terminal_clean`. A reset exception preserves an already frozen request outcome but
sets `process_unsafe`: completed output may still be delivered, while the worker is
replaced before another request. Unknown errors, missing/unknown dispositions, all
5xx, worker death, or no terminal within `worker_cancel_timeout` are
`process_unsafe`.

For a process-unsafe outcome the engine atomically marks the worker unavailable and
enters RECOVERING. It terminates the old process, drains its queues, starts a new
worker with a new instance ID, loads the same pinned runtime bundle, and runs smoke.
Only after old-process isolation and request-record cleanup may capacity be
released. A successful replacement becomes READY; failed replacement leaves the
engine unavailable. No new request is dispatched while recovering.

The terminal dispatcher validates `worker_instance_id` and `request_id`. Under the
same parent transition lock used by shutdown and caller detachment, it changes the
record to `terminal` and commits exactly one delivery outcome. A terminal that
commits while delivery is `pending` delivers its result and changes delivery to
`delivered`. A detached caller keeps `detached`, so the output is discarded. If
shutdown already committed `shutdown_error`, the caller receives public
`503 engine_draining` and a later worker terminal performs cleanup only and cannot
replace the caller-visible error. The dispatcher removes
request/activity/cancellation state exactly once. A duplicate or mismatched
terminal is stale and has no authority. Status/heartbeat traffic remains on a
separate queue and cannot consume terminals.

Shutdown is engine-owned and follows the sole authoritative sequence in
[Shutdown](#shutdown). Terminal delivery remains lock-linearized: terminal commit
first delivers its result; shutdown commit first delivers public
`503 engine_draining`; a later terminal is cleanup-only. A detached caller receives
nothing, and shutdown never waits for caller cooperation.

## Worker IPC

P0 worker IPC should be worker-level, not per-stage and not model-specific. The
implementation may combine lifecycle commands with process startup as long as
the externally visible state machine is the same.

Logical commands/events:

```text
START_WORKER -> LOAD_RUNTIME_BUNDLE -> SMOKE
RUN_REQUEST
ABORT_REQUEST
SHUTDOWN
```

`RUN_REQUEST`, `ABORT_REQUEST`, and `SHUTDOWN` use one FIFO control queue and one
listener. Each worker lease owns one bounded `multiprocessing.Queue` (`maxsize=8`
in P0); it is never retargeted or reused. All parent producers call `put_nowait`
only while holding the transition lock. Dispatch writes RUN before exposing the
record as running, so a later abort/shutdown cannot overtake it. Queue full, closed queue, or send error is
process-unsafe. Model execution runs on a separate generation thread, so the
listener remains responsive while Python/Neuron work is blocked. Status and
terminal replies remain on dedicated output queues.

The queue belongs to one provisional/current worker lease:

```python
@dataclass
class WorkerControlLease:
    worker_instance_id: str
    engine_epoch: int
    control_queue: multiprocessing.Queue
    closed: bool = False
```

Every command carries the lease worker ID and epoch. Under the transition lock,
send validates the current open lease and retirement marks it closed/detaches it;
therefore no producer can call `put_nowait` after retirement begins. Outside the
lock, cleanup terminates/joins the old process as needed, then closes and joins that
queue's feeder with a bounded fallback. A replacement receives a new queue and
cannot consume an old command.

At every initial or replacement child entry, before constructing the stage adapter
or stage engine or importing model/Neuron modules, the worker applies the bundle's validated,
allowlisted environment. P0 sets `WORLD_SIZE=1`, `LOCAL_WORLD_SIZE=1`, `RANK=0`,
and `LOCAL_RANK=0`; model components may update `LOCAL_WORLD_SIZE` during their own
NxD setup. The single resident allocation sets exact `NEURON_RT_VISIBLE_CORES` and
a consistent `NEURON_RT_NUM_CORES`, plus its effective
`NEURON_RT_VIRTUAL_CORE_SIZE` and `NEURON_LOGICAL_NC_CONFIG`; absent optional
values are removed rather than inherited. Invalid allocation capacity or child
distributed topology fails before model load.

Worker status events use a separate status queue:

```text
WORKER_STATE_CHANGE
STAGE_START / STAGE_PROGRESS / STAGE_END / STAGE_ERROR
WORKER_HEARTBEAT
```

Before each child spawn, the parent generates a new opaque
`worker_instance_id` and passes it to `_worker_main`. Every worker-originated
status event, startup/stage event, heartbeat, readiness message, and request
terminal reply includes that ID. PID is retained
for diagnostics but is not an identity because operating systems can reuse it.

The parent keeps one `current_worker_instance_id`. Process replacement atomically
retires the old ID before termination and before any replacement becomes current.
Consumers validate the ID before changing cached worker state or completing a
request. Events from a non-current ID are ignored for state purposes and may be
emitted only as rate-limited `stale_worker_event` debug diagnostics. In particular,
a retired worker's late `ready`, `busy`, heartbeat, `REQUEST_COMPLETED`,
`REQUEST_ABORTED`, or `REQUEST_ERROR` cannot change readiness, active stage, or
terminal request state.

The worker owns one request-local state machine while model applications remain
loaded:

```python
@dataclass
class WorkerRequestState:
    request_id: str
    state: Literal[
        "accepted", "resetting", "running", "abort_requested",
        "finalizing", "terminal",
    ]
    selected_outcome: Literal["completed", "aborted", "clean_error"] | None
    abort_event: threading.Event
```

For `RUN_REQUEST`, the listener acquires the worker request-state lock, rejects any
second active request, installs `WorkerRequestState(state="accepted")` and its
request-specific event, then submits execution to the generation thread. State is
therefore visible before defensive reset or model work. The generation thread moves
to `resetting`, performs defensive reset, checks the event immediately afterward,
and only then enters `running` and invokes the first model operation.

The same listener receives `ABORT_REQUEST(request_id)` while reset or generation is
busy. Under the lock it validates the accepted/active ID, sets the thread-safe
event, and changes the state to `abort_requested`. Because RUN and ABORT share one
parent-produced FIFO queue, an abort sent after dispatch cannot overtake request
installation. Abort during defensive reset remains recorded and is observed before
model execution. Wrong, stale, and duplicate aborts are no-ops. Terminal cleanup or
worker retirement removes the state/event before another request is accepted.

The listener never unloads applications or clears compiled model state. Qwen checks
before each stage and at the start of every Python denoising timestep. Flux checks
through its existing `callback_on_step_end` path and before/after non-denoising
phases. A single in-flight Neuron graph is not preempted; cancellation takes effect
at the next checkpoint.

P0 has one OS process and one host control flow. TP/CP logical ranks participate in
the graph launched by that host branch, so the generation thread reads the abort
event once at each checkpoint and either all logical ranks enter the next graph or
none do. It does not issue a distributed broadcast while a graph is running.
Independent OS-rank synchronization is part of the explicitly deferred
torchrun/MPMD design and must define its own out-of-band abort propagation before
being admitted.

At terminal selection, the worker uses one request-state lock so completion and
abort cannot both win:

- if abort is already observed before successful output is materialized, select
  `aborted` under the lock and enter `finalizing`;
- after normal execution materializes immutable output bytes, select `completed`
  under the lock and enter `finalizing` if abort has not already won;
- entering `finalizing` freezes the selected outcome and retires the abort event.
  Abort received during reset or terminal emission is stale, is ignored, does not
  change COMPLETED to ABORTED, and does not trigger worker replacement;
- exceptions follow the exhaustive classification below; an abort checkpoint is
  not reported as a generic request error.

After selecting the outcome, the worker releases the request-state lock, runs
`reset_request_state(selected_outcome)`, then reacquires the lock to enter
`terminal` and emits exactly that frozen terminal. It never holds the lock during
model cleanup. With a successful completion reset, the request is
`REQUEST_COMPLETED`, the worker is terminal-clean, and the same HBM-resident worker
returns to READY. Only a real reset exception or an independently process-unsafe
condition prevents reuse; a late abort alone never does.

Exception/reset classification is exhaustive and fail-closed:

| Condition | Frozen request outcome | Reset policy | Emitted terminal |
|---|---|---|---|
| abort observed or `RequestAborted` raised at a checkpoint | `aborted` | required selected-outcome reset | `REQUEST_ABORTED(terminal_clean)` |
| explicitly allowlisted safe request exception | `clean_error` | required selected-outcome reset | `REQUEST_ERROR(terminal_clean)` |
| unknown exception, 5xx-equivalent failure, Neuron/runtime corruption, or independently unsafe error | no reusable outcome | optional best-effort cleanup only; it cannot establish safety | `REQUEST_ERROR(process_unsafe)` |
| defensive pre-request reset failure | request startup error; model execution does not begin | no second reset attempt required | `REQUEST_ERROR(process_unsafe)` |
| terminal reset failure after frozen `completed`, `aborted`, or `clean_error` | preserve the frozen outcome | reset failed | same outcome terminal with `process_unsafe` |

Only the first two rows may return the same worker to READY, and only after their
required reset succeeds. Unsafe/best-effort cleanup never changes disposition to
terminal-clean. Every row emits exactly one terminal for the active request. A
terminal reset failure after frozen completion still delivers its immutable output,
but replacement must complete before another dispatch.

Every model stage adapter implements `reset_request_state(outcome)`. The worker calls
it defensively before starting a request and again before committing a terminal-
clean `REQUEST_COMPLETED`, `REQUEST_ABORTED`, or `REQUEST_ERROR`. It resets
request-local scheduler state, TeaCache counters/residuals, callback bindings,
cancellation/activity flags, intermediate text/latent/image tensors, and output
buffers. Flux retains its existing per-call TeaCache reset and clears callback and
interrupt references. Qwen resets its host `TeaCacheController`; a fused TeaCache
probe must unconditionally overwrite device state on the first step of every
request and ignore any prior-request delta until that initialization completes. A
profile that cannot prove this reset is process-unsafe after abort and cannot reuse
the worker. Text encoder, denoiser, VAE/decoder, weights, NEFFs, and their HBM
residency remain intact. Allocator caches may retain temporary buffers, but no
second model copy is loaded.

Reset runs for both request-local TeaCache modes. A step-mismatch baseline request
uses local `controller=None`/`probe=None` references without mutating resident
bindings, but defensive and terminal reset still clear resident controller state.
Matching, mismatched, matching, aborted, and failed request sequences therefore
cannot leak controller state.

For successful completion, the stage engine calls adapter finalization to
materialize a parent-owned
immutable output byte string that no longer references request/model buffers. The
worker then freezes `completed` under its request lock, resets request state, and
emits that frozen output. Abort arriving after the freeze is stale and cannot alter
the terminal or worker reuse. Abort and clean-error paths discard partial output
before reset. Reset failure preserves the frozen request outcome but reports
`recovery_disposition=process_unsafe`; the worker must not advertise READY or reuse
HBM state.

The parent has one terminal reader and routes a terminal only to the matching
engine-owned `RequestRecord` for the current worker instance. There is no terminal
route transfer between caller and recovery. A detached caller does not change
terminal ownership; it only prevents output delivery. If abort produces no terminal
within `worker_cancel_timeout`, the parent isolates and replaces the process.

Status traffic must not share the terminal reply queue used by `READY`,
`REQUEST_COMPLETED`, `REQUEST_ABORTED`, and `REQUEST_ERROR`. This prevents a
heartbeat consumer from stealing or reordering request completion messages.

The status queue is bounded (`maxsize=256` in P0), and worker emission is always
best-effort and non-blocking. `WorkerStatusEmitter` uses `put_nowait`; it never
waits for parent capacity from the generation thread, heartbeat thread, stage
callback, or startup path.

Backpressure policy:

- heartbeat and `stage_progress` events may be dropped or coalesced to the newest
  snapshot when the queue is full;
- lifecycle start/end/error events are written to the worker's local structured
  logger first, then mirrored best-effort to the parent status queue;
- every dropped/coalesced status event increments a counter included in the next
  successfully emitted heartbeat;
- status loss never changes request success, readiness, cancellation, or recovery;
- terminal request replies never use this queue and are never dropped by this
  policy.

The parent starts its status consumer before starting the worker and before
waiting for `READY`. On shutdown it signals the heartbeat thread, drains the
bounded queue, joins the status consumer, and only then closes the queue. A dead
or slow parent consumer therefore cannot block model execution. A consumer is
bound to the instance ID and queue created for that spawn; draining a retired
queue never reactivates its state.

In the P0 code path, `LOAD_RUNTIME_BUNDLE` and `SMOKE` may happen during worker process
startup before the worker reports ready. Health/readiness may be derived from
the process state plus cached load/smoke state rather than a separate `HEALTH`
IPC command.

`RUN_REQUEST` sends one `DiffletGenerateRequest` plus `deadline_monotonic` to
the worker. The worker builds `WorkerRequestContext`, runs the active
`StagePipelineEngine`, and returns a `DiffletGenerateOutput`.

`ABORT_REQUEST` uses the dedicated control queue so it can set the active
request's cancellation signal while the generation thread is busy. It does not
return a separate acknowledgement: `REQUEST_ABORTED` is the clean abort terminal,
while a completion race yields `REQUEST_COMPLETED`. If neither terminal arrives
within the cancel timeout, process replacement is the safety fallback.

Intermediate tensors and model objects must not cross the parent/worker IPC
boundary in P0.

## Structured Logging And Worker Heartbeat

All serving lifecycle code uses module-level Python loggers and stable structured
event names. Human-readable formatting may be selected by deployment, but event
fields remain machine-parseable. Do not log prompts, credentials, access tokens,
presigned URLs, or complete environment/profile objects.

Required startup events cover model resolution, artifact preflight/compile,
worker spawn, profile load, each Qwen stage load, Flux internal component
admission/load, readiness smoke, recovery, and shutdown. Required request events
cover queue admission, request start, every canonical stage transition, progress,
completion, cancellation, timeout, and error. Each startup/stage scope has a unique
`operation_id`. A scope that exits through Python has exactly one worker-owned end
or error event with monotonic duration and `terminal_source=worker`. Every
worker-originated event includes `worker_instance_id`.

The engine remains model-agnostic. The worker-side stage engine emits canonical
stage transitions and the parent-side engine owns their status-channel consumer:

```text
Qwen: text -> generate -> vae
Flux: pipeline
```

Flux adapter-owned phases such as prompt encoding, denoising, or decode may be
reported as `operation`, but they do not become engine stage IDs.

The worker maintains a lock-protected activity snapshot:

```python
@dataclass
class WorkerActivity:
    worker_state: str
    request_id: str | None
    stage_id: str | None
    operation_id: str | None
    operation: str | None
    request_started_at: float | None
    stage_started_at: float | None
```

Activity changes occur only through scoped helpers, not direct field mutation:

```python
with activity.request(request_id):
    with activity.stage("text", operation="execute"):
        text = encode(...)
```

Both scopes clear their fields in `finally`. The request scope receives the
terminal worker state selected by worker error/recovery policy, so a handled
request failure may return to `ready` while an unsafe failure transitions to
`recovering` or `error`.

Required transition table:

| Trigger | Worker state | Request | Stage/operation |
|---|---|---|---|
| child process started | `starting` | clear | startup operation or clear |
| profile load and smoke succeeded | `ready` | clear | clear |
| request scope entered | `busy` | set | clear |
| stage scope entered | `busy` | set | set |
| stage scope exited in `finally` | `busy` | set | clear |
| request success or safe handled error | `ready` | clear | clear |
| request aborted after cleanup | `ready` | clear | clear |
| timeout/5xx requires process recovery | `recovering` | clear | clear |
| shutdown begins | `draining` | clear | clear |
| startup or unrecoverable worker failure | `error` | clear | clear |

An exception in a stage first executes stage cleanup, then request cleanup, then
selects `ready`, `recovering`, or `error`. Timeout termination may kill a child
while its last heartbeat says `busy`; the parent recovery state becomes
authoritative as soon as recovery starts.

Forced termination and process crashes cannot execute the worker's cleanup or log
`finally` blocks. The parent keeps the last accepted activity snapshot and start
receipt time for each current `(worker_instance_id, operation_id)`. When it observes
process death or forces termination, it emits `worker_lost` or
`worker_terminated`, followed by `request_recovery_start`. If the snapshot names an
active startup/stage operation, the parent also emits one matching synthetic
`startup_step_error` or `stage_error` with `terminal_source=parent_synthetic`, the
known request/stage/operation fields, and parent-monotonic elapsed time. This closes
the observability span; it does not assert that worker-side cleanup completed.

The accepted-terminal set is keyed by `(worker_instance_id, operation_id)`. The
parent does not synthesize a terminal if a worker terminal is already accepted. A
worker terminal arriving after synthetic closure is logged as
`terminal_superseded` and cannot complete a caller, alter activity, or change
recovery state. When no active operation snapshot was accepted, worker loss emits
only worker-level loss and recovery events; it must not invent a stage identity.

A dedicated daemon thread starts immediately inside `_worker_main`, before factory
load, and emits `worker_heartbeat` every `worker_heartbeat_interval` seconds over
the status queue. A separate thread is required because profile load and Trainium
generation may block the worker's asyncio loop. The heartbeat includes:

```text
worker_pid, model_type, profile_identity
worker_instance_id
worker_state=starting|ready|busy|recovering|draining|error
active_requests=0|1, request_id|null
stage_id|null, operation|null
operation_id|null
request_elapsed_ms|null, stage_elapsed_ms|null
dropped_status_events
```

The parent adds `running_requests`, `queued_requests`, `pending_requests`, and
`request_capacity` from admission accounting before writing each heartbeat log.
These fields are present in both the text message and the structured
`worker_heartbeat` record. P0 serializes execution with
`max_running_requests=1`, so one active request/stage is sufficient. A future
concurrent worker must replace the single activity fields with a bounded list or
aggregate.

Heartbeats are observability-only in P0. Existing process-death checks, request
deadlines, and recovery remain the health authorities. If no heartbeat arrives
for three intervals, the parent emits `worker_heartbeat_stale` with the last known
state but does not terminate the worker solely for heartbeat loss.

`stage_progress` is emitted from a model callback only where the underlying loop
offers a safe and deterministic hook. Otherwise periodic heartbeat with stage and
elapsed time is the progress signal.

### Error sanitization

Normal logs, status IPC, terminal reply IPC, and HTTP responses use an allowlisted
error envelope:

```text
request_id, stage_id, operation
status_code, error_code, error_type
safe_message, exception_class, recovery_disposition
```

Known `DiffletServingError` instances may expose only messages constructed by
Difflet from non-sensitive fields. Unknown exceptions use a generic safe message
such as `worker execution failed`; normal paths must not serialize `str(exc)`,
`repr(exc)`, exception arguments, or `traceback.format_exc()` into IPC or logs.
Sanitization never upgrades recovery safety: a missing or unrecognized
`recovery_disposition` remains `process_unsafe`.

Raw tracebacks are permitted only when an explicit local debug setting is enabled.
They go to a protected worker-local debug sink and never to the status queue,
terminal reply, HTTP response, or artifact URL. Tests inject prompts, tokens,
credential-like environment values, local paths, and presigned URLs into synthetic
exceptions and assert that none appear in normal log/IPC/HTTP payloads.

## Health And Readiness

`/health` and `/ready` should reflect engine state:

- ready only after load plus serving smoke succeeds.
- not ready while starting, draining, or unhealthy.
- not ready while recovering a worker after request timeout.
- not ready after worker death.
- unhealthy after unrecoverable worker failure.

During `RECOVERING`, `/ready` returns 503 and new generation requests return
`503 engine_recovering`. `/health` may remain 200 while the process and recovery
task are alive; it should return 503 only after recovery fails or the engine is
marked unrecoverably unhealthy.

Worker death after startup should:

1. mark engine unhealthy.
2. flip `/ready` to 503.
3. fail new requests with `503 engine_unavailable`.
4. fail or cancel queued requests.

Request timeout first sends `ABORT_REQUEST`. A clean aborted or racing completed
terminal reuses the same HBM-resident worker. Process replacement occurs only when
the request cannot reach a clean terminal within the cancel timeout or returns a
process-unsafe 5xx/error: keep the active record, isolate the old process, then
restart, `LOAD_RUNTIME_BUNDLE`, and `SMOKE` before accepting the next request.
Restart after arbitrary repeated crashes outside an owned request can remain future
hardening.

## Shutdown

P0 shutdown remains engine-owned and idempotent:

```text
under the transition lock:
  reuse an existing shutdown future if present
  mark draining and closed
  prohibit new recovery spawn
  fail and remove queued RequestRecords
  mark the active RequestRecord aborting
  if active delivery is pending, commit shutdown_error and resolve its caller
  capture current worker lease
  try put_nowait ABORT_REQUEST using that lease's ID/epoch when active
  on full/closed/send error:
    mark and detach that lease process-unsafe
    set force_cleanup=true
  advance engine_epoch to fence recovery and READY publication
  claim active recovery/provisional cleanup futures
  /ready becomes 503
outside the lock:
  if not force_cleanup, wait up to worker_cancel_timeout for its one terminal
under the transition lock again:
  if not force_cleanup and the captured original lease remains current/open:
    try put_nowait SHUTDOWN using that lease's original ID/epoch
    on full/closed/send error:
      mark and detach that lease process-unsafe
      set force_cleanup=true
  otherwise:
    send no command to the retired lease
  collect current/provisional/recovery cleanup futures
outside the lock:
  if not force_cleanup, wait for orderly worker shutdown
  force-terminate a send-failed, unresponsive, or partially started worker
  await all claimed cleanup futures
under the transition lock:
  retire any remaining lease
  clear the active record and request-local parent state
outside the lock:
  join process and stop heartbeat/status consumers
  drain/close all per-spawn queues
  complete the shared shutdown future
```

Every worker exit after stage-adapter construction—including adapter load failure,
runner construction failure, readiness-smoke failure, request failure that retires
the worker, and orderly `SHUTDOWN`—uses one idempotent cleanup owner. If a stage
executor exists because runner ownership transferred, cleanup first awaits its
exhaustive shutdown; if no executor exists, that phase is skipped. Cleanup then
always awaits `stage_adapter.shutdown()` in a `finally`-equivalent path. It retains
and aggregates failures from both phases only after all applicable attempts finish;
any cleanup failure makes the worker process-unsafe.

On worker `SHUTDOWN`, the listener marks worker state draining and rejects future
RUN commands. It does not close runners or adapter state concurrently. After active
reset/execution/finalization exits, only the generation-thread owner invokes the
shared cleanup owner above. If startup, terminal, or orderly cleanup misses its
bound, the parent terminates the process and completes lease cleanup. The same rule
applies before readiness publication, so load/smoke failures cannot bypass adapter
cleanup merely because no request has run.

Both control sends are fail-closed. Their exceptions are consumed by the shared
shutdown operation rather than escaping it: the captured lease is detached under
the transition lock, orderly terminal/command waiting for that lease is skipped,
and the same idempotent cleanup future owns forced process termination plus queue
closure. The shared shutdown future resolves only after that cleanup, so repeated
shutdown calls await the same result.

Shutdown never needs caller cooperation because the engine has always owned the
record and terminal reader. Result delivery is linearized under the transition
lock: a worker terminal committed first delivers its result; shutdown committed
first delivers public `503 engine_draining`; a detached caller receives nothing. In
either case shutdown still proceeds to model/process shutdown, and a later terminal
is cleanup-only. The transition lock is not held while waiting for model execution
or process exit. Worker/epoch checks prevent a replacement or retired lease from
receiving shutdown commands. A process-unsafe terminal observed during the abort
wait cannot start a replacement because draining already owns the advanced epoch;
it contributes only its shared cleanup future to shutdown.

Tests inject completion immediately before, during, and after shutdown's locked
delivery commit, and also race shutdown with admission, queued removal, active
execution, abort terminal, process replacement, startup load/smoke, and caller
detachment. Completion requires no live worker, queued/active record, terminal
waiter, recovery/provisional task, or open per-spawn queue. Worker-side
tests inject shutdown during accepted, resetting, running graph, finalizing, and
replacement states and verify model shutdown never overlaps active execution.


## Deferred Extensions

Keep extension points for stage pools/replicas, placement, multiple resident
profiles, batching, and generalized recovery policy. P0 does not implement them.
Any extension must preserve immutable bundle binding, request/terminal ownership,
bounded admission, readiness smoke, and shutdown fencing; it must not overload the
P0 single-worker state machine with partially implemented scheduling behavior.

## Implementation Rule

Keep the P0 engine simple:

- one active profile
- one request running
- one worker-owned stage engine, adapter, and in-process executor per engine
- queue and timeout protection
- startup load plus smoke
- no generic stage scheduler yet

Add hooks as interfaces or TODO boundaries only when they do not complicate P0
control flow.
