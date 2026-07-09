# Difflet Serving Engine

This document defines the serving engine boundary for the MVP implementation.

The engine is the serving runtime layer. It owns request execution safety and
worker lifecycle. It does not own model-specific stage logic.

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
- Graceful shutdown.
- Best-effort cancel/cleanup.

MVP engines do not implement:

- multi-profile scheduling.
- co-batching.
- dynamic stage placement.
- rotating resident scheduling.
- subprocess fallback.
- cross-host distributed execution.
- generic stage DAG optimization.

Those are future architecture hooks, not P0 behavior.

## Runtime Boundary

```text
FastAPI route
  -> DiffletServingEngine.generate(request)
  -> queue / timeout / health / worker lifecycle
  -> worker IPC
  -> worker runtime creates WorkerRequestContext
  -> worker-owned ServingModelOrchestrator.generate(request, context)
  -> common orchestrator / stage adapters / pipeline inside worker
```

The engine must not hard-code model-specific stage names such as Qwen
`text/generate/vae`. It should see only:

- a `ServingProfile`
- a serving orchestrator factory or worker-owned `ServingModelOrchestrator`
- `DiffletGenerateRequest`
- `DiffletGenerateOutput`

## Engine Protocol

```python
class DiffletServingEngine(Protocol):
    active_profile: ServingProfile
    output_modalities: tuple[str, ...]

    async def start(self) -> None: ...
    async def generate(self, request: DiffletGenerateRequest) -> DiffletGenerateOutput: ...
    async def health(self) -> EngineHealth: ...
    async def shutdown(self) -> None: ...
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

The engine does not own:

- model id detection
- tokenizer/template logic
- compiled path naming
- model download
- AOT compile implementation
- Qwen stage traversal
- Flux pipeline invocation details
- R2 upload
- OpenAI response formatting

## P0 Engine Type

### `ResidentWorkerServingEngine`

Used by Qwen-Image and Flux.

```text
ResidentWorkerServingEngine
  -> owns queue/timeout/health/draining
  -> starts one Trainium worker process for the active profile
  -> sends LOAD_PROFILE / SMOKE / RUN_GENERATION over IPC
  -> receives final image bytes plus metadata
```

The worker process owns one serving orchestrator. The orchestrator can be a
single-pipeline model such as Flux or a multi-stage model such as Qwen.

Flux parent preflight:

```text
FluxServingArtifactPreparer
  ensure_artifacts()
    verify CacheSpec manifest and compiled artifact readiness
```

Flux worker layout:

```text
FluxServingOrchestrator
  active_profile
  pipe

  load()
    pipe = DiffletPipeline.from_pretrained(..., skip_compile=True)

  generate(request, context)
    context.cancellation.throw_if_cancelled()
    output = pipe(...)
    return DiffletGenerateOutput(image_bytes, "image/png")
```

Qwen worker layout:

```text
QwenServingOrchestrator
  active_profile
  prompt_encoder_stage
  denoiser_stage
  decoder_stage

  load()
    load all stage apps into the worker process

  generate(request, context)
    context.cancellation.throw_if_cancelled()
    text = prompt_encoder.generate(...)
    context.cancellation.throw_if_cancelled()
    latents = denoiser.generate(text)
    context.cancellation.throw_if_cancelled()
    image = decoder.generate(latents)
    return DiffletGenerateOutput(image_bytes, "image/png")
```

Qwen is enabled only after this shared-worker layout passes startup load and
smoke for the selected `ServingProfile`. The smoke must prove that all Qwen
stage handles co-load in the single resident worker process and can execute a
minimal request without per-stage subprocesses, rotating load/unload, or file
handoff. If this gate fails, startup fails and `/ready` never becomes healthy.

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
```

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

Artifact upload and URL generation happen after the engine returns bytes to the
OpenAI handler, so they are bounded by `artifact_store_timeout`, not by the
engine `request_timeout`. The artifact layer must configure SDK/client
timeouts and fail closed with `artifact_upload_failed` or
`artifact_store_unavailable`; it must not let a hung R2 upload or presign hold
the HTTP request indefinitely.

## Generate Flow

Generic engine logic:

```python
async def generate(self, request: DiffletGenerateRequest) -> DiffletGenerateOutput:
    if self.draining:
        raise EngineDraining()

    received_at = monotonic()
    deadline = received_at + self.request_timeout
    ticket = await self.admit_or_raise(
        request,
        received_at_monotonic=received_at,
        deadline_monotonic=deadline,
    )
    remaining = deadline - monotonic()
    if remaining <= 0:
        self.release(ticket)
        raise RequestTimeout()

    run_task = asyncio.create_task(
        self.run_one(request, ticket),
        name=f"difflet-generate-{request.request_id}",
    )
    release_ticket = True
    try:
        return await asyncio.wait_for(asyncio.shield(run_task), timeout=remaining)
    except asyncio.CancelledError:
        release_ticket = False
        self.start_inflight_recovery(
            request.request_id,
            ticket,
            run_task,
            reason="caller_cancelled",
        )
        raise
    except asyncio.TimeoutError:
        release_ticket = False
        self.start_inflight_recovery(
            request.request_id,
            ticket,
            run_task,
            reason="timeout",
        )
        raise RequestTimeout()
    finally:
        if release_ticket:
            self.release(ticket)
```

On timeout, `asyncio.shield(...)` prevents `wait_for` from cancelling the
worker-receive path. `start_inflight_recovery(...)` transfers ownership of the
execution ticket and the in-flight worker task to a recovery task. The request
can return `504` immediately, but the ticket and worker must not be reused until
recovery reaches a safe terminal state. Worker IPC replies must be tagged with
`request_id`, and recovery must drain/discard any late terminal
`GENERATION_OK`/`GENERATION_ERROR` reply for that request before reusing the
worker. If it cannot prove the reply stream is clean, it must terminate and
restart the worker. The recovery task should:

Caller cancellation, such as client disconnect or ASGI shutdown cancellation,
uses the same ownership-transfer path. The engine must not release the ticket
while the shielded worker task may still be running. After starting recovery for
`asyncio.CancelledError`, `generate()` must re-raise `CancelledError` rather
than translating it to `RequestTimeout`.

```text
mark worker RECOVERING / unavailable
/ready returns 503
send CANCEL to the worker
wait up to worker_cancel_timeout for CANCEL_ACK
drain/discard any late terminal reply tagged with request_id
if CANCEL_ACK is not received:
  terminate the worker process
restart worker
LOAD_PROFILE
SMOKE
if smoke succeeds:
  mark worker READY
  release the execution slot
else:
  keep engine unhealthy and /ready=503
```

The restart plus `LOAD_PROFILE -> SMOKE` phase is bounded by
`worker_restart_timeout`. If that timeout expires, or if restart/load/smoke
fails, the engine must leave `RECOVERING`, mark the worker state `ERROR`, keep
`/ready` at 503, and reject new generation requests with
`503 engine_unavailable` until the process is restarted or a future recovery
policy explicitly retries.

This makes `CANCEL` best-effort and recovery process-level. P0 should not try
to interrupt a Trainium model call inside the same Python process and then
immediately reuse that worker.

P0 `run_one`:

```python
async def run_one(self, request, ticket):
    return await self.worker_rpc.run_generation(
        request,
        deadline_monotonic=ticket.deadline_monotonic,
    )
```

## Worker IPC

P0 worker IPC should be worker-level, not per-stage and not model-specific.

Commands:

```text
LOAD_PROFILE
SMOKE
RUN_GENERATION
CANCEL
HEALTH
SHUTDOWN
```

`RUN_GENERATION` sends one `DiffletGenerateRequest` plus `deadline_monotonic` to
the worker. The worker builds `WorkerRequestContext`, runs the active
orchestrator internally, and returns a `DiffletGenerateOutput`.

`CANCEL` is a control command for the currently running request id. A worker
returns `CANCEL_ACK` only after the request is in a safe terminal state from the
worker's perspective: the generation stopped before entering an unsafe call, or
the generation finished and the worker discarded the result. If the worker is
blocked inside a non-interruptible Trainium/runtime call, it may not respond;
the parent must then terminate and restart the worker process.

The engine does not inject cancellation logic into model handles. Cancellation
is a worker-runtime signal:

```text
engine sends CANCEL(request_id)
worker runtime sets request_context.cancel_requested = True
orchestrator/stage code checks the signal only at safe points
worker runtime sends CANCEL_ACK after request state is gone
```

Safe points include before `orchestrator.generate(...)` starts, between Qwen
stages, between denoiser timesteps if the adapter exposes a Python loop, after a
late result is discarded, or after a handled error releases request-local state.
If the active call has no safe checkpoint, the worker should not fabricate
`CANCEL_ACK`; the parent will terminate and restart it.

Intermediate tensors and model objects must not cross the parent/worker IPC
boundary in P0.

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

Automatic worker restart after request timeout is part of P0 recovery. Restart
after arbitrary repeated crashes can remain future hardening.

## Shutdown

Shutdown sequence:

```text
mark draining
/ready returns 503
reject new requests
fail queued requests
wait for active request up to shutdown timeout
best-effort cancel if timeout is exceeded
send SHUTDOWN to worker/orchestrator
terminate unresponsive worker process
cleanup request-local state
exit
```

`shutdown()` must be idempotent.

## Future Hooks

The MVP engine should leave hooks for future architecture improvements without
implementing them now.

### Stage Scheduler Hook

Future versions may add a scheduler that sees stage-level work items:

```python
class StageScheduler(Protocol):
    async def submit(self, request: DiffletGenerateRequest, graph: StageGraph) -> DiffletGenerateOutput: ...
```

MVP should not use this hook. Qwen stage traversal stays inside
`QwenServingOrchestrator.generate(...)`.

### Placement Hook

Future versions may make placement explicit:

```python
class PlacementPlanner(Protocol):
    def plan(self, profile: ServingProfile, stages: tuple[DiffletStageSpec, ...]) -> PlacementPlan: ...
```

MVP uses fixed placement:

- Qwen: one shared worker process owns the 4-core group and all Qwen stage
  adapters.
- Flux: one shared worker process owns the active Flux pipeline profile.

### Profile Manager Hook

Future versions may switch or pool profiles:

```python
class ProfileManager(Protocol):
    async def activate(self, profile: ServingProfile) -> None: ...
    async def current(self) -> ServingProfile: ...
```

MVP loads exactly one active profile and rejects multi-profile startup.

### Batching Hook

Future versions may co-batch compatible requests:

```python
class RequestBatcher(Protocol):
    async def next_batch(self) -> list[DiffletGenerateRequest]: ...
```

MVP keeps `max_running_requests=1`.

### Recovery Hook

Future versions may restart workers automatically:

```python
class WorkerRecoveryPolicy(Protocol):
    async def recover(self, failure: WorkerFailure) -> RecoveryResult: ...
```

MVP fails readiness after worker death and requires process-level restart.

## Implementation Rule

Keep the P0 engine simple:

- one active profile
- one request running
- one worker-owned orchestrator per engine
- queue and timeout protection
- startup load plus smoke
- no generic stage scheduler yet

Add hooks as interfaces or TODO boundaries only when they do not complicate P0
control flow.
