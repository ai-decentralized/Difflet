# Difflet Serving Architecture

This document captures the serving code organization agreed for the first
text-to-image serving implementation.

## Scope

MVP serving supports:

- Qwen-Image through a resident worker that owns its staged orchestrator.
- Flux through a resident worker that owns its loaded `DiffletPipeline`.
- One active model/profile per server process.
- OpenAI-style `/v1/chat/completions` as the public API.
- R2-backed image artifact URLs.

MVP does not include subprocess serving fallback, rotating resident fallback,
multi-profile loading, local file serving, data URLs, or video serving.

## Folder Layout

```text
difflet/
  registry/
    __init__.py
    base.py
    flux.py
    qwen_image.py
    wan.py
    hunyuan_video.py
    ltx_2.py

  common/
    orchestrators/
      __init__.py
      base.py
      pipeline.py
      staged.py
      flux.py
      qwen_image.py
      ltx_2.py

  cli/
    orchestrators/
      ...

  serving/
    __init__.py
    engine.py
    model_registry.py
    factory.py
    options.py
    artifact_store.py
    orchestrators/
      __init__.py
      base.py
      qwen_image.py
      flux.py
    engines/
      __init__.py
      resident_worker.py
    openai/
      __init__.py
      api_server.py
      protocol.py
      serving_chat.py
      serving_models.py
      errors.py
    cli/
      serve.py
```

## Layer Responsibilities

### `difflet/registry`

The base registry is the source of truth for model identity and defaults.

It owns:

- `ModelEntry`
- `register_model(...)`
- `resolve_model(...)`
- `registered_models()`
- HF paths and aliases
- detector functions
- download allow patterns
- default shape
- default parallel config
- backend support
- `application_factory` for pipeline-style models

Model-specific registration should live in separate files. For example,
`difflet/registry/flux.py` owns the current `_register_builtin_flux()` logic:

```python
def register() -> None:
    def is_flux(model_id: str) -> bool:
        value = model_id.lower()
        return "flux" in value or "black-forest-labs/flux" in value

    register_model(
        name="flux",
        application_factory="difflet.models.flux.entry:create_flux_application",
        hf_paths=(
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell",
        ),
        detector=is_flux,
        default_parallel=DiffletParallelConfig(tp_degree=8),
        default_shape={"height": 1024, "width": 1024, "num_frames": None},
    )
```

`difflet/registry/__init__.py` imports/registers builtins once and preserves the
current public API.

Serving topology and serving lifecycle policy should not live in the base
registry.

### `difflet/common/orchestrators`

Common orchestrators are shared by CLI and serving.

They own reusable model execution infrastructure:

- model path resolution and download helpers.
- profile construction from `ModelEntry`, CLI args, or serving options.
- compile spec construction.
- compiled path naming.
- stage topology declarations.
- stage adapter construction.
- reusable `build_app(...)`, `compile(...)`, `load(...)`, and stage
  `generate(...)` helpers.
- pipeline helpers for Flux/LTX through `DiffletPipeline`.
- staged helpers for Qwen preserving current compiled directory naming and
  stage behavior.

CLI and serving should call this layer instead of duplicating model-specific
logic.

### `difflet/cli/orchestrators`

CLI orchestrators become thin wrappers.

They may still:

- parse CLI args.
- call common orchestrator download/compile/generate helpers.
- use `difflet.cli.runner.run_stage(...)` for offline subprocess stage
  execution.
- preserve existing `difflet download`, `difflet compile`, `difflet generate`,
  and `difflet run` behavior.

They should not be the long-term owner of reusable model stage logic.

### `difflet/serving/model_registry.py`

The serving registry is an overlay on top of `difflet.registry`.

It owns serving-specific metadata keyed by base `ModelEntry.name`:

- topology type
- stage roles
- runtime plans
- output modality
- artifact policy
- serving orchestrator factory

It should call `difflet.registry.resolve_model(...)` for model id matching,
default shape, default parallel config, backend support, and download patterns.

### `difflet/serving/orchestrators`

Serving orchestrators are serving adapters over common orchestrators.

In P0 they are worker-owned objects. The parent FastAPI process resolves the
orchestrator factory and sends `LOAD_PROFILE` to the resident worker; the worker
constructs the serving orchestrator, loads runtime handles, and runs smoke.

They own:

- active `ServingProfile`
- loaded pipeline/stage handles
- startup load and smoke readiness
- request validation integration
- serving-specific progress logs
- shutdown
- future serving-only profile switching/recovery behavior

The serving orchestrator exposes one public generation method:

```python
class ServingModelOrchestrator(Protocol):
    model_id: str
    model_type: str
    active_profile: ServingProfile

    def resolve_model_path(self, *, download_policy: DownloadPolicy) -> Path: ...
    def compile_plan(self, profile: ServingProfile) -> tuple[DiffletCompileSpec, ...]: ...
    def ensure_artifacts(self, profile: ServingProfile, policy: CompilePolicy) -> None: ...
    def load(self, profile: ServingProfile) -> None: ...
    def smoke(self) -> None: ...
    async def generate(self, request: DiffletGenerateRequest, context: WorkerRequestContext) -> DiffletGenerateOutput: ...
    def shutdown(self) -> None: ...
```

Stage traversal is internal to the orchestrator.

`WorkerRequestContext` is created by the worker runtime, not by the HTTP layer.
It carries the request deadline and one cancellation signal:

```python
@dataclass
class WorkerRequestContext:
    request_id: str
    deadline: float
    cancellation: CancellationSignal
```

The cancellation signal is the only cancellation hook model code should see.
It is checked at safe points; it is not a model-specific control channel.

For Qwen:

```text
generate(request, context)
  -> context.cancellation.throw_if_cancelled()
  -> prompt_encoder.generate(...)
  -> context.cancellation.throw_if_cancelled()
  -> denoiser.generate(...)
  -> context.cancellation.throw_if_cancelled()
  -> decoder.generate(...)
  -> DiffletGenerateOutput(bytes, mime="image/png")
```

For Flux:

```text
generate(request, context)
  -> context.cancellation.throw_if_cancelled()
  -> pipe(...)
  -> DiffletGenerateOutput(bytes, mime="image/png")
```

The serving engine should send `RUN_GENERATION` to the worker during request
execution. The worker runtime should create `WorkerRequestContext`, own the
cancellation signal, and call only `orchestrator.generate(request, context)`.

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
- `ArtifactStore` upload and URL selection.

It should not know the model stage count.

## Startup Flow

```text
parse serve args
resolve base ModelEntry through difflet.registry
resolve serving metadata through difflet.serving.model_registry
build one ServingProfile
select serving orchestrator factory
resolve/download model weights
build compile plan
check or compile artifacts
create engine
start resident worker
construct serving orchestrator inside worker
load pipeline/stage apps inside worker
run serving smoke through worker
create FastAPI app
bind HTTP
serve ready traffic
```

Download, compile, load, and smoke must emit progress logs before and after
long-running steps so startup does not appear stuck.

## Request Flow

```text
FastAPI /v1/chat/completions
  -> parse OpenAI envelope
  -> validate prompt, extra_body, profile, and modality
  -> engine.generate(request)
  -> worker IPC RUN_GENERATION
  -> worker runtime creates WorkerRequestContext
  -> worker-owned orchestrator.generate(request, context)
  -> model-specific pipeline/stage execution inside worker
  -> image bytes
  -> ArtifactStore.put_bytes(...)
  -> chat response with image_url.url
```

## MVP Model Mapping

| Model | Base registry | Serving orchestrator | Engine | Notes |
| --- | --- | --- | --- | --- |
| Qwen-Image | `qwen_image` | `serving/orchestrators/qwen_image.py` | `ResidentWorkerServingEngine` | Worker-owned `prompt_encoder -> denoiser -> decoder`. |
| Flux | `flux` | `serving/orchestrators/flux.py` | `ResidentWorkerServingEngine` | Worker-owned loaded `DiffletPipeline`. |

Each server process loads exactly one model/profile.

## Adding A Model

Adding a model should be explicit. Do not rely on automatic stage inference for
serving until the model has a registered topology and orchestrator.

### 1. Add Base Registry Entry

Create or update a model file under `difflet/registry/`.

Example:

```text
difflet/registry/new_model.py
```

The registration must define:

- model `name` / `model_type`
- HF paths and aliases
- detector function
- `application_factory` if the model can use a pipeline-style app
- default shape
- default parallel config
- backend support
- download allow patterns if the model needs non-default files

Then import/register it from `difflet/registry/__init__.py`.

This is the only place model id matching should be added.

### 2. Add Common Orchestrator

Create:

```text
difflet/common/orchestrators/new_model.py
```

The common orchestrator owns reusable model logic:

- resolve/download model path
- build profile defaults from `ModelEntry`
- compute compiled artifact paths
- build compile specs
- check artifact manifests/files
- compile missing artifacts
- build/load pipeline or stage apps
- implement per-stage `generate(...)` helpers

For a pipeline-style model, this usually wraps:

```python
DiffletPipeline.precompile(...)
DiffletPipeline.from_pretrained(..., skip_compile=True)
```

For a staged model, this usually extracts logic from the existing CLI
orchestrator into importable stage helpers.

### 3. Define Stage Topology

Define the model's serving topology in serving metadata, using generic stage
roles rather than model-specific CLI names.

Examples:

```text
Qwen text      -> prompt_encoder
Qwen generate  -> denoiser
Qwen vae       -> decoder

Wan transformer -> denoiser
Wan vae         -> decoder

Flux pipeline   -> pipeline
```

Each stage definition should include:

- stable stage id
- generic stage role
- input sources
- output keys
- whether it is final output
- final output type when applicable
- required cores
- shape/profile binding
- compiled artifact identity fields
- adapter factory

The registry declares stage metadata. The orchestrator owns execution.

### 4. Add Serving Registry Metadata

Update `difflet/serving/model_registry.py`.

Add a `ServingModelMetadata` entry keyed by `ModelEntry.name`:

```python
_SERVING_METADATA["new_model"] = ServingModelMetadata(
    model_type="new_model",
    topology_type=...,
    input_modalities=("text",),
    output_modalities=("image",),
    stage_factory=...,
    runtime_plan_factory=...,
    orchestrator_factory=...,
    artifact_policy="r2_url",
)
```

This metadata tells serving how to construct the model topology and which
serving orchestrator to use. It should not duplicate HF ids or base defaults
that already live in `difflet.registry`.

### 5. Add Serving Orchestrator

Create:

```text
difflet/serving/orchestrators/new_model.py
```

The serving orchestrator wraps the common orchestrator and owns serving state:

- active `ServingProfile`
- loaded pipe or stage app handles
- startup load
- smoke readiness
- request-time validation hooks
- `generate(...)`
- shutdown

Its public request-time API should stay:

```python
async def generate(self, request: DiffletGenerateRequest, context: WorkerRequestContext) -> DiffletGenerateOutput:
    ...
```

For multi-stage models, `generate(...)` internally walks the registered stage
topology and calls each stage adapter in order. It should check
`context.cancellation` only at safe points. The engine does not know the model's
stage count.

### 6. Add Engine Support If Needed

Use an existing engine whenever possible:

- `ResidentWorkerServingEngine` for Trainium serving, whether the model is a
  single pipeline or multiple stages.

Only add a new engine when the runtime behavior is truly new, such as a future
distributed profile pool. Do not add model-specific logic to engines.

### 7. Add OpenAI Contract Support

Update the OpenAI contract only when the model changes public behavior:

- new input modality
- new output modality
- new `extra_body` fields
- new output format
- different artifact policy
- different validation ranges

For another text-to-image model that returns PNG bytes through R2, prefer adding
adapter validation rather than changing the HTTP contract.

### 8. Add Tests And Smoke

At minimum add tests for:

- model id resolution through `difflet.registry`
- serving metadata resolution
- serving profile construction
- compiled artifact path/cache identity
- download policy
- compile policy
- load/smoke readiness
- supported and unsupported `extra_body` fields
- shape/profile mismatch
- output artifact formatting
- failure paths: queue, timeout, worker death, artifact upload failure

For Trainium models, add a real startup smoke on the target profile before
marking `/ready` successful.

## Model Addition Checklist

```text
[ ] Add base registry file under difflet/registry/
[ ] Import/register the model from difflet/registry/__init__.py
[ ] Add common orchestrator under difflet/common/orchestrators/
[ ] Define stage topology and stage roles
[ ] Add serving metadata in difflet/serving/model_registry.py
[ ] Add serving orchestrator under difflet/serving/orchestrators/
[ ] Reuse ResidentWorkerServingEngine unless runtime behavior is truly new
[ ] Add adapter-specific request validation
[ ] Add artifact/cache manifest validation
[ ] Add OpenAI contract docs if public behavior changes
[ ] Add unit tests and startup smoke
```
