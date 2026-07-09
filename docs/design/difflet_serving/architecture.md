# Difflet Serving Architecture

This document captures the serving code organization agreed for the first
text-to-image serving implementation.

This file, together with `engine.md` and `chat_completions_contract.md`, is the
authoritative P0 serving design. The older
`docs/plans/2026-07-06-difflet-serving-engine.md` keeps broader future design
notes, including rotating, subprocess, and multi-plan ideas, but those are not
part of the P0 implementation scope unless repeated here.

## Scope

MVP serving supports:

- Qwen-Image through a resident worker that owns its staged orchestrator.
- Flux through a resident worker that owns its loaded `DiffletPipeline`.
- One active model/profile per server process.
- OpenAI-style `/v1/chat/completions` as the public API.
- R2-backed image artifact URLs.

MVP does not include subprocess serving fallback, rotating resident fallback,
multi-profile loading, local file serving, data URLs, or video serving.

Qwen-Image is a P0 target only when the selected `ServingProfile` passes a real
shared-worker co-load and smoke test: prompt encoder, denoiser, and decoder must
load into the same resident Trainium worker process and run a startup smoke
without falling back to per-stage processes or file handoff. If this gate fails,
startup must fail clearly; P0 must not silently downgrade to subprocess or
rotating behavior.

## Folder Layout

```text
difflet/
  registry.py                    # existing base registry; keep unchanged

  common/
    registry/
      __init__.py
      base.py                    # common registry helpers, wraps old ModelEntry
      flux.py
      qwen_image.py
      wan.py
      hunyuan_video.py
      ltx_2.py
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
    model_registry.py              # serving overlay over common registry + difflet.registry
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
      serving_chat.py
      serving_models.py
    cli/
      serve.py
```

## Layer Responsibilities

### `difflet/registry.py`

The existing base registry module remains the source of truth for model identity
and defaults. Do not replace it with a `difflet/registry/` package in the serving
MVP, because current CLI and standalone scripts may import `difflet.registry`
directly.

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

Serving and common code should keep calling the existing public API:

```python
from difflet.registry import ModelEntry, register_model, resolve_model, registered_models
```

For example, the existing `_register_builtin_flux()` logic remains in
`difflet/registry.py`:

```python
def _register_builtin_flux() -> None:
    def is_flux(model_id: str) -> bool:
        value = model_id.lower()
        return "flux" in value or "black-forest-labs/flux" in value

    @register_model(
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
    class _FluxRegistration:
        pass
```

Serving topology and serving lifecycle policy should not live in the base
registry.

### `difflet/common/registry`

`difflet/common/registry/` is the new modular registry namespace for shared CLI
and serving infrastructure. It must not shadow `difflet.registry`; it should
wrap and reuse the existing base registry instead.

It owns common metadata that is useful outside a single entrypoint:

- common model family metadata modules keyed by `ModelEntry.name`
- serving/common stage-role declarations that are not part of the old base
  registry
- helper functions that convert a base `ModelEntry` plus profile options into
  common model descriptors
- per-model helper modules such as `difflet/common/registry/flux.py` and
  `difflet/common/registry/qwen_image.py`

It should call `difflet.registry.resolve_model(...)` for base model matching and
defaults. It should not duplicate the old `hf_paths`, detector functions, or
default shape/parallel values unless the value is serving/common-specific.

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
logic. P0 prioritizes the serving path: new serving adapters should place shared
path/cache/artifact helpers here first, while existing CLI orchestrators can
migrate incrementally as their behavior is touched.

In this document, "adapter" means a model-specific implementation of a common
serving interface, not a separate framework directory that already exists. For
P0, the concrete serving adapters are expected under
`difflet/serving/orchestrators/`, for example
`qwen_image.py` and `flux.py`. The serving registry wires them through
`preflight_factory` and `orchestrator_factory`.

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

The serving model registry is a serving-only overlay on top of
`difflet.common.registry` and the existing `difflet.registry`.

It owns serving-specific metadata keyed by base `ModelEntry.name`:

- serving-enabled checkpoint ids
- topology type
- stage roles
- runtime plans
- output modality
- artifact policy
- serving orchestrator factory

P0 keeps this overlay centralized in `difflet/serving/model_registry.py`.
Per-model common metadata lives under `difflet/common/registry/<model>.py`, for
example `difflet/common/registry/flux.py` and
`difflet/common/registry/qwen_image.py`. The serving overlay imports those
common metadata providers, registers only serving-enabled checkpoints, and
exposes `resolve_serving_model(...)` plus factory loading helpers.

It should use `difflet.common.registry` to obtain serving/common model metadata,
and that common layer should call `difflet.registry.resolve_model(...)` for
startup family resolution, default shape, default parallel config, backend
support, and download patterns. After family resolution, serving must check the
requested checkpoint id against its serving-enabled checkpoint allowlist before
building a serving spec.

### `difflet/serving/orchestrators`

Serving orchestrators are serving adapters over common orchestrators.

P0 splits cold startup work from worker runtime work:

- Parent-side preflight prepares artifacts before the worker starts.
- Parent-side request validators check model-specific request limits before
  worker admission.
- Worker-owned serving orchestrators load runtime handles, run smoke, generate,
  and shut down.

The parent FastAPI process resolves the preflight/orchestrator factories, runs
download/compile/artifact checks through the preflight object, then starts the
resident worker and sends `LOAD_PROFILE`. The worker constructs the serving
orchestrator, loads runtime handles, and runs smoke.

Parent-side preflight owns:

- model path resolution and optional download.
- compile plan construction.
- compiled artifact checks and optional compile.
- startup progress logs for cold operations.

The lifecycle order is common across models, but the implementation of each
method is model-specific. For example, Flux can build a `CacheSpec` and validate
pipeline cache manifests, while Qwen-Image must enumerate staged encoder,
denoiser, and VAE artifacts. Keep these operations in the parent-side
`ServingArtifactPreparer`; do not duplicate download, compile, or artifact
validation inside the worker `ServingModelOrchestrator`.

```python
class ServingArtifactPreparer(Protocol):
    model_id: str
    model_type: str

    def resolve_model_path(self, *, download_policy: DownloadPolicy) -> Path: ...
    def stage_specs(self, profile: ServingProfile) -> tuple[DiffletStageSpec, ...]: ...
    def compile_plan(self, profile: ServingProfile) -> tuple[DiffletCompileSpec, ...]: ...
    def ensure_artifacts(self, profile: ServingProfile, policy: CompilePolicy) -> None: ...
```

Request validators are also model-specific, but they run in the parent/FastAPI
process:

```python
class ServingRequestValidator(Protocol):
    def validate(self, request: DiffletGenerateRequest, profile: ServingProfile) -> None: ...
```

The protocol lives in `difflet/serving/orchestrators/base.py`. Concrete
validators live next to the serving adapter, for example
`FluxServingRequestValidator` in `difflet/serving/orchestrators/flux.py` and
`QwenImageServingRequestValidator` in
`difflet/serving/orchestrators/qwen_image.py`.

Use this hook for checks that must happen before worker admission, such as
prompt tokenizer/bucket limits. It must not load Trainium runtime objects or
hold Neuron cores.

Worker-owned serving orchestrators own:

- active `ServingProfile`
- loaded pipeline/stage handles
- worker-side load and smoke readiness
- serving-specific progress logs
- shutdown
- future serving-only profile switching/recovery behavior

Worker orchestrators consume the already resolved model path/profile/artifact
identity prepared during startup. They may compute local paths needed for
`load(...)`, but they must not download weights, run AOT compile, or decide that
artifacts are valid enough for readiness. If load discovers a missing or stale
artifact despite preflight, it should fail load/smoke and make startup fail.

The serving orchestrator exposes one public generation method:

```python
class ServingModelOrchestrator(Protocol):
    model_id: str
    model_type: str
    active_profile: ServingProfile

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
    deadline_monotonic: float
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
build one ServingProfile from registry defaults plus serve-flag overrides
select preflight and serving orchestrator factories
parent preflight: resolve/download model weights
parent preflight: build compile plan
parent preflight: check or compile artifacts
construct parent-side request validator
create engine
create FastAPI app with lifespan startup
lifespan startup: start resident worker
construct serving orchestrator inside worker
load pipeline/stage apps inside worker
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

Serving must not hard-code one global parallel default. `--tp-degree`,
`--cp-degree`, `--cp-mode`, `--height`, and `--width` are P0 image startup
overrides; when omitted, profile construction uses the resolved
`ModelEntry.default_parallel` and `ModelEntry.default_shape`, with any
serving-adapter default only where the model requires one. The `serve`
subcommand should therefore use serving-specific parser helpers whose startup
profile/runtime flags default to `None`. Do not reuse CLI helpers whose defaults
erase whether the operator omitted a value.

`--num-frames` is reserved for future video adapters. For Qwen/Flux P0 image
serving, a non-null startup `num_frames` override must be rejected during
startup profile validation rather than baked into `ServingProfile`.

P0 uses one model-level serving profile that can be overridden at startup with
`difflet serve --tp-degree`, `--cp-degree`, `--height`, `--width`, and related
flags. Stage-specific differences belong in serving stage metadata and
placement/core calculations. For example, Qwen P0 requires `cp_degree=1`; under
that supported profile, Qwen text and denoiser stages consume the model-level
`tp_degree`, while the Qwen decoder is a fixed-core stage owned by the adapter.
Do not expose separate per-stage TP/CP flags in the first serving milestone; add
an explicit stage-profile override only if a future model requires genuinely
different stage parallel configs.

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
  -> await ArtifactStore.put_bytes(...)
  -> await ArtifactStore.get_url(ref)
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

### 1. Ensure Base Registry Resolution

First verify that the existing `difflet/registry.py` can resolve the model id
and returns the expected `ModelEntry`. This is the only place broad model id
matching, HF paths, detector functions, default shape, default parallel config,
and `application_factory` should live.

If a future model is missing from the old base registry, update
`difflet/registry.py` as a separate compatibility-preserving change using the
current `_register_builtin_*()` pattern. Do not create `difflet/registry/`, and
do not move existing registry code as part of serving.

The base registration must define:

- model `name` / `model_type`
- HF paths and aliases
- detector function
- `application_factory` if the model can use a pipeline-style app
- default shape
- default parallel config
- backend support
- download allow patterns if the model needs non-default files

### 2. Add Common Registry Metadata

Create or update a model file under `difflet/common/registry/`.

Example:

```text
difflet/common/registry/new_model.py
```

The common registry entry should reference the base `ModelEntry.name` and define
metadata shared by CLI/common/serving infrastructure:

- common model family id
- supported exact checkpoint ids for common/serving paths
- topology family
- generic stage roles
- model-specific profile constraints that are not in the old base registry
- factories for common orchestrator helpers when useful

This layer may call the old base registry, but it must not duplicate broad model
id matching.

### 3. Add Common Orchestrator

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

The serving path must run a common `ensure_artifacts(...)` check before
`skip_compile=True` load. For Flux P0, that check must verify the selected
`CacheSpec` manifest and app-specific compiled artifact readiness before the
worker constructs or loads `DiffletPipeline`.

For a staged model, this usually extracts logic from the existing CLI
orchestrator into importable stage helpers.

### 4. Define Stage Topology

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

P0 serving metadata stores the stable topology and role/output identity.
Profile-dependent fields such as exact core count are still calculated by the
parent-side `ServingArtifactPreparer.stage_specs(profile)` because they depend
on startup flags like `--tp-degree` and `--cp-degree`.

`stage_id` is the stable artifact/runtime identifier for a model family and may
preserve existing CLI names, such as Qwen `text`, `generate`, and `vae`, so
compiled cache paths remain compatible. `stage role` is the generic serving
role, such as `prompt_encoder`, `denoiser`, `decoder`, or `pipeline`. The
engine should branch on roles/profile metadata, not on Qwen-specific
`stage_id` strings.

The registry declares stage metadata. The orchestrator owns execution.

### 5. Add Serving Registry Metadata

Add common model metadata under `difflet/common/registry/`, for example
`difflet/common/registry/new_model.py`.

Expose a `ServingModelMetadata` provider:

```python
def serving_metadata() -> ServingModelMetadata:
    return ServingModelMetadata(
        model_type="new_model",
        checkpoint_ids=("org/NewModel",),
        output_modality="image",
        output_mime_type="image/png",
        chat_content_type="image_url",
        default_steps=28,
        default_guidance_scale=3.5,
        preflight_factory="difflet.serving.orchestrators.new_model:NewModelPreparer",
        orchestrator_factory="difflet.serving.orchestrators.new_model:NewModelOrchestrator",
        request_validator_factory="difflet.serving.orchestrators.new_model:NewModelValidator",
    )
```

Then update `difflet/serving/model_registry.py` to import that provider and
register it in the serving allowlist. This metadata tells serving how to
construct the model topology and which parent-side preflight and worker-side
serving orchestrator to use, plus which parent-side request validator to run
before worker admission. It should not duplicate base defaults that already live
in `difflet.registry`, but it must list serving-enabled checkpoint ids
explicitly so a broad base registry family such as Flux does not automatically
enable every sibling checkpoint for serving.

### 6. Add Serving Orchestrator

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

### 7. Add Engine Support If Needed

Use an existing engine whenever possible:

- `ResidentWorkerServingEngine` for Trainium serving, whether the model is a
  single pipeline or multiple stages.

Only add a new engine when the runtime behavior is truly new, such as a future
distributed profile pool. Do not add model-specific logic to engines.

### 8. Add OpenAI Contract Support

Update the OpenAI contract only when the model changes public behavior:

- new input modality
- new output modality
- new `extra_body` fields
- new output format
- different artifact policy
- different validation ranges

For another text-to-image model that returns PNG bytes through R2, prefer adding
adapter validation rather than changing the HTTP contract.

### 9. Add Tests And Smoke

At minimum add tests for:

- model id resolution through `difflet.registry`
- common registry metadata resolution through `difflet.common.registry`
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
[ ] Confirm the model resolves through existing difflet/registry.py
[ ] If missing, add base ModelEntry to difflet/registry.py as a separate compatibility-preserving change
[ ] Add common registry metadata under difflet/common/registry/
[ ] Add common orchestrator under difflet/common/orchestrators/
[ ] Define stage topology and stage roles
[ ] Register serving metadata in difflet/serving/model_registry.py
[ ] Add serving orchestrator under difflet/serving/orchestrators/
[ ] Reuse ResidentWorkerServingEngine unless runtime behavior is truly new
[ ] Add adapter-specific request validation
[ ] Add artifact/cache manifest validation
[ ] Add OpenAI contract docs if public behavior changes
[ ] Add unit tests and startup smoke
```
