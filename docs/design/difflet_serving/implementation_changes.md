# Difflet Serving Implementation Changes

This document lists code changes relative to the current `feature/serving_t2i`
branch. Final contracts live in [architecture.md](architecture.md), engine lifecycle
in [engine.md](engine.md), and the public API in
[chat_completions_contract.md](chat_completions_contract.md).

## Unchanged Boundaries

- Existing staged CLI argparse, model resolution, subprocess order, tensor-file
  handoff, output paths, and calibration pathname behavior remain unchanged.
- Flux remains one opaque pipeline; its internal component topology is not converted
  to generic stages.
- P0 keeps one resident worker and one active profile. It executes the ordered
  stage list sequentially through an in-process executor. Dynamic profile
  switching, per-stage workers, MPMD, generic DAG scheduling, and numerical HBM
  estimation remain deferred.

## File And Symbol Inventory

This is the implementation checklist. A row names the existing symbols to replace
or the new contract to add; it does not authorize broader refactoring.

| File | Exact change |
|---|---|
| `difflet/common/registry/base.py` | Replace `ServingStageMetadata` and `ServingModelMetadata.stages` with shared `StageDefinition`/`PipelineDefinition`; rename `preflight_factory` to `artifact_preparer_factory` without an alias. |
| `difflet/common/registry/{qwen_image,flux}.py` | Construct one pipeline definition per model; Qwen IDs are `text/generate/vae`, Flux has only external ID `pipeline`; point metadata at the renamed factory. |
| `difflet/serving/types.py` | Add the runtime environment/allocation/stage (including extracted/opaque kind), artifact identity/binding/set, frozen adapter config, bundle with `pipeline_definition`, nominal `StagePayload` marker, generic `StageInvocation`/`StageExecutionResult`, context, and request-record contracts; extend `DiffletCompileSpec`; delete `DiffletStageSpec`, `CancellationSignal`, and TeaCache fields from `ServingProfile`. |
| `difflet/serving/orchestrators/base.py` | Replace `compile_plan(profile)`/`ensure_artifacts(...)` with the stage-adapter, artifact-preparer, request-validator, and loaded-runner protocols defined in `architecture.md`. |
| `difflet/serving/model_registry.py` | Rename `load_preflight_factory()` to `load_artifact_preparer_factory()`; resolve raw `ServeOptions` through the selected model adapter and construct validators from the frozen bundle. |
| `difflet/serving/factory.py` | Call `prepare_runtime()` once before worker startup; pass one `ResolvedRuntimeBundle` to the validator and engine; copy `ServeOptions.worker_heartbeat_interval` into `ResidentWorkerConfig`; remove independent path/profile preflight calls. |
| `difflet/serving/artifact_manager.py` | Add the common lock, lookup, staging, inventory, manifest, atomic generation publication, validation, and binding implementation. |
| `difflet/pipeline/compile_cache.py` | Preserve adapter `manifest.json` as payload; support serving identity from pinned source and explicit manager target without selecting or overwriting the common generation manifest. Existing CLI cache selection remains unchanged. |
| `difflet/pipeline/difflet_pipeline.py` | Add all-or-none `model_path_override`, `resolved_source_id`, and `compiled_path_override`; bound serving compile/load skips source/cache selection. Calls with no overrides retain current CLI behavior. |
| `difflet/common/orchestrators/qwen_image.py` | Delete the serving-only helpers listed below, retain `stage_compiled_dir_from_values()`, and expose/rehome payload validators. Add `build_text_application(...)`, `build_generate_application(...)`, and `build_vae_application(...)` from explicit model path, compiled path/target, topology, shape, dtype, and mode; these builders never resolve revisions/paths or perform CLI file I/O. |
| `difflet/common/orchestrators/flux.py` | Keep shared application construction but accept the serving adapter's pinned source, bound target/path, and prevalidated calibration object. |
| `difflet/cli/main.py` | Remove `serve` from `_validate_cfg_parallel`, `_validate_sp`, and `_validate_teacache` dispatch. Serve alone gets `--cfg-parallel/--no-cfg-parallel` -> `cfg_parallel: bool | None`, `--sp/--no-sp` -> `sp_enabled: bool | None`, and positive-float `--worker-heartbeat-interval` default `30`; staged CLI flags/defaults stay unchanged. |
| `difflet/cli/orchestrators/qwen_image.py` | Keep existing argv, download/local source resolution, subprocess ordering, `.pt` handoff, and output cleanup. Replace inline application construction in `_stage_text/_stage_generate/_stage_vae` with the common explicit-input builders; wrappers still own CLI compile/load/generate and file serialization. |
| `difflet/serving/cli/serve.py`, `difflet/serving/options.py` | Preserve the two tri-state values and raw model options through `ServeOptions`; add/validate `worker_heartbeat_interval > 0` in `_add_serve_flags`, `validate_serve_args`, and `options_from_args`; remove generic TeaCache pairing/schema and model-capability checks. |
| `difflet/models/qwen_image/{application,pipeline}.py` | Add mutually exclusive prevalidated calibration-object/path inputs and request-local TeaCache mode; keep the current pathname API for CLI callers. |
| `difflet/models/flux/{application,pipeline}.py` | Add the same prevalidated calibration-object and request-local mode boundary; keep existing CLI construction unchanged. |
| `difflet/serving/orchestrators/qwen_image.py` | Build/validate Qwen compile specs, load exact bindings, and call only common explicit-input builders. Compile runners consume `StageCompileInvocation.pinned_model_path`, `publish_target.staging_path`, and frozen options directly without CLI imports or revision/cache resolution. Implement typed runners plus initial/final payload conversion; adjacent runners consume the prior runner's logical output directly. Check abort inside long denoising loops. |
| `difflet/serving/orchestrators/flux.py` | Keep one opaque external stage; validate component topology/load order, compile into manager staging, load exact bound path, and check abort around pipeline plus its step callback. |
| `difflet/serving/engines/resident_worker.py` | Add `ResidentWorkerConfig.worker_heartbeat_interval`; pass it through `_ResidentWorkerProcess` to `_worker_main` and the dedicated heartbeat thread. Replace current run lock/pending/cancel/reply structures with engine-owned records, one FIFO control queue, terminal arbitration, and bundle-based startup/replacement. In the worker, construct `StagePipelineEngine` with `InProcessStageExecutor`, iterate the frozen ordered stages, and call one runner at a time. Shutdown-owned active callers receive the existing public `503 engine_draining`. |
| `difflet/serving/artifact_store.py` | Replace raw backend exception text with allowlisted public errors; retain details only in the protected debug log sink. |
| `difflet/serving/openai/{api_server,serving_chat}.py` | Keep public parsing/response ownership; map sanitized engine/artifact errors and never expose raw worker traceback or paths. |
| `tests/unit/serving/test_serve_cli.py`, `test_model_registry.py`, `test_resident_worker_engine.py`, `fake_worker.py` | Prove heartbeat parsing/default/rejection and exact options -> factory -> config -> process -> `_worker_main` propagation; cover shutdown as `engine_draining`, heartbeat, terminal races, and replacement. |
| `tests/unit/serving/test_qwen_common_orchestrator.py`, new artifact/stage contract tests, `tests/unit/cli/test_orchestrator_qwen.py` | Replace deleted preflight/path tests; cover extracted/opaque runner-factory invariants, bundle/pipeline-plan equality, final-output typing, stage traversal/abort, common builders with explicit paths, and compile runners forwarding the exact `StageCompileInvocation.pinned_model_path` without resolving it; retain unchanged CLI argv/file behavior. |
| `tests/unit/serving/test_flux_preflight.py` and artifact-manager tests | Cover opaque Flux admission/bound load, pre/post-call abort, TeaCache fallback, generation lock/reuse/race/crash/publication, manifest integrity, and pinned binding reload. |

## Delete Or Replace

All items below were introduced by the serving branch and have no compatibility
requirement outside it.

| Current code | Required action |
|---|---|
| `ServingArtifactPreparer.resolve_model_path()` | Remove public method; source resolution becomes an internal step of `prepare_runtime()` |
| `ServingArtifactPreparer.stage_specs()` | Remove; `PipelineDefinition` plus `RuntimePlan` are authoritative |
| `ServingArtifactPreparer.compile_plan(profile)` | Remove public path-based API; adapter uses `build_compile_plan(source, profile, config)` |
| `ServingArtifactPreparer.ensure_artifacts()` | Delete without wrapper; common artifact manager replaces it |
| `DiffletStageSpec` | Delete after all callers use `PipelineDefinition`/`RuntimePlan` |
| `DiffletCompileSpec(stage_id, artifact_path, required)` | Replace in place with path-free serving compile intent |
| `preflight_factory` / `load_preflight_factory` naming | Rename to `artifact_preparer_factory` / `load_artifact_preparer_factory`; no compatibility alias |
| Factory topology print loop over `stage_specs()` | Replace with resolved `RuntimePlan` logging |
| Worker load from model-computed cache paths | Replace with exact `ArtifactBinding.path` from the bundle |
| TeaCache fields in `ServingProfile` | Remove; raw inputs stay in `ServeOptions`, resolved values live in adapter config |
| `_run_lock`, `_admission_lock`, and `_pending` capacity accounting | Replace with transition-lock-owned bounded `RequestRecord` queue and one active record |
| Caller-coupled `_run_one()`/`run_generation()` reply loop | Replace with engine scheduler plus one terminal dispatcher; caller only waits/detaches from record future |
| Separate `_cancel_q` and `_QueueCancellationSource` | Delete; RUN/ABORT/SHUTDOWN use one bounded per-worker FIFO control queue |
| `CancellationSignal` and `WorkerRequestContext.with_timeout()` | Replace with request-specific `threading.Event`, fixed deadline, and `StageExecutionContext` adapter |
| Raw traceback in worker reply/normal logs | Remove; use allowlisted sanitized error envelopes and a protected debug sink only |
| Validator `validate(request, profile)` with model re-resolution | Replace with factory from frozen `ResolvedRuntimeBundle` and `validate(request)` using pinned tokenizer path |
| `cfg_parallel`/`sp_enabled` omission collapsed to `False` | Use serve-parser `bool | None` tri-state until registry/model adapter default resolution |
| Top-level `_validate_cfg_parallel`, `_validate_sp`, and `_validate_teacache` applied to `serve` | Remove `serve` from those shared staged-CLI validation branches; keep staged CLI command validation unchanged and pass raw serve values to the selected adapter |

### Qwen helper cleanup

Delete these serving-only path/marker/preflight functions from
`difflet/common/orchestrators/qwen_image.py` after their replacements land:

- `stage_compiled_dir(profile)` and `serving_stage_compiled_dir()`;
- `compile_plan()`, `missing_artifacts()`, and `ensure_artifacts()`;
- `artifact_ready()` and path-only serving marker validation;
- `_compile_serving_artifacts()`, `write_serving_markers()`, and
  `namespace_from_profile()`;
- `_serving_marker_payload()` and `_has_valid_serving_marker()`.

Keep `stage_compiled_dir_from_values()` because staged CLI uses it. Move/reuse
`_has_stage_artifact()`, `_has_neuron_artifact()`, and `_has_nxd_component()` behind
the Qwen adapter's `validate_compiled_artifact()` implementation.

### Serving adapter cleanup

- Delete Qwen/Flux `compile_plan(profile)` and `ensure_artifacts()` wrappers.
- Remove Qwen worker calls to `stage_compiled_dir()` and
  `serving_stage_compiled_dir()`.
- Rework Flux `_build_pipeline()` so compile receives parent-owned staging and load
  receives the finalized bound path; it must not select a cache path itself.
- Change Qwen/Flux request-validator factories to accept the frozen bundle; remove
  validator-local `resolve_model_path()` calls and load tokenizers from
  `runtime.source.pinned_model_path`.
- Rewrite tests whose only purpose is the deleted path/marker APIs. Preserve their
  payload-completeness cases under adapter/common-manager tests.

## Contract Changes

### Serving compile plan

Extend the existing serving-only type; do not add a second compile-spec class and
do not use it in staged CLI.

```python
@dataclass(frozen=True)
class DiffletCompileSpec:
    artifact_id: str
    component_id: str
    identity: CompileArtifactIdentity
    required: bool = True
```

The selected adapter exhaustively dispatches payload validation by
`component_id`. Unknown or cross-model components fail before cache lookup.
`CompileArtifactIdentity.schema_version` versions the validator/payload contract;
there is no string validator registry. Only the common artifact manager creates
`ArtifactPublishTarget` and finalized paths.

The common generation manifest is `difflet_generation_manifest.json`. Adapter
manifests such as Flux `manifest.json` remain hashed payload and are separately
validated; the common publisher never overwrites or excludes them.

### Runtime bundle

Add immutable `PipelineDefinition`, `RuntimePlan`, environment/allocation/stage
specs, `ResolvedModelSource`, artifact identity/binding/set, model-specific adapter
config, frozen path-free compile specs, and `ResolvedRuntimeBundle` contracts from
`architecture.md`.

Initial and replacement workers receive the same bundle and:

- load only `ArtifactBinding.path` with `skip_compile=True`;
- verify manifest and payload digest;
- never rebuild compile plans, select a cache path, resolve a revision, compile, or
  repair the generation.

The bundle retains the exact compile specs used during preparation. Validate unique
spec IDs, one exact spec per binding, byte-equal identities, one binding for every
required spec, and stage references through the same pair. Initial/replacement
workers pass that frozen spec to adapter payload validation and never call
`build_compile_plan()`.

`RequestRecord` stores the normalized request, receive time, queue deadline,
overall request deadline, worker identity/state, delivery state, and result future.
Queued expiry checks the overall request deadline first: equal queue/request
deadlines return `504 request_timeout`; an earlier queue deadline returns
`429 queue_timeout`.

### Model adapter

Qwen and Flux implement:

```python
resolve_adapter_config(options, source, profile)
build_compile_plan(source, profile, config)
compile(source, profile, config, spec, target)
validate_compiled_artifact(spec, artifact_root)
```

`prepare_runtime()` is the only public artifact-preparation lifecycle operation.
Common code owns identity lookup, lock, staging, inventory/digest, manifest, atomic
publication, and direct binding.

## Serve Startup Options

- Keep raw model-specific values in `ServeOptions`; generic profile construction
  resolves model/source, TP, CP, world, width, height, dtype, and output modality.
- Remove generic TeaCache pairing/file/schema checks from `validate_serve_args()`
  and `build_serving_profile()`.
- Do not run staged-CLI `_validate_cfg_parallel`, `_validate_sp`, or
  `_validate_teacache` against `serve`; model capabilities and optional fallback are
  resolved only after selecting the serving adapter.
- Qwen/Flux adapters independently parse optional calibration. Missing, unreadable,
  malformed, unsupported, or profile-mismatched calibration freezes a baseline
  config and sanitized reason instead of blocking startup.
- P0 cadence/online-delta options reach the selected adapter and are rejected there.
- Preserve `cfg_parallel`/`sp_enabled` omission as `None` through serve argparse and
  `ServeOptions`; explicit enable/disable overrides registry defaults.
- Every supported startup value enters the core profile, frozen adapter config, or
  engine config. No parsed option is silently ignored.

## TeaCache Runtime And Compile

- Calibration/controller values are runtime-only. Speedup, coefficients, threshold,
  and calibration metadata do not alter main/probe NEFF identity.
- Current adaptive mode sets `requires_teacache_probe=True`; only normalized
  `teacache_probe_enabled` changes affected artifact identity.
- Qwen/Flux application construction adds a prevalidated
  `TeaCacheCalibration | None` input mutually exclusive with the existing pathname.
  Staged CLI keeps the pathname. Serving workers reconstruct the object from frozen
  config and create a process-local controller without reopening the source file.
- If request steps differ from calibration `num_steps`, that request runs baseline
  TeaCache-off inference and logs typed reason `step_mismatch`. It is not rejected.
- Add request-local `TeaCacheRequestMode(use_teacache, fallback_reason)` from the
  orchestrator into Qwen/Flux denoising calls. Loops use local controller/probe
  references and never mutate resident bindings; baseline requests never invoke the
  probe. The value object rejects enabled mode combined with a fallback reason;
  `(False, None)` remains the normal baseline-profile mode.
- Enabled-profile startup and replacement smoke use calibration `num_steps`;
  baseline smoke may use a reduced step count.

## Qwen Stage Extraction

- Register stable serving stage IDs `text`, `generate`, and `vae`.
- Add profile-relative `RuntimePlan`; resident P0 uses one compatible world across
  all co-loaded applications, while staged CLI topology remains unchanged.
- Use strict compile-only `StageCompileInvocation` for serving Qwen compile children.
- Add process-local typed `StageRunner` adapters, opaque logical payloads,
  `StageInvocation`/`StageExecutionResult`, `StageExecutor`, and in-memory stage
  handoff through `InProcessStageExecutor`.
- Require every concrete logical payload, including all Qwen payloads and
  both Flux payloads, to inherit the nominal `StagePayload` marker. Make invocation,
  result, and runner contracts generic over their concrete payload types; erase
  those arguments only through validated `ErasedStageRunner` wrappers in the
  heterogeneous runner registry; do not use `Any` in the engine contract.
- Fail closed at three levels: executor nominal input/output checks, validated
  runner exact input/output checks, and adapter exact final-payload validation.
  Add focused tests for invalid initial payload, wrong adjacent payload type,
  non-`StagePayload` runner output, wrong final payload type, and subclass instances
  at exact input/output/final boundaries. Exact checks combine `isinstance` for
  generic narrowing with `type(value) is expected_type` for subclass rejection.
- Expose erased-runner input/output type metadata as read-only properties backed by
  private generic fields; mutable protocol attributes and unchecked casts are not
  permitted.
- Require `ValidatedStageRunner.shutdown()` to delegate exactly once to its inner
  runner. Add a structural typing check and a shutdown-delegation unit test.
- Treat pipeline/stage handles as adapter-owned only during runner construction and
  partial-failure cleanup. Successful runner return transfers them atomically and
  exclusively to executor lifecycle; adapter shutdown must not touch transferred
  handles.
- Define `StageRole`, exact `StageLoadContext(runtime, stage, artifact)` binding, and
  concrete text embedding/mask, packed-latent, and final-output value objects.
- Keep ordered traversal, stage lifecycle logs, and before/after cancellation
  checkpoints in the generic worker-side stage engine. Keep concrete payload
  validation, initial/final conversion, reset, smoke inputs/output validation, and
  async runner construction/load in the Qwen serving adapter. Transfer loaded
  runner ownership to the executor; async executor shutdown is the sole
  runner-shutdown path, followed by async adapter-state shutdown.
- Require `create_loaded_runners()` to return an insertion-ordered mapping whose
  order is both runner creation order and pipeline stage order; partial-failure and
  executor shutdown attempt every runner in strict reverse order, retain individual
  failures, and raise only after exhaustive cleanup. Adapter-state cleanup always
  runs afterward; any aggregated cleanup failure makes the worker process-unsafe.
- Route every worker exit after adapter construction—including load/smoke failure
  before readiness—through one idempotent cleanup path: clean a nullable executor
  when ownership transferred, then always clean adapter-owned state.

## Flux Serving Adapter

- Keep external stage ID `pipeline`; derive component topology and load order from
  actual `MultiComponentApplication.components()` metadata.
- Compile from the parent-pinned model source into the common manager's staging
  target.
- Load with logical model ID plus pinned model path/source ID and exact
  `compiled_path_override`; reject partial binding tuples.
- Preserve existing Flux CLI cache behavior when no serving overrides are supplied.

## Engine Changes

- Construct and pass one `ResolvedRuntimeBundle` instead of `ServingProfile` plus
  independently selected paths.
- Preserve the engine-owned request record, bounded control/status queues,
  cooperative abort, finalization/reset arbitration, process-unsafe replacement,
  and ordered shutdown contracts in `engine.md`.
- Add structured startup/stage/progress logs and a dedicated heartbeat thread.
- Add a generic worker-side stage loop over `PipelineDefinition.stages` and select
  `InProcessStageExecutor` once at worker construction. The P0 loop contains no
  `single_process` branches, stage queues, placement, or transport selection.
- Pass each non-final `StageExecutionResult.output` unchanged as the next
  `StageInvocation.input`; generic engine code never branches on Qwen payload types.
- Reserve `StageExecutor` as the future scheduling boundary. IPC/shared-memory
  handles, payload ownership, cleanup, and per-stage worker pools remain deferred.
- Readiness requires bound artifact validation, resident load, and real end-to-end
  smoke. Worker replacement loads and smokes the same bundle.
- Replace current caller-owned run/reply/cancellation structures with the
  engine-owned record/scheduler/terminal-dispatch contracts above.

## Implementation Order

1. **Contracts and cleanup:** add immutable definitions, extend
   `DiffletCompileSpec`, delete `DiffletStageSpec` and obsolete public preflight APIs,
   and add pure validation tests.
2. **Artifact manager and bundle:** implement immutable generation publication,
   `prepare_runtime()`, pinned source/config/spec flow, and direct binding.
3. **Qwen serving stages:** compile transport, runners, resident topology, and smoke.
4. **Flux serving binding:** adapter compile/validation and exact bound-path load.
5. **TeaCache handoff:** adapter parsing/fallback, prevalidated object input,
   request-step fallback, and calibrated smoke.
6. **Engine lifecycle:** bundle IPC, logging/heartbeat, abort/reset/recovery/shutdown.
7. **Delete obsolete tests/helpers:** remove every old caller and verify no stale
   interface remains.

Do not start a later step until focused unit tests and the relevant Trainium smoke
for the previous step pass.

## Verification

- `rg` finds no runtime callers of `ensure_artifacts`, `compile_plan(profile)`,
  `DiffletStageSpec`, serving path selectors, or old marker APIs.
- `rg` finds no `_run_lock`, `_pending`, `_cancel_q`, `_QueueCancellationSource`,
  `CancellationSignal`, caller-coupled reply loop, or normal-path raw traceback.
- Existing staged CLI compile/generate and artifact paths remain unchanged.
- Qwen resident text/generate/vae co-load and three repeated requests complete
  without reload.
- Flux CLI and resident serving both generate valid images; resident replacement
  loads the original bound generation.
- Initial/replacement adapter validation receives the original frozen compile spec;
  instrument `build_compile_plan()` and prove it is never called in a worker.
- Toggling only probe inclusion changes artifact identity. Runtime-only TeaCache
  changes reuse the same NEFF identity.
- Delete calibration after preflight; initial/replacement workers build the same
  controller. Run matching -> mismatched -> matching plus abort/error variants;
  baseline calls never invoke the probe or leak controller state, and calibrated
  smoke uses matching steps.
- Artifact corruption, manifest mismatch, download/preflight revision pinning,
  partial Flux overrides, environment mismatch, stale worker events,
  cancellation/reset races, and shutdown send failures all behave according to the
  architecture/engine contracts. Serving does not perform an online latest-revision
  check.
- Flux publication preserves and hashes adapter `manifest.json` alongside distinct
  `difflet_generation_manifest.json`; tampering with either fails reuse/load.
- Startup, busy, idle, recovery, and shutdown logs/heartbeats are structured,
  sanitized, non-blocking, and correctly clear request/stage state.
- Queue/request deadline ties return `request_timeout`; an earlier queue deadline
  returns `queue_timeout`. Every startup option preserves omission and propagates
  to core profile, selected adapter, or engine config. Public API/error matrix tests
  cover every documented response code.
- Incomplete TeaCache pairs and cadence/online-delta reach the selected serving
  adapter, while existing staged CLI commands retain their current shared validation
  errors and argument behavior.
- `git diff --check` and focused unit/integration tests pass before Trainium runs.
