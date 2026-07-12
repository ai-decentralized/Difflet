# Difflet Serving Design Review

## Status

- Status: complete; latest fresh reviewer converged in Round 46
- Current reviewer: `019f5141-9023-7662-b80a-9beeae711629`
- Model: `gpt-5.6-sol`, high reasoning
- Date: 2026-07-11
- Current blocking/material findings: none

Review scope:

- `architecture.md`
- `engine.md`
- `chat_completions_contract.md`
- `implementation_changes.md`

## Current Revision

### Fresh R45 revision

- **R45-M1 rejected by user:** serving does not inventory Hugging Face source bytes.
  Download/preflight resolves the requested revision to a commit SHA; the HF cache
  is trusted download-layer state. Serving pins that path/version and never checks
  latest or auto-downloads. Model update is a new download/preflight plus restart.
- **R45-M2:** define exact per-identity `.publish.lock`, numeric generation
  allocation, newest-valid reuse, `NEVER/AUTO/FORCE` behavior, staging confinement,
  fsync/rename order, crash recovery, and offline finalized-generation cleanup.
- **R45-M3:** add an exact file/symbol migration inventory, including serve-only
  tri-state spellings and unchanged staged-CLI boundaries.
- **R45-M4:** restore the complete `common`/CLI/pipeline/models/serving hierarchy;
  `common/` remains the shared layer.
- **R45-M5 (main-agent/user finding):** Qwen serving orchestrator explicitly loops
  over `PipelineDefinition.stages`, routes typed outputs, and checks abort before and
  after each stage. The engine dispatches one request and never interprets stage IDs.

### Fresh R45 follow-up

- **R45FU-B1 blocking, revised:** add `pipeline_definition` to the frozen bundle and
  make the final-stage output return directly instead of assigning it to
  `StageInputs`.
- **R45FU-M1 material, revised:** distinguish extracted and opaque pipeline stage
  definitions so Flux does not require a fake `StageRunner`, while both kinds retain
  runtime allocation/artifact binding.
- **R45FU-M2 material, revised:** complete the file/symbol inventory for heartbeat
  propagation and extraction of reusable Qwen stage builders from CLI ownership.
- **R45FU-M3 material, revised:** map shutdown-owned active requests to the existing
  public `503 engine_draining` contract instead of undefined `engine_shutdown`.
- **R45FU-M4 material, revised:** add the missing Flux post-pipeline abort check to
  the worker-layout pseudocode.

Prior status: R45-M1 remains intentionally rejected and internally consistent;
R45-M2 through R45-M4 are resolved.

### Round 46 follow-up

R45FU-B1 and R45FU-M1 through M4 are resolved. HF source inventory remains
intentionally rejected and internally consistent.

- **R46-M1 material, revised:** the migration inventory said
  `StageCompileInvocation.model_path`, but the contract field is
  `pinned_model_path`. Use the declared field and add a focused builder-path test.
- **R46-M2 material, revised:** remove unused `resource_resolver` and
  `artifact_resolver` from P0 `StageDefinition`. `RuntimePlan` already owns resource
  resolution, and adapter compile specs/common artifact manager already own artifact
  resolution; undefined resolver factories would create duplicate authorities.

Round 46 follow-up confirmed R46-M1/M2 resolved with no new blocking or material
findings. The consolidated design is implementable at the reviewed contract level.

### Fresh R41 revision

- **R41-M1:** reserve `difflet_generation_manifest.json`; adapter manifests such as
  Flux `manifest.json` remain validated, hashed payload.
- **R41-M2:** add request payload and queue deadline to `RequestRecord`; overall
  request timeout wins equal deadlines, otherwise earlier queue timeout is 429.
- **R41-M3:** construct request validators from `ResolvedRuntimeBundle`; tokenizers
  load only from the pinned model path.
- **R41-M4:** define `StageRole`, exact load context/artifact binding, and concrete
  Qwen text/generate/vae process-local input/output values.
- **R41-M5:** expand the branch-relative delta list to remove current run-lock,
  pending/cancellation/reply/traceback structures and require serve boolean tri-state
  plus complete propagation/error tests.

### Fresh R42 follow-up

- **R42-M1:** explicitly remove `serve` from top-level staged-CLI
  `_validate_cfg_parallel/_validate_sp/_validate_teacache` branches while preserving
  existing staged CLI validation. Raw serve values reach the selected adapter.
- **R42-M2:** replace impossible pre-compile manifest checking with explicit reuse,
  cache-miss, and initial/replacement-load validation order.

### Fresh R43 follow-up

- **R43-M1:** resolved. Retain exact immutable path-free `compile_specs` in
  `ResolvedRuntimeBundle`. Spec IDs/identities bind one-to-one with artifacts;
  initial/replacement workers retrieve the frozen spec for adapter validation and
  never rebuild the plan.

### Documentation consolidation

- Status: resolved; Round 39 and fresh Round 41 confirmed no contract loss or
  competing authority
- Stage/runtime and serve-option contracts now live in the original serving
  `architecture.md`; engine and public API remain separate by responsibility.
- `implementation_changes.md` lists only code additions, replacements, deletions,
  implementation order, and verification relative to the current branch.
- The standalone `stage_runtime_refactor` document set is removed after link/task
  migration; this review ledger moved into the serving directory.
- Round 39 confirmed consolidation preserved all prior contracts and introduced no
  competing authority.

## Latest User Decisions

- Generic serving validation covers resolved model/source, TP, CP, world, width,
  height, allocations, and artifact bindings. Raw model-specific options remain in
  `ServeOptions` until the selected adapter resolves them.
- Qwen/Flux adapters own extra model-specific options. Generic bundle/engine does
  not parse TeaCache calibration fields.
- Optional TeaCache input is resolved once by the adapter. Missing, unreadable,
  malformed, or unsupported model-specific fields disable TeaCache with an
  allowlisted reason; serving continues with baseline inference.
- Absent calibration `model` may use the already resolved adapter model. TP/CP/
  world/W/H always come from the normalized profile, never calibration JSON.
- Initial and replacement workers consume the same frozen adapter config and never
  reopen its source path.
- TeaCache speedup/calibration/controller values do not alter main/probe NEFF
  identity. Only optional fused-probe inclusion changes the affected artifact
  compile identity.
- `prepare_runtime()` is the serving compile boundary. Model adapters own compile
  specs/invocation; common code owns publication; worker orchestrators only load.
- Existing `DiffletCompileSpec` is branch-local serving infrastructure, not a new
  CLI abstraction. It is extended in place; no second compile-spec type is added.
- Branch-local `ensure_artifacts()` and `compile_plan(profile)` are deleted during
  bundle migration rather than retained as compatibility APIs.
- Serving passes a prevalidated calibration object into Qwen/Flux applications;
  existing staged CLI pathname arguments and behavior remain unchanged.
- Runtime/stage/serve-option design has one authority in `difflet_serving`; obsolete
  standalone runtime documents and branch-only preflight APIs are deleted.
- Current CLI argparse, source resolution, subprocess argv/dispatch, stage order,
  calibration pathname handling, and file handoff remain unchanged.
- Serving artifact compile alone uses `StageCompileInvocation`; full CLI typed
  invocation and CLI StageRunner migration are deferred.
- Resident P0 accepts only commit-addressed HF snapshots; caller-provided local
  model directories are rejected.
- Flux remains one opaque pipeline; its internal component topology stays owned by
  `NeuronFluxApplication`.
- Abort is cooperative at safe points; no in-flight Neuron graph preemption.
- Clean request reset preserves resident HBM. Process-unsafe state blocks dispatch
  until worker replacement.
- Each worker owns one bounded control queue. Parent `put_nowait` and retirement are
  serialized by one transition lock; no outbox/sender task exists.
- Hugging Face model freshness belongs to download/preflight, not serving runtime.
  Serving trusts the commit-addressed local snapshot selected by that layer.

## Resolved Review Summary

### R32-R34

- Shutdown ABORT/SHUTDOWN send failures detach the lease as process-unsafe and run
  forced idempotent cleanup before resolving the shared shutdown future.
- `CompileArtifactIdentity` stores canonical cache-input JSON bytes plus digest,
  not a mutable mapping.
- Full CLI typed transport was removed from scope; serving compile transport is
  compile-only.
- Worker examples uniformly use `context.throw_if_aborted()`.
- The earlier generic `RuntimeFileSet`/runtime-file publication proposal was
  superseded by adapter-owned frozen config per the user decision.
- R34 adapter-config ordering and review-ledger cleanup were confirmed resolved in
  Round 35.

### R35

- Existing immutable `ServeOptions` is the raw model-option transport; generic
  profile construction does not interpret TeaCache.
- Concrete frozen Qwen/Flux configs, baseline fallback, profile identity, manifest
  consistency, and process-local kwargs derivation were confirmed resolved in
  Round 36.

### R36

- Separate Qwen/Flux calibration resolvers now enforce strict finite/range/mode
  validity and map every unusable optional calibration to baseline before identity.
- P0 cadence/online-delta options reach and are rejected by the selected adapter,
  never by generic parsing/profile code. Round 37 confirmed these contracts.

### R37

- Round 38 confirmed the sole `prepare_runtime()` lifecycle, path-free serving
  compile spec, common path/publication ownership, and exact bound-path worker load.
- Round 38 confirmed the prevalidated Qwen/Flux calibration object handoff and
  unchanged staged CLI pathname behavior.

### R38

- Removed open `validator_id`; selected adapters exhaustively validate components
  and payload schema is versioned by `CompileArtifactIdentity`.
- Defined request-step baseline policy and calibrated smoke. Round 39 confirmed the
  policy but required the request-local execution mechanism in the current revision.

### R25-R31

- Flux serving compile/load injects the parent-pinned model source and exact
  immutable artifact generation while preserving logical model identity.
- Pipeline override API permits only legacy no-override mode or fully bound serving
  mode; partial tuples fail closed.
- RUN/ABORT/SHUTDOWN share a bounded per-worker FIFO queue. Worker state/event is
  installed before reset/model work, so early abort is retained.
- Completion, abort, reset, and shutdown delivery are lock-linearized. Late abort
  after frozen completion is stale.
- Shutdown owns epoch, active request, recovery/provisional cleanup, process/queue
  retirement, and model shutdown ordering.
- Documentation compression preserved canonical authority boundaries.

### R17-R24

- Compiled artifacts publish as locked immutable generations with canonical payload
  inventory/digest, fsync, atomic rename, direct generation binding, and offline-only
  finalized GC.
- Initial/replacement workers use one pinned `ResolvedRuntimeBundle`; mutable HF
  revisions are never resolved again in a worker.
- Engine owns every `RequestRecord`; caller timeout/disconnect only detaches.
- Request-specific `threading.Event`, two-phase `finalizing`, complete model reset,
  and deterministic terminal/shutdown arbitration replaced the earlier ticket,
  lease, receipt, and tombstone proposals.

### R1-R16

- Qwen serving stages are `text -> generate -> vae`; profile-specific runtime plans
  separate allocation from TP/CP/world topology.
- Flux heterogeneous components remain opaque to the generic stage schema.
- Startup-option default ownership, environment/core admission, artifact identity,
  provisional-worker cleanup, structured logs, heartbeat, and end-to-end readiness
  smoke were defined.

## Rejected Or Superseded

- Generic Flux component stages: rejected; they duplicate model-owned topology.
- Caller-owned cleanup tickets/leases/receipts: superseded by engine-owned records.
- Generic calibration schema and `RuntimeFileSet`: rejected; model adapters own
  optional calibration interpretation and safe fallback.
- Full CLI `StageInvocation` and CLI StageRunner conversion: deferred to avoid a
  large CLI behavior change.
- Generic numerical HBM estimator: deferred until model-owned estimates exist.
- Forced multi-process/MPMD serving: deferred beyond P0.
- Source-byte hashing/inventory inside serving: rejected; it duplicates the trusted
  download layer and is not needed to select or update a pinned HF revision.

## Review History

- Same-reviewer convergence reached in Round 24.
- First fresh reviewer found and closed R25-R31 lifecycle/artifact gaps.
- Additional fresh reviewer found R32-R34 scope/config ordering gaps.
- Its Round 35 follow-up found and prompted the raw-option boundary and concrete
  adapter-config contracts.
- Its Round 36 follow-up found compile-helper input and calibration-validity gaps;
  the user selected the minimal probe-flag compile design above.
- Its Round 37 follow-up confirmed calibration validity and found stale engine/spec
  and serving-to-application calibration handoff gaps addressed in the current
  revision.
- Its Round 38 follow-up confirmed those fixes and found request-step/smoke and open
  validator-ID issues addressed during document consolidation.
- Round 39 confirmed consolidation and all prior contracts; only request-local
  TeaCache fallback execution remained material.
- Round 40 confirmed that finding resolved with no blocking/material issues.
- Fresh reviewer Round 41 confirmed consolidation but found five manifest, queue,
  validator-source, stage-boundary, and branch-delta gaps revised above.
- Fresh Round 42 confirmed R41-M1 through M4 and most of M5; it found the two
  top-level validator and artifact-timing gaps revised above.
- Fresh Round 43 confirmed both R42 findings and found the missing worker-visible
  frozen compile spec revised above.
- Fresh Round 44 confirmed R43 resolved with no blocking/material findings.
- Fresh Round 45 found generation publication, migration-inventory, and hierarchy
  specificity gaps. Those are revised above. Its proposed HF source-byte inventory
  was rejected by user decision, and the user additionally required an explicit
  orchestrator-owned stage loop.
- No design implementation code was edited by reviewer agents.

## Next Action

Stop the review loop. Implementation may proceed in the order defined by
`implementation_changes.md`; Trainium/runtime validation remains required per phase.
