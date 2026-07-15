# Stage Engine Design Review

## Status

Complete. Final fresh independent sign-off found no blocking or material issues in
the nominal payload typing revision.

## Round 1

- Reviewer: independent available subagent (requested `gpt-5.6-sol high` model
  override unavailable in this environment)
- Review type: read-only design and cross-document consistency review

### Blocking

None.

### Material

1. `StageExecutionContext` conflicts across authoritative documents. Use one
   transport-neutral protocol and keep the request-specific cancellation event
   inside `WorkerRequestContext`; do not reintroduce deleted `CancellationSignal`.
2. The later "Adding A Model" guidance still prescribes the old
   `ServingModelOrchestrator.generate()` boundary. Update it to
   `ServingStageAdapter` plus the generic `StagePipelineEngine`.
3. Runner/executor lifecycle ownership is ambiguous. Separate compile-time runner
   behavior from loaded runtime `StageRunner` and select exactly one shutdown owner.
4. Payload-handle wording leaks transport representation into the logical runner
   API. Keep invocation/result payloads logical; hydrate/materialize wire handles
   entirely inside future executor/transport implementations.

### Optional

1. Remove the duplicate `stage` argument from `StageExecutor.execute(stage,
   invocation)` and use `invocation.stage` as the sole identity authority.
2. Normalize remaining "orchestrator" terminology where nearby sections now mean
   model adapter or worker-side stage engine.

## Main-Agent Recommendation

Accept all four material findings and both optional cleanup findings. They are
contract-consistency fixes and do not broaden P0 scope. Then reuse the same reviewer
to verify resolution, followed by a fresh independent pass if no material findings
remain.

## Changes Already Drafted Before Review

- Moved ordered traversal into the generic worker-side stage engine.
- Defined opaque logical payload, invocation, result, runner, executor, and adapter
  seams.
- Restricted P0 to sequential `InProcessStageExecutor` execution.
- Deferred per-stage workers, IPC/shared memory, placement, DAG scheduling, and
  overlap.
- Required readiness smoke to use the same generic traversal path.

## Next Action

Run same-reviewer round 2, then perform a fresh independent missed-issue pass after
the same reviewer reports no blocking or material findings.

## User Decision And Round 1 Revision

- User accepted all recommended findings.
- Unified `StageExecutionContext` as a transport-neutral protocol implemented by
  worker context; removed `CancellationSignal` from the proposed contract.
- Updated Adding A Model and staged-model guidance to the adapter/stage-engine
  boundary.
- Split compile-time `StageCompiler` from loaded runtime `StageRunner`.
- Made executor shutdown the sole loaded-runner shutdown path; adapter shutdown
  clears only adapter-owned state.
- Kept logical payloads in invocation/result and confined future wire handles to
  executor/transport internals.
- Removed the duplicate executor `stage` argument and normalized nearby legacy
  terminology.

## Round 2

- Reviewer: same independent available subagent
- Prior findings resolved: context contract, Adding A Model boundary,
  compile/runtime protocol split, logical/wire payload separation, duplicate stage
  argument

### Blocking

None.

### Material

1. The authoritative shutdown section still describes only old orchestrator
   shutdown and omits the required ordering: await executor shutdown first as the
   sole runner closure, then clean adapter-owned state. Rewrite shutdown ownership
   and failure behavior to match the generic contract.

### Optional

1. Normalize remaining operational uses of "orchestrator" where they mean stage
   adapter, stage engine, or executor.
2. Require `create_loaded_runners()` to clean partial runners before raising;
   ownership transfers atomically only after successful return.

## Main-Agent Round 2 Recommendation

Accept the material shutdown fix and both optional clarifications. They close the
same lifecycle boundary already accepted in round 1 and do not broaden P0 scope.

## User Decision And Round 2 Revision

- User accepted the shutdown fix and both optional clarifications.
- Shutdown now awaits executor shutdown as the sole runner closure, then clears
  adapter-owned state; failure makes the worker process-unsafe.
- `create_loaded_runners()` now cleans partial construction/load before raising and
  transfers ownership atomically only on complete successful return.
- Operational lifecycle, IPC, reset/finalize, model mapping, and implementation
  checklist terminology now distinguishes stage engine, stage adapter, and
  executor.

## Round 3

- Reviewer: same independent available subagent
- Round 2 shutdown, partial-failure ownership intent, and terminology findings are
  otherwise resolved.

### Blocking

None.

### Material

1. `StageRunner.shutdown()` is async but `create_loaded_runners()` is synchronous,
   so the adapter cannot implement the required awaited cleanup after a partial
   load failure. Make `create_loaded_runners()` async everywhere and explicitly
   await reverse-order cleanup before re-raising.

### Optional

None.

## Main-Agent Round 3 Recommendation

Accept the async signature correction. It is the smallest implementation-feasible
form of the already accepted partial-load cleanup contract.

## User Decision And Round 3 Revision

- User requested async lifecycle consistency.
- `create_loaded_runners()` is async in every contract and call site.
- Partial-load cleanup awaits runner shutdown in reverse creation order.
- Adapter-state shutdown is also async and remains ordered after executor shutdown.

## Round 4 — Same-Reviewer Convergence

- No blocking, material, or optional findings remain from the original reviewer.
- Async construction, partial cleanup, ownership transfer, and shutdown ordering
  are consistent across all reviewed documents.

## Round 5 — Fresh Independent Review

- Reviewer: new independent available subagent

### Blocking

None.

### Material

1. `architecture.md` still annotates invocation/result and adapter boundaries with
   raw `Any`, while the other authoritative documents define the named logical
   `StagePayload` seam. Add `StagePayload = Any` and use the alias consistently.
2. One Qwen worker-layout pseudocode call omits `await` on async
   `executor.execute(invocation)`, so `result.output` would operate on a coroutine.

### Optional

1. Reverse-order shutdown currently relies on generic mapping iteration. Require
   the adapter's returned mapping to preserve runner creation order, or return an
   explicitly ordered runner collection.

## Main-Agent Round 5 Recommendation

Accept both material fixes and the ordering clarification. They are localized
contract corrections with no P0 scope change.

## User Decision And Round 5 Revision

- User accepted both material fixes and made reverse-order runner shutdown a
  mandatory contract.
- Added and consistently used the named logical `StagePayload` alias.
- Added the missing `await` to executor traversal pseudocode.
- Required the runner mapping insertion order to equal creation order and pipeline
  stage order; executor and partial-failure cleanup strictly reverse that order.

## Round 6 — Reviewer Convergence

- Previous `StagePayload`, missing `await`, and mandatory ordering findings are
  resolved.
- No new blocking or material issues from the converged reviewer.

## Round 7 — Final Independent Audit

### Blocking

None.

### Material

1. Strict reverse shutdown currently stops at the first runner exception in the
   concrete pseudocode, and adapter cleanup may be skipped if executor shutdown
   raises. Require exhaustive best-effort reverse cleanup: catch/retain each error,
   continue through every runner, always await adapter cleanup afterward, aggregate
   failures, and mark the worker process-unsafe if any cleanup failed.

### Optional

None.

## Main-Agent Round 7 Recommendation

Accept the exhaustive cleanup clarification. It makes the user's mandatory strict
reverse-order guarantee hold even when an individual runner shutdown fails.

## User Decision And Round 7 Revision

- User accepted exhaustive reverse cleanup.
- Executor and partial-load cleanup now attempt every runner in strict reverse
  order, retain individual failures, and raise an aggregate error only afterward.
- Adapter-state cleanup always runs after executor cleanup through a
  finally-equivalent path.
- Failures across both cleanup phases are aggregated and make the worker
  process-unsafe.

## Round 8 — Cleanup Convergence

- Prior exhaustive-cleanup finding is resolved.
- No new blocking/material issues or P0 scope leakage in the follow-up reviewer.

## Round 9 — Release-Check Independent Audit

### Blocking

None.

### Material

1. The Flux worker-layout pseudocode still exposes old
   `FluxServingOrchestrator.load()/generate()` execution instead of one opaque
   `FluxPipelineRunner` returned by `FluxServingStageAdapter` and invoked through
   `StagePipelineEngine`.
2. Cleanup is fully defined after active request shutdown but not for every startup
   exit after adapter construction. Load or smoke failure must also run nullable
   executor cleanup when ownership transferred, then always await adapter cleanup,
   aggregating both phases.

### Optional

None.

## Main-Agent Round 9 Recommendation

Accept both fixes. They align old Flux pseudocode and startup failure handling with
the already accepted generic stage/lifecycle contracts, without changing P0 scope.

## User Decision And Round 9 Revision

- User accepted both release-check fixes.
- Flux now exposes one `FluxPipelineRunner` from
  `FluxServingStageAdapter.create_loaded_runners()` and uses the same generic stage
  engine for smoke and generation; the old adapter-owned `generate()` path is gone
  from the design.
- Every exit after adapter construction, including load/smoke failure before
  readiness, enters one idempotent cleanup path with nullable executor cleanup
  followed by mandatory adapter cleanup and aggregated failure handling.

## Round 10 — Release-Check Follow-Up

- Startup/load/smoke cleanup finding resolved.
- Reviewer found one incomplete part of the already accepted Flux fix: a
  specialized one-stage engine pseudocode path remained.
- Removed that path. Flux now uses only the previously defined generic loop, and
  architecture explicitly shows
  `StagePipelineEngine -> InProcessStageExecutor -> FluxPipelineRunner -> pipe`.

## Round 11 — Final Independent Sign-Off

- New independent reviewer found no blocking or material correctness issues.
- Confirmed one generic ordered loop for Qwen and Flux, sequential process-local P0
  execution, logical payload isolation from future wire transport, async ownership
  transfer, and exhaustive cleanup on every post-adapter-construction exit.
- `git diff --check` passed.

## Final Status

The prior stage-engine review is complete. A new nominal-payload typing revision is
active below.

## Nominal Payload Typing Revision

### Draft

- Replaced `StagePayload = Any` with a nominal marker base class.
- Made `StageInvocation`, `StageExecutionResult`, and `StageRunner` generic over
  concrete payload types.
- Required Qwen initial/text/latent/final and Flux initial/final payloads to inherit
  `StagePayload`.
- Kept `Any` only at the heterogeneous registry/executor dispatch boundary, with
  mandatory nominal and concrete runtime validation.

### Status

Round 3 shutdown-delegation finding accepted and revised; convergence check in
progress.

### Review Round 1 — Available Independent Reviewer

The requested skill-specific model override is unavailable; the user accepted the
available reviewer fallback.

#### Blocking

None.

#### Material

1. The design requires nominal `StagePayload` values but does not assign generic
   runtime validation to the executor. Because registry dispatch intentionally uses
   `Any`, `InProcessStageExecutor` must validate invocation input and result output
   with `isinstance(..., StagePayload)` around runner dispatch. Concrete runners
   still validate their exact input class, and adapter finalization validates the
   exact final payload. Add invalid initial/adjacent/output/final payload tests.

#### Optional

1. `implementation_changes.md` says all Qwen payloads and `FluxFinalPayload`; make
   this explicitly "both Flux payloads" so `FluxInitialPayload` is included.

#### Feasibility Result

- The invariant generic Protocol/dataclass shape passes a strict reduced mypy
  reproduction.
- Frozen slotted dataclass inheritance from the empty slotted marker base is
  feasible.
- Future transport handles remain private and must not inherit `StagePayload`.

#### Main-Agent Recommendation

Accept the material validation boundary and optional wording correction. They make
the nominal type contract fail closed without changing the P0 execution model.

### User Decision And Revision

- User required removal of `Any` from the heterogeneous registry/engine contract
  and accepted all runtime validation and negative-test requirements.
- Added nominal `ErasedStageRunner` for registry/executor dispatch.
- Added generic `ValidatedStageRunner` wrapper with exact input/output runtime
  validation and a single post-`isinstance` internal `cast`.
- Added executor nominal input/output validation and exact adapter final-payload
  validation requirement.
- Required tests for invalid initial payload, wrong adjacent type, non-marker
  output, and wrong final payload.
- Corrected implementation wording to include both Flux payloads.

### Review Round 2 — Same Reviewer

#### Resolved

- Generic runtime validation ownership.
- Removal of `Any` from registry/engine contracts.
- Both Flux payloads included.
- Required negative tests and exact final validation.

#### Material

1. `ErasedStageRunner.input_type/output_type` are writable protocol attributes,
   while the generic wrapper exposes narrower types. Mutable protocol attributes
   are invariant, so the wrapper does not satisfy the erased protocol under strict
   mypy, and post-construction mutation could defeat validation. Make them read-only
   properties backed by private `_input_type/_output_type` fields.
2. After `isinstance(invocation.input, self._input_type)`, strict mypy already
   narrows the value; the explicit `cast` is redundant. Remove it.

#### Main-Agent Recommendation

Accept both corrections. The read-only property form makes the erased wrapper
structurally valid and prevents validation metadata mutation.

### User Decision And Round 2 Revision

- User accepted both corrections.
- Erased runner type metadata is now exposed through read-only properties backed
  by private generic fields.
- Removed the redundant cast after exact `isinstance` narrowing.

### Review Round 3 — Same Reviewer

#### Resolved

- Read-only erased-protocol properties now satisfy strict typing.
- Redundant cast removed; generic narrowing passes the reduced strict check.

#### Material

1. `ValidatedStageRunner` does not implement the `shutdown()` required by
   `ErasedStageRunner`, so it still fails structural typing and executor shutdown
   cannot reach the inner runner. Delegate async shutdown directly to
   `inner.shutdown()` and require a delegation/static-structure test.

#### Main-Agent Recommendation

Accept. This is required both for structural typing and the existing sole-owner
runner cleanup contract.

### User Decision And Round 3 Revision

- User accepted the shutdown-delegation fix.
- `ValidatedStageRunner.shutdown()` now delegates exactly once to the inner runner.
- Implementation verification requires structural typing and delegation tests.

### Review Round 4 — Same-Reviewer Convergence

- All previous nominal payload typing findings resolved.
- No blocking or material findings; `git diff --check` passed.

### Review Round 5 — Fresh Independent Reviewer

#### Blocking

None.

#### Material

1. The contract says exact concrete payload class, but wrapper guards use only
   `isinstance`, which accepts subclasses. Retain `isinstance` for TypeVar narrowing
   and additionally require `type(value) is expected_type`; adapters use the same
   exact-class rule for final payloads. Add subclass rejection tests.
2. The early opaque-stage definition still says `runner_factory=None` means the
   adapter owns execution. Clarify it means there is no common-metadata extracted
   runner factory: the adapter constructs the opaque runtime runner, but the generic
   stage engine/executor still owns execution.

#### Optional

1. Clarify the review document's top-level status so the old completed lifecycle
   rounds are not confused with the active payload revision.

#### Main-Agent Recommendation

Accept both material fixes and the status cleanup. They make “exact type” truly
fail closed and remove a stale ownership contradiction.

### User Decision And Round 5 Revision

- User accepted both material fixes and review-status cleanup.
- Exact concrete payload validation now combines `isinstance` narrowing with class
  identity and rejects subclasses at runner input/output and adapter finalization.
- Required subclass-rejection negative tests.
- Corrected opaque-stage semantics: adapter constructs the runtime runner, while
  the generic stage engine/executor owns traversal and execution.

### Review Round 6 — Fresh-Reviewer Convergence

- Exact-class validation and opaque execution ownership findings resolved.
- No new blocking/material findings; `git diff --check` passed.

### Review Round 7 — Final Independent Sign-Off

#### Blocking

None.

#### Material

1. One architecture ownership list still says adapters own loaded pipeline/stage
   handles. Elsewhere, successful runner return atomically transfers those handles
   to executor-owned runners and adapter shutdown clears only adapter state. Clarify
   that adapters own handles only during construction/partial-failure cleanup;
   after successful return, runners/executor own them exclusively.

#### Main-Agent Recommendation

Accept the wording correction to preserve sole-owner, exactly-once cleanup.

### User Decision And Round 7 Revision

- User requested the ownership wording fix.
- Adapter ownership is now limited to construction and partial-failure cleanup.
- Successful return transfers runner-held handles atomically and exclusively to the
  executor lifecycle; adapter shutdown cannot touch transferred handles.

### Review Round 8 — Ownership Convergence

- Prior ownership contradiction resolved.
- No blocking or material findings; `git diff --check` passed.

### Review Round 9 — Final Fresh Sign-Off

- New independent reviewer found no blocking or material issues.
- Confirmed nominal payload inheritance, generic/erased runner boundary without
  `Any`, exact subclass-rejecting validation, shutdown delegation, exclusive handle
  ownership transfer, and common Qwen/Flux stage execution.

### Nominal Payload Revision Status

Complete. No unresolved blocking, material, or optional findings.
