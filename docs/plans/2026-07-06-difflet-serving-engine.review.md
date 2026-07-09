# Difflet Serving Engine Design Review

Status: complete

## Round 26

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f4683-89ed-7200-8fed-232245f2eff9`

### Findings

#### Blocking

None.

#### Material

1. P0 still named `--num-frames` as an image serving startup override.
   - Status: accepted and revised in revision 26.
   - Failure mode: implementers could bake `--num-frames` into the Qwen/Flux
     P0 `ServingProfile` even though P0 image profile matching is only
     `height`/`width` and request `num_frames` is invalid.
   - Resolution: `architecture.md` now lists only `--height` and `--width` as
     P0 image shape startup overrides and marks `--num-frames` as future
     video-only. Non-null startup `num_frames` is rejected for Qwen/Flux.

2. ArtifactStore upload/presign lacked a timeout boundary.
   - Status: accepted and revised in revision 26.
   - Failure mode: a hung R2 upload or presign could hold the HTTP handler
     indefinitely after worker generation succeeded, outside
     `request_timeout`.
   - Resolution: `engine.md` and `chat_completions_contract.md` now define
     `artifact_store_timeout`, default 60s, require SDK/client timeouts, and
     map timeout/failure to `artifact_upload_failed` or
     `artifact_store_unavailable`.

3. Missing or empty prompts lacked a machine-readable error code.
   - Status: accepted and revised in revision 26.
   - Failure mode: empty `messages`, no user prompt, or unextractable prompt
     could diverge across handler/tests.
   - Resolution: `chat_completions_contract.md` now uses
     `400 invalid_prompt` and includes it in the required error table.

#### Optional

None.

### Revision 26 Changes

- `docs/design/difflet_serving/architecture.md`
  - Removed `--num-frames` from P0 image startup overrides and marked it
    future-video-only.
- `docs/design/difflet_serving/engine.md`
  - Added `artifact_store_timeout`, default 60s, and clarified that artifact
    upload/presign is outside engine `request_timeout`.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Added artifact-store timeout behavior.
  - Added `invalid_prompt` for missing/empty/unextractable prompts.

Next planned review action:

- Stop. Fresh reviewer follow-up verified all Round 26 material findings are
  resolved and reported no remaining blocking or material issues.

## Round 25

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f4679-1789-7360-839b-6597bf95d8b9`

### Findings

#### Blocking

1. Qwen shared-process resident P0 needed an explicit go/no-go gate.
   - Status: accepted and revised in revision 25.
   - Failure mode: existing CLI/benchmark evidence still uses separate stage
     processes and file handoff, while the P0 docs require one resident worker
     process. Without a concrete co-load/smoke gate, implementation could
     discover too late that the active profile cannot co-fit.
   - Resolution: `architecture.md`, `engine.md`, and
     `chat_completions_contract.md` now state that Qwen is a P0 target only
     when all stages pass shared-worker co-load and smoke for the selected
     `ServingProfile`; failure is startup failure, not implicit fallback.

2. Caller cancellation could release the execution ticket while a shielded
   worker task kept running.
   - Status: accepted and revised in revision 25.
   - Failure mode: `asyncio.CancelledError` from client disconnect, ASGI
     cancellation, or shutdown skipped the timeout handler and reached
     `finally` with `release_ticket=True`, violating `max_running_requests=1`.
   - Resolution: `engine.md` now routes `CancelledError` through the same
     `start_inflight_recovery(...)` ownership-transfer path as timeout, keeps
     the ticket until safe recovery, and re-raises `CancelledError`.

#### Material

1. `num_frames` behavior for P0 image models was ambiguous.
   - Status: accepted and revised in revision 25.
   - Failure mode: Qwen/Flux image requests with `extra_body.num_frames` could
     reasonably produce `invalid_extra_body`, `profile_mismatch`, or a video
     rejection depending on implementer interpretation.
   - Resolution: `chat_completions_contract.md` now says Qwen/Flux reject
     non-null `num_frames` with `400 invalid_extra_body`; `null` is treated as
     absent. `num_frames` profile matching is reserved for future video
     adapters.
   - Follow-up: same reviewer found a stale "Request shape/profile fields"
     section that still listed `num_frames` as a P0 profile-matching field.
     Revision 25 follow-up removed it from P0 image profile matching and
     restated it as future video-only.

#### Optional

None.

### Revision 25 Changes

- `docs/design/difflet_serving/architecture.md`
  - Added the Qwen shared-worker co-load/smoke gate.
- `docs/design/difflet_serving/engine.md`
  - Added the Qwen shared-worker load/smoke gate to the worker layout.
  - Added `asyncio.CancelledError` recovery pseudocode and re-raise rule.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Defined `num_frames` handling for image adapters.
  - Marked Qwen P0 support as gated by shared-worker co-load/smoke.

Next planned review action:

- Same-reviewer follow-up to verify revision 25.

## Round 24

Reviewer: user-provided follow-up review

### Findings

#### Blocking

None.

#### Material

1. Qwen-Image support matrix still showed a per-stage resident core formula.
   - Status: accepted and revised in revision 24.
   - Resolution: P0 table now shows only `max(stage_cores)` for Qwen's
     shared-process resident layout. Per-stage resident sums are called out as
     future-only capacity planning.

2. `worker_restart_timeout` was defined but not used in the engine recovery
   flow.
   - Status: accepted and revised in revision 24.
   - Resolution: `engine.md` now defines restart/load/smoke timeout behavior:
     transition to `ERROR`, keep `/ready=503`, and return
     `503 engine_unavailable` until process restart or a future retry policy.

3. Split P0 docs did not state how they relate to the older long plan.
   - Status: accepted and revised in revision 24.
   - Resolution: `architecture.md`, `engine.md`, and
     `chat_completions_contract.md` now state they are authoritative for P0;
     the older long plan is broader future reference unless repeated there.

#### Optional

1. Top-level and `extra_body` duplicate-field wording was redundant after the
   top-level Difflet field precedence rule.
   - Status: accepted and revised in revision 24.
   - Resolution: wording now says flattened top-level Difflet generation/shape
     fields are invalid placement.

### Revision 24 Changes

- `docs/design/difflet_serving/chat_completions_contract.md`
  - Updated the Qwen core column to P0 shared-process only.
  - Reworded top-level Difflet generation/shape field placement.
  - Added P0 authority note.
- `docs/design/difflet_serving/engine.md`
  - Added P0 authority note.
  - Defined `worker_restart_timeout` failure behavior.
- `docs/design/difflet_serving/architecture.md`
  - Added P0 authority note.

Next planned review action:

- Stop unless another review round is requested.

## Round 23

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f4664-eb2e-7183-abdd-7effef1b5218`

### Findings

#### Blocking

None.

#### Material

1. Top-level Difflet runtime/profile fields could fall through to the wrong
   error class.
   - Status: accepted and revised in revision 23.
   - Failure mode: unknown top-level fields return `400 feature_not_supported`,
     while known Difflet generation/shape/startup/runtime fields sent top-level
     should be invalid request placement and return `400 invalid_extra_body`.
   - Reviewer fix: after ignoring top-level/`extra_body` `response_format` and
     `artifact_ttl_seconds`, any known Difflet generation/shape/startup/runtime
     or TeaCache field sent top-level returns `400 invalid_extra_body`.

2. Queue-timeout status was ambiguous in the long plan.
   - Status: accepted and revised in revision 23.
   - Failure mode: one table allowed `429` or `503` for queue wait expiry,
     while the contract requires `429 queue_timeout`.
   - Reviewer fix: make the table use exact error codes:
     `429 queue_full`, `429 queue_timeout`, and `400 profile_mismatch`.

### User Question

- Clarified that "adapter" means a model-specific implementation of the common
  serving protocols, not a preexisting package. P0 concrete adapters live under
  `difflet/serving/orchestrators/` and are wired through
  `preflight_factory` / `orchestrator_factory`.

### Revision 23 Changes

- `docs/design/difflet_serving/chat_completions_contract.md`
  - Added top-level field precedence for known Difflet fields:
    generation/shape/startup/runtime/TeaCache fields sent top-level return
    `400 invalid_extra_body`.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added matching OpenAI handler responsibility and verification tests for
    top-level Difflet fields.
  - Changed the recommended behavior table to exact queue/profile error codes.
  - Added an explicit definition of "adapter" and concrete P0 module locations.
- `docs/design/difflet_serving/architecture.md`
  - Added the same adapter terminology clarification.

Next planned review action:

- Stop.

### Round 23 Follow-Up

Reviewer: same `gpt-5.5` high explorer subagent
`019f4664-eb2e-7183-abdd-7effef1b5218`

#### Blocking

None.

#### Material

None.

#### Previous Findings

- Top-level known Difflet field error class: resolved.
- Queue full/timeout ambiguity: resolved.

#### Optional Cleanup

- Changed generic shape/profile mismatch wording to `400 profile_mismatch`.
- Changed `--queue-timeout` flag wording to `429 queue_timeout`.

### Final Review Status

- Latest fresh independent reviewer follow-up reports no blocking or material
  issues.
- Status: complete.

## Round 22

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f463b-21d2-7872-ac95-4ad8e6e18854`

### Findings

#### Blocking

None.

#### Material

1. `response_format` had conflicting error-code handling depending on where it
   appeared.
   - Status: revised in revision 22 with user-modified behavior.
   - Failure mode: top-level `response_format` could be treated as an
     unsupported chat feature returning `400 feature_not_supported`, while
     `extra_body.response_format` was documented as `400 invalid_extra_body`.
   - Reviewer fix: make response-policy fields invalid wherever present.
   - User decision: ignore `response_format` and `artifact_ttl_seconds` instead
     of returning an error. They have no effect whether top-level or inside
     `extra_body`; P0 still returns ArtifactStore/R2 URL with server-configured
     TTL.

#### Optional

1. Review document header still said `Status: active` while the prior final
   status said complete.
   - Status: accepted. The header remains `active` during this new review round
     and will be set to `complete` after follow-up convergence.

2. `ArtifactRef.uri` wording allowed an exception when `get_url(ref)` returned
   the same value.
   - Status: accepted and revised in revision 22.
   - Fix: make the invariant absolute: the handler returns only the value from
     `ArtifactStore.get_url(ref)`, never `ArtifactRef.uri` directly.

### Revision 22 Changes

- `docs/design/difflet_serving/chat_completions_contract.md`
  - Added response-policy ignored-field precedence for top-level and
    `extra_body` `response_format` / `artifact_ttl_seconds`.
  - Removed `response_format` from the generic unsupported chat feature list.
  - Made `ArtifactRef.uri` internal with no direct-return exception.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Updated OpenAI handler responsibilities to ignore response-policy fields
    wherever present.
  - Added ignored response-policy regression tests for top-level and
    `extra_body` fields.
  - Made `ArtifactRef.uri` internal with no direct-return exception.

Next planned review action:

- Same reviewer follow-up to verify Round 22 material finding is resolved.

### Round 22 Follow-Up

Reviewer: same `gpt-5.5` high explorer subagent
`019f463b-21d2-7872-ac95-4ad8e6e18854`

#### Blocking

None.

#### Material

1. Ignored response-policy fields were not verified to stay out of worker input.
   - Status: accepted and revised in revision 22.1.
   - Failure mode: tests could prove URL/TTL behavior while an implementation
     still copied `response_format` or `artifact_ttl_seconds` into
     `DiffletGenerateRequest.extra_params` or another worker-facing field.
   - Reviewer fix: extend ignored response-policy tests so a fake engine
     receives the same `DiffletGenerateRequest` with and without those fields,
     including no entries in `extra_params`.

#### Previous Findings

- `response_format` error-code conflict: resolved under the user decision to
  ignore response-policy fields.

### Revision 22.1 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Extended ignored response-policy field tests to assert those fields do not
    reach `DiffletGenerateRequest.extra_params` or any other worker-facing field.

### Round 22.1 Follow-Up

Reviewer: same `gpt-5.5` high explorer subagent
`019f463b-21d2-7872-ac95-4ad8e6e18854`

#### Blocking

None.

#### Material

None.

#### Previous Findings

- Ignored response-policy fields staying out of worker input: resolved.

### Final Review Status

- Latest fresh independent reviewer follow-up reports no blocking or material
  issues.
- Status: complete.

Next planned review action:

- Stop.

## Round 21

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f462e-9f49-7a22-8510-a765d17a4956`

### Findings

#### Blocking

None.

#### Material

1. P0 request contract still accepted non-generation response-policy fields.
   - Status: accepted by user and revised in revision 21.
   - Failure mode: `extra_body.response_format` and
     `extra_body.artifact_ttl_seconds` expanded the P0 public request API beyond
     the stated generation-plus-shape contract.
   - Reviewer fix: remove them from accepted `extra_body`, value-limit tables,
     and handler duties; always return URL and use server-configured TTL.
   - User decision: do not retain `response_format` or
     `artifact_ttl_seconds` as request fields.

2. ArtifactStore URL handoff was ambiguous enough to leak the wrong URI.
   - Status: accepted and revised in revision 21.
   - Failure mode: some prose implied `put_bytes(...)` returned the URL for
     `image_url.url`, which could cause an implementation to expose internal
     `ArtifactRef.uri`.
   - Reviewer fix: specify `ref = await store.put_bytes(...)`, then
     `url = await store.get_url(ref)`, and return only `url`.

### Revision 21 Changes

- `docs/design/difflet_serving/chat_completions_contract.md`
  - Removed request support for `response_format` and `artifact_ttl_seconds`.
    P0 always returns an ArtifactStore/R2 URL and uses server-configured TTL.
  - Defined `ArtifactRef.uri` as internal and required
    `ArtifactStore.get_url(ref)` before filling `image_url.url`.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Removed `ResponseOptions` and per-request artifact TTL handling.
  - Changed OpenAI handler responsibilities to reject response-policy fields and
    perform `put_bytes(...) -> get_url(ref) -> image_url.url`.
  - Removed `--max-artifact-ttl-seconds` from P0 serve flags.
  - Added verification that `get_url(ref)`, not `ArtifactRef.uri`, is returned.
- `docs/design/difflet_serving/architecture.md`
  - Updated request flow to include `ArtifactStore.get_url(ref)` after upload.

### Round 21 Follow-Up

Reviewer: same `gpt-5.5` high explorer subagent
`019f462e-9f49-7a22-8510-a765d17a4956`

#### Blocking

None.

#### Material

1. `extra_body` still had a loophole for advanced runtime fields.
   - Status: accepted and revised in revision 21.1.
   - Failure mode: the contract allowed TeaCache and advanced runtime fields to
     pass through when an adapter declared support, which conflicted with the P0
     goal that request `extra_body` accepts generation and request-facing shape
     fields only.
   - Reviewer fix: P0 rejects TeaCache and advanced runtime fields with
     `400 invalid_extra_body`; a future API revision may add explicit
     adapter-declared runtime fields.

#### Previous Findings

- Response-policy request fields: resolved.
- ArtifactStore URL handoff: resolved.

### Revision 21.1 Changes

- `docs/design/difflet_serving/chat_completions_contract.md`
  - Replaced the TeaCache/advanced runtime pass-through sentence with an
    explicit P0 rejection rule.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added invalid advanced runtime field tests requiring
    `400 invalid_extra_body` in P0.

### Round 21.1 Follow-Up

Reviewer: same `gpt-5.5` high explorer subagent
`019f462e-9f49-7a22-8510-a765d17a4956`

#### Blocking

None.

#### Material

None.

#### Previous Findings

- TeaCache and advanced runtime field loophole: resolved.

### Final Review Status

- Latest fresh independent reviewer follow-up reports no blocking or material
  issues.
- Status: complete.

Next planned review action:

- Stop.

## Round 1

Reviewer: `gpt-5.5` high, explorer subagent `019f40d8-efc1-75b1-b15d-ce96524600ff`

Artifacts reviewed:

- `docs/plans/2026-07-06-difflet-serving-engine.md`
- `docs/design/difflet_serving/chat_completions_contract.md`

Goal:

- Review the Difflet serving design for the 4-core Trainium text-to-image MVP.
- Check correctness, implementation risk, lifecycle/API contracts, and whether
  the design leaves extension points for stage call APIs to change later.

## Findings

### Blocking

None.

### Material

1. Final artifact contract conflicts with engine output type.
   - Status: accepted and resolved in revision 1.
   - References:
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, `DiffletGenerateOutput`
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, OpenAI Chat Handler
     - `docs/design/difflet_serving/chat_completions_contract.md`, output mapping
   - Failure mode: implementation may return base64/debug bytes in deployment
     and bypass R2/ArtifactStore.
   - Resolution: keep the engine bytes-first and storage-agnostic. The OpenAI
     handler writes `DiffletGenerateOutput.data` through `ArtifactStore`
     (`put_bytes`) for `response_format=url`, with R2 as the deployment backend.
     `data_url` is explicit local/debug behavior.

2. Request data model omits output policy fields.
   - Status: accepted and resolved in revision 1.
   - References:
     - `docs/design/difflet_serving/chat_completions_contract.md`,
       `extra_body` fields
     - `docs/plans/2026-07-06-difflet-serving-engine.md`,
       `DiffletGenerateRequest`
   - Failure mode: `output_format`, `response_format`, and
     `artifact_ttl_seconds` get buried in `extra_params`, causing inconsistent
     adapter validation.
   - Resolution: added explicit `output_format`, `response_format`, and
     `artifact_ttl_seconds` fields to `DiffletGenerateRequest`; contract now
     defines defaults.

3. Rotating resident relies on unload/detach behavior not present in current
   load primitives.
   - Status: accepted with scope reduction; resolved for MVP in revision 1.
   - References:
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, rotating scheduling
     - `difflet/backends/trainium/core/application_base.py`, `load`
   - Failure mode: all-rotating fallback may not release device state/HBM.
   - Resolution: rotating resident is now future-only for MVP. P0 uses one
     FastAPI process plus one shared-process Trainium worker. Rotating plans are
     excluded from automatic selection until adapters declare `supports_unload`
     or a worker-restart strategy.

4. Qwen staged artifact validation is under-specified for AOT-required serving.
   - Status: accepted and resolved in revision 1.
   - References:
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, compile lifecycle
     - `difflet/cli/orchestrators/qwen_image.py`, hand-built compiled dirs
     - `difflet/pipeline/compile_cache.py`, manifest-based cache
   - Failure mode: serving may reject valid CLI artifacts with no manifest, or
     accept stale artifacts from the wrong revision/toolchain/profile.
   - Resolution: added legacy Qwen staged artifact validation rules and a
     manifest strategy. `compile-policy require` may accept legacy directories
     only when expected files and profile-specific directory names match, with
     a warning if toolchain/model revision freshness cannot be fully verified.

5. `download-policy require` appears in CLI docs but not in the enum.
   - Status: accepted and resolved in revision 1.
   - References:
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, CLI flags
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, `DownloadPolicy`
   - Failure mode: parser/docs divergence.
   - Resolution: removed `download-policy require`. Download policy is
     `{auto,never}`; `auto` skips when local weights exist and downloads when
     missing.

### Optional

1. Shared-process resident should be framed as attempted/default but not
   guaranteed until startup load/warmup proves it.
   - Status: accepted and resolved in revision 1.

2. Add adapter-conformance tests to M1/M3.
   - Status: accepted and resolved in revision 1.

## Main-Agent Local Notes

- The design now has enough extension points for rewritten stage call APIs:
  `DiffletStageAdapter`, `StageRunRequest`, `StageRunResponse`,
  `DiffletRuntimePlan`, and shared-process versus per-stage-process plans.
- The most important cleanup is to make deployment output unambiguously
  ArtifactStore/R2-first while preserving local/debug `data_url` as an explicit
  mode.

## Revision 1

User decisions:

- Engine should return bytes. R2/artifact upload belongs in the serving handler
  and `ArtifactStore`, not inside the engine.
- Use OpenAI-style chat completions as the public wrapper, but treat generated
  image modality as a Difflet/vLLM-Omni-style extension. Omitted `modalities`
  means model default; `["image"]` is allowed for T2I.
- Put rotating resident later. MVP is one FastAPI process and one worker process
  that loads/runs the active model/profile.
- Startup download policy should be `auto`: skip if weights exist, download if
  missing. No `download-policy=require`.

Changes made:

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added bytes-first engine output and ArtifactStore/R2-first response
    formatting.
  - Added request fields for `output_format`, `response_format`, and
    `artifact_ttl_seconds`.
  - Made shared-process 4-core resident the P0 runtime plan and moved rotating
    resident to future-only unless unload/restart is implemented.
  - Removed `download-policy=require` from serving CLI docs and startup policy.
  - Added Qwen legacy staged artifact validation and manifest rules.
  - Added `ArtifactStore.put_bytes(...)` and adapter `supports_unload`.
  - Updated M1/M2/M3 verification and output milestones.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Clarified `modalities` compatibility and Difflet media extension behavior.
  - Clarified output defaults and ArtifactStore upload boundary.
  - Added default output format/MIME requirements.
- `tasks/todo.md`
  - Recorded the latest design decisions and documentation verification.

Rejected feedback:

- Do not make engine output carry `ArtifactRef` for MVP. Rationale: user wants
  engine decoupled from R2/storage so artifact policy can change later.

Next planned review action:

- Same reviewer follow-up to verify Round 1 findings are resolved and identify
  any new blocking/material issues.

## Round 2

Reviewer: same `gpt-5.5` high explorer subagent
`019f40d8-efc1-75b1-b15d-ce96524600ff`

### Findings

#### Blocking

None.

#### Material

1. Rotating fallback is reintroduced in a later resource section.
   - Status: accepted and resolved in revision 2.
   - References:
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, resource shortage
       mitigation section.
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, engine factory
       snippet.
   - Failure mode: implementers could preserve an unsafe automatic rotating
     path despite current Trainium load primitives lacking detach/unload.
   - Resolution: removed automatic rotating selection from the shortage section
     and added a factory/preflight guard:
     `validate_rotating_plan_has_unload_or_restart(plan)`.

2. Shared-process Neuron env semantics are under-specified for mixed-core Qwen
   stages.
   - Status: accepted and resolved in revision 2.
   - References:
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, primary resident
       stage runtime.
   - Failure mode: implementation may mutate process-level Neuron env inside a
     live shared worker, or assume the 1-core Qwen VAE can load/run under a
     4-core worker env without validation.
   - Resolution: added a shared-process Neuron env contract. The worker env is
     immutable for the runtime plan, uses plan-level core settings such as
     `NEURON_RT_NUM_CORES=max(stage_cores)`, and each adapter must validate
     load/run compatibility during startup warmup.

### Previous Findings

- Round 1 material findings 1, 2, 4, and 5: resolved.
- Round 1 material finding 3: resolved for MVP after revision 2 removed the
  stale rotating fallback.
- Round 1 optional findings: resolved.

### Revision 2 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Removed stale rotating fallback in the resource shortage/admission section.
  - Added a factory guard for any `ROTATING_RESIDENT` plan. This was later
    superseded by the 2026-07-09 P0 factory rule that accepts only
    `RESIDENT_WORKER` stages and rejects rotating plans until future work.
  - Added a shared-process immutable Neuron env contract and Qwen VAE
    compatibility caveat.

Next planned review action:

- Same reviewer follow-up to verify Round 2 material findings are resolved.

## Round 3

Reviewer: same `gpt-5.5` high explorer subagent
`019f40d8-efc1-75b1-b15d-ce96524600ff`

### Findings

#### Blocking

None.

#### Material

None.

#### Optional

1. Future rotating section still says "for MVP".
   - Status: accepted and resolved.
   - Resolution: renamed the heading to "Future rotating resident scheduling
     rules" and changed "in MVP" wording to "in the first rotating
     implementation".

2. M3 wording says Qwen workers plural despite one-worker MVP.
   - Status: accepted and resolved.
   - Resolution: changed the milestone to "Start the shared Qwen worker" and
     load prompt encoder, denoiser, and decoder adapters/apps into it.

### Round 2 Verification

- Material 1, rotating fallback: resolved.
- Material 2, shared-process Neuron env: resolved.

### Revision 3 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Clarified rotating scheduling section as future-only.
  - Clarified M3 Qwen startup as one shared worker process.

Next planned review action:

- Stop. Latest fresh independent reviewer follow-up reports no blocking or
  material issues; optional cleanup from that pass has been applied.

## Round 18

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f45ef-9b5b-7471-9aab-337858244365`

Review mode: from-scratch review with no prior round summary.

Review date: 2026-07-09

### Findings

#### Blocking

1. P0's one resident Trainium worker process invariant is not enforced
   consistently.
   - Status: accepted and revised in revision 18.4.
   - References:
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, Engine Factory
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, Resident Worker
       Runtime Strategy
   - Failure mode: the factory accepts any all-`RESIDENT_WORKER` plan, while the
     worker layout still describes `per_stage_process`. An implementer could
     build one worker per stage, violating the P0 shared-worker constraint.
   - Reviewer fix: for P0, reject runtime plans whose
     `core_allocation != "shared_process"` or whose resolved worker count is not
     exactly `1`; move `per_stage_process` to future-only text and add startup
     rejection tests.

#### Material

1. `engine_recovering` is defined by the engine but missing from the public chat
   error contract.
   - Status: accepted and revised in revision 18.4.
   - References:
     - `docs/design/difflet_serving/engine.md`, Request Admission
     - `docs/design/difflet_serving/chat_completions_contract.md`, Error
       Contract
   - Failure mode: timeout recovery returns an undocumented error code.
   - Reviewer fix: add `503 engine_recovering` to the chat contract and clarify
     health/readiness behavior during recovery.

2. `DiffletGenerateRequest` mixes generation inputs with HTTP artifact-response
   policy.
   - Status: accepted and revised in revision 18.4.
   - References:
     - `docs/plans/2026-07-06-difflet-serving-engine.md`,
       `DiffletGenerateRequest`
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, engine output boundary
   - Failure mode: `response_format` and `artifact_ttl_seconds` leak HTTP/R2
     policy into engine/worker/model code, despite the bytes-first engine
     boundary.
   - Reviewer fix: remove those two fields from `DiffletGenerateRequest`; keep
     them in an HTTP-layer response options object used after
     `engine.generate(...)`.

3. Registry package refactor conflicts with the current `difflet/registry.py`
   layout.
   - Status: accepted by user and revised in revision 18.3.
   - References:
     - `docs/design/difflet_serving/architecture.md`, Folder Layout
     - `difflet/registry.py`
   - Failure mode: Python cannot keep both `difflet/registry.py` and a
     `difflet/registry/` package at the same import path without a migration.
   - Reviewer fix: either keep the flat `difflet.registry` module for P0, or
     explicitly migrate it to a package while preserving
     `from difflet.registry import resolve_model` compatibility.

4. Qwen `cp_degree` serving overrides are not currently backed by the prompt
   encoder implementation.
   - Status: accepted and revised in revision 18.4.
   - References:
     - `docs/design/difflet_serving/architecture.md`, startup override rules
     - `difflet/cli/orchestrators/qwen_image.py`, text stage `NeuronConfig`
     - `difflet/cli/orchestrators/qwen_image.py`, compiled directory naming
   - Failure mode: docs say Qwen text and denoiser consume
     `tp_degree * cp_degree`, but the current text stage passes only
     `tp_degree` into `NeuronConfig`; `cp_degree > 1` may be falsely advertised.
   - Reviewer fix: reject Qwen serving `cp_degree > 1` until prompt-encoder CP
     is implemented and smoke-tested, or encode stage-specific CP support in
     serving metadata.

5. Flux artifact verification can be bypassed if serving follows
   `from_pretrained(..., skip_compile=True)` too literally.
   - Status: accepted and revised in revision 18.4.
   - References:
     - `difflet/pipeline/difflet_pipeline.py`, cache readiness and load path
   - Failure mode: `skip_compile=True` computes `cache_ready` but does not fail
     early on a stale or missing manifest before load is attempted.
   - Reviewer fix: require a serving `ensure_artifacts()` step for Flux that
     checks `has_valid_manifest(...)` plus app artifact readiness before worker
     pipeline construction/loading.

6. R2 upload is specified as synchronous in an async chat handler path.
   - Status: accepted and revised in revision 18.4.
   - References:
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, ArtifactStore
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, OpenAI Chat Handler
   - Failure mode: a slow `ArtifactStore.put_bytes(...)` can block FastAPI's
     event loop and affect health/readiness or other HTTP handling.
   - Reviewer fix: make `ArtifactStore` async, or require sync implementations
     to run in a bounded executor with timeout/error mapping.

#### Optional

1. Consider accepting harmless OpenAI no-op defaults such as `stream: false` or
   `n: 1` instead of rejecting all presence of those fields.
   - Status: pending user confirmation.

2. Move future video milestones out of the P0 implementation milestone list.
   - Status: pending user confirmation.

### Proposed Next Edits

- Tighten P0 engine selection: require `shared_process` and exactly one worker;
  move `per_stage_process` to future-only discussion and add tests.
- Add `engine_recovering` to the public chat error table and define
  `/health`/`/ready` behavior while recovering.
- Split HTTP response options from `DiffletGenerateRequest`.
- Clarify the registry path by leaving `difflet/registry.py` unchanged and
  placing new modular registry/common metadata under `difflet/common/registry/`.
- Add Qwen CP support metadata and reject `cp_degree > 1` for Qwen P0 until the
  text encoder path supports it.
- Add a Flux `ensure_artifacts()` startup requirement before `skip_compile=True`
  load.
- Make ArtifactStore upload async or executor-backed with timeout/error mapping.

### Revision 18.1 Changes

- Status: superseded by revision 18.2 after user clarified that the registry
  should be split now, not kept flat.
- `docs/design/difflet_serving/architecture.md`
  - Replaced the proposed `difflet/registry/` package layout with the existing
    `difflet/registry.py` module.
  - Clarified P0 serving must reuse `difflet.registry.resolve_model(...)` and
    not modify the base registry layout.
  - Updated the model-addition checklist so serving work adds new common and
    serving files, while base registry additions remain a separate
    compatibility-preserving change when needed.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Removed the P0 instruction to split `difflet/registry.py`.
  - Documented `difflet/serving/model_registry.py` as an overlay on top of the
    existing base registry.
  - Updated the proposed module layout to keep `difflet/registry.py` as-is.

### Revision 18.2 Changes

- Status: superseded by revision 18.3 after user clarified that the original
  `difflet/registry.py` must remain untouched for existing scripts, and new
  modular registry files should live under `difflet/common/registry/`.
- `docs/design/difflet_serving/architecture.md`
  - Restored the target `difflet/registry/` package layout.
  - Explicitly states that `difflet/registry.py` and `difflet/registry/` must not
    coexist because both claim the `difflet.registry` import path.
  - Documents the migration shape: move generic code to `registry/base.py`,
    move model builtins to per-model files, and re-export the old public API
    from `registry/__init__.py`.
  - Updated model-addition steps and checklist to add per-model registry files
    after the package migration.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Restored the registry package split as part of the target serving
    architecture.
  - Added the same no-side-by-side constraint and compatibility-preserving
    migration requirement.

### Revision 18.3 Changes

- `docs/design/difflet_serving/architecture.md`
  - Keeps the existing `difflet/registry.py` module unchanged.
  - Adds `difflet/common/registry/` as the new modular registry namespace for
    common/serving metadata.
  - Clarifies `difflet/common/registry/` must wrap and reuse
    `difflet.registry.resolve_model(...)` rather than shadowing or duplicating
    broad model-id matching.
  - Updates the model-addition flow and checklist to add common registry
    metadata under `difflet/common/registry/`.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Replaces the `difflet/registry/` package migration with the three-layer
    shape: old `difflet/registry.py`, new `difflet/common/registry/`, and
    `difflet/serving/model_registry.py`.
  - Documents that serving should not replace the `difflet.registry` import path.

### Revision 18.4 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Tightened the engine factory so P0 accepts only
    `core_allocation="shared_process"`, `worker_count=1`, and resident stages.
  - Added `worker_count` to `DiffletRuntimePlan` and the Qwen shared-process
    plan example.
  - Moved `per_stage_process` to future-only runtime layout and documented that
    Qwen per-stage resident would require `4 + 4 + 1 = 9` cores.
  - Split HTTP response policy into `ResponseOptions`; `DiffletGenerateRequest`
    no longer carries `response_format` or `artifact_ttl_seconds`.
  - Added Qwen P0 startup rejection for `cp_degree > 1` until text encoder CP is
    implemented and smoke-tested.
  - Added Flux/pipeline `ensure_artifacts(...)` requirements before
    `skip_compile=True` load.
  - Changed `ArtifactStore` to an async protocol and required sync SDK
    implementations to run in a bounded executor.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Added `503 engine_recovering` to the public error contract.
  - Documented `/ready=503` and `/health` behavior during recovery.
  - Clarified response options are HTTP-layer state, not worker request fields.
  - Added Qwen P0 `cp_degree=1` constraint.
  - Updated artifact upload wording to `await ArtifactStore.put_bytes(...)`.
- `docs/design/difflet_serving/engine.md`
  - Added health/readiness semantics for `RECOVERING`.
- `docs/design/difflet_serving/architecture.md`
  - Updated request flow and pipeline-style model guidance for async artifact
    upload and Flux `ensure_artifacts(...)`.

### Round 18 Follow-Up

Reviewer: same `gpt-5.5` high explorer subagent
`019f45ef-9b5b-7471-9aab-337858244365`

#### Blocking

None.

#### Material

1. `worker_count="auto"` is ambiguous against the P0 factory check.
   - Status: accepted and revised in revision 18.5.
   - Failure mode: a P0 plan that omits `worker_count` would keep the default
     `"auto"` and fail `plan.worker_count == 1` unless an undocumented
     normalization step exists.
   - Reviewer fix: make `worker_count` an `int` defaulting to `1`, or define a
     normalization step.

2. One core-handling paragraph still contradicted the shared-process P0 worker
   model.
   - Status: accepted and revised in revision 18.5.
   - Failure mode: stale wording said resident workers must be started per stage,
     reintroducing the per-stage worker design P0 now rejects.
   - Reviewer fix: distinguish future `per_stage_process` from P0
     `shared_process`.

#### Optional

1. `engine.md` Flux layout showed `skip_compile=True` without the preceding
   `ensure_artifacts(...)` guard.
   - Status: accepted and revised as optional cleanup in revision 18.5.

2. Some Qwen wording still said text and denoiser consume
   `tp_degree * cp_degree`.
   - Status: accepted and revised as optional cleanup in revision 18.5.

#### Previous Findings

- P0 worker invariant: conceptually resolved in revision 18.4; tightened again
  in revision 18.5 for `worker_count`.
- `engine_recovering` public contract: resolved.
- `DiffletGenerateRequest` HTTP policy leakage: resolved.
- Registry layout conflict: resolved with `difflet/common/registry/`.
- Qwen `cp_degree > 1`: resolved.
- Flux artifact validation: resolved.
- Async ArtifactStore: resolved.

### Revision 18.5 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Changed `DiffletRuntimePlan.worker_count` from `int | str = "auto"` to
    `int = 1`.
  - Rewrote the process-level Neuron env/core handling paragraph so P0
    `shared_process` uses one immutable plan-level worker env and future
    `per_stage_process` is explicitly separate.
  - Clarified Qwen text/generate `tp*cp` wording with P0 `cp_degree=1`.
- `docs/design/difflet_serving/engine.md`
  - Added `ensure_artifacts()` before the Flux `skip_compile=True` load snippet.
- `docs/design/difflet_serving/architecture.md`
  - Clarified Qwen P0 core wording to say supported Qwen P0 uses `tp_degree`
    because `cp_degree=1`.

### Round 18.5 Follow-Up

Reviewer: same `gpt-5.5` high explorer subagent
`019f45ef-9b5b-7471-9aab-337858244365`

#### Blocking

None.

#### Material

None.

#### Optional

1. One plan sentence still said Qwen text/denoiser use `tp_degree * cp_degree`
   in P0.
   - Status: accepted and revised as optional cleanup in revision 18.6.

#### Previous Findings

- `worker_count="auto"` ambiguity: resolved.
- Stale per-stage worker wording: resolved.
- Flux `ensure_artifacts()` snippet cleanup: resolved.
- Architecture Qwen `cp_degree=1` wording: resolved.

### Revision 18.6 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Updated the remaining Qwen P0 profile sentence to say Qwen text and denoiser
    use `tp_degree` because P0 requires `cp_degree=1`.

## Round 19

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f4611-a495-70d3-aaee-64a1724f7160`

### Findings

#### Blocking

None.

#### Material

1. Preflight ownership is still contradictory.
   - Status: accepted and revised in revision 19.1.
   - Failure mode: some sections say serving orchestrators are worker-owned, but
     startup requires download, compile plan creation, and artifact checks before
     the worker starts.
   - Reviewer fix: split parent-side cold/preflight methods from the worker-owned
     runtime orchestrator.

2. Timeout pseudocode can cancel the IPC receive path while recovery still needs
   it.
   - Status: accepted and revised in revision 19.1.
   - Failure mode: `asyncio.wait_for(self.run_one(...))` cancels `run_one` on
     timeout, potentially leaving late worker replies unread or mis-associated.
   - Reviewer fix: create an explicit worker generation task, await it with
     `asyncio.shield`, and transfer ticket plus task/request id to recovery.
     Replies must be request-id tagged and recovery must drain/discard late
     terminal replies before worker reuse, or terminate/restart.

3. `--force-compile` semantics conflict with `compile-policy=require`.
   - Status: accepted and revised in revision 19.1.
   - Failure mode: docs alternately imply force compile works regardless of
     compile policy or only when policy allows.
   - Reviewer fix: make `--force-compile` valid only with
     `--compile-policy auto`; with `require`, fail startup as invalid serving
     configuration.

### Revision 19.1 Changes

- `docs/design/difflet_serving/architecture.md`
  - Split parent-side `ServingArtifactPreparer` from worker-owned
    `ServingModelOrchestrator`.
  - Updated startup flow so parent preflight performs download, compile-plan
    creation, and artifact checks before worker startup.
  - Added `preflight_factory` to serving metadata examples.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added the same parent-side preflight and worker-side runtime split.
  - Added `preflight_factory` to `ServingModelMetadata`.
  - Updated startup artifact preparation to construct and use
    `ServingArtifactPreparer`.
  - Defined `--force-compile` as valid only with `--compile-policy auto`; with
    `require`, startup fails as `unsupported_serving_configuration`.
- `docs/design/difflet_serving/engine.md`
  - Replaced timeout `wait_for(run_one(...))` with an explicit shielded worker
    generation task.
  - Recovery now owns the ticket plus in-flight task/request id, requires
    request-id tagged worker replies, and must drain/discard late terminal
    replies or restart the worker before reuse.

### Round 19 Follow-Up

Reviewer: same `gpt-5.5` high explorer subagent
`019f4611-a495-70d3-aaee-64a1724f7160`

#### Blocking

None.

#### Material

None.

#### Optional

1. Load reuse wording still said the serving orchestrator had run
   `ensure_artifacts(...)`, but that method moved to parent-side preflight.
   - Status: accepted and revised as optional cleanup in revision 19.2.

2. `ServingArtifactPreparer` differed between architecture and plan because the
   plan included `stage_specs(...)` while architecture did not.
   - Status: accepted and revised as optional cleanup in revision 19.2.

#### Previous Findings

- Preflight ownership contradiction: resolved.
- Timeout recovery cancelling IPC receive path: resolved.
- `--force-compile` vs `compile-policy=require`: resolved.

### Revision 19.2 Changes

- `docs/design/difflet_serving/architecture.md`
  - Added `stage_specs(...)` to `ServingArtifactPreparer`.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Updated load reuse wording so parent-side preflight/preparer owns
    `ensure_artifacts(...)` before `skip_compile=True` load.

Next planned review action:

- Fresh independent reviewer pass for missed blocking/material issues.

## Round 20

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f461c-8fc4-74a1-af16-44d5b2928196`

### Findings

#### Blocking

None.

#### Material

1. Worker/preflight ownership is still contradicted in the Flux engine snippet.
   - Status: accepted and revised in revision 20.1.
   - Failure mode: `engine.md` showed `ensure_artifacts()` inside
     `FluxServingOrchestrator`, which could put artifact checks back into the
     worker despite the parent-side preflight split.
   - Reviewer fix: show `ensure_artifacts()` as parent preflight before worker
     `LOAD_PROFILE`; keep worker Flux responsibilities to `load`, `smoke`,
     `generate`, and `shutdown`.

2. `extra_body` profile-bound fields are not consistently declared as accepted
   request fields.
   - Status: superseded by product decision and revised in revision 20.2.
   - Failure mode: `tp_degree`, `cp_degree`, `cp_mode`, `cfg_parallel`, and
     `sp_enabled` were described as profile-bound but not listed in accepted
     `extra_body` tables or handler mapping, so implementations/tests could
     disagree between `invalid_extra_body` and `profile_mismatch`.
   - Reviewer fix: list them as accepted profile-match-only fields, validate them
     against `ServingProfile`, and do not send them to the worker.
   - Final decision: do not accept these fields in request `extra_body`. They
     are `difflet serve` startup-only fields and internal `ServingProfile`
     identity fields. Requests containing them return `400 invalid_extra_body`.

### Revision 20.1 Changes

- `docs/design/difflet_serving/engine.md`
  - Split Flux snippet into parent preflight
    `FluxServingArtifactPreparer.ensure_artifacts()` and worker-owned
    `FluxServingOrchestrator.load/generate`.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Added profile-match-only `extra_body` fields: `tp_degree`, `cp_degree`,
    `cp_mode`, `cfg_parallel`, and `sp_enabled`.
  - Clarified they are request assertions only and are not copied into
    `DiffletGenerateRequest` or sent to the worker.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added OpenAI handler mapping/validation rules for the same profile-match-only
    fields.

### Revision 20.2 Changes

- `docs/design/difflet_serving/chat_completions_contract.md`
  - Superseded the profile-match-only request-field approach. `tp_degree`,
    `cp_degree`, `cp_mode`, `cfg_parallel`, and `sp_enabled` are startup-only
    `difflet serve` fields; request `extra_body` containing them returns
    `400 invalid_extra_body`.
  - Kept request-time profile matching limited to shape fields: `height`,
    `width`, and future `num_frames`.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Updated `ServingProfile` validation and the OpenAI handler mapping to reject
    startup-only runtime/profile fields in requests.

### Round 20 Follow-Up

Reviewer: same `gpt-5.5` high explorer subagent
`019f461c-8fc4-74a1-af16-44d5b2928196`

#### Blocking

None.

#### Material

1. Startup-only request field behavior needs explicit regression tests.
   - Status: accepted and revised in revision 20.3.
   - Failure mode: the contract now rejects `tp_degree`, `cp_degree`, `cp_mode`,
     `cfg_parallel`, and `sp_enabled`, but the M2 invalid `extra_body` test list
     did not name them. A future implementation could accidentally accept them
     or return `profile_mismatch`.
   - Reviewer fix: add those five fields to invalid `extra_body` verification
     and require `400 invalid_extra_body`.

#### Previous Findings

- Flux preflight/worker ownership: resolved.
- Request-time tp/cp/runtime fields: resolved with superseding product decision;
  startup-only fields are invalid in request `extra_body`.

### Revision 20.3 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added M2 regression tests requiring `extra_body.tp_degree`,
    `extra_body.cp_degree`, `extra_body.cp_mode`, `extra_body.cfg_parallel`,
    and `extra_body.sp_enabled` to return `400 invalid_extra_body`, not
    `400 profile_mismatch`.

### Round 20.3 Follow-Up

Reviewer: same `gpt-5.5` high explorer subagent
`019f461c-8fc4-74a1-af16-44d5b2928196`

#### Blocking

None.

#### Material

None.

#### Previous Findings

- Startup-only request field regression tests: resolved.

### Final Review Status

- Latest same-reviewer follow-up reports no blocking or material issues.
- Status: complete.

Next planned review action:

- Stop.

## Round 17

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f45e0-611d-7893-a014-5d6dfe6cb177`

### Findings

#### Blocking

None.

#### Material

1. Startup-time Flux checkpoint support is still ambiguous and can select the
   wrong weights.
   - Status: accepted and revised in revision 17.
   - Failure mode: the base Flux registry groups `FLUX.1-dev` and
     `FLUX.1-schnell`, while the current Flux CLI orchestrator hard-codes
     `FLUX.1-dev`. If serving enables every base `ModelEntry.hf_paths` value at
     startup, `--model-id FLUX.1-schnell` could resolve as `flux` but still load
     or compile dev weights.
   - Reviewer fix: add serving metadata for supported startup checkpoint ids.
     For P0, reject `FLUX.1-schnell` at startup unless the Flux
     common/serving orchestrator is parameterized and tested for that exact
     checkpoint.

2. Serving force-compile flag name is inconsistent.
   - Status: accepted and revised in revision 17.
   - Failure mode: docs mention both `--force-compile` and `--force`, so
     implementers/tests may disagree and stale artifacts become hard to refresh
     from serving.
   - Reviewer fix: standardize serving on `--force-compile`, with `--force`
     only as an optional compatibility alias.

#### Optional

None.

### Revision 17 Changes

- `docs/design/difflet_serving/chat_completions_contract.md`
  - Clarified Flux P0 serving enables only `black-forest-labs/FLUX.1-dev`.
  - `FLUX.1-schnell` must be rejected at startup until the serving orchestrator
    is parameterized and verified for that exact checkpoint.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added serving-supported checkpoint ids as explicit serving registry
    metadata.
  - Added Flux P0 checkpoint support rules and tests.
  - Standardized serving force compile flag on `--force-compile`; `--force` is
    only an optional compatibility alias.

Next planned review action:

- Same reviewer follow-up to verify Round 17 material findings are resolved.

### Round 17 Follow-Up

Reviewer: same fresh independent `gpt-5.5` high explorer subagent
`019f45e0-611d-7893-a014-5d6dfe6cb177`

#### Blocking

None.

#### Material

1. Flux checkpoint fix is only partially resolved because the resolver shape
   still accepts by model family.
   - Status: accepted and revised in revision 17.1.
   - Failure mode: `resolve_serving_model()` resolves `FLUX.1-schnell` to base
     `ModelEntry.name == "flux"` and returns `_SERVING_METADATA["flux"]`; the
     metadata shape lacked an allowlist field to enforce the new P0 rule.
   - Reviewer fix: add `enabled_model_ids` or equivalent to
     `ServingModelMetadata`, include it in the architecture example, and make
     `resolve_serving_model()` reject ids not in that allowlist before building
     a serving spec. For P0, Flux lists only `FLUX.1-dev`.

#### Previous Findings

- Flux checkpoint ambiguity: partially unresolved before revision 17.1.
- Force compile flag inconsistency: resolved.

### Revision 17.1 Changes

- `docs/design/difflet_serving/architecture.md`
  - Serving registry now explicitly owns serving-enabled checkpoint ids.
  - Startup family resolution through `difflet.registry` must be followed by an
    enabled-checkpoint allowlist check.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added `enabled_model_ids` to `ServingModelMetadata`.
  - Updated `resolve_serving_model()` to reject requested ids not present in
    `metadata.enabled_model_ids`.
  - Added P0 Flux metadata guidance:
    `enabled_model_ids=("black-forest-labs/FLUX.1-dev",)`.

Next planned review action:

- Same reviewer follow-up to verify Round 17.1 material finding is resolved.

### Round 17.1 Follow-Up

Reviewer: same fresh independent `gpt-5.5` high explorer subagent
`019f45e0-611d-7893-a014-5d6dfe6cb177`

#### Blocking

None.

#### Material

None.

### Final Review Status

- Round 17.1 Flux checkpoint resolver enforcement: resolved.
- Latest fresh independent reviewer follow-up reports no blocking or material
  issues.
- Status: complete.

## Round 4

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f410b-96b1-72b3-a615-72c025c868b8`

### Findings

#### Blocking

None.

#### Material

1. Qwen `true_cfg_scale` contract conflicts with current Qwen capability.
   - Status: accepted and resolved in revision 4.
   - Failure mode: the advertised MVP request may either be rejected or
     implemented with incorrect true-CFG semantics.
   - Resolution: changed Qwen examples/contracts to use `guidance_scale`;
     documented `true_cfg_scale` as accepted only by true-CFG adapters and
     rejected by Qwen/Flux MVP adapters.

2. Default `auto` can silently downgrade away from the required MVP runtime.
   - Status: accepted and resolved in revision 4.
   - Failure mode: production could report ready while running the debug
     subprocess path instead of the required shared-process worker.
   - Resolution: changed `--allow-engine-downgrade` default to `false` for
     MVP/deployment and documented subprocess fallback as explicit
     debug/comparison behavior.

3. Resident core-admission wording can reject the intended 4-core
   shared-process plan.
   - Status: accepted and resolved in revision 4.
   - Failure mode: implementers may compute 9 cores for all Qwen resident modes
     and reject the 4-core shared-process plan.
   - Resolution: clarified admission is based on the selected
     `DiffletRuntimePlan.peak_cores` and `core_allocation`; the 9-core sum
     applies only to the per-stage resident plan.

4. Artifact TTL/local URL contract is incomplete.
   - Status: accepted and resolved in revision 4.
   - Failure mode: per-request TTL cannot be honored, and dev/local
     `response_format=url` can return dead URLs.
   - Resolution: added `ttl_seconds` to `ArtifactStore.put_bytes` and
     `put_file`; documented effective TTL computation in the handler; added
     `GET /v1/files/{file_id}/content` to first milestone routes when local
     artifact URLs are enabled.

#### Optional

1. `response_format=auto` is listed but not resolved precisely.
   - Status: accepted and resolved in revision 4.
   - Resolution: defined `auto` as the server configured default response
     format, with deployment defaulting to `url`.

### Revision 4 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Replaced Qwen sample `true_cfg_scale` with `guidance_scale`.
  - Clarified Qwen MVP rejects `true_cfg_scale`.
  - Changed downgrade default to false for MVP/deployment.
  - Rewrote resource admission around selected runtime plan `peak_cores`.
  - Added artifact TTL parameters and local file route.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Updated request examples and per-model fields for Qwen guidance.
  - Defined `response_format=auto`.
  - Added handler-to-ArtifactStore TTL behavior.

Next planned review action:

- Fresh reviewer follow-up to verify Round 4 material findings are resolved.

## Round 5

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f410b-96b1-72b3-a615-72c025c868b8`

### Findings

#### Blocking

None.

#### Material

1. `max_running_requests=1` is still not enforced for the MVP.
   - Status: accepted and resolved in revision 5.
   - Failure mode: concurrent requests on one shared-process Trainium worker
     can collide in worker state, tensor handoff, or Neuron execution
     assumptions.
   - Resolution: documented that P0 rejects `--max-running-requests != 1` for
     all image/video serving plans. Higher values require a future scheduler.

2. Startup ordering is contradictory around plan selection and artifact
   preparation.
   - Status: accepted and resolved in revision 5.
   - Failure mode: implementation could try to compile/check artifacts before
     knowing which plan is selected, or select a plan based on warmup before
     preparing artifacts.
   - Resolution: rewrote startup as a candidate plan loop: resolve profile,
     build candidates, select by mode/core budget, prepare artifacts for the
     candidate, build/load/warmup, accept or try the next explicitly allowed
     fallback.

3. Local artifact route is inconsistent across route lists.
   - Status: accepted and resolved in revision 5.
   - Failure mode: local `response_format=url` could return URLs without a
     first-milestone serving route.
   - Resolution: moved `GET /v1/files/{file_id}/content` into first milestone
     routes when `LocalArtifactStore` is enabled.

### Revision 5 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added hard P0 rejection for `--max-running-requests != 1`.
  - Reworked server startup flow and plan selection into an explicit candidate
    loop.
  - Moved local artifact file route into first milestone route list.

Next planned review action:

- Fresh reviewer follow-up to verify Round 5 material findings are resolved.

## Round 6

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f410b-96b1-72b3-a615-72c025c868b8`

### Findings

#### Blocking

None.

#### Material

1. `data_url` can still bypass the deployment ArtifactStore/R2 contract.
   - Status: accepted and resolved in revision 6.
   - Failure mode: deployment could return base64 bytes directly despite the
     R2/ArtifactStore URL contract.
   - Resolution: added `--allow-data-url-responses`, defaulting to false for
     deployment. `response_format=data_url` returns `400 invalid_extra_body`
     unless this local/debug option is enabled.

2. Subprocess fallback is documented but no explicit subprocess runtime plan is
   registered.
   - Status: accepted and resolved in revision 6.
   - Failure mode: `--engine-mode subprocess` or explicit downgrade could
     depend on undocumented synthetic mutation of stage kinds.
   - Resolution: added an explicit `qwen_staged_subprocess` runtime plan with
     all stages using `DiffletStageKind.SUBPROCESS` and a sequential peak core
     model.

### Revision 6 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added `--allow-data-url-responses` and deployment rejection behavior for
    `response_format=data_url`.
  - Added explicit `qwen_staged_subprocess` runtime plan.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Documented the server flag required for `data_url` responses.

Next planned review action:

- Fresh reviewer follow-up to verify Round 6 material findings are resolved.

## Round 7

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f410b-96b1-72b3-a615-72c025c868b8`

### Findings

#### Blocking

None.

#### Material

None.

#### Optional

1. `response_format=auto` could have an explicit configuration knob.
   - Status: deferred optional cleanup.
   - Rationale: deployment behavior is already safe because URL/R2 is the
     default and `data_url` requires `--allow-data-url-responses`. A later
     implementation can add `--default-response-format {url,data_url}` or
     derive `auto` from `--artifact-store` and `--allow-data-url-responses`.

### Round 6 Verification

- Material 1, `data_url` bypass: resolved.
- Material 2, missing explicit subprocess plan: resolved.

### Final Review Status

- Latest fresh independent reviewer pass reports no blocking or material
  issues.
- Remaining item is optional configuration polish only.

Status: complete

## Post-Review Scope Correction

User clarified after review completion:

- P0 should not expose `data_url`; output should go through R2 URL only.
- P0 should not include `--allow-engine-downgrade`, a subprocess runtime plan,
  or `qwen_staged_subprocess`. The goal is all active stages resident in one
  shared-process worker for the 4-core target; if that cannot load, startup
  should fail rather than silently downgrade.
- P0 should not expose local file serving through `/v1/files/{file_id}/content`
  because R2 is the artifact backend.

Changes made after this clarification:

- Removed the P0 data URL flag/route behavior from the plan and contract.
- Removed the explicit Qwen subprocess runtime plan and automatic subprocess
  fallback language.
- Removed local artifact serving route and LocalArtifactStore P0 text.

Review status for the prior broader design was complete, but the final
checked-in scope is intentionally narrower than that reviewed version.

Status: active during narrowed-scope review; later completed in Round 11.

## Round 8

Scope for the next review:

- P0 output is R2 URL only. No `data_url` responses and no local `/v1/files`
  artifact route.
- P0 runtime is shared-process resident only. No subprocess runtime plan and no
  automatic engine downgrade.
- `--engine-mode` is limited to `{auto,resident}` for P0.

Next planned review action:

- Fresh independent reviewer pass for the narrowed P0 scope.

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f412e-0ef7-7200-87eb-28db95936ea9`

### Findings

#### Blocking

None.

#### Material

1. P0 tensor handoff contract is contradictory.
   - Status: accepted and resolved in revision 8.
   - Resolution: defined P0 `TensorRef` as an in-process object/tensor handle
     inside the shared worker. Request-scoped file handoff is future
     multi-process/debug work only. M2 Qwen milestone now requires in-process
     handoff and explicitly says not to preserve CLI `text.pt` / `latents.pt`
     in the resident path.

2. Milestones still steer P0 toward Flux/pipeline before Qwen shared resident.
   - Status: accepted and resolved in revision 8; superseded by the
     2026-07-09 engine unification update.
   - Resolution at the time: made Qwen shared-process resident serving the first
     real engine milestone after fake/test serving and moved the Flux-specific
     pipeline engine path out of the first runtime.
   - Superseding update: the runtime is now named
     `ResidentWorkerServingEngine`, and Flux uses the same resident worker
     engine instead of a separate engine.

3. Chat contract still normatively defines video behavior in the first API
   contract.
   - Status: accepted and resolved in revision 8.
   - Resolution: added P0 scope rules near the top: only image-output serving
     models are registered; `["video"]`, `["text"]`, and `["audio"]` return
     `400 unsupported_modality`; video request/response shapes are future P1+
     reference only.

### Revision 8 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Clarified `TensorRef` as in-process for P0 shared worker.
  - Reordered milestones so Qwen resident comes before optional Flux pipeline.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Added P0 image-only scope.
  - Marked video behavior as future P1+ and made `["video"]` unsupported in P0.

Next planned review action:

- Same reviewer follow-up to verify Round 8 material findings are resolved.

## Round 9

Reviewer: same `gpt-5.5` high explorer subagent
`019f412e-0ef7-7200-87eb-28db95936ea9`

### Findings

#### Blocking

None.

#### Material

1. Round 8 tensor handoff finding unresolved due stale file-based names in Qwen
   stage spec and `--work-dir`.
   - Status: accepted and resolved in revision 9.
   - Resolution: changed Qwen shared-process stage outputs from `text.pt` /
     `latents.pt` to semantic in-memory keys
     `encoder_hidden_states`, `encoder_hidden_states_mask`, and `latents`.
     Reworded `--work-dir` as scratch/log/temp storage only; P0 tensor handoff
     is in-memory.

2. Round 8 video finding partially unresolved due remaining P0 video modality
   mapping and handler responsibilities.
   - Status: accepted and resolved in revision 9.
   - Resolution: made P0 output modality mapping image-only. Moved `video_url`
     to future P1+ language and removed video formatting from P0 handler
     responsibilities.

3. `--compile-policy never` conflicts with P0 AOT artifact verification.
   - Status: accepted and resolved in revision 9.
   - Resolution: removed `never` from P0 `difflet serve` compile policy. P0
     allows only `require` and `auto`.

### Revision 9 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Replaced file-like Qwen stage outputs with semantic in-memory output keys.
  - Reworded `--work-dir` as scratch only.
  - Removed `compile-policy never`.
  - Removed video output formatting from P0 handler responsibilities.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Made output modality mapping image-only for P0.
  - Moved `video_url` to future P1+ wording.

Next planned review action:

- Same reviewer follow-up to verify Round 9 material findings are resolved.

## Round 10

Reviewer: same `gpt-5.5` high explorer subagent
`019f412e-0ef7-7200-87eb-28db95936ea9`

### Findings

#### Blocking

None.

#### Material

None.

#### Optional

1. Contract shared parameter tables still say some fields apply to
   `image/video`.
   - Status: accepted and resolved.
   - Resolution: adjusted P0 fields to image-only and marked `num_frames` as
     future video.

2. Model matrix labels Flux as `P0/P1 image candidate`.
   - Status: accepted and resolved.
   - Resolution: changed Flux status to `P1 optional`.

### Round 9 Verification

- Material 1, tensor handoff: resolved.
- Material 2, video P0 wording: resolved.
- Material 3, `compile-policy never`: resolved.

### Final Review Status

- Latest reviewer pass reports no blocking or material issues.
- Optional consistency comments were also addressed.

## Round 11

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f4172-2c30-76e0-88b1-c5a3d6455aa0`

### Findings

#### Blocking

None.

#### Material

1. P0 tensor handoff is still ambiguous across the process boundary.
   - Status: accepted and resolved in revision 11.
   - Resolution: defined P0 worker IPC as a single `RUN_GENERATION` command.
     The shared worker runs `prompt_encoder -> denoiser -> decoder`
     internally and returns only final bytes plus metadata. `StageRunRequest`
     and `StageRunResponse` are now worker-internal adapter interfaces.

2. Startup warmup can falsely pass if it relies on existing `app.load(...)`
   warmup.
   - Status: accepted and resolved in revision 11.
   - Resolution: added a serving-specific readiness requirement: after loading
     all Qwen apps in the shared worker, run a full-topology smoke request or
     representative stage smoke calls. Startup fails on any exception, invalid
     tensor shape, or invalid final image bytes. Existing
     `NeuronApplicationBase.warmup()` warnings are not sufficient for `/ready`.

3. P0 silently ignores non-text chat input parts.
   - Status: accepted and resolved in revision 11.
   - Resolution: changed prompt extraction contract to reject any non-text
     content item with `400 unsupported_input_modality`.

#### Optional

1. Open decision still asks whether first milestone should include Flux,
   Qwen-Image, or both.
   - Status: accepted and resolved in revision 11.
   - Resolution: removed the stale open decision.

2. Review document chronology/status is confusing.
   - Status: accepted and resolved.
   - Resolution: moved Round 11 after Round 10 and made the top-level status
     `complete`.

### Revision 11 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added worker-level `RUN_GENERATION` IPC semantics.
  - Marked stage run envelopes as worker-internal.
  - Added serving-specific full-topology smoke readiness.
  - Removed stale Flux/Qwen/both open decision.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Replaced silent non-text input ignoring with
    `400 unsupported_input_modality`.
- `docs/plans/2026-07-06-difflet-serving-engine.review.md`
  - Reordered the final review log and unified status.

### Round 11 Follow-Up Verification

- The fresh reviewer reported no blocking or material findings after revision
  11.
- Remaining optional review-log ordering issue was fixed.

### Final Review Status

- Latest fresh independent reviewer pass reports no blocking or material
  issues.
- Status before Round 16: complete.

## Round 16

Reviewer: fresh `gpt-5.5` high explorer subagent
`019f44d0-75e6-76d0-8699-033183b68bca`

Scope:

- Re-review the latest Qwen-Image + Flux MVP after unifying both under
  `ResidentWorkerServingEngine`.
- Check for stale `PipelineServingEngine` assumptions, parent-process model
  loading, P0/future boundary drift, lifecycle risks, and verification gaps.

### Main-Agent Pre-Review Local Fix

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Fixed the factory pseudocode before delegation: P0 now accepts only
    `RESIDENT_WORKER` stages and rejects rotating plans until future work.
  - Clarified `DiffletRuntimePlan.engine_mode` as `resident` for P0, with
    `rotating_resident` future-only.

### Findings

#### Blocking

1. Timeout/cancel can release the single execution slot while the worker may
   still be running.
   - Status: pending user confirmation.
   - References:
     - `docs/design/difflet_serving/engine.md`, generic `generate(...)` flow.
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, request timeout
       semantics that allow an uninterruptible worker to finish internally.
   - Failure mode: HTTP can return `504`, release the ticket, and admit a
     second `RUN_GENERATION` while the same resident worker is still executing,
     violating `max_running_requests=1`.
   - Reviewer fix: define a terminal worker state machine. On timeout, do not
     release the execution slot until `CANCEL_ACK`, worker completion/discard,
     or worker termination. If HTTP returns `504` first, mark the worker
     busy/unavailable and reject or hold new generation until safe.

#### Material

1. P0 startup still has generic "try next candidate" language that can
   reintroduce fallback behavior.
   - Status: pending user confirmation.
   - References:
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, plan selection
       algorithm.
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, startup lifecycle.
   - Failure mode: implementers may add fallback candidate iteration even
     though P0 must fail if the selected shared-process resident plan cannot
     load/smoke.
   - Reviewer fix: state P0 candidate selection produces exactly one candidate:
     the active profile's shared-process resident plan. Move candidate
     iteration to future-only text.

2. Artifact preparation ownership conflicts with worker-owned serving
   orchestrators.
   - Status: pending user confirmation.
   - References:
     - `docs/design/difflet_serving/architecture.md`, worker-owned serving
       orchestrators.
     - `docs/design/difflet_serving/architecture.md`, startup flow.
     - `docs/design/difflet_serving/architecture.md`, `ServingModelOrchestrator`
       protocol.
   - Failure mode: implementers may construct the serving orchestrator in the
     parent and accidentally load Trainium objects there, or defer artifact
     prep into the worker after the lifecycle says it happens before HTTP bind.
   - Reviewer fix: split a parent-safe artifact/profile planner from the
     worker-owned live orchestrator. Parent-safe code owns download, compile
     plan, and artifact validation; worker-owned orchestrator owns only
     `load`, `smoke`, `generate`, and `shutdown`.

3. Flux request validation is not specified to the same level as Qwen.
   - Status: pending user confirmation.
   - Reference:
     - `docs/design/difflet_serving/chat_completions_contract.md`, model
       support and value-limit sections.
   - Failure mode: Flux may accept invalid or unsupported `extra_body` values
     differently from Qwen, causing worker crashes or inconsistent 4xx/5xx
     behavior.
   - Reviewer fix: add a Flux P0 value-limits table covering prompt length
     policy if applicable, steps default/bounds, finite guidance, seed range,
     TTL bounds, `output_format=png`, `response_format=url|auto`, and explicit
     rejection of `true_cfg_scale`, `num_frames`, CFG parallel, and unsupported
     model-specific fields.

#### Optional

1. Future video wording is still presented as a milestone.
   - Status: pending user confirmation.
   - Reference:
     - `docs/plans/2026-07-06-difflet-serving-engine.md`, `M4: Add Video
       Routes`.
   - Reviewer fix: rename this to `Future Work: Video Serving` to reduce P0
     scope creep.

### Reviewer Positive Checks

- No stale `PipelineServingEngine` language found.
- No direct parent-process loaded-pipeline requirement found.
- Current docs consistently say Flux and Qwen use
  `ResidentWorkerServingEngine`, with Flux's `DiffletPipeline` loaded inside
  the worker.

### Proposed Revision 16 Edits

- Add worker timeout/terminal-state semantics to `engine.md` and the plan:
  timeout returns `504` to the client but does not free the worker slot until
  `completed`, `cancel_ack`, `terminated`, or `failed`.
- Replace P0 candidate iteration with exactly one active resident candidate;
  move multi-candidate fallback language to future work.
- Split parent-safe planner responsibilities from worker-owned live
  orchestrator responsibilities in `architecture.md` and the plan.
- Add Flux P0 value limits to `chat_completions_contract.md`.
- Optionally rename the video milestone to `Future Work: Video Serving`.

Next planned review action:

- Await user confirmation, apply accepted edits, then send the revision back to
  the same Round 16 reviewer for follow-up.

### User Decisions During Revision 16

- `request_timeout=300s` is the external request timeout and includes queue
  wait plus generation. `queue_timeout=30s` remains the shorter admission
  timeout.
- `CANCEL_ACK` is produced by the worker runtime, not by the engine. The engine
  sends `CANCEL`, the worker runtime sets a request-scoped cancellation signal,
  and orchestrators/stage adapters check that signal at safe points.
- Model handles should not grow model-specific cancel protocols. They receive a
  single `WorkerRequestContext` / cancellation signal hook.

### Revision 16 Partial Changes

- `docs/design/difflet_serving/engine.md`
  - Added timeout recovery with worker `RECOVERING` state, `CANCEL_ACK`, and
    process-level terminate/restart/reload/smoke behavior.
  - Clarified that cancellation is a worker-runtime signal checked at safe
    points, not handle-specific control logic.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added `--worker-cancel-timeout` and `--worker-restart-timeout`.
  - Documented `request_timeout` as external wall-clock time from HTTP
    admission.
  - Added `CancellationSignal`, `WorkerRequestContext`, and worker-local
    cancellation rules.
- `docs/design/difflet_serving/architecture.md`
  - Updated worker-owned orchestrator APIs to
    `generate(request, context)`.

## Post-Review MVP Scope Update

User clarified after the final review that MVP should run both Qwen-Image and
Flux text-to-image serving.

Changes made after review completion:

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - MVP model scope now includes Qwen-Image and Flux.
  - `ResidentWorkerServingEngine` is the single P0 runtime engine for both
    Qwen-Image and Flux.
  - M2 is now "MVP Text-To-Image Engines" and includes both Qwen shared-process
    resident serving and worker-owned Flux serving.
  - Flux remains one active model/profile per server process and must pass the
    same artifact, load, smoke, request validation, R2 output, and failure-path
    contracts.
  - The plan notes that Flux's current registry default is `tp=8, cp=1`; a
    4-core Flux deployment requires an explicit supported Flux profile and
    matching artifacts, otherwise startup should fail.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Flux status changed to MVP target.
  - Model matrix now marks `black-forest-labs/FLUX.1-dev` as P0 target.

Review status:

- The prior review remains complete for the narrowed Qwen-first P0 scope.
- This post-review scope expansion has not yet received a fresh reviewer pass.

## Round 12

Reviewer: fresh `gpt-5.5` high explorer subagent
`019f4215-353d-7bb0-b0d6-69bad04f523f`

Scope:

- Re-review after narrowing P0 to one active 4-core text-to-image profile.
- Validate that the design is consistent with one FastAPI process, one
  Trainium resident worker process, Qwen-Image primary target, AOT before
  ready, R2 URL output, `max_running_requests=1`, and no P0
  subprocess/rotating fallback.

### Findings

#### Blocking

None.

#### Material

1. Shared-process Qwen feasibility is still the central unproven dependency,
   but milestones do not make it an explicit go/no-go gate.
   - Status: accepted and resolved in revision 12a.
   - Failure mode: the only in-scope 4-core runtime can fail to load because
     prompt encoder, denoiser, and VAE may not coexist under one immutable
     Neuron process env.
   - Reviewer fix: add a pre-M2 acceptance gate on real 4-core Trainium: load
     all three Qwen stage apps in one worker, run a full-topology smoke request,
     verify valid image bytes, and document that failure blocks P0 rather than
     enabling subprocess/rotating fallback.
   - Resolution: clarified Neuron placement semantics: cores are allocated to a
     process, and one process can load multiple models in its NeuronCore group.
     Added an M2 go/no-go gate requiring Qwen three-stage load plus
     full-topology smoke on the real 4-core target; failure blocks P0.

2. `--compile-policy require` may accept legacy staged artifact directories
   without a manifest, weakening the AOT/profile identity guarantee.
   - Status: accepted and resolved in revision 12b.
   - Failure mode: serving can load stale or wrong-toolchain NEFFs that match
     directory names but not the selected profile/toolchain.
   - Reviewer fix: require manifests with serialized compile spec and
     toolchain fingerprint for P0 production; gate legacy dirs behind an
     explicit dev/transition flag such as `--allow-legacy-artifacts`.
   - Resolution: production `compile-policy=require` now requires a matching
     serving manifest, serialized compile spec, and toolchain fingerprint.
     Manifestless legacy directories require explicit
     `--allow-legacy-artifacts` and log a non-production warning.

3. Artifact upload failure after successful generation is not specified in the
   API error contract.
   - Status: accepted and resolved in revision 12b.
   - Failure mode: valid image bytes may be produced, then R2 upload/presign
     fails; implementation may leak temp files, return a generic 500, or
     incorrectly fall back to local/data URL output.
   - Reviewer fix: add an error code/status such as
     `503 artifact_store_unavailable` or `502 artifact_upload_failed`; require
     cleanup of local temp data and explicitly forbid local/data URL fallback.
   - Resolution: plan and chat contract now define
     `503 artifact_store_unavailable` and `502 artifact_upload_failed`, require
     request-local cleanup, and forbid fallback to data URLs, local paths, raw
     filesystem paths, or inline bytes.

4. P0-critical timeout, health, cancellation, and cleanup behavior is specified
   earlier but deferred to M4 hardening.
   - Status: accepted and resolved in revision 12b.
   - Failure mode: an M2 Qwen server could hang requests indefinitely, keep
     `/ready` true after worker death, fail to drain cleanly, or leave queued
     requests/work dirs unresolved.
   - Reviewer fix: move minimum timeout enforcement, worker-death readiness
     behavior, queue cleanup, and idempotent shutdown into M2 acceptance
     criteria. Leave deeper polish in M4.
   - Resolution: M2 now includes bounded queue/`queue_timeout`,
     `request_timeout`, terminal cleanup, `/ready` failure on worker death,
     draining behavior, and idempotent engine/worker shutdown. M4 is limited to
     advanced recovery, dashboards, soak tests, audits, and refined
     cancellation.

5. Chat request contract needs a stricter parsing allowlist and conflict rule.
   - Status: accepted and resolved in revision 12b.
   - Failure mode: unsupported content schemas, unsupported top-level fields,
     or duplicate flattened/`extra_body` generation fields may be silently
     ignored or inconsistently applied.
   - Reviewer fix: define exact accepted content part shapes, explicit
     top-level allowlist/rejectlist, and deterministic conflict precedence for
     duplicate flattened fields and `extra_body`.
   - Resolution: chat contract now defines exact P0 text content item shape,
     rejects malformed/unknown content items, restricts top-level fields to an
     allowlist, rejects unsupported chat fields, and requires generation
     parameters to live in `extra_body`.

#### Optional

1. Runtime observability is strong for startup logs but thin for request-level
   operations.
   - Status: accepted and resolved in revision 12b.
   - Reviewer fix: require structured per-request logs/metrics for queue wait,
     generation duration, artifact upload latency, worker pid, worker health
     transitions, timeout/cancel outcome, and final error code.
   - Resolution: added required request progress logs and minimum metrics.

### Proposed Revision 12 Changes

- Add a pre-M2 shared-process Qwen feasibility gate. Resolved in revision 12a.
- Tighten P0 artifact manifest requirements and mark legacy staged dirs as
  dev/transition-only behind an explicit flag.
- Add R2/artifact upload failure handling to the plan and chat error contract.
- Move minimum timeout/readiness/worker-death/shutdown guarantees into M2.
- Tighten chat parsing with exact content schemas, field allowlists, and
  `extra_body` conflict precedence.
- Optionally add request-level structured logs/metrics.

### Revision 12a Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added Neuron placement semantics from the provided docs: NeuronCores are
    process-owned, processes do not share NeuronCores, and a single process can
    load multiple models into its assigned NeuronCore group.
  - Reframed shared-process Qwen as valid from a core-placement perspective,
    while still requiring load/smoke validation for HBM and artifact/runtime
    compatibility.
  - Added an M2 go/no-go gate: the real 4-core worker must load prompt encoder,
    denoiser, and decoder, run a full-topology smoke request, and produce valid
    PNG bytes before readiness.

### Revision 12b Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added `--allow-legacy-artifacts` as a development/transition-only flag.
  - Required production serving manifests with serialized compile spec and
    toolchain fingerprint for `compile-policy=require`.
  - Added artifact upload/presign failure handling and no-fallback behavior.
  - Added request-level structured logs and minimum metrics.
  - Moved timeout, readiness-on-worker-death, queue cleanup, and idempotent
    shutdown into M2 acceptance; narrowed M4 to advanced hardening.
  - Changed OpenAI handler responsibilities so P0 generation parameters come
    from `extra_body`, not flattened top-level extras.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Added exact text content item schema and strict content validation.
  - Added top-level allowlist/rejectlist and duplicate-field behavior.
  - Added `extra_body` validation rules for aliases, unknown fields, and
    model-specific fields.
  - Added artifact store/upload error codes and no-fallback behavior.

## User Decisions Needed

- None before Round 13 follow-up verification.

## Round 13

Reviewer: same `gpt-5.5` high explorer subagent
`019f4215-353d-7bb0-b0d6-69bad04f523f`

### Findings

#### Blocking

None.

#### Material

1. Multi-profile startup is still in the P0 CLI/build surface, conflicting with
   the narrowed single active 4-core profile scope.
   - Status: accepted and resolved in revision 13.
   - Failure mode: implementers may build `--profile`,
     `--serving-profiles`, and `eager-all` for P0, adding matching/loading
     paths that violate the one-active-profile MVP and increase HBM risk.
   - Reviewer fix: mark multi-profile startup and `eager-all` as future-only.
     For P0, reject `--profile`, `--serving-profiles`, and
     `--profile-load-policy != single-active`; document exactly one loaded
     `ServingProfile`.

2. M2 acceptance includes P0-critical safety behavior, but M2 verification only
   checks a successful curl/image output.
   - Status: accepted and resolved in revision 13.
   - Failure mode: queue timeout, request timeout, worker-death readiness,
     draining cleanup, and artifact upload failure can regress while M2 still
     appears to pass through a happy-path PNG.
   - Reviewer fix: add M2 verification using fake/controllable engine or worker
     harnesses for queue full, queue timeout, request timeout, worker death
     flipping `/ready` to 503, shutdown draining queued requests, and R2
     upload/presign failure returning the documented artifact error without
     fallback.

#### Optional

None.

### Previous Findings

- Round 12 material 1, shared-process Qwen feasibility gate: resolved.
- Round 12 material 2, manifestless legacy artifacts: resolved.
- Round 12 material 3, artifact upload failure: resolved.
- Round 12 material 4, timeout/health/cancellation/cleanup: spec resolved, but
  M2 verification coverage remains material finding 2 above.
- Round 12 material 5, chat parsing allowlist/schema/conflicts: resolved.

### Planned Revision 13 Changes

- Mark multi-profile startup, `--profile`, `--serving-profiles`, and
  `eager-all` as future-only; P0 rejects them and loads exactly one
  `ServingProfile`.
- Add M2 verification requirements for P0 failure paths and safety behavior,
  not only successful image generation.

### Revision 13 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Renamed multi-profile startup as future-only.
  - P0 now explicitly rejects `--profile`, `--serving-profiles`, and
    `--profile-load-policy` values other than `single-active`.
  - Documented exactly one loaded `ServingProfile` for P0.
  - Added M2 deterministic failure-path verification for queue full,
    `queue_timeout`, `request_timeout`, worker death/readiness, shutdown
    draining, and artifact upload/presign errors.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Clarified that P0 has exactly one `ServingProfile`; multi-profile request
    matching is future-only.

### Round 13 Follow-Up Verification

- The same reviewer reported no blocking or material issues.
- Multi-profile startup conflict: resolved.
- M2 failure-path verification gap: resolved.
- Remaining future-extension sections for rotating, subprocess debug harnesses,
  Flux, video, and multi-profile are marked out of P0 and do not conflict with
  the current build plan.

Next planned review action:

- Fresh independent reviewer pass for missed blocking/material issues.

## Round 14

Reviewer: fresh independent `gpt-5.5` high explorer subagent
`019f423c-df55-7b40-be16-3a835b5e4629`

### Findings

#### Blocking

None.

#### Material

1. Prompt token-length policy is missing.
   - Status: accepted and resolved in revision 14.
   - Failure mode: overlong prompts may be silently truncated, shape-fail, or
     behave differently from the HTTP contract.
   - Reviewer fix: add `max_prompt_tokens` / encoder sequence length to the
     serving profile or adapter contract; define whether P0 rejects overlength
     prompts with `400 prompt_too_long` or intentionally truncates after
     templating/tokenization. Add boundary tests.

2. `extra_body` value ranges are under-specified.
   - Status: accepted and resolved in revision 14.
   - Failure mode: invalid values like `num_inference_steps=0`, negative TTL,
     non-finite `guidance_scale`, unsupported `output_format`, or out-of-range
     seeds can leak into worker code and produce crashes or incorrect 5xx
     responses instead of `400 invalid_extra_body`.
   - Reviewer fix: add P0 adapter validation limits for Qwen: positive bounded
     steps, finite guidance, supported seed range, positive TTL within server
     max, `output_format=\"png\"` only, and profile-bound shape/parallel fields
     matching exactly. Add unit tests for invalid values.

### Prior Review Status

- Round 13 multi-profile scope: resolved.
- Round 13 M2 failure-path verification: resolved.
- Narrowed runtime scope remains consistent according to the fresh reviewer.

### Planned Revision 14 Changes

- Add prompt token/bucket policy to the contract and serving profile.
- Add Qwen P0 `extra_body` value-range validation.
- Add M2/unit test requirements for prompt length and invalid generation
  values.

### Revision 14 Changes

- `docs/design/difflet_serving/chat_completions_contract.md`
  - Added prompt token-length policy: P0 validates after model prompt
    templating/tokenization and returns `400 prompt_too_long`; no silent
    truncation.
  - Added Qwen P0 value limits for prompt bucket, inference steps, guidance,
    seed, artifact TTL, output format, and response format.
  - Added `prompt_too_long` to the error contract.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added `text_seq_len` / `max_prompt_tokens` to `ServingProfile`.
  - Added `--max-inference-steps`, `--max-artifact-ttl-seconds`, and
    `--max-prompt-tokens`.
  - Added OpenAI handler duties for prompt-token validation and value-range
    validation before engine admission.
  - Documented Qwen `seq256` as a serving admission bucket.
  - Added M2 verification for prompt boundary cases and invalid `extra_body`
    values.

Next planned review action:

- Same fresh reviewer follow-up to verify Round 14 material findings are
  resolved.

## Round 15

Reviewer: same fresh independent `gpt-5.5` high explorer subagent
`019f423c-df55-7b40-be16-3a835b5e4629`

### Findings

#### Blocking

None.

#### Material

1. The top-level MVP decisions still conditionally allow multi-profile
   residency.
   - Status: accepted and resolved in revision 15.
   - Failure mode: implementers could treat multi-profile startup as
     conditionally in scope for P0 after warmup, reintroducing HBM/runtime risk.
   - Reviewer fix: replace the top-level MVP sentence with: "Multiple AOT
     profile combinations may be precompiled and stored on disk, but P0 loads
     exactly one active `ServingProfile`; multi-profile loading/warmup is
     future-only."
   - Resolution: updated the MVP decisions section with that rule.

### Previous Findings

- Round 14 prompt token-length policy: resolved.
- Round 14 `extra_body` value ranges: resolved.

### Revision 15 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Removed the conditional wording that allowed multi-profile loading when
    load/warmup proved it fit.
  - MVP decisions now state P0 loads exactly one active `ServingProfile` and
    multi-profile loading/warmup is future-only.

Next planned review action:

- Same fresh reviewer follow-up to verify Round 15 material finding is
  resolved.

### Round 15 Follow-Up Verification

- The same fresh reviewer reported no blocking or material issues.
- Round 15 multi-profile wording: resolved.
- The narrowed P0 scope is consistent across the reviewed artifacts: one active
  4-core Qwen text-to-image profile, one FastAPI process, one resident Trainium
  worker, AOT before ready, R2 URL output, `max_running_requests=1`, and no P0
  subprocess/rotating/multi-profile fallback.

### Final Review Status

- Latest fresh independent reviewer pass reports no blocking or material
  issues.
- Status: complete.

## Round 16

Reviewer: fresh `gpt-5.5` high explorer subagent
`019f459f-4847-7471-94e2-586b53a7fa9e`

Context:

- Reopened review after syncing the design with latest `origin/main` behavior
  around model registry defaults, Flux `tp_degree=8`, serving startup overrides,
  and P0 `--sp` rejection.
- Design artifacts reviewed:
  - `docs/design/difflet_serving/architecture.md`
  - `docs/design/difflet_serving/engine.md`
  - `docs/design/difflet_serving/chat_completions_contract.md`
  - `docs/plans/2026-07-06-difflet-serving-engine.md`

### Findings

#### Blocking

None.

#### Material

1. Serve flag parsing can reintroduce hard-coded defaults.
   - Status: accepted and revised in revision 16.
   - Failure mode: if `difflet serve` reuses existing CLI parallel flags,
     omitted `--cp-degree` and `--cp-mode` become `1` / `gather_kv`, so serving
     cannot distinguish omitted values from explicit startup overrides. That
     violates the registry-default requirement.
   - Reviewer fix: serving-only profile-bound override flags should default to
     `None`, then resolve from `ModelEntry.default_parallel` /
     `ModelEntry.default_shape`. Add tests with non-global CP/CP-mode defaults.

2. Timeout contract references missing request state.
   - Status: accepted and revised in revision 16.
   - Failure mode: engine pseudocode uses `request.received_at`, but
     `DiffletGenerateRequest` does not define a received/deadline field.
     Implementers may accidentally start `request_timeout` after queue
     admission and exclude queue wait.
   - Reviewer fix: add `received_at_monotonic` / `deadline_monotonic`, or state
     that the engine stamps this at HTTP admission and stores it in the
     admission ticket.

3. Per-request model matching is too ambiguous with broad detectors.
   - Status: accepted and revised in revision 16.
   - Failure mode: startup detectors such as Flux's broad `"flux"` detector
     could allow a single-model server to accept an unregistered or wrong
     request `model` instead of returning `model_not_served`.
   - Reviewer fix: define `ServingProfile.accepted_model_ids` as startup
     `model_id` plus explicit `ModelEntry.hf_paths` or serving aliases. Use
     detectors only for startup resolution, not request-time acceptance.

4. Qwen prompt rejection can be lost when extracting CLI code.
   - Status: accepted and revised in revision 16.
   - Failure mode: current Qwen CLI tokenization uses `truncation=True`; if the
     helper is moved unchanged into common serving code, over-bucket prompts are
     silently truncated instead of returning `400 prompt_too_long`.
   - Reviewer fix: serving should tokenize templated prompts without truncation
     first, reject over bucket, then pad for execution.

5. Engine protocol differs between design docs.
   - Status: accepted and revised in revision 16.
   - Failure mode: `engine.md` defines async `start()`, `health()`, and
     `shutdown()`, while the plan has a different sync protocol. This can split
     FastAPI lifespan, `/health`, `/ready`, and worker startup implementations.
   - Reviewer fix: choose one protocol across artifacts. Prefer async
     `start()`, `health() -> EngineHealth`, and `shutdown()` because worker IPC
     and recovery are async.

#### Optional

None.

### User Decision

- User asked to continue the iterative review. Treat this as approval to apply
  all five non-controversial material fixes.

### Revision 16 Changes

- Make serving CLI profile-bound flags default to `None` and document that
  existing CLI flag helpers must not be reused when they erase omission state.
- Clarify that `difflet serve --tp-degree/--cp-degree` overrides the
  model-level serving profile, while stage-specific core differences are
  adapter metadata rather than separate P0 CLI flags.
- Add request/admission timestamp or deadline ownership to the engine contract.
- Add `ServingProfile.accepted_model_ids` and request-time exact/alias matching
  rules.
- Add Qwen no-truncation prompt validation before execution padding.
- Align all engine protocol examples on async `start()`, `health()`, and
  `shutdown()`.

Changed files:

- `docs/design/difflet_serving/architecture.md`
  - Changed `WorkerRequestContext.deadline` to `deadline_monotonic`.
  - Added serve-specific profile override/parser guidance and clarified that
    `difflet serve --tp-degree/--cp-degree` overrides the model-level profile,
    while stage core differences are adapter metadata.
- `docs/design/difflet_serving/engine.md`
  - Added engine-owned admission timestamp/deadline semantics.
  - Updated pseudocode to create `received_at_monotonic` and
    `deadline_monotonic` before queue admission.
- `docs/design/difflet_serving/chat_completions_contract.md`
  - Added `ServingProfile.accepted_model_ids` request-time matching.
  - Forbid request-time detector matching.
  - Added no-truncation prompt length validation before execution padding.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added `accepted_model_ids` to `ServingProfile`.
  - Updated engine protocol to async `start()`, `generate()`, `health()`, and
    `shutdown()`.
  - Clarified `DiffletGenerateRequest` does not own timeout state.
  - Added serve-specific profile parser helper with `default=None`.
  - Added tests for registry defaults vs parser defaults, accepted model ids,
    timeout including queue wait, and Qwen no-truncation validation.

Rejected feedback:

- None.

Next planned review action:

- Same reviewer follow-up to verify Round 16 material findings are resolved and
  identify any new blocking/material issues.

### Round 16 Follow-Up

Reviewer: same `gpt-5.5` high explorer subagent
`019f459f-4847-7471-94e2-586b53a7fa9e`

#### Blocking

None.

#### Material

1. `accepted_model_ids` can still allow sibling checkpoints, not aliases.
   - Status: accepted and revised in revision 16.1.
   - Failure mode: including all `ModelEntry.hf_paths` by default can make a
     server loaded with one checkpoint, such as `FLUX.1-dev`, accept a request
     for a sibling checkpoint such as `FLUX.1-schnell` and return output from
     the wrong weights.
   - Reviewer fix: `accepted_model_ids` defaults to the exact startup
     `model_id` plus explicitly declared same-checkpoint aliases only.

2. Engine timeout pseudocode still drops the worker deadline and uses the wrong
   request id field.
   - Status: accepted and revised in revision 16.1.
   - Failure mode: `run_one(request)` / `worker_rpc.run_generation(request)` do
     not receive the admission deadline, and timeout recovery used `request.id`
     instead of `request.request_id`.
   - Reviewer fix: pass the admission ticket or deadline into `run_one`, send
     `deadline_monotonic` to worker IPC, and use `request.request_id`.

#### Previous Findings

- Serve parser defaults: resolved.
- Timeout state missing: conceptually resolved; revised again for concrete
  pseudocode deadline propagation.
- Broad request-time detectors: conceptually resolved; revised again to avoid
  default sibling-checkpoint aliases.
- Qwen no-truncation prompt validation: resolved.
- Engine protocol mismatch: resolved.

### Revision 16.1 Changes

- `docs/design/difflet_serving/chat_completions_contract.md`
  - `accepted_model_ids` now defaults to exact startup model id plus explicit
    same-checkpoint aliases only.
  - Explicitly says not to include every `ModelEntry.hf_paths` value by default.
- `docs/design/difflet_serving/engine.md`
  - `run_one` receives the admission ticket and passes
    `ticket.deadline_monotonic` to worker IPC.
  - Timeout recovery uses `request.request_id`.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Same `accepted_model_ids` sibling-checkpoint rule.
  - Clarifies the admission ticket/deadline is passed through `run_one` into
    worker `WorkerRequestContext`.
  - Adds model matching tests for sibling checkpoints from the same registry
    entry.

Next planned review action:

- Same reviewer follow-up to verify Round 16.1 material findings are resolved.

### Round 16.1 Follow-Up

Reviewer: same `gpt-5.5` high explorer subagent
`019f459f-4847-7471-94e2-586b53a7fa9e`

#### Blocking

None.

#### Material

1. Flux P0 request validation is still under-specified.
   - Status: accepted and revised in revision 16.2.
   - Failure mode: Qwen has concrete P0 value limits, but Flux is also a P0
     target and lacked concrete limits for prompt bucket, steps, guidance, seed,
     output format, response format, and artifact TTL. Implementers could miss
     prompt overflow or let invalid values reach the pipeline/runtime.
   - Reviewer fix: add Flux P0 value limits mirroring Qwen and add Flux
     invalid-value and prompt-boundary tests.

#### Previous Findings

- `accepted_model_ids` sibling checkpoint issue: resolved.
- Timeout pseudocode deadline/request id issue: resolved.

### Revision 16.2 Changes

- `docs/design/difflet_serving/chat_completions_contract.md`
  - Added Flux P0 value limits for prompt bucket, steps, guidance, seed,
    artifact TTL, output format, response format, and true-CFG/negative prompt
    rejection.
- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Added Flux prompt admission rule for the existing
    `max_sequence_length=512` path.
  - Added Flux P0 value limits and Flux prompt/invalid-value test coverage.

Next planned review action:

- Same reviewer follow-up to verify Round 16.2 material finding is resolved.

### Round 16.2 Follow-Up

Reviewer: same `gpt-5.5` high explorer subagent
`019f459f-4847-7471-94e2-586b53a7fa9e`

#### Blocking

None.

#### Material

None.

#### Optional

1. Worker IPC prose still reads as if `RUN_GENERATION` sends only
   `DiffletGenerateRequest`.
   - Status: accepted and revised as optional cleanup.
   - Reviewer fix: say `RUN_GENERATION` carries the request plus engine-owned
     deadline metadata used to build `WorkerRequestContext`.

### Revision 16.3 Changes

- `docs/plans/2026-07-06-difflet-serving-engine.md`
  - Clarified `RUN_GENERATION` sends `DiffletGenerateRequest` plus
    `deadline_monotonic`, and the worker runtime uses that to build
    `WorkerRequestContext`.
- `docs/design/difflet_serving/engine.md`
  - Applied the same `RUN_GENERATION` request plus `deadline_monotonic`
    clarification.

Next planned review action:

- Fresh independent reviewer pass for missed blocking/material issues.
