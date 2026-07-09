# Difflet Serving Engine Design Review

Status: waiting for user confirmation on Round 16 edits

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

- Fresh independent reviewer pass for missed blocking/material issues.

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
