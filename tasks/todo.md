- [x] Draft Difflet serving engine plan under docs/plans, based on vLLM-Omni stage factory pattern.
- [x] Clarify generic stage roles vs runtime strategies; map Qwen-Image to prompt_encoder -> denoiser -> decoder.
- [x] Promote ResidentWorkerServingEngine as the primary server runtime and document topology template / fallback resolution policy.
- [x] Add startup artifact preparation and Trainium compile lifecycle to server/app startup plan.
- [x] Clarify serving must reuse existing Difflet download/AOT compile/cache/CLI and DiffletPipeline primitives.
- [x] Map existing common CLI flags to serving lifecycle defaults, request defaults, cache impact, and Trainium core impact.
- [x] Clarify compile stages vs generate stages and first-milestone fixed-shape serving profile.
- [x] Add current README/code evidence for shape-specific compile artifacts and compare with text serving buckets.

Review:
- Done: updated plan to make resident workers the serving target, and subprocess only an optional migration harness.
- Done: documented compile and generate as the same stage topology with different actions.
- Done: documented first serving pod as one model/topology/shape/parallel/dtype/toolchain profile.
- Done: documented existing evidence that shape is part of cache identity, plus which staged encoder artifacts can be shared across shapes.
- Done: updated MVP serving design to use one FastAPI process plus one shared-process Trainium worker for the 4-core target.
- Done: clarified that the engine returns bytes, while the OpenAI handler writes artifacts through `ArtifactStore`/R2 and returns `image_url.url`.
- Done: moved rotating resident to a future extension point until unload/restart semantics are implemented.
- Done: removed `download-policy require`; startup download policy is `auto` or `never`.
- Done: narrowed P0 serving scope to R2 URL output only, no `data_url`, no local `/v1/files` route, and no subprocess runtime fallback.
- Done: updated MVP model scope to include both Qwen-Image staged serving and worker-owned Flux serving behind `ResidentWorkerServingEngine`, with one active model/profile per server process.
- Done: narrowed P0 image request-time profile matching to `height`/`width`; `num_frames` is future video-only and non-null values are rejected by image adapters. `tp_degree`, `cp_degree`, `cp_mode`, `cfg_parallel`, and `sp_enabled` are documented as `difflet serve` startup-only fields that return `400 invalid_extra_body` if present in request `extra_body`.
- Done: documented `response_format` and `artifact_ttl_seconds` as ignored compatibility fields; P0 always returns an ArtifactStore/R2 URL with server-configured TTL, keeps those fields out of worker-facing requests, and the handler must use `put_bytes(...) -> get_url(ref)` rather than returning internal `ArtifactRef.uri`.
- Done: clarified that top-level known Difflet generation/shape/startup/runtime fields return `400 invalid_extra_body`, and that "adapter" means a model-specific serving implementation under `difflet/serving/orchestrators/*` wired by registry factories.
- Done: marked the split design docs as authoritative for P0, moved per-stage core math out of the Qwen P0 support table, and defined `worker_restart_timeout` failure behavior.
- Done: added Qwen shared-worker co-load/smoke gating, caller-cancellation recovery ownership rules, and explicit image-model `num_frames` rejection to the split P0 serving docs.
- Done: clarified that `--num-frames` is future video-only for startup, added `artifact_store_timeout`, and added `invalid_prompt` for missing/empty prompts.
- Done: implemented the initial P0 serving code path: serving registry/profile resolution, OpenAI chat request normalization, ArtifactStore/R2 boundary, resident worker engine IPC skeleton, Flux/Qwen serving adapters, FastAPI app factory, and `difflet serve` CLI wiring.
- Done: simplified public `difflet serve` flags to hide queue/timeout/artifact internals, moved common serving metadata into per-model modules under `difflet/common/registry/`, and kept `difflet/serving/model_registry.py` as the serving overlay.
- Done: aligned P0 serving implementation with the split design docs: serve now uses serve-specific profile flags, chat/completions validates omitted model/modalities/top-level fields per contract, Qwen shared-worker smoke runs a real one-step generation, and in-flight worker timeout/cancel paths recover before releasing the resident worker slot.
- Done: tightened the latest review findings: `CANCEL_ACK` is the only clean in-flight cancel terminal state, late `generation_ok` after cancel forces worker restart, idle worker death updates readiness, and `difflet serve --help` only advertises P0 image models/flags.
- Done: added parent-side model request validator factories, wired validation before worker admission, implemented Flux/Qwen prompt bucket validators, extracted shared Qwen serving preflight helpers under `difflet/common/orchestrators/qwen_image.py`, strengthened Qwen artifact readiness checks, and aligned Worker IPC docs with the implemented startup/cancel queues.
- Done: fixed latest doc/implementation mismatches: Flux preflight now fail-closes on missing compiled cache under `CompilePolicy.NEVER`, Qwen artifact readiness requires NEFF/metaneff artifacts rather than manifest-only dirs, `/health` returns 503 when unhealthy, malformed `extra_body` returns `400 invalid_extra_body`, and docs were narrowed for P0 shutdown/Flux smoke/file layout/model matching.
- Done: addressed fresh doc/code review mismatches: Qwen serving artifacts now require a profile-matched serving marker, Qwen revision is forwarded through staged subprocess compile/load helpers, engine docs describe in-flight worker 5xx recovery, and architecture docs distinguish legacy artifact `stage_id` from generic serving stage roles.
- Done: addressed follow-up fresh-agent findings: chat prompt extraction now uses only the last user message, queue wait returns `504 request_timeout` when the request deadline expires before the worker slot opens, Flux serving uses a common orchestrator helper for pipeline/artifact checks, serving registry metadata exposes stage topology/roles, and docs now match ASGI lifespan startup order plus incremental CLI/common migration.
- [x] Inspect the serving entry point, model/profile requirements, and runtime dependencies for Trainium deployment.
- [x] Connect to `16.51.176.130` and inventory the instance hardware, OS, Neuron runtime, disk, and existing model caches.
- [x] Transfer the current clean branch and install the project without overwriting unrelated remote state.
- [x] Compile Qwen-Image, run one real inference, and exercise resident-worker startup through its readiness gate.
- [x] Record exact startup logs, inference outcome, remaining blockers, and verification commands.
- Rationale: serving startup must load fixed-shape Trainium NEFF artifacts before accepting requests.
- Verification: ran `python -m pytest tests/unit/serving -q`,
  `python -m pytest tests/unit/cli/test_cli_main.py tests/unit/cli/test_cli_main_extra.py tests/unit/cli/test_cli_cfg_parallel.py tests/unit/cli/test_cli_cp_mode.py tests/unit/cli/test_cli_sp.py tests/unit/serving -q`,
  `python -m pytest tests/unit/serving tests/unit/cli/test_cli_main.py tests/unit/cli/test_cli_main_extra.py -q`,
  `python -m pytest tests/unit/cli/test_cli_main.py tests/unit/cli/test_cli_main_extra.py tests/unit/cli/test_cli_cfg_parallel.py tests/unit/cli/test_cli_cp_mode.py tests/unit/cli/test_cli_sp.py tests/unit/serving -q`,
  `python -m pytest tests/unit/serving tests/unit/cli/test_cli_main.py tests/unit/cli/test_cli_main_extra.py tests/unit/cli/test_cli_cfg_parallel.py tests/unit/cli/test_cli_cp_mode.py tests/unit/cli/test_cli_sp.py -q`,
  `python -m pytest tests/unit/serving tests/unit/cli/test_orchestrator_qwen.py::test_stage_compiled_dir_names tests/unit/cli/test_orchestrator_qwen.py::test_stage_compiled_dir_unknown_raises tests/unit/cli/test_cli_main.py tests/unit/cli/test_cli_main_extra.py tests/unit/cli/test_cli_cfg_parallel.py tests/unit/cli/test_cli_cp_mode.py tests/unit/cli/test_cli_sp.py -q`,
  `python -m pytest tests/unit/serving -q`,
  `python -m pytest tests/unit/serving tests/unit/cli/test_orchestrator_qwen.py::test_download_resolves_remote tests/unit/cli/test_orchestrator_qwen.py::test_shared_cli_args_optionals tests/unit/cli/test_orchestrator_qwen.py::test_shared_cli_args_forwards_revision tests/unit/cli/test_orchestrator_qwen.py::test_stage_compiled_dir_names tests/unit/cli/test_orchestrator_qwen.py::test_stage_compiled_dir_unknown_raises tests/unit/cli/test_cli_main.py tests/unit/cli/test_cli_main_extra.py tests/unit/cli/test_cli_cfg_parallel.py tests/unit/cli/test_cli_cp_mode.py tests/unit/cli/test_cli_sp.py -q`,
  `python -m pytest tests/unit/serving tests/unit/cli/test_orchestrator_qwen.py::test_download_resolves_remote tests/unit/cli/test_orchestrator_qwen.py::test_shared_cli_args_optionals tests/unit/cli/test_orchestrator_qwen.py::test_shared_cli_args_forwards_revision tests/unit/cli/test_orchestrator_qwen.py::test_stage_compiled_dir_names tests/unit/cli/test_orchestrator_qwen.py::test_stage_compiled_dir_unknown_raises tests/unit/cli/test_cli_main.py tests/unit/cli/test_cli_main_extra.py tests/unit/cli/test_cli_cfg_parallel.py tests/unit/cli/test_cli_cp_mode.py tests/unit/cli/test_cli_sp.py -q` after the follow-up review fixes,
  `python -m compileall -q difflet/serving difflet/common difflet/cli/main.py difflet/cli/orchestrators/qwen_image.py difflet/cli/stage.py`, and
  `git diff --check`. Full `tests/unit/cli` and full Qwen orchestrator tests still require
  `torch`/`numpy` and a writable HuggingFace/Difflet cache in this environment.

Deployment review (2026-07-10):
- Done: deployed commit `94e6c42` to EC2 `16.51.176.130` in an AWS Neuron SDK 2.30 Ubuntu 24.04 container.
- Done: downloaded Qwen-Image, compiled prompt encoder/denoiser/VAE artifacts, and generated a verified 1024x1024 RGB PNG with exit code 0.
- Done: fixed Qwen serving preflight to accept NxD `model.pt` plus `neuron_config.json` artifacts; `tests/unit/serving` passes 52 tests.
- Blocker: resident-worker co-load reaches decoder weight initialization, then segfaults in `libtorchneuron.so`; health/readiness never opens. Offline staged generation remains functional.
- Logs: `/mnt/difflet-data/logs/qwen-{compile,generate,serve-startup,serve-kernel-crash}.log` on the EC2 instance.

Qwen Trn2 mixed-TP topology investigation (2026-07-10):
- [x] Record the successful staged compile/generate path and failed resident co-load attempts.
- [x] Verify that prompt encoder and denoiser were compiled for TP=4 and VAE decoder for TP=1.
- [x] Test TP=4 and TP=1 in separate concurrent processes on the four-core Trn2 device.
- [x] Compare the cited Inf1 NCG guidance with current Trn2/torch-neuronx placement guidance.
- [x] Test same-process TP=4 -> TP=1 load with explicit placement/rank controls.
- [x] Test alternative load ordering if the placement API applies to NxD artifacts.
- [x] Implement the smallest viable serving topology after the runtime experiment.
- [x] Run unit tests plus remote startup and generation verification.
- [x] Complete `docs/design/qwen_trn2_topology/` with the final decision and measured timings.

Current findings:
- TP=4 encoder + denoiser can co-reside in one process.
- A concurrent TP=1 process is rejected because the TP=4 process owns cores 0-3.
- TP=1 decoder loads alone in 5.91s plus 0.27s warmup.
- TP=4 -> TP=1 in the current shared process reaches decoder weight initialization and segfaults in `libtorchneuron.so`.

Final decision and verification:
- Done: rejected mixed TP after implicit, explicit-placement, and reversed-order experiments.
- Done: compiled a serving-only TP=4 VAE and kept the staged CLI VAE at TP=1.
- Done: all three TP=4 stages co-load in one resident worker; four-step smoke produces a valid 1024x1024 RGB PNG.
- Done: remote target tests pass (`87 passed`), `/health` and `/ready` return 200, and a real API request completes successfully.
- Done: resident device memory is 73,401,942,984 bytes (about 68.36 GiB).
- Done: independent agent audited Wan, HunyuanVideo, Flux, LTX-2, and HunyuanVideo 1.5 topology; findings are recorded in `docs/design/qwen_trn2_topology/04_other_models_topology_audit.md`.

Flux Trn2 runtime benchmark (2026-07-10):
- [x] Confirm the existing Flux weights, compile cache, and TP/CP/world-size profile on the target instance.
- [x] Run a real Flux inference before accepting serving readiness.
- [x] Measure three independent CLI generations, including per-process model load.
- [x] Measure resident serving startup/warmup separately, then three API generations.
- [x] Record each duration, aggregate duration, output validation, and remote log paths.

Local R2 environment setup (2026-07-10):
- [x] Document how Cloudflare R2 credentials map to Difflet's `DIFFLET_R2_*` variables.
- [x] Organize `.env` without committing secrets and verify its shell syntax/config shape.
- Rationale: keep only the S3-compatible Bucket, account endpoint, Access Key ID, and Secret Access Key required by Difflet; discard the unused API token/dashboard paste and leave the optional public URL empty for presigned responses.
- Verification: `zsh -n .env`, required-key shape check, `R2ArtifactStore.from_env()` with the saved values, local boto3 presigned-URL generation, and `git diff --check` all pass without exposing credentials.

Flux benchmark status:
- Initial download returned `401 GatedRepoError`; the user-provided token was mounted temporarily, the gated weights downloaded successfully, and the remote token copy was deleted immediately afterward.
- Done: Flux compiled at TP=4/CP=1/world=4. Saved component configs confirm CLIP TP=1/W=4, T5 TP=4/W=4, transformer TP=4/W=4, and VAE TP=1/W=4.
- Done: three independent CLI runs took 48.81s, 44.66s, and 45.67s (139.14s total); all outputs are valid 1024x1024 RGB PNGs.
- Done: resident startup took 40.14s; three real serving requests took 10.87s, 10.39s, and 11.60s (32.86s total), including PNG upload and presign.
- Done: an R2 presigned URL was downloaded and validated as a 1024x1024 RGB PNG.
- Current state: `difflet-flux-service` is healthy on port 8092. Qwen is stopped because both services require cores 0-3. No EC2 stop/terminate or instance-store deletion occurred.
- Detailed record: `docs/design/qwen_trn2_topology/05_flux_runtime_validation.md`.

Serve CLI startup parity:
- [x] Document the boundary between CLI-compatible startup profile options and per-request generation/file-output fields.
- [x] Document model-specific handling for CFG, SP, host VAE, frame shape, and TeaCache options, including adapter-owned optional-calibration fallback.
- [ ] Expose all compile/load/runtime CLI options on `difflet serve`.
- [ ] Thread supported options through `ServeOptions`, `ServingProfile`, artifact identity, and resident model construction.
- [ ] Add parser, profile, model capability, and orchestrator propagation tests.
- [ ] Run focused serving/CLI tests and `git diff --check`.

Stage runtime refactor design (2026-07-10):
- [x] Audit current Qwen CLI, common preflight, serving orchestrator, and resident-worker stage ownership.
- [x] Compare Difflet's lifecycle with vLLM-Omni stage config/runtime/pool/runner boundaries.
- [x] Define the proposed `PipelineDefinition`, `StageDefinition`, `RuntimePlan`, `StageInvocation`, and process-local `StageRunner` contracts.
- [x] Document Qwen three-stage and Flux single-stage mappings, including CLI/resident topology differences.

Single-process stage engine design (2026-07-13):
- [x] Move the authoritative ordered stage traversal from the model orchestrator boundary into the generic engine design.
- [x] Define transport-neutral stage invocation, payload, result, context, runner, and executor contracts.
- [x] Keep P0 execution process-local and sequential while preserving an executor/transport seam for future stage scheduling.
- [x] Review the updated serving design for cross-document consistency and run Markdown/diff verification.

Nominal stage payload typing revision (2026-07-13):
- [x] Replace the `StagePayload = Any` logical contract with a nominal marker base.
- [x] Make invocation/result/runner contracts generic and require Qwen/Flux payload inheritance.
- [x] Run iterative same-reviewer convergence plus a fresh independent review.
- [x] Verify cross-document terminology and `git diff --check`.

Single-process stage engine implementation (2026-07-13):
- [x] Add nominal/generic stage payload and execution contracts.
- [x] Implement validated erased runners, in-process executor, and generic ordered engine.
- [x] Migrate Qwen and Flux serving implementations to stage adapters.
- [x] Route resident worker load/smoke/generate/shutdown through the stage engine.
- [x] Add negative payload, traversal, shutdown, and worker regression tests.
- [x] Run focused serving tests, compile/type checks, and `git diff --check`.
- [x] Document phased migration, failure behavior, verification matrix, estimates, non-goals, and pending decisions.
- [x] Consolidate and review the stage/runtime contracts in
  `docs/design/difflet_serving/architecture.md`.
- [x] Implement the contract/cleanup phase in
  `docs/design/difflet_serving/implementation_changes.md` only after final review.
- [ ] Implement later phases only after the previous phase passes local and Trainium validation.
- Rationale: the MVP remains single-process and sequential, while its stable stage
  contracts and executor boundary can support a future scheduler/IPC/shared-memory
  implementation without changing model payload definitions or runner APIs.
- Verification: 142 serving tests and 189 combined serving/relevant CLI tests pass;
  targeted mypy reports no issues for the generic engine contracts; Black,
  `compileall`, and `git diff --check` pass. Full `tests/unit` collection remains
  unavailable because this local environment lacks `torch`, `numpy`, and `diffusers`;
  Ruff is also not installed.

Heartbeat interval validation (2026-07-13):
- [x] Require a finite worker heartbeat interval between 5 and 120 seconds,
  inclusive, at CLI, options, resident config, and worker-process boundaries.
- [x] Cover lower/upper boundaries plus out-of-range, NaN, and infinity inputs.
- Rationale: invalid waits can either spin the heartbeat thread or terminate it;
  validating before stack construction also avoids expensive model preparation.
- Verification: 157 serving tests and 204 combined serving/relevant CLI tests pass;
  targeted mypy, Black, `compileall`, and `git diff --check` pass.

OpenAI request ID contract cleanup (2026-07-13):
- [x] Remove the unreachable client request-body `id` handling from normalization
  and HTTP logging.
- [x] Generate one server-owned request ID at the API boundary and propagate it
  through normalization, engine logging, and the chat completion response.
- [x] Document that request-body `id` is rejected while response `id` remains.
- Verification: 43 focused chat/API tests and 160 serving tests pass; targeted
  mypy, Black, `compileall`, and `git diff --check` pass.

Serve operational CLI controls (2026-07-13):
- [x] Expose queue, request, artifact-operation, cancellation, and restart controls with existing defaults.
- [x] Keep artifact-store selection and artifact URL TTL internal to server configuration.
- [x] Verify CLI help, argument propagation, validation, focused tests, and `git diff --check`.
- Rationale: operators can tune production admission and recovery behavior without
  changing the existing defaults or exposing deployment-owned artifact policy.
- Verification: 30 focused serve CLI tests and 219 combined serving/relevant CLI
  tests pass; Black, `compileall`, and `git diff --check` pass.

Serving request bounds (2026-07-13):
- [x] Reject oversized raw prompts before synchronous provider tokenization.
- [x] Bound guidance to the normal Flux/Qwen serving range before worker admission.
- [x] Update the request contract and run focused/full serving verification.
- Rationale: a cheap generic request bound protects the event loop, while the
  provider validators own guidance policy and exact model-token bucket enforcement.
- Verification: 74 focused chat/Flux/Qwen validator tests and 235 combined
  serving/relevant CLI tests pass; Black, `compileall`, and `git diff --check` pass.

Neuron core visibility and TeaCache typing (2026-07-13):
- [x] Preserve inherited Neuron core visibility for Flux and Qwen resident plans.
- [x] Default an unpartitioned current host to Neuron cores 0-3.
- [x] Restore static torch annotation visibility without a runtime import.
- [x] Run focused/full serving tests, mypy, compile, formatting, and diff checks.
- Rationale: resident workers must respect host partitioning while retaining the
  current four-core default, and type-only annotations must not force a runtime
  torch import.
- Verification: 67 focused runtime/Flux/Qwen/worker tests and 248 combined
  serving/relevant CLI tests pass; targeted mypy, Black, `compileall`, and
  `git diff --check` pass.

Serving log timestamps (2026-07-13):
- [x] Add millisecond local timestamps to console, access, and file logs.
- [x] Verify logging configuration and serving/CLI regressions.
- Verification: 31 focused serve CLI tests and 249 combined serving/relevant
  CLI tests pass; Black, `compileall`, and `git diff --check` pass.

Stage refactor design review:
- Done: stage metadata is currently descriptive; execution order, placement, artifact resolution, and dispatch are still duplicated.
- Done: selected a conservative extraction rather than a generic DAG, StagePool, replica, or cross-host control plane.
- Done: kept compile adapters and loaded runtime adapters process-local, while definitions remain immutable and reusable.
- Done: kept CLI file transport and resident in-memory transport as separate adapters over the same pipeline definition.
- Done: resolved naming, JSON invocation serialization, compatibility display
  aliases, Qwen-first migration scope, and deferred multi-process resident Qwen as
  an optional later phase.

Stage refactor design revision (2026-07-10):
- [x] Separate worker allocation from homogeneous stage topology and artifact binding.
- [x] Restrict the first StageRunner migration to Qwen and keep Flux as an opaque serving pipeline.
- [x] Add Flux serving-only component topology/artifact/load-order admission.
- [x] Bind the documented serve startup-option parity contract to resolved stage/profile inputs.
- [x] Specify structured startup, stage lifecycle, progress, and worker heartbeat logging.
- [x] Complete same-reviewer convergence and run a fresh independent review.
- [x] Revise Qwen and Flux topology examples to be profile-relative; retain TP4/W4
  only as the measured Trn2 regression profile.
- [x] Define parent-synthetic terminal log closure and deduplication for worker
  process death or forced termination.
- [x] Obtain the fresh reviewer's follow-up confirmation with no blocking/material
  findings.
- [x] Resolve GPT-5.6 Round 12 invocation, artifact identity, resource admission,
  and shutdown fencing findings; obtain same-reviewer convergence.
- [x] Resolve Round 13 pinned runtime-source and distributed/logical-NC environment
  findings; obtain same-reviewer convergence.
- [x] Resolve Round 14 duplicate artifact authority and migration phase-boundary
  findings; obtain same-reviewer convergence.
- [x] Resolve Round 15 pinned serving compile-child transport finding; obtain
  same-reviewer convergence.
- [x] Resolve Round 17 unsafe-error admission, immutable artifact publication,
  compile schema, and provisional-startup cleanup findings; obtain same-reviewer
  convergence and one fresh independent pass.
- [x] Resolve Round 18 cancellation-safe ticket ownership and crash-durable content
  validation findings; obtain same-reviewer convergence.
- [x] Resolve Round 19 shutdown ownership takeover and finalized-generation
  quarantine findings; obtain same-reviewer convergence.
- [x] Resolve Round 20 late-caller completed-cleanup observability finding; obtain
  same-reviewer convergence.
- [x] Resolve Round 21 per-ticket shutdown cleanup proof finding; obtain
  same-reviewer convergence.
- [x] Validate the user-approved engine-owned request/abort simplification against
  the prior Round 18-21 ownership findings and vLLM-Omni behavior.
- [x] Resolve Round 22 resident request-state reset, shutdown delivery arbitration,
  and thread-safe abort signal findings; obtain same-reviewer convergence.
- [x] Resolve Round 23 two-phase terminal selection/reset race; obtain same-reviewer
  convergence, then run one fresh independent review.
- [x] Resolve Round 25 Flux artifact-path injection, StageExecutionContext,
  exhaustive terminal classification, and immutable local-source admission; obtain
  fresh-reviewer convergence.
- [x] Compress stage/serving design docs into canonical ownership, state-machine,
  migration, API, and review-ledger documents; remove duplicate acceptance text,
  tutorials, future examples, and historical review transcript without dropping
  implementation contracts.
- [x] Resolve Round 26 pinned Flux source injection and FIFO pre-execution abort
  retention findings.
- [x] Obtain fresh-reviewer convergence for the applied Round 28 pipeline
  override-mode validation, per-worker control-lease fencing, and ordered shutdown.
- [x] Obtain fresh-reviewer convergence for the simplified Round 29 direct bounded
  queue and shutdown/recovery ownership revision.
- [x] Resolve Round 32 HF-snapshot admission, shutdown send-failure cleanup, and
  immutable canonical cache inputs; supersede generic runtime-file binding with the
  user-selected adapter-owned config.
- [x] Resolve Round 33 serving-only runner scope and cancellation API cleanup;
  supersede generic runtime-file publication with adapter-owned parsing/fallback.
- [x] Resolve Round 34 adapter-config-before-identity order and clean the review
  ledger; reviewer confirmed both in Round 35.
- [x] Resolve Round 35 raw adapter-option ownership and concrete Qwen/Flux config
  contracts; reviewer confirmed both in Round 36.
- [x] Resolve Round 36 adapter-owned compile hooks, probe-only compile identity,
  direct artifact binding, and calibration validity; Round 37 confirmed calibration
  rules and identified the final stale-interface/handoff gaps.
- [x] Resolve deletion of branch-local
  `ensure_artifacts()`/`compile_plan(profile)`, in-place `DiffletCompileSpec`
  extension, and prevalidated serving calibration input; Round 38 confirmed them.
- [x] Resolve Round 38 calibration step-count behavior with request-level baseline
  fallback and calibrated smoke; remove the unnecessary string validator registry.
- [x] Consolidate serving/stage/runtime/serve-option contracts, add the explicit
  implementation-change/delete list, remove standalone runtime documents, and
  obtain Round 39 confirmation that no prior contract was lost.
- [x] Obtain Round 40 same-reviewer convergence for request-local TeaCache
  step-mismatch fallback, local controller/probe references, and reset leakage tests.
- [x] Run fresh Round 41 independent review of the consolidated serving design.
- [x] Apply fresh-review manifest namespace, queue record/deadline, pinned
  validator, concrete stage-boundary, and branch-delta revisions; Round 42 narrowed
  the remaining work to two explicit gaps.
- [x] Obtain fresh Round 43 confirmation for removal of top-level shared validation
  from `serve` and exact reuse/cache-miss/load artifact-validation ordering.
- [x] Obtain fresh-review convergence for retaining exact compile specs in the
  runtime bundle and using them during initial/replacement adapter validation;
  fresh follow-up reported no blocking/material findings.
- [x] Add the explicit Qwen orchestrator loop and engine/orchestrator stage ownership
  boundary, including before/after-stage abort checks.
- [x] Keep HF freshness and revision movement in download/preflight; serving pins the
  resolved commit and does not add source-byte inventory or latest-version checks.
- [x] Specify deterministic immutable-generation locking, allocation, reuse, crash,
  and publication behavior.
- [x] Restore the full `common`/CLI/pipeline/models/serving architecture hierarchy.
- [x] Add an exact file/symbol migration inventory.
- [x] Resolve Round 45 follow-up bundle/final-output, opaque Flux stage, exact
  migration inventory, shutdown error, and Flux post-abort findings after user
  confirmation.
- [x] Obtain Round 45/46 same-reviewer convergence after those revisions.
- [x] Resolve Round 46 compile-invocation field naming and remove undefined P0 stage
  resolver factories after user confirmation.

Serving/runtime implementation (2026-07-11):
- [x] Add validated `PipelineDefinition`, extracted/opaque `StageDefinition`,
  `RuntimePlan`, allocation/topology, artifact binding, and pinned-source contracts.
- [x] Replace registry `ServingStageMetadata` with one pipeline definition and rename
  metadata loading from preflight to artifact preparer without an alias.
- [x] Preserve serve CFG/SP omission as `None`, add explicit disable flags, isolate
  staged CLI validators, and propagate the heartbeat interval into worker config and
  process arguments.
- [x] Implement immutable generation publication with bounded file lock, managed
  staging, deterministic newest-valid reuse, AUTO/NEVER/FORCE policy, recursive
  payload hashing, fsync + atomic rename, corruption fallback, and symlink rejection.
- [x] Add serving-only HF source pinning from `snapshots/<commit>` without online
  freshness checks or staged CLI behavior changes.
- [x] Verify the current local slice with 138 focused serving/CLI tests,
  `compileall`, Black, and `git diff --check`; Ruff is not installed locally.
- [x] Replace legacy path-based `DiffletCompileSpec` and `ensure_artifacts()` callers
  with adapter compile specs, `ResolvedRuntimeBundle`, and direct generation binding.
- [x] Add Qwen typed serving runners and frozen pipeline traversal; bind Qwen and
  Flux compile/load to the pinned source and exact immutable generation. Common
  Qwen application-builder extraction remains a later cleanup within the runner
  implementation.
- [ ] Replace the legacy worker cancellation/reply engine with the reviewed request
  record, status queue, heartbeat thread, stage logging, and terminal arbitration.
- [ ] Run Qwen and Flux Trainium compile/load/smoke before advancing each runtime
  migration phase.
- Blocked verification: `ubuntu@16.51.182.190` timed out during SSH banner exchange
  on 2026-07-11, so the new generation layout has not yet been exercised on Neuron.
- [x] Address serving implementation review: include `artifact_manager.py` in the
  patch, derive rectangular Qwen latent grids from the frozen profile, enforce
  request steps `1..50`, prevent closed/draining recovery restarts, and implement a
  dedicated non-blocking heartbeat pipe/thread with busy request/stage status and
  parent-side stale-heartbeat logging.
- Verification: 155 focused serving/CLI tests pass; forced worker restart no longer
  emits multiprocessing semaphore/resource-tracker warnings; `compileall` and
  `git diff --check` pass.
- [x] Fix only Qwen serving rectangular latent unpacking from the frozen profile;
  keep the staged CLI decode path unchanged.
- Rationale: packed sequence length alone does not preserve rectangular grid shape;
  the resident adapter already owns the immutable height/width profile.
- Verification: `python -m pytest tests/unit/serving -q` passes 98 tests;
  focused Qwen/profile tests pass 24 tests; `compileall` and `git diff --check` pass.

Serving review fixes (2026-07-12):
- [x] Terminalize admitted callers as `engine_draining` before worker shutdown.
- [x] Allowlist public worker errors and sanitize unknown worker/engine exceptions.
- [x] Log R2 backend failures privately and return a fixed `internal_error`.
- [x] Resolve and validate explicitly enabled TeaCache calibration once at startup;
  pass frozen calibration data to compile/load/replacement workers and leave CLI
  pathname behavior unchanged.
- [x] Run focused and full serving tests, compileall, and `git diff --check`.
- Rationale: shutdown must resolve every admitted caller promptly; backend exception
  text is private operational data; replacement workers must not reopen mutable
  user calibration files.
- Verification: 165 focused serving/CLI tests pass; Black, `compileall`, and
  `git diff --check` pass. Trainium compile/load smoke remains remote-only.

Serving allocation/compatibility review (2026-07-12):
- [x] Apply and validate the single resident worker allocation in the child before
  any model or Neuron import.
- [x] Restore backward-compatible defaults for `_merge_teacache_kwargs`.
- [x] Add child-environment and helper-signature regression tests, then rerun the
  serving/CLI/pipeline checks and static verification.
- Rationale: runtime allocation metadata must become the child process's effective
  Neuron/distributed environment, and new optional helper inputs must remain
  optional for existing callers.
- Verification: 169 focused serving/CLI tests pass; the allocation suite passes 16
  tests including a real spawned child; direct legacy helper calls pass; Black,
  `compileall`, and `git diff --check` pass. The torch-dependent pipeline test file
  cannot collect in this local environment because `torch` is not installed.

HTTP request-shape review (2026-07-12):
- [x] Route arbitrary parsed JSON bodies through Difflet normalization instead of
  FastAPI's default 422 validation.
- [x] Add HTTP-level coverage for array, null, string, and numeric request bodies.
- [x] Run serving tests and static verification.
- Rationale: framework-level body typing must not bypass the public Difflet error
  envelope for syntactically valid JSON values.
- Verification: 173 focused serving/CLI tests pass; the HTTP/API slice passes 39
  tests; Black, `compileall`, and `git diff --check` pass.

PR review and remediation (2026-07-12):
- [x] Review the complete HEAD-to-worktree diff for runtime, artifact, API, CLI,
  TeaCache, and compatibility defects.
- [x] Close and join per-spawn multiprocessing queues on startup failure, shutdown,
  termination, and replacement.
- [x] Reproduce and fix all confirmed independent-review findings.
- [x] Run the broadest locally available tests and complete a final PR re-review.
- Rationale: compile identities and serving compile environments must describe the
  exact executable artifact; adaptive TeaCache state is request-local; readiness
  requires real inference; public errors and process resources must remain bounded.
- Verification: 132 full serving tests and 186 combined serving/relevant CLI tests
  pass. All 43 changed Python files pass Black; `compileall` and
  `git diff --check` pass. Added review coverage for Qwen/Flux compile identity,
  strict Neuron environments, TeaCache calibration/request isolation, Flux real
  smoke, shutdown terminalization, queue cleanup, error sanitization, and HTTP
  non-object JSON bodies. Torch-dependent model tests and Trainium compile/load
  smoke remain unavailable in this local environment.

New Trainium serving validation (2026-07-12):
- [x] Inspect `16.26.177.239`, its Neuron SDK environment, disk, model cache, and
  artifact-store configuration.
- [x] Rsync the current local worktree without Git metadata, caches, generated
  artifacts, or local secret files.
- [x] Install the requested runtime dependencies and editable local source in the
  existing Neuron PyTorch 2.9 virtual environment.
- [x] Start Qwen serving and verify health/readiness plus one real generation.
- [x] Start Flux serving and verify health/readiness plus one real generation.
- [x] Record commands, startup/request results, logs, and any remaining defects.
- Rationale: validate the exact serving-owned download -> immutable compile ->
  resident load -> real smoke -> HTTP generation lifecycle on a clean Trn2 host;
  no standalone `difflet download`, `compile`, or `generate` command was used.
- Verification: current worktree synced to `/home/ubuntu/Difflet` on a four-core
  `trn2.3xlarge`. Qwen `serve` downloaded 29 files (~54 GiB), compiled TP4 text,
  denoiser, and serving-only TP4 VAE artifacts, co-loaded all stages, passed its
  four-step smoke, returned health/ready 200, and generated a validated 1024x1024
  RGB PNG in 9.07s. Flux `serve` downloaded 25 files, compiled CLIP/T5/transformer/
  decoder, passed the new real four-step smoke, returned health/ready 200, and
  generated a validated 1024x1024 RGB PNG in 3.86s. Logs are under
  `/home/ubuntu/Difflet/logs/{qwen,flux}-serve.log`; Flux remains healthy on
  `127.0.0.1:8092`, while Qwen was shut down normally to release all four cores.
  Follow-up Qwen and Flux warm restarts emitted zero compile lines, reused the
  immutable generations, and passed startup smoke; their logs are
  `logs/{qwen,flux}-restart.log`.
- Findings: the README's `--no-deps` install leaves `uvicorn` and `accelerate`
  absent in this clean venv, so both were installed explicitly. The instance's
  newer Neuron SDK patch versions differ from the exact `pyproject.toml` pins,
  but both models passed real hardware validation. A duplicate manually started
  Flux process briefly contended on the same download/log; it was removed while
  preserving the original process and cache. Both original services later
  received clean SIGTERM shutdowns. No serving error, traceback, segfault, OOM,
  or multiprocessing resource leak remained in the final or restart logs.

CLI versus serving benchmark (2026-07-12):
- [x] Download all remote serving logs to a local untracked artifact directory.
- [x] Freeze equivalent Qwen and Flux benchmark parameters and inventory CLI/
  serving artifact caches.
- [x] Measure and validate Qwen CLI generation versus resident HTTP generation.
- [x] Measure and validate Flux CLI generation versus resident HTTP generation.
- [x] Report one-time preparation separately from repeated generation latency and
  record output dimensions, process/load behavior, and remaining caveats.
- Rationale: compare repeated user-visible latency with identical public model,
  shape, step, guidance, seed, and prompt inputs while keeping one-time artifact
  preparation outside request latency.
- Verification: Qwen serving averaged 4.691s versus 77.455s for warm independent
  CLI processes (16.5x); Flux serving averaged 3.824s versus 48.770s (12.8x).
  All outputs are valid 1024x1024 RGB PNGs. Raw evidence and the final report are
  under `artifacts/remote-logs/16.26.177.239/` and
  `artifacts/cli-vs-serve-16.26.177.239.md`.

Serving dotenv loading (2026-07-12):
- [x] Load current-directory `.env` automatically for `difflet serve` without
  overriding exported process environment variables.
- [x] Add a sanitized `.env.example` and document the serving dependency/install
  behavior.
- [x] Add focused tests and validate a real Flux startup without `source .env`.
- Verification: 54 focused serving tests pass locally. Remote Flux PID 22277 was
  the only serving parent, loaded R2 configuration from `.env`, passed real smoke,
  and returned health/ready 200 without shell-sourcing the file.

R2 artifact latency (2026-07-12):
- [x] Reuse one lazily constructed boto3 client per R2 artifact store.
- [x] Split artifact upload and URL-generation latency logs.
- [x] Add focused regression coverage and run serving checks.
- Rationale: reuse the boto3 connection pool across requests and distinguish R2
  upload latency from local URL generation before making further optimizations.
- Verification: the focused artifact/chat suite passes 35 tests and the full
  serving suite passes 136 tests; Black and `git diff --check` pass. Mypy reaches
  only pre-existing pipeline/TeaCache errors, and Ruff is not installed locally.

HTTP disconnect, error contract, and artifact TTL review (2026-07-13):
- [x] Race each chat-completions operation against client disconnect detection,
  cancel and await queued/running generation, and prove task cleanup in tests.
- [x] Convert unexpected route failures to the fixed OpenAI-style
  `500 internal_error` payload without exposing backend exception text.
- [x] Define and test the R2 TTL contract for private presigned URLs versus public
  custom-domain URLs backed by bucket lifecycle deletion.
- [x] Run focused/full serving tests, type/format/compile/diff checks.
- [x] Inventory `16.26.110.85`, rsync the safe worktree subset, and validate real
  Flux/Qwen CLI plus serving health/readiness/HTTP generations sequentially.
- Rationale: cancellation owns both race tasks and awaits every cancelled task;
  a queued cancellation releases admission/lock while an in-flight cancellation
  enters resident-worker cancel/recovery. Private R2 mode provides exact
  presigned-URL access expiry; public custom-domain mode deliberately delegates
  object expiry to the bucket lifecycle and documents possible lifecycle/CDN lag.
- Local verification: all 188 serving tests and all 235 combined serving/relevant
  CLI tests pass. Targeted mypy, Black, `compileall`, and `git diff --check` pass.
  Regression tests cover task cleanup on disconnect, sanitized unexpected 500s,
  private `ExpiresIn`, and public custom-domain URL behavior.
- Trainium verification: current worktree was synced to `/home/ubuntu/Difflet` on
  four-core `trn2.3xlarge` host `16.26.110.85`. Qwen and Flux compile plus CLI
  generation both exited 0 and produced validated 1024x1024 RGB PNGs at
  `outputs/{qwen,flux}-cli.png`. Serving ran strictly sequentially at TP4/CP1 and
  1024x1024 on `127.0.0.1:8092`; both passed real four-step startup smoke and
  returned health/ready 200. Qwen HTTP generation returned 200 in 9.721s and Flux
  in 5.066s; both OpenAI envelopes were validated, their unlogged private R2
  presigned URLs were fetched successfully, and the downloaded outputs at
  `outputs/{qwen,flux}-serving.png` are 1024x1024 RGB PNGs. Each server handled
  SIGTERM normally; afterward there were zero child processes, zero port 8092
  listeners, and zero Neuron runtime processes. Detailed evidence is under
  `/home/ubuntu/Difflet/logs/{qwen,flux}-{cli,serve}*.log`.

Heartbeat admission visibility (2026-07-13):
- [x] Enrich parent-written worker heartbeat records with running, queued,
  pending, and total request-capacity admission counts.
- [x] Include the counts in both human-readable heartbeat messages and the
  structured `worker_heartbeat` logging payload.
- [x] Run focused and full serving verification before syncing to Trainium.
- Verification: 26 focused resident-worker tests and all 202 serving tests pass;
  targeted Mypy, Black, `compileall`, and `git diff --check` pass.

Trainium Qwen/Flux CLI and Serving revalidation (2026-07-13):
- [x] Run Qwen and Flux CLI generation sequentially at TP4/CP1, 1024x1024,
  four steps, and validate both output PNGs.
- [x] Run Qwen and Flux Serving sequentially, verify health/readiness, submit a
  real four-step HTTP generation, fetch each public R2 URL, and validate PNGs.
- Results: Qwen CLI completed in 76.58s and Qwen HTTP in 8.25s; Flux CLI
  completed in 52.54s and Flux HTTP in 2.78s. All four outputs are 1024x1024
  RGB PNGs. Flux Serving remains healthy and ready on port 8092.

Malformed JSON error contract (2026-07-13):
- [x] Convert FastAPI request-body JSON decoding failures into the stable
  OpenAI-style `400 invalid_request` payload without exposing FastAPI `detail`.
- [x] Add an HTTP-level regression test containing a raw newline inside a JSON
  string and document the public error matrix entry.
- [x] Run focused and full serving verification.
- Verification: all 9 API-server tests and all 203 serving tests pass;
  targeted Mypy, Black, `compileall`, and `git diff --check` pass.
- Trainium verification: synced to `16.26.110.85`, restarted Flux Serving, and
  confirmed malformed JSON returns the fixed OpenAI-style `400 invalid_request`
  payload while a valid four-step request returns 200 and a downloadable
  1024x1024 RGB PNG. Health and readiness remain 200.

HTTP disconnect injection regression (2026-07-13):
- [x] Require FastAPI to inject the route `Request` instead of accepting an
  optional default that silently disables disconnect handling.
- [x] Observe the underlying ASGI `http.disconnect` message without hanging
  Starlette's synchronous test client.
- [x] Verify a disconnected blocking request cancels and awaits generation with
  no leaked HTTP race tasks, then rerun serving verification.
- Rationale: HTTP cancellation is the trigger for the existing resident-engine
  cancellation/recovery path; model execution still stops only at stage-safe
  cancellation checkpoints.
- Verification: all 10 API-server tests and all 204 serving tests pass. Black,
  isolated Mypy, `compileall`, and `git diff --check` pass. Repository-wide
  import-following Mypy still reports four pre-existing errors in
  `parallel_config.py` and `difflet_pipeline.py`.
