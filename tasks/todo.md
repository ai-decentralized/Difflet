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
