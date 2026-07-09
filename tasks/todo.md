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
