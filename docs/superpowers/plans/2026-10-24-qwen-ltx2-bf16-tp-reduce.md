# bf16 tensor-parallel all-reduce by default (all models except MiniMax-H3) — Implementation & Measurement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Device phases (Tasks 9–12) also follow the `difflet-device-verify` skill (supervised runs, checkpoint after every verified step, one-bug-one-commit).

**Goal:** Make every Trainium tensor-parallel all-reduce bf16 by default in every model except MiniMax-H3 — one commit per model — keep fp32 selectable, then measure Qwen-Image and LTX-2 fp32 vs bf16 at TP4 / LNC=2 with the Wan2.1 report's metric set plus PSNR/SSIM against a CPU fp32 reference, and regression-check the other changed models.

**Architecture:** NxD's `RowParallelLinear` defaults to `reduce_dtype=torch.float32`. Flux's DiT (an NxDI fork) and Wan's DiT override it to bf16; every other construction inherits fp32. A shared helper parses a new `--tp-reduce-dtype {bf16,fp32}` flag (default `bf16`) and produces compile-cache fields; each model commit threads the dtype into its own `RowParallelLinear` sites and keys its artifacts. Cache keys are additive: a component's key gains a field only when the requested dtype differs from what that component used before this change, so every existing artifact stays valid in its legacy configuration.

**Tech Stack:** Python, PyTorch, neuronx-distributed (NxD) via `requirements-neuron.lock`, diffusers 0.38 (CPU fp32 reference), neuron-profile / neuron-explorer, scikit-image (SSIM), pytest.

**Spec:** user requests in this conversation — (1) "get qwen and ltx-2 to bf16, run them in tp4 configuration with LNC=2, collect the metrics that the report collected, and also collect PSNR/SSIM compared with CPU runs"; (2) "make the rest with bf16 by default; each model one commit for bf16 enablement, except minimax". Report referred to: claude.ai artifact "Wan2.1 Trainium Backends" (measured 2026-10-11); its raw records are not on this host, so its metric definitions are restated below.

## Global Constraints

- MiniMax-H3 is untouched: its two `RowParallelLinear` sites (`difflet/backends/trainium/minimax_h3/transformer.py:138, :156`) keep the fp32 default, and `--tp-reduce-dtype` given with a MiniMax model fails fast.
- Flag: `--tp-reduce-dtype` accepts exactly `bf16` and `fp32`; default `bf16`. Applies to every all-reduce of every TP-sharded component of the model (DiT and text encoders).
- Cache-key field: `"tp_reduce_dtype": "bfloat16" | "float32"`, emitted only when the requested dtype ≠ the component's legacy dtype (legacy table below). A component that mixed dtypes before (Flux, one artifact for DiT + encoders) always carries the field.
- Out of scope: `difflet/backends/trainium/core/modules/attention/gqa.py:1008` (no model uses `NeuronAttentionBase`); manual reduces that already run in the activation dtype (HunyuanVideo single-stream `:743`); Megatron-SP reduce-scatters; LTX-2's q/k RMS-norm statistic all-reduce (`ltx_2/transformer.py:156-157`, deliberately fp32: a `[B,S,1]` sum of squares).
- Hardware: trn2.3xlarge, `logical-neuroncore-config: 2` (4 logical NeuronCores), verified with `neuron-ls`. Device runs: `--backend trainium --tp-degree 4 --cp-degree 1`.
- Environment: `<repo>/.venv` from `./scripts/setup_env.sh` (absent on this host today); `HF_HUB_OFFLINE=1` after downloads.
- Measurement configs (June baselines in `benchmark/trn2/*.md`), all seed 42, prompt `"a cinematic shot of a red fox running through a snowy forest"`:
  - Qwen-Image `75e0b4be04f60ec59a75f475837eced720f823b6`: 1024×1024, 20 steps, guidance 4.0 (one forward per step).
  - LTX-2 `47da56e2ad66ce4125a9922b4a8826bf407f9d0a`: 480×704×49, 20 steps, guidance 1.0.
  - Flux / Wan 2.2 / HunyuanVideo: the shape, steps and revision in their `benchmark/trn2/*.md` reproduction table.
- Never delete a model or compile cache without the user's approval. Evidence under `artifacts/verification-2026-10-24-bf16-reduce/`; never commit weights, NEFFs or `.pt` files over 10 MB.

### Legacy dtype per component (the cache-key baseline)

| Model | Component | Sites | Legacy | Key carrier |
|---|---|---|---|---|
| Qwen-Image | DiT | `models/qwen_image/modeling_qwen_image.py:166` (`_row_parallel_like`) | fp32 | `orchestrators/qwen_image.py` `_stage_cache_inputs("generate")` |
| LTX-2 | DiT | `backends/trainium/ltx_2/transformer.py:128` (`_row_parallel_like`) | fp32 | `pipeline/difflet_pipeline.py` `_cache_application_kwargs` (`ltx_2`) |
| HunyuanVideo | DiT, dual-stream | `models/hunyuan_video/modeling_hunyuan_video.py:213, 262, 380, 399` | fp32 | `orchestrators/hunyuan_video.py` `_stage_cache_inputs("generate")` |
| Flux | T5, CLIP | `models/flux/t5/modeling_t5.py:88, 115`; `models/flux/clip/modeling_clip.py:90, 191` | fp32 (DiT already bf16 → mixed) | `orchestrators/flux.py` `_application_kwargs` → `_cache_application_kwargs` |
| Wan 2.1/2.2 | umT5 | `models/wan/umt5/modeling_umt5.py:114` | fp32 (DiT already bf16) | `orchestrators/wan.py` `_stage_cache_inputs` text stage (umT5); transformer stage legacy bf16 |

## Metrics (Wan2.1 report definitions)

| Metric | Definition |
|---|---|
| DiT step | wall time between consecutive denoise iterations in one real generate, step 0 excluded (n = steps − 1); median / p90 / mean / min–max |
| DiT forward | one transformer call, median |
| Host work per step | step median − forward median |
| Denoising loop | total wall of the loop |
| Compile | wall of `difflet compile`, plus per-component build lines from the log |
| E2E warm | `difflet generate` start → output written, artifacts present, page cache warm; 1 discarded run then n = 3, each listed |
| E2E page-cold | same after `sync; echo 3 > /proc/sys/vm/drop_caches`, n = 1 |
| Phase breakdown | start→latents (load, text encode, loop) and latents→output, from generate-log timestamps |
| Device profile | one forward via neuron-profile + neuron-explorer: device time by category (attention, matmul, element-wise/vector, **all-reduce not overlapped**, data-movement wait, other) and all-reduce bytes per forward |
| Quality | final latents rel. L2 + cosine; decoded image/frames PSNR mean + worst, SSIM mean; for: device fp32-reduce vs CPU fp32, device bf16-reduce vs CPU fp32, bf16 vs fp32 device |

## Review Focus

1. **Stale artifact reuse** — a bf16 run against a cache holding the legacy fp32 artifact must get a different artifact (each model task pins its key).
2. **Old caches stay valid** — requesting the legacy dtype hashes to exactly the pre-change key (each model task pins the pre-change hash).
3. **A site that bypasses the dtype** — every `RowParallelLinear(` in the five models passes `reduce_dtype`; Task 1 adds a repo-wide AST test with MiniMax-H3 and `ops_impl` as the only allow-listed omissions.
4. **Flag ignored** — MiniMax-H3 with an explicit `--tp-reduce-dtype` exits with a clear message (Task 1).
5. **Staged subprocess drops the flag** — Qwen, Wan and HunyuanVideo re-exec stage subprocesses from argv lists; each model task asserts the flag is in the child argv.

---

### Task 1: Shared flag, dtype helper, site-coverage test

**Files:**
- Create: `difflet/common/tp_reduce.py`
- Modify: `difflet/cli/main.py` (compile parser near `:62`, generate parser near `:183`), `difflet/cli/stage.py` (near `:47`)
- Modify: `difflet/cli/orchestrators/minimax_h3.py` — reject the flag
- Test: `tests/unit/common/test_tp_reduce.py`, `tests/unit/test_row_parallel_reduce_dtype_coverage.py`

**Interfaces — Produces:**
- `parse_tp_reduce_dtype(value: str | None) -> torch.dtype` — `None`/`"bf16"` → `torch.bfloat16`, `"fp32"` → `torch.float32`, anything else → `ValueError` naming `bf16, fp32`.
- `tp_reduce_cache_fields(requested: torch.dtype, *, legacy: torch.dtype | None) -> dict[str, str]` — `{}` when `requested == legacy`; otherwise `{"tp_reduce_dtype": "bfloat16" | "float32"}`; `legacy=None` (mixed) → always the field.
- argparse dest `tp_reduce_dtype` (string, default `None`, meaning bf16) on compile, generate and stage CLIs — `None` lets MiniMax distinguish "not given".

- [ ] **Step 1: Failing tests** — parse table above incl. `"fp16"` → `ValueError`; `tp_reduce_cache_fields(torch.bfloat16, legacy=torch.float32) == {"tp_reduce_dtype": "bfloat16"}`, `(torch.float32, legacy=torch.float32) == {}`, `(torch.bfloat16, legacy=None) == {"tp_reduce_dtype": "bfloat16"}`; MiniMax generate with `--tp-reduce-dtype bf16` → `SystemExit` whose message contains `"--tp-reduce-dtype is not supported for MiniMax-H3"`. Coverage test: AST-scan `difflet/` for `RowParallelLinear(` calls without a `reduce_dtype` keyword, excluding `ops_impl/` and `backends/trainium/minimax_h3/`; assert the set equals an explicit `EXPECTED_DEFAULT_SITES` list that starts as today's 13 sites (Qwen 1, LTX-2 1, HunyuanVideo 6 incl. `:686/:694`, Flux T5 2 + CLIP 2, Wan umT5 1) — each model task removes its sites from the list, so it reaches empty by Task 8.
- [ ] **Step 2: Run, confirm FAIL** — `.venv/bin/pytest tests/unit/common/test_tp_reduce.py tests/unit/test_row_parallel_reduce_dtype_coverage.py -v`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `.venv/bin/pytest tests/unit -q -x` → pass.
- [ ] **Step 5: Commit** — `feat(cli): --tp-reduce-dtype flag and cache-key helper for tensor-parallel all-reduce precision`.

### Task 2: Latent dump hook (Qwen-Image, LTX-2)

**Files:** Create `difflet/utils/latent_dump.py`; modify the Qwen generate-stage denoise loop (`difflet/cli/orchestrators/qwen_image.py` near `:283`, or `difflet/models/qwen_image/pipeline.py:287` — read first, hook the loop the stage actually runs) and `difflet/models/ltx_2/pipeline.py:557`. Test `tests/unit/utils/test_latent_dump.py`.

**Interfaces — Produces:** `dump_latents(name: str, tensor: torch.Tensor) -> None` — no-op unless env `DIFFLET_DUMP_LATENTS_DIR` is set, else `torch.save(tensor.detach().cpu(), <dir>/<name>.pt)`. Names: Qwen `initial`, `final`; LTX-2 `initial_video`, `initial_audio`, `final_video`, `final_audio`. Env var so it survives stage subprocesses.

- [ ] **Step 1: Failing tests** — `test_noop_without_env` (no file in `tmp_path`); `test_writes_cpu_tensor` (env set → reloaded tensor equal, `device.type == "cpu"`).
- [ ] **Step 2: Run, confirm FAIL.** **Step 3: Implement + hook both loops.** **Step 4: Run** new test + `tests/unit/models/qwen_image tests/unit/models/ltx_2 -q` → pass.
- [ ] **Step 5: Commit** — `feat(debug): DIFFLET_DUMP_LATENTS_DIR dumps initial/final denoise latents`.

### Task 3: Quality metrics + CPU fp32 reference scripts

**Files:** Create `scripts/quality_vs_reference.py`, `scripts/cpu_fp32_reference.py`; test `tests/unit/scripts/test_quality_vs_reference.py`. Add `scikit-image` to the dev extras if missing.

**Interfaces — Produces:**
- `latent_metrics(ref: Tensor, cand: Tensor) -> dict` — `rel_l2` (‖cand−ref‖/‖ref‖ in fp64), `cosine`.
- `frame_metrics(ref: np.ndarray, cand: np.ndarray) -> dict` — uint8 `[F,H,W,3]` (image = F 1); `psnr_mean`, `psnr_worst`, `ssim_mean` (`skimage.metrics.structural_similarity`, `channel_axis=-1`, `data_range=255`).
- `python scripts/quality_vs_reference.py --ref-dir R --cand-dir C --out metrics.json` — reads `final*.pt` and `out.png`/`out.mp4` from each dir.
- `python scripts/cpu_fp32_reference.py --model {qwen_image,ltx_2} --model-path <snapshot> --initial-latents-dir <device dump> --out-dir R [--steps N]` — diffusers `QwenImagePipeline` / `LTX2Pipeline`, fp32, CPU, Global-Constraints config, `latents=` (+ `audio_latents=`) from the dump; writes `final*.pt` (via `callback_on_step_end` on the last step) and the decoded output; loads the text encoder alone, encodes, frees it, then loads transformer + VAE (124 GB host RAM).

- [ ] **Step 1: Failing tests** — identical → `rel_l2 == 0`, `cosine == 1`, `psnr_mean == inf`, `ssim_mean == 1`; `cand = 1.1 * ref` → `rel_l2 ≈ 0.1` (abs 1e-6); uint8 frames off by 1 everywhere → `psnr_mean ≈ 48.13`.
- [ ] **Step 2: Run, confirm FAIL.** **Step 3: Implement.** **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `feat(scripts): CPU fp32 reference runner and latent/frame quality metrics`.

### Tasks 4–8: One commit per model — bf16 all-reduce by default

Each model task has the same shape. **Interfaces — Consumes:** `parse_tp_reduce_dtype`, `tp_reduce_cache_fields` (Task 1). Every construction site passes `reduce_dtype=<the model's resolved dtype>` explicitly; helpers that wrap `RowParallelLinear` take `reduce_dtype: torch.dtype` as a **required keyword-only** argument.

Steps for each:
- [ ] **Step 1: Pin the pre-change key** — on the commit before this task, compute the model's cache key for its measurement config with the legacy dtype and paste the hex string into the test as `EXPECTED_LEGACY_KEY`.
- [ ] **Step 2: Failing tests** in `tests/unit/models/<model>/test_<model>_tp_reduce_dtype.py`:
  - `test_default_is_bf16` — monkeypatch the module's `RowParallelLinear` with a kwargs recorder, build/shard a tiny 2-block model at TP 4 (CPU backend), assert every recorded call has `reduce_dtype == torch.bfloat16` and the call count equals the model's site count.
  - `test_fp32_opt_in` — same with `fp32` → all `torch.float32`.
  - `test_legacy_key_unchanged` — legacy dtype → key == `EXPECTED_LEGACY_KEY`.
  - `test_bf16_key_differs` (or, for components whose legacy is bf16, `test_fp32_key_differs`).
  - Staged models only: `test_stage_argv_carries_flag` — child argv contains `["--tp-reduce-dtype", "fp32"]` when set.
  - Remove this model's sites from `EXPECTED_DEFAULT_SITES` (Task 1 coverage test).
- [ ] **Step 3: Run, confirm FAIL.**
- [ ] **Step 4: Implement** (model-specific notes below).
- [ ] **Step 5: Run** `.venv/bin/pytest tests/unit/models/<model> tests/unit/test_row_parallel_reduce_dtype_coverage.py -q` → pass.
- [ ] **Step 6: Commit** — `perf(<model>): all-reduce tensor-parallel outputs in bf16 by default` (body: inherited NxD fp32 default with file:line, sites changed, cache-key field, `--tp-reduce-dtype fp32` restores the old artifact).

**Task 4 — Qwen-Image.** `_row_parallel_like(linear, *, input_is_parallel, reduce_dtype, reduce_output=True)`; `_replace_qwen_linears_for_tp(transformer, sp_enabled=False, *, reduce_dtype=torch.bfloat16)`; `difflet/models/qwen_image/application.py:112` sets `NeuronConfig(rpl_reduce_dtype=...)` from application kwarg `tp_reduce_dtype`, and `backends/trainium/qwen_image/transformer.py` passes `config.neuron_config.rpl_reduce_dtype` to the sharding call. Key: `tp_reduce_cache_fields(dtype, legacy=torch.float32)` merged into `_stage_cache_inputs("generate")`; flag added to the stage argv (`orchestrators/qwen_image.py:468-500`).

**Task 5 — LTX-2.** `_row_parallel_like(linear, *, input_is_parallel, reduce_dtype)`; `_shard_ltx2_transformer(transformer, tp_degree, rank_util, *, reduce_dtype=torch.bfloat16)` (call at `:571` passes the config value); `difflet/models/ltx_2/application.py:134` sets `rpl_reduce_dtype`. Key: `_cache_application_kwargs` merges `tp_reduce_cache_fields(dtype, legacy=torch.float32)` for `model_name == "ltx_2"` (follow the `attention_mask_accumulation` precedent); `orchestrators/ltx_2.py` passes `tp_reduce_dtype` in application kwargs at `:86`, `:151` and `:166`. The `_global_rms_norm` reduce stays fp32 (add a one-line comment saying so).

**Task 6 — HunyuanVideo.** Thread `reduce_dtype` through the constructors owning `:213`, `:262`, `:380`, `:399` from the model config (same `rpl_reduce_dtype` route). Leave `:686`/`:694` (no own reduce) and the manual reduce at `:743` alone, but give `:686/:694` an explicit `reduce_dtype` too so the coverage test needs no exception. Key: `_stage_cache_inputs("generate")` with `legacy=torch.float32`; argv propagation. HunyuanVideo-1.5 has no `RowParallelLinear` sites — no change.

**Task 7 — Flux text encoders.** Pass `reduce_dtype` at `t5/modeling_t5.py:88, :115` and `clip/modeling_clip.py:90, :191`, sourced from each encoder's `neuron_config` (`rpl_reduce_dtype`). The DiT keeps passing `neuron_config.torch_dtype`, now replaced by the resolved reduce dtype so `fp32` covers the whole model. Key: `orchestrators/flux.py` `_application_kwargs` adds `tp_reduce_dtype`; `_cache_application_kwargs` merges `tp_reduce_cache_fields(dtype, legacy=None)` for `flux` (mixed legacy → field always present; existing Flux artifacts recompile once).

**Task 8 — Wan (2.1 and 2.2) umT5.** Pass `reduce_dtype` at `umt5/modeling_umt5.py:114`; the DiT's `reduce_dtype=dtype` (`modeling_wan.py:326-334, :460-466`) becomes the resolved reduce dtype. Keys: text-encoder stage `legacy=torch.float32`, transformer stage `legacy=torch.bfloat16` (`orchestrators/wan.py:336`); argv propagation (`:386`). After this task `EXPECTED_DEFAULT_SITES` is empty.

### Task 9: Device setup, compile, prove the dtype in the graph

- [ ] **Step 1: Environment** — `./scripts/setup_env.sh`; `.claude/skills/difflet-device-verify/scripts/env_check.sh` → 4 idle cores, venv OK, ≥ 500 GB free. Record NxD's `RowParallelLinear` signature and its cast-before-reduce lines from `.venv/lib/python3*/site-packages/neuronx_distributed/parallel_layers/layers.py` in the evidence doc (premise of the change).
- [ ] **Step 2: Download** Qwen-Image, LTX-2, Flux.1-dev (gated: HF token), Wan 2.2, HunyuanVideo at the pinned revisions. Budget ≈ 1 TB with caches; stop and ask if free space falls under 150 GB.
- [ ] **Step 3: Compile** fp32 and bf16 variants for all five models (fp32 = legacy key, so it reuses nothing on this fresh host but proves the key); record walls.
- [ ] **Step 4: Graph fingerprint** — per artifact, count all-reduce ops by element type in the compiled HLO (`*.hlo_module.pb` under the compile workdir, decoded with neuronx-cc's `hlo_pb2`). Expected: bf16 variants have no hidden-size `f32` all-reduce (LTX-2 keeps only its `[B,S,1]` norm reduces in `f32`); fp32 variants are `f32`. Anything else → bug protocol.
- [ ] **Step 5: Checkpoint** to campaign branch `verify/bf16-reduce-2026-10-24`.

### Task 10: Full campaign — Qwen-Image and LTX-2, fp32 vs bf16

Per model × dtype, one device job at a time:
- [ ] **Step 1: Step / forward / loop** — LTX-2 `python -m benchmark.step_realloop --model ltx_2`, Qwen `benchmark/qwen_step_truth.py` (read first), both extended to pass `--tp-reduce-dtype`; run with `DIFFLET_DUMP_LATENTS_DIR`.
- [ ] **Step 2: E2E warm** (1 discarded + 3) and **page-cold** (1); keep every log and output.
- [ ] **Step 3: Device profile** — one forward, rank 0, real inputs; category split + all-reduce bytes.
- [ ] **Step 4: Sanity gate** — output is a recognisable fox, finite; bf16 vs fp32 latent cosine ≥ 0.95, else stop and triage.
- [ ] **Step 5: Checkpoint.**

### Task 11: Regression check — Flux, Wan 2.2, HunyuanVideo, fp32 vs bf16

- [ ] **Step 1:** per model × dtype: one warm generate at the benchmark config (step median from the log, n = steps − 1) and the output.
- [ ] **Step 2:** `frame_metrics` bf16 vs fp32 output; gate PSNR ≥ 30 dB (above the H.264-only loss level the Wan report measured, ≈ 33 dB) or the visual check passes — otherwise triage.
- [ ] **Step 3: Checkpoint.**

### Task 12: CPU references, quality comparison, report

- [ ] **Step 1:** smoke `cpu_fp32_reference.py --steps 1` per model to time one CPU forward, extrapolate, then run the full references (supervised; Qwen likely several hours). First assert the fp32 and bf16 runs' `initial*.pt` are byte-identical.
- [ ] **Step 2:** `quality_vs_reference.py` for fp32-vs-CPU, bf16-vs-CPU, bf16-vs-fp32, both models.
- [ ] **Step 3:** evidence doc `docs/verification/bf16-reduce-2026-10-24-evidence.md` (`git add -f`, template `references/evidence-doc-template.md`): setup + toolchain, fingerprints, per-metric fp32 | bf16 | Δ tables with the June baseline column, profile table, quality tables, Task 11 table, caveats (one seed/prompt; device text encoder bf16 vs CPU fp32).
- [ ] **Step 4:** publish the same as a private artifact page in the Wan2.1 report's style; send the link.
- [ ] **Step 5:** push; open a draft PR of Tasks 1–8 to `main` with the headline numbers; campaign evidence stays on the `verify/` branch.

### Task 13: PyTorch-native backend runs — Qwen-Image and Wan (added by the user after plan approval)

User: "跑完之后再跑下Torch Native的方式Qwen和Wan就行" — after Tasks 1–12, run Qwen and Wan on the PyTorch-native backend (`--backend neuron`, torch-neuronx; code on `origin/feat/torch-native-backend`, not on `main`).

- [ ] **Step 1:** separate worktree on `origin/feat/torch-native-backend`; environment `/opt/aws_neuronx_venv_torch_neuronx_2_14_0` (torch-neuronx 2.14, the report's native stack). Record its reduce dtype for Qwen/Wan TP linears (read `difflet/backends/neuron/ops_impl/linear.py`).
- [ ] **Step 2:** Qwen-Image at the Task 10 config (1024², 20 steps, g4.0, seed 42), TP4, LNC=2: the Metrics table (step/forward/loop, compile = first-use warm-up delta, warm e2e n=3, page-cold n=1, phase breakdown, profile) with `DIFFLET_DUMP_LATENTS_DIR` if the branch's loop supports it (else add the same hook there, uncommitted to `main`).
- [ ] **Step 3:** Wan2.1-T2V-14B at the Wan report config (480×832×9, 50 steps, g5.0, UniPC, seed 42), same metrics — directly comparable to the report's PyTorch-native column.
- [ ] **Step 4:** quality — Qwen native vs the Task 12 CPU fp32 reference and vs NxD bf16/fp32; Wan native vs NxD (this branch, bf16 DiT + bf16 umT5) frame PSNR/SSIM.
- [ ] **Step 5:** add a "PyTorch native" column to the evidence doc and the artifact page.

## Decisions made on the user's behalf (overturn any before execution)

1. fp32 stays selectable per run via `--tp-reduce-dtype fp32` (needed for the A/B; escape hatch).
2. "The rest" includes the text encoders that TP-shard with `RowParallelLinear` (Flux T5/CLIP, Wan umT5); each lands in its model's commit.
3. Full metric campaign + CPU references for Qwen-Image and LTX-2 only (the original ask); Flux, Wan and HunyuanVideo get the lighter Task 11 regression check.
4. Shared plumbing (flag, latent dump, metric scripts) lands as three commits before the five per-model commits.
5. Measurement configs copy the June benchmarks (20 steps) for comparable step times and a CPU reference that finishes in hours.
6. No bf16 CPU control run unless the fp32-vs-bf16 quality gap is ambiguous.
