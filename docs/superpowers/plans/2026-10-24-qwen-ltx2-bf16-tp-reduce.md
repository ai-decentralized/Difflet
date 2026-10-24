# Qwen-Image + LTX-2 bf16 tensor-parallel all-reduce — Implementation & Measurement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Device phases (Tasks 5–7) also follow the `difflet-device-verify` skill (supervised runs, checkpoint after every verified step, one-bug-one-commit).

**Goal:** Switch the Trainium (NxD) tensor-parallel all-reduces of Qwen-Image and LTX-2 from fp32 to bf16, keep fp32 selectable, and measure both settings at TP4 / LNC=2 on trn2.3xlarge with the metric set of the Wan2.1 backend report plus PSNR/SSIM against a CPU fp32 reference.

**Architecture:** Both models build NxD `RowParallelLinear` layers through a local `_row_parallel_like` helper that never passes `reduce_dtype`, so they inherit NxD's fp32 default (Flux and Wan pass bf16 explicitly). Make `reduce_dtype` a required argument of those helpers, carried from a new `--tp-reduce-dtype {bf16,fp32}` CLI flag (default `bf16`) through `NeuronConfig.rpl_reduce_dtype` (an existing field). The compile-cache key gains a field only when the dtype is bf16, so existing fp32 artifacts keep their hash and remain usable as the A/B baseline. A latent-dump hook and a quality-metrics script make the CPU-fp32 comparison possible; the measurement campaign then runs fp32 vs bf16 per model.

**Tech Stack:** Python, PyTorch, neuronx-distributed (NxD) via `requirements-neuron.lock`, diffusers 0.38 (CPU fp32 reference), neuron-profile / neuron-explorer, pytest.

**Spec:** this conversation's request — "get qwen and ltx-2 to bf16, run them in tp4 with LNC=2, collect the metrics the report collected, and also collect PSNR/SSIM compared with CPU runs" — plus the report it refers to: claude.ai artifact "Wan2.1 Trainium Backends" (Wan2.1 PyTorch-native vs NxD AoT, measured 2026-10-11). The report's raw records (`/home/ubuntu/plans/trainium_compare/`) are not on this host; its metric definitions are restated below.

## Global Constraints

- Hardware: trn2.3xlarge, 1 Trainium2 chip, `logical-neuroncore-config: 2` (LNC=2 → 4 logical NeuronCores); verified with `neuron-ls`. Every device run uses `--tp-degree 4 --cp-degree 1`.
- Backend: `--backend trainium` (NxD AoT). There is no PyTorch-native backend for Qwen/LTX-2 on `main`.
- Environment: `<repo>/.venv` built by `./scripts/setup_env.sh` from `requirements-neuron.lock` (absent on this host today). `HF_HUB_OFFLINE=1` after the downloads.
- Run configs (identical to the June baselines in `benchmark/trn2/{qwen_image,ltx_2}.md`, so step times are directly comparable):
  - Qwen-Image: `Qwen/Qwen-Image` rev `75e0b4be04f60ec59a75f475837eced720f823b6`, 1024×1024, 20 steps, guidance 4.0 (one DiT forward per step; spec `guidance="none"`), seed 42.
  - LTX-2: `Lightricks/LTX-2` rev `47da56e2ad66ce4125a9922b4a8826bf407f9d0a`, 480×704×49, 20 steps, guidance 1.0, seed 42.
  - Prompt for both: `"a cinematic shot of a red fox running through a snowy forest"`.
- Reduce-dtype values: CLI accepts exactly `bf16` and `fp32`; default `bf16`. Cache-key field name `tp_reduce_dtype`, value `"bfloat16"`, present only when bf16.
- The LTX-2 q/k global-RMS-norm statistic (`_global_rms_norm`, `difflet/backends/trainium/ltx_2/transformer.py:154-163`) stays an fp32 all-reduce: it is a `[B, S, 1]` sum of squares (negligible bytes) and an RMS statistic is conventionally reduced in fp32.
- Qwen SP (`reduce_output=False`) paths are out of scope; only TP4 is run.
- Never delete a model or compile cache without the user's approval. Commit evidence (logs, JSON, PNG/MP4) under `artifacts/verification-2026-10-24-bf16-reduce/`; never commit weights, NEFFs or `.pt` latents over 10 MB.

## Metrics (restated from the Wan2.1 report — every number below is collected per model × {fp32, bf16})

| Metric | Definition |
|---|---|
| DiT step | wall time between consecutive denoise iterations inside one real generate, step 0 excluded (n = 19); median / p90 / mean / min–max |
| DiT forward | one transformer call, median (n = 20 minus step 0) |
| Host work per step | step median − forward median |
| Denoising loop | total wall of the 20-step loop |
| Compile | wall of `difflet compile` (DiT stage, plus per-component build lines from the compile log) |
| End to end, warm | wall of one `difflet generate` process start → output written, artifacts present and page cache warm, mean of n = 3 (each run listed) |
| End to end, page-cold | same, after `sync; echo 3 > /proc/sys/vm/drop_caches`, n = 1 |
| Phase breakdown | start→latents (load, text encode, loop) and latents→output (decode + write), from the generate log timestamps |
| Device profile | one forward via neuron-profile: device time by category (attention, matmul, element-wise/vector, **all-reduce not overlapped**, waiting on data movement, other) + all-reduce bytes per forward |
| Quality vs CPU fp32 | final latents relative L2 error and cosine; decoded frames (LTX-2) / image (Qwen) PSNR mean and worst frame, SSIM mean; plus fp32-reduce vs bf16-reduce directly |

## Review Focus

1. **Stale artifact reuse** — a bf16 generate against a cache dir that already holds an fp32 artifact must compile/load a *different* artifact, never silently reuse the fp32 one (pinned in Task 2).
2. **Old caches stay valid** — `--tp-reduce-dtype fp32` must hash to exactly the pre-change key, or every existing user cache is invalidated (pinned in Task 2).
3. **A call site that bypasses the helper** — any `RowParallelLinear(` constructed in these models without `reduce_dtype` silently reverts to fp32; the helper takes `reduce_dtype` as required keyword-only and a test counts every constructed row-parallel layer (Task 1).
4. **Flag on an unsupported model** — `--tp-reduce-dtype fp32` with Flux/Wan/Hunyuan must fail fast with a clear message rather than be ignored (Task 2).
5. **Staged Qwen subprocess drops the flag** — the Qwen orchestrator re-execs stage subprocesses from an argv list; the flag must survive into the `generate` stage or the parent and child disagree on the cache key (Task 2).

---

### Task 1: Thread `reduce_dtype` into the Qwen and LTX-2 row-parallel layers

**Files:**
- Modify: `difflet/models/qwen_image/modeling_qwen_image.py:163-173` (`_row_parallel_like`), `:185` (`_replace_qwen_linears_for_tp`) and its four `_row_parallel_like(` call sites (`:239, :242, :251, :260`)
- Modify: `difflet/backends/trainium/qwen_image/transformer.py` — the call to `_replace_qwen_linears_for_tp`
- Modify: `difflet/backends/trainium/ltx_2/transformer.py:127-133` (`_row_parallel_like`), `:320` (`_shard_ltx2_transformer`) and its call at `:571`
- Modify: `difflet/models/qwen_image/application.py:112-117`, `difflet/models/ltx_2/application.py:134-139` — set `NeuronConfig(..., rpl_reduce_dtype=...)` from application kwarg `tp_reduce_dtype`
- Test: `tests/unit/models/qwen_image/test_qwen_tp_reduce_dtype.py`, `tests/unit/models/ltx_2/test_ltx2_tp_reduce_dtype.py`

**Interfaces:**
- Produces:
  - `_row_parallel_like(linear: nn.Linear, *, input_is_parallel: bool, reduce_dtype: torch.dtype, reduce_output: bool = True) -> RowParallelLinear` (Qwen; LTX-2 has no `reduce_output`). `reduce_dtype` is keyword-only with **no default**.
  - `_replace_qwen_linears_for_tp(transformer, sp_enabled: bool = False, *, reduce_dtype: torch.dtype = torch.bfloat16) -> None`
  - `_shard_ltx2_transformer(transformer, tp_degree: int, rank_util, *, reduce_dtype: torch.dtype = torch.bfloat16) -> None`
  - Both trainium transformers read `self.config.neuron_config.rpl_reduce_dtype or torch.bfloat16`.
  - Application kwarg `tp_reduce_dtype: str` in `{"bf16", "fp32"}` (default `"bf16"`), mapped to `torch.bfloat16` / `torch.float32` in each `_build_*_config`.

- [ ] **Step 1: Write the failing tests.** In each test file, monkeypatch the module's `RowParallelLinear` name with a recorder that stores its kwargs, force TP size 4 (monkeypatch `_safe_tensor_parallel_size` → 4 for Qwen; pass `tp_degree=4` and a stub `rank_util` for LTX-2), and shard a tiny transformer (2 blocks, small dims; diffusers `QwenImageTransformer2DModel` / `LTX2VideoTransformer3DModel` built from a minimal config).
  - `test_default_is_bf16`: every recorded call has `reduce_dtype == torch.bfloat16`; Qwen records `4 * num_blocks` calls, LTX-2 records the count of `to_out[0]` + `ff.net[2]` + `audio_ff.net[2]` replacements.
  - `test_fp32_opt_in`: with `reduce_dtype=torch.float32` every call has `torch.float32`.
  - `test_helper_requires_reduce_dtype`: `_row_parallel_like(nn.Linear(8, 8), input_is_parallel=True)` raises `TypeError`.
  - `test_application_maps_kwarg` (one per model): `_build_*_config(..., tp_reduce_dtype="fp32")` → `neuron_config.rpl_reduce_dtype is torch.float32`; omitted → `torch.bfloat16`; `"fp16"` → `ValueError` naming the allowed values.
- [ ] **Step 2: Run them and confirm they fail.** `.venv/bin/pytest tests/unit/models/qwen_image/test_qwen_tp_reduce_dtype.py tests/unit/models/ltx_2/test_ltx2_tp_reduce_dtype.py -v` → FAIL (`unexpected keyword argument 'reduce_dtype'`).
- [ ] **Step 3: Implement** the signatures above; the trainium transformers pass the config value into the sharding functions.
- [ ] **Step 4: Run the new tests plus the existing model suites.** `.venv/bin/pytest tests/unit/models/qwen_image tests/unit/models/ltx_2 -q` → all pass.
- [ ] **Step 5: Commit** — `fix(qwen,ltx_2): all-reduce tensor-parallel outputs in bf16, matching Flux and Wan` (body: the inherited NxD fp32 default, file:line evidence, fp32 kept as opt-in).

### Task 2: CLI flag and cache-key separation

**Files:**
- Modify: `difflet/cli/main.py` (compile parser near `:62`, generate parser near `:183`) — add `--tp-reduce-dtype`
- Modify: `difflet/cli/stage.py` (stage parser near `:47`) — same flag
- Modify: `difflet/cli/orchestrators/qwen_image.py:405-430` (`_stage_cache_inputs`, `generate` stage) and the stage argv builder `:468-500`
- Modify: `difflet/cli/orchestrators/ltx_2.py:166-178` (application kwargs) and the compile path `:86, :151`
- Modify: `difflet/pipeline/difflet_pipeline.py:253-278` (`_cache_application_kwargs`)
- Modify: `difflet/cli/orchestrators/{flux,wan,hunyuan_video,hunyuan_video_15,minimax_h3}.py` — reject `fp32`
- Test: `tests/unit/cli/test_tp_reduce_dtype_flag.py`

**Interfaces:**
- Consumes: application kwarg `tp_reduce_dtype` (Task 1).
- Produces: `--tp-reduce-dtype {bf16,fp32}` (default `bf16`) on `difflet compile`, `difflet generate`, and the stage CLI; cache field `"tp_reduce_dtype": "bfloat16"` added to the Qwen `generate` stage inputs and to LTX-2's `_cache_application_kwargs` output only when bf16.

- [ ] **Step 1: Write the failing tests.**
  - `test_qwen_fp32_key_unchanged`: compute `stage_cache_key(orch._stage_cache_inputs("generate", args))` for a fixed `args` (tp 4, 1024×1024) **on the current code first**, paste the 16-hex string into the test as `EXPECTED_FP32_KEY`; assert the fp32 key equals it.
  - `test_qwen_bf16_key_differs`: bf16 key != `EXPECTED_FP32_KEY`.
  - `test_ltx2_cache_kwargs`: `_cache_application_kwargs({"tp_reduce_dtype": "bf16"}, model_name="ltx_2") == {"tp_reduce_dtype": "bfloat16"}`; with `"fp32"` → `None` (identical to the pre-change result for no kwargs); host-only kwargs (`enable_host_pipeline`) stay excluded as today.
  - `test_qwen_stage_argv_carries_flag`: the generate-stage argv contains `["--tp-reduce-dtype", "fp32"]` when set.
  - `test_flag_rejected_for_other_models`: `difflet generate --model-id black-forest-labs/FLUX.1-dev --tp-reduce-dtype fp32 ...` parse+dispatch raises `SystemExit` with a message containing `"--tp-reduce-dtype is only supported for Qwen-Image and LTX-2"`.
- [ ] **Step 2: Run, confirm FAIL.** `.venv/bin/pytest tests/unit/cli/test_tp_reduce_dtype_flag.py -v`
- [ ] **Step 3: Implement.** Follow the `attention_mask_accumulation` precedent in `_cache_application_kwargs` (additive-only: no field for the old behaviour).
- [ ] **Step 4: Run** `.venv/bin/pytest tests/unit -q -x` → all pass (also `scripts/check_quick.sh` if it runs in this env).
- [ ] **Step 5: Commit** — `feat(cli): --tp-reduce-dtype for Qwen-Image and LTX-2; bf16 artifacts get their own cache key`.

### Task 3: Latent dump hook

**Files:**
- Modify: Qwen generate-stage denoise loop (`difflet/cli/orchestrators/qwen_image.py` stage loop near `:283`; fall back to `difflet/models/qwen_image/pipeline.py:287` if that is the loop the stage runs — confirm by reading before editing)
- Modify: `difflet/models/ltx_2/pipeline.py:557` loop (video and audio latents)
- Create: `difflet/utils/latent_dump.py`
- Test: `tests/unit/utils/test_latent_dump.py`

**Interfaces:**
- Produces: `dump_latents(name: str, tensor: torch.Tensor) -> None` — no-op unless env `DIFFLET_DUMP_LATENTS_DIR` is set; otherwise `torch.save(tensor.detach().cpu(), <dir>/<name>.pt)`. Names written: `initial.pt`, `final.pt` (Qwen, packed); `initial_video.pt`, `initial_audio.pt`, `final_video.pt`, `final_audio.pt` (LTX-2). Env (not a CLI flag) so it survives Qwen's stage subprocesses unchanged.

- [ ] **Step 1: Failing tests** — `test_noop_without_env` (no file created in `tmp_path`), `test_writes_cpu_tensor` (env set → file exists, reloaded tensor equal, `device.type == "cpu"`).
- [ ] **Step 2: Run, confirm FAIL.**
- [ ] **Step 3: Implement** and call it before the first step and after the last step in both loops.
- [ ] **Step 4: Run** the new test and `tests/unit/models/qwen_image tests/unit/models/ltx_2 -q` → pass.
- [ ] **Step 5: Commit** — `feat(debug): DIFFLET_DUMP_LATENTS_DIR dumps initial/final denoise latents`.

### Task 4: Quality-metrics and CPU-reference scripts

**Files:**
- Create: `scripts/quality_vs_reference.py`
- Create: `scripts/cpu_fp32_reference.py`
- Test: `tests/unit/scripts/test_quality_vs_reference.py`

**Interfaces:**
- Produces:
  - `latent_metrics(ref: Tensor, cand: Tensor) -> dict` with keys `rel_l2` (‖cand−ref‖/‖ref‖, fp64) and `cosine`.
  - `frame_metrics(ref_frames: np.ndarray, cand_frames: np.ndarray) -> dict` (uint8 `[F,H,W,3]`; an image is F=1) with `psnr_mean`, `psnr_worst`, `ssim_mean`; SSIM via `skimage.metrics.structural_similarity(channel_axis=-1, data_range=255)` (add `scikit-image` to the dev extras if missing).
  - CLI: `python scripts/quality_vs_reference.py --ref-dir R --cand-dir C --out metrics.json` reading `final*.pt` and the decoded `out.png` / `out.mp4` from each dir.
  - CLI: `python scripts/cpu_fp32_reference.py --model {qwen_image,ltx_2} --model-path <snapshot> --initial-latents-dir <device dump dir> --out-dir R` — diffusers `QwenImagePipeline` / `LTX2Pipeline` in fp32 on CPU with the Global Constraints config, `latents=` (and `audio_latents=` for LTX-2) taken from the device run's `initial*.pt`, writes `final*.pt` (via `callback_on_step_end` on the last step) and the decoded output. To fit 124 GB host RAM, encode the prompt with only the text encoder loaded, free it, then load transformer + VAE.
- [ ] **Step 1: Failing tests** — identical inputs → `rel_l2 == 0`, `cosine == 1`, `psnr_mean == inf`, `ssim_mean == 1`; `cand = ref + 0.1*ref` → `rel_l2 ≈ 0.1` (abs 1e-6); uint8 frames differing by a constant 1 everywhere → `psnr_mean ≈ 48.13` dB.
- [ ] **Step 2: Run, confirm FAIL.**
- [ ] **Step 3: Implement** both scripts.
- [ ] **Step 4: Run tests → pass.** Smoke `cpu_fp32_reference.py` with `--steps 1` (add the flag) on each model to measure seconds per CPU forward; record it and extrapolate the 20-step cost before the full run.
- [ ] **Step 5: Commit** — `feat(scripts): CPU fp32 reference runner and latent/frame quality metrics`.

### Task 5: Device setup, compile both variants, prove the dtype in the graph

Device phase — supervise long jobs with `.claude/skills/difflet-device-verify/scripts/supervise.sh` inside a Monitor.

- [ ] **Step 1: Environment.** `./scripts/setup_env.sh`; then `.claude/skills/difflet-device-verify/scripts/env_check.sh` → 4 idle NeuronCores, venv OK, disk ≥ 500 GB free. Confirm NxD's default: `grep -n "reduce_dtype" .venv/lib/python3*/site-packages/neuronx_distributed/parallel_layers/layers.py` shows `RowParallelLinear` defaulting to `torch.float32` and casting before the reduce — record the lines in the evidence doc (this is the premise of the whole change).
- [ ] **Step 2: Download** both models at the pinned revisions (`difflet download` or `huggingface-cli download --revision ...`).
- [ ] **Step 3: Compile 4 artifacts** (Qwen fp32/bf16, LTX-2 fp32/bf16) with the Global Constraints shapes and `--tp-reduce-dtype` set explicitly; save each compile log and wall time. Expected: 4 distinct cache directories (fp32 dirs carry the pre-change key).
- [ ] **Step 4: Graph fingerprint.** For each artifact, list the all-reduce ops in the compiled HLO (the `*.hlo_module.pb` / `.hlo` under the compile workdir; decode with `neuronx_cc`'s bundled `hlo_pb2` or `neuron-profile view` metadata) and count by element type. Expected: bf16 variants — every hidden-size all-reduce is `bf16`; fp32 variants — `f32`; LTX-2 both variants — additionally the `[B,S,1]` norm-statistic all-reduces in `f32`. A bf16 artifact with any hidden-size `f32` all-reduce is a FAIL → bug protocol.
- [ ] **Step 5: Checkpoint** — commit compile logs + fingerprint table to the campaign branch `verify/bf16-reduce-2026-10-24`.

### Task 6: Performance and profile runs

For each model × {fp32, bf16}, one device job at a time, nothing else on the host:

- [ ] **Step 1: DiT step / forward / loop.** LTX-2: `python -m benchmark.step_realloop --model ltx_2` extended to forward `--tp-reduce-dtype`. Qwen: `benchmark/qwen_step_truth.py` (read it first; extend the same way). Run with `DIFFLET_DUMP_LATENTS_DIR=<run dir>`. Record per-step deltas (n = 19), forward times, loop total.
- [ ] **Step 2: Warm e2e** — 1 discarded page-cache warm-up, then 3 timed `difflet generate` runs; keep each run's log and output.
- [ ] **Step 3: Page-cold e2e** — drop caches, 1 timed run.
- [ ] **Step 4: Device profile** — capture one forward with `neuron-profile capture` on the compiled DiT NEFF (rank 0) with real inputs from the dumped latents, and summarise by category with `neuron-explorer` (as the report did). Record all-reduce exposed time and bytes per forward.
- [ ] **Step 5: Sanity gate** — bf16 output must be a recognisable fox (look at it) and finite; latents cosine vs the same model's fp32-reduce run ≥ 0.95 at these step counts — otherwise stop and triage (`references/failure-triage.md`).
- [ ] **Step 6: Checkpoint** — curate logs/JSON/outputs, commit, push.

### Task 7: CPU fp32 references, quality comparison, report

- [ ] **Step 1: CPU references** (one per model, supervised; Qwen likely several hours — use the Task 4 extrapolation) from the fp32-reduce run's `initial*.pt`. Before using one initial-noise file for both variants, assert `initial*.pt` of the fp32 and bf16 runs are byte-identical (same seed ⇒ same noise; if not, the comparison is invalid).
- [ ] **Step 2: Metrics** — `quality_vs_reference.py` for: fp32-reduce vs CPU, bf16-reduce vs CPU, bf16 vs fp32 (device), each model → `metrics.json`.
- [ ] **Step 3: Evidence doc** `docs/verification/bf16-reduce-2026-10-24-evidence.md` (`git add -f`) following `references/evidence-doc-template.md`: setup table (toolchain versions from `.venv`), per-metric tables fp32 | bf16 | Δ, June-baseline column for step time, profile category table, quality tables, fingerprint table, caveats (one seed/prompt; CPU text encoder fp32 vs device bf16 encoder, as in the Wan report).
- [ ] **Step 4: Publish** the same content as a private artifact page in the style of the Wan2.1 report; give the user the link.
- [ ] **Step 5: Final commit + push**; open a draft PR from the code branch (Tasks 1–4) to `main` with the headline numbers, keeping campaign evidence on the `verify/` branch.

## Decisions made on the user's behalf (overturn any before execution)

1. **bf16 becomes the default; fp32 stays selectable** via `--tp-reduce-dtype fp32` — needed anyway for the A/B, and an escape hatch if a later prompt shows a regression.
2. **LTX-2's RMS-norm statistic all-reduce stays fp32** (tiny, numerically sensitive).
3. **Run configs copy the June benchmarks** (Qwen 1024² 20 steps; LTX-2 480×704×49 20 steps g1.0) rather than the CLI's 50/40-step verification shapes — directly comparable step times and a CPU reference that finishes in hours, not a day.
4. **NxD backend only** — no PyTorch-native backend exists for these models on `main`; the report's "PyTorch native vs NxD" axis becomes "fp32-reduce vs bf16-reduce".
5. **No bf16 CPU control run** (the Wan report flagged it as open); add it only if the fp32-vs-bf16 quality gap is ambiguous.
