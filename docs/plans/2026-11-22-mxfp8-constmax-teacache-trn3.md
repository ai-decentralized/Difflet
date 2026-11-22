# MXFP8 + const-max + ScalarE exp + TeaCache on trn3 (arm E) — test plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, for all seven models, how fast the combined trn3 configuration runs against the bf16 baseline: DiT step, warm end-to-end, and quality against CPU fp32.

**Status:** plan only. **No device or CPU run starts until the user says so.**

**Prerequisite P0 — met 2026-11-22:** const-max is on main as 5727fe76 (#174, squash). This plan now lives on `exp/arm-e-mxfp8-constmax-teacache`, cut from that main; every run uses this branch's tree. The old branch `exp/mxfp8-constmax-teacache` (on the unsquashed const-max commits) is left as it was. Main's merge carries the `--attention` flag and the seven models' const-max tables, but **not** the 11-18 arm runner scripts, collectors or evidence docs (they remain on `origin/exp/attn-constmax-rebased`), so Task 4 writes one new runner instead of editing six old ones. One run was started by mistake on 2026-11-22 10:30 UTC (HunyuanVideo arm A, cat prompt, 11-18 tree, device 5) and killed after about a minute; `hunyuan_video/A_cat/` holds only its partial log.

**Branch:** `exp/arm-e-mxfp8-constmax-teacache`, from main 5727fe76.

## The two sides of the comparison (user, 2026-11-22)

| | baseline A | arm E |
|---|---|---|
| precision | bf16 | `--quant mxfp8` (HV1.5: bf16, see D4) |
| attention | attention_cte, softmax exp on **VectorE** | `--attention const-max`; layers not routed to const-max run attention_cte with the exp on **ScalarE** (this tree's default, main #164) |
| TeaCache | off | E1: `--teacache-online-delta 0.6` (primary). E2: `--teacache-cadence 2` (only on devices left idle) |
| parallelism | tp4 (HV1.5: tp2 x CFG2) | same |
| source | the 2026-11-18 arm A numbers, **reused** | measured in this plan |

A is not re-measured: the 11-18 tree predates #164, so its arm A is exactly "bf16 + cte + VectorE exp". The one exception is HunyuanVideo's cat prompt (D5).

## Baseline A (verified on disk, `/mnt/nvme/difflet-exp/constmax-models/<model>/results.json`)

| model | shape and settings | A step (ms) | A warm e2e (s) | A vs fp32 (dB) |
|---|---|---|---|---|
| Qwen-Image | 1024x1024, tp4, 50 steps, unguided | 304.7 | 16.23 | 36.34 |
| FLUX.1-dev | 1024x1024, tp4, 50 steps, guidance 3.5 | 222.6 | 11.55 | 23.36 |
| LTX-2 | 512x768x121, tp4, 50 steps | 1,287.1 | 74.81 | 16.55 |
| Wan 2.1 | 480x832x81, tp4, 50 steps, CFG 5.0 | 10,107.5 | 526.24 | 20.15 |
| Wan 2.2 | 480x832x81, tp4, 40 steps, CFG 4.0/3.0, two experts | 10,115.2 | 425.49 | 11.00 |
| HunyuanVideo | 544x960x129, tp4, 50 steps, guidance 6.0 (fox prompt; see D5) | 18,485.6 | 1,003.04 | 15.36 (fox) |
| HunyuanVideo-1.5 | 480x848x121, tp2 x CFG2, 50 steps, CFG 6.0 | 9,917.5 | 564.2 | 16.09 |

CPU fp32 references exist at these settings for all seven (`/home/ubuntu/fp8-bench/fp32_ref/`, `/mnt/nvme/difflet-exp/fp32_ref_long/`); only HunyuanVideo needs a new one (D5).

## Decisions

- **D1 TeaCache:** E1 = online-delta 0.6 on every model; E2 = cadence 2 where a device is idle. Known before running: online-delta has never run on the neuron backend; on Trainium AOT at 50 steps alpha 0.6 gave HunyuanVideo 25.3 dB against 33.0 dB for cadence 2 (same 20 skipped steps), and it cannot skip more steps than cadence 2.
- **D2 Qwen-Image and LTX-2:** TeaCache is wired on the neuron backend first (Tasks 2, 3), probe-free modes only. Qwen-Image runs unguided (true CFG does not combine with TeaCache in the shared loop).
- **D3 Settings:** the 11-18 settings above.
- **D4 HV1.5** stays tp2 x CFG2 and runs **without MXFP8** (user, 2026-11-22): its arm E is bf16 + const-max + ScalarE exp + TeaCache, no `--quant`, no `DIFFLET_QUANT_TARGETS`. HV1.5 has no default MXFP8 target set (11-09 decision: every measured set costs 4.7-7.5 dB and buys no speed). Every table and the evidence doc mark the HV1.5 row "no MXFP8"; its 11-18 comparison arm is B (bf16 + const-max), not D.
- **D5 HunyuanVideo prompt:** "a cat walking through a garden". A is re-measured in full with the cat prompt (step, warm e2e, output), as a two-request timed run from the 11-18 tree (so the exp stays on VectorE; its own cache, no recompile). The fox A numbers are kept only as a cross-check. A new CPU fp32 reference with the cat prompt is needed (the fox one took 22 h 49 m); like all CPU fp32 work it runs last, after every device step. B, C, D are not re-run and are marked "fox prompt".
- **D6 `--requests`:** the CLI's TeaCache refusal is removed for a single replica. It stays for `--dp > 1`, which was not traced.

## Facts the plan rests on (read-only investigation, workflow wf_25aa1d23-f23)

- Qwen-Image / LTX-2: the shared denoise loops already hold the cadence and online-delta logic and reset the controller per call; the neuron applications only refuse the kwargs and do not pass them on.
- `--quant` refuses only `--teacache-speedup`; online-delta and cadence pass `validate_quant`.
- Per-request reset exists in all pipelines (FLUX `flux/pipeline.py:606`, Wan `wan/pipeline.py:373`, HV1 `hunyuan_video/pipeline.py:216`, HV1.5 `neuron/hunyuan_video15/pipeline.py:172`); the router already forwards the TeaCache flags.
- The warm measurement needs `--requests` chiefly for HV1.5: its first request pays 189-228 s in step 1 (generate loads with `skip_warmup=True`).
- Wan with `--cfg-parallel` silently turns TeaCache off; the Wan arms here do not use it.
- Not yet checked: which engine the const-max kernel's own exp runs on (`attention_const_max_kernel.py` imports `_has_vector_engine_exp`). Task 4 step 1 reads it and the doc states it; "ScalarE" above is established only for the cte layers.

## Global constraints

- Every run: LNC=2, seed 42, the table's steps and prompt.
- Before every device launch: `fuser /dev/neuron<d>` (check holders' `NEURON_RT_VISIBLE_CORES`) and `mpstat -P ALL 2 1`; idle devices only, one lane per device, lanes in parallel.
- Wrap device commands in `flock /home/ubuntu/ws/trainium-kernel-agent/locks/corenode.lock`; `NEURON_NUMA_PIN=0`, `taskset`, `HF_HUB_OFFLINE=1`.
- CPU-only jobs: `TORCH_DEVICE_BACKEND_AUTOLOAD=0`; pytest with `--basetemp=/mnt/nvme/...`, narrow dirs only.
- Disk: `/mnt/nvme` has 324 GB free. Reuse existing cache dirs in place; stop below 100 GB.
- The coordinator owns every long wait. Commit per task, push each commit.

## Review focus

- Request 1 and request 2 of an E arm differ in output or in `[teacache] stats`: state leaked; the arm is invalid.
- No `[teacache] stats` line, or zero skipped steps (online-delta 0.6 may legitimately skip few; zero means the flag was dropped or alpha never triggers — report which).
- LTX-2: the audio stream on a skipped step (the loop keeps its own audio residual).
- Wan 2.2: controller reset at the expert switch.
- The log's "step median" mixes full and skipped steps; it is not the DiT step.

---

### Task 1: remove the TeaCache refusal for `--requests`

**Files:** `difflet/cli/main.py` (`_validate_dp`, ~1283-1293); `tests/unit/cli/test_main_dp.py` (`test_teacache_rejected_in_batch_mode`, line 40)

- [ ] Tests first: `--requests` + each TeaCache flag is accepted with one replica; `--dp 2` + TeaCache still exits 1.
- [ ] Change the condition to `dp > 1`; reword the message.
- [ ] `pytest tests/unit/cli/test_main_dp.py tests/unit/cli/dp -q`; commit; push.

### Task 2: TeaCache (cadence, online-delta) for Qwen-Image on neuron

**Files:** `difflet/models/qwen_image/neuron_application.py` (`_TEACACHE_KWARGS` keeps only the adaptive names; `_build_pipeline` passes `teacache_cadence`, `teacache_online_delta_alpha`); `difflet/cli/orchestrators/qwen_image.py` (`_torchrun_unsupported_options` lets the two flags through, `_TORCHRUN_RUNS` text); tests `tests/unit/cli/test_orchestrator_qwen_neuron.py` (:504-510, :538, :1434), `tests/unit/backends/test_neuron_qwen_image_q2.py` (:213-214)

- [ ] Tests first: the application accepts `teacache_cadence=2` and `teacache_online_delta_alpha=0.6` and hands them to the orchestrator; `teacache_speedup` is still refused; true CFG + TeaCache still raises.
- [ ] Implement; run the two test files; commit; push.

### Task 3: TeaCache (cadence, online-delta) for LTX-2 on neuron

**Files:** `difflet/models/ltx_2/neuron_application.py` (same two changes); `difflet/cli/orchestrators/ltx_2.py` (`_torchrun_unsupported_options`); tests `tests/unit/backends/test_neuron_ltx2_l2.py` (:120), `tests/unit/cli/test_ltx2_neuron_cli.py` (:28)

- [ ] Tests first, as Task 2; adaptive still refused.
- [ ] Implement; run the two test files; commit; push.

### Task 4: arms E1 / E2 in the runner scripts

**Files:** `scripts/attn_constmax/{arms.sh,arms_wan.sh,hv1_arms.sh,hv15_arms.sh,qwen_lane.sh,ltx2_lane.sh}`

- [ ] Step 1: read `attention_const_max_kernel.py` and record which engine its exp uses on trn3.
- [ ] Step 2: add arms `E1` (`--quant mxfp8 --teacache-online-delta 0.6`, const-max on) and `E2` (`--teacache-cadence 2`); `PROMPT` and `TAG` overridable in `hv1_arms.sh`; device-lock dir from `${DEVLOCKS:-/mnt/nvme/difflet-exp/devlocks}`; HV1.5's E arms carry no `--quant` (D4).
- [ ] Step 3: `bash -n`; unknown arm exits 2; commit; push.

### Task 5: device runs — seven lanes, E1 first

Per model, on an idle device, in that model's existing cache dir:

- [ ] compile (TeaCache is not in the cache key; against 11-18's D only the cte layers changed with #164, so expect a partial recompile — HV1 up to 4 h);
- [ ] 10-step smoke: exit 0, finite latents, `[teacache] stats` printed;
- [ ] timed: one process, two identical requests;
- [ ] gates: request 1 and 2 equal in md5 and stats. A failed gate stops the lane; no tuning.
- [ ] E2 on any device still idle after E1's lanes are placed; it shares E1's cache.
- [ ] HunyuanVideo only: arm A with the cat prompt from a snapshot of the 11-18 tree (63b0ae2c, its own cache, no recompile), timed with two requests; its step and warm e2e replace the fox A's in the table. Report the difference from 18,485.6 ms / 1,003.04 s.

### Task 6: collect

**Files:** `scripts/attn_constmax/collect_e.py`, `tests/unit/scripts/test_attn_constmax_collect_e.py`

- [ ] Test first on a synthetic log: `parse_arm_e(text) -> dict` gives per request `full_steps`, `skipped_steps`, `full_step_median_ms`, `effective_step_ms` (DiT loop time / scheduler steps), `wall_s`; raises `ValueError` without a stats line.
- [ ] Write arms `E1` / `E2` into each `results.json`: `step_ms` (effective), `full_step_ms`, `skipped_steps`, `warm_e2e_s` (request 2, the lane's own definition), `md5`. Commit; push.

### Task 7: quality against CPU fp32 (CPU only)

- [ ] Runs last, after every device step is done (user). Then start the HunyuanVideo cat-prompt fp32 reference (`scripts/fp32_ref/hunyuan_video_long.py`, prompt from `FP32_REF_PROMPT`, output `fp32_ref_long/hunyuan_video_long_cat/`, `nice -n 19`, one NUMA node, about 23 h).
- [ ] Score each E arm's request-2 output against the fp32 reference with the scorer arms A-D used; also `E_vs_A` and, where D has the same prompt, `E_vs_D`. PSNR (mean, worst frame), SSIM, LPIPS, latent cosine.

### Task 8: evidence doc

**Files:** `docs/verification/2026-11-22-mxfp8-constmax-teacache.md`

- [ ] One table, 7 models x {A, D, E1, E2}: DiT step (E: effective and full-step), warm e2e, vs fp32; ratios E/A; skipped steps. Each number labelled measured / reused; HunyuanVideo's fp32 cells "pending" until Task 7's reference finishes; B/C/D for HunyuanVideo marked "fox prompt". Commit; push.

## Arm F: arm E's features at each model's best parallel layout (user, 2026-11-22)

**ON HOLD (user, 2026-11-22): do not run arm F or Task 9 until the user says so. Kept in the plan, not deleted.**

**Priority (when released):** below E. Order on the devices: all E1 lanes, then E2, then F on whatever is idle; CPU fp32 last. F is compared with the same baseline A and scored against the same fp32 references (shape, steps, seed, prompt unchanged — only the parallel layout changes, still 4 cores per run).

| model | E layout | F layout | F features |
|---|---|---|---|
| FLUX.1-dev | tp4 | tp4 + SP | as E |
| Qwen-Image | tp4 | tp4 + SP | as E (unguided) |
| Wan 2.1 | tp4 | tp2 x CFG2 + SP | as E |
| Wan 2.2 | tp4 | tp2 x CFG2 + SP | as E; run at 480x832x81 like A (the user's list says "9 frames"; assumed to be the shape the layout was found at — confirm) |
| LTX-2 | tp4 | tp2 x CFG2 | as E |
| HunyuanVideo | tp4 | tp4 + SP | as E, cat prompt |
| HunyuanVideo-1.5 | tp2 x CFG2 | tp2 x CFG2 | identical to E: no separate F run |

Known conflicts, to be settled by Task 9 before any F launch:

- **Wan + CFG-parallel turns TeaCache off silently** (`difflet/models/wan/pipeline.py:365`). As the code stands, Wan 2.1 / 2.2 arm F would run without TeaCache.
- **MXFP8 with SP is documented as "do not use"**: HunyuanVideo MXFP8 + SP measured 1,263.8 ms against 659.4 ms bf16 SP (1.92x slower; rotary op after the MX kernel). Affects FLUX, Qwen-Image, Wan and HunyuanVideo in F.
- **const-max under SP / CFG-parallel has never run**: the per-rank attention shape changes, and the routed-layer tables were measured at tp4.
- **LTX-2 TeaCache under CFG-parallel**: the loop caches the CFG-combined velocity; with the two branches on different ranks this path is unverified.

### Task 9: arm F preflight (read-only, then one 10-step smoke per model)

- [ ] Per model, read the code path for the F layout and answer: does each of MXFP8, const-max, TeaCache stay active; what flag set selects the layout on 4 cores; does `--quant` need a new quantized directory (tp degree is part of the spec).
- [ ] Report the result to the user before any timed F run. A feature that the layout disables is not worked around: that model's F row states which features actually ran.
- [ ] Then per model: compile, 10-step smoke, two-request timed run, same gates and collection as E (arm name `F`).

## Not in this plan

- Adaptive TeaCache with MXFP8; TeaCache with true CFG on Qwen-Image; TeaCache with `--dp > 1`.
- An alpha sweep; a TeaCache-only bf16 arm; cold e2e, compile time, memory; multi-clip quality sets.
