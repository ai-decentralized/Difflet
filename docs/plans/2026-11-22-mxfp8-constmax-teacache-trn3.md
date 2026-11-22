# MXFP8 + const-max + ScalarE exp + TeaCache on trn3 (arm E) — test plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, for seven models, how fast the combined trn3 configuration runs against the bf16 baseline: DiT step, warm end-to-end, and quality against CPU fp32. HunyuanVideo-1.5's combined configuration carries no MXFP8 (D4).

**Branch:** `exp/arm-e-mxfp8-constmax-teacache`, from main 5727fe76 (const-max, #174). The user gave the start signal on 2026-11-22; this revision applies the Fable 5 plan review (workflow wf_1b11602d-a8e, 29 findings verified, 9 added by the critic).

## The two sides of the comparison (user, 2026-11-22)

| | baseline A | arm E |
|---|---|---|
| precision | bf16 | `--quant mxfp8` (HV1.5: bf16, D4) |
| attention | attention_cte, softmax exp on VectorE | `--attention const-max`. The const-max kernel's own exp runs on VectorE on trn3 (no switch). ScalarE exp applies to the calls that stay on attention_cte: the unrouted layers (Qwen-Image 3, FLUX 0, LTX-2 3 video attn1, Wan 4 attn1 per expert, HunyuanVideo 5, HV1.5 2) and every cross-attention / audio call |
| TeaCache | off | E1: `--teacache-online-delta 0.6` (primary). E2: E1's exact command with that flag replaced by `--teacache-cadence 2` |
| layout | tp4 (HV1.5: tp2 x CFG2) | same |
| source | the 2026-11-18 arm A numbers, reused (HunyuanVideo: re-timed with the cat prompt) | measured here |

So arm E differs from 11-18's arm D (HV1.5: arm B) by TeaCache and by ScalarE exp in the few cte calls.

## Baseline A (verified on disk: `/mnt/nvme/difflet-exp/constmax-models/<model>/results.json`)

| model | A's tree | shape, steps | guidance | prompt | A step (ms) | A warm e2e (s) | A vs fp32 (dB) |
|---|---|---|---|---|---|---|---|
| Qwen-Image | b05c71d6 | 1024x1024, 50 | `--guidance-scale 7.5` (ignored: unguided) | a cat sitting on a bench | 304.7 | 16.23 | 36.34 |
| FLUX.1-dev | 63b0ae2c | 1024x1024, 50 | embedded 3.5 | a cat sitting on a bench | 222.6 | 11.55 | 23.36 |
| LTX-2 | 63b0ae2c | 512x768x121, 50 | CLI default | a cat walking through a garden | 1,287.1 | 74.81 | 16.55 |
| Wan 2.1 | 7c41de7d | 480x832x81, 50 | CFG 5.0, card negative | a cat walking through a garden | 10,107.5 | 526.24 | 20.15 |
| Wan 2.2 | 7c41de7d | 480x832x81, 40, two experts | CFG 4.0 / 3.0, card negative | a cat walking through a garden | 10,115.2 | 425.49 | 11.00 |
| HunyuanVideo | ceab09f4 | 544x960x129, 50 | embedded 6.0 | fox (11-18) -> **cat** (D5) | 18,485.6 (fox) | 1,003.04 (fox) | 15.36 (fox) |
| HunyuanVideo-1.5 | 47c7df2d | 480x848x121, 50, tp2 x CFG2 | CFG 6.0 (default) | a cat walking through a garden | 9,917.5 | 564.2 | 16.09 |

All seven: seed 42, LNC=2, 12 host threads on NUMA node 1, one process with two identical requests (request 2 is the warm one). All five trees predate main #164, so the exp is on VectorE. They live on the local branch `exp/attn-constmax-models` only (not on origin; the user decides whether to push or tag it). The 11-18 runner scripts are at `10af3dcc^`, the evidence docs at `618365c3^`, both reachable from `origin/exp/mxfp8-constmax-teacache` (kept until the evidence doc is done).

CPU fp32 references exist at these settings (`/home/ubuntu/fp8-bench/fp32_ref/{qwen_image,flux,ltx_2}`, `/mnt/nvme/difflet-exp/fp32_ref_long/*_long`); HunyuanVideo needs a new one with the cat prompt.

## Decisions

- **D1 TeaCache:** E1 = online-delta 0.6; E2 = cadence 2 on spare devices. Not verified on neuron (HV1.5 possibly: 738ec877, no log). Expected skips, from trn2 AOT bf16 runs at other shapes: FLUX, Qwen-Image, HunyuanVideo about 20 of 50; Wan 2.1 18-20; HV1.5 about 17; LTX-2 about half of 20; Wan 2.2 unknown. On Trainium AOT at 50 steps alpha 0.6 gave HunyuanVideo 25.3 dB against cadence 2's 33.0 dB. Online-delta skipping zero steps is reported, not treated as a failure.
- **D2 Qwen-Image and LTX-2:** probe-free TeaCache is wired on neuron first (Tasks 2, 3). Qwen-Image unguided.
- **D3 Settings:** the table above. E's `requests.jsonl` is A's file with only `output` rewritten (HunyuanVideo: prompt replaced by the cat prompt).
- **D4 HV1.5:** tp2 x CFG2, **no MXFP8** (bf16 + const-max + ScalarE + TeaCache); marked "no MXFP8" everywhere; its 11-18 comparison arm is B. It decodes on one core like arm A (`--vae-cores 1`; main's default of 4 cores, #165, would add about 25 s of unrelated gain).
- **D5 HunyuanVideo prompt:** "a cat walking through a garden" for A and E. A is re-timed in full from the ceab09f4 snapshot (VectorE). B, C, D are not re-run and are marked "fox prompt". Ratios use A_cat; the fox A is a cross-check only.
- **D6 `--requests`:** the TeaCache refusal goes for one replica; `--dp > 1` keeps it (not traced, not used here).
- **D7 CPU fp32 last**, and it does not wait for arm F while F is on hold.

## Global constraints

- **Snapshot:** all code tasks (1-6) are finished and pushed first; then `git archive` to `/mnt/nvme/difflet-exp/arm-e/src-<sha>`; every lane runs with `PYTHONPATH` on that snapshot and logs the sha. No rebase between the first device launch and the evidence doc. An engine fix mid-campaign means a new snapshot, and the table marks the sha per row.
- **Devices and CPUs:** E1 lanes on devices 0-6 (each model's 11-18 device: Qwen 0, FLUX 1, LTX-2 2, Wan 2.1 3, Wan 2.2 4, HunyuanVideo 5, HV1.5 6), A_cat on 7, E2 on 8-15. Cores `4d..4d+3`. CPU slice, always 12 threads: `d < 8`: `48+6d..53+6d,144+6d..149+6d`; `d >= 8`: `6(d-8)..6(d-8)+5,96+6(d-8)..96+6(d-8)+5` (NUMA node 0; rows run there are marked). The runner asserts `taskset -c $CPUS nproc` = 12.
- **Before every launch:** the device has no holder whose `NEURON_RT_VISIBLE_CORES` overlaps (every runtime also opens neuron0, so bare `fuser /dev/neuron0` is not the test), and the lane's 12 CPUs show %idle >= 90 in `mpstat -P <slice> 2 1`. Otherwise re-check every 5 min; after 30 min report to the user.
- **Locks:** `flock -s` (shared) on `/home/ubuntu/ws/trainium-kernel-agent/locks/corenode.lock`; exclusive per-device lock in `/home/ubuntu/.claude/jobs/3bceca85/tmp/devlocks` while that directory exists, else `/mnt/nvme/difflet-exp/arm-e/devlocks`.
- **Environment:** `NEURON_NUMA_PIN=0`, `HF_HUB_OFFLINE=1`, `HF_HUB_CACHE=/mnt/nvme/difflet-quant-bench-hub`, `--work-dir` on `/mnt/nvme` with `--keep-work-dir`, a wall-clock `timeout` of about twice the expected time on every device command.
- **Caches (reused in place, no copies):** FLUX `constmax-models/flux/cache/flux-mxcm`, Qwen-Image `qwen_image/cache/D`, LTX-2 `ltx_2/cache/mxcm`, Wan `wan-cache`, HunyuanVideo `hunyuan_video/cache`, HV1.5 `hunyuan_video_15/cache`. E2 starts only after that model's E1 compile has exited 0 and runs generate only. A_cat uses a hard-link copy (`cp -al`, no extra disk) of the HunyuanVideo cache. Budget a full DiT recompile per lane: the killed A_cat run missed the NEFF cache from a new source path. Stop below 100 GB free on `/mnt/nvme` (321 GB now).
- **Outputs:** everything new goes under `/mnt/nvme/difflet-exp/arm-e/<model>/<arm>/`; nothing under `constmax-models/` is modified.
- **Host busy:** each timed run logs `uptime` at start and end; a run ending with 1-min load above 30 is marked "host busy" and re-run once. One retry per lane at most; after killing a lane confirm the device is released.
- **CPU-only jobs:** `TORCH_DEVICE_BACKEND_AUTOLOAD=0`; pytest with `--basetemp=/mnt/nvme/...`, narrow dirs, pinned off the lanes' CPU slices while a timed run is up.
- **Commits:** engine commits first and contiguous (Tasks 1-3, 5), then campaign commits (runner, collector, doc). Push each.

## Review focus

- Request 1 and 2 of an E arm differ in output md5 (png / mp4) or in `[teacache] stats`: the arm is invalid.
- Any two ranks' stats lines differ within one request: the ranks decided differently; the arm is invalid.
- Wan 2.2: the stats line covers the low-noise stage only (reset at the expert switch). Cadence 2 skips 14 of 40 but prints 17 full / 10 skipped.
- A 10-step run cannot skip (warm-up 5, cool-down 5); zero skips below 12 steps is expected.
- The log's "step median" is the full-step time as long as fewer than half the steps are skipped; the effective step is computed separately.

---

### Task 1: TeaCache with `--requests` (one replica)

**Files:** `difflet/cli/main.py` (`_validate_dp`, ~1283-1293); `tests/unit/cli/test_main_dp.py` (`test_teacache_rejected_in_batch_mode`)

- [ ] Tests first: `--requests` with `--teacache-cadence` / `--teacache-online-delta` is accepted on one replica; `--dp 2` + TeaCache still exits 1.
- [ ] Refuse only when `dp > 1`; reword the message. Commit (engine).

### Task 2: probe-free TeaCache for Qwen-Image on neuron

**Files:** `difflet/models/qwen_image/neuron_application.py` (`_TEACACHE_KWARGS` keeps the adaptive names; `_build_pipeline` passes `teacache_cadence`, `teacache_online_delta_alpha`); `difflet/models/qwen_image/pipeline.py` (`_denoise`: after the loop, `print(f"[teacache] stats: {controller.stats()}", flush=True)` when a controller exists); `difflet/cli/orchestrators/qwen_image.py` (`_torchrun_unsupported_options`, `_TORCHRUN_RUNS`); tests `tests/unit/cli/test_orchestrator_qwen_neuron.py` (:504-510, :538, :1434), `tests/unit/backends/test_neuron_qwen_image_q2.py` (:213-214)

- [ ] Tests first: both kwargs reach the orchestrator; the stats line is printed; `teacache_speedup` still refused; true CFG + TeaCache still raises.
- [ ] Implement; run the two test files; commit (engine).

### Task 3: probe-free TeaCache for LTX-2 on neuron

**Files:** `difflet/models/ltx_2/neuron_application.py`; `difflet/cli/orchestrators/ltx_2.py` (`_torchrun_unsupported_options`); tests `tests/unit/backends/test_neuron_ltx2_l2.py` (:120), `tests/unit/cli/test_ltx2_neuron_cli.py` (:28)

- [ ] Tests first as Task 2; adaptive still refused. Implement; commit (engine).

### Task 4: the router forwards `--vae-cores`; the long HunyuanVideo fp32 script takes a prompt

**Files:** `difflet/cli/dp/router.py` (`worker_cli_args`), `tests/unit/cli/dp/test_router.py`; `scripts/fp32_ref/hunyuan_video_long.py` (`--prompt`, default the pinned fox text; output key suffix when set), `tests/unit/scripts/test_fp32_ref_long.py`

- [ ] Tests first: `--vae-cores 1` appears in the worker argv; the long script with `--prompt X` records X and writes to a different key. Implement; two commits (engine, script).

### Task 5: runner and collector (campaign)

**Files:** `scripts/arm_e/run.sh`, `scripts/arm_e/collect.py`

- [ ] `run.sh <model> <arm> <compile|smoke_d|smoke_tc|timed> <device>`; arms `E1`, `E2`. Model flags copied from the six 11-18 scripts at `10af3dcc^` (revisions, `--dual-expert`, Qwen's `--guidance-scale 7.5`, HV1.5's `--vae-cores 1`), requests built from A's `requests.jsonl` (D3), env, slices, locks, timeout and NEFF counts before / after per Global constraints; timed output piped through `scripts/quant_bench/stamp.py`.
  - `smoke_d`: 10 steps, E's flags without TeaCache (the arm-D configuration on this tree).
  - `smoke_tc`: 14 steps (Wan 2.2: 20) with `--teacache-cadence 2`.
- [ ] `collect.py <model>` writes `/mnt/nvme/difflet-exp/arm-e/<model>/results.json`. Per request: `full_step_ms` (the log's step median; assert skipped < n/2), `effective_step_ms` = (request - first-step segment - after-last-step) / n (HV1.5: denoise / 50; the doc says which), `skipped_steps` from the stats lines (Wan 2.2: n - round(loop time / full step), labelled "derived"), stats equal across ranks and requests with `world_size x 2` lines, output md5s. `warm_e2e_s` per lane as on 11-18: Qwen-Image and Wan = request 2 + parent decode-and-save; FLUX and HunyuanVideo = request 2; HV1.5 = encode + denoise + decode; LTX-2 = request 2 (no host decode), plus a second figure with the host decode.
- [ ] `collect.py --self-check`: run on the existing arm A logs it must return 304.7 / 16.23, 222.6 / 11.55, 1287.1 / 74.81, 10107.5 / 526.24, 10115.2 / 425.49, 18485.6 / 1003.04, 9917.5 / 564.2. `bash -n run.sh`; unknown arm exits 2. Commit (campaign).

### Task 6: snapshot

- [ ] All of Tasks 1-5 pushed; narrow CPU tests green; `git archive HEAD` to `/mnt/nvme/difflet-exp/arm-e/src-<sha>`.

### Task 7: device runs

Per lane, in order: `compile` -> `smoke_d` -> `smoke_tc` -> `timed` (E1); E2 `timed` on a device in 8-15 once E1's compile is done.

Gates (a failed gate stops the lane and is reported; no tuning):

- [ ] `smoke_d`: exit 0, finite latents, the `MXFP8 transformer (nki)` line present (absent for HV1.5) with the 11-18 digest where known (Qwen d1231d8ec61363a8, HunyuanVideo 46e0952c1c5a9f74, Wan 2.1 6d28df7cc78386b5), `unwarmed_shapes=[]`, no fallbacks, NEFFs written by this compile contain `attention_const_max`; latent cosine against the 11-18 arm D smoke where one was kept (md5 equality is not expected).
- [ ] `smoke_tc`: every rank prints `skipped_steps 2, full_steps 12` (Wan 2.2: stats present, latents finite).
- [ ] `timed`: the Review focus checks; E's full-step median within 5% of 11-18's arm D step (HV1.5: arm B's 6,272.5 ms) — D: Qwen 216.8, FLUX 165.1, LTX-2 894.8, Wan 2.1 8,018.4, Wan 2.2 8,018.6, HunyuanVideo 8,648.9; Wan prints "36 of 40"; no NEFF added during generate (FLUX: at most 24 small ones, as on main); HV1.5's decode about 59 s.
- [ ] HunyuanVideo A_cat: `/mnt/nvme/difflet-exp/arm-e/hv1_a_cat.sh` (snapshot ceab09f4, device 7, shared corenode lock, device lock, hard-linked cache, the killed run's directory renamed first); two-request timed run. Gate: step median within 2% of 18,485.6 ms. Budget 42 min plus up to 1 h compile.

Lane estimates from the 11-18 logs (compile is a lower bound): Qwen-Image and FLUX 10-20 min, LTX-2 about 30 min, Wan 1-1.5 h each, HV1.5 1-1.75 h, HunyuanVideo 1.7-2.4 h; in parallel about 2.5-3.5 h wall.

### Task 8: collect and first table

- [ ] `collect.py` for the seven models; a first evidence doc with every device number and the fp32 cells empty.

### Task 9: quality against CPU fp32 (last)

- [ ] After the device steps and once the user has seen the first table. Score E1 / E2 request-2 outputs for the six models with existing references, with the scorer arms A-D used (`10af3dcc^:scripts/attn_constmax/score.sh`, `hv15_score.sh`; `compare_difflet_runs.py`): vs fp32, vs A, vs D (HV1.5: vs B). PSNR (mean, worst frame), SSIM, LPIPS, latent cosine. CPUs on node 1, `nice -n 19`.
- [ ] In parallel on node 0 (CPUs 0-47,96-143), `nice -n 19`, `setsid`: the HunyuanVideo cat-prompt reference (`hunyuan_video_long.py --prompt ...`, own snapshot, new output directory, progress file; not `run_long_queue.sh setup`). About 23 h on an idle host. Afterwards check the recorded prompt and noise md5 `df925d6d08d0004d0c1c807cb1df6fea`.
- [ ] Then score HunyuanVideo A_cat, E1, E2 against it and fill the cells.

### Task 10: evidence doc

**Files:** `docs/verification/2026-11-22-mxfp8-constmax-teacache.md`

- [ ] Long table, one row per (model, arm): arm, prompt, precision, attention (const-max with N cte layers on ScalarE), TeaCache mode, full step, effective step, skipped / total, warm e2e (footnote: what it contains for that model), ratio to the baseline, PSNR vs fp32 with the model's 11-18 A-D range beside it, measured / reused, tree sha, device and NUMA node.
- [ ] The headline DiT-step ratio is E's effective step over A's step; the full-step ratio is a separate column. HunyuanVideo's denominator is A_cat. HV1.5 rows say "no MXFP8" and compare with B.
- [ ] Notes: on HunyuanVideo and Wan 2.2, differences under 2 dB against fp32 are not conclusions (single seed, the 11-18 arms scatter by that much); LTX-2's A and E both keep bf16 host state (81a6c597 is not on this tree); md5 of the seven 11-18 `results.json` files.

## Arm F: arm E's features at each model's best parallel layout (user, 2026-11-22)

**ON HOLD (user, 2026-11-22): do not run arm F or its preflight until the user says so. Kept in the plan, not deleted.**

| model | E layout | F layout |
|---|---|---|
| FLUX.1-dev | tp4 | tp4 + SP |
| Qwen-Image | tp4 | tp4 + SP |
| Wan 2.1 | tp4 | tp2 x CFG2 + SP |
| Wan 2.2 | tp4 | tp2 x CFG2 + SP (480x832x81 like A; the user's list says "9 frames" — confirm) |
| LTX-2 | tp4 | tp2 x CFG2 |
| HunyuanVideo | tp4 | tp4 + SP |
| HunyuanVideo-1.5 | tp2 x CFG2 | identical to E |

Known conflicts for a preflight to settle before any F launch: Wan + CFG-parallel turns TeaCache off silently (`difflet/models/wan/pipeline.py:365`); MXFP8 with SP is documented as "do not use" (HunyuanVideo 1,263.8 ms against 659.4 ms bf16 SP); const-max under SP / CFG-parallel has never run; LTX-2 TeaCache under CFG-parallel is unverified. A feature the layout disables is not worked around: the row states which features ran.

## Not in this plan

- Adaptive TeaCache with MXFP8; TeaCache with true CFG on Qwen-Image; TeaCache with `--dp > 1`.
- An alpha sweep; a TeaCache-only bf16 arm; cold e2e, compile time, memory; multi-clip quality sets.
