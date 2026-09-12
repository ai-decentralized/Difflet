# Working log — probe-free TeaCache on TPU, on-device verification

Date: 2026-09-11
Branch: `tpu-teacache` @ `8e682a3` (code: `993b56a`)
Plan: `docs/plans/2026-09-08-tpu-teacache.md` — Status there is "implemented, CPU-verified, on-device verification pending". This log records the device run.

## Environment (checked 2026-09-11)

| item | value |
|---|---|
| host | GCP, Linux 5.19, 188 GiB RAM, `/mnt/models` 1.6 TiB free |
| chips | `/dev/vfio/0-3` present, no python holders (`ps` clean) |
| venv | `/mnt/models/tpuenv312` — Python 3.12.14, torch 2.9.0+cpu, torch_xla 2.9.0, jax 0.7.1 |
| gotcha | needs `LD_LIBRARY_PATH=/mnt/models/uvpy/cpython-3.12.14-linux-x86_64-gnu/lib` or `import _XLAC` fails with `libpython3.12.so.1.0: cannot open shared object file` (documented in `benchmark/v5e/wan_2_2.md`) |
| weights | `HF_HOME=/mnt/models/hf`: Qwen-Image, Wan2.2-T2V-A14B, Wan2.1, LTX-2, HunyuanVideo snapshots present |
| difflet | editable install → `/mnt/models/Difflet/difflet` |

Common env for every command below:

```bash
export LD_LIBRARY_PATH=/mnt/models/uvpy/cpython-3.12.14-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH
export HF_HOME=/mnt/models/hf DIFFLET_BACKEND=tpu
P=/mnt/models/tpuenv312/bin/python
```

Results go under `benchmark/v5e-teacache-<mode>/` (`DIFFLET_BENCH_DEVICE`) so the
committed `benchmark/v5e/` baseline is never overwritten.

## Progress

### Step 0 — unit tests in the TPU venv (no chip) — DONE, green

- `pytest` was not in `tpuenv312`; installed with `/mnt/models/uvbin/uv pip install --python /mnt/models/tpuenv312/bin/python pytest` (also `python-dotenv`, which `difflet serve` requires and the venv lacked).
- Plan's step-0 files: `tests/unit/serving/test_qwen_common_orchestrator.py tests/unit/models/wan/test_wan_tpu_application.py` → **32 passed**.
- Wider sweep `tests/unit/serving/test_model_registry.py test_serve_cli.py test_wan_video_adapter.py tests/unit/pipeline` → 356 passed / 5 failed, all environmental and pre-existing on `main`:
  3× `No module named 'neuronx_distributed'` (Wan skeleton / Wan-LTX2 teacache tests; same 3 the plan lists), 2× dotenv tests that ran before `python-dotenv` was installed.
- Gotcha: with the bare venv the backend auto-detects **tpu** (torch_xla importable), and `tests/unit/pipeline/test_pipeline.py` uses `unit_dummy` models registered for trainium only → 12 extra failures that vanish with `DIFFLET_BACKEND=trainium`. Not TeaCache-related; a test-isolation nit.

### Step 1 — Qwen-Image A/B (baseline / cadence 2 / online-delta 0.6) — DONE

Command (baseline; the A/B runs add `DIFFLET_BENCH_TEACACHE_CADENCE=2` / `DIFFLET_BENCH_TEACACHE_ONLINE_DELTA=0.6`):

```bash
DIFFLET_BENCH_DEVICE=v5e-teacache-baseline DIFFLET_BENCH_SAVE_PNG=/mnt/models/teacache_runs/qwen_baseline.png \
  $P -m benchmark.bench --model qwen_image --backend tpu --skip-download --iters 3 \
  > /mnt/models/teacache_runs/qwen_baseline.log 2>&1
```

- **Run 1 (baseline) died after denoise** in every rank:
  `TypeError: QwenImageServingStageAdapter._decode() missing 1 required positional argument: 'request'`
  (`benchmark/adapters/tpu.py:201`). Pre-existing on `main`: `f31e433` changed `_decode(packed, request)` and left the bench worker calling `_decode(latents)`. Not caused by the TeaCache commit; it means the TPU Qwen benchmark has been broken since Aug 31. Fixed and committed as **`f9689ec`** (request now carries `height`/`width`, passed to `_decode`). Also killed the orphaned parent (workers had exited, parent was blocked on the reply queue — the chips were free again, no `/dev/vfio` holders).
- **Run 2 (baseline, after fix): ok.** `benchmark/v5e-teacache-baseline/qwen_image.json`, PNG `/mnt/models/teacache_runs/qwen_baseline.png` (1,066,634 B).
  e2e cold 80.6 s (load 8.4 s + first-execution compile), e2e warm 13.19 s (n=3), stage: encode 2.04 / **denoise 10.12** / decode 1.21 s; per-step **503 ms synced**, **286 ms natural** (warm natural e2e 8.72 s); peak HBM 9.62 GB; latents finite, mean 0.0136 std 0.474.
  → reproduces `benchmark/v5e` (10.14 s / 508 / 291) within noise. The `*** SIGTERM received …` stack dumps at the end of the log are the parent's shutdown of the workers, not a crash.
- **Run 3 (cadence 2): ok.** `benchmark/v5e-teacache-cadence2/qwen_image.json`, PNG `/mnt/models/teacache_runs/qwen_cadence2.png`. Worker log shows `[teacache] probe-free controller enabled for qwen_image/1024x1024: cadence=2 online_delta_alpha=0.0` on all 4 ranks.
  e2e cold 77.4 s, e2e warm **10.81 s** (11.04 / 10.55 / 10.85), stage: encode 2.04 / **denoise 7.60** / decode 1.21 s; per-full-step 500.6 ms median synced (14 deltas = **15 full steps → 5 skipped**, exactly the plan's {6,8,10,12,14}); natural e2e 7.54 / 7.96 s, natural per-step 292–298 ms; peak HBM 9.62 GB (unchanged); latents finite, mean 0.01365 std 0.4742.
  → **denoise 7.60 / 10.12 = 0.751×**, the predicted 0.75×. Per-full-step latency unchanged (500 vs 500 ms) → no recompiles/scalar leaks. First warm iteration 11.04 s vs later 10.55/10.85: the extra compile for the with-residual graph is ≲ 0.3 s, not DiT-sized.
  Quality vs. baseline PNG: **mean abs 0.00188/pixel, max 0.22, 0.08 % of pixels > 0.05, PSNR 47.3 dB**; side-by-side (`/mnt/models/teacache_runs/side_baseline_cadence2.png`) indistinguishable.
- **Bug 2 found while reading the results:** the JSON/MD has no `teacache` block although `benchmark/README.md` (993b56a) says it does — `run_generate` returned it but `BenchResult` had no field and `bench.py` dropped it; the skip count was only inferable from the 14-vs-19 delta count. Fixed as **`4ce1dff`** (`BenchResult.teacache`, carried from the warm generate, rendered under the stage breakdown). The online-delta run below is the first with the block in the output; cadence 2 will be re-run at the end to get it too.
- **Run 4 (online-delta 0.6): ok**, `benchmark/v5e-teacache-online-delta/qwen_image.json` — first run with the `teacache` block: `{'full_steps': 15, 'skipped_steps': 5, 'probe_calls': 0, 'last_delta_estimate': 0.0514, 'cache_initialized': True}`.
  e2e cold 78.1 s, e2e warm 10.88 s (11.10 / 10.83 / 10.70), stage: encode 1.84 / **denoise 7.62** / decode 1.24 s; per-full-step 500.5 ms median synced (n=14); **natural e2e 10.29 / 10.09 s, natural per-step 485 ms**; peak 9.62 GB; latents finite.
  Quality vs. baseline: mean abs 0.00266/px, PSNR 44.3 dB (vs. cadence 2: 0.00328, 42.8 dB — it picked different steps; the stats block only carries counts, not indices).
  → On the **synced** basis it matches cadence 2 (7.62 vs 7.60 s: same 5 skips). On the **natural** basis — what serving actually runs — it is **slower than the baseline**: 10.1–10.3 s vs 8.72 s baseline / 7.5–8.0 s cadence 2. The plan predicted the mechanism: `record_full_step` does `float(...)` on every full step, which forces a device sync and destroys the tracing/execution overlap the natural loop has (485 ms/step natural ≈ the 500 ms synced figure, vs. 286 ms natural for baseline). 15 synced steps (15 × 0.50 = 7.5 s) cost more than 20 overlapped ones (20 × 0.29 = 5.7 s).
  **Conclusion for Qwen on TPU: use `--teacache-cadence`; `--teacache-online-delta` is a net loss in serving unless the sync is removed** (e.g. decide from a delta that is one step stale so the readback can overlap, or keep the comparison on-device and read a single bool — a follow-up, not in this branch).

- **Run 5 (cadence 2 re-run at `dc215f5`, for the JSON block): ok**, `benchmark/v5e-teacache-cadence2/qwen_image.{json,md}` now carry `teacache: {'full_steps': 15, 'skipped_steps': 5, …}` and the report line "TeaCache: 15 full / 5 skipped steps". denoise 7.63 s, warm e2e 10.59 s, natural e2e 7.35 / 7.88 s, per-full-step 500 ms — and the PNG is **bit-identical** to run 3's. Reproducible.

### Step 2 — Wan 2.2 A/B — DONE

```bash
$P benchmark/wan_tpu_run.py --steps 20 --iters 3 --out /mnt/models/teacache_runs/wan_<mode> [--teacache-cadence 2 | --teacache-online-delta 0.6]
```

- **Run 1 (baseline) died before loading a weight**, every rank: `ModuleNotFoundError: No module named 'benchmark'` at `wan_tpu_run.py:126 from benchmark.harness import RealLoopStepTimer`. Running the script by path puts `benchmark/` at `sys.path[0]`; the spawn workers inherit that. The import came in with `9fffe91` (the commit that wrote `benchmark/v5e/wan_2_2.md`), so that run must have used `PYTHONPATH=.` or `-m`; the documented command never worked as written. Parent blocked on its reply queue → killed. **Bug 3**, fixed as **`2bf93f3`** (repo root inserted at `sys.path[0]` in the script).
- **Run 2 (baseline, after fix): ok.** `/mnt/models/teacache_runs/wan_baseline.{json,npy,mp4,latents.pt}`. 480×832×9, 20 steps, guidance 1.0, seed 42, single expert, tp=4.
  cold 54.2 s (load 4.7 + compile), warm wall **12.24 / 12.18 s** synced, **12.17 / 12.14 s** natural; **609 ms/step**; encode 0.68 s; HBM 6.985 GB after load, 7.51 GB peak; latents finite mean −0.058 std 0.461; host VAE decode 24.8 s. `teacache: null` in every record. Matches `benchmark/v5e/wan_2_2.md` (12.20 s / 610 ms).
- **Run 3 (cadence 2): ok.** `/mnt/models/teacache_runs/wan_cadence2.{json,npy,mp4}`. Log: `[teacache] probe-free controller enabled for wan/480x832x9: cadence=2`, and after every generate `[teacache] stats: {'full_steps': 15, 'skipped_steps': 5, 'probe_calls': 0, …}` on all ranks; the JSON `teacache` block carries the same.
  cold 49.8 s, warm wall **9.09 / 9.11 s** synced, **9.11 / 9.11 s** natural (as expected, no difference on Wan — UniPC syncs every step anyway); per-full-step **606 ms** (14 deltas), unchanged from 609; encode 0.70 s; peak HBM 7.513 GB (unchanged); latents finite mean −0.0589 std 0.4608; host decode 23.6 s.
  → **9.10 / 12.18 = 0.747×**, predicted ≈ 9.2 s. First warm iteration shows no extra compile (9.09 vs 9.11).
  Quality vs. baseline video (decoded, 9 frames, 0..1): **mean abs 0.0126/px, max 0.54, PSNR 31.3 dB**, per-frame 0.0140 → 0.0121 (falls with frame index). Visibly identical at 1× (`/mnt/models/teacache_runs/side_wan_frame4.png`: baseline / cadence 2 / 4× diff); the diff concentrates on fur texture and tree-bark edges, no structural or motion change. Note this is ~7× the Qwen figure — video at 20 UniPC steps is more sensitive to the residual extrapolation than the image at 20 Euler steps; nothing in the plan sets a video threshold, flagging it for the operator rather than calling it a pass/fail.
- **Run 4 (online-delta 0.6): ok.** `/mnt/models/teacache_runs/wan_online_delta.{json,npy,mp4}`. `teacache: {'full_steps': 15, 'skipped_steps': 5, 'last_delta_estimate': 0.082, …}` every iteration.
  cold 50.4 s, warm wall **9.11 / 9.12 s** synced, **9.16 / 9.11 s** natural; per-full-step 607–611 ms; peak 7.513 GB; latents finite mean −0.0584.
  → Same 0.75× as cadence 2 and — unlike Qwen — **no natural-basis penalty**, exactly as the plan said (Wan's UniPC loop already ends every step on the host, so the `float()` sync is free here).
  Quality vs. baseline video: **mean abs 0.0194/px, PSNR 29.1 dB** (cadence 2: 0.0126 / 31.3 dB); vs. cadence 2: 0.0208 / 28.2 dB — it chose different steps and drifts more. Same 5-skip count, worse fidelity → for Wan, cadence 2 is also the better default; alpha 0.6 is untuned for this model on TPU.

### Step 3 — serving smoke — DONE (Qwen and Wan)

```bash
$P_BIN/difflet serve --model-id Qwen/Qwen-Image --revision 75e0b4be… --tp-degree 4 --cp-degree 1 \
    --height 1024 --width 1024 --host 127.0.0.1 --port 8093 --teacache-cadence 2
curl -X POST :8093/v1/chat/completions  # same prompt/seed/steps/guidance as the MATRIX row
```

- **Server 1** (code at `f3e2021`): startup 21:13:53 → ready 21:16:41 (2 m 48 s incl. 4-step startup smoke on all 4 replicas, "Qwen shared-worker generation smoke passed" ×4). `[teacache] probe-free controller enabled for qwen_image/1024x1024: cadence=2` ×4. `/ready` 200.
  One 1024²/20-step request: **HTTP 200 in 7.47 s** (RESULTS.md's no-TeaCache serving figure: 8.75 s → −1.3 s ≈ 5 skipped steps × ~0.29 s natural, consistent with the bench). PNG 1,066,368 B; vs. the bench cadence-2 PNG mean abs 0.00126/px, PSNR 50 dB (bench and serve are not bit-identical even without TeaCache — different process/encode path; not a TeaCache effect).
  SIGTERM → process gone in 10 s, no leftover workers, chips free.
- **Bug 5:** the plan's acceptance line `qwen.tpu_teacache stats=…` never appeared — not in the console log nor in `logs/Difflet-20260911_211353.log`. Root cause: the resident worker is a spawned process with no logging handler, so **no** worker-side `logger.info` reaches any log (checked: `worker_main start` is absent too). Wan/LTX-2/Hunyuan `print(..., flush=True)` for the same reason. Fixed as **`dc215f5`** (Qwen TPU loop now prints `[teacache] stats: {...}`); plan doc's expected line updated to match.
- **Server 2 (code at `dc215f5`): pass.** Ready after 130 s. Startup smoke printed `[teacache] stats: {'full_steps': 4, 'skipped_steps': 0, …}` ×4 (4 steps → nothing inside the skip window, as the plan says). Two 20-step requests: **HTTP 200 in 7.75 s and 7.57 s**, each followed by `[teacache] stats: {'full_steps': 15, 'skipped_steps': 5, 'probe_calls': 0, …}`. The served PNGs are **bit-identical** to each other and to server 1's (mean abs 0.00000) — deterministic across requests and process restarts. SIGTERM shutdown clean, chips released.
- **Wan serving smoke** (added — `wan_tpu_run.py` calls `build_wan_orchestrator` directly, so the serving adapter's `_teacache_kwargs` → `TpuWanApplication.load_eager` path was only unit-tested): `difflet serve --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --tp-degree 4 --height 480 --width 832 --num-frames 9 --port 8094 --teacache-cadence 2`, log `/mnt/models/teacache_runs/serve_wan_cadence2.log`. **Pass.** `[teacache] probe-free controller enabled for wan/480x832x9: cadence=2` ×4 → the adapter's `_teacache_kwargs` → `TpuWanApplication.load_eager` → `build_wan_orchestrator` chain works on device. Startup smoke is a 1-step generate: `stats: {'full_steps': 1, 'skipped_steps': 0, 'cache_initialized': False}` ×4. Ready 21:24:39. One `POST /v1/videos/sync` (832×480×9, 20 steps, guidance 1.0, seed 42): **HTTP 200 in 30.4 s** (≈ 9 s denoise + ~24 s host VAE decode, i.e. the bench's numbers), 205,708 B mp4 at `/mnt/models/teacache_runs/serve_wan_cadence2.mp4`, followed by `stats: {'full_steps': 15, 'skipped_steps': 5, …}` ×4. SIGTERM → gone in 10 s, chips free.

### Step 4 — quality — DONE (figures inline in steps 1–3)

| pair | mean abs /px | PSNR | note |
|---|---|---|---|
| Qwen baseline vs cadence 2 | 0.00188 | 47.3 dB | same magnitude as the fused-attention A/B's 0.002 acceptance figure |
| Qwen baseline vs online-delta 0.6 | 0.00266 | 44.3 dB | |
| Qwen cadence 2: bench vs serve | 0.00126 | 50.0 dB | bench/serve differ slightly even without TeaCache |
| Qwen cadence 2: run vs run (bench, serve×3) | 0.00000 | ∞ | bit-identical |
| Wan baseline vs cadence 2 (decoded video) | 0.0126 | 31.3 dB | fur/bark texture only, no structure/motion change |
| Wan baseline vs online-delta 0.6 | 0.0194 | 29.1 dB | |

Helpers: `/mnt/models/teacache_runs/compare_png.py`, `compare_npy.py`; visuals `side_baseline_cadence2.png`, `side_wan_frame4.png`.

## Conclusions

1. **The wiring works on device, both models, both modes, both paths (bench + `difflet serve`).** Skip counts are exactly the plan's 5 of 20; per-full-step latency and peak HBM are unchanged (no recompiles, no scalar leaks); outputs are finite and deterministic.
2. **Speed: cadence 2 = 0.75× denoise on both models** (Qwen 10.12 → 7.60 s, Wan 12.18 → 9.10 s), matching the prediction to 1 %.
3. **Online-delta is the wrong default for Qwen on TPU.** Same 5 skips, but its per-full-step `float()` sync destroys the lazy-XLA overlap: natural-basis e2e 10.1–10.3 s vs 8.7 s *without* TeaCache. On Wan it costs nothing (UniPC already syncs) but drifts more than cadence 2 at the same skip count. → Recommend `--teacache-cadence 2` for both; treat `--teacache-online-delta` on TPU Qwen as "measured, not recommended" until the sync is made asynchronous.
4. **Quality:** Qwen cadence 2 is within the repo's existing 0.002/px agreement figure. Wan video is at 0.0126/px, visually clean; there is no video threshold on record, so this is reported, not judged.
5. **Five bugs, five commits**, three of them pre-existing on `main` (the TPU Qwen bench and the Wan runner were both unrunnable as documented; the bench parent hung for an hour on worker death).

## Follow-ups (not done here)

- Online-delta without a sync on the Qwen TPU loop (decide from a one-step-stale delta, or compare on device and read back a bool).
- Expose `warmup/cooldown` as serving flags (5/5 → 3/3 would skip 7 of 20).
- A unit test for `benchmark/adapters/tpu.py`'s worker call sequence (bug 1 would have been caught).
- Decide a video-quality acceptance figure for Wan TeaCache.
- Test-isolation nit: `tests/unit/pipeline/test_pipeline.py` assumes the ambient backend is trainium; in a venv with torch_xla it auto-detects tpu and 12 tests fail.

## Commits on `tpu-teacache` from this session

| sha | subject |
|---|---|
| `f9689ec` | fix(bench): TPU Qwen adapter called `_decode` without the request |
| `4ce1dff` | fix(bench): the TeaCache stats block never reached the results JSON |
| `2bf93f3` | fix(bench): `wan_tpu_run.py` workers could not import `benchmark.harness` when run as documented |
| `f3e2021` | fix(bench): TPU adapter hung for an hour after its workers had died |
| `dc215f5` | fix(qwen/tpu): TeaCache stats were logged at INFO from a worker that has no log handler |

Unit suites after all five (TPU venv, `DIFFLET_BACKEND=trainium`): `tests/unit/serving` + Wan TPU application tests → 560 passed, 2 failed (`test_video_storage.py` path-validation tests — reproduce on a clean `main` worktree, unrelated).

Artifacts: `/mnt/models/teacache_runs/` (logs, PNGs, mp4/npy, JSON), `benchmark/v5e-teacache-{baseline,cadence2,online-delta}/` (bench JSON+MD; not committed, `benchmark/v5e/` untouched).

## Results

| model | mode | denoise warm (s) | step, full steps (ms) | skipped / full | first-request extra compile (s) | quality vs. baseline |
|---|---|---|---|---|---|---|
| Qwen-Image | baseline | 10.12 (warm e2e 13.19) | 503 synced / 286 natural | 0 / 20 | — (cold 80.6 total) | — |
| Qwen-Image | cadence 2 | **7.60** (warm e2e 10.81) | 501 synced / 295 natural | 5 / 15 | ≲ 0.3 (11.04 vs 10.55 first warm) | mean abs 0.0019/px, PSNR 47.3 dB |
| Qwen-Image | online-delta 0.6 | 7.62 synced (warm e2e 10.88) — **natural e2e 10.1–10.3, slower than baseline 8.72** | 500 synced / 485 natural (sync every full step) | 5 / 15 | ≲ 0.4 | mean abs 0.0027/px, PSNR 44.3 dB |
| Wan 2.2 | baseline | 12.18 wall (=denoise+0.68 encode) | 609 | 0 / 20 | — (cold 54.2) | — |
| Wan 2.2 | cadence 2 | **9.10** wall | 606 | 5 / 15 | none visible | mean abs 0.0126/px, PSNR 31.3 dB (video) |
| Wan 2.2 | online-delta 0.6 | 9.11 wall (no natural penalty) | 608 | 5 / 15 | none visible | mean abs 0.0194/px, PSNR 29.1 dB (video) |

## Bugs found

| # | where | symptom | cause | fix |
|---|---|---|---|---|
| 1 | `benchmark/adapters/tpu.py` | TPU Qwen bench dies after denoise: `_decode() missing 1 required positional argument: 'request'` | `f31e433` changed the orchestrator's `_decode` signature, bench worker not updated; no test covers the bench adapter | `f9689ec` |
| 2 | `benchmark/bench.py`, `harness.py`, `report.py` | results JSON/MD lack the `teacache` block README promises | adapter returned it, `BenchResult` had no field, `bench.py` never copied it | `4ce1dff` |
| 3 | `benchmark/wan_tpu_run.py` | `python benchmark/wan_tpu_run.py` (documented form) → every rank `No module named 'benchmark'` | script-by-path puts `benchmark/` not the repo root on `sys.path`; spawn workers inherit it | `2bf93f3` |
| 5 | `difflet/serving/orchestrators/qwen_image.py` | TeaCache stats line never appears in serving logs | `logger.info` from a spawned worker with no handler; no worker INFO line reaches the log | `dc215f5` — `print(..., flush=True)` like the other pipelines |
| 4 | `benchmark/adapters/tpu.py` | parent sits silent for 3600 s after all workers have crashed (seen with bug 1 — had to be killed by hand) | `reply_q.get(timeout=3600)` never checks `Process.is_alive()` | `f3e2021` — `_wait_reply` polls in 5 s slices and raises with the dead ranks' exit codes |

**Hang-proofing for the rest of this session** (the "no response" problem): every run is launched with `nohup … &` and waited on with a `kill -0` loop plus a hard timeout; if a run stops producing log lines, check `ps` for dead workers and kill the parent instead of waiting. `wan_tpu_run.py` already breaks out of its wait when no rank is alive (worst case 60 s); the Qwen adapter now does the same (bug 4).


## Addendum (same day) — HunyuanVideo, after its TPU port

`benchmark/hunyuan_tpu_run.py --steps 20 --iters 2 --natural-iters 1 [--teacache-cadence 2]`, 320×512×61, tp=4:

| mode | denoise (s) | per full step (ms) | skipped / full | quality vs baseline |
|---|---|---|---|---|
| baseline | 20.12 / 20.14 / 20.11 (natural) | 1006 | 0 / 20 | — |
| cadence 2 | **15.07 / 15.08 / 15.19** (0.749×) | 1004 | 5 / 15 | mean abs 0.0064/px, PSNR 36.9 dB (61 frames) |

Same 0.75× as Qwen-Image and Wan; no natural-basis penalty (the orchestrator's Euler step is
host-side, so every step syncs regardless). Online-delta not measured on Hunyuan. Host VAE decode
(167 s) dominates the request either way.

## Addendum (2026-09-12) — FLUX.1-dev, after its TPU port

Port written weight-free (`0389f74`, 16 CPU tests incl. module-vs-diffusers at 1e-5 and
loop-vs-scheduler at 1e-6), proven on the chips with a synthetic sharded checkpoint (cos
0.9999995 in fp32) and the full geometry with random weights (10.18 GB/rank, 0.335 s/forward),
then — the token arrived 04:11 — on the real weights within 20 minutes: parity cos 0.99946 @1024²
(0.99974 vs diffusers' own bf16 @256²), bench **8.7 s to latents / 187 ms per step** (trn2:
268 ms — the first model where the v5e is faster per step), cadence 2 skips 9/28 (DiT 0.69×,
0.0041/px), `difflet serve` ready in 216 s and **200 in 10.1 / 9.3 s**, PNGs bit-identical to
the bench; VAE on the chip 0.31 s warm (116 s first compile). At 1024² the device is 0.99986 from
diffusers' own bf16 while both are 0.99946 from fp32. No device bugs. Commits: `0389f74`
(scaffold + tests), `c7b88ea` (runner, probes), `a73efb0` (row + records). Records:
`benchmark/v5e/flux_1_dev.md`, evidence doc Phase 7, `benchmark/v5e/RESULTS.md`.
