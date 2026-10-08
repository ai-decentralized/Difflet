# Probe-free TeaCache on the TPU backend

Date: 2026-09-08

Status: Implemented; **verified on a v5e 2026-09-11** — see `docs/worklog/2026-09-11-tpu-teacache-device.md`

Branch: `tpu-teacache`, commit `993b56a` (on top of `main` @ `967e38f`)

## Goal

Cut denoise time on the TPU backend (Cloud TPU v5e, eager torch_xla) by letting
the two ported models — Qwen-Image and Wan 2.2 — skip DiT forwards with the
TeaCache controller that already ships in `difflet/pipeline/teacache.py`. No
modeling code changes; the work is wiring in the serving adapters, the TPU
application, and the benchmark runners, plus the option-layer validation that
had been refusing the flags.

## Where TeaCache stood (anchored to `main` @ `967e38f`)

The controller is pipeline-layer, host-side and backend-neutral. It has three
modes:

| mode | flag | needs | decision |
|---|---|---|---|
| adaptive | `--teacache-speedup X --teacache-calibration PATH` | block-0 modulated-input signal (fused probe NEFF on Trainium; CPU shadow for Wan) + per-shape calibration | polynomial on the signal |
| fixed cadence | `--teacache-cadence N` | nothing | skip every N-th step inside `[warmup, steps - cooldown)` |
| online-delta | `--teacache-online-delta ALPHA` | nothing | skip iff the previous full step's output rel-L1 delta < `ALPHA` × the first post-warmup delta; never two in a row |

On Trainium all three are wired for every model (README feature matrix). On TPU
none of them reached the loop:

- **Only two inference paths exist on TPU.** `difflet serve` and the benchmark
  runners. The CLI (`difflet generate/run`) goes through `DiffletPipeline`, and
  `TpuBackend.prepare_runtime` raises `NotImplementedError` there.
- **Qwen-Image serving** (`difflet/serving/orchestrators/qwen_image.py`,
  `_denoise_tpu`) is a hand-written device-resident flow-matching Euler loop —
  written that way so every per-step scalar is a device tensor and XLA compiles
  one graph for the whole loop. It never consulted a controller.
- **Wan serving** (`difflet/models/wan/tpu_application.py`, `load_eager`) built
  the backend-neutral `WanOrchestrator` without the TeaCache kwargs that
  `NeuronWanApplication` forwards, so `--teacache-cadence` would have produced
  an orchestrator with no controller at all.
- **Option layer** (`difflet/serving/options.py`, `build_serving_profile`)
  rejected `--teacache-cadence` / `--teacache-online-delta` for *every* model
  ("resident serving does not yet implement …"), so `difflet serve` could not
  reach either adapter with them.
- **Adaptive is not available on TPU.** Its signal is produced by a fused probe
  NEFF (Qwen, Flux, HunyuanVideo) or a CPU shadow that lives under
  `difflet/backends/trainium/` (Wan). Neither exists for TPU, and the gate that
  chose the method per model (`difflet/pipeline/teacache_gate.py`) already
  preferred online-delta over adaptive for Qwen (signal Pearson 0.30 vs. delta
  autocorrelation 0.93).

## What changed (commit `993b56a`)

| layer | file | change |
|---|---|---|
| profile | `difflet/serving/types.py` | `ServingProfile.teacache_cadence` / `.teacache_online_delta` |
| options | `difflet/serving/options.py` | `_validate_probe_free_teacache`: mutually exclusive with each other and with `--teacache-speedup`; cadence an `int >= 2`; alpha finite and `> 0`; accepted for `qwen_image` and `wan` only (`_PROBE_FREE_TEACACHE_SERVING_MODELS`) |
| Qwen adapter | `difflet/serving/orchestrators/qwen_image.py` | `_tpu_denoise_loop` (the Euler loop, now with the controller); `_build_tpu_teacache`; Trainium path forwards the kwargs to `NeuronQwenImageApplication` and passes `teacache_enabled=None` for probe-free profiles |
| Wan adapter | `difflet/serving/models/wan.py` | `_teacache_kwargs(profile)` reaches both backends' applications; kept out of `_application_kwargs` (compile-cache identity); `_validate_profile` still rejects adaptive |
| Wan TPU app | `difflet/models/wan/tpu_application.py` | `build_wan_orchestrator` split out of `load_eager`; forwards `teacache_cadence` / `teacache_online_delta_alpha` |
| rejections | `difflet/serving/models/hunyuan_video.py`, `ltx_2.py` | `_validate_profile` rejects the probe-free fields too (belt and braces behind the option layer) |
| CLI help | `difflet/cli/main.py` | `serve` flag help no longer says "not supported by serving" |
| benchmark | `benchmark/adapters/tpu.py`, `benchmark/wan_tpu_run.py` | `DIFFLET_BENCH_TEACACHE_CADENCE` / `DIFFLET_BENCH_TEACACHE_ONLINE_DELTA`; `--teacache-cadence` / `--teacache-online-delta`; results carry a `teacache` block |
| tests | `tests/unit/serving/…`, `tests/unit/models/wan/test_wan_tpu_application.py` | see "Verification so far" |
| docs | `DEVELOPER.md`, `README.md`, `benchmark/README.md` | serving + TPU notes, A/B knobs |

### Design notes

**Why only the probe-free modes on TPU.** See above: no probe, no shadow. The
option layer never lets an adaptive profile reach the TPU adapter, so this is
an explicit "not implemented", not a silent downgrade.

**XLA behaviour of the Qwen loop** (`_tpu_denoise_loop`). The controller's
`prev_noise_pred` and `cached_residual` are device tensors: `record_full_step`
computes `noise_pred - prev` lazily on the chip and `skip_noise_pred` is one
lazy add, so a skipped step costs an elementwise op instead of a 20B forward.
Over a request XLA sees three graph shapes — step 0 (no residual yet), a full
step (forward + residual + Euler update), a skipped step (add + Euler update) —
and caches each after its first compile. The velocity is recorded in fp32 (the
dtype the Euler update already casts to), so the residual is not bf16-quantized
on the way through.

- *Fixed cadence* forces no device sync. The loop keeps the tracing/execution
  overlap that the "natural" basis in `benchmark/v5e/RESULTS.md` measures.
- *Online-delta* calls `float(...)` on one scalar in `record_full_step`, which
  is a device sync on every **full** step. On this loop that is the synced
  basis; skipped steps still pay nothing. (Wan's orchestrator already ends
  every step in `.cpu()` for UniPC, so neither mode adds a sync there.)

**Compile-cache identity.** Neither mode changes the compiled graph. On Wan the
kwargs are added in `_build_application`, not in `_application_kwargs`, which
`CacheSpec.application_kwargs` hashes — toggling a cadence on Trainium must not
recompile. Pinned by `test_teacache_kwargs_stay_out_of_the_compile_identity`.

**Trainium side effect.** Since the option layer is backend-agnostic, Qwen and
Wan *Trainium* serving now accept the two flags too. For both it is a kwargs
pass-through into pipelines that already implement the modes (verified there
by the CLI matrix). One behavioural detail: Qwen's adaptive gate
(`request_uses_teacache`) emits `teacache_enabled=False` on a step mismatch;
for probe-free profiles the adapter now passes `None`, because `False` would
silently disable the cadence the operator asked for.

**Skip window.** `[warmup_steps, num_steps - cooldown_steps)` with the
controller's 5/5 defaults. 20 steps at cadence 2 → steps 6, 8, 10, 12, 14
skipped (5 of 20). A 4-step startup smoke skips nothing.

## Usage

```bash
# serving (Qwen-Image or Wan), TPU
DIFFLET_BACKEND=tpu difflet serve --model-id Qwen/Qwen-Image --teacache-cadence 2
DIFFLET_BACKEND=tpu difflet serve --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --teacache-online-delta 0.6

# benchmark A/B (baseline = unset)
DIFFLET_BENCH_TEACACHE_CADENCE=2 DIFFLET_BACKEND=tpu \
    python -m benchmark.bench --backend tpu --model qwen_image --iters 3
DIFFLET_BACKEND=tpu python benchmark/wan_tpu_run.py --steps 20 --iters 3 --teacache-cadence 2
```

Both runners report the controller's `{full_steps, skipped_steps, …}` as a
`teacache` block beside the timings. `step_seconds` covers full steps only (a
skipped step never calls the DiT, so the timer never ticks), so read the saving
off `denoise_seconds` / `wall_seconds`.

## Verification so far (CPU only)

Unit tests, run in a venv without the Neuron toolchain:

- `test_tpu_denoise_loop_fixed_cadence_skips_the_dit_and_extrapolates` — exact
  skip indices `{6, 8, 10, 12, 14}` at 20 steps, 15 DiT calls, one `mark_step`
  per step, output equal to a hand-rolled reference of the residual
  extrapolation (bf16-quantized velocities, fp32 arithmetic).
- `test_tpu_denoise_loop_online_delta_skips_only_flat_steps` — skips only after
  the trajectory flattens, never in warmup/cooldown, never two in a row.
- `test_tpu_denoise_loop_resets_the_controller_between_requests`.
- `test_orchestrator_receives_fixed_cadence` / `…_online_delta` /
  `…_ignores_adaptive_calibration_on_tpu` (Wan TPU application).
- `test_build_application_forwards_probe_free_teacache_on_tpu` / `…_on_neuron`,
  `test_teacache_kwargs_stay_out_of_the_compile_identity` (Wan adapter).
- `test_serving_profile_carries_probe_free_teacache_modes`,
  `…_rejects_probe_free_teacache_for_unwired_adapters`,
  `…_rejects_invalid_probe_free_teacache` (option layer).

870 passed across `tests/unit/serving`, `tests/unit/pipeline`, the CLI main
tests and the TPU backend tests. 3 failures are pre-existing (missing
`neuronx_distributed` in the local venv; reproduced on a clean `main`
worktree).

## On-device verification plan

Prerequisites (from `benchmark/v5e/wan_2_2.md`): the Python 3.12 venv with
`torch_xla 2.9` + `jax[tpu] 0.7.1` (fused attention), `DIFFLET_BACKEND=tpu`,
weights under `/mnt/models`, chips free (`tpu-info` lists holders of
`/dev/vfio/*`).

0. **Unit tests in the TPU venv** (no chip):
   `python -m pytest tests/unit/serving/test_qwen_common_orchestrator.py tests/unit/models/wan/test_wan_tpu_application.py -q`
1. **Qwen-Image A/B** — baseline, then cadence 2, then online-delta 0.6, same
   `MATRIX` row (1024², 20 steps, tp=4, seed 42). `benchmark.bench` runs only
   the cold generate by default, so pass `--iters 3` for warm ones. Read the
   *second and later* warm iterations: the first request with TeaCache pays one extra DiT-sized
   XLA compile (the full-step-with-residual graph is a new shape; the skip
   graph is tiny). Save the PNG (`DIFFLET_BENCH_SAVE_PNG=…`) for each run.
2. **Wan 2.2 A/B** — `wan_tpu_run.py --steps 20 --iters 3` baseline vs.
   `--teacache-cadence 2` vs. `--teacache-online-delta 0.6`; compare the later
   iterations and the decoded `.mp4`.
3. **Serving smoke** — `difflet serve … --teacache-cadence 2`, one request
   through `/v1/chat/completions`; the worker log must show
   `[teacache] stats: {'full_steps': 15, 'skipped_steps': 5, …}` (Qwen)
   or `[teacache] stats: {...}` (Wan).
4. **Quality** — same seed, compare against the baseline output: mean absolute
   per-pixel difference (the fused-attention A/B used 0.002/pixel as its
   agreement figure) and a visual check. Cadence 2 is the configuration the
   Trainium CLI matrix passes; the TPU numerics path is the same controller
   over the same modeling, but it has not been looked at on TPU.

### Results (fill in on device)

| model | mode | denoise warm (s) | step, full steps (ms) | skipped / full | first-request extra compile (s) | quality vs. baseline |
|---|---|---|---|---|---|---|
| Qwen-Image | baseline | 10.14 (`benchmark/v5e`) | 508 synced / 291 natural | 0 / 20 | — | — |
| Qwen-Image | cadence 2 | **7.60** | 501 synced / 295 natural | 5 / 15 | ≲ 0.3 | mean abs 0.0019/px, PSNR 47.3 dB |
| Qwen-Image | online-delta 0.6 | 7.62 synced; **natural e2e 10.1–10.3 s, slower than baseline (8.7)** | 500 synced / 485 natural | 5 / 15 | ≲ 0.4 | 0.0027/px, 44.3 dB |
| Wan 2.2 | baseline | 12.20 (`benchmark/v5e`) | 610 | 0 / 20 | — | — |
| Wan 2.2 | cadence 2 | **9.10** wall | 606 | 5 / 15 | none | 0.0126/px, 31.3 dB (video) |
| Wan 2.2 | online-delta 0.6 | 9.11 wall | 608 | 5 / 15 | none | 0.0194/px, 29.1 dB (video) |

**Expected, unmeasured:** cadence 2 at 20 steps removes 5 of 20 forwards, so
denoise ≈ 0.75× baseline — Qwen ≈ 7.6 s, Wan ≈ 9.2 s — with the per-full-step
figure unchanged. Online-delta's count depends on the trajectory; on Trainium
the gate preferred it for Qwen. `ALPHA = 0.6` is the gate's default, not a
TPU-tuned value.

### Triage

| symptom | first thing to check |
|---|---|
| `skipped_steps == 0` | controller never attached (`_tpu_teacache is None` — profile fields missing?) or `num_steps` not synced (cooldown guard fires on every step) |
| per-full-step slower than baseline | recompiles: `PT_XLA_DEBUG=1` / `XLA_IR_DEBUG=1`; a Python scalar leaking into the graph; online-delta's sync is expected, cadence's is not |
| `Attempting to allocate … free` | the residual + prev tensors are two extra `[1, 4096, 64]` fp32 buffers (~2 MB); not plausible at 1024², but the headroom after the weight shard was measured at ~6 GiB |
| NaN / drift in output | compare against the CPU reference in `_reference` (test file); the residual is fp32 by construction |

## Not verified / known gaps

- Nothing has run on a chip yet. The Trainium serving wiring has not run on
  Trainium either (kwargs pass-through only).
- `warmup_steps` / `cooldown_steps` are the controller's 5/5 defaults and not
  exposed as serving flags. With 20 steps that leaves a 10-step window; 3/3
  would skip 7 instead of 5 at cadence 2. A flag is a small follow-up if the
  measured quality allows it.
- Adaptive TeaCache stays Trainium-only.
- The benchmark's per-step timer measures full steps only (documented in
  `benchmark/README.md`).

## Other TPU optimizations worth considering

Ranked by expected value; evidence from `benchmark/v5e/` and the handoff in
`docs/plans/2026-08-16-tpu-backend-support.md` (branch `tpu`).

| item | evidence | expected |
|---|---|---|
| Wan VAE decode on the chip | host decode 26.8–42.7 s vs. 12.2 s denoise (`wan_2_2.md`); blocked by `AutoencoderKLWan` raising an unsupported negative index under torch_xla, not by HBM (7.6 GB free) | ~2× on Wan e2e |
| Batched CFG for Wan (`batch=2`, one forward) | guidance > 1 runs cond + uncond as two forwards per step; the benchmark used guidance 1.0 so this is invisible in the tables; `WanOrchestrator._denoise` already has the `torch.cat([latents, latents])` form in its cfg-parallel branch, and tp=2 cfg-parallel does not fit 16 GB | large on real workloads |
| Sequence parallelism | the two row-parallel all-reduces were 15% of a block before attention fusion and are a larger share now; `reduce_scatter_to_sequence_parallel_region` etc. exist in `backends/tpu/ops_impl/collectives.py`; the Qwen TPU config has no `sp` field and the Wan TPU config lists it as unsupported | ~10–15% per step |
| TeaCache window flags | 5/5 warmup/cooldown hard-coded; see above | 25% → 35% skips at cadence 2 |
| Qwen text encoder | host fp32, 2.0 s of a 13.3 s warm e2e (15%); one rank encodes and broadcasts already | 1–2 s |
| Persistent XLA compilation cache | 20–56 s compile per process start; the Phase 0 spike hit `UNIMPLEMENTED: Deserializing serialized executable not supported` on torch_xla 2.9 / libtpu 0.0.21 — that is the PJRT persistent-cache path, worth retrying on a newer libtpu | cold start only |
| Matmul precision audit | `configure_matmul_precision("highest")` is process-wide; the DiT is bf16 so the MXU path is unaffected, but any fp32 matmul left in the hot path runs bf16×6 emulation | unknown; profile first |
| int8 weight-only | v5e int8 MXU peak is ~2× bf16; 9.6 GB → ~5 GB per chip would let both Wan experts co-reside (quality item as much as speed) | long-term |

## Status log

### 2026-09-11 — verified on v5e

Five commits `f9689ec`..`dc215f5` on `tpu-teacache`. Both models, both modes,
bench + `difflet serve`: 5 of 20 steps skipped, denoise 0.75× baseline,
per-full-step and HBM unchanged, outputs deterministic. Online-delta is a
net loss on the Qwen TPU loop in the natural basis (per-step sync) — use
cadence. Full log: `docs/worklog/2026-09-11-tpu-teacache-device.md`.

### 2026-09-08 — implemented, CPU-verified

Commit `993b56a` on `tpu-teacache`. 20 files. Unit tests green (see above).
Device run pending; results table above is empty on purpose.
