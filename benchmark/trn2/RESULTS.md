# Benchmark results — summary

> **2026-09-12 update.** The table directly below is the 2026-06/07 tp4 run, kept as history. A new **parallel-topology campaign** (main @ 38e863e, fresh toolchain, all five models re-measured with one per-step method, plus tp2cp2 / tp4sp / tp2cfg and a native NxDI FLUX baseline) is at the [bottom of this file](#2026-09-12-parallel-topology-campaign-main--38e863e--campaign-branch); its per-model files are `<slug>.json` (tp4, overwritten by the new run) and `<slug>_<config>.json`.

Measured on **trn2.3xlarge** (1 Neuron device, 4 NeuronCores × 24 GB), bf16,
`tp=4`, via the Neuron inference venv
(`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`), with **per-rank presharding
and jemalloc default-on** (2026-06-30/07-01 — see the Presharding + jemalloc notes below). The device is serial, so
every run had it to itself (an early contended run skewed badly — LTX-2 970 s vs clean).
**e2e cold** is a *true* cold start: the OS page cache is dropped
(`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before the run, so the
weight load is a real cold disk read. **e2e warm** is the very next run, with the
weights now in the page cache. Each row links to a detailed per-model report.

| model | kind | shape | compile¹ | **e2e cold**² | **e2e warm**³ | load cold→warm⁴ | **DiT per-step**⁰ | output | status |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| [LTX-2](ltx_2.md) | video+audio | 480×704×49 | 30.6 min | **778 s** (13.0 min) | **58 s** | 315→15 s⁸ | **441.8 ms** (2.26/s)ᵇᵉ | (1,49,3,480,704) ✓ | ok |
| [Wan 2.1 14B](wan_2_1.md) | video (T2V) | 480×832×9 | 131 min⁵ | **394 s** (6.6 min) | **56 s** | 343→31 s | **554.8 ms** (1.80/s)ᵈ | (1,3,9,480,832) ✓ | ok |
| [Wan 2.2 A14B](wan_2_2.md) | video (T2V) | 480×832×9 | (shares 2.1)⁶ | **394 s** (6.6 min) | **57 s** | 342→30 s | **554.8 ms** (1.80/s)ᵈ | (1,3,9,480,832) ✓ | ok⁶ |
| [Qwen-Image](qwen_image.md) | image (T2I) | 1024×1024 | 21.9 min | **509 s** (8.5 min) | **63 s** | 453→32 s | **447 ms** (2.24/s) | (1,3,1024,1024) ✓ | ok |
| [HunyuanVideo](hunyuan_video.md) | video (T2V) | 320×512×61 | 47.1 min | **667 s** (11.1 min) | **144 s** | 513→49 s | **850.6 ms** (1.18/s)ᶜ | (1,3,61,320,512) ✓ | ok |
| [FLUX.1-dev](flux_1_dev.md) | image (T2I) | 1024×1024 | 24.7 min⁹ | **321 s** (5.3 min) | **35 s** | 280→20 s | **267.6 ms** (3.74/s)ᵇ | 1024² PNG ✓ | ok |
| [HunyuanVideo-1.5](hunyuan_video_15.md) | video (T2V) | 480×848×121 | — | — | — | — | — | — | pending⁷ |

✓ = output is finite (no NaN/Inf) with a sensible value range — see each report.

## Presharding (per-rank weight cache) — default-on, 2026-06-30

`save_sharded_checkpoint` now defaults to **True** (`config.py`): the per-rank shards
(`weights/tp{rank}_sharded_checkpoint.safetensors`) are written **once at compile** and
read directly at load, removing the per-run load-time reshard. The load path is now
**guarded** (`application_base.py`): if a component's shards are missing it logs a warning
and falls back to shard-on-load instead of raw-crashing with `FileNotFoundError` — so the
old footgun (presharding enabled but shards absent) is closed, and components that don't
go through the standard shard path (e.g. the Qwen-Image LLM text-encoder) take the
fallback cleanly. **Lossless**: output is bit-identical to load-time sharding (verified on
FLUX, md5-equal across cold/warm × on/off).

The **warm e2e in the top table** now also includes **jemalloc** (see the jemalloc note below);
the OFF→ON table here **isolates presharding alone** (pre-jemalloc) vs the load-time-shard
baseline on trn2.3xlarge (warm is the reliable metric; cold is disk/page-cache-noisy):

| model | warm e2e OFF→ON | Δ warm | warm load OFF→ON |
|---|---|---:|---|
| LTX-2 | 103 → 62 s | **−40%** | 52 → 15 s |
| Wan 2.1 14B | 97 → 60 s | **−38%** | 54 → 31 s |
| Wan 2.2 A14B | 93 → 59 s | **−36%** | 51 → 30 s |
| HunyuanVideo | 220 → 178 s | **−19%** | 37 → 49 s* |
| FLUX.1-dev | 47 → 38 s | **−19%** | 28 → 20 s |
| Qwen-Image | 74 → 65 s | **−12%** | 41 → 32 s |

Large DiTs gain most — presharding eliminates the full load-time shard of the transformer
(~20–40 s). **`step_latency` is unchanged** (presharding moves only weight load, not DiT
compute), so the per-step column above is identical to the prior run. Per-rank shards add
to the one-time compile (the transformer shard-write); the compile column includes that
cost. *HunyuanVideo's warm-load delta is not a clean comparison — this run compiles the
VAE on-chip whereas the prior baseline decoded VAE on the host, so the load profiles
differ.

> Caveat: parts of this run's compiles and cold reads overlapped large model downloads,
> so absolute **cold** e2e and **compile** minutes carry some disk-contention noise. The
> **warm** e2e and per-step latency are unaffected (page-cache / pure compute).

## jemalloc allocator — default-on, 2026-07-01

On top of presharding, difflet now preloads jemalloc (`libjemalloc.so`, bundled in
torch_neuronx) via a one-time re-exec for `generate`/`run` — `cli/main.py`
`_ensure_jemalloc`. The parallel per-rank weight load (`_parallel_load`, one thread per
rank) is malloc/page-fault heavy; glibc's shared arena/mmap-lock serializes the threads,
jemalloc's per-thread arenas remove that contention. **~17% faster load, ~2–5 s off warm
e2e, bit-identical.** Gated to generate/run (compiling under jemalloc crashes the
neuronx-cc worker); opt out with `DIFFLET_NO_JEMALLOC=1`.

Clean LTX-2 A/B (warm, median n=5, 2 warmups): **OFF 63.4 s → ON 58.4 s (−5.0 s)**, and
available RAM after was equal/higher with jemalloc (no page-cache penalty). LTX/Hunyuan
gain the most because they are **host-staged** (text-encoder/VAE on CPU), so jemalloc also
speeds the host allocation, not just the Neuron load.

> ⚠️ **Big-model warm caveat**: LTX-2 (86 GB weights) and HunyuanVideo barely fit this
> box's ~100 GB page cache, so their warm e2e is only stable after ≥2 cache-warming
> runs (a single under-warmed run reads cold and inflates the number — e.g. an LTX-2
> mean of 186 s vs a true 63 s). The values above are medians of properly-warmed runs.

## Corrections (2026-06-27)

This run corrects measurement-method and wiring errors in the earlier numbers. The
old values were not flagged as wrong anywhere — they were recorded as if correct — so
they are kept here with the reason they changed (not silently overwritten).

- **ᵇ Per-step is now measured the same way as H100** (`benchmark/step_realloop.py`):
  inter-step deltas of a *real* generate loop (device-synced, step 0 excluded) — the
  same quantity H100's `callback_on_step_end` measures. The earlier trn2 per-step used
  **three different methods** across models (n=20 isolated synthetic-input transformer
  forward for Wan/Qwen/Hunyuan; an n=1 parity script for LTX-2; the tqdm denoise-loop
  rate for FLUX), so the old cross-device table was not apples-to-apples. Re-measured:
  **FLUX 266→267.6 ms** (n=27) and **LTX-2 473→477 ms** (n=19) — i.e. the old FLUX/LTX
  numbers were already about right; only the *method* was inconsistent. **Qwen and Wan
  are NOT yet re-measured this way** (still the old isolated-timer numbers).
- **ᶜ HunyuanVideo per-step 3719 → 850.6 ms (4.37× faster) — it was running on SDPA,
  not attention_cte.** Its joint self-attn carries a text key-padding mask; difflet's
  masked path fell back to `F.scaled_dot_product_attention` after commit `cd54d0f`
  dropped the in-graph mask→bounds auto-route. The `config_label` said "attention_cte"
  but the compiled graph used SDPA. Re-wired the contiguous key-padding mask to
  attention_cte's `bound_min`/`bound_max` (trace-safe sum in `dual_stream_attention`,
  CPU-validated lossless cosine 1.0). 850.6 ms is measured with the existing
  step_latency method (same as the old 3719) using a synthetic all-ones mask
  (`bound_max`=full seq) → a *conservative* upper bound; a real padded prompt attends
  fewer keys. The HunyuanVideo **compile (41.5 min) and e2e (551/220 s) rows are stale**:
  this run also compiles the **VAE on-chip** (`unet-inference` NEFF) whereas the old e2e
  decoded VAE on the host, so e2e must be re-measured before it is trusted.
- **ᵈ Wan per-step 1144 → 554.8 ms (2.06×) — its attention was *replicated* across the 4
  TP cores, now head-sharded.** Wan's `qk_norm="rms_norm_across_heads"` runs over the full
  inner_dim, so difflet had gathered Q/K/V to every rank (`gather_output=True`) and run the
  attention replicated (`tp=4` parallelized only the FFN). Re-wired to head-sharded
  attention (`gather_output=False` + `RowParallelLinear` output + local heads) with a
  TP-aware global RMS (cross-rank sum-of-squares + per-rank norm-weight slice, the pattern
  difflet's LTX-2 `_global_rms_norm` already uses; HunyuanVideo avoids it because its
  `qk_norm` is per-head and shards for free). Strict parity vs the replicated baseline on
  the same input: **cosine 0.999768** (rel_l2 2.0e-2, bf16). trn2 now **matches H100**
  (554.8 vs 554.2 ms, ≈par) — was 2.0× behind. Both Wan 2.1 and the single-expert Wan 2.2 use this
  per-step (only Wan 2.1 was independently measured).
- **ᵉ LTX-2 text cross-attn 477 → 441.8 ms (7%) — moved from masked SDPA to unmasked
  attention_cte.** LTX-2's self-attn was already on attention_cte; only its two text
  cross-attns (video←text, audio←text, text_seq_len=1024) fell to SDPA because they carry a
  key-padding mask and attention_cte's bound path is self-attn only (q==kv; cross-attn q!=kv
  fails neuronx-cc NCC_IBIR243). Dropping the mask and running unmasked attention_cte (q!=kv
  is supported unmasked, like wan/qwen) is lossless here — **full-generate parity vs the
  masked baseline, same seed, cosine 0.999934** — so the text padding is benign. The gain is
  small: cross-attn was only ~7% of the per-step, so the residual H100 1.41× lead is genuine
  self-attn(cte)+FFN compute, like Qwen. (Caveat: padding-benign is an empirical property of
  the text encoder, validated on the benchmark prompt.)

**The headline finding: e2e is load-dominated, not compute-bound.** For the
pure-Neuron pipelines warm is **5–9× faster** than cold (Qwen 509→63 s, Wan
394→56 s, FLUX 321→35 s); HunyuanVideo is the exception at 4.6× (667→144 s) because
its host VAE decode is not weight-load and doesn't speed up with a warm
cache. The cold→warm gap is otherwise the one-time cold disk read of the weights —
the Neuron denoise compute is small (per-step × steps). So the two metrics that actually characterize
the hardware are the **cold weight-load** and the **DiT per-step**; absolute cold
e2e mostly measures disk + page-cache state. FLUX.1-dev is the fastest warm e2e
(47 s) and fastest per-step (267.6 ms) of the set.

⁰ **DiT per-step** = warm in-process transformer-forward latency (median n=20, p90
within 1 ms of median for every model), the stable pure-Neuron-compute metric, from
`benchmark/step_latency.py`. Multiply by `steps` for the Neuron denoise-loop floor
(e.g. Qwen 0.447 s × 20 ≈ 9 s of the 74 s warm e2e; the rest is load + host
text-encode/VAE decode).

¹ One-time AOT compile (cached afterwards; reused by every run via `--cache-dir`, no
recompile). ² **e2e cold** = true cold start (page cache dropped first) → real cold
disk read; this replaces earlier values that were accidentally taken with the host
weights already cached (artificially low, e.g. the old LTX-2 "293 s" had an 8 s host
load vs 372 s cold). ³ **e2e warm** = the immediately-following run, same session,
weights served from the OS page cache. ⁴ **load cold→warm** = total Neuron weight
load (sum over all pipeline stages) on the cold vs warm run — this delta is the bulk
of the cold→warm speedup. ⁵ Wan 2.1 compile is dominated by the video VAE decoder
(~100 min on neuronx-cc; the transformer is 431 s). ⁶ The wan orchestrator keys its
compile cache by *shape*, not model id, so Wan 2.2 reused Wan 2.1's NEFF; difflet
also runs Wan 2.2 with only the high-noise expert (`enable_transformer_2=False`),
i.e. single-transformer — see report caveats. ⁷ HunyuanVideo-1.5 orchestrator is a
stub (`NotImplementedError`). (FLUX.1-dev is gated; it was benchmarked here after
authenticating with an HF token.) ⁸ host stages: LTX-2 runs its text-encoder + VAE
on the host (`enable_host_pipeline`) and HunyuanVideo decodes VAE on the host, so
that work is **not** in the Neuron load total — it sits in the compute residual.
That is why HunyuanVideo's warm e2e is still 220 s (its host VAE decode of a
61-frame video is ~185 s and does not benefit from a warm weight cache), while the
pure-Neuron pipelines drop to ~47–100 s warm.

¹⁰ HunyuanVideo's 40 min is the full `difflet compile` wall; its neuronx-cc build
sub-phase is only ~11 min (652 s). The ~29 min difference is the one-time host load
+ HLO trace + weight shard/save of the Llama-8B encoder + 13B DiT (the largest such
overhead in the suite) — not directly comparable to the other models' compile, whose
host load was small. Compile is one-time/cached regardless.

⁹ FLUX compile (21.2 min) is dominated by the VAE decoder (~1078 s of 1257 s on
neuronx-cc; CLIP 21 s, T5 7 s, transformer only 151 s) — like Wan, the image/video
decoder is the expensive AOT artifact, not the transformer. ᵃ FLUX per-step is the
**warm denoise-loop rate** (3.76 it/s ⇒ 266 ms/step, 28 steps in ~7 s) read from the
generate log, not the in-process timer used for the others: flux's compiled-graph
input order differs from its wrapper `forward()` signature, so the generic isolated
timer doesn't fit it; the denoise-loop rate is the equivalent warm per-step.

## Notes on "optimal end-to-end"

- The **best-performing configs currently available** on a single trn2.3xlarge:
  `tp=4` (the max with 4 cores), bf16, attention_cte self-attn where the model routes
  through it. On a trn2.48xlarge, higher `tp` (e.g. FLUX `tp=8`) and context-parallel
  would lift these further.
- **Optimal warm e2e** (weights cached) is the realistic steady-state for a served
  deployment that keeps weights hot: 38–65 s for the image/short-video pipelines
  (FLUX 35 s, Wan 56–57 s, Qwen 63 s, LTX-2 58 s), 144 s for HunyuanVideo — all with
  presharding default-on (weights cached, no load-time reshard). The **Neuron per-step**
  is the lossless compute floor (presharding-independent; corrected
  values, matching the table above): FLUX 267.6 ms, Qwen 447 ms, LTX-2 441.8 ms
  (validated cosine 0.99992 vs CPU), Wan 554.8 ms, HunyuanVideo 850.6 ms.

## Reproduce

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
# one-time compile (cached afterwards):
python -m benchmark.bench --model <slug>            # e.g. ltx_2, qwen_image, wan_2_1
# per-step (warm, in-process):
python -m benchmark.step_latency --model <slug>
# true cold + warm e2e (drops the page cache before the cold run):
python -m benchmark.cold_warm_e2e --model <slug>
```

<!-- campaign:begin -->
## 2026-09-12 parallel-topology campaign (main @ 38e863e + campaign branch)

Same host class as above (**trn2.3xlarge**, 4 NeuronCores under LNC=2, 96 GB HBM, 124 GB RAM), fresh venv from `requirements-neuron.lock`; toolchain `torch=2.9.1`, `torch-neuronx=2.9.0.2.15.32035+de43f57c`, `neuronx-cc=2.26.6360.0+6f180f47`, `neuronx-distributed=0.19.28492+435aae2b`, `diffusers=0.38.0`. Every cell is `compile` (one-time AOT, timed) → **true cold** e2e (page cache dropped) → **warm** e2e (the next process) → **DiT per-step** by the real-loop rule (`benchmark/step_realloop.py`: inter-step deltas of one real generate, device-synced, step 0 excluded — now the same method for all five models). Features ran in the order tp4 → tp2cp2 → tp4sp → tp2cfg, all models per feature, device otherwise idle. Files: `<slug>.json` (tp4) and `<slug>_<config>.json` per cell; run any cell with `python -m benchmark.{bench,cold_warm_e2e,step_realloop} --model <slug> --config <label>`.

### Feature: tp4 — tensor parallel over all 4 cores (the baseline topology)

| model | shape / steps | compile¹ | **e2e cold**² | **e2e warm**³ | load cold→warm⁶ | **DiT per-step**⁰ | outputs/hr (warm)⁴ | cost / 1k⁵ | guidance | output | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| [FLUX.1-dev](flux_1_dev.md) | 1024×1024 / 28 | 21.2 min | **308 s** | **41 s** | 269→23 s | **270.7 ms (n=27)** | 87 | $10.41 | 3.5 | ✓ finite | ok |
| [Qwen-Image](qwen_image.md) | 1024×1024 / 20 | 24.7 min | **494 s** | **65 s** | 446→35 s | **417.3 ms (n=19)** | 55 | $16.52 | 4.0 | ✓ finite | ok |
| [LTX-2](ltx_2.md) | 480×704×49 / 20 | 18.0 min | **769 s** | **56 s** | 310→11 s | **459.2 ms (n=19)** | 64 | $14.21 | 1.0 | ✓ finite | ok |
| [HunyuanVideo](hunyuan_video.md) | 320×512×61 / 20 | 82.3 min | **594 s** | **115 s** | 519→58 s | **814.1 ms (n=19)** | 31 | $28.98 | 6.0 | ✓ finite | ok |
| [Wan 2.1 14B](wan_2_1.md) | 480×832×9 / 20 | 111.6 min | **407 s** | **85 s** | 354→52 s | **575.5 ms (n=19)** | 43 | $21.40 | 1.0 | ✓ finite | ok |

#### FLUX.1-dev tp4: difflet vs the native NxDI baseline (same weights, 1024², 28 steps, seed 42, guidance 3.5, bf16)

| engine | compile | **e2e cold** | **e2e warm** | Neuron load cold→warm | host-side load (warm) | **DiT per-step** | outputs/hr (warm) | output |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| [difflet `flux_1_dev` tp4](flux_1_dev.md) | 21.2 min | **308 s** | **41 s** | 269→23 s | — | **270.7 ms (n=27)** | 87 | ✓ finite |
| [**native NxDI** `generate_flux.py` setup, tp4](flux_1_dev_nxdi.md) | 15.1 min | **330 s** | **57 s** | 290→39 s | 0 s | **271.9 ms (n=27)** | 64 | ✓ finite |

NxDI's `NeuronFluxApplication` loads the full diffusers pipeline on the host in every process (the host-side column, inside its e2e) and runs a warm-up forward per component inside `load()`; difflet loads only the Neuron stages from presharded per-rank checkpoints. The DiT per-step (same attention_cte lineage, same compiler flags) is the like-for-like number; e2e differences are mostly load-path design. Measured by `benchmark/trn2/nxdi_flux_baseline.sh` with the campaign rules (timed compile into a fresh workdir; cold = page cache dropped; one generate per process, no warm-up; real-loop per-step).

### Feature: tp2cp2 — tp=2 × context parallel 2, `--cp-mode ulysses`

| model | shape / steps | compile¹ | **e2e cold**² | **e2e warm**³ | load cold→warm⁶ | **DiT per-step**⁰ | outputs/hr (warm)⁴ | cost / 1k⁵ | guidance | output | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| [FLUX.1-dev](flux_1_dev_tp2cp2.md) | 1024×1024 / 28 | 19.7 min | **491 s** | **44 s** | 452→25 s | **263.1 ms (n=27)** | 82 | $11.05 | 3.5 | ✓ finite | ok |
| [Qwen-Image](qwen_image_tp2cp2.md) | 1024×1024 / 20 | 34.2 min | **820 s** | **71 s** | 771→39 s | **454.3 ms (n=19)** | 51 | $17.88 | 4.0 | ✓ finite | ok |
| LTX-2 | — | — | — | — | — | — | — | — | — | — | **N/A** — LTX-2 has no context-parallel path |
| [HunyuanVideo](hunyuan_video_tp2cp2.md) | 320×512×61 / 20 | 88.8 min | **748 s** | **116 s** | 672→58 s | **849.0 ms (n=19)** | 31 | $29.38 | 6.0 | ✓ finite | ok |
| [Wan 2.1 14B](wan_2_1_tp2cp2.md) | 480×832×9 / 20 | 24.8 min | **623 s** | **87 s** | 570→54 s | **575.5 ms (n=19)** | 41 | $21.99 | 1.0 | ✓ finite | ok |

### Feature: tp4sp — tp=4 + Megatron sequence parallel (`--sp`)

| model | shape / steps | compile¹ | **e2e cold**² | **e2e warm**³ | load cold→warm⁶ | **DiT per-step**⁰ | outputs/hr (warm)⁴ | cost / 1k⁵ | guidance | output | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| [FLUX.1-dev](flux_1_dev_tp4sp.md) | 1024×1024 / 28 | 18.9 min | **307 s** | **40 s** | 269→22 s | **278.8 ms (n=27)** | 89 | $10.21 | 3.5 | ✓ finite | ok |
| [Qwen-Image](qwen_image_tp4sp.md) | 1024×1024 / 20 | 26.4 min | **508 s** | **65 s** | 461→35 s | **365.6 ms (n=19)** | 56 | $16.32 | 4.0 | ✓ finite | ok |
| LTX-2 | — | — | — | — | — | — | — | — | — | — | **N/A** — LTX-2 has no sequence-parallel path |
| [HunyuanVideo](hunyuan_video_tp4sp.md) | 320×512×61 / 20 | 82.7 min | **596 s** | **112 s** | 520→56 s | **790.6 ms (n=19)** | 32 | $28.29 | 6.0 | ✓ finite | ok |
| [Wan 2.1 14B](wan_2_1_tp4sp.md) | 480×832×9 / 20 | 16.1 min | **406 s** | **79 s** | 353→47 s | **578.2 ms (n=19)** | 45 | $20.06 | 1.0 | ✓ finite | ok |

### Feature: tp2cfg — tp=2 × CFG-parallel (uncond/cond branches on separate core pairs, guidance 2.0)

| model | shape / steps | compile¹ | **e2e cold**² | **e2e warm**³ | load cold→warm⁶ | **DiT per-step**⁰ | outputs/hr (warm)⁴ | cost / 1k⁵ | guidance | output | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| FLUX.1-dev | — | — | — | — | — | — | — | — | — | — | **N/A** — guidance-distilled model |
| Qwen-Image | — | — | — | — | — | — | — | — | — | — | **N/A** — guidance-distilled model |
| [LTX-2](ltx_2_tp2cfg.md) | 480×704×49 / 20 | 24.9 min | **1077 s** | **921 s** | 596→447 s | **779.4 ms (n=19)** | 4 | $232.82 | 2.0 | ✓ finite | ok |
| HunyuanVideo | — | — | — | — | — | — | — | — | — | — | **N/A** — guidance-distilled model |
| [Wan 2.1 14B](wan_2_1_tp2cfg.md) | 480×832×9 / 20 | 22.6 min | **640 s** | **102 s** | 577→59 s | **1070.6 ms (n=19)** | 35 | $25.91 | 2.0 | ✓ finite | ok |

### Feature: tp4cfg2 — tp=4 at guidance 2.0 (two sequential CFG branches: the same-work baseline for tp2cfg)

| model | shape / steps | compile¹ | **e2e cold**² | **e2e warm**³ | load cold→warm⁶ | **DiT per-step**⁰ | outputs/hr (warm)⁴ | cost / 1k⁵ | guidance | output | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| FLUX.1-dev | — | — | — | — | — | — | — | — | — | — | **N/A** — guidance-distilled model |
| Qwen-Image | — | — | — | — | — | — | — | — | — | — | **N/A** — guidance-distilled model |
| [LTX-2](ltx_2_tp4cfg2.md) | 480×704×49 / 20 | 7 s | **792 s** | **77 s** | 310→11 s | **918.1 ms (2 calls × 459.1, n=39)** | 47 | $19.49 | 2.0 | ✓ finite | ok |
| HunyuanVideo | — | — | — | — | — | — | — | — | — | — | **N/A** — guidance-distilled model |
| [Wan 2.1 14B](wan_2_1_tp4cfg2.md) | 480×832×9 / 20 | 12 s | **419 s** | **94 s** | 354→50 s | **1151.5 ms (2 calls × 575.8, n=39)** | 38 | $23.73 | 2.0 | ✓ finite | ok |

### Feature: tp4sdpa — tp=4 with `--attention-impl sdpa` (PyTorch SDPA through XLA instead of the attention_cte megakernel routing)

| model | shape / steps | compile¹ | **e2e cold**² | **e2e warm**³ | load cold→warm⁶ | **DiT per-step**⁰ | outputs/hr (warm)⁴ | cost / 1k⁵ | guidance | output | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| [FLUX.1-dev](flux_1_dev_tp4sdpa.md) | 1024×1024 / 28 | 15.8 min | **317 s** | **50 s** | 266→20 s | **682.3 ms (n=27)** | 71 | $12.73 | 3.5 | ✓ finite | ok |
| [Qwen-Image](qwen_image_tp4sdpa.md) | 1024×1024 / 20 | 17.8 min | **503 s** | **69 s** | 447→32 s | **788.8 ms (n=19)** | 52 | $17.47 | 4.0 | ✓ finite | ok |
| [LTX-2](ltx_2_tp4sdpa.md) | 480×704×49 / 20 | 8.5 min | **770 s** | **58 s** | 308→10 s | **506.8 ms (n=19)** | 63 | $14.56 | 1.0 | ✓ finite | ok |
| [HunyuanVideo](hunyuan_video_tp4sdpa.md) | 320×512×61 / 20 | 85.7 min | **651 s** | **168 s** | 516→53 s | **3641.3 ms (n=19)** | 21 | $42.39 | 6.0 | ✓ finite | ok |
| [Wan 2.1 14B](wan_2_1_tp4sdpa.md) | 480×832×9 / 20 | 5.5 min | **416 s** | **88 s** | 352→45 s | **1034.8 ms (n=19)** | 41 | $22.23 | 1.0 | ✓ finite | ok |

### Feature: tp4tc2 — tp=4 + TeaCache fixed cadence 2 (`--teacache-cadence 2`, warmup/cooldown 5 steps, same artifact)

| model | shape / steps | compile¹ | **e2e cold**² | **e2e warm**³ | load cold→warm⁶ | **DiT per-step**⁰ | outputs/hr (warm)⁴ | cost / 1k⁵ | guidance | output | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| [FLUX.1-dev](flux_1_dev_tp4tc2.md) | 1024×1024 / 28 | 21 s | **306 s** | **39 s** | 270→23 s | **270.8 ms (n=18)** (n=18 < 27) | 93 | $9.78 | 3.5 | ✓ finite | ok |
| [Qwen-Image](qwen_image_tp4tc2.md) | 1024×1024 / 20 | 12.9 min | **493 s** | **62 s** | 447→34 s | **417.4 ms (n=14)** (n=14 < 19) | 58 | $15.70 | 4.0 | ✓ finite | ok |
| [LTX-2](ltx_2_tp4tc2.md) | 480×704×49 / 20 | 6 s | **768 s** | **56 s** | 310→12 s | **459.4 ms (n=14)** (n=14 < 19) | 65 | $14.10 | 1.0 | ✓ finite | ok |
| [HunyuanVideo](hunyuan_video_tp4tc2.md) | 320×512×61 / 20 | 17 s | **594 s** | **109 s** | 521→56 s | **814.7 ms (n=14)** (n=14 < 19) | 33 | $27.56 | 6.0 | ✓ finite | ok |
| [Wan 2.1 14B](wan_2_1_tp4tc2.md) | 480×832×9 / 20 | 12 s | **403 s** | **78 s** | 353→48 s | **576.9 ms (n=14)** (n=14 < 19) | 46 | $19.78 | 1.0 | ✓ finite | ok |

### Feature: tp4tcod — tp=4 + TeaCache online-delta adaptive (`--teacache-online-delta 0.6`, calibration-free, same artifact)

| model | shape / steps | compile¹ | **e2e cold**² | **e2e warm**³ | load cold→warm⁶ | **DiT per-step**⁰ | outputs/hr (warm)⁴ | cost / 1k⁵ | guidance | output | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| [FLUX.1-dev](flux_1_dev_tp4tcod.md) | 1024×1024 / 28 | 8 s | **307 s** | **38 s** | 270→22 s | **271.3 ms (n=18)** (n=18 < 27) | 94 | $9.68 | 3.5 | ✓ finite | ok |
| [Qwen-Image](qwen_image_tp4tcod.md) | 1024×1024 / 20 | 19 s | **494 s** | **64 s** | 447→36 s | **417.5 ms (n=14)** (n=14 < 19) | 56 | $16.18 | 4.0 | ✓ finite | ok |
| [LTX-2](ltx_2_tp4tcod.md) | 480×704×49 / 20 | 6 s | **767 s** | **54 s** | 310→10 s | **459.3 ms (n=15)** (n=15 < 19) | 66 | $13.73 | 1.0 | ✓ finite | ok |
| [HunyuanVideo](hunyuan_video_tp4tcod.md) | 320×512×61 / 20 | 17 s | **596 s** | **109 s** | 523→56 s | **814.9 ms (n=14)** (n=14 < 19) | 33 | $27.59 | 6.0 | ✓ finite | ok |
| [Wan 2.1 14B](wan_2_1_tp4tcod.md) | 480×832×9 / 20 | 12 s | **404 s** | **80 s** | 354→49 s | **576.1 ms (n=14)** (n=14 < 19) | 45 | $20.19 | 1.0 | ✓ finite | ok |

### Feature: tp4tcad — tp=4 + TeaCache calibrated adaptive (`--teacache-speedup` at cadence 2's skip budget + per-model `--teacache-calibration`; probe NEFF for FLUX / Qwen-Image / HunyuanVideo, host signal for Wan / LTX-2)

| model | shape / steps | compile¹ | **e2e cold**² | **e2e warm**³ | load cold→warm⁶ | **DiT per-step**⁰ | outputs/hr (warm)⁴ | cost / 1k⁵ | guidance | output | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| [FLUX.1-dev](flux_1_dev_tp4tcad.md) | 1024×1024 / 28 | 7 s | **310 s** | **41 s** | 273→25 s | **274.4 ms (n=18)** (n=18 < 27) | 88 | $10.33 | 3.5 | ✓ finite | ok |
| [Qwen-Image](qwen_image_tp4tcad.md) | 1024×1024 / 20 | 17 s | **508 s** | **68 s** | 460→40 s | **420.5 ms (n=14)** (n=14 < 19) | 53 | $17.24 | 4.0 | ✓ finite | ok |
| [LTX-2](ltx_2_tp4tcad.md) | 480×704×49 / 20 | 6 s | **773 s** | **56 s** | 310→10 s | **511.6 ms (n=14)** (n=14 < 19) | 64 | $14.28 | 1.0 | ✓ finite | ok |
| HunyuanVideo | 320×512×61 / 20 | — | — | — | — | — | — | — | — | — | **BLOCKED** — HBM exhausted: the DiT already fills the core-pair HBM at this shape, so the added TeaCache probe NEFF (calibrated-adaptive only) cannot be resident with it |
| [Wan 2.1 14B](wan_2_1_tp4tcad.md) | 480×832×9 / 20 | 13 s | **417 s** | **83 s** | 354→48 s | **863.2 ms (n=14)** (n=14 < 19) | 43 | $21.08 | 1.0 | ✓ finite | ok |

### TeaCache vs tp4 (same artifact, same seed)

| model | steps | mode | skipped steps (evidence) | warm e2e: tp4 → TC | loop ms/step: tp4 → TC⁷ | DiT call (ms) | output vs tp4 (PSNR)⁸ |
|---|---:|---|---|---:|---:|---:|---|
| FLUX.1-dev | 28 | cadence 2 | **9/28** (stats line) | 41 → **39 s** (1.06×) | 271 → **184** (1.47×) | 270.8 (n=18) | 41.3 dB |
| FLUX.1-dev | 28 | online-δ α=0.6 | **9/28** (stats line) | 41 → **38 s** (1.07×) | 271 → **184** (1.47×) | 271.3 (n=18) | 39.7 dB |
| FLUX.1-dev | 28 | calibrated adaptive (target 1.474×, R² 0.61) | **9/28** (stats line) | 41 → **41 s** (1.01×) | 271 → **186** (1.45×) | 274.4 (n=18) | 37.4 dB |
| Qwen-Image | 20 | cadence 2 | **5/20** (DiT-call count) | 65 → **62 s** (1.05×) | 417 → **353** (1.18×) | 417.4 (n=14) | 45.2 dB |
| Qwen-Image | 20 | online-δ α=0.6 | **5/20** (DiT-call count) | 65 → **64 s** (1.02×) | 417 → **344** (1.21×) | 417.5 (n=14) | 45.0 dB |
| Qwen-Image | 20 | calibrated adaptive (target 1.333×, R² 0.98) | **5/20** (DiT-call count) | 65 → **68 s** (0.96×) | 417 → **322** (1.30×) | 420.5 (n=14) | 45.0 dB |
| LTX-2 | 20 | cadence 2 | **5/20** (stats line) | 56 → **56 s** (1.01×) | 459 → **345** (1.33×) | 459.4 (n=14) | 36.6 dB |
| LTX-2 | 20 | online-δ α=0.6 | **4/20** (stats line) | 56 → **54 s** (1.04×) | 459 → **368** (1.25×) | 459.3 (n=15) | 36.5 dB |
| LTX-2 | 20 | calibrated adaptive (target 1.333×, R² 0.97) | **5/20** (stats line) | 56 → **56 s** (1.00×) | 459 → **381** (1.20×) | 511.6 (n=14) | 36.2 dB |
| HunyuanVideo | 20 | cadence 2 | **5/20** (stats line) | 115 → **109 s** (1.05×) | 814 → **624** (1.30×) | 814.7 (n=14) | 31.8 dB |
| HunyuanVideo | 20 | online-δ α=0.6 | **5/20** (stats line) | 115 → **109 s** (1.05×) | 814 → **623** (1.31×) | 814.9 (n=14) | 24.1 dB |
| HunyuanVideo | 20 | tp4tcad | **BLOCKED** — HBM exhausted: the DiT already fills the core-pair HBM at this shape, so the added TeaCache probe NEFF (calibrated-adaptive only) cannot be resident with it | — | — | — | — |
| Wan 2.1 14B | 20 | cadence 2 | **5/20** (stats line) | 85 → **78 s** (1.08×) | 575 → **441** (1.31×) | 576.9 (n=14) | 36.7 dB |
| Wan 2.1 14B | 20 | online-δ α=0.6 | **5/20** (stats line) | 85 → **80 s** (1.06×) | 575 → **439** (1.31×) | 576.1 (n=14) | 36.2 dB |
| Wan 2.1 14B | 20 | calibrated adaptive (target 1.333×, R² 0.61) | **5/20** (stats line) | 85 → **83 s** (1.01×) | 575 → **640** (0.90×) | 863.2 (n=14) | 35.7 dB |

⁷ loop ms/step = denoise-loop wall (first DiT call entry → last call exit) ÷ scheduler steps, so a skipped step counts as ~0 — the per-step figure TeaCache actually changes; the DiT call column is the unchanged cost of one real call. tp4 skips nothing, so its loop figure is its DiT call time (× calls per step). ⁸ PSNR of this cell's output against the tp4 output at the same seed (pixel space; SSIM when scikit-image is installed); an identical output means the controller skipped nothing.

### TeaCache online-delta α sweep (same tp4 artifact, same seed, one host)

| model | α (cell) | skipped / steps (evidence) | skipped step indices | loop ms/step: tp4 → TC⁷ | warm e2e | output vs tp4 (PSNR / SSIM)⁸ | trace what-if⁹ |
|---|---|---:|---|---:|---:|---|---|
| FLUX.1-dev | cadence 2 (ref) | **9/28** (stats line) | — | 271 → **184** (1.47×) | 41 → 39 s | 41.3 dB | — |
| FLUX.1-dev | 0.6 (committed 2026-09-13 host) | **9/28** (stats line) | — | 271 → **184** (1.47×) | 41 → 38 s | 39.7 dB | first skip @ step 5; knee α ≈ 0.05 (from tp4tcod01) |
| FLUX.1-dev | 0.1 (tp4tcod01) | **6/28** (stats line) | 9, 13, 15, 17, 19, 22 | 271 → **213** (1.27×) | 41 → 41 s | 42.8 dB | first skip @ step 9; knee α ≈ 0.05 (from tp4tcod01) |
| FLUX.1-dev | 0.2 (tp4tcod02) | **8/28** (stats line) | 5, 7, 9, 13, 15, 17, 19, 21 | 271 → **194** (1.40×) | 41 → 39 s | 37.6 dB | first skip @ step 5; knee α ≈ 0.05 (from tp4tcod01) |
| FLUX.1-dev | 0.3 (tp4tcod03) | **9/28** (stats line) | 5, 7, 9, 11, 13, 16, 18, 20, 22 | 271 → **184** (1.47×) | 41 → 40 s | 39.0 dB | first skip @ step 5; knee α ≈ 0.05 (from tp4tcod01) |
| FLUX.1-dev | 0.4 (tp4tcod04) | **9/28** (stats line) | 5, 7, 9, 11, 13, 15, 17, 19, 21 | 271 → **184** (1.47×) | 41 → 38 s | 39.7 dB | first skip @ step 5; knee α ≈ 0.05 (from tp4tcod01) |
| FLUX.1-dev | 0.5 (tp4tcod05) | **9/28** (stats line) | 5, 7, 9, 11, 13, 15, 17, 19, 21 | 271 → **184** (1.47×) | 41 → 39 s | 39.7 dB | first skip @ step 5; knee α ≈ 0.05 (from tp4tcod01) |
| FLUX.1-dev | 0.6 (tp4tcod06) | **9/28** (stats line) | 5, 7, 9, 11, 13, 15, 17, 19, 21 | 271 → **184** (1.47×) | 41 → 40 s | 39.7 dB | first skip @ step 5; knee α ≈ 0.05 (from tp4tcod01) |
| FLUX.1-dev | 0.8 (tp4tcod08) | **9/28** (stats line) | 5, 7, 9, 11, 13, 15, 17, 19, 21 | 271 → **184** (1.47×) | 41 → 39 s | 39.7 dB | first skip @ step 5; knee α ≈ 0.05 (from tp4tcod01) |
| Qwen-Image | cadence 2 (ref) | **5/20** (DiT-call count) | — | 417 → **353** (1.18×) | 65 → 62 s | 45.2 dB | — |
| Qwen-Image | 0.6 (committed 2026-09-13 host) | **5/20** (DiT-call count) | — | 417 → **344** (1.21×) | 65 → 64 s | 45.0 dB | first skip @ step 5; knee α ≈ 0.14 (from tp4tcod02) |
| Qwen-Image | 0.1 (tp4tcod01) | not measured | | | | | |
| Qwen-Image | 0.2 (tp4tcod02) | **3/20** (stats line) | 7, 10, 13 | 417 → **392** (1.07×) | 65 → 64 s | 46.1 dB | first skip @ step 7; knee α ≈ 0.14 (from tp4tcod02) |
| Qwen-Image | 0.3 (tp4tcod03) | **4/20** (stats line) | 5, 7, 10, 12 | 417 → **380** (1.10×) | 65 → 67 s | 44.8 dB | first skip @ step 5; knee α ≈ 0.14 (from tp4tcod02) |
| Qwen-Image | 0.4 (tp4tcod04) | **5/20** (stats line) | 5, 7, 9, 11, 14 | 417 → **343** (1.22×) | 65 → 62 s | 45.8 dB | first skip @ step 5; knee α ≈ 0.14 (from tp4tcod02) |
| Qwen-Image | 0.5 (tp4tcod05) | **5/20** (stats line) | 5, 7, 9, 11, 13 | 417 → **357** (1.17×) | 65 → 68 s | 45.0 dB | first skip @ step 5; knee α ≈ 0.14 (from tp4tcod02) |
| Qwen-Image | 0.6 (tp4tcod06) | **5/20** (stats line) | 5, 7, 9, 11, 13 | 417 → **343** (1.22×) | 65 → 63 s | 45.0 dB | first skip @ step 5; knee α ≈ 0.14 (from tp4tcod02) |
| Qwen-Image | 0.8 (tp4tcod08) | **5/20** (stats line) | 5, 7, 9, 11, 13 | 417 → **353** (1.18×) | 65 → 64 s | 45.0 dB | first skip @ step 5; knee α ≈ 0.14 (from tp4tcod02) |
| LTX-2 | cadence 2 (ref) | **5/20** (stats line) | — | 459 → **345** (1.33×) | 56 → 56 s | 36.6 dB | — |
| LTX-2 | 0.6 (committed 2026-09-13 host) | **4/20** (stats line) | — | 459 → **368** (1.25×) | 56 → 54 s | 36.5 dB | first skip @ step 5; knee α ≈ 0.36 (from tp4tcod02) |
| LTX-2 | 0.1 (tp4tcod01) | not measured | | | | | |
| LTX-2 | 0.2 (tp4tcod02) | **0/20** (stats line) | none | 459 → **460** (1.00×) | 56 → 57 s | **identical (no-op)** | no skip; knee α ≈ 0.36 (from tp4tcod02) |
| LTX-2 | 0.3 (tp4tcod03) | **0/20** (stats line) | none | 459 → **460** (1.00×) | 56 → 58 s | **identical (no-op)** | no skip; knee α ≈ 0.36 (from tp4tcod02) |
| LTX-2 | 0.4 (tp4tcod04) | **1/20** (stats line) | 7 | 459 → **437** (1.05×) | 56 → 57 s | 36.9 dB | first skip @ step 7; knee α ≈ 0.36 (from tp4tcod02) |
| LTX-2 | 0.5 (tp4tcod05) | **3/20** (stats line) | 5, 7, 10 | 459 → **391** (1.17×) | 56 → 57 s | 37.1 dB | first skip @ step 5; knee α ≈ 0.36 (from tp4tcod02) |
| LTX-2 | 0.6 (tp4tcod06) | **4/20** (stats line) | 5, 7, 10, 13 | 459 → **368** (1.25×) | 56 → 56 s | 36.5 dB | first skip @ step 5; knee α ≈ 0.36 (from tp4tcod02) |
| LTX-2 | 0.8 (tp4tcod08) | **5/20** (stats line) | 5, 7, 9, 11, 14 | 459 → **345** (1.33×) | 56 → 56 s | 37.2 dB | first skip @ step 5; knee α ≈ 0.36 (from tp4tcod02) |
| HunyuanVideo | cadence 2 (ref) | **5/20** (stats line) | — | 814 → **624** (1.30×) | 115 → 109 s | 31.8 dB | — |
| HunyuanVideo | 0.6 (committed 2026-09-13 host) | **5/20** (stats line) | — | 814 → **623** (1.31×) | 115 → 109 s | 24.1 dB | first skip @ step 5; knee α ≈ 0.06 (from tp4tcod01) |
| HunyuanVideo | 0.1 (tp4tcod01) | **3/20** (stats line) | 9, 12, 14 | 814 → **704** (1.16×) | 115 → 113 s | 35.5 dB | first skip @ step 9; knee α ≈ 0.06 (from tp4tcod01) |
| HunyuanVideo | 0.2 (tp4tcod02) | **4/20** (stats line) | 7, 9, 12, 14 | 814 → **667** (1.22×) | 115 → 111 s | 33.2 dB | first skip @ step 7; knee α ≈ 0.06 (from tp4tcod01) |
| HunyuanVideo | 0.3 (tp4tcod03) | **5/20** (stats line) | 5, 8, 10, 12, 14 | 814 → **623** (1.31×) | 115 → 109 s | 22.5 dB | first skip @ step 5; knee α ≈ 0.06 (from tp4tcod01) |
| HunyuanVideo | 0.4 (tp4tcod04) | **5/20** (stats line) | 5, 8, 10, 12, 14 | 814 → **630** (1.29×) | 115 → 110 s | 22.5 dB | first skip @ step 5; knee α ≈ 0.06 (from tp4tcod01) |
| HunyuanVideo | 0.5 (tp4tcod05) | **5/20** (stats line) | 5, 7, 9, 11, 13 | 814 → **623** (1.31×) | 115 → 109 s | 24.1 dB | first skip @ step 5; knee α ≈ 0.06 (from tp4tcod01) |
| HunyuanVideo | 0.6 (tp4tcod06) | **5/20** (stats line) | 5, 7, 9, 11, 13 | 814 → **623** (1.31×) | 115 → 106 s | 24.1 dB | first skip @ step 5; knee α ≈ 0.06 (from tp4tcod01) |
| HunyuanVideo | 0.8 (tp4tcod08) | **5/20** (stats line) | 5, 7, 9, 11, 13 | 814 → **626** (1.30×) | 115 → 114 s | 24.1 dB | first skip @ step 5; knee α ≈ 0.06 (from tp4tcod01) |
| Wan 2.1 14B | cadence 2 (ref) | **5/20** (stats line) | — | 575 → **441** (1.31×) | 85 → 78 s | 36.7 dB | — |
| Wan 2.1 14B | 0.6 (committed 2026-09-13 host) | **5/20** (stats line) | — | 575 → **439** (1.31×) | 85 → 80 s | 36.2 dB | first skip @ step 5; knee α ≈ 0.27 (from tp4tcod02) |
| Wan 2.1 14B | 0.1 (tp4tcod01) | not measured | | | | | |
| Wan 2.1 14B | 0.2 (tp4tcod02) | **0/20** (stats line) | none | 575 → **584** (0.99×) | 85 → 83 s | **identical (no-op)** | no skip; knee α ≈ 0.27 (from tp4tcod02) |
| Wan 2.1 14B | 0.3 (tp4tcod03) | **2/20** (stats line) | 8, 11 | 575 → **526** (1.09×) | 85 → 80 s | 37.5 dB | first skip @ step 8; knee α ≈ 0.27 (from tp4tcod02) |
| Wan 2.1 14B | 0.4 (tp4tcod04) | **3/20** (stats line) | 7, 9, 12 | 575 → **497** (1.16×) | 85 → 79 s | 36.7 dB | first skip @ step 7; knee α ≈ 0.27 (from tp4tcod02) |
| Wan 2.1 14B | 0.5 (tp4tcod05) | **4/20** (stats line) | 5, 7, 9, 11 | 575 → **469** (1.23×) | 85 → 82 s | 36.4 dB | first skip @ step 5; knee α ≈ 0.27 (from tp4tcod02) |
| Wan 2.1 14B | 0.6 (tp4tcod06) | **5/20** (stats line) | 5, 7, 9, 11, 13 | 575 → **441** (1.30×) | 85 → 79 s | 36.2 dB | first skip @ step 5; knee α ≈ 0.27 (from tp4tcod02) |
| Wan 2.1 14B | 0.8 (tp4tcod08) | **5/20** (stats line) | 5, 7, 9, 11, 13 | 575 → **444** (1.29×) | 85 → 77 s | 36.2 dB | first skip @ step 5; knee α ≈ 0.27 (from tp4tcod02) |

The controller (`difflet/pipeline/teacache.py`, online-delta mode) skips step *s* iff the rel-L1 delta of the last full step is < α × baseline, where the baseline is the first measured delta (step 1, inside the 5-step warmup, where the trajectory moves fastest) and never skips two steps in a row — so skips are capped at cadence 2's count (9/28, 5/20) whatever α, and α only chooses WHICH of the eligible steps go. Warm e2e is load-dominated and shown only for completeness. ⁹ what-if = read off the lowest-α run's per-step delta trace: the first eligible step the rule would skip at this α, and the knee α below which it would skip nothing; exact only up to that run's own first skip (a skip changes the trajectory after it).

### Serving layer: `difflet serve` (resident model, tp4) vs the CLI

| model | endpoint | startup → /ready: first (compiles) / warm⁹ | c=1 p50 / p90 / p99 | c=2 p50 | c=4 p50 | throughput (any c) | CLI warm e2e → resident speedup | NeuronCore util (c=1)¹⁰ | device mem | errors | cost / 1k¹¹ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| FLUX.1-dev | `/v1/chat/completions` | 21 min / **330 s** | **8.213** / 8.22 / 8.221 s | 16.429 s | 32.853 s | **438 img/h** | 41 s → 5.0× | 91.5% | 53.26 GB | 0 | $2.08 |
| Qwen-Image | `/v1/chat/completions` | 28 min / **1339 s** | **9.02** / 9.032 / 9.033 s | 18.058 s | 36.09 s | **399 img/h** | 65 s → 7.2× | 85.9% | 68.8 GB | 0 | $2.28 |
| LTX-2 | `/v1/videos/sync` | 34 min / **926 s** | **37.537** / 38.471 / 39.144 s | 75.752 s | 149.752 s | **95 videos/h** | 56 s → 1.5× | 22.8% | 42.31 GB | 0 | $9.54 |
| HunyuanVideo | `/v1/videos/sync` | 3 min / **1487 s** | **34.053** / 34.118 / 34.165 s | 68.197 s | 119.548 s | **106 videos/h** | 115 s → 3.4× | 88.2% | 87.59 GB | 0 | $8.61 |
| Wan 2.1 14B | `/v1/videos/sync` | 103 min / **426 s** | **13.182** / 13.191 / 13.194 s | 26.343 s | 52.823 s | **273 videos/h** | 85 s → 6.4× | 85.3% | 70.23 GB | 0 | $3.33 |

Closed loop: c in-flight requests until 8 complete (HunyuanVideo 6), no think time, after 2 warm-up requests; latency = client wall per request including queueing; throughput = successes ÷ level wall. `difflet serve` runs **one resident worker** (`max_running_requests=1`), so c = 2 / 4 measure queueing (p50 ≈ c × service time) and throughput is flat — parallel execution needs `--dp` replicas, which need ≥ 2 cores each. Image requests are JSON on `/v1/chat/completions` (base64 PNG back); video requests are multipart on `/v1/videos/sync` (mp4 bytes back), admitted through the video service FIFO (`--max-queued-requests 8`, `--request-timeout 1800`). ⁹ Serving has its own immutable artifact generation under `~/.cache/difflet/serving/`: the first start compiles it from scratch (the CLI artifacts are not reused); the second figure is a restart against the published generation (no compile, load only) — measured after the other models had evicted this model's files from the page cache, so it is a cold-cache load; a restart right after publish, page cache warm, took 185 s for HunyuanVideo. ¹⁰ Mean over all 4 cores of neuron-monitor's `neuroncore_utilization` sampled every 1 s during the c=1 level. ¹¹ At the indicative hourly price stated above.

### DiT per-step vs tp4 (lower is better; ratio = tp4 / config; a step with two sequential CFG calls counts both calls)

| model | tp4 | tp2cp2 | tp4sp | tp2cfg | tp4cfg2 | tp4sdpa | tp4tc2 | tp4tcod | tp4tcad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLUX.1-dev | 270.7 ms | 263.1 ms (1.03×) | 278.8 ms (0.97×) | N/A | N/A | 682.3 ms (0.40×) | 183.9 ms (1.47×) | 184.2 ms (1.47×) | 186.3 ms (1.45×) |
| Qwen-Image | 417.3 ms | 454.3 ms (0.92×) | 365.6 ms (1.14×) | N/A | N/A | 788.8 ms (0.53×) | 313.0 ms (1.33×) | 313.1 ms (1.33×) | 315.4 ms (1.32×) |
| LTX-2 | 459.2 ms | N/A | N/A | 779.4 ms (0.59×) | 918.1 ms (0.50×) | 506.8 ms (0.91×) | 344.6 ms (1.33×) | 367.4 ms (1.25×) | 383.7 ms (1.20×) |
| HunyuanVideo | 814.1 ms | 849.0 ms (0.96×) | 790.6 ms (1.03×) | N/A | N/A | 3641.3 ms (0.22×) | 611.0 ms (1.33×) | 611.1 ms (1.33×) | — |
| Wan 2.1 14B | 575.5 ms | 575.5 ms (1.00×) | 578.2 ms (1.00×) | 1070.6 ms (0.54×) | 1151.5 ms (0.50×) | 1034.8 ms (0.56×) | 432.7 ms (1.33×) | 432.1 ms (1.33×) | 647.4 ms (0.89×) |

⁰ DiT per-step: mean of the inter-step deltas (n = steps − 1). ¹ compile = full `difflet compile` wall (all stages, incl. per-rank presharding); stage caches shared across features are reused, so a later feature's compile can be shorter than tp4's. ² cold = `sync; echo 3 > drop_caches` then one generate. ³ warm = the immediately following generate. ⁴ outputs/hr = 3600 / warm e2e (one image or one video per generate, batch 1, fresh process each — a served deployment with a resident model does better). ⁵ cost / 1k outputs = hourly price ÷ outputs/hr × 1000; AWS publishes no list price for trn2.3xlarge; $0.91/h is trn2.48xlarge on-demand ($14.5556/h, us-east-2, third-party listing sparecores.com fetched 2026-09-12) ÷ 16 chips — indicative only.. ⁶ Neuron weight load summed over the pipeline's stages (from the generate log), cold vs warm — the bulk of the cold→warm gap; LTX-2's text encoder and VAE run on the host and are not in it.

N/A cells are by design (the gate is named in the cell's `.md`): guidance-distilled models (FLUX, Qwen-Image, HunyuanVideo) have no second CFG branch to parallelise; LTX-2 has no CP or SP path. tp2cfg runs the two true-CFG models at guidance 2.0 (the tp4 row is single-branch at guidance 1.0), so its per-step is a two-branch step and is not a same-work comparison with tp4.

<!-- campaign:end -->

<!-- campaign-findings:begin -->
## 2026-09-12 campaign — findings vs the 2026-06/07 run (hand-written)

**tp4 per-step, now the real-loop rule for all five models** (old value in parentheses):
FLUX **270.7 ms** (267.6, +1%) · Qwen-Image **417.3 ms** (447, −7%: the old number was the
isolated synthetic-input timer, this is the first real-loop measurement) · LTX-2 **459.2 ms**
(441.8, +4%) · HunyuanVideo **814.1 ms** (850.6, −4%: measured with the real padded text mask
instead of the all-ones-bound upper bound) · Wan 2.1 **575.5 ms** (554.8, +4%, old = isolated
timer). All n = steps − 1, p90 within 1 ms of the median for every model.

**Warm e2e** (n=1, the generate immediately after the true-cold one): FLUX 41 s (35) ·
Qwen 65 s (63) · LTX-2 56 s (58) · HunyuanVideo **115 s** (144: the VAE decoder now runs on
Neuron instead of the host) · Wan 2.1 **85 s** (56). The Wan regression sits in the stage
loads (warm log): VAE decoder **30.4 s** (was 13.6), UMT5 12.4 s (8.3), transformer 9.2 s (6.5)
— the denoise loop itself is unchanged. Not chased inside the campaign (it would have changed
the code under test); flagged as a follow-up: new toolchain / NEFF size / first-warm-run
effect (the old row was the median of n=3 after a discarded warm-up).

**Cold e2e**: FLUX 308 s (321) · Qwen 494 s (509) · LTX-2 769 s (778) · HunyuanVideo 594 s
(667) · Wan 2.1 407 s (394) — the same load-dominated picture; the load column shows the
cold→warm gap is the disk read of the weights.

**Compile**: FLUX 21.2 min (24.7) · Qwen 24.7 min (21.9) · LTX-2 18.0 min (30.6) ·
HunyuanVideo **82.3 min** (47.1 — the old row was stale, host-VAE; now the VAE decoder is
compiled on-chip inside the DiT stage and its neuronx-cc build is 62 of the 82 min, paid
again for every topology) · Wan 2.1 111.6 min (131, VAE-dominated).

**Native NxDI FLUX baseline** (table under tp4): per-step **271.9 ms** vs difflet 270.7 ms —
identical within noise, as expected: difflet's FLUX backbone is a fork of NxDI's and both run
attention_cte with the same compiler flags. The e2e rows differ by load path: NxDI warm 57 s
(process) with 39 s of component loads incl. a warm-up forward per component in `load()`,
difflet 41 s with 23 s of presharded loads; NxDI cold 330 s vs 308 s. NxDI's compile is
shorter (15.1 vs 21.2 min) because difflet's compile also writes the per-rank presharded
checkpoints that make its later loads faster — a one-time cost moved from every load to the
compile. Measured with the same rules by `benchmark/trn2/nxdi_flux_baseline.sh`; the AWS
example's own "Average generation time" (resident model, 5 warm-ups) corresponds to NxDI's
`generate_s` = 8.0 s here (difflet's realloop generate: 7.9 s).

**tp2cp2 (ulysses) vs tp4, per-step**: FLUX **263.1 ms** (270.7, 1.03× faster) · Qwen-Image
**454.3 ms** (417.3, 0.92×) · HunyuanVideo **849.0 ms** (814.1, 0.96×) · Wan 2.1 **575.5 ms**
(575.5, 1.00× — identical to within 0.1 ms; it is a different artifact — its own compile,
`cp_mode=ulysses` manifest, a 4-shard `tp2w4-cp` weight entry whose cold read is 570 s vs
354 s — so at 480×832×9 the Wan step is FFN/TP-bound and the attention layout does not
move it). On a single 4-core chip CP trades TP width for sequence sharding, so a small loss
or a wash is the expected outcome; the win case is more cores per replica. **Cold e2e is
uniformly worse under CP** (FLUX 491 s vs 308, Qwen 820 vs 494, Wan 623 vs 407): the CP
weight-store entry holds 4 ranks × a tp2 shard = 2× the bytes of the tp4 entry, and cold
e2e is the disk read of those bytes; warm e2e is within a few seconds of tp4. Compile is
shorter where a stage cache is shared (Wan 24.8 min: the VAE from tp4 is reused; Hunyuan
88.8 min: its VAE is inside the DiT stage and recompiles per topology).

**tp4sp (tp4 + Megatron sequence parallel) vs tp4, per-step**: Qwen-Image **365.6 ms**
(417.3, **1.14× faster** — the one clear win of the campaign: Qwen's joint-attention blocks
carry the most per-layer LayerNorm/residual traffic, which SP shards instead of
replicating) · HunyuanVideo **790.6 ms** (814.1, 1.03×) · FLUX **278.8 ms** (270.7, 0.97×) ·
Wan 2.1 **578.2 ms** (575.5, 1.00×). SP reuses the tp4 weight shards (same store entry), so
cold and warm e2e match tp4 within noise (FLUX 307/40 s, Qwen 508/65 s, Hunyuan 596/112 s,
Wan 406/79 s) and compile is the transformer NEFF only where a stage cache is shared
(Wan 16.1 min). Combined with tp2cp2 this gives the per-topology recommendation on a
4-core chip: **Qwen-Image → tp4sp**, **FLUX → tp2cp2 (ulysses)**, HunyuanVideo → tp4sp
(marginal), Wan 2.1 and LTX-2 → tp4.

**tp2cfg (tp2 × CFG-parallel, guidance 2.0)** — only the two true-CFG models; the ratio
column reads 0.5–0.6× because a tp2cfg step is a *two-branch* (uncond + cond) step while the
tp4 row is single-branch at guidance 1.0, so the honest comparison is against two sequential
tp4 branches: Wan 2.1 **1070.6 ms** vs a notional 2 × 575.5 = 1151 ms (≈ 7% better than serial
CFG at tp4), LTX-2 **779.4 ms** vs 2 × 459.2 = 918 ms (≈ 15% better) — notional because tp4
was not run at guidance 2.0. **LTX-2's tp2cfg warm e2e (921 s, load 447 s) is not a
steady-state number**: the `tp2w4-cfg` weight entry (4 ranks × a tp2 shard of the 86 GB
model) plus the host-side text encoder/VAE no longer fit this box's ~100 GB page cache, so
the "warm" process reads most of it from disk again (cold 1077 s); a served deployment keeps
the model resident and never pays this. Wan 2.1: cold 640 s, warm 102 s (load 59 s).

**tp4sdpa — `--attention-impl sdpa` vs the megakernel default (branch
`feat/attention-impl-cli`, same tp4 topology, own compile-cache identity)**, per-step
megakernel → sdpa: FLUX 270.7 → **682.3 ms** (sdpa 2.52× slower) · Qwen-Image 417.3 →
**788.8 ms** (1.89×) · LTX-2 459.2 → **506.8 ms** (1.10×) · HunyuanVideo 814.1 →
**3641.3 ms** (4.47× — the ~40k-token joint sequence materialises the full q×k score
matrix under SDPA; this reproduces the 3719 ms "was running on SDPA" number of the 2026-06
correction) · Wan 2.1 575.5 → **1034.8 ms** (1.80×). The gap tracks attention's share of the
step: largest for the long-sequence video models and FLUX at 1024², smallest for LTX-2 at
480×704×49 where the FFN dominates. Warm e2e moves by the denoise-loop delta only (FLUX
41 → 50 s, Hunyuan 115 → 168 s); loads are unchanged. Compile is *shorter* under SDPA
(FLUX 15.8 vs 21.2 min, LTX-2 8.5 vs 18.0, Wan transformer 5.5 vs 16.1) — no NKI kernel
build. Evidence: `[difflet] DiT attention policy: sdpa` / `tracing attention kernel: sdpa`
in every `<slug>_tp4sdpa_compile.log`, `attention_impl: sdpa` in the artifact manifests,
outputs finite. So the megakernel routing (attention_cte) is worth 1.1–4.5× per step and
is the right default; `sdpa` is the fallback/reference path, not a performance option.

**Follow-up 2 — the measured CFG-parallel baseline (tp4cfg2 = tp4 at guidance 2.0, two
sequential branches on the tp4 artifact, no recompile)**: LTX-2 step = 2 × 459.1 = **918 ms**
vs tp2cfg **779.4 ms** → CFG-parallel **1.18× faster** per step; Wan 2.1 step = 2 × 575.8 =
**1152 ms** vs **1070.6 ms** → **1.08×**. Both match the notional estimates above. e2e at
guidance 2.0: LTX-2 warm **77 s** on tp4 vs tp2cfg's page-cache-bound 921 s (see follow-up 3
for the resident number); Wan 2.1 warm **94 s** on tp4 vs **102 s** tp2cfg — the CFG-parallel
step win is eaten by the 4-shard `tp2w4-cfg` load (50 → 59 s), so at this shape tp4 is the
better *process-level* choice for Wan and tp2cfg only pays off with a resident model.

**Follow-up 1 — Wan 2.1 warm e2e 85 s vs the old 56 s: real, and it is NEFF device-init,
not the page cache.** Steady state with the old protocol (2 discarded warm-ups, then n=3):
**81.7 s** (80.7 / 81.7 / 82.7), stage loads 48–49 s every time (`benchmark/trn2fu/`).
Per-stage `nxd_model.initialize` (the presharded reads are page-cache instant: 140 MB in
0.00 s, 28.6 GB in 0.32 s), new vs the 2026-06 files: Wan VAE decoder **32.0 s vs 13.6**,
UMT5 7.8 vs 8.3, transformer 9.2 vs 6.5 — and the same 2–3× on other models' small NEFFs:
FLUX VAE decoder **6.5 vs 2.2 s**, HunyuanVideo Llama **28.7 vs 12.9 s** (LTX-2's DiT went
the other way, 10.4 vs 13.9). The denoise loop is unchanged (576 vs 555 ms). The only
in-repo load-path change since July is a JSON read (`world_check`, 4f59547); the toolchain
moved from neuronx-cc 2.25.3371 / torch-neuronx 2.14 to 2.26.6360 / 2.15 with runtime 2.34,
so the working hypothesis is a slower NEFF device-init in the new toolchain (the Wan VAE NEFF
is 242 MB). To confirm, re-run one VAE stage under a 2.25 venv (not on this host) or bisect
the runtime package; a served deployment never pays it — resident Wan generates in
**26.0–26.4 s** (first 34.8 s) with the model loaded once.

**Follow-up 3 — LTX-2 tp2cfg steady state.** Process-level warm converges only after the
page cache has been fought over: **1061 → 650 → 510 s** (2 discarded warm-ups, then 1
measured; Neuron load 600 → 173 → 41 s), and the remaining ~470 s of the 510 s is the
host-side text encoder + VAE being re-read from disk — the `tp2w4-cfg` shard entry is
**72 GB** (tp4's is 38 GB) and, with LTX-2's ~40 GB of host-side weights, exceeds this
124 GB box's page cache. That is the honest process-level number on this host. **Resident**
(model loaded once, `step_realloop --generates 3`): **52.4 s per video** (77.5 s for the
first, 52.7 / 52.4 after), per-step 779.7 ms — vs tp4 at guidance 2.0: warm 77 s per process.
So for LTX-2 with true CFG on a 4-core chip: serve it resident on tp2cfg (52 s/video, the
1.18× per-step win), never as one process per request. Files: `benchmark/trn2fu/`.

**TeaCache on the tp4 artifact (no recompile; table "TeaCache vs tp4" above).** *Fixed
cadence 2* skips exactly what the controller's fixed 5-step warmup / 5-step cooldown allows:
9 of 28 steps for FLUX, 5 of 20 for the 20-step models (25% — a 50-step run would skip 20).
The DiT call itself is unchanged (271 / 417 / 459 / 815 / 577 ms), so the gain is purely the
skipped calls: **denoise loop per step 1.47× (FLUX), 1.33× (LTX-2), 1.31× (Wan), 1.30×
(HunyuanVideo), 1.18× (Qwen-Image)**; at the process level warm e2e moves only 1.01–1.08×
because e2e is load-dominated (LTX-2: the host-side encode/decode swamps the loop entirely).
Output vs the same-seed tp4 image/video: PSNR 41.3 dB (FLUX), 45.2 (Qwen), 36.7 (Wan),
36.6 (LTX-2), 31.8 (HunyuanVideo) — different outputs (the controller did work; Qwen prints
no stats line, its evidence is the 15-of-20 DiT-call count and the changed image), with the
video models' lower PSNR being the visible cost of skipping a quarter of the steps at 20
steps. *Adaptive* = **online-delta (α = 0.6)**, the calibration-free controller (skip when
the last full step's relative-L1 delta < α × the latched baseline; never two skips in a row):
it was the calibration-free adaptive mode measured in the 2026-09-13 run; the calibrated
`--teacache-speedup` mode is now wired for all five CLIs and measured on device (2026-09-17,
the **tp4tcad** rows above and the next paragraph).
*Online-delta result*: at these step counts it skipped **the same number of steps as cadence 2**
(9/28 FLUX, 5/20 Qwen / HunyuanVideo / Wan; LTX-2 4/20) — the controller latches its
baseline delta on the first full step, when deltas are largest, so the α = 0.6 threshold is
generous and the no-two-skips-in-a-row latch makes it behave like cadence 2 — with the same
loop speedups (1.47× / 1.21× / 1.25× / 1.31× / 1.31×) and warm e2e within ±2 s of cadence 2.
Where it differs is *which* steps it skips, and that shows in the output: **HunyuanVideo drops
to 24.1 dB PSNR vs tp4 (cadence 2: 31.8 dB)**, FLUX 39.7 vs 41.3, the rest within 0.5 dB. So on
this evidence fixed cadence is the better probe-free choice: equal speed, more predictable
quality; online-delta would need a post-warmup baseline (the code comment already describes
that intent) or a tighter α to earn its "adaptive" label.

**Calibrated adaptive (`tp4tcad`, `--teacache-speedup` + per-model `--teacache-calibration`;
measured 2026-09-17).** This is the TeaCache-paper controller: a per-model degree-4 polynomial
maps the block-0 modulated-input signal to the expected output change, accumulated to a
threshold. Each calibration was fit on device from 3 record-only generates on prompts *distinct
from* the benchmark prompt (the block-0 signal and the noise-prediction rel-L1 the pipeline
actually feeds the controller; JSONs under `benchmark/trn2/teacache_calib/`), and the threshold
was tuned to **cadence 2's skip budget** (9/28 FLUX, 5/20 the rest) so the three modes are
compared at equal skips. Cross-host caveat handled: this host's plain-tp4 real-loop
(`benchmark/trn2check/`) reproduces the 2026-09-13 DiT-call times within ~1 ms (FLUX 271.1 vs
270.7, Qwen 418.0 vs 417.3, LTX-2 460.2 vs 459.2, Wan 576.5 vs 575.5), so the DiT-call and
loop comparisons below are like-for-like; e2e is load-dominated and not compared across hosts.
Results, and the one finding that decides it — **where the adaptive *signal* comes from**:

- *Probe models (FLUX, Qwen-Image) — the fused probe rides the DiT graph, no per-step host
  cost.* FLUX: 9/28 skips, loop 271 → **186 ms/step (1.45×)** vs cadence 2's 1.47×; the probe
  adds **3.3 ms** to the DiT call (274.4 vs the 271.1 same-host tp4). Qwen: 5/20, loop 417 →
  **322 ms/step (1.30×)** vs cadence 2's 1.18×, probe **+2.5 ms** (420.5 vs 418.0). So on the
  probe models calibrated adaptive matches (FLUX) or slightly beats (Qwen) fixed cadence at the
  same skips. Cost: a one-time probe compile — FLUX **834 s** (its own artifact identity, the
  "FLUX needs a full recompile" cost the earlier campaign flagged), Qwen **114 s** (an additive
  `teacache_probe` component of the tp4 DiT stage; the DiT/VAE NEFFs are reused untouched).
- *Host-signal models (LTX-2, Wan) — the block-0 signal is computed on a host CPU shadow /
  transformer every non-skipped step, and that cost outweighs the skips.* LTX-2: 5/20, but the
  signal adds **+51 ms** to each DiT call (511.6 vs 460.2 same-host tp4), so the loop is only
  459 → **381 ms/step (1.20×)** — worse than cadence 2's 1.33×. Wan is decisive: the CPU shadow
  (block-0 of a 14B expert) adds **+287 ms/step** (863.2 vs 576.5), so the loop is 575 →
  **640 ms/step (0.90× — slower than plain tp4)**. For these two models the probe-free modes
  (cadence / online-delta, no per-step signal) are strictly better.
- *Signal quality varies and, at the matched budget, doesn't change the skip count.* Pearson /
  R² of the fit: Qwen **0.98**, LTX-2 **0.97**, FLUX **0.61** (Pearson 0.705), Wan **0.61**
  (Pearson 0.68). All four still land on cadence 2's exact skip budget because the threshold was
  tuned to it — so calibrated adaptive here chooses *which* steps to skip, not *how many*. Its
  only path to beating fixed cadence is a higher target speedup (skip more of the genuinely-flat
  steps a strong signal identifies); that upside was deliberately not exercised, to keep the
  quality comparison fair. Output vs same-seed tp4: FLUX 37.4 dB (cadence 41.3), Qwen 45.0
  (45.2), LTX-2 36.2 (36.6), Wan 35.7 (36.7) — equal-or-slightly-lower than cadence everywhere
  (the controller picks different, sometimes worse, skip steps; Wan's weak signal costs the most
  PSNR).
- *HunyuanVideo: **BLOCKED (HBM)**.* The DiT at 320×512×61 already holds **22.0 GB of the ~24 GB**
  per core-pair (LNC size 2), so the added probe NEFF OOMs (`nrt_tensor_allocate` ret=-12;
  `benchmark/trn2/logs/tp4tcad_prep/hunyuan_video_collect0.log`). The probe-free tc2/tcod modes
  compile no NEFF, which is why they measured fine. Diagnosed fix (recorded in
  `hunyuan_video_tp4tcad.json`): wire `--host-vae` into the HV CLI generate so the VAE's HBM
  frees for the probe (host decode exists only in serving today, `_load_host_vae`); its cost
  includes that a fair PSNR-vs-tp4 then needs the tp4 baseline re-decoded on host too.

**Bottom line vs the earlier fixed-cadence result:** at the matched skip budget, calibrated
adaptive is **not a win over fixed cadence 2**. It ties cadence on the probe models (FLUX/Qwen)
— at the price of a per-model calibration and a probe compile — loses on the host-signal models
(LTX-2 slower, Wan slower than no caching at all), matches-or-slightly-trails on PSNR, and can't
run at all on HunyuanVideo. Fixed cadence 2 remains the better probe-free default; calibrated
adaptive earns its keep only for a probe model (fused, on-device signal) run at a higher target
than cadence's budget — a follow-up this campaign did not measure.

**Online-delta α sweep (`tp4tcod01`…`tp4tcod08`, `--teacache-online-delta` 0.1 / 0.2 / 0.3 / 0.4 /
0.5 / 0.6 / 0.8; measured 2026-09-18, one host; table "TeaCache online-delta α sweep" above).**
The campaign's `tp4tcod` rows all used the repo default α = 0.6; this sweep varies it per model on
the same tp4 artifact (the flag is generate-only and outside the compile-cache key, so no
recompile), with the 0.6 point re-measured here as the cross-host check: skip counts, skipped
steps and PSNRs reproduce the 2026-09-13 rows exactly and loop ms/step within 2 ms. For the sweep
the controller now records *which* steps it skipped and the per-step rel-L1 delta trace (JSON
`teacache.stats`, and the `[teacache] stats` line — Qwen-Image prints one now too), which is
what makes the curve readable. Two mechanics decide everything:

- *The skip count is capped at cadence 2's by construction.* The controller never skips two
  steps in a row and only inside the 5-step warmup / cooldown window, so the most it can skip is
  9/28 (FLUX) or 5/20 — every model reaches that cap somewhere between α = 0.3 and 0.8 and is
  flat above it (0.6 and 0.8 are identical on all five). α therefore chooses *which* eligible
  steps go, not how many; the loop speedup at the cap is cadence 2's (1.47× / 1.22× / 1.33× /
  1.31× / 1.30× for FLUX / Qwen / LTX-2 / HV / Wan).
- *The baseline is the step-1 delta* (the first measured one, inside the warmup, where the
  trajectory moves fastest — not the post-warmup delta the code comment described; corrected).
  Per-model delta traces show why the knees differ so much: by step 5 the delta has fallen to
  ~0.1× the baseline on FLUX and HunyuanVideo, ~0.2× on Qwen, but only ~0.4× on LTX-2 and ~0.3×
  on Wan. So the same α is "loose" on the first two and "tight" on the last two — **there is no
  single right α**; the knee (the α below which nothing is skipped) is ≈ 0.05 FLUX, 0.06 HV,
  0.14 Qwen, 0.27 Wan, 0.36 LTX-2.

Per model (skips / loop ms/step / PSNR vs same-seed tp4; cadence 2 in brackets):

- *FLUX.1-dev.* 0.1: **6/28, 213 ms (1.27×), 42.8 dB** — fewer skips than cadence 2 but a
  *better* image than it (41.3); 0.2: 8/28, 194 ms, 37.6 dB (skips 13–21 in a run, the worst
  quality point); 0.3: 9/28, 39.0 dB; **0.4 and above: 9/28 at exactly the cadence-2 pattern
  (5,7,…,21), 184 ms, 39.7 dB** — the controller *becomes* cadence 2.
- *Qwen-Image.* 0.2: 3/20, 392 ms (1.07×), 46.1 dB; 0.3: 4/20, 44.8; 0.4: 5/20 (5,7,9,11,14),
  **343 ms (1.22×), 45.8 dB** — at the cap with the best quality; 0.5 and above: the cadence-2
  pattern, 45.0 dB (cadence 45.2). Every point is within 1.3 dB; Qwen is insensitive to which
  steps go.
- *LTX-2.* Nothing skipped at 0.2 or 0.3 (output bit-identical to tp4); 0.4: 1/20 (step 7),
  36.9 dB; 0.5: 3/20, 391 ms, 37.1; 0.6: 4/20, 368 ms, 36.5 (the committed row); **0.8: 5/20,
  345 ms (1.33×), 37.2 dB** — the only model where going *above* 0.6 changes anything, and it
  reaches cadence 2's speed with better PSNR than cadence (36.6).
- *HunyuanVideo — the one that matters.* 0.1: 3/20 (9,12,14), 704 ms (1.16×), **35.5 dB**;
  0.2: 4/20 (7,9,12,14), 667 ms (1.22×), **33.2 dB**; 0.3–0.4: 5/20 but skipping **step 5**
  (5,8,10,12,14) → **22.5 dB**; 0.5–0.8: the cadence-2 pattern → 24.1 dB (the committed 0.6
  row; cadence 2 itself 31.8). The step-5 skip alone costs ~10 dB: HunyuanVideo's early
  post-warmup steps are still shaping the video even though their output delta already looks
  flat (0.1× baseline). The α = 0.6 "cliff" reported for this model is therefore a threshold
  choice, not an inherent cost of the controller — at 0.2 it beats fixed cadence by 1.4 dB for
  one fewer skip.
- *Wan 2.1.* Nothing skipped at 0.2; 0.3: 2/20, 526 ms, **37.5 dB** (above cadence's 36.7);
  0.4: 3/20, 497 ms, 36.7; 0.5: 4/20, 469 ms, 36.4; 0.6 and above: 5/20 cadence-2 pattern,
  441 ms (1.30×), 36.2. A smooth, ±1 dB trade of skips for quality.

**Bottom line for the online-delta mode:** as shipped (step-1 baseline, no consecutive skips)
it is a way of *selecting* a subset of cadence 2's skip slots, and the default α = 0.6 lands on
plain cadence 2 for four of the five models — so the earlier "equal speed, more predictable
quality → prefer fixed cadence" conclusion stands for the default. What the sweep adds is a
per-model operating point that fixed cadence cannot express: **HunyuanVideo at α = 0.2 (1.22×,
33.2 dB) or 0.1 (1.16×, 35.5 dB)** instead of cadence 2's 1.30× at 31.8 dB, **FLUX at 0.1
(1.27×, 42.8 dB)** for quality-first use, and **LTX-2 at 0.8** (cadence-2 speed, +0.6 dB). A
single global default is wrong; if one number is needed, 0.4 is the least bad (cap reached on
FLUX / Qwen, HV at its bad pattern though) — better to set α per model, or to fix the two
mechanics (post-warmup baseline; a minimum step index for the first skip) and re-sweep.

**Serving layer (`difflet serve`, one resident worker, tp4; table "Serving layer" above).**
Resident per-request latency is the denoise loop plus on-device encode/decode, with no
weight reload: FLUX **8.2 s** (438 img/h), Qwen-Image **9.0 s** (399 img/h), Wan 2.1
**13.2 s** (273 videos/h), HunyuanVideo **34.1 s** (106 videos/h), LTX-2 **37.5 s** (95
videos/h) — **3.4–7.2× the CLI's per-request warm e2e** for the Neuron-resident models, only
1.5× for LTX-2, whose serving path keeps the text encoder and VAE on the host: its
NeuronCore utilisation during requests is **22.8%** against **85–92%** for the other four,
so LTX-2 serving is host-bound at this shape. Concurrency 2 and 4 leave throughput flat and
multiply p50 (queueing on the single worker; every request 200, no 429 / timeouts with
`--max-queued-requests 8`); scaling needs `--dp` replicas, i.e. ≥ 2 cores per replica.
Startup: serving compiles its own artifact generation on first start — 21 / 28 / 34 / 103 /
≈ 90 min (HunyuanVideo, from the aborted attempt's log) — and a restart reuses it but the
load is page-cache-bound: 330 s (FLUX) to 1487 s (HunyuanVideo) with a cold cache vs 185 s
for HunyuanVideo right after publish. Three incidents, recorded so the next campaign avoids
them: (1) a 1 h `/ready` bound cut Wan's 100-min VAE compile; (2) killing that neuronx-cc
left a 0-byte `model.hlo_module.pb.lock` in `/var/tmp/neuron-compile-cache`, and the next
HunyuanVideo compile waited on it for 4 h printing "Another process must be compiling";
(3) HunyuanVideo's first worker load (cold, freshly written files; 288 s for the DiT init
alone) exceeded the engine's default 900 s `--worker-restart-timeout`, so the server exited
with `503 engine_unavailable` after its compile — set it to 3600 s for cold starts.

**Campaign totals**: 15 measured cells + 5 by-design N/A, every output finite, no failed
cell, no deleted cache (1.5 TB disk, ~1.1 TB used at the end incl. 285 GB of HF weights).
Device time ≈ 14 h; the largest single items were the HunyuanVideo VAE-decoder compiles
(3 × ~62 min, once per topology) and the Wan 2.1 VAE compile (once, ~100 min).

**New capability**: HunyuanVideo context parallel now runs with `--cp-mode ulysses` (its
padded-Llama key-padding mask is expressed as attention_cte bounds inside the ulysses op —
commit on this branch); the tp2cp2 section carries the first measurement (849 ms/step at
tp2 cp2 vs 814 ms at tp4: CP costs ~4% at this 61-frame shape, as expected for a 4-core box).
<!-- campaign-findings:end -->
