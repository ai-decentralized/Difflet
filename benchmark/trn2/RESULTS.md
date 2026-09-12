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

### DiT per-step vs tp4 (lower is better; ratio = tp4 / config)

| model | tp4 |
|---|---:|
| FLUX.1-dev | 270.7 ms |
| Qwen-Image | 417.3 ms |
| LTX-2 | 459.2 ms |
| HunyuanVideo | 814.1 ms |
| Wan 2.1 14B | 575.5 ms |

⁰ DiT per-step: mean of the inter-step deltas (n = steps − 1). ¹ compile = full `difflet compile` wall (all stages, incl. per-rank presharding); stage caches shared across features are reused, so a later feature's compile can be shorter than tp4's. ² cold = `sync; echo 3 > drop_caches` then one generate. ³ warm = the immediately following generate. ⁴ outputs/hr = 3600 / warm e2e (one image or one video per generate, batch 1, fresh process each — a served deployment with a resident model does better). ⁵ cost / 1k outputs = hourly price ÷ outputs/hr × 1000; AWS publishes no list price for trn2.3xlarge; $0.91/h is trn2.48xlarge on-demand ($14.5556/h, us-east-2, third-party listing sparecores.com fetched 2026-09-12) ÷ 16 chips — indicative only. ⁶ Neuron weight load summed over the pipeline's stages (from the generate log), cold vs warm — the bulk of the cold→warm gap; LTX-2's text encoder and VAE run on the host and are not in it.

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

**New capability**: HunyuanVideo context parallel now runs with `--cp-mode ulysses` (its
padded-Llama key-padding mask is expressed as attention_cte bounds inside the ulysses op —
commit on this branch); the tp2cp2 section carries the first measurement (849 ms/step at
tp2 cp2 vs 814 ms at tp4: CP costs ~4% at this 61-frame shape, as expected for a 4-core box).
<!-- campaign-findings:end -->
