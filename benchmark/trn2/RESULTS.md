# Benchmark results — summary

Measured on **trn2.3xlarge** (1 Neuron device, 4 NeuronCores × 24 GB), bf16,
`tp=4`, via the Neuron inference venv
(`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`). The device is serial, so every
run had it to itself (an early contended run skewed badly — LTX-2 970 s vs clean).
**e2e cold** is a *true* cold start: the OS page cache is dropped
(`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before the run, so the
weight load is a real cold disk read. **e2e warm** is the very next run, with the
weights now in the page cache. Each row links to a detailed per-model report.

| model | kind | shape | compile¹ | **e2e cold**² | **e2e warm**³ | load cold→warm⁴ | **DiT per-step**⁰ | output | status |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| [LTX-2](ltx_2.md) | video+audio | 480×704×49 | 28.6 min | **803 s** (13.4 min) | **103 s** | 335→52 s⁸ | **441.8 ms** (2.26/s)ᵇᵉ | (1,49,3,480,704) ✓ | ok |
| [Wan 2.1 14B](wan_2_1.md) | video (T2V) | 480×832×9 | 108 min⁵ | **722 s** (12.0 min) | **97 s** | 657→54 s | **554.8 ms** (1.80/s)ᵈ | (1,3,9,480,832) ✓ | ok |
| [Wan 2.2 A14B](wan_2_2.md) | video (T2V) | 480×832×9 | (shares 2.1)⁶ | **635 s** (10.6 min) | **93 s** | 570→51 s | **554.8 ms** (1.80/s)ᵈ | (1,3,9,480,832) ✓ | ok⁶ |
| [Qwen-Image](qwen_image.md) | image (T2I) | 1024×1024 | 13.4 min | **517 s** (8.6 min) | **74 s** | 462→41 s | **447 ms** (2.24/s) | (1,3,1024,1024) ✓ | ok |
| [HunyuanVideo](hunyuan_video.md) | video (T2V) | 320×512×61 | 41.5 minᶜ | 551 sᶜ | 220 sᶜ | 342→37 s | **850.6 ms** (1.18/s)ᶜ | (1,3,61,320,512) ✓ | okᶜ |
| [FLUX.1-dev](flux_1_dev.md) | image (T2I) | 1024×1024 | 21.2 min⁹ | **320 s** (5.3 min) | **47 s** | 279→28 s | **267.6 ms** (3.74/s)ᵇ | 1024² PNG ✓ | ok |
| [HunyuanVideo-1.5](hunyuan_video_15.md) | video (T2V) | 480×848×121 | — | — | — | — | — | — | pending⁷ |

✓ = output is finite (no NaN/Inf) with a sensible value range — see each report.

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
pure-Neuron pipelines warm is **5–8× faster** than cold (Qwen 517→74 s, Wan
635→93 s, FLUX 320→47 s); HunyuanVideo is the exception at 2.5× (551→220 s) because
its host VAE decode (~185 s) is not weight-load and doesn't speed up with a warm
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
  deployment that keeps weights hot: 47–103 s for the image/short-video pipelines
  (FLUX 47 s, Wan 93–97 s, Qwen 74 s, LTX-2 103 s), 220 s for HunyuanVideo (a stale
  host-VAE-decode measurement; its VAE is now compiled on-chip, e2e pending re-measure
  — see Corrections ᶜ). The **Neuron per-step** is the lossless compute floor (corrected
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
