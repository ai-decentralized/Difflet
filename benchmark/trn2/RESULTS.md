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
| [LTX-2](ltx_2.md) | video+audio | 480×704×49 | 28.6 min | **803 s** (13.4 min) | **103 s** | 335→52 s⁸ | **473 ms** (2.11/s) | (1,49,3,480,704) ✓ | ok |
| [Wan 2.1 14B](wan_2_1.md) | video (T2V) | 480×832×9 | 108 min⁵ | **722 s** (12.0 min) | **97 s** | 657→54 s | **1144 ms** (0.87/s) | (1,3,9,480,832) ✓ | ok |
| [Wan 2.2 A14B](wan_2_2.md) | video (T2V) | 480×832×9 | (shares 2.1)⁶ | **635 s** (10.6 min) | **93 s** | 570→51 s | **1144 ms** (0.87/s) | (1,3,9,480,832) ✓ | ok⁶ |
| [Qwen-Image](qwen_image.md) | image (T2I) | 1024×1024 | 13.4 min | **517 s** (8.6 min) | **74 s** | 462→41 s | **447 ms** (2.24/s) | (1,3,1024,1024) ✓ | ok |
| [HunyuanVideo](hunyuan_video.md) | video (T2V) | 320×512×61 | 40 min¹⁰ | **551 s** (9.2 min) | **220 s**⁸ | 342→37 s | **3719 ms** (0.27/s) | (1,3,61,320,512) ✓ | ok |
| [FLUX.1-dev](flux_1_dev.md) | image (T2I) | 1024×1024 | 21.2 min⁹ | **320 s** (5.3 min) | **47 s** | 279→28 s | **266 ms**ᵃ (3.76/s) | 1024² PNG ✓ | ok |
| [HunyuanVideo-1.5](hunyuan_video_15.md) | video (T2V) | 480×848×121 | — | — | — | — | — | — | pending⁷ |

✓ = output is finite (no NaN/Inf) with a sensible value range — see each report.

**The headline finding: e2e is load-dominated, not compute-bound.** For the
pure-Neuron pipelines warm is **5–8× faster** than cold (Qwen 517→74 s, Wan
635→93 s, FLUX 320→47 s); HunyuanVideo is the exception at 2.5× (551→220 s) because
its host VAE decode (~185 s) is not weight-load and doesn't speed up with a warm
cache. The cold→warm gap is otherwise the one-time cold disk read of the weights —
the Neuron denoise compute is small (per-step × steps). So the two metrics that actually characterize
the hardware are the **cold weight-load** and the **DiT per-step**; absolute cold
e2e mostly measures disk + page-cache state. FLUX.1-dev is the fastest warm e2e
(47 s) and fastest per-step (266 ms) of the set.

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
  (FLUX 47 s, Wan 93–97 s, Qwen 74 s, LTX-2 103 s), 220 s for HunyuanVideo (host VAE
  bound). The **Neuron per-step** is the lossless compute floor: FLUX 266 ms, Qwen
  447 ms, LTX-2 473 ms (validated cosine 0.99992 vs CPU), Wan 1144 ms, HunyuanVideo
  3719 ms.

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
