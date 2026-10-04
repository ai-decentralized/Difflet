# Benchmark — Qwen/Qwen-Image

**Status:** ok  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-09-12 14:00 UTC

> Best-performing configuration: tp=2 x cp=2 (ulysses); tp=4, bf16, joint attention via attention_cte

## Configuration

| key | value |
|---|---|
| model type | qwen_image |
| dtype | bf16 |
| parallel | tp=2 cp=2 ulysses |
| shape | {'height': 1024, 'width': 1024, 'num_frames': None} |
| steps | 20 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 34.2 min (2055 s) |
| **e2e generate — cold start** (page cache dropped) | **13.7 min (820 s)** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 12.9 min (771 s) |
| **e2e generate — warm cache** | **70.72 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 39.24 s |

> Cold vs warm: **13.7 min (820 s) → 70.72 s** (11.6× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 454.3 ms | 454.2 ms | 454.4 ms | 453.7 ms | 19 |
| end-to-end (warm) | 70.72 s | 70.72 s | 70.72 s | 70.72 s | 1 |

**Throughput:** 2.201 DiT steps/s

## Compile breakdown

Per component (neuronx-cc AOT). `other` = layout-optimize + weight-shard + neff-save tail (not timed by a single log line).

| component | module load | HLO gen | priority-HLO compile | all-HLO compile | other | **build total** |
|---|---:|---:|---:|---:|---:|---:|
| text_encoder | 252.0 ms | 3.86 s | 98.51 s | 8.04 s | 2.1 min (125 s) | **3.9 min (235 s)** |
| transformer | 88.68 s | 19.75 s | 36.47 s | 7.0 ms | 4.5 min (268 s) | **6.9 min (413 s)** |
| vae_decoder | 399.0 ms | 1.18 s | 6.7 min (405 s) | 0.0 ms | 9.67 s | **6.9 min (416 s)** |
| **Σ component builds** | | | | | | **17.7 min (1065 s)** |

> The headline **compile = 34.2 min (2055 s)** is the full `difflet compile` wall; the **Σ component builds = 17.7 min (1065 s)** above is only the neuronx-cc build sub-phase. The difference is one-time host model load + HLO trace + weight shard/save before/around the builds (largest for big multi-encoder pipelines).

## End-to-end breakdown (cold generate)

difflet runs the pipeline stages sequentially in one process, each (re)loading its component to device. e2e cold is **load-dominated**, not compute-bound.

| stage | weight shard | weight load |
|---|---:|---:|
| text_encoder | — | 2.1 min (123 s) |
| transformer (denoise loop) | — | 10.6 min (639 s) |
| vae_decoder | — | 9.22 s |
| **weights load total** | 0.0 ms | **12.9 min (771 s)** |

- **weights load total:** 12.9 min (771 s) of 13.7 min (820 s) wall
- **compute + overhead (residual):** 49.14 s = text-encode + denoise loop + VAE decode + process/runtime startup

## Output validity

| field | value |
|---|---|
| shape | None |
| dtype | None |
| finite (no NaN/Inf) | None |
| note | saved qwen_image_tp2cp2_out.png |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- per-step = 454.3 ms/DiT-step (median 454.2, p90 454.4, n=19) — measured the SAME way as H100: inter-step deltas of a real 20-step generate (wrapping NeuronQwenImageTransformerApplication.__call__, synced, step 0 excluded), NOT the old isolated synthetic-input timer. 20 DiT calls timed; warm generate 38s; output finite=True.
- compile-only run: e2e/per-step come from cold_warm_e2e / step_realloop
- e2e_cold = 820 s — TRUE cold start (OS page cache dropped before the run), so the weight load is a real cold disk read.
- e2e_warm = 71 s (n=1, warm OS page cache from the immediately-preceding cold run; same session as the 820 s cold start). difflet reloads weights every process, so warm = warm disk cache -> faster load, not a resident model.

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `Qwen/Qwen-Image` |
| HF revision (pinned) | `75e0b4be04f60ec59a75f475837eced720f823b6` |
| model type | qwen_image |
| dtype | bf16 |
| parallel | tp=2, cp=2 ulysses |
| shape (H×W×F) | 1024×1024 |
| steps | 20 |
| guidance scale | 4.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=2 x cp=2 (ulysses); tp=4, bf16, joint attention via attention_cte |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
    --tp-degree 2 --cp-degree 2 --cp-mode ulysses --height 1024 --width 1024
difflet generate --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
    --tp-degree 2 --cp-degree 2 --cp-mode ulysses --height 1024 --width 1024 \
    --steps 20 --guidance-scale 4.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.png

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2 \
    python -m benchmark.cold_warm_e2e --model qwen_image --config tp2cp2    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2 \
    python -m benchmark.step_latency  --model qwen_image --config tp2cp2    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model qwen_image --config tp2cp2   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
