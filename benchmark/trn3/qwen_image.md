# Benchmark — Qwen/Qwen-Image

**Status:** ok  
**Backend:** trainium  
**Device:** trn3pd98.3xlarge / 4 NeuronCores / 144 GB/device  
**Timestamp:** 2026-06-30 03:29 UTC

> Best-performing configuration: tp=4, bf16, joint attention via attention_cte

## Configuration

| key | value |
|---|---|
| model type | qwen_image |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 1024, 'width': 1024, 'num_frames': None} |
| steps | 20 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 19.8 min (1189 s) |
| **e2e generate — cold start** (page cache dropped) | **6.7 min (400 s)** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 6.2 min (373 s) |
| **e2e generate — warm cache** | **54.56 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 25.79 s |

> Cold vs warm: **6.7 min (400 s) → 54.56 s** (7.3× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 324.1 ms | 324.2 ms | 324.3 ms | 323.8 ms | 20 |
| end-to-end (warm) | 54.56 s | 53.20 s | 57.75 s | 52.74 s | 3 |

**Throughput:** 3.085 DiT steps/s

## Compile breakdown

Per component (neuronx-cc AOT). `other` = layout-optimize + weight-shard + neff-save tail (not timed by a single log line).

| component | module load | HLO gen | priority-HLO compile | all-HLO compile | other | **build total** |
|---|---:|---:|---:|---:|---:|---:|
| text_encoder | 167.0 ms | 2.47 s | 52.07 s | 5.32 s | 31.07 s | **91.09 s** |
| transformer | 82.33 s | 9.65 s | 29.25 s | 5.0 ms | 2.1 min (128 s) | **4.2 min (249 s)** |
| vae_decoder | 340.0 ms | 367.0 ms | 5.3 min (318 s) | 0.0 ms | 6.38 s | **5.4 min (325 s)** |
| **Σ component builds** | | | | | | **11.1 min (665 s)** |

> The headline **compile = 19.8 min (1189 s)** is the full `difflet compile` wall; the **Σ component builds = 11.1 min (665 s)** above is only the neuronx-cc build sub-phase. The difference is one-time host model load + HLO trace + weight shard/save before/around the builds (largest for big multi-encoder pipelines).

## End-to-end breakdown (cold generate)

difflet runs the pipeline stages sequentially in one process, each (re)loading its component to device. e2e cold is **load-dominated**, not compute-bound.

| stage | weight shard | weight load |
|---|---:|---:|
| text_encoder | 100.16 s | 2.1 min (124 s) |
| transformer (denoise loop) | — | 4.0 min (242 s) |
| vae_decoder | 118.0 ms | 6.53 s |
| **weights load total** | 100.28 s | **6.2 min (373 s)** |

- **weights load total:** 6.2 min (373 s) of 6.7 min (400 s) wall
- **compute + overhead (residual):** 26.74 s = text-encode + denoise loop + VAE decode + process/runtime startup

## Output validity

| field | value |
|---|---|
| shape | None |
| dtype | None |
| finite (no NaN/Inf) | None |
| note | saved qwen_image_out.png |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.14.27725+e2ff0410
- `neuronx-cc` = 2.25.3371.0+f524f7f8
- `neuronx-distributed` = 0.19.28093+fc70b593
- `diffusers` = 0.38.0

## Notes

- per-step = 324.1 ms/DiT-forward (warm, in-process, n=20) via benchmark.step_latency — the stable Neuron-compute metric (e2e generate is load-dominated/noisy across processes).
- e2e_warm = 55 s (n=3; reported after 1 discarded cache-warming run(s) so the OS page cache is warm). The difflet CLI reloads weights every process, so 'warm' = warm disk cache -> faster load, not a resident model; cf. e2e cold and the load/compute breakdown.

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `Qwen/Qwen-Image` |
| HF revision (pinned) | `75e0b4be04f60ec59a75f475837eced720f823b6` |
| model type | qwen_image |
| dtype | bf16 |
| parallel | tp=4, cp=1 |
| shape (H×W×F) | 1024×1024 |
| steps | 20 |
| guidance scale | 4.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4, bf16, joint attention via attention_cte |
| measured on | trn3pd98.3xlarge / 4 NeuronCores / 144 GB/device (device folder `trn3`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
    --tp-degree 4 --cp-degree 1 --height 1024 --width 1024
difflet generate --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
    --tp-degree 4 --cp-degree 1 --height 1024 --width 1024 \
    --steps 20 --guidance-scale 4.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.png

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn3 \
    python -m benchmark.cold_warm_e2e --model qwen_image    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn3 \
    python -m benchmark.step_latency  --model qwen_image    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model qwen_image   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
