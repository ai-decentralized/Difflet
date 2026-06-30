# Benchmark — hunyuanvideo-community/HunyuanVideo

**Status:** ok  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-06-30 20:05 UTC

> Best-performing configuration: tp=4, bf16, attention_cte

## Configuration

| key | value |
|---|---|
| model type | hunyuan_video |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 320, 'width': 512, 'num_frames': 61} |
| steps | 20 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 47.1 min (2825 s) |
| **e2e generate — cold start** (page cache dropped) | **11.1 min (667 s)** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 8.5 min (513 s) |
| **e2e generate — warm cache** | **3.0 min (178 s)** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 48.61 s |

> Cold vs warm: **11.1 min (667 s) → 3.0 min (178 s)** (3.8× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | — | — | — | — | — |
| end-to-end (warm) | 3.0 min (178 s) | 3.0 min (178 s) | 3.0 min (178 s) | 3.0 min (178 s) | 1 |

## Compile breakdown

Per component (neuronx-cc AOT). `other` = layout-optimize + weight-shard + neff-save tail (not timed by a single log line).

| component | module load | HLO gen | priority-HLO compile | all-HLO compile | other | **build total** |
|---|---:|---:|---:|---:|---:|---:|
| text_encoder_clip | 279.0 ms | 418.0 ms | — | 19.76 s | 99.0 ms | **20.56 s** |
| text_encoder | 99.0 ms | 2.26 s | 80.82 s | 8.89 s | 54.23 s | **2.4 min (146 s)** |
| transformer | 15.98 s | 31.12 s | 3.6 min (213 s) | 3.0 ms | 2.8 min (168 s) | **7.1 min (429 s)** |
| **Σ component builds** | | | | | | **9.9 min (596 s)** |

> The headline **compile = 47.1 min (2825 s)** is the full `difflet compile` wall; the **Σ component builds = 9.9 min (596 s)** above is only the neuronx-cc build sub-phase. The difference is one-time host model load + HLO trace + weight shard/save before/around the builds (largest for big multi-encoder pipelines).

## End-to-end breakdown (cold generate)

difflet runs the pipeline stages sequentially in one process, each (re)loading its component to device. e2e cold is **load-dominated**, not compute-bound.

| stage | weight shard | weight load |
|---|---:|---:|
| text_encoder_clip | — | 6.75 s |
| text_encoder | 115.88 s | 2.1 min (125 s) |
| transformer (denoise loop) | — | 6.3 min (381 s) |
| **weights load total** | 115.88 s | **8.5 min (513 s)** |

- **weights load total:** 8.5 min (513 s) of 11.1 min (667 s) wall
- **compute + overhead (residual):** 2.6 min (155 s) = text-encode + denoise loop + VAE decode + process/runtime startup
- VAE decode runs on the host (no Neuron load line); the residual is CLIP+Llama encode + denoise loop + host VAE decode.

## Output validity

| field | value |
|---|---|
| shape | [1, 3, 61, 320, 512] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [-2.3750, 2.1562] (mean -0.5341, std 0.5848) |
| note | saved hunyuanvideo_out.pt |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.14.27725+e2ff0410
- `neuronx-cc` = 2.25.3371.0+f524f7f8
- `neuronx-distributed` = 0.19.28093+fc70b593
- `diffusers` = 0.38.0

## Notes

- e2e_cold = 667 s — TRUE cold start (OS page cache dropped before the run), so the weight load is a real cold disk read.
- e2e_warm = 178 s (n=1, warm OS page cache from the immediately-preceding cold run; same session as the 667 s cold start). difflet reloads weights every process, so warm = warm disk cache -> faster load, not a resident model.

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `hunyuanvideo-community/HunyuanVideo` |
| HF revision (pinned) | `e8c2aaa66fe3742a32c11a6766aecbf07c56e773` |
| model type | hunyuan_video |
| dtype | bf16 |
| parallel | tp=4, cp=1 |
| shape (H×W×F) | 320×512×61 |
| steps | 20 |
| guidance scale | 6.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4, bf16, attention_cte |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id hunyuanvideo-community/HunyuanVideo --revision e8c2aaa66fe3742a32c11a6766aecbf07c56e773 \
    --tp-degree 4 --cp-degree 1 --height 320 --width 512 --num-frames 61
difflet generate --model-id hunyuanvideo-community/HunyuanVideo --revision e8c2aaa66fe3742a32c11a6766aecbf07c56e773 \
    --tp-degree 4 --cp-degree 1 --height 320 --width 512 --num-frames 61 \
    --steps 20 --guidance-scale 6.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2 \
    python -m benchmark.cold_warm_e2e --model hunyuan_video    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2 \
    python -m benchmark.step_latency  --model hunyuan_video    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model hunyuan_video   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
