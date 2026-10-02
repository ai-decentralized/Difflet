# Benchmark — hunyuanvideo-community/HunyuanVideo

**Status:** ok  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-10-02 08:25 UTC

> Best-performing configuration: tp=4, bf16, attention_cte; FP8 PTQ (weight-only) on the DiT linears

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
| compile (AOT, one-time) | 73.9 min (4437 s) |
| **e2e generate — cold start** (page cache dropped) | **8.8 min (528 s)** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 2.2 min (130 s) |
| **e2e generate — warm cache** | **114.28 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 10.80 s |

> Cold vs warm: **8.8 min (528 s) → 114.28 s** (4.6× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 833.5 ms | 840.3 ms | 848.8 ms | 805.3 ms | 19 |
| end-to-end (warm) | 114.28 s | 114.28 s | 114.28 s | 114.28 s | 1 |

**Throughput:** 1.200 steps/s

Per-step basis: **real-loop DiT wall time per step, device-synced, step 0 excluded** — device-synced inter-step deltas of a real generate loop, step 0 excluded, the same rule the other device folders use (`benchmark/harness.py::RealLoopStepTimer`).

## Compile breakdown

Per component (neuronx-cc AOT). `other` = layout-optimize + weight-shard + neff-save tail (not timed by a single log line).

| component | module load | HLO gen | priority-HLO compile | all-HLO compile | other | **build total** |
|---|---:|---:|---:|---:|---:|---:|
| transformer | 16.03 s | 54.35 s | 2.9 min (172 s) | 4.0 ms | 4.2 min (252 s) | **8.2 min (494 s)** |
| vae_decoder | 2.03 s | 403.0 ms | 59.6 min (3577 s) | 1.0 ms | 95.34 s | **61.2 min (3674 s)** |
| **Σ component builds** | | | | | | **69.5 min (4169 s)** |

> The headline **compile = 73.9 min (4437 s)** is the full `difflet compile` wall; the **Σ component builds = 69.5 min (4169 s)** above is only the neuronx-cc build sub-phase. The difference is one-time host model load + HLO trace + weight shard/save before/around the builds (largest for big multi-encoder pipelines).

## End-to-end breakdown (cold generate)

difflet runs the pipeline stages sequentially in one process, each (re)loading its component to device. e2e cold is **load-dominated**, not compute-bound.

| stage | weight shard | weight load |
|---|---:|---:|
| text_encoder | — | 2.2 min (130 s) |
| **weights load total** | 0.0 ms | **2.2 min (130 s)** |

- **weights load total:** 2.2 min (130 s) of 8.8 min (528 s) wall
- **compute + overhead (residual):** 6.6 min (398 s) = text-encode + denoise loop + VAE decode + process/runtime startup
- VAE decode runs on the host (no Neuron load line); the residual is CLIP+Llama encode + denoise loop + host VAE decode.

## Output validity

| field | value |
|---|---|
| shape | None |
| dtype | None |
| finite (no NaN/Inf) | None |
| note | saved hunyuanvideo_fp8_tensor_wo_out.mp4 |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- e2e_cold = 528 s — TRUE cold start (OS page cache dropped before the run), so the weight load is a real cold disk read.
- e2e_warm = 114 s (n=1, warm OS page cache from the immediately-preceding cold run; same session as the 528 s cold start). difflet reloads weights every process, so warm = warm disk cache -> faster load, not a resident model.

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
| best-perf knobs | tp=4, bf16, attention_cte; FP8 PTQ (weight-only) on the DiT linears |
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
    python -m benchmark.cold_warm_e2e --model hunyuan_video_fp8_wo    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2 \
    python -m benchmark.step_latency  --model hunyuan_video_fp8_wo    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model hunyuan_video_fp8_wo   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
