# Benchmark — Wan-AI/Wan2.2-T2V-A14B-Diffusers

**Status:** ok  
**Backend:** cuda  
**Device:** CUDA / NVIDIA B300 SXM6 AC  
**Timestamp:** 2026-06-27 04:42 UTC

> Best-performing configuration: tp=4, bf16, A14B (high/low-noise experts), attention_cte

## Configuration

| key | value |
|---|---|
| model type | wan |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 480, 'width': 832, 'num_frames': 9} |
| steps | 20 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 0.0 ms |
| **e2e generate — cold start** (page cache dropped) | **16.02 s** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 7.98 s |
| **e2e generate — warm cache** | **17.48 s** |
| peak device memory | 70.9 GB |

> Cold vs warm: **16.02 s → 17.48 s** (0.9× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 240.7 ms | 240.0 ms | 240.8 ms | 235.3 ms | 19 |
| end-to-end (warm) | 17.48 s | 17.48 s | 17.48 s | 17.48 s | 1 |

**Throughput:** 4.155 steps/s

## Compile breakdown

| component | build time |
|---|---|
| eager | 0.0 ms |

## End-to-end breakdown (cold generate)

difflet runs the pipeline stages sequentially in one process, each (re)loading its component to device. e2e cold is **load-dominated**, not compute-bound.

| stage | weight shard | weight load |
|---|---:|---:|
| pipeline load (from_pretrained → device) | — | 7.98 s |
| **weights load total** | — | **7.98 s** |

- **weights load total:** 7.98 s of 16.02 s wall
- **compute + overhead (residual):** 8.03 s = text-encode + denoise loop + VAE decode + process/runtime startup
- eager diffusers (gpu-resident): one fused load of all components; residual is text-encode + denoise loop + VAE decode (no AOT compile). VAE tiling/slicing enabled (standard single-GPU video-decode setting; identical frames, DiT per-step unaffected).

## Output validity

| field | value |
|---|---|
| shape | [1, 9, 3, 480, 832] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [0.0000, 1.0000] (mean 0.4220, std 0.2041) |
| note | diffusers frames (video) |

## Toolchain

- `torch` = 2.9.1+cu128
- `diffusers` = 0.38.0
- `transformers` = 4.57.6
- `accelerate` = 1.14.0
- `cuda` = 12.8

## Notes

- H100/CUDA reference runs single-GPU DENSE via stock diffusers (eager, no AOT compile). The tp=4/cp=1 shown in Configuration/Reproduction is the Trainium sharding for the difflet recipe — NOT how this GPU run executed (effective tp=1). Per-step latency is the load-independent metric comparable to trn2.

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| HF revision (pinned) | `5be7df9619b54f4e2667b2755bc6a756675b5cd7` |
| model type | wan |
| dtype | bf16 |
| parallel | tp=4, cp=1 |
| shape (H×W×F) | 480×832×9 |
| steps | 20 |
| guidance scale | 1.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4, bf16, A14B (high/low-noise experts), attention_cte |
| measured on | CUDA / NVIDIA B300 SXM6 AC (device folder `b300`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9
difflet generate --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9 \
    --steps 20 --guidance-scale 1.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=b300 \
    python -m benchmark.cold_warm_e2e --model wan_2_2    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=b300 \
    python -m benchmark.step_latency  --model wan_2_2    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model wan_2_2   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
