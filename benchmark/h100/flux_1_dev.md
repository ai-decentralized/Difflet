# Benchmark — black-forest-labs/FLUX.1-dev

**Status:** ok  
**Backend:** cuda  
**Device:** CUDA / NVIDIA H100 PCIe  
**Timestamp:** 2026-06-27 06:22 UTC

> Best-performing configuration: tp=4 (registry default tp=8 -> 4 on trn2.3xlarge), bf16, attention_cte

## Configuration

| key | value |
|---|---|
| model type | flux |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 1024, 'width': 1024, 'num_frames': None} |
| steps | 28 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 0.0 ms |
| **e2e generate — cold start** (page cache dropped) | **15.88 s** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 6.38 s |
| **e2e generate — warm cache** | **15.77 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 6.43 s |
| peak device memory | 36.3 GB |

> Cold vs warm: **15.88 s → 15.77 s** (1.0× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 310.8 ms | 310.8 ms | 313.0 ms | 305.1 ms | 27 |
| end-to-end (warm) | 15.77 s | 15.77 s | 15.77 s | 15.77 s | 1 |

**Throughput:** 3.217 steps/s

## Compile breakdown

| component | build time |
|---|---|
| eager | 0.0 ms |

## End-to-end breakdown (cold generate)

difflet runs the pipeline stages sequentially in one process, each (re)loading its component to device. e2e cold is **load-dominated**, not compute-bound.

| stage | weight shard | weight load |
|---|---:|---:|
| pipeline load (from_pretrained → device) | — | 6.38 s |
| **weights load total** | — | **6.38 s** |

- **weights load total:** 6.38 s of 15.88 s wall
- **compute + overhead (residual):** 9.50 s = text-encode + denoise loop + VAE decode + process/runtime startup
- eager diffusers (gpu-resident): one fused load of all components; residual is text-encode + denoise loop + VAE decode (no AOT compile). VAE tiling/slicing enabled (standard single-GPU video-decode setting; identical frames, DiT per-step unaffected).

## Output validity

| field | value |
|---|---|
| shape | [1, 3, 1024, 1024] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [0.0000, 1.0000] (mean 0.5765, std 0.2446) |
| note | diffusers images (image) |

## Toolchain

- `torch` = 2.9.1+cu128
- `diffusers` = 0.38.0
- `transformers` = 4.57.6
- `accelerate` = 1.14.0
- `cuda` = 12.8

## Notes

- H100/CUDA reference runs single-GPU DENSE via stock diffusers (eager, no AOT compile). The tp=4/cp=1 shown in Configuration/Reproduction is the Trainium sharding for the difflet recipe — NOT how this GPU run executed (effective tp=1). Per-step latency is the load-independent metric comparable to trn2.
- e2e_warm = 16 s (n=1) — warm-cache generate(s) run in a SEPARATE process (an 80 GB H100 can't hold a second in-process generate after the cold run; the numeric equivalent of B300's in-process second iter). Weights were on disk from the immediately-preceding cold run, so warm = warm disk cache -> faster load, not a resident model.

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `black-forest-labs/FLUX.1-dev` |
| HF revision (pinned) | `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` |
| model type | flux |
| dtype | bf16 |
| parallel | tp=4, cp=1 |
| shape (H×W×F) | 1024×1024 |
| steps | 28 |
| guidance scale | 3.5 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4 (registry default tp=8 -> 4 on trn2.3xlarge), bf16, attention_cte |
| measured on | CUDA / NVIDIA H100 PCIe (device folder `h100`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id black-forest-labs/FLUX.1-dev --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
    --tp-degree 4 --cp-degree 1 --height 1024 --width 1024
difflet generate --model-id black-forest-labs/FLUX.1-dev --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
    --tp-degree 4 --cp-degree 1 --height 1024 --width 1024 \
    --steps 28 --guidance-scale 3.5 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.png

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=h100 \
    python -m benchmark.cold_warm_e2e --model flux_1_dev    # true cold + warm e2e
# (no in-process step_latency loader for model_type 'flux'; its per-step comes from the warm denoise-loop rate in the generate log — see Notes)

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model flux_1_dev   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
