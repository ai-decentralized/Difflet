# Benchmark — Qwen/Qwen-Image

**Status:** ok  
**Backend:** tpu  
**Device:** Cloud TPU v5litepod-4 / 4 chips / topology 2x2 / 16 GB HBM per chip  
**Timestamp:** 2026-08-24 02:17 UTC

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
| compile (AOT, one-time) | 0.0 ms |
| **e2e generate — cold start** (page cache dropped) | **80.57 s** |
| **e2e generate — warm cache** | **13.52 s** |
| peak device memory | 9.6 GB |

> Cold vs warm: **80.57 s → 13.52 s** (6.0× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 502.3 ms | 499.0 ms | 500.9 ms | 498.0 ms | 19 |
| end-to-end (warm) | 13.52 s | 13.52 s | 13.79 s | 13.25 s | 2 |

**Throughput:** 1.991 steps/s

Per-step basis: **synced** — device-synced inter-step deltas of a real generate loop, step 0 excluded, the same rule the other device folders use (`benchmark/harness.py::RealLoopStepTimer`).

| same loop, other bases | per step |
|---|---|
| throughput (denoise wall clock / steps) | 505.9 ms |

## Stage breakdown (one warm generate)

| stage | seconds |
|---|---|
| text_encode | 1.91 |
| denoise | 10.12 |
| vae_decode | 1.22 |

## Compile breakdown

| component | build time |
|---|---|
| eager | 0.0 ms |

## Output validity

| field | value |
|---|---|
| shape | [1, 4096, 64] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [-1.4554, 1.2626] (mean 0.0136, std 0.4743) |
| note | packed latents; 1066634 byte PNG after decode |

## Toolchain

- `torch` = 2.9.0+cpu
- `torch-xla` = 2.9.0
- `libtpu` = 0.0.20
- `diffusers` = 0.38.0
- `transformers` = 5.15.1
- `accelerator_type` = v5litepod-4

## Notes

- The config_label comes from models.py::MATRIX and describes the TRAINIUM recipe — 'attention_cte' is a Neuron kernel and is NOT what ran here; the TPU backend uses scaled_dot_product_attention. The tp=4/cp=1 sharding IS accurate: the DiT is split across 4 v5e chips, one worker process per chip. Attention uses the Pallas fused kernel (torch_xla.experimental.custom_kernel) above 32M score elements and scaled_dot_product_attention below it. compile_seconds=0 means no AOT artifact was built or reused, not that compilation is free — XLA compiles on each process's first execution, inside e2e_cold. step_latency is on a throughput basis (denoise wall clock / steps): unsynced inter-step deltas under lazy XLA measure the enqueue rate, not device time.

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
| measured on | Cloud TPU v5litepod-4 / 4 chips / topology 2x2 / 16 GB HBM per chip (device folder `v5e`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
    --tp-degree 4 --cp-degree 1 --height 1024 --width 1024
difflet generate --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
    --tp-degree 4 --cp-degree 1 --height 1024 --width 1024 \
    --steps 20 --guidance-scale 4.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.png

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=v5e \
    python -m benchmark.cold_warm_e2e --model qwen_image    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=v5e \
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
