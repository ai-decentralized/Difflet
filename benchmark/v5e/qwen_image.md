# Benchmark — Qwen/Qwen-Image

**Status:** ok  
**Backend:** tpu  
**Device:** Cloud TPU v5litepod-4 / 4 chips / topology 2x2 / 16 GB HBM per chip  
**Timestamp:** 2026-09-12 06:07 UTC

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
| **e2e generate — cold start** (page cache dropped) | **2.9 min (172 s)** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 83.39 s |
| **e2e generate — warm cache** | **92.42 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 8.02 s |
| **request on the resident process** (weights already on the device, n=2) | **8.77 s** |
| peak device memory | 11.1 GB |

> e2e cold and warm are **fresh processes** (weights reloaded, XLA compiled again where the backend is eager), the same definition as every other device folder. The resident-process row is the same request with the model already loaded: what a served request costs. Its trn2 counterpart is warm e2e minus the warm load (`compute + overhead` in the breakdown).

> Cold vs warm: **2.9 min (172 s) → 92.42 s** (1.9× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 276.8 ms | 276.2 ms | 279.1 ms | 274.1 ms | 38 |
| end-to-end (warm, fresh process) | 92.42 s | 92.32 s | 92.72 s | 92.21 s | 3 |
| request (resident process) | 8.77 s | 8.77 s | 8.81 s | 8.73 s | 2 |

**Throughput:** 3.612 steps/s

Per-step basis: **synced** — device-synced inter-step deltas of a real generate loop, step 0 excluded, the same rule the other device folders use (`benchmark/harness.py::RealLoopStepTimer`).

| same loop, other bases | per step |
|---|---|
| throughput (denoise wall clock / steps) | 288.9 ms |
| enqueue rate (unsynced deltas — **not device time**) | 275.6 ms |

### Natural basis (no per-step sync)

The same generate with no per-step device sync — what a real serving loop delivers, as opposed to what the cross-device rule measures. Both are real; the sync serialises work a lazy backend would otherwise overlap, so the gap is large on XLA and small on an eager backend.

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (natural) | 287.9 ms | 287.9 ms | 288.3 ms | 287.4 ms | 2 |
| end-to-end warm (natural) | 8.62 s | 8.62 s | 8.64 s | 8.61 s | 2 |

## Protocol

| step | value |
|---|---|
| page cache dropped before the cold run | yes |
| warm runs discarded (cache warming) | 1 |
| warm runs reported (fresh process each) | 3 |
| resident requests (synced per-step) | 2 |
| resident requests, natural (no per-step sync) | 2 |

## Stage breakdown (one warm generate)

| stage | seconds |
|---|---|
| text_encode | 1.92 |
| denoise | 5.78 |
| vae_decode | 1.04 |

## Compile breakdown

| component | build time |
|---|---|
| eager | 0.0 ms |

## End-to-end breakdown (cold generate)

One worker process per chip; every stage below is loaded once per process. e2e cold is **load-dominated**, not compute-bound.

| stage | weight shard | weight load |
|---|---:|---:|
| text_encoder (Qwen2.5-VL 7B, host fp32, ordinal 0) | — | 3.50 s |
| transformer (DiT -> chips) | — | 79.61 s |
| vae (host; moved to the chip per decode) | — | 286.0 ms |
| **weights load total** | 0.0 ms | **83.39 s** |

- **weights load total:** 83.39 s of 2.9 min (172 s) wall
- **compute + overhead (residual):** 88.50 s = text-encode + denoise loop + VAE decode + process/runtime startup
- fresh worker processes, one per chip. The residual is process start + imports + one request (text encode + denoise + decode on the primary replica). XLA's first-execution compile is inside the load stages where the model's load_eager warms up (HunyuanVideo, LTX-2, FLUX) and inside the request otherwise (Qwen-Image, Wan).

## End-to-end breakdown (warm generate, fresh process)

| stage | weight load |
|---|---:|
| text_encoder (Qwen2.5-VL 7B, host fp32, ordinal 0) | 1.74 s |
| transformer (DiT -> chips) | 5.96 s |
| vae (host; moved to the chip per decode) | 312.0 ms |
| **weights load total** | **8.02 s** |

- **weights load total:** 8.02 s of 92.72 s wall
- **compute + overhead (residual):** 84.71 s

## Output validity

| field | value |
|---|---|
| shape | [1, 3, 1024, 1024] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [0.0157, 0.9765] (mean 0.6827, std 0.1981) |
| note | decoded image (BCHW, zero_to_one); saved qwen_image_warm2.png |

## Toolchain

- `torch` = 2.9.0+cpu
- `torch-xla` = 2.9.0
- `libtpu` = 0.0.20
- `jax` = 0.7.1
- `diffusers` = 0.38.0
- `transformers` = 5.15.1
- `accelerator_type` = v5litepod-4

## Notes

- The config_label comes from models.py::MATRIX and describes the TRAINIUM recipe -- 'attention_cte' is a Neuron kernel and is NOT what ran here; the TPU backend uses the Pallas fused attention kernel (torch_xla.experimental.custom_kernel) above 32M score elements and scaled_dot_product_attention below it. The tp=4/cp=1 sharding IS accurate: the DiT is split across 4 v5e chips, one worker process per chip. compile_seconds=0 means no AOT artifact was built or reused, not that compilation is free -- XLA compiles on each process's first execution, inside e2e cold AND e2e warm (both are fresh processes, as on trn2), and torch_xla cannot persist the executables. The resident-process request row is the same request with the weights already on the chips: what `difflet serve` delivers. Per-step is the device-synced real-loop rule (harness.RealLoopStepTimer), taken from the resident synced iterations.
- e2e_cold = 172 s -- TRUE cold start (OS page cache dropped before the run), so the weight load is a real cold disk read.
- e2e_warm = 92 s (n=3; reported after 1 discarded cache-warming run(s) so the OS page cache is warm). Every run is a fresh process that reloads the weights, so 'warm' = warm disk cache -> faster load, not a resident model.

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
# Cloud TPU (eager XLA, one worker per chip). The same MATRIX row through the
# same harness; the adapter restarts its workers per run to match trn2's
# fresh-process cold/warm definition, then measures resident requests.
DIFFLET_BENCH_DEVICE=v5e DIFFLET_BACKEND=tpu HF_HOME=/mnt/models/hf \
    python -m benchmark.bench --backend tpu --model qwen_image --skip-download \
        --iters 3 --warm-discard 1 --resident-iters 2 --natural-iters 2 \
        --save-dir artifacts/benchmark-v5e

# the same request through serving:
DIFFLET_BACKEND=tpu difflet serve --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
    --tp-degree 4 --cp-degree 1 --height 1024 --width 1024
```

**Measurement protocol** (identical to the trn2 folder, see there):
- **compile**: none ahead of time. XLA compiles on each process's first execution; that lands in the load stages (where the model's `load_eager` warms up) or in the first request, and is paid again by every fresh process.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`), then one fresh set of worker processes -> a decoded output.
- **e2e warm**: fresh worker processes again, weights served from the page cache; runs after the discarded cache-warming run(s).
- **resident request**: one more request on the workers left running -- the served-request cost.
- **DiT per-step**: device-synced inter-step deltas of the real generate loop, step 0 excluded (`benchmark/harness.py::RealLoopStepTimer`), from the resident synced requests.
- device otherwise **idle**; one model at a time.
