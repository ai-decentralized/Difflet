# Benchmark — Lightricks/LTX-2

**Status:** ok  
**Backend:** tpu  
**Device:** Cloud TPU v5litepod-4 / 4 chips / topology 2x2 / 16 GB HBM per chip  
**Timestamp:** 2026-09-12 06:43 UTC

> Best-performing configuration: tp=4, bf16, TP-sharded transformer + attention_cte self-attn, guidance=1.0 (batch-1 NEFF)

## Configuration

| key | value |
|---|---|
| model type | ltx_2 |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 480, 'width': 704, 'num_frames': 49} |
| steps | 20 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 0.0 ms |
| **e2e generate — cold start** (page cache dropped) | **8.2 min (494 s)** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 3.5 min (210 s) |
| **e2e generate — warm cache** | **4.5 min (269 s)** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 2.5 min (152 s) |
| **request on the resident process** (weights already on the device, n=2) | **53.98 s** |
| peak device memory | 12.5 GB |

> e2e cold and warm are **fresh processes** (weights reloaded, XLA compiled again where the backend is eager), the same definition as every other device folder. The resident-process row is the same request with the model already loaded: what a served request costs. Its trn2 counterpart is warm e2e minus the warm load (`compute + overhead` in the breakdown).

> Cold vs warm: **8.2 min (494 s) → 4.5 min (269 s)** (1.8× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 1.46 s | 1.46 s | 1.47 s | 1.45 s | 38 |
| end-to-end (warm, fresh process) | 4.5 min (269 s) | 4.5 min (268 s) | 4.5 min (271 s) | 4.5 min (267 s) | 3 |
| request (resident process) | 53.98 s | 53.98 s | 54.56 s | 53.41 s | 2 |

**Throughput:** 0.685 steps/s

Per-step basis: **synced** — device-synced inter-step deltas of a real generate loop, step 0 excluded, the same rule the other device folders use (`benchmark/harness.py::RealLoopStepTimer`).

| same loop, other bases | per step |
|---|---|
| throughput (denoise wall clock / steps) | 1462.8 ms |
| enqueue rate (unsynced deltas — **not device time**) | 1458.3 ms |

### Natural basis (no per-step sync)

The same generate with no per-step device sync — what a real serving loop delivers, as opposed to what the cross-device rule measures. Both are real; the sync serialises work a lazy backend would otherwise overlap, so the gap is large on XLA and small on an eager backend.

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (natural) | 1.46 s | 1.46 s | 1.46 s | 1.46 s | 2 |
| end-to-end warm (natural) | 53.18 s | 53.18 s | 54.00 s | 52.36 s | 2 |

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
| text_encode | 24.38 |
| denoise | 29.26 |
| vae_decode | 0.93 |

## Compile breakdown

| component | build time |
|---|---|
| eager | 0.0 ms |

## End-to-end breakdown (cold generate)

One worker process per chip; every stage below is loaded once per process. e2e cold is **load-dominated**, not compute-bound.

| stage | weight shard | weight load |
|---|---:|---:|
| transformer (DiT -> chips + XLA first-execution compile) + Gemma-3 12B (host fp32, ordinal 0) + connectors + VAEs (video VAE on the primary chip) | — | 3.5 min (210 s) |
| **weights load total** | 0.0 ms | **3.5 min (210 s)** |

- **weights load total:** 3.5 min (210 s) of 8.2 min (494 s) wall
- **compute + overhead (residual):** 4.7 min (284 s) = text-encode + denoise loop + VAE decode + process/runtime startup
- fresh worker processes, one per chip. The residual is process start + imports + one request (text encode + denoise + decode on the primary replica). XLA's first-execution compile is inside the load stages where the model's load_eager warms up (HunyuanVideo, LTX-2, FLUX) and inside the request otherwise (Qwen-Image, Wan).

## End-to-end breakdown (warm generate, fresh process)

| stage | weight load |
|---|---:|
| transformer (DiT -> chips + XLA first-execution compile) + Gemma-3 12B (host fp32, ordinal 0) + connectors + VAEs (video VAE on the primary chip) | 2.5 min (152 s) |
| **weights load total** | **2.5 min (152 s)** |

- **weights load total:** 2.5 min (152 s) of 4.5 min (268 s) wall
- **compute + overhead (residual):** 116.68 s

## Output validity

| field | value |
|---|---|
| shape | [1, 49, 3, 480, 704] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [0.0000, 0.8789] (mean 0.3701, std 0.1447) |
| note | decoded video (BFCHW, zero_to_one); saved ltx_2_warm2.mp4, ltx_2_warm2_frames_0_24_48.png |

## Toolchain

- `torch` = 2.9.0+cpu
- `torch-xla` = 2.9.0
- `libtpu` = 0.0.20
- `jax` = 0.7.1
- `diffusers` = 0.38.0
- `transformers` = 5.15.1
- `accelerator_type` = v5litepod-4

## Notes

- Default registry shape 512x768x121 also compiles; 480x704x49 used here as the representative fast shape. CFG (guidance>1) needs a batch-2 NEFF.
- The config_label comes from models.py::MATRIX and describes the TRAINIUM recipe -- 'attention_cte' is a Neuron kernel and is NOT what ran here; the TPU backend uses the Pallas fused attention kernel (torch_xla.experimental.custom_kernel) above 32M score elements and scaled_dot_product_attention below it. The tp=4/cp=1 sharding IS accurate: the DiT is split across 4 v5e chips, one worker process per chip. compile_seconds=0 means no AOT artifact was built or reused, not that compilation is free -- XLA compiles on each process's first execution, inside e2e cold AND e2e warm (both are fresh processes, as on trn2), and torch_xla cannot persist the executables. The resident-process request row is the same request with the weights already on the chips: what `difflet serve` delivers. Per-step is the device-synced real-loop rule (harness.RealLoopStepTimer), taken from the resident synced iterations.
- e2e_cold = 494 s -- TRUE cold start (OS page cache dropped before the run), so the weight load is a real cold disk read.
- e2e_warm = 269 s (n=3; reported after 1 discarded cache-warming run(s) so the OS page cache is warm). Every run is a fresh process that reloads the weights, so 'warm' = warm disk cache -> faster load, not a resident model.

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `Lightricks/LTX-2` |
| HF revision (pinned) | `47da56e2ad66ce4125a9922b4a8826bf407f9d0a` |
| model type | ltx_2 |
| dtype | bf16 |
| parallel | tp=4, cp=1 |
| shape (H×W×F) | 480×704×49 |
| steps | 20 |
| guidance scale | 1.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4, bf16, TP-sharded transformer + attention_cte self-attn, guidance=1.0 (batch-1 NEFF) |
| measured on | Cloud TPU v5litepod-4 / 4 chips / topology 2x2 / 16 GB HBM per chip (device folder `v5e`) |

```bash
# Cloud TPU (eager XLA, one worker per chip). The same MATRIX row through the
# same harness; the adapter restarts its workers per run to match trn2's
# fresh-process cold/warm definition, then measures resident requests.
DIFFLET_BENCH_DEVICE=v5e DIFFLET_BACKEND=tpu HF_HOME=/mnt/models/hf \
    python -m benchmark.bench --backend tpu --model ltx_2 --skip-download \
        --iters 3 --warm-discard 1 --resident-iters 2 --natural-iters 2 \
        --save-dir artifacts/benchmark-v5e

# the same request through serving:
DIFFLET_BACKEND=tpu difflet serve --model-id Lightricks/LTX-2 --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a \
    --tp-degree 4 --cp-degree 1 --height 480 --width 704 --num-frames 49
```

**Measurement protocol** (identical to the trn2 folder, see there):
- **compile**: none ahead of time. XLA compiles on each process's first execution; that lands in the load stages (where the model's `load_eager` warms up) or in the first request, and is paid again by every fresh process.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`), then one fresh set of worker processes -> a decoded output.
- **e2e warm**: fresh worker processes again, weights served from the page cache; runs after the discarded cache-warming run(s).
- **resident request**: one more request on the workers left running -- the served-request cost.
- **DiT per-step**: device-synced inter-step deltas of the real generate loop, step 0 excluded (`benchmark/harness.py::RealLoopStepTimer`), from the resident synced requests.
- device otherwise **idle**; one model at a time.
