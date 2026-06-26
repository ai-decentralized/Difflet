# Benchmark — Wan-AI/Wan2.2-T2V-A14B-Diffusers

**Status:** ok  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-06-25 06:51 UTC

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
| compile (AOT, one-time) | — |
| **e2e generate — cold start** (page cache dropped) | **10.6 min (635 s)** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 9.5 min (570 s) |
| **e2e generate — warm cache** | **92.76 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 51.20 s |

> Cold vs warm: **10.6 min (635 s) → 92.76 s** (6.9× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 1.14 s | 1.14 s | 1.14 s | 1.14 s | 20 |
| end-to-end (warm) | 92.76 s | 92.76 s | 92.76 s | 92.76 s | 1 |

**Throughput:** 0.874 DiT steps/s

## End-to-end breakdown (cold generate)

difflet runs the pipeline stages sequentially in one process, each (re)loading its component to device. e2e cold is **load-dominated**, not compute-bound.

| stage | weight shard | weight load |
|---|---:|---:|
| text_encoder (UMT5) | 533.0 ms | 94.53 s |
| transformer (denoise loop) | 7.3 min (439 s) | 7.7 min (462 s) |
| vae_decoder | 1.72 s | 14.06 s |
| **weights load total** | 7.4 min (441 s) | **9.5 min (570 s)** |

- **weights load total:** 9.5 min (570 s) of 10.6 min (635 s) wall
- **compute + overhead (residual):** 65.05 s = text-encode + denoise loop + VAE decode + process/runtime startup

## Output validity

| field | value |
|---|---|
| shape | [1, 3, 9, 480, 832] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [-0.7227, -0.3164] (mean -0.4743, std 0.0791) |
| note | saved wan2_2_t2v_a14b_diffusers_out.pt |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.14.27725+e2ff0410
- `neuronx-cc` = 2.25.3371.0+f524f7f8
- `neuronx-distributed` = 0.19.28093+fc70b593
- `diffusers` = 0.38.0

## Notes

- difflet runs Wan 2.2 with enable_transformer_2=False -> only the high-noise expert (single transformer), not the full A14B MoE.
- per-step = 1143.6 ms/DiT-forward (warm, in-process, n=20) via benchmark.step_latency — the stable Neuron-compute metric (e2e generate is load-dominated/noisy across processes).
- e2e_cold = 635 s — TRUE cold start (OS page cache dropped before the run via sudo drop_caches), so the weight load is a real cold disk read; this replaces an earlier value taken with the host weights already cached (artificially low).
- e2e_warm = 93 s (n=1, warm OS page cache from the immediately-preceding cold run, same session). difflet reloads weights every process, so warm = warm disk cache -> faster load, not a resident model.
- compile NOT actually performed: the wan orchestrator keys its NEFF cache by shape only, and Wan 2.2 shares Wan 2.1's shape (480x832x9), so `difflet compile` was a false cache hit (~14 s) reusing Wan 2.1's graph. compile_seconds set to null; a real Wan 2.2 compile would be on the order of Wan 2.1's (~108 min, VAE-dominated).

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
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9
difflet generate --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers --revision 5be7df9619b54f4e2667b2755bc6a756675b5cd7 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9 \
    --steps 20 --guidance-scale 1.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2 \
    python -m benchmark.cold_warm_e2e --model wan_2_2    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2 \
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
