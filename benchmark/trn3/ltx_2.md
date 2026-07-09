# Benchmark — Lightricks/LTX-2

**Status:** ok  
**Backend:** trainium  
**Device:** trn3pd98.3xlarge / 4 NeuronCores / 144 GB/device  
**Timestamp:** 2026-06-30 04:03 UTC

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
| compile (AOT, one-time) | 15.0 min (902 s) |
| **e2e generate — cold start** (page cache dropped) | **14.3 min (859 s)** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 5.7 min (345 s) |
| **e2e generate — warm cache** | **57.74 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 13.91 s |

> Cold vs warm: **14.3 min (859 s) → 57.74 s** (14.9× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 345.0 ms | 344.5 ms | 346.0 ms | 343.7 ms | 19 |
| end-to-end (warm) | 57.74 s | 57.74 s | 57.74 s | 57.74 s | 1 |

**Throughput:** 2.899 DiT steps/s

## Compile breakdown

Per component (neuronx-cc AOT). `other` = layout-optimize + weight-shard + neff-save tail (not timed by a single log line).

| component | module load | HLO gen | priority-HLO compile | all-HLO compile | other | **build total** |
|---|---:|---:|---:|---:|---:|---:|
| transformer | 79.20 s | 18.32 s | 105.73 s | 10.0 ms | 3.4 min (206 s) | **6.8 min (409 s)** |
| **Σ component builds** | | | | | | **6.8 min (409 s)** |

> The headline **compile = 15.0 min (902 s)** is the full `difflet compile` wall; the **Σ component builds = 6.8 min (409 s)** above is only the neuronx-cc build sub-phase. The difference is one-time host model load + HLO trace + weight shard/save before/around the builds (largest for big multi-encoder pipelines).

## End-to-end breakdown (cold generate)

difflet runs the pipeline stages sequentially in one process, each (re)loading its component to device. e2e cold is **load-dominated**, not compute-bound.

| stage | weight shard | weight load |
|---|---:|---:|
| transformer (denoise loop) [Neuron] | — | 5.7 min (345 s) |
| **weights load total** | 0.0 ms | **5.7 min (345 s)** |

- **weights load total:** 5.7 min (345 s) of 14.3 min (859 s) wall
- **compute + overhead (residual):** 8.6 min (514 s) = text-encode + denoise loop + VAE decode + process/runtime startup
- text-encoder and VAE decode run on the host (enable_host_pipeline/enable_decode_components), so only the transformer is a Neuron load; the residual is host text-encode + denoise + host VAE decode.

## Output validity

| field | value |
|---|---|
| shape | [1, 49, 3, 480, 704] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [0.0000, 0.8750] (mean 0.3691, std 0.1456) |
| note | saved ltx_2_out.pt |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.14.27725+e2ff0410
- `neuronx-cc` = 2.25.3371.0+f524f7f8
- `neuronx-distributed` = 0.19.28093+fc70b593
- `diffusers` = 0.38.0

## Notes

- per-step = 345.0 ms/DiT-step (median 344.5, p90 346.0, n=19) — measured the SAME way as H100: inter-step deltas of a real 20-step generate (wrapping NeuronLTX2Application.forward_dit, synced, step 0 excluded), NOT the old isolated synthetic-input timer. 20 DiT calls timed; warm generate 54s; output finite=True.
- Default registry shape 512x768x121 also compiles; 480x704x49 used here as the representative fast shape. CFG (guidance>1) needs a batch-2 NEFF.
- e2e_warm = 58 s (n=1; reported after 1 discarded cache-warming run(s) so the OS page cache is warm). The difflet CLI reloads weights every process, so 'warm' = warm disk cache -> faster load, not a resident model; cf. e2e cold and the load/compute breakdown.

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
| measured on | trn3pd98.3xlarge / 4 NeuronCores / 144 GB/device (device folder `trn3`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Lightricks/LTX-2 --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a \
    --tp-degree 4 --cp-degree 1 --height 480 --width 704 --num-frames 49
difflet generate --model-id Lightricks/LTX-2 --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a \
    --tp-degree 4 --cp-degree 1 --height 480 --width 704 --num-frames 49 \
    --steps 20 --guidance-scale 1.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn3 \
    python -m benchmark.cold_warm_e2e --model ltx_2    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn3 \
    python -m benchmark.step_latency  --model ltx_2    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model ltx_2   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
