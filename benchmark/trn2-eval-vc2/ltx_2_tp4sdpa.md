# Benchmark — Lightricks/LTX-2

**Status:** ok  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-10-12 05:41 UTC

> Best-performing configuration: tp=4, --attention-impl sdpa (PyTorch SDPA via XLA instead of attention_cte); tp=4, bf16, TP-sharded transformer + attention_cte self-attn, guidance=1.0 (batch-1 NEFF)

## Configuration

| key | value |
|---|---|
| model type | ltx_2 |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 480, 'width': 704, 'num_frames': 49} |
| steps | 40 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 8.9 min (535 s) |
| **e2e generate — cold start** (page cache dropped) | **13.0 min (782 s)** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 5.1 min (309 s) |
| **e2e generate — warm cache** | **67.35 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 9.63 s |

> Cold vs warm: **13.0 min (782 s) → 67.35 s** (11.6× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 514.2 ms | 514.0 ms | 514.5 ms | 513.5 ms | 39 |
| end-to-end (warm) | 67.35 s | 67.35 s | 67.35 s | 67.35 s | 1 |

**Throughput:** 1.945 DiT steps/s

## Compile breakdown

Per component (neuronx-cc AOT). `other` = layout-optimize + weight-shard + neff-save tail (not timed by a single log line).

| component | module load | HLO gen | priority-HLO compile | all-HLO compile | other | **build total** |
|---|---:|---:|---:|---:|---:|---:|
| transformer | 84.03 s | 18.56 s | 110.54 s | 13.0 ms | 5.2 min (313 s) | **8.8 min (526 s)** |
| **Σ component builds** | | | | | | **8.8 min (526 s)** |

> The headline **compile = 8.9 min (535 s)** is the full `difflet compile` wall; the **Σ component builds = 8.8 min (526 s)** above is only the neuronx-cc build sub-phase. The difference is one-time host model load + HLO trace + weight shard/save before/around the builds (largest for big multi-encoder pipelines).

## End-to-end breakdown (cold generate)

difflet runs the pipeline stages sequentially in one process, each (re)loading its component to device. e2e cold is **load-dominated**, not compute-bound.

| stage | weight shard | weight load |
|---|---:|---:|
| transformer (denoise loop) [Neuron] | — | 5.1 min (309 s) |
| **weights load total** | 0.0 ms | **5.1 min (309 s)** |

- **weights load total:** 5.1 min (309 s) of 13.0 min (782 s) wall
- **compute + overhead (residual):** 7.9 min (473 s) = text-encode + denoise loop + VAE decode + process/runtime startup
- text-encoder and VAE decode run on the host (enable_host_pipeline/enable_decode_components), so only the transformer is a Neuron load; the residual is host text-encode + denoise + host VAE decode.

## Output validity

| field | value |
|---|---|
| shape | None |
| dtype | None |
| finite (no NaN/Inf) | None |
| note | saved ltx_2_tp4sdpa_out.mp4 |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- per-step = 514.2 ms/DiT-step (median 514.0, p90 514.5, n=39) — measured the SAME way as H100: inter-step deltas of a real 40-step generate (wrapping NeuronLTX2Application.forward_dit, synced, step 0 excluded), NOT the old isolated synthetic-input timer. 40 DiT calls timed; warm generate 48s; output finite=True.
- Default registry shape 512x768x121 also compiles; 480x704x49 used here as the representative fast shape. CFG (guidance>1) needs a batch-2 NEFF.
- compile-only run: e2e/per-step come from cold_warm_e2e / step_realloop
- e2e_cold = 782 s — TRUE cold start (OS page cache dropped before the run), so the weight load is a real cold disk read.
- e2e_warm = 67 s (n=1, warm OS page cache from the immediately-preceding cold run; same session as the 782 s cold start). difflet reloads weights every process, so warm = warm disk cache -> faster load, not a resident model.

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
| steps | 40 |
| guidance scale | 1.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4, --attention-impl sdpa (PyTorch SDPA via XLA instead of attention_cte); tp=4, bf16, TP-sharded transformer + attention_cte self-attn, guidance=1.0 (batch-1 NEFF) |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2-eval-vc2`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Lightricks/LTX-2 --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a \
    --tp-degree 4 --cp-degree 1 --height 480 --width 704 --num-frames 49
difflet generate --model-id Lightricks/LTX-2 --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a \
    --tp-degree 4 --cp-degree 1 --height 480 --width 704 --num-frames 49 \
    --steps 40 --guidance-scale 1.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2-eval-vc2 \
    python -m benchmark.cold_warm_e2e --model ltx_2 --config tp4sdpa    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2-eval-vc2 \
    python -m benchmark.step_latency  --model ltx_2 --config tp4sdpa    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model ltx_2 --config tp4sdpa   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
