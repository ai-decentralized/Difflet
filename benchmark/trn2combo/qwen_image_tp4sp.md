# Benchmark — Qwen/Qwen-Image

**Status:** compiled  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-10-04 21:12 UTC

> Best-performing configuration: tp=4 + sequence parallel; tp=4, bf16, joint attention via attention_cte

## Configuration

| key | value |
|---|---|
| model type | qwen_image |
| dtype | bf16 |
| parallel | tp=4 cp=1 sp |
| shape | {'height': 1024, 'width': 1024, 'num_frames': None} |
| steps | 20 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 14.3 min (855 s) |
| **e2e generate — cold start** (page cache dropped) | **—** |
| **e2e generate — warm cache** | **63.94 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 35.17 s |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 365.3 ms | 365.2 ms | 365.5 ms | 364.8 ms | 19 |
| end-to-end (warm) | 63.94 s | 64.05 s | 64.46 s | 63.10 s | 5 |

**Throughput:** 2.738 DiT steps/s

## Compile breakdown

Per component (neuronx-cc AOT). `other` = layout-optimize + weight-shard + neff-save tail (not timed by a single log line).

| component | module load | HLO gen | priority-HLO compile | all-HLO compile | other | **build total** |
|---|---:|---:|---:|---:|---:|---:|
| transformer | 86.54 s | 18.75 s | 45.48 s | 7.0 ms | 2.7 min (162 s) | **5.2 min (313 s)** |
| **Σ component builds** | | | | | | **5.2 min (313 s)** |

> The headline **compile = 14.3 min (855 s)** is the full `difflet compile` wall; the **Σ component builds = 5.2 min (313 s)** above is only the neuronx-cc build sub-phase. The difference is one-time host model load + HLO trace + weight shard/save before/around the builds (largest for big multi-encoder pipelines).

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- per-step = 365.3 ms/DiT-step (median 365.2, p90 365.5, n=19) — measured the SAME way as H100: inter-step deltas of a real 20-step generate (wrapping NeuronQwenImageTransformerApplication.__call__, synced, step 0 excluded), NOT the old isolated synthetic-input timer. 20 DiT calls timed; warm generate 28s; output finite=True.
- compile-only run: e2e/per-step come from cold_warm_e2e / step_realloop
- e2e_warm = 64 s (n=5; reported after 1 discarded cache-warming run(s) so the OS page cache is warm). The difflet CLI reloads weights every process, so 'warm' = warm disk cache -> faster load, not a resident model; cf. e2e cold and the load/compute breakdown.

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `Qwen/Qwen-Image` |
| HF revision (pinned) | `75e0b4be04f60ec59a75f475837eced720f823b6` |
| model type | qwen_image |
| dtype | bf16 |
| parallel | tp=4, cp=1 sp |
| shape (H×W×F) | 1024×1024 |
| steps | 20 |
| guidance scale | 4.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4 + sequence parallel; tp=4, bf16, joint attention via attention_cte |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2combo`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
    --tp-degree 4 --cp-degree 1 --sp --height 1024 --width 1024
difflet generate --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
    --tp-degree 4 --cp-degree 1 --sp --height 1024 --width 1024 \
    --steps 20 --guidance-scale 4.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.png

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.cold_warm_e2e --model qwen_image --config tp4sp    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.step_latency  --model qwen_image --config tp4sp    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model qwen_image --config tp4sp   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
