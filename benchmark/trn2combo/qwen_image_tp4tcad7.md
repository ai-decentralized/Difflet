# Benchmark — Qwen/Qwen-Image

**Status:** compiled  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-10-05 00:24 UTC

> Best-performing configuration: tp=4 + TeaCache calibrated adaptive, 7-skip budget; tp=4, bf16, joint attention via attention_cte

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
| compile (AOT, one-time) | 17.16 s |
| **e2e generate — cold start** (page cache dropped) | **—** |
| **e2e generate — warm cache** | **65.27 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 37.21 s |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 419.4 ms | 417.9 ms | 423.8 ms | 416.7 ms | 12 |
| end-to-end (warm) | 65.27 s | 65.41 s | 66.31 s | 64.51 s | 5 |

**Throughput:** 2.384 DiT steps/s

## Compile breakdown

| component | build time |
|---|---|
| wall_total_s | 17.16 s |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- per-step = 419.4 ms/DiT-step (median 417.9, p90 423.8, n=12) — measured the SAME way as H100: inter-step deltas of a real 20-step generate (wrapping NeuronQwenImageTransformerApplication.__call__, synced, step 0 excluded), NOT the old isolated synthetic-input timer. 13 DiT calls timed; warm generate 29s; output finite=True.
- compile-only run: e2e/per-step come from cold_warm_e2e / step_realloop
- e2e_warm = 65 s (n=5; reported after 1 discarded cache-warming run(s) so the OS page cache is warm). The difflet CLI reloads weights every process, so 'warm' = warm disk cache -> faster load, not a resident model; cf. e2e cold and the load/compute breakdown.

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
| best-perf knobs | tp=4 + TeaCache calibrated adaptive, 7-skip budget; tp=4, bf16, joint attention via attention_cte |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2combo`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
    --tp-degree 4 --cp-degree 1 --height 1024 --width 1024
difflet generate --model-id Qwen/Qwen-Image --revision 75e0b4be04f60ec59a75f475837eced720f823b6 \
    --tp-degree 4 --cp-degree 1 --height 1024 --width 1024 \
    --steps 20 --guidance-scale 4.0 --seed 42 --teacache-speedup 1.538 --teacache-calibration /home/ubuntu/Difflet/.claude/worktrees/flux-best-combo/benchmark/trn2combo/teacache_calib/qwen_image_tp4tcad_s7.json \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.png

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.cold_warm_e2e --model qwen_image --config tp4tcad7    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.step_latency  --model qwen_image --config tp4tcad7    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model qwen_image --config tp4tcad7   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
