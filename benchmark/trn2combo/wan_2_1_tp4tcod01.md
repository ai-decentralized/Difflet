# Benchmark — Wan-AI/Wan2.1-T2V-14B-Diffusers

**Status:** compiled  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-10-04 15:46 UTC

> Best-performing configuration: tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.1; alpha sweep); tp=4, bf16, single-transformer (no MoE), attention_cte, 2-stage (transformer + VAE) subprocess pipeline

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
| compile (AOT, one-time) | 11.89 s |
| **e2e generate — cold start** (page cache dropped) | **—** |
| **e2e generate — warm cache** | **80.98 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 49.18 s |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 574.6 ms | 574.4 ms | 574.9 ms | 574.1 ms | 19 |
| end-to-end (warm) | 80.98 s | 80.79 s | 81.78 s | 80.39 s | 3 |

**Throughput:** 1.740 DiT steps/s

## Compile breakdown

| component | build time |
|---|---|
| wall_total_s | 11.89 s |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- per-step = 574.6 ms/DiT-step (median 574.4, p90 574.9, n=19) — measured the SAME way as H100: inter-step deltas of a real 20-step generate (wrapping NeuronWanBackboneApplication.__call__, synced, step 0 excluded), NOT the old isolated synthetic-input timer. 20 DiT calls timed; warm generate 35s; output finite=True.
- compile-only run: e2e/per-step come from cold_warm_e2e / step_realloop
- e2e_warm = 81 s (n=3; reported after 1 discarded cache-warming run(s) so the OS page cache is warm). The difflet CLI reloads weights every process, so 'warm' = warm disk cache -> faster load, not a resident model; cf. e2e cold and the load/compute breakdown.

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| HF revision (pinned) | `38ec498cb3208fb688890f8cc7e94ede2cbd7f68` |
| model type | wan |
| dtype | bf16 |
| parallel | tp=4, cp=1 |
| shape (H×W×F) | 480×832×9 |
| steps | 20 |
| guidance scale | 1.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.1; alpha sweep); tp=4, bf16, single-transformer (no MoE), attention_cte, 2-stage (transformer + VAE) subprocess pipeline |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2combo`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9
difflet generate --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9 \
    --steps 20 --guidance-scale 1.0 --seed 42 --teacache-online-delta 0.1 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.cold_warm_e2e --model wan_2_1 --config tp4tcod01    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.step_latency  --model wan_2_1 --config tp4tcod01    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model wan_2_1 --config tp4tcod01   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
