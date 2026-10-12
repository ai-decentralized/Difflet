# Benchmark — Wan-AI/Wan2.1-T2V-14B-Diffusers

**Status:** compiled  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-10-12 01:33 UTC

> Best-performing configuration: tp=4 + TeaCache fixed cadence 2 (--teacache-cadence 2); tp=4, bf16, single-transformer (no MoE), attention_cte, official 81 frames / 50 steps / CFG 5.0, chunked FP32 VAE on device

## Configuration

| key | value |
|---|---|
| model type | wan |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 480, 'width': 832, 'num_frames': 81} |
| steps | 50 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 12.75 s |
| **e2e generate — cold start** (page cache dropped) | **—** |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 6.82 s | 6.59 s | 7.46 s | 6.38 s | 59 |

**Throughput:** 0.147 DiT steps/s

## Compile breakdown

| component | build time |
|---|---|
| wall_total_s | 12.75 s |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- per-step = 6821.7 ms/DiT-step (median 6592.7, p90 7463.3, n=59) — measured the SAME way as H100: inter-step deltas of a real 50-step generate (wrapping NeuronWanBackboneApplication.__call__, synced, step 0 excluded), NOT the old isolated synthetic-input timer. 60 DiT calls timed; warm generate 916s; output finite=True.
- compile-only run: e2e/per-step come from cold_warm_e2e / step_realloop

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| HF revision (pinned) | `38ec498cb3208fb688890f8cc7e94ede2cbd7f68` |
| model type | wan |
| dtype | bf16 |
| parallel | tp=4, cp=1 |
| shape (H×W×F) | 480×832×81 |
| steps | 50 |
| guidance scale | 5.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4 + TeaCache fixed cadence 2 (--teacache-cadence 2); tp=4, bf16, single-transformer (no MoE), attention_cte, official 81 frames / 50 steps / CFG 5.0, chunked FP32 VAE on device |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2-eval-vc2`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 81
difflet generate --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 81 \
    --steps 50 --guidance-scale 5.0 --seed 42 --teacache-cadence 2 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2-eval-vc2 \
    python -m benchmark.cold_warm_e2e --model wan_2_1 --config tp4tc2    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2-eval-vc2 \
    python -m benchmark.step_latency  --model wan_2_1 --config tp4tc2    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model wan_2_1 --config tp4tc2   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
