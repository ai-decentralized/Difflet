# Benchmark — Wan-AI/Wan2.1-T2V-14B-Diffusers

**Status:** compiled  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-10-11 02:00 UTC

> Best-performing configuration: tp=4, 33 frames, 8-step loop, NKI attention; tp=4, bf16, single-transformer (no MoE), attention_cte, official 81 frames / 50 steps / CFG 5.0, chunked FP32 VAE on device

## Configuration

| key | value |
|---|---|
| model type | wan |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 480, 'width': 832, 'num_frames': 33} |
| steps | 8 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 12.4 min (744 s) |
| **e2e generate — cold start** (page cache dropped) | **—** |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 2.44 s | 2.44 s | 2.44 s | 2.44 s | 15 |

**Throughput:** 0.410 DiT steps/s

## Compile breakdown

Per component (neuronx-cc AOT). `other` = layout-optimize + weight-shard + neff-save tail (not timed by a single log line).

| component | module load | HLO gen | priority-HLO compile | all-HLO compile | other | **build total** |
|---|---:|---:|---:|---:|---:|---:|
| text_encoder_t5 | 441.0 ms | 942.0 ms | 4.60 s | 1.0 ms | 28.39 s | **34.37 s** |
| transformer | 2.14 s | 2.6 min (158 s) | 5.9 min (354 s) | 3.0 ms | 3.0 min (182 s) | **11.6 min (695 s)** |
| **Σ component builds** | | | | | | **12.2 min (730 s)** |

> The headline **compile = 12.4 min (744 s)** is the full `difflet compile` wall; the **Σ component builds = 12.2 min (730 s)** above is only the neuronx-cc build sub-phase. The difference is one-time host model load + HLO trace + weight shard/save before/around the builds (largest for big multi-encoder pipelines).

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- per-step = 2439.7 ms/DiT-step (median 2439.8, p90 2442.4, n=15) — measured the SAME way as H100: inter-step deltas of a real 8-step generate (wrapping NeuronWanBackboneApplication.__call__, synced, step 0 excluded), NOT the old isolated synthetic-input timer. 16 DiT calls timed; warm generate 102s; output finite=True.
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
| shape (H×W×F) | 480×832×33 |
| steps | 8 |
| guidance scale | 5.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4, 33 frames, 8-step loop, NKI attention; tp=4, bf16, single-transformer (no MoE), attention_cte, official 81 frames / 50 steps / CFG 5.0, chunked FP32 VAE on device |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2-eval`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 33
difflet generate --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 33 \
    --steps 8 --guidance-scale 5.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2-eval \
    python -m benchmark.cold_warm_e2e --model wan_2_1 --config tp4f33    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2-eval \
    python -m benchmark.step_latency  --model wan_2_1 --config tp4f33    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model wan_2_1 --config tp4f33   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
