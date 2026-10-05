# Benchmark — hunyuanvideo-community/HunyuanVideo

**Status:** compiled  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-10-05 05:53 UTC

> Best-performing configuration: tp=2 x cp=2 (ulysses); tp=4, bf16, attention_cte

## Configuration

| key | value |
|---|---|
| model type | hunyuan_video |
| dtype | bf16 |
| parallel | tp=2 cp=2 ulysses |
| shape | {'height': 320, 'width': 512, 'num_frames': 61} |
| steps | 20 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 82.2 min (4934 s) |
| **e2e generate — cold start** (page cache dropped) | **—** |
| **e2e generate — warm cache** | **2.0 min (122 s)** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 58.56 s |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 849.9 ms | 849.7 ms | 850.0 ms | 849.5 ms | 19 |
| end-to-end (warm) | 2.0 min (122 s) | 117.76 s | 2.2 min (133 s) | 116.60 s | 3 |

**Throughput:** 1.177 DiT steps/s

## Compile breakdown

Per component (neuronx-cc AOT). `other` = layout-optimize + weight-shard + neff-save tail (not timed by a single log line).

| component | module load | HLO gen | priority-HLO compile | all-HLO compile | other | **build total** |
|---|---:|---:|---:|---:|---:|---:|
| text_encoder | 102.0 ms | 2.77 s | 97.64 s | 14.63 s | 88.48 s | **3.4 min (204 s)** |
| transformer | 16.83 s | 56.84 s | 2.4 min (145 s) | 5.0 ms | 3.8 min (226 s) | **7.4 min (444 s)** |
| vae_decoder | 1.90 s | 377.0 ms | 60.5 min (3630 s) | 1.0 ms | 99.03 s | **62.2 min (3731 s)** |
| **Σ component builds** | | | | | | **73.0 min (4379 s)** |

> The headline **compile = 82.2 min (4934 s)** is the full `difflet compile` wall; the **Σ component builds = 73.0 min (4379 s)** above is only the neuronx-cc build sub-phase. The difference is one-time host model load + HLO trace + weight shard/save before/around the builds (largest for big multi-encoder pipelines).

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- per-step = 849.9 ms/DiT-step (median 849.7, p90 850.0, n=19) — measured the SAME way as H100: inter-step deltas of a real 20-step generate (wrapping NeuronHunyuanVideoBackboneApplication.__call__, synced, step 0 excluded), NOT the old isolated synthetic-input timer. 20 DiT calls timed; warm generate 89s; output finite=True.
- compile-only run: e2e/per-step come from cold_warm_e2e / step_realloop
- e2e_warm = 122 s (n=3; reported after 1 discarded cache-warming run(s) so the OS page cache is warm). The difflet CLI reloads weights every process, so 'warm' = warm disk cache -> faster load, not a resident model; cf. e2e cold and the load/compute breakdown.

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `hunyuanvideo-community/HunyuanVideo` |
| HF revision (pinned) | `e8c2aaa66fe3742a32c11a6766aecbf07c56e773` |
| model type | hunyuan_video |
| dtype | bf16 |
| parallel | tp=2, cp=2 ulysses |
| shape (H×W×F) | 320×512×61 |
| steps | 20 |
| guidance scale | 6.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=2 x cp=2 (ulysses); tp=4, bf16, attention_cte |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2combo`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id hunyuanvideo-community/HunyuanVideo --revision e8c2aaa66fe3742a32c11a6766aecbf07c56e773 \
    --tp-degree 2 --cp-degree 2 --cp-mode ulysses --height 320 --width 512 --num-frames 61
difflet generate --model-id hunyuanvideo-community/HunyuanVideo --revision e8c2aaa66fe3742a32c11a6766aecbf07c56e773 \
    --tp-degree 2 --cp-degree 2 --cp-mode ulysses --height 320 --width 512 --num-frames 61 \
    --steps 20 --guidance-scale 6.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.cold_warm_e2e --model hunyuan_video --config tp2cp2    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.step_latency  --model hunyuan_video --config tp2cp2    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model hunyuan_video --config tp2cp2   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
