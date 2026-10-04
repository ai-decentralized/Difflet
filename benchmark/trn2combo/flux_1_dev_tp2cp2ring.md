# Benchmark — black-forest-labs/FLUX.1-dev

**Status:** compiled  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-10-03 23:24 UTC

> Best-performing configuration: tp=2 x cp=2 (ring); tp=4 (registry default tp=8 -> 4 on trn2.3xlarge), bf16, attention_cte

## Configuration

| key | value |
|---|---|
| model type | flux |
| dtype | bf16 |
| parallel | tp=2 cp=2 ring |
| shape | {'height': 1024, 'width': 1024, 'num_frames': None} |
| steps | 28 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 3.9 min (234 s) |
| **e2e generate — cold start** (page cache dropped) | **—** |
| **e2e generate — warm cache** | **44.84 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 26.82 s |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 267.4 ms | 267.3 ms | 267.7 ms | 267.0 ms | 27 |
| end-to-end (warm) | 44.84 s | 44.74 s | 45.34 s | 44.45 s | 3 |

**Throughput:** 3.740 DiT steps/s

## Compile breakdown

Per component (neuronx-cc AOT). `other` = layout-optimize + weight-shard + neff-save tail (not timed by a single log line).

| component | module load | HLO gen | priority-HLO compile | all-HLO compile | other | **build total** |
|---|---:|---:|---:|---:|---:|---:|
| text_encoder_clip | 290.0 ms | 349.0 ms | — | 41.0 ms | 87.0 ms | **768.0 ms** |
| text_encoder_t5 | 366.0 ms | 686.0 ms | — | 79.0 ms | 74.0 ms | **1.20 s** |
| transformer | 1.83 s | 13.03 s | 56.30 s | 3.0 ms | 2.5 min (151 s) | **3.7 min (223 s)** |
| vae_decoder | 213.0 ms | 198.0 ms | — | 785.0 ms | 331.0 ms | **1.53 s** |
| **Σ component builds** | | | | | | **3.8 min (226 s)** |

> The headline **compile = 3.9 min (234 s)** is the full `difflet compile` wall; the **Σ component builds = 3.8 min (226 s)** above is only the neuronx-cc build sub-phase. The difference is one-time host model load + HLO trace + weight shard/save before/around the builds (largest for big multi-encoder pipelines).

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- per-step = 267.4 ms/DiT-step (median 267.3, p90 267.7, n=27) — measured the SAME way as H100: inter-step deltas of a real 28-step generate (wrapping NeuronFluxBackboneApplication.__call__, synced, step 0 excluded), NOT the old isolated synthetic-input timer. 28 DiT calls timed; warm generate 8s; output finite=True.
- compile-only run: e2e/per-step come from cold_warm_e2e / step_realloop
- e2e_warm = 45 s (n=3; reported after 1 discarded cache-warming run(s) so the OS page cache is warm). The difflet CLI reloads weights every process, so 'warm' = warm disk cache -> faster load, not a resident model; cf. e2e cold and the load/compute breakdown.
- resident generate = 7.8 s (generate 3 of 3 in one process, model loaded once; walls [7.8, 7.8, 7.8]) -- the serving steady state; per-step above is from the last generate.

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `black-forest-labs/FLUX.1-dev` |
| HF revision (pinned) | `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` |
| model type | flux |
| dtype | bf16 |
| parallel | tp=2, cp=2 ring |
| shape (H×W×F) | 1024×1024 |
| steps | 28 |
| guidance scale | 3.5 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=2 x cp=2 (ring); tp=4 (registry default tp=8 -> 4 on trn2.3xlarge), bf16, attention_cte |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2combo`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id black-forest-labs/FLUX.1-dev --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
    --tp-degree 2 --cp-degree 2 --cp-mode ring --height 1024 --width 1024
difflet generate --model-id black-forest-labs/FLUX.1-dev --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
    --tp-degree 2 --cp-degree 2 --cp-mode ring --height 1024 --width 1024 \
    --steps 28 --guidance-scale 3.5 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.png

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.cold_warm_e2e --model flux_1_dev --config tp2cp2ring    # true cold + warm e2e
# (no in-process step_latency loader for model_type 'flux'; its per-step comes from the warm denoise-loop rate in the generate log — see Notes)

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model flux_1_dev --config tp2cp2ring   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
