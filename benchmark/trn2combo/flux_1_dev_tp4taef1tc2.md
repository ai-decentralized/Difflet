# Benchmark — black-forest-labs/FLUX.1-dev

**Status:** compiled  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-10-04 01:08 UTC

> Best-performing configuration: tp=4 + TAEF1 decoder + TeaCache cadence 2; tp=4 (registry default tp=8 -> 4 on trn2.3xlarge), bf16, attention_cte

## Configuration

| key | value |
|---|---|
| model type | flux |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 1024, 'width': 1024, 'num_frames': None} |
| steps | 28 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 8.13 s |
| **e2e generate — cold start** (page cache dropped) | **—** |
| **e2e generate — warm cache** | **31.56 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 17.09 s |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 270.1 ms | 270.1 ms | 270.4 ms | 269.9 ms | 18 |
| end-to-end (warm) | 31.56 s | 31.59 s | 32.50 s | 30.63 s | 5 |

**Throughput:** 3.702 DiT steps/s

## Compile breakdown

| component | build time |
|---|---|
| wall_total_s | 8.13 s |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- per-step = 270.1 ms/DiT-step (median 270.1, p90 270.4, n=18) — measured the SAME way as H100: inter-step deltas of a real 28-step generate (wrapping NeuronFluxBackboneApplication.__call__, synced, step 0 excluded), NOT the old isolated synthetic-input timer. 19 DiT calls timed; warm generate 5s; output finite=True.
- compile-only run: e2e/per-step come from cold_warm_e2e / step_realloop
- resident generate = 5.2 s (generate 3 of 3 in one process, model loaded once; walls [5.3, 5.3, 5.2]) -- the serving steady state; per-step above is from the last generate.
- e2e_warm = 32 s (n=5; reported after 1 discarded cache-warming run(s) so the OS page cache is warm). The difflet CLI reloads weights every process, so 'warm' = warm disk cache -> faster load, not a resident model; cf. e2e cold and the load/compute breakdown.

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `black-forest-labs/FLUX.1-dev` |
| HF revision (pinned) | `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` |
| model type | flux |
| dtype | bf16 |
| parallel | tp=4, cp=1 |
| shape (H×W×F) | 1024×1024 |
| steps | 28 |
| guidance scale | 3.5 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4 + TAEF1 decoder + TeaCache cadence 2; tp=4 (registry default tp=8 -> 4 on trn2.3xlarge), bf16, attention_cte |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2combo`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id black-forest-labs/FLUX.1-dev --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
    --tp-degree 4 --cp-degree 1 --height 1024 --width 1024
difflet generate --model-id black-forest-labs/FLUX.1-dev --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
    --tp-degree 4 --cp-degree 1 --height 1024 --width 1024 \
    --steps 28 --guidance-scale 3.5 --seed 42 --teacache-cadence 2 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.png

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2combo \
    python -m benchmark.cold_warm_e2e --model flux_1_dev --config tp4taef1tc2    # true cold + warm e2e
# (no in-process step_latency loader for model_type 'flux'; its per-step comes from the warm denoise-loop rate in the generate log — see Notes)

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model flux_1_dev --config tp4taef1tc2   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
