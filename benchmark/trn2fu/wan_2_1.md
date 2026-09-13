# Benchmark — Wan-AI/Wan2.1-T2V-14B-Diffusers

**Status:** ok  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-09-12 11:07 UTC

> Best-performing configuration: tp=4, bf16, single-transformer (no MoE), attention_cte, 2-stage (transformer + VAE) subprocess pipeline

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
| compile (AOT, one-time) | 111.6 min (6699 s) |
| **e2e generate — cold start** (page cache dropped) | **6.8 min (407 s)** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 5.9 min (354 s) |
| **e2e generate — warm cache** | **81.72 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 49.28 s |

> Cold vs warm: **6.8 min (407 s) → 81.72 s** (5.0× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 576.3 ms | 576.2 ms | 576.4 ms | 575.9 ms | 19 |
| end-to-end (warm) | 81.72 s | 81.70 s | 82.72 s | 80.75 s | 3 |

**Throughput:** 1.735 DiT steps/s

## Compile breakdown

Per component (neuronx-cc AOT). `other` = layout-optimize + weight-shard + neff-save tail (not timed by a single log line).

| component | module load | HLO gen | priority-HLO compile | all-HLO compile | other | **build total** |
|---|---:|---:|---:|---:|---:|---:|
| text_encoder_t5 | 598.0 ms | 875.0 ms | 6.43 s | 1.0 ms | 28.22 s | **36.12 s** |
| transformer | 2.14 s | 22.57 s | 71.72 s | 3.0 ms | 2.9 min (174 s) | **4.5 min (271 s)** |
| vae_decoder | 590.0 ms | 2.78 s | 92.8 min (5567 s) | 1.0 ms | 9.77 s | **93.0 min (5580 s)** |
| **Σ component builds** | | | | | | **98.1 min (5886 s)** |

> The headline **compile = 111.6 min (6699 s)** is the full `difflet compile` wall; the **Σ component builds = 98.1 min (5886 s)** above is only the neuronx-cc build sub-phase. The difference is one-time host model load + HLO trace + weight shard/save before/around the builds (largest for big multi-encoder pipelines).

## End-to-end breakdown (cold generate)

difflet runs the pipeline stages sequentially in one process, each (re)loading its component to device. e2e cold is **load-dominated**, not compute-bound.

| stage | weight shard | weight load |
|---|---:|---:|
| text_encoder (UMT5) | — | 89.00 s |
| transformer (denoise loop) | — | 3.9 min (235 s) |
| vae_decoder | — | 29.48 s |
| **weights load total** | 0.0 ms | **5.9 min (354 s)** |

- **weights load total:** 5.9 min (354 s) of 6.8 min (407 s) wall
- **compute + overhead (residual):** 52.95 s = text-encode + denoise loop + VAE decode + process/runtime startup

## Output validity

| field | value |
|---|---|
| shape | None |
| dtype | None |
| finite (no NaN/Inf) | None |
| note | saved wan2_1_t2v_14b_diffusers_out.mp4 |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- per-step = 576.3 ms/DiT-step (median 576.2, p90 576.4, n=19) — measured the SAME way as H100: inter-step deltas of a real 20-step generate (wrapping NeuronWanBackboneApplication.__call__, synced, step 0 excluded), NOT the old isolated synthetic-input timer. 20 DiT calls timed; warm generate 26s; output finite=True.
- compile-only run: e2e/per-step come from cold_warm_e2e / step_realloop
- e2e_cold = 407 s — TRUE cold start (OS page cache dropped before the run), so the weight load is a real cold disk read.
- e2e_warm = 82 s (n=3; reported after 2 discarded cache-warming run(s) so the OS page cache is warm). The difflet CLI reloads weights every process, so 'warm' = warm disk cache -> faster load, not a resident model; cf. e2e cold and the load/compute breakdown.
- resident generate = 26.4 s (generate 3 of 3 in one process, model loaded once; walls [34.8, 26.0, 26.4]) -- the serving steady state; per-step above is from the last generate.

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
| best-perf knobs | tp=4, bf16, single-transformer (no MoE), attention_cte, 2-stage (transformer + VAE) subprocess pipeline |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9
difflet generate --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers --revision 38ec498cb3208fb688890f8cc7e94ede2cbd7f68 \
    --tp-degree 4 --cp-degree 1 --height 480 --width 832 --num-frames 9 \
    --steps 20 --guidance-scale 1.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2 \
    python -m benchmark.cold_warm_e2e --model wan_2_1    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2 \
    python -m benchmark.step_latency  --model wan_2_1    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model wan_2_1   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
