# Benchmark — black-forest-labs/FLUX.1-dev

**Status:** skipped  
**Backend:** trainium  
**Device:**   
**Timestamp:** 2026-09-13 02:17 UTC

> Best-performing configuration: tp=4 at guidance 2.0 (two sequential CFG branches; baseline for tp2cfg); tp=4 (registry default tp=8 -> 4 on trn2.3xlarge), bf16, attention_cte

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
| compile (AOT, one-time) | — |
| **e2e generate — cold start** (page cache dropped) | **—** |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | — | — | — | — | — |

## Notes

- N/A by design (tp4cfg2): guidance-distilled model: guidance is a conditioning input of its single forward pass, so 'guidance 2.0 on tp4' is not a two-branch CFG baseline -- the tp2cfg cell it would baseline is N/A for this model too

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
| guidance scale | 2.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4 at guidance 2.0 (two sequential CFG branches; baseline for tp2cfg); tp=4 (registry default tp=8 -> 4 on trn2.3xlarge), bf16, attention_cte |
| measured on |  (device folder `trn2`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id black-forest-labs/FLUX.1-dev --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
    --tp-degree 4 --cp-degree 1 --height 1024 --width 1024
difflet generate --model-id black-forest-labs/FLUX.1-dev --revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
    --tp-degree 4 --cp-degree 1 --height 1024 --width 1024 \
    --steps 28 --guidance-scale 2.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.png

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2 \
    python -m benchmark.cold_warm_e2e --model flux_1_dev --config tp4cfg2    # true cold + warm e2e
# (no in-process step_latency loader for model_type 'flux'; its per-step comes from the warm denoise-loop rate in the generate log — see Notes)

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model flux_1_dev --config tp4cfg2   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
