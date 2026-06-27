# Benchmark — hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v

**Status:** failed  
**Backend:** cuda  
**Device:** CUDA / NVIDIA H100 PCIe  
**Timestamp:** 2026-06-26 19:09 UTC

> Best-performing configuration: tp=4, bf16, attention_cte + MX precision ops

## Configuration

| key | value |
|---|---|
| model type | hunyuan_video_15 |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 480, 'width': 848, 'num_frames': 121} |
| steps | 20 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 0.0 ms |
| **e2e generate — cold start** (page cache dropped) | **—** |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | — | — | — | — | — |

## Compile breakdown

| component | build time |
|---|---|
| eager | 0.0 ms |

## Toolchain

- `torch` = 2.9.1
- `diffusers` = 0.38.0
- `transformers` = 4.57.6
- `accelerate` = 1.14.0
- `cuda` = 12.8

## Notes

- H100/CUDA reference runs single-GPU DENSE via stock diffusers (eager, no AOT compile). The tp=4/cp=1 shown in Configuration/Reproduction is the Trainium sharding for the difflet recipe — NOT how this GPU run executed (effective tp=1). Per-step latency is the load-independent metric comparable to trn2.
- FAILED: CUDA out of memory. Tried to allocate 51.05 GiB. GPU 0 has a total capacity of 79.19 GiB of which 39.43 GiB is free. Including non-PyTorch memory, this process has 39.75 GiB memory in use. Of the allocated memory 39.13 GiB is allocated by PyTorch, and 35.42 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` |
| HF revision (pinned) | `286be7ce72277246578a3e3cc2487e95ddae5bcf` |
| model type | hunyuan_video_15 |
| dtype | bf16 |
| parallel | tp=4, cp=1 |
| shape (H×W×F) | 480×848×121 |
| steps | 20 |
| guidance scale | 6.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=4, bf16, attention_cte + MX precision ops |
| measured on | CUDA / NVIDIA H100 PCIe (device folder `h100`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --revision 286be7ce72277246578a3e3cc2487e95ddae5bcf \
    --tp-degree 4 --cp-degree 1 --height 480 --width 848 --num-frames 121
difflet generate --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --revision 286be7ce72277246578a3e3cc2487e95ddae5bcf \
    --tp-degree 4 --cp-degree 1 --height 480 --width 848 --num-frames 121 \
    --steps 20 --guidance-scale 6.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=h100 \
    python -m benchmark.cold_warm_e2e --model hunyuan_video_15    # true cold + warm e2e
# (no in-process step_latency loader for model_type 'hunyuan_video_15'; its per-step comes from the warm denoise-loop rate in the generate log — see Notes)

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model hunyuan_video_15   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
