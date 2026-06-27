# Benchmark — hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v

**Status:** pending  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-06-25 07:02 UTC

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
| compile (AOT, one-time) | — |
| **e2e generate — cold start** (page cache dropped) | **—** |

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | — | — | — | — | — |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.14.27725+e2ff0410
- `neuronx-cc` = 2.25.3371.0+f524f7f8
- `neuronx-distributed` = 0.19.28093+fc70b593
- `diffusers` = 0.38.0

## Notes

- compile not implemented in difflet (orchestrator stub: needs Qwen2.5-VL/ByT5/image-semantic encoders) — see benchmark/hunyuan_video_15.md

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
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2`) |

> ⚠️ This model is **pending** (not yet runnable in difflet) — the commands below are the *intended* recipe, not a reproduced run.

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --revision 286be7ce72277246578a3e3cc2487e95ddae5bcf \
    --tp-degree 4 --cp-degree 1 --height 480 --width 848 --num-frames 121
difflet generate --model-id hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v --revision 286be7ce72277246578a3e3cc2487e95ddae5bcf \
    --tp-degree 4 --cp-degree 1 --height 480 --width 848 --num-frames 121 \
    --steps 20 --guidance-scale 6.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

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
