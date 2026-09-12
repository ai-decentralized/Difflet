# Benchmark — Lightricks/LTX-2

**Status:** skipped  
**Backend:** trainium  
**Device:**   
**Timestamp:** 2026-09-12 14:50 UTC

> Best-performing configuration: tp=2 x cp=2 (ulysses); tp=4, bf16, TP-sharded transformer + attention_cte self-attn, guidance=1.0 (batch-1 NEFF)

## Configuration

| key | value |
|---|---|
| model type | ltx_2 |
| dtype | bf16 |
| parallel | tp=2 cp=2 ulysses |
| shape | {'height': 480, 'width': 704, 'num_frames': 49} |
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

## Notes

- N/A by design (tp2cp2): LTX-2 has no context-parallel path (registry supports_cp=False; difflet/models/ltx_2/entry.py raises NotImplementedError: the tri-stream video+audio+text transformer has no CP foundation yet)

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `Lightricks/LTX-2` |
| HF revision (pinned) | `47da56e2ad66ce4125a9922b4a8826bf407f9d0a` |
| model type | ltx_2 |
| dtype | bf16 |
| parallel | tp=2, cp=2 ulysses |
| shape (H×W×F) | 480×704×49 |
| steps | 20 |
| guidance scale | 1.0 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | tp=2 x cp=2 (ulysses); tp=4, bf16, TP-sharded transformer + attention_cte self-attn, guidance=1.0 (batch-1 NEFF) |
| measured on |  (device folder `trn2`) |

```bash
# difflet (Neuron / trn2) — compile is one-time and cached (reused, never recompiled):
difflet compile  --model-id Lightricks/LTX-2 --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a \
    --tp-degree 2 --cp-degree 2 --cp-mode ulysses --height 480 --width 704 --num-frames 49
difflet generate --model-id Lightricks/LTX-2 --revision 47da56e2ad66ce4125a9922b4a8826bf407f9d0a \
    --tp-degree 2 --cp-degree 2 --cp-mode ulysses --height 480 --width 704 --num-frames 49 \
    --steps 20 --guidance-scale 1.0 --seed 42 \
    --prompt "a cinematic shot of a red fox running through a snowy forest" --output out.mp4

# benchmark harness on this device (writes benchmark/<device>/):
DIFFLET_BENCH_DEVICE=trn2 \
    python -m benchmark.cold_warm_e2e --model ltx_2 --config tp2cp2    # true cold + warm e2e
DIFFLET_BENCH_DEVICE=trn2 \
    python -m benchmark.step_latency  --model ltx_2 --config tp2cp2    # warm per-step

# other backends (H100/B300) reproduce the SAME model+config via the generic runner:
#   python -m benchmark.bench --backend cuda --model ltx_2 --config tp2cp2   # diffusers CUDA reference adapter
```

**Measurement protocol** (so the numbers above are comparable across hardware):
- **compile**: AOT, one-time, cached via `--cache-dir` and reused by every run (no recompilation). The headline figure is the full `difflet compile` wall; see the compile-breakdown for the neuronx-cc build sub-phase.
- **e2e cold**: OS page cache dropped (`sync; echo 3 > /proc/sys/vm/drop_caches`) immediately before a single generate → a true cold disk read of the weights.
- **e2e warm**: the very next generate, weights served from the OS page cache.
- **DiT per-step**: warm steady-state transformer-forward latency (in-process, n=20; for FLUX the warm denoise-loop rate, n=1 indicative for LTX-2) — the load-independent compute metric.
- The `cold_warm_e2e`/`step_latency` helpers are the **Trainium path**; on other accelerators use `benchmark.bench --backend <cuda|cpu>` with the same MATRIX config.
- device otherwise **idle** (serial accelerator); one model at a time.
