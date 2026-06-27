# Benchmark — Wan-AI/Wan2.1-T2V-14B-Diffusers

**Status:** ok  
**Backend:** trainium  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-06-25 06:14 UTC

> Best-performing configuration: tp=4, bf16, single-transformer (no MoE), attention_cte, 2-stage (transformer + VAE) subprocess pipeline — head-sharded attention + TP-aware global RMS (was replicated)

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
| compile (AOT, one-time) | 108.5 min (6507 s) |
| **e2e generate — cold start** (page cache dropped) | **12.0 min (722 s)** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 10.9 min (657 s) |
| **e2e generate — warm cache** | **96.77 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 54.43 s |

> Cold vs warm: **12.0 min (722 s) → 96.77 s** (7.5× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 554.8 ms | 554.8 ms | 554.8 ms | 554.8 ms | 20 |
| end-to-end (warm) | 96.77 s | 96.77 s | 96.77 s | 96.77 s | 1 |

**Throughput:** 1.802 DiT steps/s

## Compile breakdown

| component | build time |
|---|---|
| text_encoder(UMT5) | 31.00 s |
| transformer(WanTransformer3DModel, 14B) | 7.2 min (431 s) |
| vae_decoder | 100.8 min (6045 s) |

## End-to-end breakdown (cold generate)

difflet runs the pipeline stages sequentially in one process, each (re)loading its component to device. e2e cold is **load-dominated**, not compute-bound.

| stage | weight shard | weight load |
|---|---:|---:|
| text_encoder (UMT5) | 2.9 min (173 s) | 3.0 min (181 s) |
| transformer (denoise loop) | 7.3 min (438 s) | 7.7 min (461 s) |
| vae_decoder | 1.72 s | 14.41 s |
| **weights load total** | 10.2 min (613 s) | **10.9 min (657 s)** |

- **weights load total:** 10.9 min (657 s) of 12.0 min (722 s) wall
- **compute + overhead (residual):** 65.41 s = text-encode + denoise loop + VAE decode + process/runtime startup

## Output validity

| field | value |
|---|---|
| shape | [1, 3, 9, 480, 832] |
| dtype | torch.float32 |
| finite (no NaN/Inf) | True |
| value range | [-0.8711, -0.2578] (mean -0.5518, std 0.0840) |
| note | saved wan2_1_t2v_14b_diffusers_out.pt |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.14.27725+e2ff0410
- `neuronx-cc` = 2.25.3371.0+f524f7f8
- `neuronx-distributed` = 0.19.28093+fc70b593
- `diffusers` = 0.38.0

## Notes

- CORRECTION: per-step 1144 -> 554.8 ms (2.06x) — the attention was REPLICATED across the 4 TP cores (gather_output=True + full-width across-heads RMSNorm to match HF), so tp=4 only parallelized the FFN, not the attention. Re-wired to head-sharded attention (gather_output=False + RowParallel out + local heads) with a TP-aware global RMS (cross-rank sum-of-squares + per-rank norm-weight slice, like LTX-2's _global_rms_norm). Strict parity vs the replicated baseline on the same input: cosine 0.999768 (rel_l2 2.0e-2, bf16). trn2 now matches H100 (554.8 vs 563 ms) — was 2.0x behind. Measured with the isolated step_latency timer (same as the old 1144 number).
- compile time from a dedicated clean run; the VAE decoder dominates (~100 min) — the Wan video VAE is conv-heavy and slow on neuronx-cc.
- single-transformer (no MoE); attention is unmasked -> attention_cte.
- e2e_cold = 722 s — TRUE cold start (OS page cache dropped before the run via sudo drop_caches), so the weight load is a real cold disk read; this replaces an earlier value taken with the host weights already cached (artificially low).
- e2e_warm = 97 s (n=1, warm OS page cache from the immediately-preceding cold run, same session). difflet reloads weights every process, so warm = warm disk cache -> faster load, not a resident model.
- cold transformer weight load 461 s (shard 438 s) is ~2x the size-implied cold disk read and ~60x the warm shard (7 s) — the cold TP-shard path faults in the mmapped 14B weights during the host-side scatter. Warm load (29 s) is normal; treat the cold shard as a cold-cache artifact, not steady-state.

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
| best-perf knobs | tp=4, bf16, single-transformer (no MoE), attention_cte, 2-stage (transformer + VAE) subprocess pipeline — head-sharded attention + TP-aware global RMS (was replicated) |
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
