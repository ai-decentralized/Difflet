# Benchmark results — B300 (summary)

Measured on **NVIDIA B300 SXM6 (275 GB, Blackwell, sm_103)**, bf16, **single-GPU dense** via stock Hugging Face **diffusers** (eager — no AOT compile), through the backend-generic harness' CUDA reference adapter (`benchmark.bench --backend cuda`). The device was idle and serial — one model at a time. Run with `--iters 1` so each model reports **both** an e2e cold and one warm e2e iteration (n=1), matching the trn2 warm method.

**Input size (H×W×F) and step count are identical to the trn2 runs** (same `MATRIX` config + pinned HF revision); the adapter refuses to run any other shape. Video models use VAE tiling/slicing (the standard single-GPU decode setting — output-identical, DiT per-step unaffected; disabled for LTX-2, see ⁵). Each row links to a detailed report.

Toolchain: torch 2.9.1+cu128, diffusers 0.38.0, transformers 4.57.6, accelerate 1.14.0, CUDA 12.8 — **identical to the H100 reference**, so B300-vs-H100 is the same adapter, same method, same versions.

| model | kind | shape (==trn2) | steps | **e2e cold**¹ | **e2e warm**ᵉ | load² | **DiT per-step**³ | peak mem⁴ | output | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| [LTX-2](ltx_2.md) | video | 480×704×49 | 20 | 12.7 s | 12.7 s | 7.4 s | **159.5 ms** (6.27/s) | 76.7 GB | (1, 49, 3, 480, 704) ✓ | ok |
| [Wan 2.1 14B](wan_2_1.md) | video | 480×832×9 | 20 | 15.9 s | 13.6 s | 5.1 s | **271.2 ms** (3.69/s) | 42.3 GB | (1, 9, 3, 480, 832) ✓ | ok |
| [Wan 2.2 A14B](wan_2_2.md) | video | 480×832×9 | 20 | 16.0 s | 17.5 s | 8.0 s | **240.7 ms** (4.15/s) | 70.9 GB | (1, 9, 3, 480, 832) ✓ | ok |
| [FLUX.1-dev](flux_1_dev.md) | image | 1024×1024 | 28 | 8.1 s | 7.9 s | 3.2 s | **134.1 ms** (7.46/s) | 35.4 GB | (1, 3, 1024, 1024) ✓ | ok |
| [Qwen-Image](qwen_image.md) | image | 1024×1024 | 20 | 10.5 s | 10.7 s | 5.2 s | **140.0 ms** (7.14/s) | 58.3 GB | (1, 3, 1024, 1024) ✓ | ok |
| [HunyuanVideo](hunyuan_video.md) | video | 320×512×61 | 20 | 30.4 s | 27.8 s | 4.3 s | **874.5 ms** (1.14/s) | 59.5 GB | (1, 61, 3, 320, 512) ✓ | ok |
| [HunyuanVideo-1.5](hunyuan_video_15.md) | video | 480×848×121 | 20 | 438.8 s | 439.1 s | 3.9 s | **—**ᶠ | **99.2 GB** | (1, 121, 3, 480, 848) ✓ | **ok** (H100 OOM'd) |

✓ = output finite (no NaN/Inf), sensible range — see each report. Video output is frames-first `(B, F, C, H, W)` in diffusers vs trn2's `(B, C, F, H, W)`; same video.

## B300 vs H100 vs trn2 — DiT per-step (the comparable metric)

Same model, **same input size + step count**. The B300 and H100 per-steps are both the diffusers denoise-step latency from the *same* CUDA adapter (cuda-synced, first step dropped) — so **B300-vs-H100 is an identical-method, identical-version comparison**. The trn2 per-step is the matching real-loop inter-step delta (FLUX/LTX-2 via `step_realloop.py`) or the isolated DiT-forward timer (HunyuanVideo/Wan/Qwen), so **B300-vs-trn2 carries the same cross-method caveat as H100-vs-trn2**. `speedup` columns are `other ÷ B300` (**> 1 means B300 is faster**).

| model | **B300 per-step** | H100 per-step | trn2 per-step | B300 vs H100 | B300 vs trn2 |
|---|---:|---:|---:|---:|---:|
| FLUX.1-dev | **134.1 ms** | 310.8 ms | 267.6 ms | **2.32×** | **2.00×** |
| Qwen-Image | **140.0 ms** | 297.7 ms | 447.1 ms | **2.13×** | **3.19×** |
| LTX-2 | **159.5 ms** | 313.1 ms | 441.8 ms | **1.96×** | **2.77×** |
| Wan 2.2 A14B | **240.7 ms** | 553.7 ms | 554.8 ms | **2.30×** | **2.31×** |
| Wan 2.1 14B | **271.2 ms** | 554.2 ms | 554.8 ms | **2.04×** | **2.05×** |
| HunyuanVideo | **874.5 ms** | 1503.2 ms | 850.6 ms | **1.72×** | 0.97× (trn2 ≈ par) |
| HunyuanVideo-1.5 | —ᶠ (ran; H100 OOM) | OOM (>80 GB) | — (stub) | — | — |

### Headline

**B300 is ~2× faster than H100 across the board on the comparable DiT per-step, and it is the only one of the three that runs HunyuanVideo-1.5 at all.** Against the same stock-diffusers CUDA path (same adapter, same torch/diffusers versions), Blackwell-Ultra lands **2.0–2.3× ahead of Hopper (H100)** on every model with a per-step number — FLUX 2.32×, Wan 2.2 2.30×, Qwen 2.13×, Wan 2.1 2.04×, LTX-2 1.96× — and **1.72×** on HunyuanVideo. The one model where B300 is *not* ~2× ahead of trn2 is HunyuanVideo (0.97×, trn2 marginally faster): that is exactly the model the trn2 stack was hand-tuned on (masked joint-attn re-wired from SDPA to attention_cte, 3719→850.6 ms — see [trn2/RESULTS.md](../trn2/RESULTS.md) Corrections), so trn2's tuned kernel pulls level with an untuned eager-diffusers B300 run. Everywhere the GPU path is also "just eager diffusers," B300 doubles H100.

**HunyuanVideo-1.5 is the memory story.** At 480×848×**121 frames** its activation working set peaks at **99.2 GB** — over the 80 GB H100 (which OOM'd) but comfortable inside the B300's 275 GB. It produces a valid (1, 121, 3, 480, 848) clip in 438.8 s e2e. trn2 never ran it either (orchestrator stub), so **B300 is the only device in this matrix that completes it**. Per-step is N/Aᶠ — the HunyuanVideo-1.5 diffusers pipeline doesn't expose `callback_on_step_end`, so the adapter can't time individual steps (an adapter limitation, not a failure; no device has a per-step for this model).

**e2e cold/warm are *not* comparable across devices** (each reads just-downloaded weights from cache, not a true cold disk read; the CUDA adapter reloads the full pipeline every generate, so warm ≈ cold here) — only **per-step** is. They are reported for completeness and to match the trn2 schema.

¹ **e2e cold** = one full `from_pretrained` (weights→GPU) + generate, wall clock (eager reloads weights per process). ᵉ **e2e warm** = one extra full generate after the cold one (`--iters 1`, n=1), matching the trn2 warm method; for the eager CUDA adapter this reloads the pipeline too, so warm ≈ cold. ² **load** = the `from_pretrained(...).to(cuda)` portion. ³ **DiT per-step** = the load-independent compute metric, directly comparable across devices. ⁴ peak `torch.cuda.max_memory_allocated`. ⁵ **LTX-2**: VAE tiling disabled (`DIFFLET_BENCH_VAE_TILING=0`) — its small latents tile into a size-1 dim that crashes the decoder conv in diffusers 0.38.0. **Wan 2.2 A14B**: B300 loads BOTH 14B experts resident (70.9 GB peak); per-step compute is single-expert-equivalent (each expert is 14B). ᶠ **HunyuanVideo-1.5 per-step N/A**: its diffusers pipeline `__call__` has no `callback_on_step_end`, so the adapter's per-step timer never fires (clean run, no traceback; e2e and output are valid).

## Reproduce

```bash
source ~/.venvs/difflet-b300/bin/activate           # CUDA torch 2.9.1+cu128 + diffusers 0.38.0
export HF_HOME=/root/hf DIFFLET_BENCH_DEVICE=b300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=...                                  # for gated FLUX.1-dev
python -m benchmark.bench --backend cuda --model <slug> --iters 1          # e.g. flux_1_dev, wan_2_1
DIFFLET_BENCH_VAE_TILING=0 python -m benchmark.bench --backend cuda --model ltx_2 --iters 1
```

The full matrix was driven serially (one model at a time, HF cache pruned between models to fit the 422 GB disk) by `benchmark/b300/drive_all.sh`.
