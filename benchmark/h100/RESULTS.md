# Benchmark results — H100 (summary)

Measured on **NVIDIA H100 PCIe (80 GB)**, bf16, **single-GPU dense** via stock Hugging Face **diffusers** (eager — no AOT compile), through the backend-generic harness' CUDA reference adapter (`benchmark.bench --backend cuda`). The device was idle and serial — one model at a time. Weights cached on a 738 GB scratch disk.

**Input size (H×W×F) and step count are identical to the trn2 runs** (same `MATRIX` config + pinned HF revision); the adapter refuses to run any other shape. Video models use VAE tiling/slicing (the standard single-GPU decode setting — output-identical, DiT per-step unaffected; disabled for LTX-2, see ⁵). Each row links to a detailed report.

| model | kind | shape (==trn2) | steps | **e2e cold**¹ | load² | **DiT per-step**³ | peak mem⁴ | output | status |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| [LTX-2](ltx_2.md) | video | 480×704×49 | 20 | **24.6 s** | 17.0 s | **319 ms** (3.14/s) | 76.7 GB | (1, 49, 3, 480, 704) ✓ | ok |
| [Wan 2.1 14B](wan_2_1.md) | video | 480×832×9 | 20 | **25.1 s** | 10.6 s | **563 ms** (1.78/s) | 42.3 GB | (1, 9, 3, 480, 832) ✓ | ok |
| [Wan 2.2 A14B](wan_2_2.md) | video | 480×832×9 | 20 | **39.9 s** | 25.2 s | **564 ms** (1.77/s) | 70.9 GB | (1, 9, 3, 480, 832) ✓ | ok |
| [FLUX.1-dev](flux_1_dev.md) | image | 1024×1024 | 28 | **15.2 s** | 5.7 s | **316 ms** (3.17/s) | 36.3 GB | (1, 3, 1024, 1024) ✓ | ok |
| [Qwen-Image](qwen_image.md) | image | 1024×1024 | 20 | **18.0 s** | 9.6 s | **302 ms** (3.32/s) | 58.3 GB | (1, 3, 1024, 1024) ✓ | ok |
| [HunyuanVideo](hunyuan_video.md) | video | 320×512×61 | 20 | **46.7 s** | 8.8 s | **1525 ms** (0.66/s) | 59.5 GB | (1, 61, 3, 320, 512) ✓ | ok |
| [HunyuanVideo-1.5](hunyuan_video_15.md) | video | 480×848×121 | 20 | **—** | — | — | — | — | **failed** — OOM (>80GB) |

✓ = output finite (no NaN/Inf), sensible range — see each report. Video output is frames-first `(B, F, C, H, W)` in diffusers vs trn2's `(B, C, F, H, W)`; same video.

## H100 vs trn2 — DiT per-step (the only directly comparable metric)

Same model, **same input size + step count**. The trn2 per-step is the warm in-process transformer-forward latency from its report; the H100 per-step is the diffusers denoise-step latency (cuda-synced, first step dropped). Speedup = trn2 ÷ H100.

| model | H100 per-step | trn2 per-step | H100 speedup |
|---|---:|---:|---:|
| LTX-2 | 319 ms | 473 ms | **1.5×** |
| Wan 2.1 14B | 563 ms | 1144 ms | **2.0×** |
| Wan 2.2 A14B | 564 ms | 1144 ms | **2.0×** |
| FLUX.1-dev | 316 ms | 266 ms | **0.8×** (trn2 faster) |
| Qwen-Image | 302 ms | 447 ms | **1.5×** |
| HunyuanVideo | 1525 ms | 3719 ms | **2.4×** |
| HunyuanVideo-1.5 | — | — | — |

### Headline

**It is model-dependent — not a blanket win for either side.** On the models where difflet is less tuned, the H100's mature CUDA path leads (HunyuanVideo 2.4×, Wan 2.0×, Qwen 1.5×); but on **FLUX.1-dev — difflet's flagship — trn2 is *faster* per-step (266 ms vs 316 ms, 0.8×)**, despite being 4×24 GB NeuronCores vs one 80 GB GPU. So the gap reflects **software-stack maturity as much as silicon**: where difflet's AOT NEFF is optimized it is competitive-to-faster; elsewhere stock diffusers' kernels win. **e2e cold is *not* comparable** across the two (trn2's is a true page-cache-dropped cold disk read; the H100's reads just-downloaded weights from cache) — only per-step is.

¹ **e2e cold** = one full `from_pretrained` (weights→GPU) + generate, wall clock (eager reloads weights per process). ² **load** = the `from_pretrained(...).to(cuda)` portion. ³ **DiT per-step** = the load-independent compute metric, directly comparable to trn2. ⁴ peak `torch.cuda.max_memory_allocated`. ⁵ **LTX-2**: VAE tiling disabled (`DIFFLET_BENCH_VAE_TILING=0`) — its small latents tile into a size-1 dim that crashes the decoder conv in diffusers 0.38.0. **HunyuanVideo-1.5** (480×848×121): OOM at default config — the 121-frame attention activation alone exceeds 80 GB (trn2 never ran it either; its orchestrator is a stub). **Wan 2.2 A14B**: H100 loads BOTH 14B experts resident (70.9 GB peak); trn2's per-step is single-expert but per-step compute is equivalent (each expert is 14B).

## Reproduce

```bash
source ~/.venvs/difflet-h100/bin/activate            # CUDA torch + diffusers
export HF_HOME=/ephemeral/hf DIFFLET_BENCH_DEVICE=h100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=...                                   # for gated FLUX.1-dev
python -m benchmark.bench --backend cuda --model <slug>          # e.g. flux_1_dev, wan_2_1
DIFFLET_BENCH_VAE_TILING=0 python -m benchmark.bench --backend cuda --model ltx_2
```
