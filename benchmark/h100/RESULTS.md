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

Same model, **same input size + step count**. The H100 per-step is the diffusers
denoise-step latency (cuda-synced, first step dropped); the trn2 per-step is the matching
real-loop inter-step delta (FLUX/LTX-2 via `step_realloop.py`) or the isolated DiT-forward
timer (HunyuanVideo/Wan/Qwen). `trn2 speedup` = H100 ÷ trn2 (**> 1 means trn2 is faster**).

| model | H100 per-step | trn2 per-step | trn2 speedup |
|---|---:|---:|---:|
| LTX-2 | 319 ms | 477 ms | 0.67× (H100 faster) |
| Qwen-Image | 302 ms | 447 ms | 0.68× (H100 faster) |
| Wan 2.1 14B | 563 ms | 554.8 ms | **1.01×**ᵈ |
| Wan 2.2 A14B | 564 ms | 554.8 ms | **1.02×**ᵈ |
| FLUX.1-dev | 316 ms | 267.6 ms | **1.18×** |
| HunyuanVideo | 1525 ms | 850.6 ms | **1.79×**ᶜ |
| HunyuanVideo-1.5 | — | — | — |

### Headline

**It is software-stack maturity, not silicon — and the corrected numbers make that sharper.** Two trn2 per-steps were mis-measured or mis-configured, both in difflet's favor once fixed. **HunyuanVideo's 2.4× "H100 win" was a difflet bug**: its masked joint self-attn fell back to `F.scaled_dot_product_attention` instead of attention_cte (the `config_label` said attention_cte; the compiled graph used SDPA — commit `cd54d0f` had dropped the mask→bounds wiring). Re-wiring the key-padding mask to attention_cte's `bound_min`/`bound_max` flips it to **trn2 1.79× faster (1525 → 850.6 ms)ᶜ**. On the flagship **FLUX.1-dev trn2 is faster** (267.6 vs 316 ms, trn2 speedup 1.18×), despite 4×24 GB NeuronCores vs one 80 GB GPU. **Wan was the same story**: its 2.0× gap was difflet *replicating* the attention across the 4 TP cores (`gather_output=True` for HF qk_norm parity) instead of head-sharding it; head-sharding it (with a TP-aware global RMS) gives **2.06× and pulls trn2 level with H100 (554.8 vs 563 ms)ᵈ**, parity cosine 0.9998. So where difflet routes through attention_cte and shards properly, trn2 is competitive-to-faster (FLUX, HunyuanVideo, Wan); the residual H100 leads — LTX-2 1.50× (its cross-attn is still SDPA), Qwen 1.5× (not yet re-examined) — are smaller and model-specific, not a silicon verdict. **e2e cold is *not* comparable** across the two (trn2's is a true page-cache-dropped cold disk read; the H100's reads just-downloaded weights from cache) — only per-step is. trn2 per-step is now measured the H100 way (real-loop inter-step deltas)ᵇ for FLUX/LTX; **Qwen/Wan are still the older isolated-timer numbers**.

¹ **e2e cold** = one full `from_pretrained` (weights→GPU) + generate, wall clock (eager reloads weights per process). ² **load** = the `from_pretrained(...).to(cuda)` portion. ³ **DiT per-step** = the load-independent compute metric, directly comparable to trn2. ⁴ peak `torch.cuda.max_memory_allocated`. ⁵ **LTX-2**: VAE tiling disabled (`DIFFLET_BENCH_VAE_TILING=0`) — its small latents tile into a size-1 dim that crashes the decoder conv in diffusers 0.38.0. **HunyuanVideo-1.5** (480×848×121): OOM at default config — the 121-frame attention activation alone exceeds 80 GB (trn2 never ran it either; its orchestrator is a stub). **Wan 2.2 A14B**: H100 loads BOTH 14B experts resident (70.9 GB peak); trn2's per-step is single-expert but per-step compute is equivalent (each expert is 14B).

ᵇ trn2 per-step re-measured the H100 way — inter-step deltas of a real generate loop, device-synced, step 0 excluded (`benchmark/step_realloop.py`), replacing the earlier per-model mix (isolated synthetic-forward / n=1 parity / tqdm-rate). FLUX 266→267.6 ms (n=27), LTX-2 473→477 ms (n=19): the old values were ~right, only the method was inconsistent. ᶜ **HunyuanVideo correction**: 3719→850.6 ms (4.37×) — was SDPA, now attention_cte via the key-padding→bounds re-wire (CPU-validated lossless). Measured with the step_latency method (synthetic all-ones mask, `bound_max`=full seq) → a conservative upper bound; a real padded prompt attends fewer keys, so the H100 lead flips by at least 1.79×. trn2 also now compiles HunyuanVideo's **VAE on-chip** (the old e2e decoded it on host), so e2e is not yet re-measured. ᵈ **Wan correction**: 1144→554.8 ms (2.06×) — attention was replicated across the 4 TP cores (`gather_output=True` + full-width across-heads RMSNorm), now head-sharded with a TP-aware global RMS (cross-rank sum-of-squares + per-rank norm-weight slice). Strict parity vs the replicated baseline on the same input: cosine 0.999768. Measured with the isolated step_latency timer; Wan 2.2 (single high-noise expert) inherits Wan 2.1's per-step.

## Reproduce

```bash
source ~/.venvs/difflet-h100/bin/activate            # CUDA torch + diffusers
export HF_HOME=/ephemeral/hf DIFFLET_BENCH_DEVICE=h100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=...                                   # for gated FLUX.1-dev
python -m benchmark.bench --backend cuda --model <slug>          # e.g. flux_1_dev, wan_2_1
DIFFLET_BENCH_VAE_TILING=0 python -m benchmark.bench --backend cuda --model ltx_2
```
