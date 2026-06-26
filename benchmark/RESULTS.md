# Benchmark results — summary

Measured on **trn2.3xlarge** (1 Neuron device, 4 NeuronCores × 24 GB), bf16,
`tp=4`, via the Neuron inference venv
(`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`). All end-to-end numbers were
taken **clean** (no concurrent downloads) — the device is serial, so concurrent
I/O badly skews timings (an early contended LTX-2 run measured e2e 970 s vs the
clean 293 s below). Each row links to a detailed per-model report.

| model | kind | shape | compile¹ | weights load | **e2e generate (cold)** | **DiT per-step**⁰ | output | status |
|---|---|---|---:|---:|---:|---:|---|---|
| [LTX-2](ltx_2.md) | video+audio | 480×704×49 | 8.3 min | 82 s | **293 s** (4.9 min) | **473 ms** (2.11/s) | (1,49,3,480,704) ✓ | ok |
| [Wan 2.1 14B](wan_2_1.md) | video (T2V) | 480×832×9 | 108 min² | 15 s | **713 s** (11.9 min) | **1144 ms** (0.87/s) | (1,3,9,480,832) ✓ | ok |
| [Wan 2.2 A14B](wan_2_2.md) | video (T2V) | 480×832×9 | (shares 2.1)³ | 16 s | **616 s** (10.3 min) | **1144 ms** (0.87/s) | (1,3,9,480,832) ✓ | ok³ |
| [Qwen-Image](qwen_image.md) | image (T2I) | 1024×1024 | 13.4 min | 6 s | **496 s** (8.3 min) | **447 ms** (2.24/s) | (1,3,1024,1024) ✓ | ok |
| [HunyuanVideo](hunyuan_video.md) | video (T2V) | 320×512×61 | 40 min | 210 s | **527 s** (8.8 min) | **3719 ms** (0.27/s) | (1,3,61,320,512) ✓ | ok |
| [HunyuanVideo-1.5](hunyuan_video_15.md) | video (T2V) | 480×848×121 | — | — | — | — | — | pending⁴ |
| [FLUX.1-dev](flux_1_dev.md) | image (T2I) | 1024×1024 | — | — | — | — | — | pending⁵ |

✓ = output is finite (no NaN/Inf) with a sensible value range — see each report.

⁰ **DiT per-step** = warm in-process transformer-forward latency (median of n=20,
p90 within 1 ms of median for every model), the stable pure-Neuron-compute metric —
measured by `benchmark/step_latency.py`. We report this separately from e2e because
each `difflet generate` is a fresh process whose text-encoder/VAE load (5–11 GB)
swings with OS page-cache warmth, making e2e load-dominated and noisy across runs;
the per-step isolates the denoise compute. Multiply by `steps` for the Neuron
denoise-loop floor (e.g. LTX-2 0.473 s × 30 ≈ 14 s; HunyuanVideo 3.72 s × 30 ≈
112 s), the rest of e2e is host text-encode + VAE decode.

¹ One-time AOT compile (cached afterwards). ² Wan 2.1 compile is dominated by the
video VAE decoder (~100 min on neuronx-cc; the transformer is 431 s). ³ The wan
orchestrator keys its compile cache by *shape*, not model id, so Wan 2.2 reused
Wan 2.1's NEFF (the 14 s is a false cache hit); difflet also runs Wan 2.2 with
only the high-noise expert (`enable_transformer_2=False`), i.e. single-transformer
— see report caveats. ⁴ difflet's HunyuanVideo-1.5 orchestrator is a stub
(`NotImplementedError` at compile). ⁵ FLUX.1-dev is a gated HF repo; no token on
this box.

## Notes on "optimal end-to-end"

- **e2e cold** includes the one-time CPU host stages (text-encode, VAE decode) plus
  the Neuron denoise loop. For video models the CPU text-encoder load + VAE decode
  dominate (e.g. HunyuanVideo `load`=210 s); the Neuron transformer compute is a
  smaller fraction. The clean Neuron per-step (warm, in-process, n=20) is now
  measured for every model: **Qwen-Image 447 ms**, **LTX-2 473 ms** (attention_cte
  self-attn, validated lossless vs CPU, cosine 0.99992), **Wan 2.1 / 2.2 1144 ms**,
  **HunyuanVideo 3719 ms** — see the per-step column above and each report.
- These are the **best-performing configs currently available** on a single
  trn2.3xlarge: `tp=4` (the max with 4 cores), bf16, attention_cte where the model
  routes through it. On a trn2.48xlarge, higher `tp` (e.g. FLUX `tp=8`) and
  context-parallel would lift these further.

## Reproduce

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
python -m benchmark.bench --model <slug> --skip-download    # e.g. ltx_2, qwen_image
# or the whole matrix:
python -m benchmark.bench --all
```
