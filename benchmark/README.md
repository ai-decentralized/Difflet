# Difflet diffusion benchmark

A **backend-generic** benchmark harness for the diffusion models difflet supports.
It measures end-to-end compile + inference performance with one set of universal
metrics, so results are directly comparable across accelerators (Trainium today;
CUDA/CPU via the reference adapter, and any future backend by adding one adapter).

## Design — why it's backend-generic

The harness (`harness.py`) defines a tiny contract and a fixed metric schema; it
imports **no** torch and **no** backend SDK. Each backend ships an *adapter* that
implements four hooks:

```
BackendAdapter
├── device_info()                      -> str        # accelerator description
├── toolchain()                        -> dict        # versions (provenance)
├── prepare(spec)                                     # download weights
├── compile(spec) -> (seconds, breakdown)             # AOT build (0 for eager backends)
└── run_generate(spec) -> {wall_seconds, load_seconds, step_seconds[], output, ...}
```

The runner (`bench.py`) calls those hooks the same way for every backend and emits
the **same** metric set. Adding a backend = adding one file under `adapters/`; the
metrics, timing, stats, and report renderer are shared and untouched.

Adapters provided:
- `adapters/trainium.py` — drives the `difflet` CLI (compile/generate) in the
  Neuron inference venv and parses difflet's emitted timings + subprocess wall-clock.
- `adapters/diffusers_ref.py` — stock Hugging Face diffusers on CPU/CUDA (eager,
  `compile_seconds=0`); the apples-to-apples reference that proves the harness is
  not Trainium-specific.

## Universal metrics

| metric | meaning |
|---|---|
| `compile_seconds` (+breakdown) | one-time AOT build/trace/compile (0 for eager backends) |
| `load_seconds` | weights load onto the device (per process) |
| `e2e_cold_seconds` | first full prompt→output generate (incl. one-time host costs) |
| `e2e_warm` (Stats) | steady-state generate over N iters (`--iters`) |
| `step_latency` (Stats) | per denoising-step DiT latency (core compute) — the metric directly comparable across backends |
| `throughput` | derived (e.g. steps/s) |
| `peak_device_mem_gb` | peak accelerator memory during inference |
| `output` | shape / dtype / finite (no NaN/Inf) / value-range of the result |

Timing uses warmup + N iters with percentile stats (`harness.Stats`).

**Per-step is measured the H100 way** (`benchmark/step_realloop.py`): the inter-step
deltas of a *real* generate loop (device-synced, step 0 dropped) — the same quantity the
diffusers CUDA reference takes via `callback_on_step_end`, so the cross-device numbers are
apples-to-apples. This replaces an earlier per-model mix (isolated synthetic-input forward
/ n=1 parity script / tqdm denoise-rate). `benchmark/step_latency.py` is the older
isolated-forward timer, kept for the masked models' before/after. See the **Corrections**
section in [trn2/RESULTS.md](trn2/RESULTS.md) for which numbers changed and why.

```bash
python -m benchmark.step_realloop --model flux_1_dev   # H100-consistent per-step (real loop)
```

## Usage

All runs use the Neuron inference venv:

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

python -m benchmark.bench --model ltx_2                  # one model (download+compile+generate)
python -m benchmark.bench --model ltx_2 --skip-download  # weights already local
python -m benchmark.bench --model ltx_2 --skip-compile   # reuse an existing compile cache
python -m benchmark.bench --all                          # the whole matrix
python -m benchmark.bench --model flux_1_dev --backend diffusers   # CPU/CUDA reference
```

The **best-performing configuration** per model (shape, tp/cp, dtype, steps,
guidance, **pinned HF revision**, seed) lives in `models.py::MATRIX` — edit there to
retune. Shapes default to what fits a single **trn2.3xlarge** (1 Neuron device, 4
cores × 24 GB), so `tp=4` is the max (FLUX's registry default of tp=8 is overridden
to 4).

## Results are namespaced per hardware

Each hardware target gets its own folder so reproductions sit side-by-side:

```
benchmark/
  trn2/                      # measured here (Trainium trn2.3xlarge)
    RESULTS.md               # cross-model summary for this device
    <slug>.json  <slug>.md   # machine-readable + detailed report
    logs/                    # raw compile/generate logs (gitignored)
  h100/   b300/              # measured here too (NVIDIA, diffusers CUDA reference); same layout
```

The runner writes to `benchmark/<device>/`; the device defaults to `trn2` and is set
with `DIFFLET_BENCH_DEVICE` (e.g. `DIFFLET_BENCH_DEVICE=h100 python -m benchmark.bench …`).
Every per-model report carries a **Reproduction** section with the exact,
hardware-agnostic test conditions (model id + pinned revision, shape, tp/cp, dtype,
steps, guidance, seed, prompt) and the precise commands + measurement protocol — so
H100/B300 can replicate the *same* run and compare against the trn2 numbers.

**[trn2/RESULTS.md](trn2/RESULTS.md)** — cross-model summary table (trn2).
**[h100/RESULTS.md](h100/RESULTS.md)** — cross-model summary + H100-vs-trn2 per-step
comparison (NVIDIA H100 PCIe 80 GB, stock-diffusers reference, single-GPU dense).
**[b300/RESULTS.md](b300/RESULTS.md)** — cross-model summary + B300-vs-H100-vs-trn2
per-step comparison (NVIDIA B300 SXM6 275 GB, stock-diffusers reference, single-GPU dense).

| model | slug | report (trn2) | status |
|---|---|---|---|
| LTX-2 (video+audio) | `ltx_2` | [trn2/ltx_2.md](trn2/ltx_2.md) | see report |
| Wan 2.1 14B (T2V) | `wan_2_1` | [trn2/wan_2_1.md](trn2/wan_2_1.md) | see report |
| Wan 2.2 A14B (T2V) | `wan_2_2` | [trn2/wan_2_2.md](trn2/wan_2_2.md) | see report |
| Qwen-Image (T2I) | `qwen_image` | [trn2/qwen_image.md](trn2/qwen_image.md) | see report |
| HunyuanVideo (T2V) | `hunyuan_video` | [trn2/hunyuan_video.md](trn2/hunyuan_video.md) | see report |
| HunyuanVideo-1.5 (T2V) | `hunyuan_video_15` | [trn2/hunyuan_video_15.md](trn2/hunyuan_video_15.md) | pending (orchestrator stub) |
| FLUX.1-dev (T2I) | `flux_1_dev` | [trn2/flux_1_dev.md](trn2/flux_1_dev.md) | see report (gated; needs HF token) |

Each report records the exact config + pinned revision, phase timings, latency
distribution, compile breakdown, e2e cold/warm load split, output validity, toolchain
versions, and the full reproduction commands.

### H100 reference (NVIDIA H100 PCIe 80 GB)

Reproduced via the **diffusers CUDA reference adapter** (`--backend cuda`), at the
**same input size + step count + pinned revision** as trn2. The only directly
comparable metric is the **DiT per-step** (load-independent compute); e2e cold is
*not* comparable (trn2's is a true cold disk read, the H100's reads cached weights).
See **[h100/RESULTS.md](h100/RESULTS.md)** for the full table and caveats.

`trn2 speedup` = H100 ÷ trn2 per-step (**> 1 means trn2 is faster**).

| model | DiT per-step — H100 | DiT per-step — trn2 | trn2 speedup |
|---|---:|---:|---:|
| Qwen-Image | 297.7 ms | 447.1 ms | 0.67× (H100 faster) |
| LTX-2 | 313.1 ms | 441.8 ms | 0.71× (H100 faster) |
| Wan 2.1 14B | 554.2 ms | 554.8 ms | **1.00×** (≈par) |
| Wan 2.2 A14B | 553.7 ms | 554.8 ms | **1.00×** (≈par) |
| FLUX.1-dev | 310.8 ms | 267.6 ms | **1.16×** |
| HunyuanVideo | 1503.2 ms | 850.6 ms | **1.77×** |
| HunyuanVideo-1.5 | OOM (>80 GB) | — (stub) | — |

### B300 reference (NVIDIA B300 SXM6 275 GB)

Reproduced via the **same diffusers CUDA reference adapter** (`--backend cuda`), at the
**same input size + step count + pinned revision + toolchain** as H100 (torch 2.9.1+cu128,
diffusers 0.38.0) — so **B300-vs-H100 is an identical adapter/method/version comparison**.
Run with `--iters 1` (e2e cold + one warm iter, matching the trn2 warm method). Full table
and caveats in **[b300/RESULTS.md](b300/RESULTS.md)**.

`B300 speedup` columns = `other ÷ B300` per-step (**> 1 means B300 is faster**).

| model | DiT per-step — B300 | vs H100 | vs trn2 |
|---|---:|---:|---:|
| FLUX.1-dev | **134.1 ms** | 2.32× | 2.00× |
| Qwen-Image | **140.0 ms** | 2.13× | 3.19× |
| LTX-2 | **159.5 ms** | 1.96× | 2.77× |
| Wan 2.2 A14B | **240.7 ms** | 2.30× | 2.31× |
| Wan 2.1 14B | **271.2 ms** | 2.04× | 2.05× |
| HunyuanVideo | **874.5 ms** | 1.72× | 0.97× (trn2 ≈ par) |
| HunyuanVideo-1.5 | — (ran; per-step N/A) | H100 **OOM** | trn2 stub |

**B300 is ~2× faster than H100 across the board on the comparable per-step, and the only
device here that runs HunyuanVideo-1.5** (480×848×121 peaks at **99.2 GB** — over the 80 GB
H100, inside the 275 GB B300; trn2 never ran it). The lone exception to the ~2× gap is
HunyuanVideo, where trn2's hand-tuned attention_cte kernel (the model the trn2 stack was
tuned on) pulls level with an untuned eager-diffusers B300 run (0.97×). Per-step for
HunyuanVideo-1.5 is N/A — its diffusers pipeline exposes no `callback_on_step_end`.

**It is software-stack maturity, not silicon — two apparent H100 wins were difflet bugs.**
HunyuanVideo and Wan originally looked 2.0–2.4× behind H100; both were difflet
inefficiencies, not the chip: **HunyuanVideo's masked joint-attn had silently fallen back
to SDPA instead of attention_cte** (3719→850.6 ms once re-wired to the kernel's
bound_min/bound_max), and **Wan ran its attention *replicated* across the 4 TP cores**
instead of head-sharding it (1144→554.8 ms once sharded, parity cosine 0.9998 vs the
replicated baseline). After those fixes trn2 is **competitive-to-faster on FLUX,
HunyuanVideo, and Wan**; the residual H100 leads — Qwen 1.5× and LTX-2 1.41× — are
smaller and model-specific (LTX-2's text cross-attn was also moved off SDPA to unmasked
attention_cte, lossless parity cosine 0.999934, but it was only ~7% of per-step, so the
residual is genuine self-attn+FFN compute like Qwen). See **Corrections** in
[trn2/RESULTS.md](trn2/RESULTS.md) / [h100/RESULTS.md](h100/RESULTS.md) for the old
numbers and exactly why each changed. (HunyuanVideo-1.5's 121-frame attention exceeds
80 GB at default config; trn2 never ran it either — orchestrator stub.)

### e2e warm — trn2 vs H100 vs B300

End-to-end **warm** generate (weights served from the OS page cache, n=1) on each device,
same MATRIX config + pinned revision. Unlike per-step, **e2e warm is load-dominated, not a
clean compute comparison**: every backend reloads the full pipeline each generate. On trn2
the warm e2e is dominated by the per-process **Neuron weight-load** (each generate reloads
the weights onto the NeuronCores); the denoise compute is small. The GPUs likewise reload
the full pipeline from cache and run it on-device. Read it as the practical steady-state
latency per device, **not** a silicon ranking — that is the per-step table above.

| model | trn2 warm | H100 warm | B300 warm |
|---|---:|---:|---:|
| FLUX.1-dev | 46.7 s | 15.8 s | 7.9 s |
| Qwen-Image | 73.6 s | 18.3 s | 10.7 s |
| LTX-2 | 102.7 s | 24.9 s | 12.7 s |
| Wan 2.1 14B | 96.8 s | 33.7 s | 13.6 s |
| Wan 2.2 A14B | 92.8 s | 49.3 s | 17.5 s |
| HunyuanVideo | 220.2 s | 47.6 s | 27.8 s |
| HunyuanVideo-1.5 | — (stub) | OOM (>80 GB) | 439.1 s |

trn2's warm e2e is the largest of the three purely because its per-process **Neuron
weight-load** dominates (the denoise compute is small: per-step × steps). Every component —
text-encoder, transformer **and** VAE — is compiled and runs on the NeuronCores (no host
stages), and the difflet CLI starts a fresh process per generate, so each warm run reloads
all weights onto the cores. The HunyuanVideo 220 s is a **stale** outlier: it was measured
when its VAE still decoded on the host (~185 s); difflet now compiles HunyuanVideo's VAE
on-chip too, so that e2e is pending re-measure (see trn2 Corrections). On the GPUs warm ≈
one cached full-pipeline reload + the denoise loop. **HunyuanVideo-1.5 runs only on the
B300** (480×848×121 peaks at 99.2 GB — over the 80 GB H100; trn2's orchestrator is a stub).
