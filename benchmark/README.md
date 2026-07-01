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
  trn2/                      # measured here (Trainium trn2.3xlarge, Trainium2) — presharding default-on
    RESULTS.md               # cross-model summary for this device
    <slug>.json  <slug>.md   # machine-readable + detailed report
    logs/                    # raw compile/generate logs (gitignored)
  trn3/                      # Trainium trn3pd98.3xlarge (Trainium3); same layout (per-rank presharding A/B)
  h100/   b300/              # NVIDIA (diffusers CUDA reference); same layout
```

The runner writes to `benchmark/<device>/`; the device defaults to `trn2` and is set
with `DIFFLET_BENCH_DEVICE` (e.g. `DIFFLET_BENCH_DEVICE=h100 python -m benchmark.bench …`).
Every per-model report carries a **Reproduction** section with the exact,
hardware-agnostic test conditions (model id + pinned revision, shape, tp/cp, dtype,
steps, guidance, seed, prompt) and the precise commands + measurement protocol — so
H100/B300 can replicate the *same* run and compare against the trn2 numbers.

**[trn2/RESULTS.md](trn2/RESULTS.md)** — cross-model summary table (trn2, Trainium2;
**presharding default-on**).
**[trn3/RESULTS.md](trn3/RESULTS.md)** — cross-model summary + per-rank presharding A/B +
**trn3-vs-trn2 per-step** (trn3pd98.3xlarge, Trainium3, 144 GB).
**[h100/RESULTS.md](h100/RESULTS.md)** — cross-model summary + H100-vs-trn2 per-step
comparison (NVIDIA H100 PCIe 80 GB, stock-diffusers reference, single-GPU dense).
**[b300/RESULTS.md](b300/RESULTS.md)** — cross-model summary + B300-vs-H100-vs-trn2
per-step comparison (NVIDIA B300 SXM6 275 GB, stock-diffusers reference, single-GPU dense).

| model | slug | report (trn2) | report (trn3) | status |
|---|---|---|---|---|
| LTX-2 (video+audio) | `ltx_2` | [trn2](trn2/ltx_2.md) | [trn3](trn3/ltx_2.md) | see report |
| Wan 2.1 14B (T2V) | `wan_2_1` | [trn2](trn2/wan_2_1.md) | [trn3](trn3/wan_2_1.md) | see report |
| Wan 2.2 A14B (T2V) | `wan_2_2` | [trn2](trn2/wan_2_2.md) | — (shares 2.1 NEFF) | see report |
| Qwen-Image (T2I) | `qwen_image` | [trn2](trn2/qwen_image.md) | [trn3](trn3/qwen_image.md) | see report |
| HunyuanVideo (T2V) | `hunyuan_video` | [trn2](trn2/hunyuan_video.md) | [trn3](trn3/hunyuan_video.md) | see report |
| HunyuanVideo-1.5 (T2V) | `hunyuan_video_15` | [trn2](trn2/hunyuan_video_15.md) | — | pending (orchestrator stub) |
| FLUX.1-dev (T2I) | `flux_1_dev` | [trn2](trn2/flux_1_dev.md) | [trn3](trn3/flux_1_dev.md) | see report (gated; needs HF token) |

Each report records the exact config + pinned revision, phase timings, latency
distribution, compile breakdown, e2e cold/warm load split, output validity, toolchain
versions, and the full reproduction commands.

## Cross-device comparison

Trainium columns (**trn2** = Trainium2, **trn3** = Trainium3) are `tp=4` AOT-compiled with
**presharding on**; the GPU columns (**H100**, **B300**) are the **diffusers CUDA reference**
(`--backend cuda`, single-GPU dense eager, effective tp=1) at the same shape + step count +
pinned revision. Raw values only — `—` = not run on that device, `stub` = orchestrator not
implemented, `OOM` = out of memory, `N/A` = metric not exposed.

### e2e warm (s)

Warm generate, weights hot in the page cache (n=1). **Load-dominated** — read it as the
practical steady-state latency per device, not a silicon ranking (use per-step for that):
Trainium reloads weights onto the cores each generate, the GPUs reload the cached pipeline.

| model | trn2 | trn3 | H100 | B300 |
|---|---:|---:|---:|---:|
| FLUX.1-dev | 37.7 | 35.1 | 15.8 | 7.9 |
| LTX-2 | 61.7 | 57.7 | 24.9 | 12.7 |
| Wan 2.1 14B | 59.5 | 52.0 | 33.7 | 13.6 |
| Wan 2.2 A14B | 59.1 | — | 49.3 | 17.5 |
| Qwen-Image | 64.9 | 54.5 | 18.3 | 10.7 |
| HunyuanVideo | 177.6 | 159.7 | 47.6 | 27.8 |
| HunyuanVideo-1.5 | stub | — | OOM | 439.1 |

Trainium warm is with presharding on; the trn3 column is its clean ON arm with the
jemalloc allocator preloaded (default-on load path — see [trn3/RESULTS.md](trn3/RESULTS.md)).
HunyuanVideo-1.5
runs only on B300 (480×848×121 peaks at 99.2 GB — over the 80 GB H100; trn2/trn3 stub).

### DiT per-step (ms)

The load-independent compute metric — the cleanest cross-device comparison (mean
`step_latency` on Trainium; mean inter-step delta on the GPUs). Presharding-independent.

| model | trn2 | trn3 | H100 | B300 |
|---|---:|---:|---:|---:|
| FLUX.1-dev | 268.1 | 241.7 | 310.8 | 134.1 |
| LTX-2 | 437.9 | 345.0 | 313.1 | 159.5 |
| Wan 2.1 14B | 554.8 | 442.5 | 554.2 | 271.2 |
| Wan 2.2 A14B | 554.8 | — | 553.7 | 240.7 |
| Qwen-Image | 447.1 | 324.1 | 297.7 | 140.0 |
| HunyuanVideo | 850.6 | 650.0 | 1503.2 | 874.5 |
| HunyuanVideo-1.5 | stub | — | OOM | N/A |

trn3 FLUX/LTX per-step captured via the realloop method (241.7 / 345.0 ms, n=27/19). HunyuanVideo-1.5 per-step is
N/A (its diffusers pipeline exposes no `callback_on_step_end`). The trn2 numbers reflect two
correctness fixes that pulled it level on the heavy models — HunyuanVideo SDPA→attention_cte
(was 3719 ms) and Wan replicated→head-sharded attention (was 1144 ms); see **Corrections** in
[trn2/RESULTS.md](trn2/RESULTS.md) and [h100/RESULTS.md](h100/RESULTS.md).
