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
| `step_latency` (Stats) | per denoising-step transformer-forward latency (core compute) |
| `throughput` | derived (e.g. steps/s) |
| `peak_device_mem_gb` | peak accelerator memory during inference |
| `output` | shape / dtype / finite (no NaN/Inf) / value-range of the result |

Timing uses warmup + N iters with percentile stats (`harness.Stats`); the Trainium
adapter additionally relies on difflet's internal warmup so `step_latency` reflects
steady-state device latency, not first-iteration cost.

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
  h100/   b300/   …          # future: same layout, one folder per device
```

The runner writes to `benchmark/<device>/`; the device defaults to `trn2` and is set
with `DIFFLET_BENCH_DEVICE` (e.g. `DIFFLET_BENCH_DEVICE=h100 python -m benchmark.bench …`).
Every per-model report carries a **Reproduction** section with the exact,
hardware-agnostic test conditions (model id + pinned revision, shape, tp/cp, dtype,
steps, guidance, seed, prompt) and the precise commands + measurement protocol — so
H100/B300 can replicate the *same* run and compare against the trn2 numbers.

**[trn2/RESULTS.md](trn2/RESULTS.md)** — cross-model summary table (trn2).

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
