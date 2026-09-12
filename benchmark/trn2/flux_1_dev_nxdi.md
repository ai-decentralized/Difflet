# Benchmark — black-forest-labs/FLUX.1-dev

**Status:** ok  
**Backend:** nxdi  
**Device:** trn2.3xlarge / 4 NeuronCores / 96 GB/device  
**Timestamp:** 2026-09-12 13:30 UTC

> Best-performing configuration: NATIVE NxDI baseline (neuronx_distributed_inference 0.10.18399+ed62453e), examples/generate_flux.py setup: backbone tp=4, world=4, bf16, CLIP tp=1, T5 tp=world, VAE decoder tp=1

## Configuration

| key | value |
|---|---|
| model type | flux |
| dtype | bf16 |
| parallel | tp=4 cp=1 |
| shape | {'height': 1024, 'width': 1024, 'num_frames': None} |
| steps | 28 |

## End-to-end performance

| phase | time |
|---|---|
| compile (AOT, one-time) | 15.1 min (908 s) |
| **e2e generate — cold start** (page cache dropped) | **5.5 min (330 s)** |
| &nbsp;&nbsp;↳ of which weights load (cold disk read) | 4.8 min (290 s) |
| **e2e generate — warm cache** | **56.51 s** |
| &nbsp;&nbsp;↳ of which weights load (from page cache) | 39.28 s |

> Cold vs warm: **5.5 min (330 s) → 56.51 s** (5.8× faster warm). e2e is load-dominated; the gap is the one-time cold disk read of the weights (warm = weights already in the OS page cache). The stable compute metric is the per-step latency below.

## Latency distribution

| metric | mean | median | p90 | min | n |
|---|---|---|---|---|---|
| per denoise step (transformer fwd) | 271.9 ms | 271.7 ms | 272.1 ms | 270.0 ms | 27 |
| end-to-end (warm) | 56.51 s | 56.51 s | 56.51 s | 56.51 s | 1 |

**Throughput:** 3.677 DiT steps/s

Per-step basis: **real-loop (NxDI backbone forward, synced)** — device-synced inter-step deltas of a real generate loop, step 0 excluded, the same rule the other device folders use (`benchmark/harness.py::RealLoopStepTimer`).

## Compile breakdown

| component | build time |
|---|---|
| host_pipeline_load_s | 1.82 s |
| wall_total_s | 15.1 min (908 s) |

## Output validity

| field | value |
|---|---|
| shape | None |
| dtype | None |
| finite (no NaN/Inf) | True |
| note | saved flux_1_dev_nxdi_out.png |

## Toolchain

- `torch` = 2.9.1
- `torch-neuronx` = 2.9.0.2.15.32035+de43f57c
- `neuronx-cc` = 2.26.6360.0+6f180f47
- `neuronx-distributed` = 0.19.28492+435aae2b
- `diffusers` = 0.38.0

## Notes

- Native NxDI baseline for the difflet FLUX tp4 row, same weights/shape/steps/seed/prompt/guidance. Measured with the campaign rules: compile = timed NeuronFluxApplication.compile into a fresh workdir; e2e cold = a fresh process after dropping the OS page cache, load + ONE generate, no warm-up; e2e warm = the next identical process; per-step = inter-step deltas of the backbone forward inside that real generate, step 0 excluded (n=27).
- NOT apples-to-apples with difflet: NxDI's NeuronFluxApplication loads the full diffusers pipeline on the host in every process (host_pipeline_load 2 s cold / 0 s warm, inside e2e) and load() runs a warm-up forward per component; difflet loads only the Neuron stages from presharded per-rank checkpoints. The DiT per-step is the comparable number.
- NxDI's own metric (examples/generate_flux.py 'Average generation time', resident model, after 5 warm-ups) corresponds to generate_s = 8.0 s here (1 generate, no warm-up).

## Reproduction

Exact test conditions. The **model + config rows are hardware-agnostic** — an H100/B300 (or any backend) must match these to reproduce; only the toolchain and the launch backend differ. The pinned HF `revision` fixes the exact weights.

| key | value |
|---|---|
| model id | `black-forest-labs/FLUX.1-dev` |
| HF revision (pinned) | `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` |
| model type | flux |
| dtype | bf16 |
| parallel | tp=4, cp=1 |
| shape (H×W×F) | 1024×1024 |
| steps | 28 |
| guidance scale | 3.5 |
| seed | 42 |
| prompt | "a cinematic shot of a red fox running through a snowy forest" |
| best-perf knobs | NATIVE NxDI baseline (neuronx_distributed_inference 0.10.18399+ed62453e), examples/generate_flux.py setup: backbone tp=4, world=4, bf16, CLIP tp=1, T5 tp=world, VAE decoder tp=1 |
| measured on | trn2.3xlarge / 4 NeuronCores / 96 GB/device (device folder `trn2`) |

```bash
# native NxDI (neuronx_distributed_inference) baseline, campaign rules:
source .venv/bin/activate
DIFFLET_BENCH_DEVICE=trn2 bash benchmark/trn2/nxdi_flux_baseline.sh
#   = python -m benchmark.nxdi_flux_baseline compile|generate|record (see its docstring)
```

**Measurement protocol**: see the Notes above — compile is a timed `NeuronFluxApplication.compile` into a fresh workdir; e2e cold/warm are one fresh process each (page cache dropped before the cold one), load + one generate, no warm-up; per-step = inter-step deltas of the NxDI backbone forward inside that generate, step 0 excluded.
