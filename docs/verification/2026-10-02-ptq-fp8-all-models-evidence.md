# FP8 PTQ across the Difflet models — on-device evidence

Campaign: 2026-10-02, host `trn2.3xlarge` (1 Neuron device, 4 NeuronCores, 96 GB HBM,
LNC=2, 12 vCPU, 124 GB RAM, 1.5 TB disk), branch `quantization`.
Design: `docs/superpowers/specs/2026-10-02-ptq-fp8-all-models-design.md`.
Plan: `docs/superpowers/plans/2026-10-02-ptq-fp8-all-models.md`.
Predecessor (Wan 2.1, where the core was fixed on the device):
`docs/verification/2026-10-01-ptq-fp8-wan-evidence.md`.

Scope: the FP8 PTQ core that Wan 2.1 proved (per-tensor e4m3 weights at the Trainium 240
range, Difflet's own per-tensor dynamic activation path, bf16-typed quantized layers,
schema-keyed artifacts) wired to **FLUX.1-dev, Qwen-Image, LTX-2 (single mode),
HunyuanVideo 1.0**, and **Wan 2.2** (A14B, two experts) on the existing Wan wiring. Each
model is verified on the device in three arms through the benchmark harness —
**bf16**, **fp8 weight-only** (`--quant fp8 --quant-act none`) and **fp8 dynamic**
(`--quant fp8`, per-tensor dynamic activations) — with the same prompt / seed / steps, then
the fp8 outputs are compared against the bf16 output (PSNR / SSIM / LPIPS).

Curated evidence (harness reports, logs, compare JSONs, rendered outputs; no `.pt`
latents, no weights, no NEFFs) lives under `artifacts/verification-2026-10-02/ptq-all/<slug>/`.
Every number below is transcribed from a file in that directory or from a quoted command;
inferences are labelled. Per-model tables are rendered by `scripts/ptq_model_section.py`.

## Toolchain

Same venv as the Wan 2.1 campaign (from the harness reports' `toolchain` field):

| component | version |
|---|---|
| Python | 3.12.3 |
| torch / torch-neuronx | 2.9.1 / 2.9.0.2.15.32035+de43f57c |
| neuronx-cc | 2.26.6360.0+6f180f47 |
| neuronx-distributed | 0.19.28492+435aae2b |
| diffusers | 0.38.0 |

neuronx-cc 2.27 was tried and parked (pip `ResolutionImpossible` against this SDK set).

## Runner

`scripts/ptq_model_verify.sh <bf16-slug>` (gate on an idle host → `difflet quantize` →
for each arm `benchmark.bench --skip-download --iters 1` and `benchmark.cold_warm_e2e` →
`scripts/ptq_compare_outputs.py` vs the bf16 output → copy reports / outputs / store listing
into the evidence dir). The five runs were chained on the device in the order FLUX,
Qwen-Image, LTX-2, HunyuanVideo, Wan 2.2 (one run at a time; each bf16 compile was cold on
this host). Idle gate output per model: `artifacts/.../<slug>/gate.txt`.

What each arm measures: `compile wall s` = the harness's compile stage (cold for bf16; the
fp8 arms recompile only the transformer and reuse the shared text-encoder / VAE stages);
`e2e cold s` = first generation after a page-cache drop (weights read from disk);
`e2e warm s` = median of the warm repeats; `transformer load cold s` = the transformer
stage's weight load inside the cold e2e; `DiT step ms` = per-step latency from the
`--iters 1` bench (median / mean over the steps after the first); quality is the fp8 output
vs the bf16 output of the same run configuration.

## Results

_Per-model sections are appended below as each device run completes._

