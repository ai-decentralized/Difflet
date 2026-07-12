# Difflet CLI vs Resident Serving Benchmark

Date: 2026-07-12  
Host: `16.26.177.239` (`trn2.3xlarge`, 4 logical NeuronCores, 96 GiB HBM)  
Source: current local worktree synced to `/home/ubuntu/Difflet`

## Fixed Parameters

| Parameter | Qwen | Flux |
|---|---:|---:|
| Model | `Qwen/Qwen-Image` | `black-forest-labs/FLUX.1-dev` |
| TP / CP | 4 / 1 | 4 / 1 |
| Shape | 1024x1024 | 1024x1024 |
| Steps | 4 | 4 |
| Guidance | 4.0 | 3.5 |
| Seed | 42 | 42 |
| Prompt | `A small red sailboat on a calm blue lake, clean studio illustration` | same |

The serving measurement covers the HTTP request through inference, PNG encoding,
R2 upload, and URL signing. The separate R2 download used to validate each image
is not included. Each CLI measurement starts a new `difflet generate` process and
includes model loading, Neuron initialization/warmup, inference, and local PNG
write. One-time CLI compilation is reported separately.

## Results

| Model | Mode | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---:|---:|---:|---:|
| Qwen | Resident serving HTTP | 4.804s | 4.805s | 4.463s | **4.691s** |
| Qwen | CLI, independent process | 340.04s | 77.18s | 77.73s | **164.98s** |
| Flux | Resident serving HTTP | 3.766s | 4.093s | 3.613s | **3.824s** |
| Flux | CLI, independent process | 107.53s | 49.33s | 48.21s | **68.36s** |

The first CLI process includes a cold OS/page-cache state. Using runs 2 and 3 as
the repeated-process steady state:

| Model | CLI runs 2-3 mean | Serving mean | Serving speedup |
|---|---:|---:|---:|
| Qwen | 77.455s | 4.691s | **16.5x** |
| Flux | 48.770s | 3.824s | **12.8x** |

Including the first CLI process, the mean speedups are 35.2x for Qwen and 17.9x
for Flux. The operational comparison should use the steady-state figures above;
resident serving is intentionally already loaded, while CLI always creates a new
process.

## 20-Step CLI And Serving Rerun

A follow-up run used the same TP4/CP1, 1024x1024, seed 42 profile with 20 inference
steps, guidance 3.5, and prompt `a small red sailboat on a calm blue lake`. Each
CLI measurement is a new `difflet generate` process. Each serving measurement is
the complete localhost HTTP request, including validation, resident inference, PNG
encoding, R2 upload, and public URL construction.

| Model | Mode | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---:|---:|---:|---:|
| Qwen | Resident serving HTTP | 14.804s | 9.267s | 9.322s | **11.131s** |
| Qwen | CLI, independent process | 83.37s | 82.70s | 80.74s | **82.27s** |
| Flux | Resident serving HTTP | 6.833s | 6.321s | 6.324s | **6.493s** |
| Flux | CLI, independent process | 109.73s | 53.70s | 54.10s | **72.51s** |

Runs 2 and 3 remove the most obvious first-request/process cold effects:

| Model | CLI runs 2-3 mean | Serving runs 2-3 mean | Serving speedup |
|---|---:|---:|---:|
| Qwen | 81.720s | 9.294s | **8.79x** |
| Flux | 53.900s | 6.322s | **8.53x** |

The Qwen serving worker itself was stable at 8.982-9.000s for all three requests.
Its first HTTP request spent an additional approximately 5.29s lazily loading the
parent-side request-validator tokenizer; runs 2 and 3 did not repeat that cost.
Flux's 109.73s first CLI process reproduced the earlier cold page-cache outlier,
while its two warm independent processes were within 0.4s of each other.

The detailed serving phases were:

| Model | Phase | Run 1 | Run 2 | Run 3 |
|---|---|---:|---:|---:|
| Qwen | Resident engine | 8.999s | 8.985s | 8.982s |
| Qwen | R2 upload + URL | 0.513s | 0.279s | 0.338s |
| Flux | Resident engine | 6.045s | 6.033s | 6.031s |
| Flux | R2 upload + URL | 0.637s | 0.286s | 0.290s |

The R2 bucket for this rerun was located in Oceania near the `ap-southeast-4`
Trainium host. Uploading the 0.74-0.95 MiB PNGs took 0.28-0.64s, compared with
2.85-2.96s when the previous bucket was in Western North America. Public URL
construction itself took only 0.02-0.04ms.

Both models produced deterministic output within each mode. Flux CLI and serving
were byte-identical with SHA256
`fcffdb037da9045076d557296496c20d3aa0bbe5c7deafcbd56054691972f5fa`.
Qwen CLI and serving were internally deterministic but differed by approximately
0.499/255 mean absolute RGB channel value because the CLI and resident serving VAE
topologies differ. All 12 downloaded images are valid 1024x1024 RGB PNGs.

This run also created new immutable serving generations. Preparation plus load and
startup smoke took approximately 9 minutes for Qwen and 3 minutes 40 seconds for
Flux, with the underlying Neuron compiler cache hit. These one-time startup costs
are excluded from HTTP latency and will be reused by later warm starts.

The complete follow-up evidence is under:

```text
artifacts/remote-logs/16.26.177.239/cli-vs-serve/benchmark-cli-20steps-20260712T074956Z/
```

## Comparison With The Previous Flux Run

The 2026-07-10 Flux benchmark used the same TP4/CP1/1024 profile but 28 steps. Its
three CLI runs were 48.81s, 44.66s, and 45.67s (46.38s mean). The current warm
runs are 49.33s and 48.21s (48.77s mean), only about 5.2% slower and therefore
consistent with the earlier result. The current 107.53s first run is the outlier:
it was the first load after creating a new CLI artifact and paid cold filesystem/
page-cache costs. The earlier benchmark ran after compile and other model work on
an already warm host, so it did not capture an equivalent cold first process.

The previous Qwen record did not measure complete staged CLI wall time. It only
reported resident stage loads (12.32s text, 16.85s denoiser, 2.11s VAE), a 2.80s
four-step in-memory smoke, and an approximately 12.5s external HTTP request. Those
resident measurements are not comparable to a staged CLI invocation, which loads
and tears down three independent processes. In the current warm CLI runs, text,
denoiser, and VAE weight loading alone consumed roughly 13s, 29s, and 9s before
the remaining process, tokenization, inference, handoff, decode, and file costs.

## One-Time CLI Preparation

The CLI artifact cache was initially empty, but model weights and the Neuron
compiler cache had already been warmed by serving.

| Model | CLI compile wall time | Details |
|---|---:|---|
| Qwen | **327.37s** | text 34.56s, generate 260.11s, VAE 12.00s; cached NEFFs used |
| Flux | **881.95s** | CLIP 17.67s, T5 6.89s, transformer 155.48s, decoder 694.08s |

Flux transformer and decoder produced different HLO module identities from the
serving compile and were recompiled. Serving immutable generations and legacy
CLI cache entries are separate artifact contracts; a serving compile does not
currently make CLI artifacts directly reusable.

## Output Validation

- Every measured output is a valid 1024x1024 RGB PNG.
- Every repeated run within a model/mode is byte deterministic.
- Flux CLI and serving output are byte-identical:
  `807706358b3d8ecd0a62c6c126a7d9bcb8d29b59da175ae0b712989f8d1a00dd`.
- Qwen CLI and serving are each internally byte-identical, but differ across
  modes. Mean absolute RGB channel difference is approximately 0.5/255. Qwen CLI
  uses its public default single-core VAE while serving uses a TP4 resident VAE,
  so minor rounding differences are expected despite identical public request
  parameters.
- Average validation-only R2 download time was 0.681s for Qwen and 0.662s for
  Flux; it is excluded from the serving latency table.

## Environment Loading Validation

`difflet serve` now loads `.env` from the current working directory with
`override=False`, so exported process environment variables retain precedence.
The final Flux process was started without `source .env`, passed real startup
smoke, and returned health/ready 200. A sanitized `.env.example` documents all
required and optional R2 variables.

## Local Evidence

All remote evidence is stored under:

```text
artifacts/remote-logs/16.26.177.239/
```

Key benchmark files:

```text
cli-vs-serve/qwen-http-20260712T022453Z/summary.json
cli-vs-serve/qwen-cli-compile-20260712T022556Z.{log,status}
cli-vs-serve/qwen-cli-generate-20260712T023148Z/summary.tsv
cli-vs-serve/flux-serve-runs.txt
cli-vs-serve/flux-cli-compile-20260712T024024Z.{log,status}
cli-vs-serve/flux-cli-generate-20260712T025537Z/summary.tsv
cli-vs-serve/flux-dotenv-final-20260712T031433Z.log
cli-vs-serve/flux-dotenv-validation.png
```

The failed pre-fix Flux restart that omitted R2 environment loading is retained
as `flux-serve-after-cli-20260712T030326Z.log`. A later duplicate startup failed
because another worker already owned all four cores; the final state was cleaned
to one Flux parent/worker pair before the automatic `.env` validation.
