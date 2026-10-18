# Difflet Developer Guide

This guide covers Difflet's internals: the architecture, the backend-neutral op surface, the
compile cache, parallelism, the runtime protocol, performance tooling, and how to port a new
model. For installation and CLI usage, see the [README](README.md).

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [The op surface and backends](#the-op-surface-and-backends)
- [Compile cache](#compile-cache)
- [Parallelism](#parallelism)
- [Runtime protocol](#runtime-protocol)
- [Performance and profiling](#performance-and-profiling)
- [Adding a new model](#adding-a-new-model)
- [Development workflow](#development-workflow)

## Overview

Difflet turns a Hugging Face diffusion model id into a generated image or video on AWS Trainium.
The flow is the same for every model:

```
download weights → AOT-compile NEFFs (cache on disk) → load + denoise → decode
```

A single public entry point — `DiffletPipeline` — resolves the model registry, manages the
compile cache, and dispatches to a per-model `Application`. The CLI (`difflet/cli/`) wraps the
same applications with subprocess orchestration for models whose stages cannot co-fit on one
card. Model code never touches Trainium APIs directly: it imports only from `difflet.ops`, a
frozen backend-neutral op surface, and the real Trainium implementations live under
`difflet/backends/trainium/`.

## Architecture

```
User script / difflet CLI
   │
   ▼
DiffletPipeline             single public entry; resolves registry, manages
   │  from_pretrained,      compile cache, dispatches to <Model>Application
   │  precompile, __call__
   ▼
ModelEntry (registry.py)    per-model metadata: factory, default parallel +
   │                        shape, download patterns, allowed backends
   ▼
<Model>Application          composes encoder / DiT / VAE sub-apps; exposes
   │                        compile() + load() + __call__()
   ▼
difflet.ops                 backend-neutral op surface (attention / linear /
   │                        norm / collectives / embeddings / platform).
   │                        Frozen v1; additive only; dispatch frozen at
   │                        first import per process.
   ▼
difflet/backends/<hw>/ops_impl/   per-hardware implementations.
                            trainium = the real backend; cpu = pure torch
                            numerical reference; cuda/rocm stubs.
```

`DiffletPipeline` only invokes `compile() / load() / __call__()` on an application. Multi-component
models extend the shared `MultiComponentApplication` base
(`difflet/backends/trainium/core/`), which provides race-safe compile with `model.pt` markers and
SPMD barriers, plus ordered load with the biggest-TP component first. All current model
applications (Flux, Wan, HunyuanVideo, Qwen-Image, LTX-2) build on it.

### CLI orchestration

Each model has an orchestrator under `difflet/cli/orchestrators/` implementing `download()`,
`compile()`, and `generate()`. The base class (`orchestrators/base.py`) defines `run()` as
`download() → compile() → generate()`. Single-process models (Flux, LTX-2) run in-process;
staged models (Wan, HunyuanVideo, Qwen-Image) spawn one subprocess per stage and pass tensors
through `--work-dir` files, because the encoder, DiT, and VAE cannot all co-fit on a 4-core card.

## Repository layout

```
difflet/
├── pipeline/        Difflet-authored public API (DiffletPipeline, compile cache,
│                    parallel config, HF path resolver)
├── cli/             difflet CLI: main.py, stage.py, orchestrators/<model>.py
├── registry.py      @register_model + ModelEntry
├── ops/             Backend-neutral op surface (frozen v1)
├── backends/
│   ├── trainium/    Real backend (NXD + nkilib + torch_neuronx)
│   │   ├── core/    AOT base classes, attention, custom_calls
│   │   ├── utils/   compile_env, runtime_env, distributed, snapshot
│   │   ├── ops_impl/    Trainium impls of difflet.ops
│   │   ├── nki_kernels/    NKI custom kernels (MX microscaling, etc.)
│   │   └── flux/ wan/ hunyuan_video/ qwen_image/ ltx_2/    per-model wrappers
│   ├── cpu/         Pure-torch numerical reference
│   └── cuda/ rocm/  Stubs
├── utils/           Hardware-neutral utilities (HF / diffusers adapters)
├── layers/          Diffusion-specific layers; import only via difflet.ops
└── models/          flux/ wan/ hunyuan_video/ qwen_image/ ltx_2/ — each:
                     modeling, pipeline, application, entry, checkpoint
                     (where needed). HunyuanVideo 1.5 is served from
                     hunyuan_video/ (model_version="1.5").
```

`difflet/backends/trainium/{core,modules}` follows upstream Neuron coding style so periodic
rebases stay clean; `difflet/{pipeline, ops, cli, registry.py, models/<new>}` is Difflet-authored
and formatted with Black.

## The op surface and backends

`difflet.ops` is a **frozen v1** op surface: a backend-neutral API covering attention, linear,
norm, collectives, embeddings, and platform queries. Model and layer code imports only from
`difflet.ops` — never `neuronx_distributed`, `torch_neuronx`, or `nkilib` directly. This keeps
modeling code portable and lets a CPU reference backend mirror every Trainium op for numerical
parity testing.

Rules:

- The surface is **additive only**. Dispatch is frozen at first import per process.
- Each op has a Trainium implementation under `difflet/backends/trainium/ops_impl/`, a CPU
  reference under `difflet/backends/cpu/ops_impl/` (the numerical oracle), a TPU
  implementation under `difflet/backends/tpu/ops_impl/`, and a CUDA implementation under
  `difflet/backends/cuda/ops_impl/`. `rocm/` is a stub.
- The Megatron-style parallel layers, the shard-at-load helpers and the safetensors loader are
  shared by the torch backends in `difflet/backends/_torch_megatron/`; TPU and CUDA each bind
  them to their own collectives (`build_parallel_layers(collectives)`).

The repo-wide import guard (`scripts/test_imports.sh`) enforces that modeling files import only
from `difflet.ops`, `torch`, stdlib, `diffusers`, and `transformers`. It runs in two tiers: the
backend-neutral modules always, the Trainium modules only when the Neuron SDK is importable.

### CUDA backend (Release 1)

`DIFFLET_BACKEND=cuda` runs Qwen-Image at `tp=1` on one NVIDIA GPU, with the DiT going through
`difflet.ops` and the text encoder, VAE and scheduler loop on the same device in one process.
There is no ahead-of-time compile step: `compile()` writes `cuda_manifest.json`, `load()` builds
the module eagerly from the diffusers safetensors. Validated on an RTX PRO 6000 (Blackwell,
96 GB) with torch 2.11.0+cu128: 50 steps at 1024×1024 in **16.5 s eager (0.33 s/step)**, 12.4 s with
`--torch-compile`, 10.0 s with `--teacache-cadence 2`; 58 GB peak; the DiT parity gates (all-ones
and padded masks) and the whole-pipeline gate vs diffusers (PSNR 41.2 dB eager, 40.2 dB compiled)
passing. Upstream diffusers' pipeline runs the same 50 steps in 15.2 s on that card. The DiT
forward alone: 323 ms eager, 243 ms compiled, 300 ms for diffusers (`scripts/cuda_bench_dit.py`). Plan, decisions and status log:
`docs/plans/2026-10-16-cuda-backend-support.md`.

```bash
# environment (CUDA 12.8+ wheel; Blackwell needs it) — see requirements-cuda.lock
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
uv pip install --python .venv/bin/python -e ".[test,cuda]"
scripts/cuda_env_check.sh                       # driver, capability, torch build, bf16 matmul

# first image through the CLI (one process; download + eager "compile" + generate)
hf download Qwen/Qwen-Image
DIFFLET_BACKEND=cuda difflet run --model-id Qwen/Qwen-Image --tp-degree 1 \
    --prompt "a photorealistic cat sitting in a sunlit garden" --output /tmp/qwen.png --steps 50
scripts/cuda_smoke.sh --prompt "..." --output /tmp/qwen.png     # same, logged to /tmp/logs
DIFFLET_BACKEND=cuda python scripts/cuda_smoke.py --steps 50     # per-phase timings, peak memory
DIFFLET_BACKEND=cuda difflet run --model-id Qwen/Qwen-Image --prompt "..." --output /tmp/qwen.png \
    --torch-compile --teacache-cadence 2             # measured options: compiled DiT blocks,
                                                     # skip every 2nd step after warm-up; both off by default

# DiT forward, Difflet vs diffusers on identical inputs: timed median + kernel categories
DIFFLET_BACKEND=cuda python scripts/cuda_bench_dit.py --model-path <snapshot dir> [--torch-compile]

# oracles against upstream diffusers: DiT-level (all-ones and padded-prompt masks), then
# whole pipeline (prompt in, image out, same initial noise)
DIFFLET_BACKEND=cuda python tests/numerical/cuda_parity_gpu.py --model-path <snapshot dir>
DIFFLET_BACKEND=cuda python scripts/cuda_e2e_vs_diffusers.py --model-path <snapshot dir>
DIFFLET_RUN_CUDA_NUMERICAL=1 pytest tests/numerical/test_cuda_vs_diffusers.py tests/numerical/test_cuda_e2e_vs_diffusers.py -v

# the CUDA column of the CLI matrix (download → compile → timed generate, results.json)
python scripts/verify_cli_cuda.py --steps 50

# OpenAI-compatible serving: one resident process, the model stays loaded between requests
DIFFLET_BACKEND=cuda difflet serve --model-id Qwen/Qwen-Image --port 8091
scripts/cuda_serve_smoke.sh --output /tmp/qwen-serve.png --steps 50   # start, /ready, 2 requests, timings

# CPU-host gate (no SDK, no GPU) — also what CI runs
scripts/cpu-gate.sh
```

**Prompt padding mask.** Difflet pads every prompt to 1024 text tokens; a typical prompt is
about 16. The shared Qwen modeling code drops the padding mask (it does not trace under XLA),
so on Trainium and TPU the DiT attends over the padding. On CUDA the padding is sliced off before
the DiT (`_QwenImageTransformerTraceModule(apply_text_mask=True, trim_text_padding=True)`, set only
by the CUDA transformer application): the same image output as masking it (masked keys contribute
nothing; the dropped positions' own outputs are discarded), but the joint sequence is ~4112 tokens
instead of 5120 and SDPA takes the flash kernel again — 516 → 323 ms per DiT forward. Ragged batches
keep a mask for the shorter rows. `tests/unit/models/qwen_image/test_qwen_text_mask.py` pins both
against diffusers on CPU. Trainium/TPU outputs are unchanged by this.

**Measured options (both off by default; eager is the premise).** `--torch-compile`
(`difflet serve --torch-compile`, or `application_kwargs={"torch_compile": True}`) compiles each
DiT block with inductor (`max-autotune-no-cudagraphs`, `dynamic=True` so one graph serves every
prompt length): 0.33 → 0.25 s/step. Expect two compilations at load (the warm-up runs twice under
the flag: dynamo's guard on a `None` block argument forces the second; ~16 s together with a warm
inductor cache, more on a fresh box), then no recompiles — `TORCH_LOGS=recompiles` shows them. Parity moves but
stays inside the bf16 noise (rel_l1 0.88× the bf16 control vs fp32). CUDA-only: the other backends
refuse the flag. `--teacache-cadence N` runs the existing probe-free TeaCache controller (skip every
Nth step after warm-up): cadence 2 → 0.20 s/step at PSNR 38.5 dB against the full run. The
calibrated and online-delta TeaCache modes need the probe path on CUDA and are refused.

**Flux on CUDA (verified on the RTX PRO 6000).** `modeling_flux.py` is backend-neutral since
the Trainium wrapper classes live in `difflet/models/flux/trainium_backbone.py` (bodies
unchanged), the layer-boundary markers resolve per backend in `difflet/layers/markers.py`
(identity off Trainium), and `FluxDiTConfig` (`difflet/models/flux/dit_config.py`) replaces the
Trainium `InferenceConfig` elsewhere. On CUDA, Difflet's own `NeuronFluxTransformer2DModel` runs
on the CUDA ops from the diffusers safetensors (`difflet/backends/cuda/flux/transformer.py`; the
single-stream block's `skip_bias_add` / `reduce_output=False` / `process_group=` contract is now
part of the shared and CUDA ops) and a thin `FluxDiTAdapter` lets diffusers' `FluxPipeline` drive
it with CLIP, T5, the scheduler and the VAE on the same GPU. The modeling is pinned against
diffusers' `FluxTransformer2DModel` on CPU (`tests/unit/models/flux/test_flux_vs_diffusers_cpu.py`,
and a two-step `FluxPipeline` run around the adapter in `test_flux_cuda_application.py`).

```bash
# Both FLUX repos are gated: accept the license on the model page, then `hf auth login`.
hf download black-forest-labs/FLUX.1-dev --exclude "flux1-dev.safetensors" --exclude "ae.safetensors"
DIFFLET_BACKEND=cuda difflet run --model-id black-forest-labs/FLUX.1-dev --tp-degree 1 \
    --prompt "..." --output /tmp/flux.png --steps 28 --guidance-scale 3.5
scripts/cuda_smoke_flux.sh --prompt "..." --output /tmp/flux.png --steps 28
DIFFLET_BACKEND=cuda python tests/numerical/cuda_parity_flux_gpu.py --model-path <snapshot dir>
DIFFLET_BACKEND=cuda python scripts/cuda_e2e_flux_vs_diffusers.py --model-path <snapshot dir>
DIFFLET_RUN_CUDA_NUMERICAL=1 pytest tests/numerical/test_cuda_flux_vs_diffusers.py tests/numerical/test_cuda_flux_e2e_vs_diffusers.py -v
python scripts/verify_cli_cuda.py                # Qwen-Image at 50 steps and FLUX.1-dev at 28
```

Measured on the RTX PRO 6000 (2026-10-18): `difflet run` writes a 1024×1024 image in 23.2 s for
28 steps after a 36.5 s load; DiT parity vs diffusers rel_l1 0.987× the bf16 control (same-dtype
distance 0.398×); whole-pipeline e2e vs diffusers' `FluxPipeline` PSNR 40.35 dB / cosine 0.99980
(floors 35 dB / 0.998); the CLI matrix passes both cells. Flux on CUDA refuses TeaCache, SP, CP, CFG-parallel, TAEF1,
`--host-vae`, bucketed shapes and `--torch-compile` as F5b. The Trainium byte-identity of the
split is argued (verbatim move, CPU suites) but not re-run on a trn2.

**Wan 2.2 on CUDA (verified on the RTX PRO 6000).** `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
runs on one GPU at `tp=1` through Difflet's own backend-neutral Wan DiT (`modeling_wan.py`) with
**both 14B experts resident** (`enable_transformer_2=True` is the CUDA default; Trainium/TPU keep
False) and the true-CFG path: `difflet/models/wan/cuda_application.py` holds one
`CudaWanTransformerApplication` per expert (`difflet/backends/cuda/wan/transformer.py`, diffusers
safetensors loaded through the shared `_torch_megatron` loader with the `ffn.net_in/net_out` rename
from the neutral `difflet/models/wan/dit_config.py`, lifted from the TPU port), UMT5 and
`AutoencoderKLWan` on the GPU behind callables, and drives the backend-neutral `WanOrchestrator`
(boundary-ratio expert switch at timestep 875, cond + uncond passes per step, loop latents on the
host in fp32 as on TPU). `difflet run --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers` writes an MP4
(`difflet/cli/orchestrators/wan_cuda.py`, 40 steps, `--num-frames` default 9, `--guidance-scale`
default 4.0 with `guidance_scale_2` 3.0 below the boundary). Measured 2026-10-18 (Workstation
Edition card): load 703 s cold from the volume (118 GB of fp32 shards cast to bf16), 9 frames at
480×832 in 55 s for 40 steps (1.38 s/step, two passes per step), 71.3 GiB peak (73,035 MiB)
with both experts resident; DiT parity vs diffusers per expert rel_l1 1.051× / 1.043× the bf16
control (same-dtype 0.870× / 0.819×, cosine 0.9988 / 0.9997); whole-video e2e vs diffusers'
`WanPipeline` from the same noise PSNR 26.3 dB / cosine 0.9961 with the fp32 timestep (27.5 dB /
0.9970 under Release 1's bf16 cast), floors 25 dB / 0.99. Diffusers generates the same clip in
44.6 s; the 10 s gap is unmeasured (the host latents round trip is far too small to explain it)
and belongs to F4b. The e2e PSNR is well below Qwen's and Flux's; F1b measured the first suspect,
the timestep: the CUDA DiT now receives it in fp32 (`timestep_dtype="float32"`, a per-component
property the shared orchestrator reads; Trainium and TPU keep their model-dtype cast), which
brings each DiT call closer to diffusers (rel_l1 1.084× → 1.039× the control at timestep 999,
where bf16 rounds to 1000) but leaves the 40-step trajectory 1.2 dB further away (26.3 dB vs 27.5
dB with the old cast; the old-cast `model` variant ran on both the Workstation and the Server
Edition cards and matched to the digit, the fp32 variant on the Server Edition only), so a
single-seed A/B does not support the timestep as the gap. Both sides combine CFG in bf16 (Difflet
on host tensors the DiT returned in bf16, diffusers on the device), so the remaining suspects are
the scheduler step on the host CPU vs the device, the bf16 latents/text casts, and the
empty-prompt negative embeds; measure the e2e metric's own noise floor (diffusers vs diffusers
with a perturbed timestep, several seeds) before chasing any of them. Both sides decode the VAE in
bf16, so it is not a parity suspect. A Wan 2.2 snapshot without `transformer_2/` is refused (a
partial download), not run on one expert.
`Wan-AI/Wan2.1-T2V-14B-Diffusers` is routed to the same path but unverified on a GPU; diffusers'
Wan 2.1 defaults (50 steps, guidance 5.0) differ from the 40 / 4.0 used here.

**Two GPUs on CUDA (verified on 2× RTX PRO 6000 over PCIe).** Decision D6's order: data
parallel and CFG-parallel first. `difflet run --dp 2 --requests requests.jsonl` runs one
single-GPU replica per card through the existing request router (`difflet/cli/dp/`): each worker
gets `CUDA_VISIBLE_DEVICES` for its card, loads once, claims requests from the shared manifest and
writes each output; the one-process orchestrator's batch mode does the loop. Measured: four
50-step Qwen-Image requests in 58 s wall on two cards against 190 s as four serial `difflet run`
calls and 79 s as a `--dp 1 --requests` batch on one card (one load, four generates); every
compared image (all four from `--dp 2`, two from `--dp 1`) is byte-identical to its reference.
`difflet run --model-id Wan-AI/... --cfg-parallel` re-execs itself under `torchrun --standalone
--nproc_per_node=2` (`difflet/cli/runner.py`): `CudaBackend.prepare_runtime` joins NCCL at
`WORLD_SIZE > 1`, the CUDA mesh builds one `torch.distributed` group per non-trivial axis from
`MeshSpec.axis_groups`, and the unchanged Wan modeling scatters the batch of two to one branch per
rank and all-gathers the result each step; both ranks run the whole orchestrator loop (SPMD) and
rank 0 writes. Measured: the 40-step, 9-frame clip in 26.1 s on two cards against 50.9 s serially
on one (1.95×, one all-gather per step over PCIe), bit-identical to the serial video (max_abs 0);
68,074 / 67,519 MiB resident on the two cards. The torchrun child dies with its parent
(`PR_SET_PDEATHSIG`); a batch request that fails inside a two-rank world fails that worker fast so
the router records it. Still refused: context parallel, tensor parallel across PCIe and
`--sp`, `--cfg-parallel` for Qwen-Image and Flux (single-pass / distilled); `--dp` with
`--cfg-parallel` needs four cards and is unverified.

**Verifying the CUDA backend.** A GPU session runs, per model, the smoke (`difflet run`), the DiT
parity script under `tests/numerical/` and its gate, and the e2e script under `scripts/` and its
gate; then `scripts/verify_cli_cuda.py` and the serving smoke. The results — per-gate status, the
support matrix (PASS / LIMIT / N/A / NOT MEASURED) and the list of what was never run — are
summarised in `docs/verification/cuda-release1-evidence.md`.

**Serving on CUDA.** `difflet serve` keeps the serving engine's one resident worker
process; on CUDA that worker holds a single opaque `pipeline` stage —
`DiffletPipeline.from_pretrained(backend="cuda")` kept loaded, `app.generate()` per request —
instead of the three Neuron stage applications (`difflet/serving/orchestrators/qwen_image_cuda.py`,
selected by backend name in `difflet/serving/model_registry.py`). The worker loads the model with
the same `DiffletPipeline.from_pretrained` arguments as `difflet run` (pinned to the commit the
server resolved at startup) and runs each request through the same `generate()` call:
`num_inference_steps` (default 50), `seed`, the profile's one shape. Three deliberate
differences from the CLI: the HTTP layer caps steps at 1..50; `guidance_scale` is accepted only
at its no-op value 4.0 and refused otherwise (`feature_not_supported` — the model has
`guidance_embeds=False`, so any value would be a silent no-op; the CLI refuses the flag
outright); prompts beyond 1024 tokens past the chat template are refused (`prompt_too_long`)
where the CLI truncates silently. Other shapes are refused with `profile_mismatch`. At startup,
`--tp-degree > 1` / `--sp` are refused naming F6 and `--shapes` naming F3, before the snapshot is
resolved; `--cp-degree > 1`, `--cfg-parallel` and a TeaCache flag are refused earlier still by
the shared profile builder with its own wording (they are F2b / F4 work). `--tp-degree` defaults
to 1 on CUDA. One request runs at a time; a 50-step request takes what `difflet run` takes on
the same card (measured 38.6 s served vs 38.5 s CLI on an RTX PRO 6000 Workstation Edition,
~17 s on the Server Edition; ready in 56 s including load and the startup smoke). A request
that waits longer than `--queue-timeout` (30 s default — shorter than one 50-step request on
the Workstation card, so raise it when clients queue) gets 429 `queue_timeout`, and a request
past `--request-timeout` restarts the worker (full reload plus the 4-step smoke) because
`generate()` is not interruptible mid-denoise. Trainium and TPU serving are
untouched.

Not in Release 1 (follow-ups F1–F8 in the plan; Wan, Flux, serving and the second card have
landed as F1, F5, F3b and F2 for the shapes and models above, with their own follow-ups):
context parallel, tensor parallel across PCIe (need-triggered), multi-shape serving on
CUDA, Wan and Flux serving on CUDA, serving on two GPUs, Wan follow-ups (F1b:
TeaCache, `--shapes`, 81-frame clips, I2V, `--negative-prompt`, batch > 1), Flux follow-ups,
CUDA graphs, calibrated/online-delta TeaCache, FP8, `flash-attn`, custom kernels.

## Compile cache

Difflet maintains a content-addressed cache of AOT-compiled artifacts.

```
~/.cache/difflet/<model>/<sha256-prefix>/
├── manifest.json
├── text_encoder/   model.pt + neuron_config.json
├── text_encoder_2/ model.pt + neuron_config.json
├── transformer/    model.pt + neuron_config.json
└── decoder/        model.pt + neuron_config.json
```

The cache key hashes:

- model id, registry name, revision
- parallel configuration (`tp_degree`, `cp_degree`, `cfg_parallel_enabled`)
- dtype (normalized — `"bf16"`, `"bfloat16"`, `torch.bfloat16` collapse to one key)
- shape (`height`, `width`, `num_frames`)
- toolchain versions (Python major.minor, torch, neuronx-cc, neuronx-distributed, nki,
  libneuronxla, torch-neuronx, torch-xla, diffusers, transformers)

`model_path` and the Python patch version are recorded in the manifest for debugging but
excluded from the key, so caches are portable across hosts and survive Python patch upgrades.

Override the cache root with the `DIFFLET_COMPILE_CACHE` environment variable, `--cache-dir`
(CLI), or `compile_cache_dir=` in `from_pretrained`. Pass `--force` (CLI) or `force_compile=True`
(API) to bypass a valid cache hit.

Separate from this cache, neuronx-cc dumps scratch into the process **working directory** —
16-hex-char per-kernel cache dirs, `neuronxcc-<id>/` work dirs, and a few diagnostic files (all
gitignored). `difflet clean` (`difflet/cli/clean.py`) sweeps them; `--dry-run` lists first. It
matches only direct children by those fixed patterns, never follows symlinks, and skips a
hash-named directory whose contents don't look like compiler output.

## Parallelism

`DiffletParallelConfig` exposes four orthogonal parallelism axes, organized as a named device
mesh `{dp, cfg, cp, tp}` (tp innermost, dp outermost):

- `tp_degree` — tensor-parallel degree (must divide the visible NeuronCore count).
- `cp_degree` — context-parallel degree (1 = disabled); the attention strategy is selectable
  with `cp_mode` (`gather_kv` or `ring`).
- `cfg_parallel_enabled` — splits the CFG conditional/unconditional batch across a cfg axis of
  size 2. Mutually exclusive with `cp_degree > 1` at the pipeline level, and only valid for
  true-CFG models (Wan, LTX-2) — guidance-distilled models (Flux, HunyuanVideo, HunyuanVideo
  1.5, Qwen-Image) run a single forward pass and reject it.
- `sp_enabled` — Megatron-style sequence parallelism: shards the otherwise-replicated
  norm/modulation/residual regions along the sequence axis across the *existing*
  tensor-parallel group (reduce-scatter replaces the row-parallel all-reduce). Adds no mesh
  axis, so `world_size` is unchanged. Mutually exclusive with `cp_degree > 1`. Device-verified
  for Flux, Wan 2.1/2.2, HunyuanVideo, and Qwen-Image (support is read from each model's
  registry `ModelCapabilities.supports_sp`, not a hand-list; parity gates live in
  `scripts/*_sp_parity_smoke.sh`).
- `dp_degree` — data-parallel replica axis (reserved: groups exist, the replication feature is
  not wired yet). The dp axis carries **no** per-layer/per-step collective — enforced by
  `tests/unit/test_no_dp_parasites.py`.

`world_size` is always the product `dp * cfg * cp * tp`, with
`rank = tp + T*(cp + C*(cfg + G*dp))`:

```python
DiffletParallelConfig(tp_degree=4)                          # world_size=4
DiffletParallelConfig(tp_degree=4, cfg_parallel_enabled=1)  # world_size=8
DiffletParallelConfig(tp_degree=4, cp_degree=2)             # world_size=8
DiffletParallelConfig(tp_degree=4, cp_degree=4)             # world_size=16
```

The mesh math (rank ↔ coords mapping, per-axis subgroup meshes) lives in
`difflet/pipeline/parallel_mesh.py` (`MeshSpec`, exposed as
`DiffletParallelConfig.mesh_spec`). On Trainium,
`difflet/backends/trainium/core/parallel_mesh.py` materializes one process group per
non-trivial axis (`init_parallel_mesh(config)`, `get_cfg_group()`, `get_cp_group()`,
`get_cfg_rank_spmd()`, `get_cp_rank_spmd()`, reachable via `difflet.ops`). Model code must use
its own axis's group for every collective: the CFG cond/uncond merge fires on the cfg group,
CP sequence scatters/gathers and ring K/V rotation on the cp group, TP/SP on NxD's
tensor-parallel group. NxD's legacy `get_data_parallel_group` is no longer part of the ops
surface. The migration is bit-exact: `scripts/mesh_regression_smoke.sh` compiles and runs the
Wan backbone at `tp=2×cfg=2` and `tp=2×cp=2` against the pre-refactor baseline and asserts
byte-identical outputs on device.

On the 4-core `trn2.3xlarge`, the four-core configurations are `tp=4`, `tp=2 cp=2`,
`tp=2 + --cfg-parallel` (true-CFG models), and `tp=4 --sp` (SP-capable models); LTX-2 and
HunyuanVideo 1.5 support neither CP nor SP and use `tp=4`. All combinations are exercised
end-to-end by `scripts/verify_cli.py` (see Development workflow).

## Runtime protocol

Difflet diffusion artifacts have hard constraints that differ from typical LLM serving setups.
Violating them produces silent SIGSEGVs or `global communicator` errors.

1. **Use one Python process with multiple visible NeuronCores.** Set `NEURON_RT_NUM_CORES=N` and
   run the application in a single process. Do **not** launch with `torchrun --nproc_per_node=N`
   for the diffusion path — `torchrun`'s MPMD model assigns one core per process, which is
   incompatible with how these traced artifacts initialize the runtime communicator.

2. **All components in a pipeline must share the same `world_size`.** For Flux on 4 cores: T5 and
   the transformer run as `TP=4, DP=1`; CLIP and the VAE decoder run as `TP=1, DP=4`. All four
   components have `world_size=4`. Mixing `world_size=1` and `world_size=4` artifacts in one
   process crashes during weight initialization.

3. **Component load order matters.** The first component loaded fixes the process-wide
   NeuronCore communicator. Tensor-parallel components must load before replicated ones. Flux
   load order: `text_encoder_2 → transformer → text_encoder → decoder`.

These rules are enforced in `difflet/models/flux/application.py`; new model ports should follow
the same pattern. For staged models, each stage is its own process, which sidesteps the co-fit
limit but requires inter-stage tensors to flow through `--work-dir` files.

## Performance and profiling

### Baselines

**Flux.1-dev — 1024×1024, 28 steps, bf16, cache hit** (`trn2.3xlarge`, `NEURON_RT_NUM_CORES=4`,
`--tp-degree 4`, `--skip-warmup`):

| Stage | Time | Gate |
|---|---:|---:|
| `from_pretrained` (load + shard + NRT init) | 64.6 s | not gated |
| 28-step forward | 7.8 s | ≤ 8.5 s |
| Denoise throughput (steady state) | 3.78 it/s | not gated |
| Compile cache size on disk | 113 MB | n/a |
| First-time AOT compile (cold) | ~683 s | n/a |

**Wan 2.2 T2V — 480×832, 9 frames, bf16, cache hit, 2 inference steps:**

| Stage | Time |
|---|---:|
| Stage 1 load (text encoder + DiT, TP=4) | 19.4 s |
| Stage 1 forward (UMT5 + 2 denoise steps) | 3.7 s |
| Stage 2 load (VAE decoder, TP=1) | 15.3 s |
| Stage 2 forward (single decode) | 1.0 s |
| Total wall clock (`wan_smoke.sh`) | ~53 s |

### TeaCache

Difflet ships an optional TeaCache step-skipping path (`difflet/pipeline/teacache*.py`) to cut
denoise time. Three modes are exposed on `difflet generate` / `difflet run`:

- `--teacache-cadence N` — skip every N-th DiT step (fixed cadence, no calibration).
- `--teacache-online-delta ALPHA` — online-delta gating (no calibration).
- `--teacache-speedup X` — adaptive target speedup; requires `--teacache-calibration PATH`.

The modes are mutually exclusive. The latent-metrics helpers (`difflet/pipeline/latent_metrics.py`)
support a CPU-shadow gate for calibrating skip decisions against a reference trajectory.

## Adding a new model

To port a diffusion model, add three things:

1. **`difflet/models/<name>/`** — implementation: `application.py` composing the
   encoder / backbone / decoder sub-applications, `pipeline.py` (or a thin orchestrator like
   `difflet/models/wan/pipeline.py`), plus `modeling_<name>.py` for the DiT backbone.
2. **`difflet/models/<name>/entry.py`** — a factory
   `create_<name>_application(model_path, parallel, dtype, shape, **kwargs)`.
3. **`difflet/registry.py`** — a `@register_model` entry pointing to the factory by string (lazy
   import) plus default parallel config, shape, HF download patterns, and supported
   `backends=("trainium", ...)`.

To expose the model through the CLI, also add an orchestrator under
`difflet/cli/orchestrators/<name>.py` and register it in `difflet/cli/main.py`
(`VALID_MODELS`, `_MODEL_TYPE`, and `_get_orchestrator`).

Hard rules for new modeling code (enforced by `scripts/test_imports.sh`):

- model files import **only** from `difflet.ops`, `torch`, stdlib, `diffusers`, and
  `transformers`;
- no direct `neuronx_distributed`, `torch_neuronx`, `nkilib`, `difflet.core`, or old
  `difflet.utils.{compile_env,runtime_env,distributed,snapshot}` imports;
- if a primitive is missing, add it to `difflet.ops` first (with at least the Trainium
  implementation under `difflet/backends/trainium/ops_impl/`, ideally also a CPU reference under
  `difflet/backends/cpu/ops_impl/`).

`DiffletPipeline` itself does not change — multi-component models extend the shared
`MultiComponentApplication` base.

## Development workflow

### CLI verification matrix

`scripts/verify_cli.py` is the on-device release gate: for each model it runs
`difflet download` once, then `difflet compile` + a **timed** `difflet generate` for every
supported four-core parallel config (`tp4`, `tp2cp2`, `tp2cfg`, `tp4sp`), with sample-grade
step counts so outputs are visually verifiable against the prompt. Unsupported combinations
auto-SKIP (the skip-rule sets are pinned to `difflet.cli.main` by drift-guard unit tests);
documented gaps are XFAIL cells. Results land in `/tmp/logs/verify_matrix_<ts>/` as per-cell
logs, output artifacts, a matrix summary, and `results.json`; the exit code is nonzero on any
unexpected FAIL/XPASS.

```bash
python scripts/verify_cli.py                          # full 7-model x 4-config matrix
python scripts/verify_cli.py --models wan --configs tp4sp
```

**Validation lesson (hard-won):** parity/smoke tests must compare against the **diffusers
reference**, not difflet-to-difflet. Two output-destroying bugs (unzeroed Wan pad embeddings,
a silently-dropped attention mask on the CPU backend) survived every internal parity gate
because both sides of each comparison shared the bug. Exit code 0 plus an artifact on disk is
not verification either — check mode-specific side effects (e.g. an `*sp*` compiled dir, a
plausible compile time) when validating a flag.

### Known toolchain issues (neuronx-cc 2.25.3371, trn2)

Repro artifacts preserved under `/tmp/logs/` on the validation host; ticket material for
`aws-neuron-sdk`:

1. `NCC_INLA001` — internal compiler error (SBUF allocation, `concatenate [140,1536,1]`)
   compiling the HunyuanVideo DiT at `tp=2 cp=2`. Deterministic.
2. `NCC_EVRF007` — the Wan VAE decoder graph exceeds the 5M compiler-instruction budget
   beyond ~9 frames (23.5M at 49 frames). Worked around with `--host-vae`; the real fix is a
   per-frame Neuron VAE decode mirroring diffusers' causal cache.
3. On-device video-DiT sample fidelity degrades relative to the CPU reference with identical
   inputs, worsening with attention sequence length (fine ≲5k tokens for image DiTs and
   LTX-2; degraded at ≳13k — HunyuanVideo tiles, Wan long-clip washout). Difflet modeling is
   validated exact vs diffusers on CPU (cosine ≥ 0.9995); one-step NEFF parity harnesses:
   `scripts/wan_m25c_neff_cpu_alignment.sh`, `scripts/hunyuan_real_weight_one_step_alignment.py`.

### Helper scripts

Project-local helper scripts:

```bash
./scripts/check_quick.sh                      # imports + unit tests
./scripts/test_unit.sh                        # pytest tests/unit -q
./scripts/test_imports.sh                     # smoke import difflet + key submodules
./scripts/flux_smoke.sh                       # 1-step Flux smoke (load + 1 forward)
./scripts/flux_baseline_28.sh                 # 28-step Flux baseline gate

# Wan 2.2 spike + alignment
./scripts/wan_smoke.sh                        # M2 spike e2e (text + DiT + VAE, 4-core sequential)
./scripts/wan_e2e_smoke.sh                    # single-process Wan e2e (component-level debug)
./scripts/wan_e2e_sequential_smoke.sh         # explicit two-stage split, no env-flag magic
./scripts/wan_backbone_compile_smoke.sh       # AOT compile DiT only
./scripts/wan_text_encoder_compile_smoke.sh   # AOT compile UMT5 only
./scripts/wan_vae_compile_smoke.sh            # AOT compile VAE only (~78 min @ 480×832×9)

# M2.5 numerical alignment gates
./scripts/wan_m25_numerical_alignment.sh      # M2.5-A: VAE NEFF vs CPU
./scripts/wan_m25b_text_dit_alignment.sh      # M2.5-B: UMT5 / DiT real-weight CPU parity
./scripts/wan_m25c_neff_cpu_alignment.sh      # M2.5-C: UMT5 / DiT NEFF vs CPU
./scripts/wan_vae_real_alignment.sh           # VAE real-weight CPU + NEFF parity (used by M2.5-A)

# Wan utilities
./scripts/wan_convert_checkpoint.sh           # HF → Difflet state-dict conversion CLI
```

All scripts auto-set the Neuron venv on `PATH`, the project on `PYTHONPATH`, and
`NEURON_RT_NUM_CORES`. The M2.5 scripts use a 115 GB peak-RSS gate
(`DIFFLET_M25{B,C}_PEAK_RSS_MAX_GB`) to avoid OOM on the 4-core spike host.

**Environment.** `./scripts/setup_env.sh` builds the reference environment at `<repo>/.venv`
from `requirements-neuron.lock` (recent Neuron DLAMIs no longer ship the old
`/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` venv). Scripts default to the legacy `/opt`
venv path; when it is absent, export `PYTHON_BIN=<repo>/.venv/bin/python` (the flux smoke
scripts fall back to `<repo>/.venv` automatically). Keep the lock in sync with the venv —
regenerate after any toolchain change with:

```bash
.venv/bin/pip freeze --exclude-editable > requirements-neuron.lock
```

Identical pins are what make `~/.cache/difflet` compile caches portable across hosts: the cache
key includes the Python minor version and the torch / neuronx-cc / neuronx-distributed /
diffusers / transformers versions, so any drift silently forces a full recompile.

Known-bad pin: `neuronx-cc==2.27.5334.0` fails with an internal compiler error
(`[NCC_ISMP902] Simplifier error: is_subset()`) on the Flux CLIP text encoder; the lock pins
`2.26.6360.0`, which compiles everything cleanly.

Run unit tests directly:

```bash
PYTHONPATH=. pytest tests/unit -q
```

Format Difflet-authored code:

```bash
black difflet/pipeline difflet/cli difflet/registry.py examples tests
```
