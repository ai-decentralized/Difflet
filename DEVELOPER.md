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
- Each op has a Trainium implementation under `difflet/backends/trainium/ops_impl/` and,
  ideally, a CPU reference under `difflet/backends/cpu/ops_impl/`.
- `cuda/` and `rocm/` are stubs.

The repo-wide import guard (`scripts/test_imports.sh`) enforces that modeling files import only
from `difflet.ops`, `torch`, stdlib, `diffusers`, and `transformers`.

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

Run unit tests directly:

```bash
PYTHONPATH=. pytest tests/unit -q
```

Format Difflet-authored code:

```bash
black difflet/pipeline difflet/cli difflet/registry.py examples tests
```
