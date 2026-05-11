# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Nova is a Trainium-native inference framework for diffusion transformers (Flux,
Wan 2.2; Hunyuan / Qwen-Image / LTX planned). One public entry point —
`NovaPipeline.from_pretrained(...)` in `nova/pipeline/nova_pipeline.py` —
handles HF download, AOT compilation, content-addressed compile cache, and
SPMD multi-core execution.

The README is current and authoritative for status, performance baselines,
quick-start commands, parallel modes, and compile-cache semantics. Read it
once before non-trivial work.

## Common commands

The Neuron toolchain lives at `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`.
Helper scripts auto-set `PATH`, `PYTHONPATH=$ROOT`, and `NEURON_RT_NUM_CORES`,
so prefer them over raw `python` invocations.

```bash
./scripts/check_quick.sh                 # imports + tests/unit (use as default sanity gate)
./scripts/test_unit.sh                   # passes extra args through to pytest
./scripts/test_unit.sh tests/unit/test_pipeline.py::test_name   # single test
./scripts/test_imports.sh                # smoke import nova + key submodules

./scripts/flux_smoke.sh                  # 1-step Flux load + forward
./scripts/flux_baseline_28.sh            # 28-step Flux baseline gate (≤ 8.5 s denoise)
./scripts/wan_smoke.sh                   # Wan 2.2 spike e2e (two-stage, 4-core)

black nova/pipeline nova/registry.py examples tests   # format Nova-authored code only
```

`pyproject.toml` excludes `nova/{core,layers,utils}` and `nova/models/flux/**`
from black/isort — those are upstream-derived files kept at upstream
formatting until rebase. New model dirs (`nova/models/wan/`, future ports)
**are** black-formatted.

## Architecture in one paragraph

`NovaPipeline` resolves a `ModelEntry` from `nova/registry.py`, calls a
per-model `<Model>Application` (e.g. `nova/models/flux/application.py`,
`nova/models/wan/application.py`) which composes 2–4 sub-applications
(text encoder(s), DiT, VAE) and exposes only `compile()`, `load()`,
`__call__()`. Modeling code imports **only** from `nova.ops` (the frozen
v1 backend-neutral op surface in `nova/ops/`); `nova/ops/_dispatch.py`
selects an implementation under `nova/backends/<hw>/ops_impl/` on first
import and freezes it for the process. `trainium` is the only real
backend; `cpu` is a numerical reference; `cuda` / `rocm` are stubs.

`nova/core/` and the four legacy `nova/utils/{compile_env, runtime_env,
distributed, snapshot}` paths **have been removed**. Modeling and
runtime code lives under `nova/backends/trainium/{core,utils}/`. A
repo-wide AST guard in `scripts/test_imports.sh` rejects any
reintroduction.

## Non-negotiable rules

**Modeling code import allowlist** (enforced by `scripts/test_imports.sh`'s
repo-wide AST guard plus per-model AST tests in
`tests/unit/test_modeling_*.py`). Files under `nova/models/<name>/` may
import only from: `torch`, stdlib, `diffusers`, `transformers`, and
`nova.ops`. Forbidden roots: `neuronx_distributed`, `nkilib`,
`torch_neuronx`, `nova.core`, and `nova.utils.{compile_env,
runtime_env, distributed, snapshot}` (removed paths — the guard rejects
any reintroduction). If a primitive is missing, add it to `nova.ops`
first (Trainium impl required, CPU reference preferred), then use it
from the model.

**Runtime invariants** (violations produce silent SIGSEGVs or
`global communicator` errors, not Python tracebacks):

1. One Python process, multiple visible NeuronCores
   (`NEURON_RT_NUM_CORES=N python …`). Do **not** use `torchrun` for the
   diffusion path — its MPMD model is incompatible with these traced
   artifacts.
2. All components in a pipeline must share the same `world_size`. For
   Flux at TP=4: T5 + transformer run TP=4/DP=1; CLIP + VAE run
   TP=1/DP=4 — both yield `world_size=4`. Mixing `world_size=1` and
   `world_size=4` artifacts in one process crashes during weight init.
3. Component load order matters. The first component to load fixes the
   process-wide NeuronCore communicator, so tensor-parallel components
   must load **before** replicated ones (Flux:
   `text_encoder_2 → transformer → text_encoder → decoder`). The
   pattern is enforced inside `nova/models/flux/application.py`; new
   ports must follow it.

The 4-core `trn3pd98.3xlarge` cannot fit Wan's TP=4 transformer + TP=1
VAE in one process — `scripts/wan_smoke.sh` splits into two sequential
stages and shuttles latents through `.nova-cache/wan_smoke_latents.pt`.

## Compile cache

`~/.cache/nova/<model>/<sha256-prefix>/` (override with
`NOVA_COMPILE_CACHE` or `compile_cache_dir=`). The key hashes model id +
parallel config + dtype (normalized) + shape + toolchain versions
(`torch`, `neuronx-cc`, `neuronx-distributed`, `nki`, `libneuronxla`,
`torch-neuronx`, `torch-xla`, `diffusers`, `transformers`, Python
major.minor). `model_path` and Python patch version are stored in the
manifest only — caches survive Python patch upgrades and are portable
across hosts. Pass `force_compile=True` to bypass.

## Adding a new model

`README.md` §"Adding a new model" is the canonical checklist:
`nova/models/<name>/{application.py, pipeline.py, modeling_<name>.py,
entry.py}` plus a `@register_model(...)` line in `nova/registry.py`
with `backends=("trainium", ...)`. Wan is the most recent worked
example — mirror its layout, not Flux's (Flux predates the
`nova.ops`-only rule and the upstream-fork formatting carve-out).
Lifting the multi-component compile/load orchestration into a shared
base class is a deferred Phase B/M3 cleanup
(`cclogs/m2-wan/17-M2-wan-spike-retro.md` §2.2).

## Logs and history

`cclogs/` is the per-milestone session log, indexed by
`cclogs/README.md` and entry-pointed by `cclogs/00-plan.md`. Folders:
`m0-m1/`, `backend/`, `m2-wan/`. Add new logs with monotonic numeric
prefixes inside the milestone folder that owns the work. The retros
(`06-M1-closure.md`, `17-M2-wan-spike-retro.md`,
`23-phase-b-closure.md`) capture decisions that aren't otherwise in
code.
