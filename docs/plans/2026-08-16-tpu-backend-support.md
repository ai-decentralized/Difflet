# TPU Backend Support Plan

Date: 2026-08-16

## Goal

Add TPU as a second real hardware backend for Difflet, reusing the existing
modeling code unchanged. This plan targets the **`difflet.ops` path**: model
files (e.g. `difflet/models/wan/modeling_wan.py`) do not change at all;
only the backend implementation layer underneath `difflet.ops` and the
component compile/load lifecycle are new.

**Scope caveat (see "Model portability audit"):** the zero-change claim holds
only for the newer-generation models (Wan, HunyuanVideo, Qwen-Image, LTX-2
backbones). Flux is a legacy NxDI fork whose modeling file imports Trainium
code directly and is *not* covered by the zero-change claim.

## Why this path and not a JAX-native model

Two paths were considered for TPU support:

1. **Model expressed in JAX/Flax**, using JAX's native GSPMD parallelism.
   This bypasses almost the entire existing stack — `difflet.ops` is a
   PyTorch-tensor-in/out contract, `MultiComponentApplication`/
   `NeuronApplicationBase` are `torch.nn.Module`-based, and the Pipeline
   classes (e.g. `NeuronFluxPipeline`) subclass diffusers' PyTorch
   pipelines. A JAX model would require re-porting every DiT architecture
   from scratch and building an independent compile/serve stack, sharing
   only the outermost CLI/registry/serving shell.
2. **Model expressed via `difflet.ops`** (this plan). The modeling layer is
   already backend-neutral by design (`difflet.ops` is documented as a
   "frozen v1, additive-only" op surface specifically so a second hardware
   backend can be added without touching model code). The work is scoped to
   `difflet/backends/tpu/` plus a TPU-specific component lifecycle class.
   This is also the first real test of whether that abstraction is actually
   hardware-neutral — `cuda`/`rocm` are unimplemented stubs today, so the
   frozen contract has never been exercised against a second real backend.

## Existing contract this plan must fit into

Three nested layers, from loosest/most portable to most hardware-specific:

1. **`DiffletPipeline` ↔ `Application`** (`difflet/pipeline/difflet_pipeline.py`) —
   duck-typed, reflection-based (`inspect.signature`). Minimum required
   surface: `app.compile(path[, debug])`, `app.load(path[, start_rank_id,
   local_ranks_size, skip_warmup])`, optional `app.has_compiled_artifacts(path)`,
   and `app(*args, **kwargs)`. Hardware-neutral today; must stay that way.
2. **`MultiComponentApplication` ↔ `ComponentSpec.component`**
   (`difflet/backends/trainium/core/multi_component_application.py`) —
   generic race-safe compile, load-priority ordering, artifact validation.
   Algorithm is hardware-neutral, but it reads
   `spec.component.config.neuron_config.{tp_degree,world_size,start_rank_id,
   local_ranks_size}` directly, which is NxD's `NeuronConfig` shape. A TPU
   backend needs either a parallel class with the same algorithm, or a
   compatible config shape.
3. **`NeuronApplicationBase`** (`difflet/backends/trainium/core/application_base.py`) —
   100% NxD/NEFF-specific: `ModelBuilder.trace()`, `model.pt`,
   `nxd_model.initialize()`, safetensors weight sharding. Not reusable;
   needs a TPU-specific equivalent.

## Model portability audit (verified 2026-08-16)

The repo contains **two generations of model code** with very different TPU
portability. Verified by grepping for `backends.trainium` / `torch_neuronx` /
`neuronx_distributed` imports:

- **New-style (backend-neutral) modeling** — `modeling_wan.py`,
  `modeling_hunyuan_video.py`, and the Qwen-Image / LTX-2 backbone files
  import only `torch`, pure-torch diffusers utilities, and `difflet.ops`;
  their docstrings explicitly commit to this. Trainium-specific wrappers live
  separately under `difflet/backends/trainium/<model>/` (e.g.
  `wan/backbone.py`, `wan/text_encoder.py`, `wan/vae.py`). These are the
  models the zero-change claim actually applies to.
- **Legacy NxDI fork (Flux)** — `modeling_flux.py` fuses the modeling and
  the application in one file and imports `NeuronApplicationBase`,
  `ModelWrapper`, `core.config`, and the layer-boundary markers straight from
  `difflet.backends.trainium`. It also sets `NEURON_PLATFORM_TARGET_OVERRIDE`
  at import time. Its sub-encoders (`flux/t5`, `flux/clip`, `flux/vae`) have
  the same coupling, and `difflet/layers/normalization.py` (used by Flux)
  unconditionally imports the Trainium marker module — so **importing Flux
  modeling on a host without the Neuron toolchain fails outright**.
- **In between** — every model's `application.py` imports Trainium core
  (expected; Phase 3 replaces that layer), and some non-backbone components
  of new-style models are not clean either (e.g. `wan/umt5/modeling_umt5.py`,
  `wan/checkpoint/`).

Consequences:

1. The first end-to-end TPU model must be a new-style model, **not Flux**
   (Phase 4 revised accordingly).
2. Porting Flux to TPU is a separate "legacy untangling" workstream: either
   lift its modeling onto `difflet.ops` (conflicts with the repo's
   keep-upstream-formatting-for-rebases policy for `difflet/models/flux/`)
   or write a parallel implementation. Explicitly out of v1 scope.
3. The layer-boundary marker (`ModuleMarkerStartWrapper`/`EndWrapper`) needs
   to become a `difflet.ops` primitive with a no-op CPU/TPU implementation
   before any Flux-family code can even be imported off-Neuron.
4. Note the current "CPU backend" runs *inside the Neuron venv* (it swaps
   `ops_impl` dispatch, not the installed packages), so the codebase has
   likely never been imported on a host without the Neuron toolchain. A TPU
   host is the first such host; expect import-time breakage beyond what the
   audit above lists, and budget for it.

**Quarantine mechanics (verified):** ignoring the legacy files on TPU costs
nothing, because the codebase is lazy end-to-end — `difflet/__init__.py`
lazy-exports via module `__getattr__`; registry factories are lazy import
strings (registering Flux imports nothing); `difflet.ops` dispatch imports
only `difflet.backends.<selected>.ops_impl.*`; and the CLI imports
orchestrators inside `_get_orchestrator()`, with model imports inside
methods. Crucially, `from_pretrained` calls `entry.require_backend(...)`
*before* `entry.create_application(...)`, so simply **not** adding `"tpu"`
to Flux's registry `backends` tuple yields a clean
"does not support backend 'tpu'" error without ever importing
`modeling_flux.py`. The Trainium-marker-importing `difflet/layers/` files
are only referenced by Flux modeling and are quarantined with it. Residual
boundaries: shared code we *want* to reuse (e.g. the
`MultiComponentApplication` algorithm, which imports Trainium utils at
module top) must be copied, not imported; CLI startup has Trainium-flavored
steps (jemalloc re-exec, Neuron prewarm) to gate by backend in Phase 6; and
Flux-touching unit tests need a skip marker on TPU CI.

## Phase overview

| Phase | Goal | Key output | Depends on |
|---|---|---|---|
| 0 | De-risk the biggest unknown | Written decision memo | — |
| 1 | Backend scaffolding | `difflet/backends/tpu/` stub, registered | Phase 0 (partial) |
| 2 | `ops_impl/` implementation | collectives → linear/norm/embeddings → attention → platform | Phase 1 |
| 3 | Component lifecycle | `TpuApplicationBase`, replacing `NeuronApplicationBase` | Phase 0 decision |
| 4 | End-to-end single model (new-style, not Flux) | One model generates real output on TPU, tp=1 | Phase 2 + 3 |
| 5 | Parallelism + numerical validation | TP works; CPU-reference parity | Phase 4 |
| 6 | CLI/serving integration | `difflet run --backend tpu`, verify matrix | Phase 5 |

## Phase 0 — De-risk the compile/load model (no code yet)

**Core question:** what does `compile()` actually produce on TPU, and what
does `load()` read back, given the existing two-phase contract assumes NxD's
explicit AOT model (trace once → NEFF on disk → load separately), while
`torch_xla` on TPU traditionally does lazy execution with a persistent HLO
cache (compile-on-first-call)?

Two candidate directions:

- **Direction A — explicit AOT export.** Use `torch_xla`'s newer AOT export
  path (`torch.export` + StableHLO export) to produce a serializable artifact
  at `compile()` time, loaded independently at `load()` time. Closest fit to
  the existing contract; API maturity for multi-component models and dynamic
  shapes is unverified.
- **Direction B — lazy execution + persistent cache.** `compile()` becomes
  "run a warm-up forward pass so `XLA_PERSISTENT_CACHE_PATH` is populated";
  `load()` becomes "verify the cache directory exists"; real compilation
  happens lazily on the first real `__call__()`. More natural fit for how
  `torch_xla`/TPU is normally used, but weakens the "compile and load are
  independent phases" assumption, and `has_compiled_artifacts()` no longer
  means "compilation is actually done." Also note: warm-up compilation needs
  materialized weights on device (random values suffice), unlike NxD's
  `initialize_model_weights=False` trace — this changes the memory profile
  of `difflet compile`.
- **Direction C — torchax / torch_xla2 (torch-on-JAX interop).** Run the
  torch modules through the JAX runtime (`jax.jit` AOT + JAX persistent
  compilation cache), which is the direction Google has been pushing for
  torch-on-TPU. Would change Phase 2's implementation substrate (collectives
  via JAX under `shard_map` rather than `xm.*`). Maturity as of writing is
  unverified — check current status during the spike before investing.

**Action:** a throwaway spike script (not merged, similar in spirit to
`scripts/mx_smoke.py`) that takes one small module and tries the directions.
**The spike module must include at least one collective op** (e.g. an
`all_gather` in a 2-device mesh) — a single-core-only spike would give false
confidence, because exporting/caching graphs that contain collectives is
exactly where AOT-export paths tend to break, and the real models are full
of them. Record: compile time, whether the artifact survives a fresh process
(`load()` in a new process without recompiling), and whether "load"
genuinely avoids recompilation. This decision determines the class design in
Phase 3 — spike before designing.

**Secondary decisions** (can proceed in parallel, don't block Phase 1–2):

- Hand-rolled Megatron-style parallel layers (via `torch_xla`
  `mark_sharding`) vs. GSPMD auto-sharding for TP — defer to Phase 5.
- `BackendCapabilities.supports_torchrun_mpmd` for TPU — likely the
  **opposite** of Trainium's `False`. `torch_xla` traditionally runs
  multi-core via `xmp.spawn`/`torchrun` (one process per device), which is
  exactly what Trainium's runtime protocol forbids. This must be decided
  explicitly in Phase 1, not copied from `TrainiumBackend`.
- Single TPU host vs. multi-host TPU pod — recommend explicitly excluding
  multi-host from v1 scope; it adds a materially different runtime topology.

## Phase 1 — Backend scaffolding

Mirror the existing `cuda`/`rocm` placeholder pattern:

```
difflet/backends/tpu/
├── __init__.py
├── runtime.py          # TpuBackend(BackendRuntime); capabilities per Phase 0
└── ops_impl/            # empty + __init__.py, filled in Phase 2
```

```python
# difflet/backends/registry.py
_BACKEND_FACTORIES = {
    ...,
    "tpu": "difflet.backends.tpu.runtime:create_backend",
}
```

Three more small changes that are easy to miss (verified against the code):

- **Registry `backends` tuples.** Every `ModelEntry` currently pins
  `backends=("trainium",)` (or defaults to it), and
  `entry.require_backend("tpu")` raises `ValueError` before
  `prepare_runtime` is ever reached. The target model's registration in
  `difflet/registry.py` must add `"tpu"` to its tuple.
- **`_auto_detect_backend()`** (`difflet/backends/registry.py`) has no TPU
  branch: on a TPU host (no `torch_neuronx`, no CUDA, no HIP) it silently
  falls back to `"trainium"`. Add `torch_xla`/`libtpu` detection.
- **`CacheSpec` hashing** (`difflet/pipeline/compile_cache.py`): the
  toolchain list is Trainium-specific (`neuronx-cc`, `torch-neuronx`, ...).
  On a TPU host those resolve to `None`, which accidentally differentiates
  the hash, but the **backend name itself is not part of the key**. Add the
  backend name explicitly plus the TPU-relevant packages (`torch-xla`,
  `libtpu`) rather than relying on absent-package `None`s.

**Exit criterion:** `DiffletPipeline.from_pretrained(..., backend="tpu")` on
a model whose entry lists `"tpu"` passes `require_backend` and reaches
`get_backend("tpu").prepare_runtime()`, failing with an explicit "not
implemented" rather than an unrelated error.

## Phase 2 — `ops_impl/` implementation, highest-reuse first

**2a. Collectives (highest reuse).** The collective calls already used in
`difflet/backends/trainium/core/modules/attention/attention_base.py`
(`xm.all_to_all`, `xm.all_gather`, `xm.reduce_scatter`, `xm.collective_permute`)
are standard `torch_xla` APIs — TPU is their native target, more mature than
the Neuron PJRT plugin path. Work here is mostly building
`difflet/backends/tpu/ops_impl/collectives.py` (`init_parallel_mesh`,
`get_cfg_group`, `get_cp_group`, etc. against `torch_xla.runtime`'s mesh API
instead of NxD's `parallel_state`), not rewriting the communication logic
itself. Still need to verify that `groups=`/`pin_layout=` semantics match
between the Neuron and TPU PJRT plugins — don't assume, write a unit test.
(If Phase 0 selects Direction C/torchax, this subsection's substrate changes
to JAX collectives under `shard_map`; the reuse-ordering logic still holds.)

**2b. `linear.py` / `norm.py` / `embeddings.py`.** Current implementation is
a thin wrapper over `neuronx_distributed.parallel_layers.layers`
(`difflet/ops/linear.py:3`). For TPU: start with a plain, non-parallel
implementation (`tp_degree=1` behavior — ordinary `nn.Linear`/`nn.LayerNorm`)
and defer actual tensor-parallel sharding to Phase 5. This unblocks an
end-to-end result faster.

**2c. `attention.py`.** Single-core (no CP) is a plain
`F.scaled_dot_product_attention` or equivalent XLA op. Defer ring/ulysses/
gather_kv CP modes to Phase 5.

**2d. `platform.py`.** Small — add a TPU branch to `get_platform_target()`/
`hardware()`.

**Exit criterion:** each `ops_impl/*.py` file has a corresponding
`tests/unit/ops/` test (mirror the existing Trainium test structure with the
backend swapped); CPU-reference vs. TPU numerical sanity check on small,
non-distributed tensors.

## Phase 3 — Component lifecycle: `TpuApplicationBase`

Highest-uncertainty phase; driven directly by the Phase 0 decision. Target:
a `TpuApplicationBase` (parallel to `NeuronApplicationBase`) implementing:

```python
class TpuApplicationBase(torch.nn.Module):
    def compile(self, compiled_model_path: str, debug: bool = False) -> None: ...
    def load(self, compiled_model_path: str, start_rank_id=None,
             local_ranks_size=None, skip_warmup=False) -> None: ...
    def has_compiled_artifacts(self, compiled_model_path: str) -> bool: ...
```

Signatures must match the existing top-layer contract exactly so
`DiffletPipeline` can call it without modification. Internals are fully
independent of NxD, following whichever direction Phase 0 selected.

`convert_hf_to_neuron_state_dict()`-style weight remapping (see the Flux
`proj_out_attn`/`proj_out_mlp` split in `modeling_flux.py:1697`) will likely
still be needed — if TPU-side parallel layers split/fuse weights differently
than NxD does, the remapping function needs a TPU-specific rewrite even for
architecturally identical models. Don't assume the Trainium version transfers.

**Multi-component models** (Wan/HunyuanVideo/Qwen-Image): decide whether to
port `MultiComponentApplication`'s algorithm (rewrite `_compiled_config_matches`
etc. against whatever config shape the TPU components expose) now, or narrow
v1 scope to single-process models only (Flux/LTX-2) and defer multi-component
lifecycle. **Recommend narrowing** — prove out the single-component path
first.

**Exit criterion:** a toy model (a plain `nn.Linear`, not a real DiT) round-trips
through compile → save → restart process → load → forward, confirming the
artifact genuinely survives a fresh process. This is the concrete check on
whether the Phase 0 direction was correct.

## Phase 4 — End-to-end: a new-style model first (NOT Flux)

An earlier draft of this plan picked Flux ("only single-process model,
lowest complexity"). The portability audit reversed that: Flux is the legacy
NxDI fork whose modeling imports Trainium code directly — it is the *least*
portable model in the repo and is deferred to a separate legacy-untangling
workstream.

**Candidate first models** (all have backend-neutral DiT backbones):

- **Qwen-Image** — image output (simplest artifact path), guidance-distilled
  (single forward pass, no CFG branch). Its 3-stage CLI orchestration is a
  Trainium co-fit workaround, not an architectural requirement; via the
  `DiffletPipeline` API the components compose in one process.
- **LTX-2** — the precedent for single-process composition even on Trainium;
  video output adds mp4/temporal-VAE complexity.
- **Wan** — backbone modeling is the cleanest (explicitly documented
  backend-neutral), but its UMT5 encoder (`wan/umt5/modeling_umt5.py`) is
  Trainium-coupled and true-CFG adds a second forward branch.

The choice also depends on whether the model fits at tp=1 in the target TPU
generation's HBM (v5e 16 GB vs. v5p ~95 GB per chip) — check memory fit
before committing; this is listed under open decisions.

Steps (whichever model is chosen):

1. Reuse the model's backbone modeling file unchanged on top of the TPU
   `ops_impl`. Any place it turns out to bypass `difflet.ops` is concrete
   evidence the "frozen" contract has a leak — log it and patch the ops
   surface (additive-only).
2. **Simplification for v1 at tp=1:** run the text encoder(s) and VAE as
   stock HF `transformers`/`diffusers` modules on the XLA device, and route
   only the DiT backbone through difflet modeling. The custom parallel
   encoder implementations only pay off when sharding; at tp=1 they add
   porting cost for nothing. This sidesteps e.g. Wan's Trainium-coupled UMT5.
3. Add a TPU branch to the model's `entry.py` factory that swaps the Neuron
   application for a TPU-backed equivalent. Note: all current applications
   extend `MultiComponentApplication` (even single-process ones — Flux
   composes clip/t5/transformer/decoder), so the multi-component question
   from Phase 3 cannot be fully avoided regardless of model choice.
4. Get one full generation working at tp=1, small shape, bf16 or fp32 —
   correctness first, performance later.

**Exit criterion:** `difflet run --model-id <chosen model> --backend tpu
--tp-degree 1 ...` produces recognizable output, not noise.

## Phase 5 — Numerical validation + parallelism

- **Numerical validation:** reuse the `tests/numerical/` pattern, CPU
  reference as the shared oracle. Per the existing hard-won lesson in
  `DEVELOPER.md`, always validate against the **diffusers reference**, never
  TPU-vs-Trainium self-comparison (two independently-buggy implementations
  can agree with each other and both be wrong). Target cosine ≥ 0.9995,
  matching the existing Trainium bar.
- **Parallelism:** implement the TP layer deferred from Phase 2b (hand-rolled
  vs. GSPMD, per Phase 0). Don't try to port every Trainium parallel mode
  (gather_kv/ring/ulysses/cfg-parallel/sp) at once — land TP first, prioritize
  the rest afterward.

## Phase 6 — CLI/serving integration

- Add `--backend tpu` validation/routing to `difflet/cli/main.py`.
- Add a TPU branch (or standalone script) mirroring `scripts/verify_cli.py`'s
  download→compile→timed-generate matrix, producing a `results.json`.
- Add a "TPU Runtime protocol" section to `DEVELOPER.md`, mirroring the
  Trainium section's three hard rules — in particular documenting whatever
  Phase 0 decided about `supports_torchrun_mpmd`, since it likely inverts the
  Trainium rule.

## Open decisions needing a team call (not decidable from the codebase alone)

1. ~~Compile model: AOT export vs. lazy-execution + persistent cache vs.
   torchax/JAX interop (Phase 0)~~ — **SETTLED 2026-08-16 by the spike:
   Direction A (AOT StableHLO export) + a mandatory opaque-custom-op wrapper
   for collectives. B is broken on torch_xla; C's eager path is broken at
   torch 2.9. See the status log for the measurements and the caveat that
   load still pays backend compilation.**
2. Hand-rolled parallel layers vs. GSPMD auto-sharding — affects Phase 5
   effort and the resulting performance profile.
3. First end-to-end model (Qwen-Image vs. LTX-2 vs. Wan) — gated on a
   memory-fit check against the target TPU generation (v5e vs. v5p HBM).
4. Target TPU generation itself — determines whether tp=1 is even viable
   for the larger DiTs, which feeds back into decision 3.
5. Single TPU host vs. multi-host pod — recommend excluding multi-host from
   v1.
6. Whether/when to fund the Flux legacy-untangling workstream (lifting
   `modeling_flux.py` and `difflet/layers/` off their direct Trainium
   imports) — required before Flux can join the TPU matrix, and it conflicts
   with the keep-upstream-formatting-for-NxDI-rebases policy.

## Recommended first step

The Phase 0 spike. It's the one step where being wrong invalidates the class
design in Phase 3 and possibly the phase ordering itself; everything else in
this plan is comparatively low-risk sequencing. Worth resolving before
committing engineering time to the rest.

## Status log

### 2026-08-16 — plan written, reviewed, Phase 1 implemented

**Plan revisions after code-verified review** (all reflected in the sections
above):

- Portability audit added: two generations of model code identified; Flux is
  the legacy NxDI fork and the *least* portable model — Phase 4 first model
  switched from Flux to a new-style model (Qwen-Image / LTX-2 / Wan).
- Quarantine mechanics verified end-to-end: lazy imports + the
  `require_backend` guard mean the legacy files are ignored on TPU at zero
  cost (confirmed by assertion: requesting Flux with `backend="tpu"` raises
  a clean `ValueError` and `difflet.models.flux` never enters `sys.modules`).
- Phase 0 gained Direction C (torchax) and the requirement that the spike
  graph contain a collective op.
- Phase 1 gained three concrete fixes: registry `backends` tuples,
  `_auto_detect_backend` TPU branch, `CacheSpec` backend keying.

**Phase 1 — implemented and locally verified** (working tree, not committed):

- New: `difflet/backends/tpu/{__init__,runtime}.py` (`TpuBackend` stub;
  capabilities marked provisional pending Phase 0; `prepare_runtime` raises
  `NotImplementedError` pointing at this plan), `difflet/backends/tpu/ops_impl/`
  (empty until Phase 2).
- New: `tests/unit/backends/test_tpu_backend.py` — 8 tests: backend
  resolution, auto-detect precedence (Neuron-first; `torch_xla`+`libtpu` →
  tpu), cache-key stability, Flux quarantine guard. To run in the Neuron
  venv / CI (the local dev Mac has no torch/pytest; verified there with
  equivalent plain-python assertions instead — all four checks passed).
- Modified: `difflet/backends/registry.py` (factory entry + auto-detect
  branch), `difflet/pipeline/compile_cache.py` (`CacheSpec.backend` field,
  additive-only: `None`/`"trainium"` keys byte-identical to pre-change,
  `"tpu"` adds backend name + `libtpu` version to the hash),
  `difflet/pipeline/difflet_pipeline.py` (passes `backend_runtime.name` into
  `CacheSpec`).
- Deliberately NOT done: no model's registry entry lists `"tpu"` yet — that
  lands with the Phase 4 model decision.

**Phase 0 — spike script ready, awaiting hardware:**

- `scripts/tpu_phase0_spike.py` (throwaway probe, never imported by the
  package). Probe module contains an `all_gather`. Direction A: torch.export
  → StableHLO export → save → load. Direction B: persistent compilation
  cache; run the script **twice** — a marker file records the first run's
  cold-compile time so the second process can tell whether compilation was
  actually skipped. Direction C: torchax availability probe.
- Needs a Cloud **TPU VM** (the current architecture: SSH into a host with
  directly-attached TPU chips — the GCP analog of the `trn2.3xlarge`
  workflow). Smallest VM suffices for a first pass; re-run on ≥2 chips
  (e.g. `v5litepod-4` / `v4-8`) before trusting the collective-export
  result. Colab/Kaggle TPU runtimes work for a weaker free-tier pass.

**Next actions:**

1. Run the spike on a TPU VM (twice), bring back the three direction
   verdicts → settles open decision 1.
2. Design `TpuApplicationBase` (Phase 3) from the verdict.
3. Start Phase 2 `ops_impl/` (collectives first) in parallel once the
   direction is known.

### 2026-08-16 (later) — Phase 0 spike RUN; open decision 1 settled

**Hardware:** no provisioning was needed — the dev box *is* a TPU VM.
`machine-type n2d-192-112-v5lite-tpu`, `accelerator-type v5litepod-4`:
4× v5e chips, 2x2 topology, single host, us-west4-a, chips exposed via
`/dev/vfio/{0..3}`. This satisfies the plan's "re-run on ≥2 chips before
trusting a PASS" requirement directly.

**Toolchain:** torch 2.9.0+cpu, torch_xla 2.9.0, libtpu 0.0.21 (the exact
version `torch_xla[tpu]==2.9.0` pins — not a mismatch of our making). Note
this matches the repo's existing torch 2.9 baseline.

#### Direction A — AOT StableHLO export: **VIABLE, with a required wrapper**

| Probe | Result |
|---|---|
| Collective-free graph: export → StableHLO → save → load → execute | PASS |
| Same graph with raw `xm.all_gather` | **FAIL** at `torch.export` |
| Same graph, collective wrapped as opaque custom op | PASS |
| 4-chip mesh (4 procs, world=4), save then load in **fresh** processes | PASS |

The raw-collective failure is exactly what the plan predicted:
`RuntimeError: The tensor has a non-zero number of elements, but its data is
not allocated yet ... it is likely that we are erroneously tracing into a
custom kernel.` The error message names its own fix, and the fix works:
wrapping the collective with `torch.library.custom_op` + `register_fake`
makes `torch.export` treat it as opaque. Crucially the collective is **not**
left as an unlowered `custom_call` — the saved StableHLO contains a real
`all-gather` with `replica_groups`, i.e. it genuinely compiles.

On the real 2x2 mesh via `torch_xla.launch` (4 processes, world=4): all 4
ranks export+save; 4 *fresh* processes then load (0.00s) and execute
(0.25–0.36s) producing shape `(16, 16)` = 4 batch × 4 world — a correct
4-way gather. **The artifact survives the process boundary.**

*Consequence for Phase 2a:* every collective in `ops_impl/collectives.py`
needs an opaque-custom-op wrapper (with a `register_fake` shape rule) if the
graph is ever to be exported. This is a new, concrete work item.

#### Direction B — lazy execution + persistent cache: **FAILS its core promise**

Both entry points (`xr.initialize_cache` and `XLA_PERSISTENT_CACHE_PATH`)
write cache files, but the second process logs:

```
Failed to deserialize executable: UNIMPLEMENTED: Deserializing serialized
executable not supported.
```

Cold vs. warm first-step timings across process boundaries: 1.26s → 1.26s,
0.51s → 0.51s, 1.30s → 1.31s. **Zero cross-process benefit.** Two controls
were run to place the blame correctly:

- **Not a libtpu version issue** — reproduced identically with libtpu 0.0.17
  (the version JAX pulls) against the same torch_xla 2.9.
- **Not a hardware or XLA limitation** — JAX on the same v5e chips with an
  equivalently-shaped probe goes 1.30s cold → **0.08s warm across a fresh
  process** (16×). Executable serialization on v5e works fine.

The gap is specifically `torch_xla`'s PJRT computation client.

#### Direction C — torchax / torch-on-JAX: **split verdict**

- Eager/device interop (`torchax.default_env()` + `.to("jax")`) **hard-crashes**
  with torch 2.9: `SIGABRT`, `c10::Error` at `getDeviceGuardImpl`. torchax
  0.0.7 is evidently not built against torch 2.9. Not usable today.
- Export interop (`torch.export` → `torchax.export.exported_program_to_jax`
  → `jax.jit`) **PASS** (0.05s on TPU) — and it inherits JAX's working
  persistent compilation cache.

#### The decisive cross-cutting finding

**On this stack, nothing avoids XLA backend compilation at load time.**
Direction A persists the *traced/lowered graph*, not the compiled
executable: `load` is 0.00s but `exec` still pays 0.27s compiling
StableHLO → TPU executable. Direction B, which is the only mechanism that
would cache the executable itself, is broken on torch_xla. So via torch_xla
today, `difflet compile` **cannot** produce an artifact meaning "compilation
is actually done" — `has_compiled_artifacts()` can only ever mean "the graph
is exported," with backend compile paid on every process start.

This is the single most consequential input to the Phase 3 class design, and
it is the one thing that would push toward Direction C: the JAX runtime is
the only path measured here that gets a working persistent compile cache.

#### Verdict / recommendation

Adopt **Direction A** (torch.export → StableHLO) as the compile front-end,
plus a mandatory opaque-custom-op wrapper layer for every collective. Two
reasons beyond its own PASS: it is the closest fit to the existing
`compile()`/`load()` contract, and `torch.export` is *also* Direction C's
front-end — so committing to it keeps the JAX escape hatch open if the
backend-compile-caching problem later proves decisive. Treat "load still
recompiles" as a known, documented v1 limitation rather than a blocker.

#### Other findings worth carrying into Phase 2

1. **Phase 2a's reuse premise holds.** All four collectives named in the plan
   (`all_gather`, `all_to_all`, `reduce_scatter`, `collective_permute`)
   still exist in torch_xla 2.9. But `xm.get_ordinal()` and
   `xm.xrt_world_size()` were **removed** — they are now
   `xr.global_ordinal()` / `xr.world_size()`.
2. **`groups=` semantics do NOT transfer from Neuron.** TPU's HLO verifier
   rejects a replica group that doesn't cover every replica:
   `RET_CHECK failure ... In cross_replica mode, replica groups should
   contain 4 replicas, but found 1`. The plan's "don't assume, write a unit
   test" is now confirmed necessary, not precautionary.
3. **A failed `torch.export` poisons XLA process state** — a later
   `mark_step()` SIGSEGVs. Compile-path probes (and probably the real
   compile path) must run in their own process; this also explains the
   initial confusing spike crash.
4. `scripts/tpu_phase0_spike.py` was fixed for the removed `xm.get_ordinal`
   (now `xr.global_ordinal`). Its single-participant `groups=[[ordinal]]`
   is invalid whenever >1 chip is visible — it needs either the 1-chip env
   override (`TPU_VISIBLE_CHIPS=0` + `TPU_*_BOUNDS=1,1,1`) or an
   all-replica group.

### 2026-08-16 (later still) — Phase 2a implemented and hardware-verified

**New:** `difflet/backends/tpu/ops_impl/{parallel_mesh,collectives}.py`,
`tests/unit/backends/test_tpu_collectives.py` (17 tests).

- **Mesh math is reused, not reimplemented.** `difflet/pipeline/parallel_mesh.py`
  turned out to be genuinely backend-agnostic, and `MeshSpec.axis_groups()`
  already returns a *full partition* of the world — exactly what XLA's HLO
  verifier demands. The TPU mesh module is thin: no `torch.distributed`
  groups (torch_xla collectives take replica-group lists directly), and it
  *does* own TP, which the Trainium manager delegates to NxD's
  `parallel_state`.
- **Every collective is wrapped in an opaque custom op**, per the Phase 0
  finding. Two mechanical consequences: the custom-op schema has no
  `int[][]`, so replica groups cross the boundary flattened; and the fake
  (meta) rules must encode each collective's shape math.
- **Verified on the real 2x2 v5e mesh, 4 processes: 40/40 checks pass** —
  numerics for `gather_tp_dim` / `reduce_tp` / `scatter_tp_dim` /
  `reduce_scatter_to_sequence_parallel_region`, plus `torch.export` on a
  module using them (collectives preserved as opaque nodes) and StableHLO
  save with the collectives fully lowered (no residual `custom_call`).
- **Two bugs the process caught, worth recording:**
  1. The partial-replica-group guard first checked only that member ranks
     were contiguous from 0 — which *passes* `[[0]]` on a 4-replica world,
     the exact case XLA rejects. It must compare against the mesh world size.
     Now a regression test.
  2. `tests/unit/test_no_dp_parasites.py` (an existing architecture lint)
     rejected a `get_dp_group` re-export. The dp axis must have no named
     accessor outside the Trainium manager; the TPU module now exposes none.
     Note the lint is a raw token search, so even naming it in a comment
     trips it.
- Full unit suite: 1635 passed / 107 failed — the 107 is the unchanged
  off-Trainium baseline (see the earlier entry); no regressions.

### 2026-08-16 (later still) — Phase 2b/2c/2d implemented, TP moved earlier

Proceeding on the TP-first reading of the sequencing problem below: TP is
implemented **now** rather than deferred to Phase 5, because no candidate
model fits on a 16 GB v5e chip at tp=1.

**New:** `difflet/backends/tpu/ops_impl/{linear,norm,embeddings,attention,platform}.py`
and `tests/unit/backends/test_tpu_layers.py` (19 tests).

- **`linear.py` shards for real**, Megatron-style, rather than the CPU
  backend's tp=1 stand-in: `ColumnParallelLinear` splits the output dim
  (optional `gather_output`), `RowParallelLinear` splits the input dim and
  finishes with an all-reduce, `ParallelEmbedding` shards vocab (masked
  lookup + all-reduce) or the embedding dim. Bias on `RowParallelLinear` is
  added **after** the reduce; folding it in before would add it `tp` times.
- **`attention.py`** is SDPA-based and local — under TP the heads are already
  split by the surrounding projections, so no collective belongs inside
  attention. CP modes (ring/ulysses/joint) raise `NotImplementedError`
  pointing at Phase 5 rather than silently degrading.
- **`norm.py`/`embeddings.py`** are pure torch and deliberately byte-identical
  to the CPU backend's math, so the CPU oracle stays valid for Phase 5.
- **Verified on the real 2x2 v5e mesh: 44/44** — every parallel layer at tp=4
  reproduces the *full-weight* `nn.Linear`/`nn.Embedding` oracle (not a
  tp=1 self-comparison, which would prove nothing), plus attention, RMSNorm,
  and `torch.export` of a TP block.

**The most consequential finding: XLA's default TPU matmul precision is not fp32.**

The MXU multiplies fp32 inputs at reduced precision unless told otherwise.
Measured on v5e: ~1e-2 absolute error per matmul at the default, versus ~2e-6
with `torch_xla.backends.set_mat_mul_precision("highest")`. Compounded through
a DiT this makes the Phase 5 target (cosine ≥ 0.9995 vs. the diffusers
reference) unreachable — and it fails *silently*: every shape is correct and
nothing raises. This is now `platform.configure_matmul_precision()`, and
`TpuBackend.prepare_runtime` carries a note that Phase 3 must call it.

**Two more bugs the process caught:**

1. Layer construction read the runtime ordinal (for shard metadata and the
   embedding's vocab offset), so building a model required a live XLA
   runtime. Both are now resolved lazily — `get_axis_rank` short-circuits to
   0 on a trivial axis, and `vocab_start` moved into `forward`.
2. Parameters were allocated with `torch.empty` and left uninitialized. A test
   passed or failed depending on what ran before it, because the arena
   sometimes held NaN. Beyond the test, uninitialized parameters mean a
   checkpoint that misses a weight yields silent NaN rather than something
   obviously wrong. They are now initialized like `nn.Linear`/`nn.Embedding`,
   using the *unsharded* fan-in so sharding does not change the distribution.

Full unit suite: 1654 passed / 107 failed — 107 is the unchanged
off-Trainium baseline; the only non-dependency failures are the same three
pre-existing ones.

**Phase ordering change — TP moved from Phase 5 into Phase 2 (decided):**

Open decision 4 is answered by the hardware: **v5e, 16 GB HBM per chip**.
Every model's `default_parallel` in `difflet/registry.py` is `tp_degree=4`
or `8`, and no candidate first model's DiT fits in 16 GB at bf16 (a 14B
backbone alone is ~28 GB). Phase 4's "get one generation working at tp=1,
defer TP to Phase 5" premise is therefore not viable, and Phase 2b's "start
with plain non-parallel `nn.Linear`" inherited the problem. **TP is now
implemented in Phase 2b** (done, above); Phase 5 keeps numerical validation
and the remaining parallel modes (CP: ring/ulysses/gather_kv, cfg-parallel,
sp).

Phase 4's exit criterion changes accordingly: `--tp-degree 1` is not a
reachable milestone on v5e, so the first end-to-end run targets tp=4 on this
`v5litepod-4`. Open decision 3 (which model first) is still open, now
constrained by what fits in **4 × 16 GB = 64 GB** aggregate rather than 16 GB.

### 2026-08-16 (later still) — Phase 3 implemented, exit criterion met

**New:** `difflet/backends/tpu/core/{application_base,weights}.py`,
`tests/unit/backends/test_tpu_application.py` (19 tests).

- **`TpuApplicationBase`** implements Direction A: `compile()` exports this
  rank's graph via `torch.export` → StableHLO into
  `<path>/tpu_rank<N>/`, `load()` reloads it, `has_compiled_artifacts()`
  checks a manifest plus *every* rank's directory so a partially-written
  artifact does not read as ready. Signatures match what `DiffletPipeline`
  reflects on, and a unit test asserts the parameter *names* — the pipeline
  only passes arguments it finds by name, so a rename would silently drop
  them.
- **`weights.py`** is the `convert_hf_to_neuron_state_dict` analogue. It
  splits full host checkpoints using each parameter's own `._difflet_shard`
  declaration rather than a per-model table of weight names, so a layer and
  its loader cannot disagree. Re-sharding an already-sharded checkpoint is a
  no-op rather than a double split (the failure mode that silently yields a
  quarter of a model).
- **A manifest records the mesh**, and `load()` rejects an artifact compiled
  for a different one — including a *different factorization of the same
  world size* (tp=4 vs. tp=2×cp=2), where the collectives are wired
  differently but nothing about the shapes would complain.
- **Phase 3 exit criterion met on real hardware:** 4 processes compile and
  export at tp=4; **4 fresh processes** then load and run, matching the
  unsharded full-weight CPU oracle to **4.77e-07**. That single check covers
  weight sharding, export, the artifact process boundary, and the
  collectives at once.

`start_rank_id`/`local_ranks_size` are accepted and ignored: they encode
Trainium's MPMD rank-range protocol, and on TPU each process owns one chip
and takes its rank from the XLA runtime. Ignoring them explicitly beats
reinterpreting them into something that looks plausible.

Full unit suite: 1673 passed / 107 failed (the unchanged off-Trainium
baseline; same three pre-existing non-dependency failures). 63 TPU unit
tests, plus 128 checks on the real mesh across the three verification runs.

### 2026-08-16 — SPMD vs. one-process-per-device: measured, and it is a real fork

Everything above assumes `torch_xla.launch`, one process per chip. That
assumption was inherited from the Phase 0 spike and never justified — and it
matters, because it means `DiffletPipeline` runs N times over (scheduler
loop, encoders, VAE, output writing all duplicated). Trainium forbids exactly
that shape, and difflet's `MultiComponentApplication` is written for
single-process multi-core.

The alternative is torch_xla's **SPMD** mode: one process drives all chips,
`mark_sharding` declares the partitioning, XLA does the rest. Measured on the
v5litepod-4:

| Probe | Result |
|---|---|
| Single process sees all 4 chips, `xr.use_spmd()` | PASS |
| `mark_sharding` + forward | PASS |
| Sharded result vs. unsharded CPU oracle | PASS (1.07e-06) |
| `torch.export` → StableHLO **under SPMD** | **FAIL** |
| Export in a non-SPMD process, load under SPMD | **FAIL** |

Both failures are the same: `RuntimeError: Check failed:
string_to_device_.find(device) != string_to_device_.end(): Unknown device
SPMD:0`. Splitting the phases across processes does not help — the StableHLO
path simply does not understand the virtual SPMD device.

**So SPMD and Direction A are mutually exclusive on torch_xla 2.9.** That
leaves three combinations, and the choice is not obvious:

1. **Multi-process + AOT export** (what is implemented). Works today,
   end-to-end, verified. Cost: the pipeline layer must become rank-aware —
   output writing guarded to rank 0, encoders/VAE either sharded too or
   replicated on every rank (each holding a full copy). This is standard
   practice for TP inference (vLLM, Megatron-LM all run every rank's Python
   redundantly), but it is a real change to difflet's single-process design,
   and it belongs in the Phase 4/6 estimate rather than being assumed free.
2. **SPMD + lazy compile.** Fits difflet's architecture, but there is then no
   compile artifact at all, and Direction B cannot cache executables on
   torch_xla — so every process start recompiles the whole DiT. Likely
   unacceptable for serving.
3. **SPMD + JAX (Direction C).** The only combination that gets *both* the
   single-process architecture and a working persistent compile cache (JAX's
   cache measured 16× on this same hardware). Blocked today by torchax's
   eager path crashing on torch 2.9 — though its export path worked.

Recommendation: keep (1) for now, since it is implemented and verified, but
treat the rank-aware pipeline work as a known Phase 4 cost rather than a
surprise. Re-evaluate (3) if that cost turns out to be larger than it looks,
or if torchax catches up to torch 2.9.

### 2026-08-16 — Phase 4 started: model chosen, ops parity closed, DiT de-risked

**Open decision 3 settled: Qwen-Image.** Measured component sizes (from HF
file metadata, no download; note several repos ship fp32, so bf16 halves them):

| Model | transformer (bf16) | text encoder | per-chip at tp=4 |
|---|---|---|---|
| **Qwen-Image** | 38.1 GiB | 15.4 GiB | 9.5 GiB ✓ |
| HunyuanVideo | 23.9 GiB | 14.0 GiB | 6.0 GiB ✓ |
| Wan2.1-T2V-14B | 26.6 GiB | 10.6 GiB | 6.7 GiB ✓ |
| Wan2.2-T2V-A14B | 53.2 GiB (two experts) | 10.6 GiB | 13.3 GiB ✗ |
| LTX-2 | 35.2 GiB + 158 GiB root | 93.5 GiB | ✗ |

Because the text encoder can run as a separate stage, the binding constraint
is `max(transformer, encoder)` rather than the sum — so the top three all fit
and the decision falls back to complexity, which is where the plan already
put Qwen-Image first: image output (simplest artifact path) and
guidance-distilled (one forward pass, no CFG branch).

**The frozen ops surface had a real leak, now closed.** Qwen-Image's
transformer imports 13 names from `difflet.ops`, and **6 existed only in the
Trainium backend** (`SPMDRank`, `get_world_group`,
`scatter_to_process_group_spmd`, `get_cp_rank_spmd`,
`get_tensor_model_parallel_size`,
`gather_from_tensor_model_parallel_region_with_dim`). `from difflet.ops
import (...)` resolves every name eagerly, so a missing one breaks at import
time even on a code path that never runs. All six are now implemented on TPU,
along with the rest of the Trainium surface — **the two backends now export
the same 50 ops**, enforced by a test.

The `*_spmd` family collapses on TPU: NxD traces one graph for all ranks, so
the rank must be a runtime tensor; TPU exports a per-rank artifact, so the
rank is a Python constant. `scatter_to_process_group_spmd` therefore rejects
a traced tensor rank explicitly rather than producing a graph that is correct
for one rank out of four. MX ops raise instead of silently running
unquantized — MX is Trainium hardware with no TPU equivalent.

**Phase 4 de-risked on real hardware.** A genuine diffusers
`QwenImageTransformer2DModel` (small config), with 34 of its linears swapped
for `ColumnParallelLinear`, at tp=4 on the 2x2 v5e mesh: **matches the
unsharded oracle to 2.44e-05, and `torch.export` succeeds on the whole model
with all 34 opaque collective nodes preserved** (536 nodes). 16/16 checks
across 4 ranks.

**What the remaining Phase 4 work actually is.** Qwen-Image's difflet code
lives in `difflet/backends/trainium/qwen_image/transformer.py`, which fuses
modeling with Trainium plumbing — the "in between" category from the
portability audit, so the zero-change claim does not apply as written. But
the split is clean: of its 8 classes, only 3 touch Trainium core
(`QwenImageTransformerInferenceConfig`, `ModelWrapperQwenImageTransformer`,
`NeuronQwenImageTransformerApplication`). The 5 modeling classes
(`_StaticQwenImageRealRope`, `_ZeroQwenImageAttention`, `_ZeroLikeModule`,
`_QwenImageTrainiumAttnProcessor`, `_QwenImageTransformerTraceModule`) use
only torch, diffusers and `difflet.ops` — they are backend-neutral already,
just trapped in a file whose module-level imports pull NxD.

So the port is **code motion**: extract those 5 into a backend-neutral module
(where Wan and HunyuanVideo already keep theirs) and have both backends
import it. The DiT verification above is what makes that mechanical rather
than speculative.

**That refactor was blocked on not being able to verify the Trainium side —
it no longer is.** See the next entry.

Full unit suite: 1699 passed / 107 failed (the then-current off-Trainium
baseline; same three pre-existing non-dependency failures).

### 2026-08-16 — the Neuron toolchain DOES install off-host; the baseline is gone

Everything above was measured against a "107 failures is just how it is off
Trainium" baseline. That baseline was wrong. The toolchain installs on this
GCP TPU VM, and with it the suite goes from **1699 passed / 107 failed / 29
collection errors** to **2090 passed / 3 failed / 0 errors** — and the 3 are
the same pre-existing failures verified against clean HEAD.

Reproducible via `scripts/setup_neuron_venv_offhost.sh`. Four things block
the toolchain off-host, each with a supported escape:

1. `libtorchneuron.so` needs `libnrt.so.1`, which ships only in the
   `aws-neuronx-runtime-lib` **apt** package. Extracted from the `.deb` with
   `dpkg-deb -x` — no root, no system modification.
2. `torch_xla` shells out to `libneuronpjrt-path`, a console script inside
   the venv, so the venv's `bin` must be on `PATH`.
3. `libneuronxla` hard-codes `/opt/aws/neuron/lib/libnrt.so.1` and checks it
   exists (so `LD_LIBRARY_PATH` alone is not enough) —
   `NEURON_INTERNAL_SKIP_LIBNRT_CHECK=1` bypasses the check.
4. `torch_neuronx` reads `/sys/devices/virtual/dmi/id/product_name` and
   refuses on non-EC2 (`Unsupported Platform - Google Compute Engine`) —
   `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` names the target directly. This is
   the same variable `modeling_flux.py` already sets at import time.

Plus one pin: `transformers<5`. `neuronx_distributed` imports
`transformers.utils.fx`, which 5.x removed — that alone accounted for 106 of
the failures.

**What this does and does not buy.** It is import-level and CPU-level
coverage: there is no Neuron driver and no device, so anything that actually
executes on hardware still cannot run. A green run means "this change did not
break the Trainium code path", never "verified on Trainium". That is exactly
the coverage the Qwen-Image extraction needs, though — the risk there was
import-surface breakage, not device behaviour.

**It also invalidates a claim made earlier in this log.** The portability
audit said the codebase had "likely never been imported on a host without the
Neuron toolchain" and to "budget for import-time breakage". With the toolchain
installed that budget mostly disappears; the genuine off-toolchain import
breakage found earlier (`difflet/utils/tensor_replacement/registry.py` →
`torch_xla`, `difflet/layers/normalization.py`) is still real, but it is now
a *portability* question rather than a *can we even test* question.

One test of mine was wrong and this caught it: `test_import_does_not_require_torch_xla`
asserted `"torch_xla" not in sys.modules`, a whole-process property that is
legitimately false in a Neuron venv. It now blocks the import in a subprocess,
which is what it always meant to test.

### 2026-08-16 — Phase 4 milestone: difflet's real Qwen-Image DiT runs on TPU at tp=4

**The extraction is done.** `difflet/models/qwen_image/modeling_qwen_image.py`
now holds the backend-neutral modeling (494 lines, moved verbatim);
`backends/trainium/qwen_image/transformer.py` keeps only what is genuinely
Trainium (the `InferenceConfig` subclass, `ModelWrapper`,
`NeuronApplicationBase`) and re-exports the rest, so all four existing
importers — the application, the TeaCache probe, and two test modules — work
unchanged. One deliberate non-verbatim change: the trace module's
`config: QwenImageTransformerInferenceConfig` annotation was dropped, because
a backend-neutral module should not name a Trainium class. The config was
always duck-typed.

Checked first that `ltx_2/transformer.py` and `hunyuan_video/backbone15.py`
define their *own* copies of `_env_flag`, `_column_parallel_like` etc. rather
than importing Qwen's, so there is no hidden cross-dependency.

**Result, on the real 2x2 v5e mesh at tp=4 — 20/20 across all ranks:**

| Check | Result |
|---|---|
| Import the modeling on the TPU backend | PASS — `neuronx_distributed` never enters `sys.modules` |
| Build difflet's real `_QwenImageTransformerTraceModule` | PASS |
| Shard stock diffusers weights in via `_difflet_shard` | PASS — 77 tensors, 0 unmatched |
| **tp=4 output vs. the stock diffusers reference** | **cosine = 1.000000, maxdiff 2.60e-05** |
| `torch.export` of the real trace module | PASS — 17 opaque collectives, 619 nodes |

That is the same code Trainium runs, unmodified, against the diffusers
reference as DEVELOPER.md requires (never TPU-vs-Trainium, which would let two
independently-wrong implementations agree). It clears the 0.9995 bar with room
to spare, and it exercises the Phase 3 weight loader end to end.

**Trainium side verified intact:** 2092 passed / 3 failed in the Neuron venv,
the 3 being the known pre-existing failures. The move caught one real
regression on the way: `tests/unit/models/qwen_image/test_qwen_groups.py`
reads the transformer module's source to assert the CP collectives ride the cp
axis and not dp — after the move that source no longer contained them. The
test now checks both halves, with the dp prohibition applying to each, so
relocating code cannot become a way around the lint.

### 2026-08-16 — real checkpoints, real geometry, and a three-tier test ladder

**Real checkpoint loading (`backends/tpu/core/checkpoint.py`).** Reads
safetensors *lazily*: each rank slices only its own sub-range off disk. This
is not an optimization — four ranks each materializing Qwen-Image's 38 GiB
tensor set would need ~152 GiB of host RAM before any shard is taken.
Verified against the real files: 0 missing, 0 unexpected, cosine 1.000000.

**Real 20B geometry now fits, measured not estimated.** Building the actual
60-layer model requires `accelerate.init_empty_weights(include_buffers=False)`
— buffers must stay real because the trace module precomputes the static RoPE
in `__init__`, which needs `pos_freqs` data. With that:

| | |
|---|---|
| Build time (structure only) | 1.3 s |
| Params per rank at tp=4 | 5.11 B (of 20.4 B) |
| Parameters still on meta | 1933 / 1933 — zero allocation |
| Host RSS after build | 0.4 GiB |
| **bf16 footprint per rank** | **9.5 GiB** |

That confirms the 9.5 GiB/chip figure derived earlier from file sizes, and
leaves ~6.5 GiB per v5e chip for activations.

**A design bug this surfaced.** Shard metadata used to be an attribute on each
`nn.Parameter`. `accelerate.init_empty_weights` re-creates parameters and
forwards their attributes as constructor kwargs, so it died with
`Parameter.__new__() got an unexpected keyword argument '_difflet_shard'`.
The declaration now lives on the *module* (`{parameter name: split dim}`),
which is where it belonged anyway: sharding is a property of the layer's
design, and module-level metadata survives the parameter re-creation that
`to_empty()`, wrapper layers, and `init_empty_weights` all perform.

**Test ladder — three tiers, so the 54 GiB download is not the dev loop:**

| Tier | What | Size | Covers |
|---|---|---|---|
| `hf-internal-testing/tiny-qwenimage-pipe` | diffusers' own CI fixture, all 5 components | **35 MiB** | real checkpoint files, full pipeline path, tp=1 |
| synthetic small config | real `QwenImageTransformer2DModel` class, shrunk | 0 bytes | tp=4 sharding + collectives + export |
| `Qwen/Qwen-Image` | the real thing | 54 GiB | memory fit, and eventually a real image |

The tiny fixture has `num_attention_heads: 3`, which does not divide tp=4 —
hence the synthetic config for sharding work. Between them the download is
needed only for the memory check (done) and the final image.

### 2026-08-16 — Qwen-Image wired through difflet's own stack on TPU

The tiny fixture is a raw diffusers checkpoint, not a difflet-supported model
(its id does not even match the `is_qwen_image` detector — `qwenimage` has no
separator). It stays useful as a fast smoke checkpoint, but the model under
test is `Qwen/Qwen-Image`, which difflet already supports.

**Now wired end to end through difflet's own path:**

- `difflet/registry.py` — Qwen-Image's entry lists `backends=("trainium", "tpu")`,
  closing the item Phase 1 deliberately deferred until the model was chosen.
- `difflet/models/qwen_image/entry.py` — a `tpu` branch in the factory.
- `difflet/models/qwen_image/tpu_application.py` — `TpuQwenImageApplication`,
  mirroring `NeuronQwenImageApplication`'s outward contract
  (`compile`/`load`/`has_compiled_artifacts`/`dit_input_contract`/`__call__`)
  over the TPU component lifecycle. Transformer-only, per the plan's advice to
  prove the single-component path first.
- `difflet/backends/tpu/qwen_image/{config,transformer}.py` — geometry config
  recomputed without NxD, plus the `TpuApplicationBase` subclass.
- `difflet/models/qwen_image/contract.py` — third extraction of the same kind:
  `QwenImageDiTInputBundle` and `normalize_dtype` were trapped in
  `application.py`, which imports NxD at module level. Both are pure;
  `application.py` re-exports them.

**The real 20B model loads on TPU through `create_qwen_image_application`:**

| | |
|---|---|
| `dit_input_contract` at 1024x1024 | hidden `(1, 4096, 64)`, text `(1, 1024, 3584)` |
| Weights loaded from disk | 10.5 s |
| Params on this rank | 5.11 B, bf16 |
| Host RSS | 15.2 GiB (9.5 GiB of it parameters) |
| Non-finite parameters | 0 |
| Static RoPE buffers | finite, `(4096, 128)` |

**Three bugs found on the way, each of the silent kind:**

1. `Module.to_empty()` blanks **buffers** as well as parameters — and the
   static RoPE is a buffer computed in `__init__`, which is exactly what
   `init_empty_weights(include_buffers=False)` was protecting. Using it would
   have produced uninitialized rotary embeddings: correct shapes, no error,
   garbage images. Replaced with a materializer that touches only meta tensors.
2. Converting the *incoming* tensor to bf16 while the target parameter was
   fp32 just upcast it straight back — 20.4 GiB per rank instead of 9.5 GiB,
   which does not fit a 16 GiB v5e chip. The dtype has to be applied when the
   storage is allocated.
3. The loader treated non-persistent buffers as missing checkpoint entries.
   `persistent=False` means "computed, never stored", so their absence is
   correct by definition — now derived from `state_dict()` keys.

**And one portability leak in the newly-extracted modeling:**
`_safe_tensor_parallel_size()` caught only `AssertionError`, which is how
*NxD* reports an uninitialized TP group. The TPU mesh raises `RuntimeError`,
so the "fall back to tp=1" path never fired and construction died instead. A
module advertised as backend-neutral was quietly assuming NxD's exception
type. Now catches both.

Trainium side re-verified after all of this: 2111 passed / 3 failed (the
known pre-existing three). `test_qwen_entry.py` needed updating — it asserted
the rejection message said "trainium backend"; it now checks the supported
set is `trainium and tpu` and that the tpu branch really builds.

### 2026-08-16 — the real 20B runs on 4 v5e chips, and it is fast enough

Built through `create_qwen_image_application(..., backend="tpu")`, real
weights, four processes, tp=4. **4/4 ranks pass, identical numbers:**

| | |
|---|---|
| Weights loaded from disk | 16–18 s |
| **HBM after weights** | **9.52 / 15.7 GiB** |
| HBM peak during forward | 9.56 GiB — activations add 0.04 GiB |
| Output | `(1, 4096, 64)`, finite |
| Forward latency | 38.0 s, 36.9 s, then **0.85 s, 0.85 s, 0.85 s** |
| `CompileTime` counter | **2** |

The 9.5 GiB/chip figure predicted from file sizes, then confirmed on meta
tensors, now holds in actual HBM — with 6.1 GiB of headroom, and activations
at 1024x1024 costing almost nothing.

**Steady-state is 0.85 s per forward.** Sanity check: 20 B params × 2 FLOPs ×
4096 tokens ≈ 164 TFLOP against 4 × 197 TFLOPS bf16 = 788 TFLOPS, so 0.21 s at
100 % MFU — 0.85 s is ~25 % MFU. That is an unremarkable-but-real number for a
port with no performance work done at all, and it puts a 50-step denoise at
~42 s per image.

**It also puts a price on the Direction B failure.** The two compiles cost
~75 s, they happen on *every* process start, and torch_xla cannot persist
them (`UNIMPLEMENTED: Deserializing serialized executable not supported`).
For a long-lived server that is a one-time warm-up; for a one-shot CLI run it
is 75 s of pure overhead on top of a 42 s generation. This is now a measured
cost rather than an abstract limitation, and it is the strongest argument for
revisiting Direction C — JAX's persistent cache was measured working on this
same hardware.

Two compiles rather than one is the ordinary torch_xla warm-up pattern (the
first trace differs slightly from the steady-state one); it stabilizes after.


---

# Handoff: state, context, and what is left

Written 2026-08-17, after Phases 0-4 and most of 6. Everything below was
measured on the machine described in "Context", not estimated.

## Where this landed

`difflet serve` runs Qwen-Image on four v5e chips and returns real images over
the OpenAI Chat Completions API. Benchmarked through the repo's own harness:
**19.10 s warm end-to-end** at 1024x1024 / 20 steps / tp=4 — 3.3x faster than
trn2, level with an H100 PCIe, at about a sixth of the peak device memory.
Full numbers and reproduction in `benchmark/v5e/RESULTS.md`.

Trainium is unaffected: 2119 passed / 3 failed on the full unit suite, and the
three failures pre-date this work (verified against a clean HEAD worktree).

## Context a newcomer needs

**The dev box IS the TPU.** `machine-type n2d-192-112-v5lite-tpu`,
`accelerator-type v5litepod-4`: 4 x v5e, 2x2, 16 GB HBM per chip, us-west4-a,
chips at `/dev/vfio/{0..3}`. No provisioning step exists or is needed. Read it
from GCP metadata under `instance/attributes/tpu-env`.

**Two separate virtualenvs, neither of them the system Python** (which has no
torch at all):

| purpose | contents | notes |
|---|---|---|
| TPU work | torch 2.9.0+cpu, torch_xla 2.9.0, libtpu 0.0.21, diffusers 0.38.0, torchvision 0.24.0 | `python -m virtualenv` — `python3-venv` is not installed |
| Trainium tests | the Neuron toolchain | built by `scripts/setup_neuron_venv_offhost.sh` |

The Neuron venv matters more than it sounds: without it the unit suite reports
~107 failures and 29 collection errors purely from missing imports, which hides
real regressions. With it, 2119/3. It gives import-level and CPU-level coverage
only — no driver, no device — so a green run means "did not break the Trainium
code path", never "verified on Trainium".

**Weights live under `/mnt/models/`** (a 2 TB volume; the root filesystem has
~88 GB and a single checkpoint would fill it). Qwen-Image is at
`/mnt/models/hf/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be...` — the same
revision `models.py::MATRIX` pins.

**Chips are held until the holder dies.** A worker that outlives its parent
(a SIGKILLed server orphans its daemon replicas) keeps `/dev/vfio/*` open, and
the next run fails with "Device or resource busy". `tpu-info` lists the PIDs.

## Design decisions and why

| decision | why |
|---|---|
| Direction A (torch.export -> StableHLO) | The only direction that works. B cannot deserialize executables on torch_xla; C's eager interop crashes on torch 2.9. |
| Collectives wrapped as opaque custom ops | `torch.export` fails outright on a raw `xm.all_gather`, and the wrapper still lowers to a real HLO all-gather. |
| One process per chip | SPMD and AOT export are mutually exclusive on torch_xla 2.9 (`Unknown device SPMD:0`), and export is required by the compile contract. |
| TP implemented in Phase 2, not deferred to 5 | No candidate model fits a 16 GB chip at tp=1. |
| Serving runs eagerly | The compile path exists but is unproven at 20B scale, and an artifact would save tracing, not compilation. |

## What is left, roughly by value

**1. Numerical validation of the real model (Phase 5).** The 20B has only been
checked for "finite" and "the image looks right". The cosine 1.000000 result is
from a small config. `tests/numerical/` exists and DEVELOPER.md requires the
diffusers reference as the oracle — never TPU-vs-Trainium, since two
independently-wrong implementations can agree.

**2. Direction A on the real model.** `TpuApplicationBase` implements
compile/load and was verified on a toy module across a process boundary, but
the 60-layer 20B has never been exported. Collectives were exactly where export
broke before, so this is a real unknown, and the whole Phase 0 decision rests
on it. Serving does not depend on it today.

**3. Parallel modes beyond TP.** Context parallelism (ring / ulysses /
gather_kv), cfg-parallel and sequence parallelism all raise
`NotImplementedError`. Sequence parallelism is the interesting one: difflet's
ops surface already has `scatter_to_sequence_parallel_region` and friends, and
activations at block boundaries are currently replicated on every rank.

**4. Performance.** Two known items, both measured:
   - The text encoder runs on the host. After the thread fix it is ~1 s, so it
     is no longer the bottleneck, but it leaves the chips idle.
   - Attention has no fused kernel. The TPU Pallas flash-attention kernel needs
     jax 0.7.1, which requires Python >= 3.11 while this toolchain is on 3.10;
     a standalone 3.12 (e.g. `uv python install`) would unblock it without
     touching the system Python or the 3.10 Neuron venv.

**5. Serving completeness.**
   - `difflet generate --backend tpu` is not wired; only `serve` is.
   - There is no `/v1/images/generations` endpoint — image generation goes
     through `/v1/chat/completions`. Adding the OpenAI Images shape would
     mirror what the Videos API already does.
   - No web UI ships with difflet. A standalone one exists outside the repo
     (page + server-side proxy, because difflet installs no CORS middleware).

**6. More models.** Only Qwen-Image is ported. Wan, HunyuanVideo and LTX-2 have
backend-neutral backbones and should follow the same path; Flux stays
quarantined as the legacy NxDI fork. Each needs its `backends` tuple extended
and a TPU branch in its entry factory.

**7. Artifact layout.** Exported artifacts are per-rank because each rank's
weight shard differs. Separating graph from weights would let ranks share one
graph.

**8. Multi-host TPU pod.** Deliberately out of v1 scope; a materially different
runtime topology.

**9. The Trainium side of the Qwen-Image extraction** has only import-level and
CPU-level verification. It is pure code motion with re-export and the suite is
green, but it has not run on Trainium hardware.

## Mistakes worth not repeating

Recorded because each cost real time and none would be obvious from the code:

- **A Python scalar in the denoise loop** (`delta = float(...)`) is
  constant-folded into the graph, so every step becomes a different graph and
  XLA recompiles all of them. 2.45 s/step against 0.25 s of device work. Pass
  per-step scalars as device tensors.
- **PyTorch defaults to one thread per core per process.** Four replicas
  oversubscribed the host four-fold; a text encode measured 22 s in the worker
  against 0.43 s standalone.
- **A missing `torch.no_grad()`** kept every block's intermediates alive and
  overran HBM. The structural fix is `requires_grad_(False)` at load.
- **`Module.to_empty()` blanks buffers**, including the RoPE computed in
  `__init__` — correct shapes, no error, garbage images.
- **Converting the incoming tensor to bf16 while the parameter is fp32** just
  upcasts it back: 20.4 GiB per rank instead of 9.5 GiB.
- **XLA's default TPU matmul precision is not fp32** (~1e-2 error per matmul).
  Silent: every shape is right and nothing raises.
- **Forcing a device sync per step to measure it changes the workload**, not
  just the clock — 0.858 s/step measured, 0.276 s actual.
