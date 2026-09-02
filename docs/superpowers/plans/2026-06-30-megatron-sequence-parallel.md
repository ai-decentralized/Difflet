# Megatron-style Sequence Parallelism (SP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`)
> syntax for tracking.

**Goal:** Add **Megatron-style sequence parallelism** as an opt-in, TP-coupled mode
(`sp_enabled`) for the four CP-capable DiT backbones (Wan, Flux, HunyuanVideo, Qwen-Image),
selectable from the CLI (`--sp`). SP shards the otherwise-replicated norm / modulation /
residual regions along the sequence axis across the existing tensor-parallel group, replacing
the row-parallel all-reduce with reduce-scatter and adding an all-gather before each
column-parallel projection. No new world-size axis is introduced.

## 1. Definition — what "Megatron-style sequence parallelism" is

Source: Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*
(2022) — the sequence-parallel extension of Megatron-LM tensor parallelism.

A transformer layer has two kinds of region:

| Region | Activation layout | Compute |
|---|---|---|
| **Tensor-parallel (TP)** — attention core, MLP | full sequence, **hidden/head sharded** `[B, S, H/tp]` | column-parallel in, row-parallel out |
| **Sequence-parallel (SP)** — LayerNorm/RMSNorm, AdaLN modulation, residual adds *between* TP regions | **sequence sharded**, full hidden `[B, S/tp, H]` | per-token / elementwise |

In vanilla Megatron-TP the norm/residual regions are **replicated**: every TP rank redundantly
recomputes them on the full `[B, S, H]`, wasting activation memory by a factor of `tp`.
Megatron-SP shards those regions along the **sequence** axis across the *same* TP ranks, so the
region boundaries change their collective:

- **`g` (SP→TP, entering attention/MLP):** `all-gather` the seq-sharded input along sequence →
  full `[B, S, H]` before the column-parallel projection.
- **`ḡ` (TP→SP, leaving attention/MLP):** replace the row-parallel `all-reduce` with
  `reduce-scatter` along sequence → `[B, S/tp, H]`.

**Key invariant:** `all-gather + reduce-scatter` moves exactly the same bytes as one
`all-reduce`. SP therefore adds **zero communication** over plain TP while cutting
norm/residual/modulation activation memory by `tp×`. It is **not** a new world-size axis — it
reuses the TP group; `world_size` is unchanged.

**Distinct from the existing CP** (`cp_degree` / `cp_mode`): CP shards the sequence across the
*data-parallel* group and rings/gathers it *through attention* (to scale attention compute);
SP shards the sequence across the *TP* group only in the *norm/MLP boundary* regions and
**gathers the full sequence for attention** (to cut activation memory at no comm cost). The two
are orthogonal in principle; this increment keeps them mutually exclusive
(`sp_enabled` requires `cp_degree == 1`).

## 2. Per-model feasibility analysis

All five backbones normalize over the **hidden** axis and apply modulation/residuals
**per-token**, so sharding the sequence axis in the norm/residual regions is numerically sound
everywhere. The sequence axis is `dim=1` (`[B, S, H]`) in every model. Feasibility differences
are about stream structure and which forward code is CPU-reachable (coverage scope):

| Model | Streams | TP boundary (`ḡ` = all-reduce today) | Forward in coverage scope? | Verdict |
|---|---|---|---|---|
| **Wan** | single (self-attn + text cross-attn + FFN) | row-parallel `to_out` / `net_out` | `difflet/models/wan/modeling_wan.py` ✅ | **In scope** |
| **Flux** | MMDiT double + single stream | row-parallel out-projs | `difflet/models/flux/modeling_flux.py` ✅ | **In scope** |
| **HunyuanVideo** | dual-stream (latent‖text) + single | row-parallel `to_out`, `proj_out` | `difflet/models/hunyuan_video/modeling_hunyuan_video.py` ✅ | **In scope** |
| **Qwen-Image** | single-stream concat (text‖image) | row-parallel `to_out` / `to_add_out` | forward under `difflet/backends/trainium/qwen_image/` (omitted) | **In scope (logic), low coverage weight** |
| **LTX-2** | tri-stream (video/audio/text) + cross-modality; CFG-parallel only, no CP | row-parallel per stream | under `backends/trainium/` (omitted) | **Deferred** (tri-stream + no CP foundation) |

**Scope (this increment):** Wan, Flux, HunyuanVideo — **device-verified** (dense-vs-SP
cosine ≥ 0.999 at tp=4 on trn2). Qwen-Image landed 2026-09-02 via the `modeling_qwen`
fork (dual-stream pure SP) with its own device parity gate; LTX-2 **deferred** (see below).

### On-device verification (trn2.3xlarge, 4 NeuronCores, tp=4)

`scripts/{wan,flux,hunyuan,qwen}_sp_parity_smoke.sh` compile a dense backbone and an SP
backbone, run a fixed-seed forward on each, and compare:

| Model | dense-vs-SP cosine | verdict |
|---|---|---|
| Wan | 0.99993 | ✅ |
| Flux | 0.99997 | ✅ |
| HunyuanVideo | 0.999997 | ✅ |
| Qwen-Image | 0.999991 (tp=4, 2026-09-02) | ✅ |

All three also generate end-to-end via `difflet generate --sp --tp-degree 4`.

Two device-only bugs were found and fixed during bring-up (both invisible to CPU
tests, which use identity collectives — see [[difflet-sp-cpu-identity-testing]]):
1. **Sequence scatter under SPMD tracing** — nxd's `scatter_to_sequence_parallel_region`
   resolves `group.rank()` to a compile-time constant, so every rank kept the same
   chunk. Fixed by scattering with the materialized SPMD rank buffer via
   `scatter_to_process_group_spmd` (see [[difflet-spmd-rank-scatter]]).
2. **Bias double-count** — `RowParallelLinear(reduce_output=False)` adds the full
   bias to each rank's partial, which the ḡ reduce-scatter then sums `tp×`. Fixed by
   subtracting `(tp-1)·bias` after the reduce-scatter.

Both proved that **device parity, not CPU equivalence, is the real correctness gate.**

### Resolved: Qwen-Image (was Deferred)

Qwen's runtime originally **monkey-patched the upstream diffusers
`QwenImageTransformer2DModel`** rather than reimplementing the forward (as
Wan/Flux/HunyuanVideo do). In that wrapped trace, the `SPMDRank` per-rank id used by
the entry sequence scatter was not a live/loaded graph input — it constant-folded to
its `zeros(1)` init, so every rank read rank 0 and kept chunk 0 (device parity stuck
at ~0.30, bit-stable).

**Resolution (commit 9a00746 + 2026-09-02 completion):** `difflet/models/qwen_image/
modeling_qwen.py` now subclasses the diffusers model with SP-aware blocks — per-block
`g`/`ḡ` around the joint attention and each stream's MLP, norm/modulation/residual on
sequence shards, `_sp_unbias` on the four row-parallel outputs, and the entry scatter
through the **materialized SPMD rank buffer** (`scatter_to_process_group_spmd`), the
same primitive wan's validated SP path uses. `--sp` is accepted for Qwen; the device
parity gate is `scripts/qwen_sp_parity_smoke.sh`. LTX-2 (tri-stream, no CP foundation)
and HunyuanVideo-1.5 (segmented runtime) remain out of scope.

## 3. Reality checks that shape the plan

1. **CPU backend = identity collectives (`world_size == 1`).** SP is a numeric **no-op** on CPU
   (`S/tp` with `tp == 1` is the whole sequence; all-gather / reduce-scatter are identity).
   Host verification therefore asserts: (a) **SP path == non-SP path** bit-for-bit on CPU,
   (b) collective **placement / shape bookkeeping** is correct, (c) config + CLI **threading**.
   On-device numeric parity needs a Trainium run (env-gated NEFF / parity smoke scripts).
2. **Coverage omits `difflet/backends/trainium/**`.** SP logic that must count toward the >90%
   target lives in `pipeline/parallel_config.py`, `ops/collectives.py` (+ CPU impl),
   `cli/`, and the in-scope model forwards. This is why SP is wired **explicitly at the
   modeling level** (visible, testable) rather than hidden inside the nxd linears.

## 4. Architecture

- **Config:** new `sp_enabled: bool = False` on `DiffletParallelConfig`. `world_size` unchanged.
  Validation: `sp_enabled` requires `tp_degree > 1` is **not** required for host tests but is the
  intended deployment; `sp_enabled` and `cp_degree > 1` are mutually exclusive. Cache key is
  **additive-only** — `sp_enabled` omitted from `to_cache_dict()` when `False`, so legacy keys
  stay byte-identical.
- **Ops:** two new backend-dispatched collectives in `difflet.ops`:
  - `gather_from_sequence_parallel_region(tensor, *, dim)` — the `g` op.
  - `reduce_scatter_to_sequence_parallel_region(tensor, *, dim)` — the `ḡ` op.
  - `scatter_to_sequence_parallel_region(tensor, *, dim)` — forward-entry seq scatter across TP.
  CPU impls are **identity** (world_size 1). Trainium impls wrap the nxd
  `gather_from_sequence_parallel_region` / `reduce_scatter_to_tensor_model_parallel_region_with_dim`
  / `scatter_to_process_group_spmd` over the **tensor-model-parallel** group, dim-aware.
- **Row-parallel linears under SP** are constructed with `reduce_output=False` so the modeling
  code performs the `ḡ` reduce-scatter explicitly (CPU swallows the kwarg → plain `nn.Linear`,
  identity-correct; Trainium returns the un-reduced partial).
- **Model forward (per block, when `sp_enabled`):** keep the inter-region tensor seq-sharded
  `[B, S/tp, H]`; norm/modulation/residual run on the shard; `gather_from_sequence_parallel_region`
  before each column-parallel projection; `reduce_scatter_to_sequence_parallel_region` after each
  row-parallel projection. Forward entry scatters along seq across TP; forward exit gathers.

**Tech stack:** Python, PyTorch, `neuronx_distributed` (parallel state / collectives), pytest.

## 5. Global constraints

- **Backend-neutral modeling:** `modeling_*.py` import only `torch` / `diffusers` / `difflet.ops`.
  SP collectives are reached via `difflet.ops`, never imported from `neuronx_distributed` directly.
- **Default off:** `sp_enabled` defaults to `False`; a default config leaves the compile-cache key
  byte-identical to legacy (additive-only, mirroring `cp_mode` / `CandidateConfig`).
- **SP activates only when** `sp_enabled` is `True`. Mutually exclusive with `cp_degree > 1`.
- **Divisibility:** per-stream sequence length must be divisible by `tp_degree` on device (pad
  otherwise). Enforced/​documented at the model boundary; irrelevant on CPU (`tp == 1`).
- TDD: host-runnable tests run in normal `pytest`; device parity tests follow the env-var-gated
  NEFF pattern. Commit + push after each phase; squash at the end.

## 6. Phases (commit + push after each)

- **Phase A — config + ops foundation:** `sp_enabled` field + validation + additive cache key;
  new seq collectives in `difflet.ops` + CPU identity impls + Trainium impls; unit tests.
- **Phase B — CLI threading:** `--sp` flag in `cli/main.py`; validation guards (sp ⊥ cp);
  thread through the four orchestrators; unit tests.
- **Phase C — model wiring:** explicit SP region collectives in Wan / Flux / HunyuanVideo /
  Qwen-Image forwards; CPU equivalence + threading tests; Trainium parity smoke scripts.
- **Phase D — coverage:** add tests to keep CPU-testable coverage > 90%.
- **Phase E — squash** all phase commits into one; push.

## 7. Verification

- Host: `pytest tests/unit` green; SP-path == non-SP-path equivalence tests pass; total coverage
  (`--cov=difflet`, CPU scope per `pyproject.toml` omit list) > 90%.
- Device (operator-run, not in this host's reach): `scripts/*_sp_parity_smoke.py` assert
  SP vs non-SP trajectory cosine ≥ 0.999 on Trainium.
