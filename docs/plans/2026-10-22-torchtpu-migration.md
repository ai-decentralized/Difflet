# TorchTPU migration — replacing `torch_xla` on the TPU backend

Date: 2026-10-22 · Branch: `tpu-port-hunyuan` (rebased onto `main` @ `117ca45`)

## Goal

Move the TPU backend off `torch_xla` and onto **TorchTPU** (`torch_tpu`), Google's
native PyTorch TPU backend, without touching the Trainium backend, the model files,
or the `difflet.ops` contract.

Two things motivate this, in order of weight:

1. **The v5e backend's worst number is warm start, and TorchTPU fixes its root cause.**
2. `torch_xla` is deprecated and its repo is archived 2026-11-15.

## Why now

### The warm-start problem is structural, and it has a named cause

`benchmark/v5e/RESULTS.md`, same MATRIX rows, same protocol as `benchmark/trn2/`:

| model | trn2 warm | v5e warm | ratio |
|---|---:|---:|---:|
| ltx_2 | 58.5 s | **268.8 s** | 4.6× |
| flux_1_dev | 35.3 s | **211.7 s** | 6.0× |
| wan_2_1 | 56.4 s | **92.9 s** | 1.6× |
| wan_2_2 | 57.4 s | **93.8 s** | 1.6× |
| qwen_image | 62.7 s | **92.4 s** | 1.5× |

RESULTS.md names the cause: "the TPU pays XLA's compile again (44–170 s of DiT
compile, plus the VAE/loop graphs on the first request)". And
`difflet/backends/tpu/core/application_base.py:16` names why no workaround exists:

> Direction B, the only mechanism that would cache the executable, fails on
> torch_xla (`UNIMPLEMENTED: Deserializing serialized executable not supported`),
> so no path avoids this today.

TorchTPU ships a three-tier compilation cache whose tier-2 (machine-local shared
memory, cross-process) and tier-3 (remote filesystem, **cross host, program and
job restart**) hold *serialized compiled executables*. That is precisely the
mechanism torch_xla lacked, and it is the whole reason the backend went Direction A
(`torch.export` → StableHLO) in the first place.

If tier-3 delivers, three things collapse at once:

- the per-rank AOT export/load lifecycle (`core/application_base.py`, ~330 lines),
- the `torch.library.custom_op` + `register_fake` wrapper layer in
  `ops_impl/collectives.py`, which exists *only* because `xm.all_gather` makes
  `torch.export` fail,
- the `DIFFLET_TPU_COMPILE_SLOTS` flock gate that rations the ~43 GB/rank host
  memory peak of XLA compile (`application_base.py:warmup_eager`).

### The deprecation clock — and what it actually means

PyTorch/XLA's docs now read "This project is deprecated and replaced by TorchTPU",
and "The PyTorch/XLA repo will be archived on November 15, 2026."

**Nothing breaks on Nov 15.** The `torch_xla==2.9.0` wheel stays on PyPI and the
current v5e setup keeps running. What stops is fixes, and compatibility work
against future `libtpu` and torch releases. So this is a plan to execute
deliberately, not a fire drill.

## Scope — what is and is not being replaced

| area | torch_xla refs | files | in scope |
|---|---:|---:|---|
| `difflet/backends/tpu` | 35 | 6 | **yes** |
| `difflet/models/*/tpu_application.py` | 87 | 6 | **yes** (1763 lines) |
| `difflet/serving` | 23 | 3 | **yes** (partial) |
| `benchmark` | 103 | 13 | **yes** (adapters + runners) |
| `tests` | 87 | 19 | **yes** |
| `scripts` | 86 | 19 | mostly one-off probes — port on demand |
| `difflet/backends/trainium` | 22 | 4 | **no** |

The Trainium backend's `xm.collective_permute` / `xm.all_to_all` (ring CP, ulysses,
`backends/trainium/ops_impl/attention.py`) run on AWS Neuron's own `torch-xla`
build, shipped inside the Neuron toolchain venv. That is a separate artifact from
Google's upstream repo and a separate machine; Google's archival does not touch it.
**This migration must not edit anything under `difflet/backends/trainium/`.**

## API inventory and mapping

Counts are from `difflet/` only. The mapping column is assembled from TorchTPU's
`docs/public_apis.md` and the launch posts — **there is no official PyTorch/XLA →
TorchTPU mapping table**; every row marked "medium" or below must be verified in
Phase 0, not assumed.

| current | n | TorchTPU | confidence |
|---|---:|---|---|
| `torch_xla.device()` | 20 | `torch.device("tpu")` | high |
| `xr.global_ordinal()` / `xr.world_size()` | 9 | `dist.get_rank()` / `dist.get_world_size()` | high |
| `xm.wait_device_ops()` | 6 | `torch.accelerator.synchronize()` | high |
| `xm.mark_step()` | 23 | mostly deleted (DeferAndFuse picks materialization points); `synchronize()` where a hard boundary is required | medium |
| `xm.collective_broadcast(list, root_ordinal=0)` | 5 | `dist.broadcast(t, src=0)`, per tensor | high |
| `xm.all_gather(dim=, groups=)` | 4 | `torch.tpu.ops.all_gather_dim` (experimental) or `dist.all_gather_into_tensor` on a subgroup | medium |
| `xm.all_reduce(REDUCE_SUM, groups=)` | 4 | `dist.all_reduce` — `torch.tpu.ops` has **no** `all_reduce` | medium |
| `xm.reduce_scatter(groups=)` | 2 | `torch.tpu.ops.reduce_scatter_dim` (experimental) | medium |
| replica groups `[[0,1],[2,3]]` | all of `ops_impl/parallel_mesh.py` | `dist.new_group(ranks)` subgroups, or `torch.tpu.distributed.topology_aware_mesh` + DeviceMesh | medium |
| `backends.set_mat_mul_precision("highest")` | 1 | `torch.set_float32_matmul_precision("highest")`, plus scoped `torch.tpu.precision` | high |
| `custom_kernel.flash_attention` | 1 | `torch.tpu.ops.splash_sdpa` (experimental), else the existing SDPA fallback | **low** |
| `stablehlo.save_torch_model_as_stablehlo` / `StableHLOGraphModule.load` | lifecycle | no equivalent → `torch.compile(backend="tpu")` + compilation cache | — |
| `torch_xla.launch` | 1 | one process per chip + `dist.init_process_group("tpu_dist")`; `torch.tpu.distributed.set_tpu_launch_env` | medium |

Note on `torch.tpu.ops`: TorchTPU's public API doc states that **anything not on
that list is undefined behavior**, and every collective on it is marked
experimental. Prefer `torch.distributed` where both exist.

## Migration strategy: a runtime shim, not a forked backend

The naive options are both bad. Editing `backends/tpu/` in place destroys the
ability to A/B against the recorded v5e numbers. Forking a parallel
`backends/torchtpu/` means forking the 1763 lines of `models/*/tpu_application.py`
too, where 87 of the call sites live.

Instead: **introduce one thin runtime shim, port every TPU call site onto it with
no behavior change, then implement the shim a second time on `torch_tpu`.**

```
difflet/backends/tpu/ops_impl/device_runtime.py   # new, the only module that imports torch_xla or torch_tpu
    device()            -> torch.device
    rank() / world_size()
    sync()              # xm.mark_step + wait_device_ops  |  torch.accelerator.synchronize
    step()              # xm.mark_step                     |  no-op
    broadcast(tensors, src=0)
    set_matmul_precision(p)
    device_kind()
```

Selected by `DIFFLET_TPU_RUNTIME=torch_xla|torch_tpu` (default `torch_xla` until
Phase 5 flips it). The collectives already have a clean seam —
`ops_impl/collectives.py` is the single implementation of `difflet.ops`'
tensor-parallel surface — so its three `torch.ops.difflet_tpu.*` custom ops get a
second implementation behind the same public functions rather than a second file.

Benefit: the A/B is one env var, on one process, against the identical benchmark
harness. No forked model code, no divergent benchmark protocol.

## Phase overview

| Phase | Goal | Key output | Gate |
|---|---|---|---|
| 0 | De-risk; decide go/no-go | Spike memo in the status log | — |
| 1 | Runtime shim, still on torch_xla | `device_runtime.py`, all call sites ported, **benchmark unchanged** | Phase 0 go |
| 2 | TorchTPU behind the shim | `DIFFLET_TPU_RUNTIME=torch_tpu` runs one model at tp=4 | 1 |
| 3 | Drop the AOT lifecycle | `core/application_base.py` reduced to compile-cache management | 0d proven |
| 4 | Per-model bring-up | 5 models, parity ≥ 0.9995 each | 2, 3 |
| 5 | Benchmark + flip the default | `benchmark/v5e/RESULTS.md` rewritten, default is torch_tpu | 4 |
| 6 | Remove torch_xla from the TPU path | docs, pins, dead code deleted | 5 |

## Phase 0 — de-risk (no production code)

Prerequisite: **there is no TPU node right now** (the v5e was deleted 2026-09-12).
Recreate in `us-west1-c` (the zone that provisioned in 65 s when `us-west4-a`
queued for days), reattach the `tpu-models-2tb` restore at `/mnt/models`. Build a
**new venv** — do not touch `/mnt/models/tpuenv312`; it is the rollback.

Four spikes, throwaway scripts in the spirit of `scripts/mx_smoke.py`. Run 0a
first; it can kill the whole plan on its own.

### 0a — Can the dependency set even co-exist?

`torch_tpu==0.2.0` needs Python 3.11+ and some torch version the README does not
state (it only points at PyTorch's compatibility matrix). The v5e box today runs
Python 3.12.14, torch 2.9.0+cpu, diffusers 0.38.0.

**Measure:** which torch `torch_tpu==0.2.0` resolves to; whether diffusers 0.38 and
the repo's own pins survive that torch; whether `jax` is still needed at all (today
it is, for the Pallas flash kernel — `ops_impl/attention.py:_flash_kernel`).

**Exit:** a working venv that imports `torch`, `torch_tpu` and `diffusers`, and runs
a `torch.device("tpu")` matmul. **If this forces a torch bump that breaks diffusers,
stop and re-plan** — that is a much larger blast radius than this document covers.

### 0b — Collective semantics on a real 2×2 mesh

The current mesh (`ops_impl/parallel_mesh.py`) hands XLA explicit replica-group
lists and relies on XLA's rule that groups must partition *every* replica. TorchTPU
expresses the same thing as `torch.distributed` process groups.

**Measure:** 4 processes, `dist.init_process_group("tpu_dist")`. For
`all_gather_dim`, `all_reduce`, `reduce_scatter_dim`, `broadcast`: compare against
the current `xm.*` results bit-for-bit, at tp=4 and at the tp=2 × cp=2 shapes the
mesh can produce. Check both `torch.tpu.ops.*` and the `dist.*` equivalents.

**Exit:** a documented, bit-identical mapping for all four, with the chosen API per
collective written down. Any mismatch is a finding, not a rounding issue.

### 0c — Precision

`ops_impl/platform.py` sets `"highest"` globally because XLA's TPU default measured
~1e-2 error per matmul against ~2e-6 at highest, which made the ≥ 0.9995 cosine
target unreachable. TorchTPU defaults to `medium` (bf16 MXU, fp32 accumulate) and
documents `high` as ~3× and `highest` as ~6× MXU cost, with a warning on use.

**Measure:** one DiT block, cosine vs. the diffusers fp32 oracle and ms/step, at
`medium` / `high` / `highest`; then the same with `highest` scoped via
`torch.tpu.precision` to only the numerically sensitive ops.

**Exit:** the cheapest setting that holds cosine ≥ 0.9995, with its ms/step cost.
A scoped result that beats today's global `highest` is a *win*, not just parity.

### 0d — The go/no-go: does the compilation cache survive a fresh process?

This is the spike that justifies the whole migration.

**Measure:** run one real DiT at tp=4 with tier-3 enabled on a shared path. Kill
all four processes. Start fresh. Record time to first completed step, and confirm
from the profile that no XLA compile occurred. Repeat after a reboot, and from a
second host if one is available.

**Exit:** warm start of a fresh 4-process worker set is **under 40 s** for
Qwen-Image (today: 92.4 s; trn2: 62.7 s). If the cache does not survive a fresh
process, the main reason for this migration is gone and the plan becomes a
pure deprecation-driven port — re-decide priority at that point.

## Phase 1 — the shim, with torch_xla still underneath

Add `ops_impl/device_runtime.py` and route every call site through it:
`models/*/tpu_application.py` (87), `serving/` (23), `backends/tpu/` (35),
`benchmark/adapters/tpu*.py` and the five `benchmark/*_tpu_run.py` runners.

**No behavior change.** The gate is `benchmark.bench --backend tpu` reproducing the
recorded v5e numbers within noise on at least Qwen-Image and Wan 2.1, plus the unit
suite green. This phase is deliberately boring and is what makes every later phase
reversible by one env var.

## Phase 2 — TorchTPU behind the shim

Implement `device_runtime.py` and `collectives.py` on `torch_tpu` per the 0b
mapping. Launch becomes `dist.init_process_group("tpu_dist")`, one process per
chip, replacing the `torch_xla.launch`-shaped path that
`serving/engines/resident_worker.py:956` reimplements by hand.

Delete, once green: the `torch.library.custom_op` / `register_fake` wrappers, and
`SPMDRank` / `_global_rank` indirection that exists for NxD shape-compat.

**Gate:** Qwen-Image (the smallest application, 139 lines) generates a correct
1024² image at tp=4 under `DIFFLET_TPU_RUNTIME=torch_tpu`, cosine ≥ 0.9995.

## Phase 3 — drop the AOT lifecycle

Conditional on 0d. Replace export/load in `core/application_base.py` with
`torch.compile(backend="tpu")` plus cache configuration. Remove the per-rank
artifact directories, `tpu_manifest.json`, and the `DIFFLET_TPU_COMPILE_SLOTS`
flock gate — the last of which should become unnecessary, since compilation moves
into the cache-miss path instead of every process start.

`BackendCapabilities.requires_aot` for the TPU backend flips to `False`, and
`runtime.py:prepare_runtime` can finally be implemented rather than raising.

**Gate:** `difflet serve` reaches `/ready` and answers a request with no AOT
artifact on disk, twice in a row, with the second start hitting the cache.

## Phase 4 — per-model bring-up

Order by ascending risk, same ladder the original port used:
Qwen-Image → Wan 2.1 → Wan 2.2 → FLUX.1-dev → HunyuanVideo → LTX-2.

Each model: parity ≥ 0.9995 vs. the diffusers oracle, one served request, and the
probe-free TeaCache pair (`--teacache-cadence`, `--teacache-online-delta`) still
honored. LTX-2 last — it is the one with the on-chip VAE and the masked
cross-attention, and it was the hardest model in the original port.

Attention is the open risk here: if `splash_sdpa` does not match the current Pallas
wrapper's padded-and-masked semantics (`ops_impl/attention.py:_flash_attention`),
fall back to SDPA and record the regression rather than shipping a wrong mask.

## Phase 5 — benchmark and flip

Re-run the full `benchmark.bench --backend tpu` campaign on the trn2 protocol —
page cache dropped for cold, fresh workers for warm, n=3, decoded output checks —
and rewrite `benchmark/v5e/RESULTS.md` with a third column set. Keep the torch_xla
column; it is the before/after evidence.

Then flip `DIFFLET_TPU_RUNTIME`'s default to `torch_tpu`.

## Phase 6 — remove torch_xla from the TPU path

Delete the torch_xla branch of the shim, drop the dependency from the TPU install
docs, pin `torch_tpu`, and update README note 6, `DEVELOPER.md`, and the runtime
features table. `difflet/backends/trainium/` keeps its torch_xla imports untouched.

## Open decisions needing a call

1. **torch version.** If 0a shows `torch_tpu` requires a torch that diffusers 0.38
   or the repo's pins reject, this becomes a dependency-upgrade project first.
   Decide then whether to bump repo-wide or pin the TPU venv separately.
2. **Flash attention.** Accept an SDPA-speed regression in exchange for dropping
   the jax dependency, or invest in porting the wrapper onto `splash_sdpa` /
   `torch.tpu.pallas.jax_op`? Recommend: ship on SDPA, record the number, revisit.
3. **Precision policy.** Global `highest` (today) vs. scoped per 0c. Recommend
   whichever 0c measures as cheapest at cosine ≥ 0.9995.
4. **Keep an export path at all?** If 0d succeeds there is no reason to, but
   `BackendCapabilities.requires_aot=True` is currently load-bearing for how the
   CLI reasons about TPU. Confirm nothing else depends on it.
5. **Multi-host.** Out of scope, as in the original plan. Tier-3 cache across hosts
   is worth *measuring* in 0d, but multi-host serving is not in this plan.

## Risks

| risk | impact | mitigation |
|---|---|---|
| `torch_tpu` 0.2.0 is 2 days old; collectives all marked experimental | high | Phase 0b is bit-for-bit; shim keeps torch_xla one env var away through Phase 5 |
| Dependency conflict forces a torch bump | high | 0a runs first and can stop the plan |
| 0d fails — cache does not survive a fresh process | high | the migration loses its main payoff; re-prioritize rather than push through |
| `splash_sdpa` semantics differ from the Pallas wrapper | medium | existing SDPA fallback (`DIFFLET_TPU_FLASH=0`) is already the correctness path |
| No official migration guide exists | medium | the mapping table above is unverified by construction; 0b/0c verify it |
| v5e capacity | medium | us-west1-c provisioned in 65 s last time; probe with a synchronous queued-resource create |

## Recommended first step

Recreate the v5e node in `us-west1-c`, build a clean venv, and run **0a alone**.
It is an hour of work, it is the cheapest possible falsification of the whole plan,
and everything downstream is wasted if it fails.

## Status log

### 2026-10-22 — plan written

Desk research only: no TorchTPU code has been run, and no TPU node exists. The
API mapping table is assembled from TorchTPU's `docs/public_apis.md`, the Google
Developers and PyTorch launch posts, and the Cloud TPU docs — not from an official
PyTorch/XLA migration guide, which does not appear to exist. Treat every row below
"high" confidence as a hypothesis for Phase 0 to test.

### 2026-10-22 — v5e reacquired; **Phase 0a PASS**

Node: `v5e-probe-us-west1-c` (v5litepod-4, us-west1-c, yotta-dev-453606), queued
resource `v5e-probe-us-west1-c-qr`, **ACTIVE after 180 s** — in line with the 65 s
and 188 s this zone gave on 2026-09-11. External IP `8.231.52.76` (it changes on
every recreate). `tpu-models-2tb` reattached and mounted at `/mnt/models` by UUID
with `nofail`; 492 G used of 2 T, and `Difflet` (at `b27886d`), `tpuenv312`,
`tpuenv`, `uvpy`, `uvbin`, `hf` (403 G) all survived.

New venv `/mnt/models/tt-venv` (uv, CPython 3.12.14). `tpuenv312` untouched — it is
the rollback. Note `uv venv` needs `--seed`; without it the venv has no `pip` and
the official install line fails with `No module named pip` *while the shell still
exits 0*.

**The answer to 0a's core unknown: `torch_tpu==0.2.0` pulls `torch 2.14.1+cpu`.**
The box ran 2.9.0+cpu. It also pulls `libtpu 0.0.51` (against 0.0.20 today) and
**does not need jax at all** — today's Pallas flash kernel is the only reason jax
is installed, so dropping that kernel drops a dependency.

| | `tpuenv312` (torch_xla, works) | `tt-venv` (torch_tpu, new) |
|---|---|---|
| torch | 2.9.0+cpu | **2.14.1+cpu** |
| TPU runtime | torch_xla 2.9.0 | torch_tpu 0.2.0 |
| libtpu | 0.0.20 | 0.0.51 |
| diffusers | 0.38.0 | 0.38.0 |
| transformers | 5.15.1 | 5.20.2 |
| accelerate | 1.14.0 | 1.15.0 |
| numpy | 2.5.2 | 2.5.4 |
| tokenizers | 0.22.2 | 0.23.3 |
| safetensors | 0.8.0 | 0.9.0rc1 |
| jax | 0.7.1 | not installed |

**Exit criteria, all met.** `diffusers==0.38.0` — the hard pin — installs unchanged
on torch 2.14.1 and is not downgraded; `pip check` reports no broken requirements;
`import torch, torch_tpu, diffusers, transformers` and
`from diffusers import FluxTransformer2DModel, AutoencoderKL` all succeed; the
chips stay visible afterwards. The plan's "if 0a forces a torch bump that breaks
diffusers, stop and re-plan" branch did **not** fire.

**Device smoke.** `torch.accelerator.current_accelerator()` → `tpu`,
`device_count` → 4, a 512² fp32 matmul on `torch.device("tpu")` round-trips with
**cosine 0.99999863 vs CPU at the default `medium` precision**. That is already far
better than the ~1e-2 per-matmul error `ops_impl/platform.py` documents for
torch_xla's default, which is the first evidence that 0c may end cheaper than
today's global `"highest"`.

**Two findings for 0d, free from the smoke's logs.** The compilation cache
announces its own state — `Tier-3 compilation cache root directory: ` (empty, so
disabled) and `Tier-2 compilation cache is disabled for world size 1` — so 0d can
read cache tier and hit rate straight from the logs rather than inferring them from
wall clock. And a single process warns `Only using 1 device out of all the 4
addressable devices`: tp=4 needs four processes, as expected.

**Two things to carry into the lockfile work.**

- `safetensors 0.9.0rc1` — a prerelease was resolved. Pin to 0.8.x.
- `transformers 5.15.1 → 5.20.2` is a minor bump, **not** the 4.x → 5.x jump that
  `requirements-neuron.lock` (`transformers==4.57.6`) suggests at a glance. That
  lock is Trainium's, a different machine; the TPU box has been on 5.x all along.

**Per-device torch versions are already the architecture, not a new problem.**
`pyproject.toml` deliberately declares no torch (lines 20–30): difflet installs
*into* a venv that provides it. trn2 gets torch 2.9 + torch-neuronx from AWS's
Neuron venv, the TPU box gets its own, and a dev laptop gets a third. TorchTPU
needing 2.14 therefore cannot reach the Trainium backend. What is missing is
bookkeeping, and it should be added with this migration: a `requirements-tpu.lock`
and `requirements-torchtpu.lock` beside the Neuron one (the TPU side has never had
a lockfile — this recreate only went smoothly because the data disk outlived the
node), plus CI running the shared unit suite under both torch versions, since
`difflet/models/*/modeling_*.py` is shared and a torch-2.14-only API would pass on
TPU and fail on trn2.

### 2026-10-22 — **Phase 0b PASS**: every collective is bit-identical

Method: the same deterministic rank-distinct payload (`arange(32).reshape(4,8) +
rank*1000`, exactly representable in fp32) through both runtimes on the same
chips, each rank saving a `.pt`, then an exact `torch.equal` diff per rank. The
torch_xla side calls the same `xm.*` entry points with the same
`groups=`/`pin_layout=False` arguments that `ops_impl/collectives.py` uses in
production, so this compares against what ships, not against a toy.

Two meshes: tp=4 (`[[0,1,2,3]]`) and tp=2 × cp=2 (`[[0,1],[2,3]]`).

| collective | `torch.distributed` | `torch.tpu.ops` |
|---|---|---|
| all_gather dim 0, tp4 | bit-identical | bit-identical |
| all_gather dim 0, tp2 | bit-identical | bit-identical |
| all_reduce SUM, tp4 | bit-identical | n/a — no `all_reduce` in `torch.tpu.ops` |
| all_reduce SUM, tp2 | bit-identical | n/a |
| reduce_scatter SUM dim 0, tp4 | bit-identical | bit-identical |
| reduce_scatter SUM dim 0, tp2 | bit-identical | bit-identical |
| broadcast src=0 | bit-identical | n/a |

**Both spellings work and agree**, so the mapping table's "medium" confidence rows
for the four collectives are now confirmed. Recommendation: use
`torch.distributed` throughout (`all_gather_into_tensor`, `all_reduce`,
`reduce_scatter_tensor`, `broadcast`) — it covers all four, including the two
`torch.tpu.ops` has no entry for, and it is the stable surface.

**Replica groups map cleanly to process groups.** `parallel_mesh.axis_groups()`
already returns a full partition of the world, so each partition becomes a set of
`dist.new_group(ranks=...)` calls, created by every rank in the same order, with
each rank keeping the one it belongs to. XLA's "groups must cover every replica"
rule — the one `collectives.py::_groups` re-checks by hand because a violation
surfaces as a RET_CHECK deep inside XLA — has no analogue to worry about here:
`new_group` is explicit about membership. That hand-rolled validation can go.

**The launch finding, which cost the first attempt.** `torchrun` alone is not
enough. `init_process_group("tpu_dist")` fails with

    RuntimeError: missing required environment variables for distributed
    training: TORCH_TPU_SLICEBUILDER_ADDRESSES, TORCH_TPU_TOPOLOGY; please run
    the program via torchrun or similar tools so that the environment is set up
    properly

— and the advice in that message is misleading, because torchrun does not set
them. `torch.tpu.distributed.set_tpu_launch_env(nproc_per_node=N)` does, and its
docstring is explicit that it must run **in the parent, before any worker spawns**.
The working shape is a parent that calls it and then `os.execv`s
`torch.distributed.run`: exec hands the environment down and drops whatever the
parent had attached (`tt_launch.py` on the box). This matters for Phase 2 —
`serving/engines/resident_worker.py` hand-rolls its own per-chip spawn precisely
because `torch_xla.launch` owns the process lifecycle, and it will need the same
parent-side call before it forks its workers.

**`torch.tpu.ops` surface, for the record** (27 names): `align_jagged_to_multiple,
all_gather_dim, max_pool{1,2,3}d(_backward), optimization_barrier,
ragged_all_to_all, ragged_dot, ragged_dot_weight_grad, recv, reduce_scatter_dim,
send, sparse_dense_matmul*, sparse_gather*, sparse_iota, sparse_map_row,
splash_sdpa, strip_jagged_padding`. Note `splash_sdpa` is present — the Phase 4
attention risk has a candidate to test rather than only the SDPA fallback.

### 2026-10-22 — **Phase 0c**: `medium` holds the bar on a 4-block proxy at 2.42x the speed

Measured on FLUX's own block geometry (dim 3072, 24 heads, head_dim 128, 4096
image + 512 text tokens), four blocks deep, random weights at a fixed seed,
against a CPU fp32 oracle. Rotary embeddings omitted on purpose — they are
elementwise, so matmul precision cannot reach them.

| mode | 1−cos @4 blocks | max abs err | ms/block | vs `highest` |
|---|---:|---:|---:|---:|
| `medium` (TorchTPU default) | 1.61e-07 | 4.80e-03 | **21.0** | **2.42x faster** |
| `high` | 8.50e-12 | 3.19e-05 | 29.8 | 1.70x faster |
| `highest` (what the repo pins today) | 4.86e-13 | 1.91e-06 | 50.8 | — |
| `medium` + attention-only `HIGHEST` | 1.38e-07 | 4.75e-03 | 35.5 | 1.43x faster |

All four clear cos ≥ 0.9995 by orders of magnitude, so **the bar does not
discriminate at this depth** and the decision rests on the error columns.

**Selective scoping is not the win it looked like.** Raising precision inside
attention only costs 69% more than `medium` and leaves the error essentially
where `medium` had it (1.38e-07 against 1.61e-07). The error therefore lives in
the linear/MLP matmuls, not attention, and `torch.tpu.precision` as a targeted
escape hatch would have to wrap those instead — at which point it approaches the
global setting's cost. Drop the "scope highest to the sensitive layers" idea
unless a real model's parity run says otherwise.

**Error compounds with depth at `medium`, and is flat at `highest`.** Per-depth
1−cos runs 3.93e-08 → 7.98e-08 → 1.20e-07 → 1.61e-07 on `medium` (roughly linear
in depth) against 4.54e-13 → 4.86e-13 on `highest`. Linear extrapolation to
FLUX's 57 blocks lands near 2e-06, i.e. cos ≈ 0.999998 — still three orders
inside the bar, and HunyuanVideo's 60 blocks are no worse. **This is
extrapolation from a 4-block random-weight proxy, not proof**: real checkpoints
and a real residual stream may compound differently, and Phase 4's per-model
parity run against the diffusers oracle is what actually decides.

> **Superseded — see "Per-step ablation" below.** This spike ran FLUX blocks with fp32 random weights, and the precision setting governs **fp32** matmuls only (torch_xla's own docstring: "for 32bit inputs"). Production DiTs run in bf16, which it does not touch: on Qwen-Image under torch_tpu, `medium` against `highest` moved per-step by 1%. The 2.42x is real for fp32 and irrelevant to the models.

**Recommendation:** default the TorchTPU path to `medium`, carry `highest` as a
per-model override, and let Phase 4 move any model that misses parity. Dropping
`platform.py`'s global `"highest"` is worth 2.42x on every matmul, which is a
larger effect than most of the rest of this migration.

**Two methodology notes, both learned the hard way.**

- *Cosine has no resolution here.* The first run of this spike printed
  `1.000000` for all four modes and would have been recorded as "no difference".
  At a relative error of 1e-3 over 12M elements, cosine is 1 − 5e-7, which
  rounds to 1 at six decimals. Report `1−cos` in scientific notation, or an
  absolute error, whenever the quantity being compared is this close.
- *`max relative error` is not decision-grade as computed.* It reads 1.56e+03 on
  `medium`, which is an artifact of dividing by reference elements near zero
  (clamped at 1e-6), not a real 1000x error. Max absolute error against
  order-1 activations is the honest column.
- *A context manager around the whole forward is the global setting by another
  name.* The first attempt at the scoped row wrapped `forward_capture` entirely
  and reproduced global `highest` to three significant figures. Scoping has to
  be selective — here, per-submodule — to measure anything.

**Not measured:** the same proxy under torch_xla at its default and at
`"highest"`. The decision this spike had to make is TorchTPU-internal
(`medium` vs `highest` on the same runtime), so the cross-runtime comparison was
not on the critical path, and `platform.py`'s "~1e-2 per matmul" is a
per-matmul figure that is not directly comparable to a through-4-blocks one.

### 2026-10-22 — **Phase 0d PASS**: the executable survives a fresh process, and only one rank compiles

The spike the migration rests on. Measured on the same FLUX 4-block stack,
timing `t_first_forward` — weights already on the chip, inputs ready, to
synchronized completion — so XLA compile is isolated from the host-side build.
The knobs are `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` and
`TORCH_TPU_TIER2_COMPILATION_CACHE`, **found by running `strings` over the
shipped binary**: they appear in neither the docs nor `public_apis.md`.

**Single process, four fresh starts:**

| condition | t_first_forward | num_compilations |
|---|---:|---:|
| tier-3 disabled (today's situation) | 3.47 s | 37 |
| tier-3 enabled, cache empty (cold) | 3.47 s | 37 |
| tier-3 enabled, cache warm, **fresh process** | **0.20 s** | **0** |
| again | **0.20 s** | **0** |

`num_compilations` goes 37 → 0 across a process boundary: the executables are
deserialized from disk, which is exactly what torch_xla answers
`UNIMPLEMENTED: Deserializing serialized executable not supported` to. 0.20 s
against a pure-execute 0.084 s means the remainder is cache read and dispatch.

**tp=4, four concurrent workers — and the better finding:**

| phase | rank 0 | rank 1 | rank 2 | rank 3 |
|---|---:|---:|---:|---:|
| cold, t_first | 3.51 s | 3.05 s | 2.14 s | 3.55 s |
| cold, num_compilations | **37** | **0** | **0** | **0** |
| warm, t_first | 0.20 s | 0.21 s | 0.20 s | 0.20 s |
| warm, num_compilations | 0 | 0 | 0 | 0 |

**On a cold cache only one rank compiles; the other three take the result from
the tier-2 machine-local cache.** Setting the tier-3 root switches tier-2 on
(it is off for a single-device program), and the on-disk cache is 6.0 MB in 37
files — the same as the single-process run, so ranks do not each write a copy.

That removes the reason `DIFFLET_TPU_COMPILE_SLOTS` exists. Today
`application_base.py::warmup_eager` rations compile slots with an flock because
XLA's compile of a 60-block DiT peaked at **~43 GB per rank**, and four ranks
compiling at once plus the host text encoder does not fit a 188 GB v5e host.
With tier-2 dedup there is one compiler, not four, so the gate has nothing left
to prevent. Phase 3 should delete it rather than port it.

**Caveat, stated plainly.** This is a 4-block random-weight stack, not a real
DiT: it compiles in 3.5 s where a real one takes 44–170 s, and the mechanism
scaling to that size is assumed, not shown. The plan's exit criterion — a fresh
four-worker Qwen-Image set warm-starting under 40 s against today's 92.4 s — is
Phase 4/5's to prove on the real model. What is settled here is the binary
question, and it is settled: **a compiled executable survives a fresh process,
and survives four of them.**

**Three operational findings, each of which cost a run.**

1. *Do not call `torch.accelerator.set_device_index(local_rank)` under this
   launcher.* Each worker already owns one chip, so it sees exactly one
   addressable device and any index but 0 raises `Device index N is out of
   bounds of the number of addressable devices 1`. Ranks 1–3 died on it while
   rank 0 sat in `do_poll` waiting for peers that were already gone — a 13
   minute hang whose only visible symptom was host memory stuck at 4 GB of 188.
   The 0b spike made the same call inside a `try/except`, so it had been failing
   there too, unnoticed, which is itself the proof that the launcher assigns
   chips on its own.
2. *TPU chips are exclusive, and an orphaned worker keeps holding one.*
   `timeout` kills torchrun but not its children, and the survivors keep
   `/dev/vfio/N` open, so the next run dies with `Failed to acquire a TPU device
   node '/dev/vfio/1' because it is already opened by another process on this
   host` — an error whose text points nowhere near its cause. Free the chips by
   their actual holders (`sudo fuser /dev/vfio/*`) before and after each run.
   **This matters for Phase 2:** `serving/engines/resident_worker.py` owns its
   per-chip worker lifecycle, and every crash and timeout path must release the
   chip or the next `difflet serve` will not start.
3. *Never hang a long-running job directly on `gcloud compute tpus tpu-vm ssh`.*
   On a connection wobble gcloud silently reconnects and **replays the whole
   command**, which for a job that spawns processes means another full set of
   them. One deadlocked run left four separate generations of chip-holding
   workers this way. Start the job detached on the box and let ssh poll a log.

---

## Phase 0 verdict: **GO**

| spike | question | result |
|---|---|---|
| 0a | can the dependency set coexist? | **PASS** — torch 2.14.1, diffusers' 0.38.0 pin intact, `pip check` clean |
| 0b | do the collectives match? | **PASS** — all four bit-identical on tp=4 and tp=2×cp=2 |
| 0c | what does precision cost? | **PASS** — `medium` holds the bar at 2.42x the speed of today's `highest` |
| 0d | does the cache survive a fresh process? | **PASS** — 37 compilations → 0, 3.47 s → 0.20 s, and one compiler instead of four |

Nothing found in Phase 0 argues against the migration, and 0c and 0d each found
more than the plan asked for: a 2.42x matmul win that is independent of the
cache, and a compile-dedup that deletes a whole subsystem. Proceed to Phase 1.

### 2026-10-22 — **Phase 1**: the shim is in, and it costs nothing

`difflet/backends/tpu/ops_impl/device_runtime.py` is now the only module on the
TPU path that imports a TPU runtime library, selected by `DIFFLET_TPU_RUNTIME`
(default `torch_xla`). Only the torch_xla implementations exist; the torch_tpu
ones raise and name Phase 2. Ported: the four `ops_impl` modules, the component
lifecycle, four `models/*/tpu_application.py`, the Qwen serving orchestrator,
the resident worker's replica init, and `benchmark/adapters/tpu.py`.

**Shared files were handled under a standing rule — leave the other devices
alone — and each decision is evidence, not judgement.**

| shared file | done | why |
|---|---|---|
| `backends/registry.py` | inline `find_spec`, **no new import** | it is on every backend's path; "is this a TPU host" is a different question from "which runtime do TPU ops call" |
| `serving/engines/resident_worker.py` | ported | unreachable on Trainium *structurally*: the caller guards `replica_count > 1` and `_replica_count()` returns 1 whenever `supports_torchrun_mpmd` is False |
| `serving/orchestrators/qwen_image.py` | ported | all 14 changed lines are inside `_load_*_tpu` / `_encode_prompt_tpu` / `_denoise_tpu` / `_decode_tpu` |
| `benchmark/bench.py`, `harness.py` | **untouched** | an AST pass found **zero** live references; every mention is in a docstring, and the sync is injected as a callable |
| `models/flux/pipeline.py` | **untouched, deliberately** | it defines `NeuronFluxPipeline`, and `requirements-neuron.lock` pins `torch-xla==2.9.0`, so its `XLA_AVAILABLE` / `xm.mark_step()` pair is **Trainium's**, not the TPU backend's |

That last row is the trap: those two call sites look exactly like the ones the
refactor missed. They are noted in `device_runtime.py`'s docstring so a future
tidy-up does not "finish the job" and drag the production backend through a TPU
module.

The invariant that keeps `resident_worker` off the Trainium path had no test.
It does now (`test_a_backend_without_torchrun_mpmd_gets_exactly_one_replica`).

**Gate: Qwen-Image, and the baseline had to be rebuilt first.**

The phase's exit criterion was written as "reproduces the recorded numbers",
and that was the wrong criterion: `benchmark/v5e/RESULTS.md` was measured on
2026-09-12 on a VM that has since been deleted. Against it, the refactor looked
1.9% *faster* — which a shim that only adds a Python call cannot cause.

Three runs on identical code gave a run-to-run spread of 0.9%, so 1.9% was
**outside** it: systematic, not noise. The experiment that settles it is old
code against new code on **one machine, one protocol**:

| metric | new code (this box, n=3) | old code (this box, n=2) | **new vs old** | old vs Sept |
|---|---|---|---:|---:|
| per-step synced | 0.273 / 0.270 / 0.272 | 0.273 / 0.270 | **−0.0%** | −1.8% |
| per-step p90 | 0.274 / 0.271 / 0.273 | 0.275 / 0.272 | **−0.2%** | −1.9% |
| denoise | 5.710 / 5.662 / 5.695 | 5.751 / 5.662 | **−0.3%** | −1.2% |
| HBM / chip | 11.069 ×3 | 11.069 ×2 | **±0.0%** | ±0.0% |

**The shim costs nothing** — new against old is inside the 0.9% spread on every
row and bit-identical on HBM. The whole −1.9% is this VM instance against the
September one, and the old code shows the same offset.

**Gate: Wan 2.1**, which adds what Qwen-Image does not exercise — `reduce_scatter`,
the video VAE decode (20 s, the largest single piece of the request) and a video
latent's broadcast. Two runs each way, same machine, same protocol, all four
`status=ok`:

| metric | new code (n=2) | old code (n=2) | **new vs old** | Sept |
|---|---|---|---:|---:|
| per-step synced | 0.610 / 0.611 | 0.608 / 0.608 | **+0.3%** | 0.613 |
| per-step p90 | 0.613 / 0.613 | 0.611 / 0.612 | **+0.3%** | 0.617 |
| denoise | 12.208 / 12.231 | 12.203 / 12.164 | **+0.3%** | 12.245 |
| vae_decode | 20.425 / 19.998 | 20.084 / 20.570 | **−0.6%** | 20.040 |
| HBM / chip | 7.513 ×2 | 7.513 ×2 | **±0.0%** | 7.513 |

Across both models the shim moves nothing: ±0.3%, sign random, HBM bit-identical.

Note that Wan does **not** show Qwen-Image's 1.9% offset against September —
0.610 against 0.613 is 0.5%. So the machine-to-machine difference is not a flat
whole-box effect; it varies by model, which makes a recorded number an even
worse "before" than a single figure would suggest.

**Carry into Phase 5: never use a recorded number as the "before".** The
torch_xla → torch_tpu comparison must measure both sides on one machine in one
session. A ~2% machine-to-machine offset exists on per-step, which is small
against what TorchTPU should deliver but large enough to corrupt a per-step
conclusion.

**A cross-device hazard found and fixed on the way.** `--backend` and the output
folder are independent: the folder comes from `DIFFLET_BENCH_DEVICE`
(`benchmark/models.py`), which defaults to `trn2`. A failed TPU run therefore
overwrote `benchmark/trn2/qwen_image.{json,md}` — Trainium's recorded data,
restored from git. `benchmark/bench.py` now refuses to write one backend's
results into a folder whose existing results name another backend, reading the
expected backend out of the folder's own JSONs so the rule cannot drift and a
new device folder needs no change.

**Deliberately not done** (per the owner's call to focus on product code):
`tests/` (19 files) and the superseded `benchmark/*_tpu_run.py` runners and
one-off probes still import torch_xla directly. Consequence to remember: when
Phase 2 flips `DIFFLET_TPU_RUNTIME=torch_tpu`, those still exercise torch_xla.

### 2026-10-22 — **Phase 2**: torch_tpu runs a real model, and two defaults had to change

`DIFFLET_TPU_RUNTIME=torch_tpu` now runs Qwen-Image end to end at tp=4 —
`status=ok`, no tracebacks, four workers, 20 steps, VAE decode, PNG out.

**Shim verification first.** The same 0b payload, but driven through
`device_runtime` rather than the runtime APIs, so what is tested is the
translation layer: all four collectives bit-identical to torch_xla on tp=4 and
tp=2×cp=2. The pieces with real logic in them are the group-list →
`ProcessGroup` cache (`dist.new_group` is collective, so every rank must create
every group in the same order), the dim handling in all_gather /
reduce_scatter (`dist` only ever works along dim 0), broadcast over a list, and
`all_reduce` needing a clone because `dist`'s is in place while `xm`'s returns
a new tensor.

**Launch is not what torchrun does.** difflet spawns workers with `mp.Process`,
so nothing sets the `env://` rendezvous variables. `prepare_launch()` now picks
a free port and sets `MASTER_ADDR`/`MASTER_PORT`/`WORLD_SIZE` in the parent
before spawn; `initialize_replica()` sets `RANK`/`LOCAL_RANK` from its
arguments, which is what `pjrt.initialize_multiprocess` took directly. All
`setdefault`, so a genuine torchrun launch keeps its own values.

**Two capabilities are off by default and both had to be turned on in the
shim.** Leaving either to an environment variable the operator must know is the
same as not having it.

| | default | cost of leaving it | documented? |
|---|---|---|---|
| `TORCH_TPU_DEFER_AND_FUSE` | off (Strict eager) | per-step **2.9x** | mentioned in the launch posts |
| `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` | off | every process recompiles | **nowhere** — found with `strings` |

**Qwen-Image, 1024², tp=4, against torch_xla on the same box:**

| metric | torch_xla | tt Strict | tt +fuse | **tt +fuse +cache (warm)** |
|---|---:|---:|---:|---:|
| e2e cold | 92.24 s | 69.17 s | 64.23 s | **41.49 s** (−55%) |
| HBM / chip | 11.07 GB | 9.52 GB | 9.61 GB | **9.61 GB** (−13%) |
| per-step | 0.270 s | 1.004 s | 0.342 s | 0.355 s (+31%) |
| load | 7.83 s | 26.71 s | 20.78 s | 21.02 s (+168%) |

`num_compilations` across a fresh four-worker set: `8, 12, 21, 15, 0, 0, 2, 0`
cold, then **all zeros** warm, with 101 MB in 58 files on disk. Phase 0d
measured this on a 4-block proxy and the plan recorded that scaling to a real
DiT was assumed; it is now measured.

**per-step's remaining +31% is the attention kernel, and the arithmetic says so.**

> **Superseded — refuted by Wan 2.1, see "Corrections after a second model" below.** Wan's attention share is higher and its per-step came out at parity (−1.9%), so attention cannot be the cause of Qwen-Image's +31%. The 87 ms ≈ 80 ms agreement was a coincidence, and the 80 ms rested on a 60-block count that was never checked.
The gap against Strict eager was 734 ms/step. SDPA costs 1.33 ms per call at
4096 tokens, so across a 60-block DiT attention can account for ~80 ms — about
10%. Enabling DeferAndFuse removed 647 ms, leaving 87 ms. The two numbers meet.

**Attention is not a drop-in swap, and the kernel is not obviously the answer.**

- `torch.tpu.ops.splash_sdpa` defaults to **`is_causal=True`** while its own
  docstring calls it a "Replacement for F.scaled_dot_product_attention", whose
  default is False. It does not error — it silently returns cosine 0.40. A
  diffusion DiT is bidirectional, so this would be wrong everywhere.
  With `is_causal=False` it is correct (cosine 0.99998).
- It is also **slower than SDPA** at the shapes that matter: 0.68x at 4096
  tokens / 24 heads with the best working block configuration, against 2.7x
  *faster* at 1024 tokens. Sweeping `block_q`/`block_kv` did not help — the
  512/512 default was the best that ran, and `block_kv >= 1024` raised
  ValueError and poisoned the device state for every later config in the
  process.
- It has no `q_segment_ids`/`kv_segment_ids`, which the current wrapper uses
  for two things: masking the block padding of unaligned lengths (without it,
  Wan's 4680 silently returns 5.4e-2 relative error), and HunyuanVideo's
  per-row key window. Neither has a splash equivalent.
- Installing plain `jax` does **not** restore the current kernel: that wrapper
  imports `torch_xla.experimental.custom_kernel`, and this venv has no
  torch_xla. It does make `splash_sdpa` importable, which otherwise fails on
  `import jax` — and plain `jax` 0.12 left libtpu 0.0.51 and torch 2.14.1
  alone, so the note in `attention.py` about `jax[tpu]` downgrading libtpu does
  not apply here.

**load's +168% is host-to-device bandwidth, and TorchTPU's is 20-27x slower.**

> **Superseded in part — see "Corrections after a second model" below.** The h2d measurement itself stands (forced on both runtimes). The inference that it *explains load* does not: `load_seconds` is not comparable across the two runtimes, because torch_xla's load timer stops before its lazy transfers complete.

Ruled out first: compilation (tier-3 took `num_compilations` to zero and load
did not move — 21.04 cold against 21.02 warm), page cache, and DeferAndFuse.
Timing the three load stages showed the transformer stage taking 20.7-31.9 s
per rank while the VAE took 0.085 s, which points at moving weights, not at
building anything.

Measured single-process on an empty chip, same code both runtimes, **every copy
forced to materialise with a reduction read back to the host**:

| size | torch_tpu | torch_xla |
|---|---:|---:|
| 0.25 GB | 0.20 GB/s | 5.41 GB/s |
| 1 GB | 0.30 GB/s | 5.60 GB/s |
| 2 GB | 0.22 GB/s | 5.67 GB/s |

That forcing matters: without it torch_xla reported 20-46 GB/s, because `.to()`
on a lazy runtime records the copy and `wait_device_ops()` need not materialise
a tensor nothing consumes. The repo already knows this shape of error —
`benchmark/v5e/RESULTS.md` notes that an unsynced delta is the enqueue rate,
not device time. The 5.6 GB/s figure is flat across three sizes and the
checksums agree, so it is the real one.

Arithmetic: Qwen-Image at tp=4 puts ~9.6 GB of weights on each rank. At
5.6 GB/s that is ~1.7 s; at 0.25 GB/s it is ~38 s. Measured stage: 20-32 s,
with the weights loaded shard by shard rather than in one block.

This is a defect, not a setting. DeferAndFuse and tier-3 were both on, the chip
was empty, the process was alone. 0.25 GB/s is far below what the link should
do, and it is worth reporting upstream.

Consequences, which differ sharply by workload:

- **Resident serving** pays it once at startup. 21 s against 7.8 s is absorbed,
  and e2e cold is still 55% better because of the compilation cache.
- **Cold-start-heavy use** — switching models, restarting workers — pays 13 s
  every time.
- **Bigger models scale it linearly.** HunyuanVideo and LTX-2 carry more
  weight per rank, so the gap grows with them. Phase 4 must measure this per
  model rather than assume Qwen-Image's 13 s generalises.

**Dependencies.** `pip install -e .` plus torchvision plus jax into the
torch_tpu venv left torch at 2.14.1+cpu throughout, `pip check` clean.


### 2026-10-22 — Corrections after a second model

Two Phase 2 attributions above were each built on one model and a story that fit
it. Running Wan 2.1 through both runtimes on the same box, same protocol, same
session (launch scripts diffed line by line; they differ only in venv, runtime
and output folder) refuted both.

| metric | torch_xla | torch_tpu | tt vs xla |
|---|---|---|---:|
| per-step | 0.61 / 0.61 | 0.60 / 0.60 | −1.9% |
| denoise | 12.24 / 12.17 | 12.08 / 11.90 | −2.2% |
| HBM / chip | 7.51 / 7.51 | 7.00 / 7.00 | −6.8% |
| vae_decode | 19.93 / 20.08 | 21.12 / 21.22 | +5.7% |
| e2e cold | 99.79 / 92.78 | 103.72 / 79.97 | −13.8% |
| load | 6.34 / 6.78 | 46.22 / 25.81 | +280.7% |

> **Superseded — refuted by direct ablation, see "Per-step ablation" below.** Removing the Pallas kernel from torch_xla alone costs Qwen-Image +117%. The inference from Wan assumed torch_tpu's SDPA trails torch_xla's flash kernel by a fixed ratio; it does not — torch_tpu's SDPA is 44% faster than torch_xla's, and the gap to the fused kernel varies with shape.

**Attention is not why Qwen-Image is +31% per step.** Wan carries a longer
sequence and a larger attention share, so if SDPA-versus-Pallas were the cause
Wan would be worse. It is at parity. Qwen-Image's +31% is open, and specific to
it.

**Host-to-device bandwidth does not explain load across models.** The model
predicted Wan's load gap would be *smaller* than Qwen-Image's, since Wan holds
less weight per rank (7.0 GB against 9.6 GB). It is larger: +19.3 s against
+13.2 s.

**The reason: `load_seconds` measures different things on the two runtimes.**
`_load_denoiser_stage_tpu` ends with `step()` and no `wait()`; torch_xla's
`mark_step()` dispatches without waiting, and the h2d probe already showed that
even `wait_device_ops()` does not materialise a tensor nothing consumes. So
torch_xla's load timer stops while weights are still in flight, and the
transfer lands in the first request. torch_tpu's `.to()` completes before
returning. Subtracting load from e2e cold makes the shift visible:

| e2e cold − load | torch_xla | torch_tpu |
|---|---:|---:|
| Qwen-Image | 84.4 s | 20.5 s |
| Wan 2.1 | 86.0 s | 54.2 s |

torch_xla's post-load phase is far longer, carrying both the deferred transfer
and the recompile it pays on every process. That stretch cannot yet be split
between the two, which is why `benchmark/adapters/tpu.py` now records
`weights_load_materialized_s` alongside the unchanged `load_seconds`: the same
load followed by a read that forces every device tensor to have arrived.

**What survives two models:** cross-process compilation cache (`num_compilations`
to zero on both), lower HBM on both, better e2e cold on both. **Everything else is
model-dependent with no cause yet identified** — per-step, vae_decode, and how
much of load is real.

**Method notes, for the next phase:**

- Any stage-boundary timing is suspect across a lazy and an eager runtime.
  Only end-to-end wall clock and explicitly materialised measurements compare.
- One model plus a plausible story was enough, three times today, to write a
  conclusion here. A second model refuted it each time. Phase 4 must not record
  a cross-runtime cause without at least two models agreeing.
- Constants in an argument get checked before the argument is believed. The
  "60 blocks" behind the attention estimate never was.

### 2026-10-22 — FLUX.1-dev: a third model, a deadlock, an op gap, and a walk-back

**A deadlock Phase 2 introduced.** FLUX hung for 17 minutes with no output.
`/proc/locks` showed two ranks holding the two `compile_slot` flock files and
the other two in `nanosleep` polling for one. `compile_slot` wraps
`warmup_eager`'s forward, and its own docstring states the condition it relies
on: the slot is released after `mark_step()` dispatches and *before* waiting,
so a rank stuck in its first collective does not hold a slot. That holds only
on a lazy runtime. Under torch_tpu the tp collectives in the forward block on
the host inside the slot, and the two holders wait forever for the two
waiters. Qwen-Image and Wan never hit it because they do not call
`warmup_eager`; FLUX, HunyuanVideo and LTX-2 do. `compile_slot` is now a no-op
under torch_tpu — and not only to avoid the hang: tier-2 already gives one rank
the compile and the rest the result (Phase 0d: 37 / 0 / 0 / 0), so the 43 GB/rank
peak it guards against does not stack.

The general lesson for the rest of this migration: **any code whose safety
argument mentions `mark_step` assumes lazy execution and must be re-read.**

**An op gap in TorchTPU.** With the deadlock gone, FLUX failed in the VAE
decode — after all 28 denoise steps had succeeded:

    RuntimeError: view(): cannot create a view of shape [1, 32, 262144] from the
    input tensor of shape [1, 512, 128, 128] and strides [512, 1, 65536, 512]

A minimal repro (`scripts/torchtpu_groupnorm_repro.py`) separated it:

| question | answer |
|---|---|
| does conv2d on tpu return channels-last? | **no** — output contiguous |
| does conv → group_norm fail? | no |
| does group_norm on an explicitly channels-last tensor fail? | **yes**, same error |
| does `.contiguous()` first fix it? | yes, 1−cos 2.6e-6 vs CPU |

So TorchTPU's `group_norm` rejects non-contiguous input where CPU and CUDA
PyTorch accept any strides. The first hypothesis — that conv produced the
channels-last tensor — was wrong; the repro is why it did not get recorded as
the cause. What in `AutoencoderKL`'s decoder produces it was not traced, and the
fix does not depend on it. `device_runtime._install_compat()` wraps
`F.group_norm` to make tpu-resident non-contiguous input contiguous, only under
torch_tpu; host tensors are untouched. Qwen-Image and Wan were never affected:
their VAEs normalise with RMS-style layers. Worth an upstream report.

**Results, same box, same protocol, n=2 each:**

| metric | torch_xla | torch_tpu | |
|---|---:|---:|---:|
| e2e cold | 214.5 s | 60.2 s | −72% |
| vae_decode | 0.312 s | 0.136 s | −57% |
| HBM / chip | 10.455 GB | 10.403 GB | −0.5% |
| per-step | 0.188 s | 0.217 s | +16% |
| load, as reported | 52.3 s | 40.5 s | −23% |
| load, materialised | 55.6 s | 40.6 s | −27% |
| — of which the materialise step | 3.3 s | 0.09 s | |

**Walk-back of the previous entry.** "Corrections after a second model" argued
that load's cross-runtime gap was mostly torch_xla's timer stopping before its
lazy transfers finished. The materialised metric this entry introduced puts
that deferral at **3.3 s** on FLUX. That is real, and it is why `load_seconds`
alone cannot compare the runtimes — but it is far too small to explain Wan's
+19 s. The earlier correction over-corrected: Qwen-Image's and Wan's load gaps
are probably mostly genuine, consistent with the 20-27x slower host-to-device
copy that was measured with forced materialisation. Confirming that needs
Qwen-Image and Wan re-run with the materialised metric; until then it stays a
probability, not a finding.

**And load is not one quantity across models either.** FLUX's TPU application
calls `warmup_eager` inside load, so its torch_xla load of ~52 s contains the
DiT compile (~44 s); torch_tpu takes that from the cache. Qwen-Image and Wan do
not warm up during load. Comparing load across models compares different work.

**Three models together:**

| | Qwen-Image | Wan 2.1 | FLUX.1-dev | agree? |
|---|---:|---:|---:|---|
| e2e cold | −55% | −14% | −72% | **yes, all better** |
| HBM / chip | −13% | −7% | −0.5% | **yes, all ≤** |
| compile cache, fresh process | 0 compiles | 0 | 0 | **yes** |
| per-step | +31% | −2% | +16% | no |
| vae_decode | +49% | +6% | −57% | no |

Per-step has a *possible* shape — both image models at 1024² are slower, the
video model is at parity — but three points in two classes is a hypothesis.
Under the rule written into the previous entry, it is not recorded as a cause.

**TorchTPU's sharp edges, collected.** Two days after release, every model has
found one:

| issue | default | cost |
|---|---|---|
| DeferAndFuse | off | 2.9x per step |
| tier-3 compilation cache | off, undocumented | every process recompiles |
| `splash_sdpa` `is_causal` | True | silently wrong (cosine 0.40) |
| `group_norm` | rejects non-contiguous | unmodified diffusers VAEs crash |
| host-to-device copy | — | 20-27x slower than torch_xla |

"Change two lines to migrate" does not hold for diffusion models on 0.2.0.
Each further model should be expected to turn up its own.

### 2026-10-22 — Why torch_tpu loaded weights slowly, and the fix

The load regression has one cause, found by testing hypotheses as a matrix
rather than by reasoning (`scripts/torchtpu_h2d_why.py`): every 1 GiB copy,
forced to arrive, timed for wall and process CPU.

| default config | GB/s | cpu/wall |
|---|---:|---:|
| fp32 pageable | 0.21 | 1.07 |
| fp32 pinned | 0.28 | 1.09 |
| `non_blocking=True` | 0.21 | 1.08 |
| 3-D weight-shaped / bf16 / 2-D | 0.28 | 1.08-1.21 |
| **`TPU_PREMAPPED_BUFFER_SIZE=4GiB`** | **5.5-13.0** | 3.0-6.7 |

Pinning, async, dtype and shape change nothing; cpu/wall ≈ 1.1 means one host
core doing the copy. The premapped buffer changes everything.

**The rule** (`scripts/torchtpu_h2d_buffer_fit.py`, 256 MiB buffer, pinned): a
transfer goes at 10.9-12.7 GB/s if the whole tensor fits, and at 0.20 GB/s if
it does not — at 256 MiB exactly as well as at 512. It does **not** chunk
through the buffer. A 60x cliff, gated on the largest single tensor, not on
total weight.

**Which is why Wan was worst.** From the safetensors headers
(`scripts/torchtpu_max_tensor.py`), the largest tensor each model puts on the
chip: Qwen-Image 108 MiB, HunyuanVideo 108, FLUX.1-dev 126, LTX-2 216 (its VAE
runs on chip), **Wan 2.1 600 — its checkpoint is F32**. Larger tensors, more of
them over the cliff. The "host-to-device × weight volume" model in the earlier
entries used the wrong variable.

**Fix:** `device_runtime._size_premapped_buffer()` sets the variable to 2 GiB
by default, in `prepare_launch()` (parent, before spawn, so workers inherit it
before the runtime allocates the buffer). 2 GiB clears every supported model and
LTX-2's 1.35 GiB text connector. Cost: 2 GiB of premapped host memory per
worker. `setdefault`, so it can be traded down.

**Verified on Wan 2.1, launch script not setting the variable:**

| run 2 (warm) | torch_xla | tt before | tt + buffer |
|---|---:|---:|---:|
| load | 6.78 s | 25.81 s | **7.08 s** |
| e2e cold | 92.78 s | 79.97 s | **57.26 s** (−38% vs xla) |
| per-step | 0.61 s | 0.60 s | 0.59 s |
| HBM | 7.51 GB | 7.00 GB | 7.03 GB |

Load is now at parity with torch_xla; per-step and HBM did not move, so the fix
touched only what it should. With the buffer large enough the fast path is
*faster* than torch_xla's 5.6 GB/s.

**Not settled:**

- Run 1 is slower than run 2 both before and after the fix (46.2 / 25.8 s,
  then 28.1 / 7.1 s), so it is not this change. The materialise step took 21 s
  on run 1 and 0.2 s on run 2 — that is the new reduction graph compiling
  once, the instrument's own cost. Why run 1's *load* is also 21 s slower is
  unexplained; an on-chip F32→bf16 conversion compiling per shape would fit,
  and was not checked.
- The fallback call in `initialize_replica()` was not exercised on its own. If
  the runtime has already initialised by then, it may not take effect.
- That the slow path pins and maps pages on the fly is inferred from its
  single-threaded, low-bandwidth signature. TorchTPU's source was not read.

Upstream, the right fix is chunked staging through the buffer, so any size
reaches full bandwidth. Worth reporting with the `group_norm` gap.


### 2026-10-22 — Per-step ablation: it is the attention kernel, and it is not precision

Qwen-Image's per-step gap had two refuted explanations and no measured one.
This flips one variable at a time through the same harness and the same
full-step metric, rather than comparing two runtimes' profilers (which would
be the cross-runtime-semantics mistake again).

| | runtime | attention | matmul | per-step | vs A |
|---|---|---|---|---:|---:|
| A | torch_xla | Pallas flash | highest | 0.276 s | — |
| B | torch_xla | **SDPA** | highest | **0.598 s** | **+117%** |
| C | torch_tpu | SDPA | highest | 0.333 s | +21% |
| D | torch_tpu | SDPA | **medium** | 0.330 s | +20% |
| E | torch_xla | Pallas flash | medium | failed | — |

D's override was confirmed to apply: C logs 8 TorchTPU HIGHEST-precision
warnings, D logs none. E failed because torch_xla does not accept the name
`medium`; see below.

**Precision is ruled out.** D against C is 1%. The setting governs fp32
matmuls and the DiT runs in bf16; what is left is a few fp32 ops such as the
timestep embedding. That also retires Phase 0c's recommendation, which was
measured on fp32 random weights.

**The attention kernel is the gap.** B against A: without its fused kernel,
torch_xla is 2.17x slower. torch_tpu has no fused kernel in use and runs SDPA,
yet C lands only 21% behind A — because torch_tpu's SDPA is itself **44%
faster than torch_xla's** (C against B). So torch_tpu's non-attention path is
at least as good, and what is missing is a fused attention kernel as fast as
torch_xla's Pallas one. The earlier refutation reasoned from Wan's parity on
the assumption of a fixed ratio between torch_tpu's SDPA and torch_xla's flash
kernel; B and C show the ratio moves, and with shape it can net to zero.

This is strongly indicated, not closed. Closing it takes the fourth corner —
torch_tpu with a fast fused kernel — and `splash_sdpa` measured slower than
SDPA in isolation. That needs re-measuring inside the model.

**A shim bug this surfaced.** The two runtimes name the same three settings
differently: torch_xla takes `default`/`high`/`highest`, torch and TorchTPU
`medium`/`high`/`highest`, where `default` and `medium` are one bf16 pass.
`set_matmul_precision` now translates per runtime and rejects unknown names.
Separately, `platform.configure_matmul_precision()` passed its own default
explicitly, which made `DIFFLET_TPU_MATMUL_PRECISION` unreachable from every
caller; it now defers.

**Third revision of this question today.** The attention attribution was made
from arithmetic, withdrawn on a second model, and restored by ablation. The
first two were each one inference from one or two data points. Only the
ablation, which changes one variable and holds the rest, could separate the
kernel from the runtime.

### 2026-10-22 — splash_sdpa in the model: 70% slower than SDPA

The ablation left one corner open: torch_tpu with a fused kernel. TorchTPU's
own fused kernel is `torch.tpu.ops.splash_sdpa`, so `attention.py` gained an
opt-in path (`DIFFLET_TPU_SPLASH=1`) and Qwen-Image ran through it.

It applied to every joint-attention call — logged
`splash USED q=(6, 5120, 128) k=(6, 5120, 128)` — and 5120 being a block
multiple meant no key padding, so no mask was needed and none was missing.

| | per-step |
|---|---:|
| A torch_xla, Pallas flash | 0.276 s |
| C torch_tpu, SDPA | 0.333 s |
| **F torch_tpu, splash** | **0.567 s** |

Splash is 70% slower than torch_tpu's SDPA in the model, as it was in
isolation. **torch_tpu's SDPA is the fastest attention it has, and it trails
torch_xla's Pallas kernel by 21%.** Nothing TorchTPU 0.2.0 ships closes that.

The path stays, off by default, for two reasons: the next TorchTPU release can
be re-measured with one variable instead of new code, and it encodes the two
correctness constraints that make splash usable at all — `is_causal=False`
passed explicitly (its default is causal), and refusal of any key length that
would need padding (it has no segment ids to mask it).

**The way to parity is torch_xla's own kernel, not TorchTPU's.** The Pallas
flash kernel torch_xla calls is a JAX kernel, and TorchTPU's public API has
`torch.tpu.pallas.jax_op` for running JAX/Pallas kernels on torch tensors. That
is the next experiment: wrap the same kernel, keep the existing
segment-id padding mask, and measure whether torch_tpu then matches or beats
0.276 s. Its non-attention path already does as well as torch_xla's, so the
whole gap should be in reach.
