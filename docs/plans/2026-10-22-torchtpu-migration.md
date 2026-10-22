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
