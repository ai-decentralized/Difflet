# Parallel Mesh Refactor — Final Report (2026-07-05)

**Branch:** `worktree-restruct-group` (13 commits on top of baseline `93fac4c`)
**Goal:** make DP, CFG, CP, TP four orthogonal named axes with independent process
subgroups, and de-parasitize CFG/CP off NxD's `dp_group` — groups only, no DP
feature yet.

## What changed

### Rank layout and mesh (new)

Rank layout, tp innermost / dp outermost, sizes `(D, G, C, T)`:

```
rank(dp, cfg, cp, tp) = tp + T*(cp + C*(cfg + G*dp))

tp  = rank % T                    subgroup stride 1
cp  = (rank // T) % C             subgroup stride T
cfg = (rank // (T*C)) % G         subgroup stride T*C
dp  =  rank // (T*C*G)            subgroup stride T*C*G
```

An axis's subgroup is the set of ranks differing only in that coordinate. For
every pre-refactor configuration (exactly one of cfg/cp non-trivial, dp=1) the
non-trivial axis's mesh coincides with NxD's legacy dp-group column mesh
`[[j, j+T, …] for j in range(T)]` — which is why the migration is bit-exact.

- `difflet/pipeline/parallel_mesh.py` — `MeshSpec(dp, cfg, cp, tp)`: pure math,
  `coords_of` / `rank_of` / `axis_groups` / `axis_rank`, validated axes.
- `DiffletParallelConfig` — new `dp_degree` field (default 1, validated) and a
  `mesh_spec` property; `world_size = dp·cfg·cp·tp`. Compile-cache keys stay
  byte-identical for default configs (`dp_degree` omitted at 1).
- `difflet/backends/trainium/core/parallel_mesh.py` — ProcessGroupManager
  singleton: `init_parallel_mesh(config)` asserts `product == world_size`, then
  builds ONE `torch.distributed` group per **non-trivial** axis with the full
  axis mesh in `xla_pg_options` (replica groups under SPMD tracing). Distilled
  models (cfg=1) never construct a cfg group. Idempotent; conflicting re-init
  raises. TP delegates to NxD's tensor-parallel group (coincides with the tp
  axis since tp is innermost). `get_dp_group()` exists but has **zero**
  consumers in this increment.
- `difflet.ops` surface — added `init_parallel_mesh`, `get_cfg_group`,
  `get_cp_group`, `get_cfg_rank_spmd`, `get_cp_rank_spmd` (Trainium + CPU
  backends); **removed** `get_data_parallel_group` and `get_dp_rank_spmd`, so
  any parasitic regression is an ImportError.

### Call sites migrated (every collective now fires in its own axis subgroup)

| Model / file | CFG (cfg_group, size 2) | CP (cp_group) |
|---|---|---|
| Wan `models/wan/modeling_wan.py` | batch scatter + exit merge | seq/RoPE scatter, per-block gather-KV, exit gather |
| Flux `models/flux/modeling_flux.py` | dormant path (CLI-blocked) scatter + merge | RoPE/seq split, double+single-stream gather-KV, ring, exit gather |
| HunyuanVideo `models/hunyuan_video/modeling_hunyuan_video.py` | n/a (distilled) | seq/RoPE scatter, per-block gather-KV, exit gather |
| Qwen-Image `backends/trainium/qwen_image/transformer.py` | n/a (distilled) | RoPE scatter, seq scatter, KV gather, exit gather |
| LTX-2 `backends/trainium/ltx_2/transformer.py` | 10-input scatter + video/audio merge | n/a |
| ring collectives `backends/trainium/ops_impl/attention.py` | — | `ring_attention` / `joint_ring_attention` replica groups + permute pairs from `get_cp_mesh()` |

Per-branch/shard indices moved from the merged `rank // tp` to per-axis
`get_cfg_rank_spmd` (`(rank // (T·C)) % G`) and `get_cp_rank_spmd`
(`(rank // T) % C`) — integer-exact, reduce to the legacy value in every
pre-refactor config.

### Policy (per user decisions)

- CFG×CP mutual exclusion **kept** at the pipeline level; the mesh can express
  combined specs (unit-tested for `(1,2,2,·)`, `(1,1,4,·)`, `(4,1,1,·)`,
  `(2,2,1,·)`) but pipelines still reject the combo.
- Distilled models forced to cfg=1, unchanged guards: FLUX.1-dev, HunyuanVideo,
  HunyuanVideo-1.5, Qwen-Image (CLI `_DISTILLED_MODELS` + entry
  `NotImplementedError`s). Flux's Python-API CFG path left in place (dormant),
  migrated to cfg_group.
- LTX-2 cfg axis is ONLY cond/uncond (=2); STG (`perturbed_attn`) remains
  rejected under CFG-parallel, so no third guidance branch can reach the axis.
- NxDI verbatim-fork files (`attention_process_groups.py`, `attention_base.py`,
  `utils/distributed.py`) untouched — their intra-TP "dp/cp" machinery is
  unrelated to the pipeline axes.

## Verification

### Unit suite (CPU)

```
PYTHONPATH=. pytest tests/unit -q
1323 passed, 27 skipped
```

New suites: `tests/unit/pipeline/test_parallel_mesh.py` (18 tests: rank
round-trip for all four required combos, partition/vary-only-that-axis
properties, legacy-mesh equivalence), `tests/unit/backends/
test_trainium_parallel_mesh.py` (11 tests: spec derivation, product==world
assert, only-non-trivial-axis group construction, idempotency, SPMD rank
equivalence), per-model wiring tests, and `tests/unit/test_no_dp_parasites.py`.

### dp carries nothing (assertion tests)

`tests/unit/test_no_dp_parasites.py` enforces, repo-wide:
1. `difflet.ops` exports none of the dp helpers and all five axis ops;
2. no source file outside the NxDI-fork `utils/distributed.py` references
   `get_data_parallel_group` / `get_dp_rank_spmd`;
3. nothing outside the manager calls `get_dp_group` — the dp axis group (built
   only when dp>1) has no per-layer/per-step consumer.

### On-device bit-identity regression + smoke (trn2.3xlarge, 4 cores) — ALL models

`scripts/mesh_regression_smoke.sh`: per-model tiny seeded diffusers
checkpoints (identical weights for all runs), each DiT backbone compiled +
loaded + run end-to-end (doubling as the smoke test) under both revisions —
baseline `93fac4c` via a detached git worktree on `PYTHONPATH`, refactor from
the branch — with fixed-seed inputs. Coverage = every (model, parallel-axis)
pair the refactor changed on device, at `tp=2`, `world=4`:

```
[mesh-regression] wan:cfg            shape=(2, 16, 1, 16, 16) bit_identical=True PSNR=inf
[mesh-regression] wan:cp             shape=(1, 16, 1, 16, 16) bit_identical=True PSNR=inf
[mesh-regression] flux:cp            shape=(1, 256, 64)       bit_identical=True PSNR=inf
[mesh-regression] hunyuan:cp:sample  shape=(1, 16, 3, 32, 32) bit_identical=True PSNR=inf
[mesh-regression] qwen:cp            shape=(1, 256, 64)       bit_identical=True PSNR=inf
[mesh-regression] ltx2:cfg:video     shape=(2, 8, 8)          bit_identical=True PSNR=inf
[mesh-regression] ltx2:cfg:audio     shape=(2, 2, 8)          bit_identical=True PSNR=inf
[mesh-regression] PASS: all model/mode pairs bit-identical to 93fac4c
[mesh-regression] PASS: hyv15 smoke
[mesh-regression] ALL PASS
```

Byte-exact (`torch.equal`, max_abs_diff = 0.0) ⇒ PSNR → ∞, as required: the
refactor changes only which subgroup each collective fires in, and for these
configs the new axis meshes equal the legacy dp mesh (confirmed in the NxD
init log: legacy `dp_groups=[[0, 2], [1, 3]]` at tp=2/world=4 — exactly the
cfg/cp axis mesh). HunyuanVideo-1.5 supports neither CFG nor CP (no
collective changed), so it runs a tp-only compile/load/forward smoke
(`scripts/hunyuan15_tiny_compile_smoke.py`, tp=4) on the refactor checkout —
PASS. Two tiny-checkpoint geometry notes baked into the runner: the Flux
wrapper hardcodes a 128-dim rope (`FluxPosEmbed(axes_dim=(16,56,56))`), and
HunyuanVideo's dual-stream attention kernel needs head_dim=128 with
128-multiple sequence shards.

### Reproduce

```bash
PYTHONPATH=. pytest tests/unit -q                 # CPU suite
bash scripts/mesh_regression_smoke.sh             # device regression + smoke
```

## Known limitations / follow-ups

- **DP feature unwired** (by design, this task = groups only): `dp_degree > 1`
  builds a dp group but no request routing / replica logic consumes it yet.
- **cfg=2 × cp=2 expressible but rejected** by pipelines; enabling it for Wan
  is a natural follow-up now that the axes are orthogonal (validate against
  serial-CFG at cp=2).
- **cp degree is inferred** in the backend (`world / (tp·cfg·dp)`) because the
  backbone configs carry only the boolean flag, exactly as pre-refactor; the
  product==world assertion catches any mismatch. Threading the explicit degree
  can ride the DP feature.
- The regression uses tiny random-weight backbones (bit-identity needs
  identical weights, not real ones); real-checkpoint parity gates
  (`wan_ring_parity_smoke.sh` etc.) still apply when weights are on disk.
- Ring cp_mode is exercised at the replica-group level (same `get_cp_mesh()`
  feeds both modes) but the device pairs above ran gather_kv; the existing
  `*_ring_parity_smoke.sh` gates cover ring numerics when real weights are
  available.

## Commits

```
343ad8c test(device): bit-identity mesh regression (cfg=2 and cp=2 vs pre-refactor) + smoke
97954ce test: enforce dp_group carries no collectives (static audit + ops surface)
97cab00 refactor(ltx2): CFG merge on cfg_group
96bc1ec refactor(qwen): CP collectives on cp_group
8ec5dfc refactor(hunyuan): CP collectives on cp_group
822ae23 refactor(flux): CFG/CP collectives on their own axis groups; conditional group init
74656ea refactor(wan): CFG merge on cfg_group, CP on cp_group; de-parasitize dp_group
2b8ce3a refactor(cp): ring/joint-ring replica groups from the cp axis mesh, not NxD dp_group
914c640 feat(ops): axis-explicit group ops; drop get_data_parallel_group/get_dp_rank_spmd from the surface
4fce8ca feat(mesh): Trainium ProcessGroupManager building per-axis subgroups from one spec
01e21a0 feat(mesh): dp_degree axis + mesh_spec on DiffletParallelConfig (cache-key additive)
2d3c67c feat(mesh): MeshSpec rank<->(dp,cfg,cp,tp) math with per-axis subgroup meshes
28ffa13 docs: implementation plan for orthogonal dp/cfg/cp/tp parallel mesh
```
