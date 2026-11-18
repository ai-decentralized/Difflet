# Static block-sparse attention on trn3: kernel level at `wan_81f` (E2)

Plan: `docs/plans/2026-11-18-attn-constmax-sparse-experiments.md`, section E2. Branch
`exp/attn-constmax-sparse`. Run on 2026-11-18, devices 8-13. Kernel at `098dca20`, probe at
`2c98f795`. The kernel and mask files did not change after `098dca20`. E1 committed to the same
branch during the runs, which is why the JSONs record later heads; every JSON has
`kernel_dirty: false`.

## 0. Summary

**Measured**

- The static K/V-tile mask compiles. The NKI compiler took the masked loop structure with no
  change to the kernel's loops.
- **Full density is the unmasked kernel.** Bit-identical outputs on the CPU simulator (6 cases)
  and on the device at `wan_81f` (4 runs, 0 differing elements). On the device it is also
  bit-identical to the earlier campaign's attention_cte outputs (ScalarE and VectorE exp).
- **Masked outputs are right.** Against the masked fp32 reference, the masked kernel has rel L2
  2.8e-3 to 3.1e-3 at `wan_81f`. The unmasked kernel has 3.106e-3 against dense fp32.
- **Time is affine in tile density.** Fits give about 10 to 12 ms of fixed cost plus 41 ms
  (ScalarE exp) or 54 ms (VectorE exp) per unit of density.
  - At about 25% density: 0.36 to 0.41 of full.
  - At about 50%: 0.59 to 0.64 of full.
  - At about 75%: 0.79 to 0.85 of full.
- **Fast paths.**
  - ScalarE exp at `temporal:2` (24.0%): 20.79 ms, 0.400 of the 51.97 ms cte baseline.
  - VectorE exp at `temporal:2`: 22.88 ms, 0.360 of the 63.54 ms shipped baseline.
- **What the engine captures show at 51% density.**
  - Per-tile work scales exactly with density: MATMUL, the MM1 evict, the exp and `sync/WRITE`.
  - Per-(Q group, section) work does not scale: DMA (20,480 instructions), memsets (20,400), and
    the flash bookkeeping on ScalarE and VectorE.

**Inferred**

- The fixed cost is the per-(group, section) flash chain, and the latency it exposes.
  - That chain is the row-max reduce, the correction, and the read-modify-write of the output
    through HBM between sections.
  - Every (group, section) pair runs it, even the 57% of pairs that `temporal:2` leaves empty.
- With iid N(0,1) inputs, the rel L2 against dense fp32 (0.59 to 1.86) is set by density alone.
  It follows sqrt(1/density - 1). It says nothing about real-model quality, which needs a model
  run. That run is not part of this dispatch.

## 1. The kernel change

`difflet/backends/neuron/ops_impl/mx_attention_kernel.py` is the attention_cte fork of
feat/mxfp8-sage-attention at `5d5b83c5`. It was copied verbatim in `4df8a36b`, with its header
(nkilib 2.14 attention_cte sha256 `608f0c03...`), its CPU reference `difflet/quant/mx_attention.py`
and its tests (92 pass here). The MX paths are kept as they were. The mask refuses them (below).

`098dca20` adds `kv_tiles` to `mx_attention_fwd` and `_attention_cte`.

- **Format.** One tuple per 128-query group (`ceil(S_q / 128)` of them). Each tuple lists the
  global 512-key tiles the group computes.
- **Static.** It is a static kernel argument. nki's trace-cache key holds it by value
  (`CompileKernel._hashable_arg`). There is no runtime indirection.
- **Skipping.** `_tile_kept` gates three selections that already existed:
  - `_k_tile_selection`: MM1 and the evict with row max.
  - `_exp_tile_selection`: the exp, the row-sum chain and the P transpose.
  - `_pv_mm2_tile_selection`: MM2, and the PSUM-accumulate evict decision.

  A skipped tile issues no instruction.
- **Online softmax.** The running max, sum and correction code is unchanged.
  - A skipped tile leaves its partial max at `_FLOAT32_MIN` and its exp sum at 0.
  - A section where the group keeps no tile leaves the running max as it was (correction 1). It
    must add a zero output, so `_write_back_impl` zeroes `mm2_sb` for such a section
    (`_section_has_tiles`). Without that zeroing, `mm2_sb` holds an earlier group's MM2 result,
    and the structured test cases fail (mutation check, section 3).
  - Before the group's first kept tile, the running max is `_FLOAT32_MIN`. The first kept
    section's correction, exp(`_FLOAT32_MIN` - max) = 0, drops the zeros that came before.
- **Refused inputs.** An empty row (0/0), a wrong row count, an out-of-range tile, and `kv_tiles`
  together with `pv_mx`, `qk_mx` or bounds. The bounded and MX combinations are untested.

## 2. Masks

`difflet/backends/neuron/ops_impl/attn_tile_masks.py` holds `window_tiles` (pure Python),
`density` and `masked_attention_reference` (fp32, the skipped scores at -inf).

**Geometry.** Wan 2.1 at 480x832x81 has 21 latent frames of 30 x 52 tokens: 32,760 tokens in
frame-major raster order. That order is the Conv3d patch embedding's
`[B, C, F, H, W].flatten(2)` (`modeling_wan.py:978`). This gives 256 query groups and 64 key
tiles; the last tile has 504 keys and the last group has 120 queries.

**Rule.** A group attends a tile when one of its queries and one of the tile's keys fall inside
the window. So every in-window token pair is kept, and the tile mask is a superset of the token
window.

**Windows.** A 512-key tile covers about 10 whole rows (52 columns each), so columns cannot be
windowed in this order. The two families are:

- **temporal:** key frames within +-W of a query frame;
- **STA-like 3D:** +-WT frames and +-WH rows. WH = 6 (13 of 30 rows) throughout. WT is chosen
  for density.

This "STA-like" family runs on Wan's native token order. A real STA would first permute tokens
into 3D cubes, which is a model-side change outside this experiment.

| mask | window | tile density | empty (group, section) pairs | contiguous tile runs per group |
|---|---|---|---|---|
| `temporal:2` | +-2 frames | 0.240 | 0.568 | 1.00 |
| `temporal:6` | +-6 frames | 0.534 | 0.307 | 1.00 |
| `temporal:10` | +-10 frames | 0.756 | 0.118 | 1.00 |
| `sta:3,6` | +-3 frames x +-6 rows | 0.243 | 0.516 | 4.39 |
| `sta:8,6` | +-8 frames x +-6 rows | 0.511 | 0.231 | 8.80 |
| `sta:16,6` | +-16 frames x +-6 rows | 0.753 | 0.000 | 12.93 |

The w values were picked from the density tables: temporal w = 0..20 and the (WT, WH) grid. A
"section" is the kernel's 8,192-key flash section: 16 tiles, 4 sections at `wan_81f`.

`tests/unit/backends/test_attn_tile_masks.py` (13 tests) checks:

- the six densities, the row format, and that density rises with the window up to full;
- symmetry: with 512-token query blocks the mask is symmetric;
- refinement: each 512 block's row is the union of its four 128-query groups' rows;
- by brute force on a 4 x 5 x 6 video, that a tile is kept exactly when it holds an in-window
  key;
- that the reference ignores skipped keys and equals SDPA without a mask.

## 3. Correctness

**CPU simulator.** `tests/unit/backends/test_neuron_mx_attention_kernel.py`, 20 new tests, all
passing. Shapes:

- `partial`: 1 x 602 x 602, a partial group and tile, one section;
- `sections`: 1 x 602 queries x 11,053 keys, two flash sections of 16 + 6 tiles, the last 301
  keys;
- `lnc2_odd`: 3 x 1,152 x 11,053, the LNC2 split of an odd batch over the sequence.

Masks:

- `random`: each (group, tile) kept with p = 0.5, seed 1, and an empty row given one random tile.
- `structured`: rows cycled over the groups — the first tile alone; the last (partial) tile
  alone, so section 0 is empty; section 0 only, so the last section is empty; tiles straddling
  the sections with gaps; a run to the end.

The reference is fp32 with the skipped scores at -inf. Numbers are from
`/mnt/nvme/difflet-exp/sparse/sim_table.json`.

| shape | mask | density | exp | full mask bit-identical to no mask | rel L2 vs masked fp32 | rel L2 vs dense fp32 |
|---|---|---|---|---|---|---|
| partial | full | 1 | ScalarE / VectorE | yes / yes | 2.25e-3 / 2.25e-3 | same |
| sections | full | 1 | ScalarE / VectorE | yes / yes | 2.70e-3 / 2.70e-3 | same |
| sections | random | 0.473 | ScalarE / VectorE | | 2.71e-3 / 2.71e-3 | 1.10 |
| sections | structured | 0.218 | ScalarE / VectorE | | 2.49e-3 / 2.49e-3 | 3.61 |
| lnc2_odd | full | 1 | ScalarE / VectorE | yes / yes | 2.75e-3 / 2.75e-3 | same |
| lnc2_odd | random | 0.500 | ScalarE / VectorE | | 2.71e-3 / 2.71e-3 | 1.09 |
| lnc2_odd | structured | 0.172 | ScalarE / VectorE | | 2.49e-3 / 2.49e-3 | 3.83 |

How to read the table:

- **Tolerance.** The test bound is 4e-3. That is the fork's own distance from fp32 at these
  shapes (2.7e-3) plus margin. The masked kernel stays within the unmasked kernel's error.
- **Mutation check.** With the `mm2_sb` zeroing removed, the 4 structured cases fail (random
  rows never leave a section empty).
- **Refusals.** 6 tests cover them.

**Device, `wan_81f`.** These are the probe outputs, compared as int16 views
(`/mnt/nvme/difflet-exp/sparse/probe/ident.json`).

| a | b | differing elements of 41,932,800 |
|---|---|---|
| ScalarE full, device 8 | ScalarE no mask, device 12 | 0 |
| ScalarE full, device 9 | ScalarE no mask | 0 |
| VectorE full, device 10 | VectorE no mask, device 12 | 0 |
| VectorE full, device 11 | VectorE no mask | 0 |
| ScalarE no mask | attention_cte `allow_vector_exp=False` (2026-11-13 evidence, section 9) | 0 |
| VectorE no mask | shipped attention_cte (2026-11-13 evidence, section 15h) | 0 |

The no-mask and full-mask arms compiled to the same NEFFs: the cache holds 14 NEFFs for the 16
(mask, exp) arms.

## 4. Timing at `wan_81f`

**Method.** `scripts/block_sparse/probe.py`, after the fork probe of feat/mxfp8-sage-attention
(evidence 2026-11-13 section 4c):

- **Inputs.** 10 x 32,760 x 128, unbounded; q, k, v ~ N(0,1) bf16, seed 0, scale 128^-0.5. These
  are that probe's inputs.
- **Region.** The compiled region holds the kernel's torch op alone, with no
  `ops_impl.attention` around it. E1 edits that file, and on the unbounded path it only reshapes.
- **Timing.** One warm call, then 20 calls and one sync, in each of 3 worker processes. Report
  the median and the spread, (max - min) / median.
- **Pinning.** One logical core (`NEURON_RT_VISIBLE_CORES=4d`) and a 16-thread CPU slice per
  lane, with `NEURON_NUMA_PIN=0`. `fuser` was idle before each worker, and the CPU slice was idle
  at each arm's start (`lanes.log`).
- **Lanes.** Each lane is one device, one CPU set and one exp engine, running full density and
  one family's three densities in sequence. The ratio is to that lane's own full-density arm.
- **Rerun.** The first full-density files of devices 8 and 10 were overwritten by devices 9 and
  11 (same file name). Those two arms were rerun as `full_dev8` / `full_dev10`, and reproduced
  their first medians, 52.329 -> 52.327 and 63.734 -> 63.727 ms (`lanes.log`).
- **Baselines.** The last column divides by the baselines of the 2026-11-13 evidence: 51.97 ms
  (cte with ScalarE exp, section 9) and 63.54 ms (shipped cte, section 7).

| exp | device (CPUs) | mask | density | runs (ms) | median ms | spread | ratio to full (same device) | ratio to baseline |
|---|---|---|---|---|---|---|---|---|
| ScalarE | 8 (0-7,96-103) | `full` | 1.000 | 52.33 / 52.32 / 52.33 | 52.33 | 0.02% | 1.000 | 1.007 |
| ScalarE | 8 | `temporal:2` | 0.240 | 20.79 / 20.78 / 20.79 | 20.79 | 0.08% | **0.397** | 0.400 |
| ScalarE | 8 | `temporal:6` | 0.534 | 32.76 / 32.74 / 32.76 | 32.76 | 0.06% | **0.626** | 0.630 |
| ScalarE | 8 | `temporal:10` | 0.756 | 41.72 / 41.75 / 41.74 | 41.74 | 0.06% | **0.798** | 0.803 |
| ScalarE | 9 (8-15,104-111) | `full` | 1.000 | 51.97 / 51.97 / 51.97 | 51.97 | 0.01% | 1.000 | 1.000 |
| ScalarE | 9 | `sta:3,6` | 0.243 | 21.30 / 21.29 / 21.30 | 21.30 | 0.04% | **0.410** | 0.410 |
| ScalarE | 9 | `sta:8,6` | 0.511 | 33.17 / 33.17 / 33.18 | 33.17 | 0.05% | **0.638** | 0.638 |
| ScalarE | 9 | `sta:16,6` | 0.753 | 44.40 / 44.38 / 44.37 | 44.38 | 0.05% | **0.854** | 0.854 |
| ScalarE | 12 (32-39,128-135) | no mask | 1.000 | 52.32 / 52.29 / 52.29 | 52.29 | 0.06% | | 1.006 |
| VectorE | 10 (16-23,112-119) | `full` | 1.000 | 63.73 / 63.73 / 63.72 | 63.73 | 0.01% | 1.000 | 1.003 |
| VectorE | 10 | `temporal:2` | 0.240 | 22.88 / 22.88 / 22.89 | 22.88 | 0.03% | **0.359** | 0.360 |
| VectorE | 10 | `temporal:6` | 0.534 | 38.30 / 38.26 / 38.53 | 38.30 | 0.69% | **0.601** | 0.603 |
| VectorE | 10 | `temporal:10` | 0.756 | 50.02 / 50.03 / 50.04 | 50.03 | 0.03% | **0.785** | 0.787 |
| VectorE | 11 (24-31,120-127) | `full` | 1.000 | 63.55 / 63.55 / 63.55 | 63.55 | 0.01% | 1.000 | 1.000 |
| VectorE | 11 | `sta:3,6` | 0.243 | 23.30 / 23.30 / 23.32 | 23.30 | 0.08% | **0.367** | 0.367 |
| VectorE | 11 | `sta:8,6` | 0.511 | 37.73 / 37.71 / 38.04 | 37.73 | 0.86% | **0.594** | 0.594 |
| VectorE | 11 | `sta:16,6` | 0.753 | 51.47 / 51.49 / 51.49 | 51.49 | 0.03% | **0.810** | 0.810 |
| VectorE | 12 | no mask | 1.000 | 63.64 / 63.66 / 63.65 | 63.65 | 0.02% | | 1.002 |

Every spread is below 1%. Full density costs the same as no mask: 52.33 vs 52.29 ms, and 63.73 vs
63.65 ms. Devices differ by up to 0.7% at the same NEFF (devices 8 and 9: 52.33 vs 51.97), which
is why each ratio uses its own device's full-density arm.

**Least-squares fit, t = a + b * density** (four points each, full included):

| family | exp | a, fixed (ms) | b, per unit density (ms) | max residual (ms) | a / t(full) |
|---|---|---|---|---|---|
| temporal | ScalarE | 10.75 | 41.36 | 0.29 | 0.21 |
| STA-like | ScalarE | 11.94 | 41.11 | 1.49 | 0.23 |
| temporal | VectorE | 9.85 | 53.59 | 0.35 | 0.16 |
| STA-like | VectorE | 10.46 | 53.53 | 0.72 | 0.16 |

**Reading.**

- **Measured.** Time falls with density close to linearly, but not in proportion: a fixed
  10 to 12 ms remains. That is 16% (VectorE) to 23% (ScalarE) of the full-density time.
- **Measured.** The STA-like family costs more than the temporal family at matched density.
  - At ~75% the gap is 6% on ScalarE (0.854 vs 0.798) and 3% on VectorE.
  - Its fit has the larger residual: on ScalarE, `sta:16,6` sits 1.5 ms above the line.
- **Inferred.** The STA cost comes from scattered tiles: 4 to 13 runs per group against 1 for
  the temporal family. Large tiles hold 1 to 3 kept tiles, which breaks the four-tile row-max
  chain and adds per-large-tile read-outs. The capture's `DVE_READ_ACCUMULATOR` count supports
  this, but only at 51%; 75% was not captured.

## 5. Where the fixed cost sits (engine capture)

**Method.** The optional capture, read-only use of `/mnt/nvme/difflet-mxattn/probe/3b/capture.sh`
and `engines.py`:

- The probe's NEFFs were replayed on device 13 (CPUs 40-47,136-143) under `neuron-explorer`.
  This was executions 1-4, with the 3rd profiled.
- Busy time is the union of instruction intervals per physical core, averaged over the 2 cores.
- The exp opcode identifies the engine (the cache holds the NEFFs by hash).
- 51% density is `sta:8,6`. The `temporal:6` NEFF did not exist yet when the captures started.

| | ScalarE full | ScalarE `sta:8,6` | VectorE full | VectorE `sta:8,6` |
|---|---|---|---|---|
| span ms | 51.88 | 33.11 | 63.61 | 37.71 |
| tensor busy ms | 45.27 | 23.78 | 44.25 | 23.76 |
| vector busy ms | 42.64 | 23.89 | 62.03 | 34.25 |
| scalar busy ms | 39.56 | 20.86 | 1.63 | 1.63 |
| gpsimd busy ms | 1.29 | 1.17 | 5.58 | 1.66 |
| hbm_dma busy ms | 7.19 | 7.20 | 16.26 | 12.75 |
| `tensor/MATMUL` ms (instr) | 45.19 (818,911) | 23.72 (419,000) | 44.21 (819,200) | 23.73 (419,000) |
| `vector/TENSOR_SCALAR_CACHE_REDUCE` (MM1 evict) | 39.44 (163,807) | 20.40 (83,800) | 37.66 (163,840) | 19.57 (83,800) |
| `vector/EXPONENTIAL` | 0 | 0 | 20.34 (163,840) | 10.59 (83,800) |
| `scalar/ACTIVATE` (exp + flash correction) | 39.54 (179,134) | 20.85 (99,160) | 1.63 (15,360) | 1.63 (15,360) |
| `hbm_dma/DMA_DIRECT2D` | 7.19 (20,472) | 7.20 (20,480) | 16.26 (20,480) | 12.75 (20,480) |
| `gpsimd/MEMSET` | 1.28 (20,397) | 1.16 (20,400) | 5.58 (20,400) | 1.65 (20,400) |
| `vector/TENSOR_REDUCE` | 1.33 (20,473) | 1.37 (20,480) | 1.52 (20,480) | 1.41 (20,480) |
| `vector/SCALAR_TENSOR_TENSOR` | 1.11 (7,675) | 1.12 (7,680) | 1.19 (7,680) | 1.18 (7,680) |
| `vector/TENSOR_TENSOR` | 0.43 (7,677) | 0.55 (7,680) | 0.67 (7,680) | 0.64 (7,680) |
| `vector/COPY` | 1.99 (17,913) | 1.44 (15,550) | 1.66 (17,920) | 1.38 (15,550) |
| `vector/DVE_READ_ACCUMULATOR` | 2.14 (40,951) | 1.42 (27,350) | 2.64 (51,200) | 1.82 (35,220) |
| `sync/WRITE` | 3.31 (168,301) | 1.70 (86,148) | 3.31 (168,410) | 1.70 (86,148) |

The ScalarE full capture is missing ~0.04% of instructions (818,911 MATMUL against 819,200).

**Measured**

- Per-tile work scales with the tile density exactly: 419,000 / 819,200 = 83,800 / 163,840 =
  0.5115, the mask's density. This covers MATMUL, the MM1 evict, the exp and `sync/WRITE`.
- These counts do not move with density:
  - DMA instructions: 20,480;
  - GpSimd memsets: 20,400;
  - ScalarE flash-correction activations: 15,360 = 2 per (group, section) x 3 later sections x
    256 groups x 10 heads;
  - VectorE `TENSOR_REDUCE`, `SCALAR_TENSOR_TENSOR` and `TENSOR_TENSOR`: 20,480 / 7,680 / 7,680.
- On ScalarE, the span minus the busiest engine grows from 6.6 ms (full) to 9.2 ms (51%). On
  VectorE it grows from 1.6 to 3.5 ms.

**Inferred**

- The fixed 10 to 12 ms is the per-(Q group, flash section) chain that the mask does not remove:
  - partial-max memsets and reduce;
  - the running max / sum / correction;
  - the output read-modify-write through HBM between the 4 sections (`_write_back_impl`: a store
    per group, section and head, plus a load in sections 1-3);
  - the K/V/Q loads. The 20,480 `DMA_DIRECT2D` instructions are 256 x 10 x (1 + 3 x 2) = 17,920
    output stores and loads, plus 2,560 V-tile loads.
  - and the latency of this chain, which less per-tile work hides less well.
- The chain runs for every (group, section) pair, including empty ones: 57% of pairs at
  `temporal:2`, 52% at `sta:3,6`.
- Two kernel changes could cut this fixed cost. Neither was tried here.
  1. Skip the bookkeeping and the output round trip for a (group, section) with no kept tile. The
     last section still needs its normalization.
  2. Process a sparse group's kept tiles as one section when they fit in SBUF (at most 16
     tiles), so no output round trip through HBM is needed.

## 6. Accuracy at `wan_81f`

The first worker's output of each arm, against fp32 references on the host. Each head is
evaluated in 1,024-query chunks, cached under `/mnt/nvme/difflet-exp/sparse/ref/`. rel L2 is
taken over all 10 heads, in float64.

| mask | density | exp | rel L2 vs dense fp32 | sqrt(1/density - 1) | rel L2 vs masked fp32 (worst head) | non-finite |
|---|---|---|---|---|---|---|
| no mask / `full` (all 4 devices) | 1.000 | both | 3.106e-3 | 0 | 3.106e-3 (3.214e-3) | 0 |
| `temporal:2` | 0.240 | ScalarE / VectorE | 1.823 | 1.782 | 2.816e-3 / 2.817e-3 (2.831e-3) | 0 |
| `temporal:6` | 0.534 | ScalarE / VectorE | 0.984 | 0.935 | 2.985e-3 / 2.984e-3 (3.015e-3) | 0 |
| `temporal:10` | 0.756 | ScalarE / VectorE | 0.615 | 0.568 | 3.097e-3 / 3.097e-3 (3.151e-3) | 0 |
| `sta:3,6` | 0.243 | ScalarE / VectorE | 1.857 | 1.766 | 2.844e-3 / 2.844e-3 (2.859e-3) | 0 |
| `sta:8,6` | 0.511 | ScalarE / VectorE | 1.048 | 0.977 | 3.021e-3 / 3.020e-3 (3.055e-3) | 0 |
| `sta:16,6` | 0.753 | ScalarE / VectorE | 0.592 | 0.573 | 3.094e-3 / 3.094e-3 (3.167e-3) | 0 |

- **Measured, against masked fp32.** The kernel's own error is 2.8e-3 to 3.1e-3, at or below the
  dense kernel's 3.106e-3. This is the kernel tolerance: the bf16 P and the bf16 output.
- **Measured, against dense fp32.** 0.59 to 1.86: the approximation error of dropping keys, not
  a kernel error.
- **Inferred: what the dense-fp32 number means on these inputs.**
  - With iid N(0,1) q, k and v, every key is equally relevant and the output averages
    independent values. So keeping a fraction p of the keys gives an expected rel L2 near
    sqrt(1/p - 1), and the measured values sit 2 to 8% above it.
  - On these inputs the number is therefore set by density alone. It says nothing about Wan,
    whose attention has the spatio-temporal locality these windows assume.
  - **Real-model quality needs a model run** (plan E2's follow-up: Wan 2.1 480x832x81 with a
    temporal mask, PSNR against bf16 and its 22.75 dB floor). That run is not part of this
    dispatch.

## 7. Caveats

- **Kernel level only.** One attention call at the per-rank shape. Step time and quality of a
  Wan run are not measured.
- **Inputs.** Synthetic N(0,1) (section 6). Timing does not depend on the values: the masks are
  static, and full density is bit-identical to no mask.
- **STA-like family.** It is defined on Wan's native token order, so it windows frames and rows
  only. A true STA (tokens permuted into 3D cubes) is a model-side change that was not tried.
- **Scope of the mask.** It covers unbounded bf16 attention only. Bounds (HunyuanVideo,
  Qwen-Image) and the MX paths are refused with `kv_tiles`, because they are untested.
- **Capture coverage.** The engine capture is one execution per NEFF and covers 51% density
  only. The fixed-cost attribution in section 5 is inferred from instruction counts and engine
  busy times, not from a kernel change that removes it.
- **Rerun.** Devices 8 and 10's first full-density files were overwritten (section 4). Their
  reruns reproduced the logged medians within 0.01%.

## 8. Paths

- **Code.**
  - `difflet/backends/neuron/ops_impl/mx_attention_kernel.py` (`kv_tiles`)
  - `difflet/backends/neuron/ops_impl/attn_tile_masks.py`
  - `tests/unit/backends/test_neuron_mx_attention_kernel.py` (the `kv_tiles` tests at the end)
  - `tests/unit/backends/test_attn_tile_masks.py`
  - `scripts/block_sparse/probe.py`
  - `scripts/block_sparse/lane.sh`
- **Evidence, on the host only.** This repo ignores everything under `docs/verification/` except
  `*.md`. All of it is under `/mnt/nvme/difflet-exp/sparse/`:
  - `probe/<scalar|vector>/<mask>.json`: one per arm, with every run's ms, the per-head rel L2,
    the CPUs, the `fuser` record and the tree. Next to each: `.out.pt`, `.run<i>.{json,log}`.
  - `probe/lanes.log`, `probe/ident.json`;
  - `sim_table.json`;
  - `cap/<name>/`: the captures (4.5 GB), with `engines.json`. Here `full_a` / `sta86_a` are
    VectorE and `full_b` / `sta86_b` are ScalarE.
  - `ref/`: fp32 references per head. `cache/`: NEFF, HLO and NKI trace caches.
  - The scripts as run: `sim_table.py`, `ident.py`, `table.py`, `cap_table.py`,
    `empty_sections.py`, `launch.sh`, `rerun_full.sh`, `captures.sh`.
- **Reproduce one arm.**

  ```
  NEURON_RT_VISIBLE_CORES=<4d> NEURON_NUMA_PIN=0 taskset -c <cpus> python scripts/block_sparse/probe.py --mask temporal:6 [--vector-exp] --out <dir>/temporal_6.json
  ```

  Or use `ROOT=<dir> scripts/block_sparse/lane.sh <device> <cpus> <scalar|vector> <masks...>`.
