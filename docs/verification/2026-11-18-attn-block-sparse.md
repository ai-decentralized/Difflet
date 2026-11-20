# Static block-sparse attention on trn3: kernel level at `wan_81f`, and Wan 2.1 (E2)

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
  It follows sqrt(1/density - 1). It says nothing about real-model quality; section 9 measures
  that.

**Model level, Wan 2.1 480x832x81 (section 9)**

- **Measured.**
  - **Gate.** The routing gate passed: at full density the new route gives the shipped bf16
    clip bit for bit (`35177856`, 10 steps).
  - **Speed.** `temporal:6` (density 0.534) runs the DiT step at 8,119 ms against bf16's
    10,121 ms: 0.802. Warm e2e is 0.802. `temporal:10` (0.756) gives 0.896. The model got 98%
    and 96% of the saving the kernel numbers predict.
  - **Quality fails the floor by far.**
    - `temporal:6`: 11.08 dB PSNR against bf16, worst frame 10.35. SSIM 0.273, LPIPS 0.597.
    - `temporal:10`: 12.70 dB, worst frame 10.88. SSIM 0.326, LPIPS 0.478.
    - For comparison: the floor is 22.75 dB, and MXFP8 scored 15.08 dB.
  - **What the `temporal:6` clip shows.**
    - It drifts to another scene and another cat without a cut.
    - It has broken motion around frame 12: a ghosted, doubled body.
    - Its last frames are oversaturated.
  - **What the `temporal:10` clip shows.** It keeps one cat and smooth motion, but in another
    composition from bf16, and its last frame is posterized.
- **Inferred.** A fixed temporal window, applied at every step and every layer, cuts the
  long-range attention Wan uses for global consistency. The speed is real; this mask family is
  not usable as is.

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
  - **Real-model quality needs a model run.** Section 9 has it: Wan 2.1 480x832x81 with
    temporal masks, scored against bf16 and its 22.75 dB floor.

## 7. Caveats

- **Kernel level (sections 1-8).** One attention call at the per-rank shape. Section 9 is the
  Wan run, with its own caveats.
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
## 9. Model level: Wan 2.1 480x832x81

The plan's conditional step: time scaled close to density (section 4), so one Wan 2.1 run with the
temporal mask. Run on 2026-11-18 from `7631c73a`, devices 8-10.

### 9a. The hook

`DIFFLET_EXP_ATTN_SPARSE` is opt-in, for example `temporal:6` or `sta:8,6`. It routes Wan's
self-attention (`attn1`) of every block to the fork with the `kv_tiles` mask. Cross-attention
(`attn2`) is unchanged, and with the variable unset nothing changes.

- **Spec.** `WanTransformer3DModel.forward` stamps each `attn1` with the spec plus the call's
  patch grid, for example `temporal:6@21x30x52`.
- **Call path.** `_attn_kernel` then calls the backend's `block_sparse_attention`. It is resolved
  through `difflet.ops` dispatch at model construction, so `modeling_wan` stays backend-neutral.
- **Neuron op.** `ops_impl/block_sparse_attention.py` registers the torch op
  `difflet_neuron::block_sparse_attention`, using the pattern of feat/mxfp8-sage-attention's
  `ops_impl/mx_attention.py`.
  - It runs the fork with `allow_vector_exp=True`, shipped attention_cte's exp, so the mask is
    the only difference from shipped bf16.
  - It builds `kv_tiles` from the spec when the kernel is traced
    (`attn_tile_masks.kv_tiles_for`).
  - E1's `ops_impl/attention.py` is not touched.
- **CPU.** The CPU backend's op is the masked fp32 reference, used by the routing tests.

**Shapes.**

- **CFG.** CFG runs as two sequential batch-1 forwards on the tp4 group (bench.sh D3).
- **Heads.** Each rank holds 40 / 4 = 10 heads, so the kernel gets `[10, 32760, 128]`. That is
  exactly the probe's `wan_81f`. Any batch row or head takes the one mask.
- **No padding.** The kernel takes 32,760 tokens as 255 full query groups plus one of 120, and
  63 full key tiles plus one of 504, so nothing is padded.
- **Refused inputs.** A spec whose grid holds a different token count from the call, and context
  parallelism (it shards the sequence the mask indexes).

**Tests.**

- `tests/unit/models/wan/test_wan_attn_sparse.py` (5 CPU tests):
  - unset routes nothing;
  - set, only `attn1` of each block is routed, with the call's grid;
  - a window over every frame equals dense attention;
  - context parallelism is refused;
  - a grid mismatch is refused.
- `test_attn_tile_masks.py`: the spec parser.

**Device check of the op.** At `wan_81f`, `difflet_neuron::block_sparse_attention` with
`temporal:6@21x30x52` equals the kernel probe's VectorE `temporal:6` output in all 41,932,800
elements (`/mnt/nvme/difflet-exp/sparse/op_check.{py,json}`).

### 9b. Routing gate

**Method.** `scripts/block_sparse/wan_arms.sh` runs bench.sh's `wan21_official` flags, read from
`scripts/quant_bench/bench.sh`:

- revision `38ec498c`, 480x832x81, tp4;
- seed 42, CFG 5.0, the model card's negative prompt;
- prompt "a cat walking through a garden";
- bench.sh's environment hygiene, and `DIFFLET_NKI_TRACE_CACHE=1`.

Each arm compiles into its own `--cache-dir`. The other arms link bf16's `weights-bf16` copy
once it is complete; the copies are byte-identical, and this saves disk.

| arm | spec |
|---|---|
| bf16 | unset (shipped attention_cte) |
| full | `temporal:20`: every tile, the gate |
| t6 | `temporal:6`, density 0.534 |
| t10 | `temporal:10`, density 0.756 |

**Result (measured).** 10-step generates of bf16 (device 8) and full (device 9), in parallel:

| arm | clip md5 | latents md5 | step median (n=9) |
|---|---|---|---|
| bf16 | `35177856` | `a56b064c` | 10,113.6 ms |
| full | `35177856` | `a56b064c` | 10,113.2 ms |

**The route was taken.** The full arm's four DiT-block NEFFs (one per rank, ~89.6 MB) sit under
cache keys that bf16's cache does not have. The custom call is the new op. The other 291 NEFFs
match by key. The t6 arm's block NEFFs are 62.4 MB.

### 9c. Timed arms

**Method.**

- One generate process per arm, with two identical 50-step requests (`--requests`).
- The arms ran back to back on device 8 (cores 32-35) and CPUs 0-15,96-111: bf16, then t6, then
  t10 (2026-11-18 09:53-10:4x). `fuser` was idle and the CPU set idle at each start (`runs.log`).
- The step median is the `[wan]` request line's: n=49, per request.
- The arm's spread is (max - min) / median of its two request medians.
- e2e is the request time: prompt encode, 50 steps and decode. "Warm" is the second request.
- **Overlap.** t10's compile ran on device 9 and CPUs 16-31,112-127 from 09:53 to 10:06. That
  overlapped bf16's first request (09:55-10:04), which is 1.1 ms slower than its second. The
  second requests saw no other job of this experiment.

| arm | spec | density | step median ms (request 1 / 2) | spread | step / bf16 | warm e2e s (request 2) | e2e / bf16 | process wall | clip md5 (both requests) |
|---|---|---|---|---|---|---|---|---|---|
| A: bf16 | unset | 1 | 10,122.0 / 10,120.9 | 0.01% | 1.000 | 506.23 | 1.000 | 19:32 | `963566ba` |
| B: t6 | `temporal:6` | 0.534 | 8,116.8 / 8,119.1 | 0.03% | **0.802** | 406.22 | **0.802** | 15:51 | `c7a85ea1` |
| t10 | `temporal:10` | 0.756 | 9,066.5 / 9,068.2 | 0.02% | **0.896** | 453.59 | **0.896** | 17:43 | `0cb275d2` |

- **Ratios** are the second requests'. Each step's worst iteration is within 0.6% of its median
  (A 10,179.0 / 10,149.7, B 8,141.9 / 8,153.3, t10 9,103.1 / 9,095.4 ms).
- **Baseline reproduced.** A's clip `963566ba` and first latents `cb3e28cd` are bit-identical to
  the shipped bf16 `wan21_official` run of feat/mxfp8-sage-attention (2026-11-15, its task6
  `bf16/timed`, step median 10,114.8 / 10,126.8 ms there).
- **Deterministic.** Each arm's two requests give one clip md5.
- **Against the kernel numbers.** Section 4's VectorE lane on device 10 gives per call, against
  63.73 ms at full density: `temporal:6` 38.30 ms and `temporal:10` 50.03 ms. With 80 `attn1`
  calls per step (40 blocks x 2 sequential CFG forwards), that predicts:
  - **t6.** A saving of 2,034 ms per step (8,087 ms). Measured: 2,003 ms (98.5%).
  - **t10.** A saving of 1,096 ms (9,025 ms). Measured: 1,054 ms (96%).
  - **Inferred.** The rest of the step does not change, and `attn1` is 80 x 63.7 = 5.1 s, about
    50% of the shipped step.

### 9d. Quality

**Method.** `scripts/compare_difflet_runs.py`, each arm's first clip and latents against A's:

- latent rel L2 and cosine, in float64;
- PSNR, SSIM and LPIPS (AlexNet) per frame on the decoded `.mp4`, then the mean, the worst frame
  and its index.

The floor and the screen are the 2026-11-07 evidence, section 12c (`quality.py`, the same
per-frame mp4 metrics):

- the floor is bf16 at tp8 against bf16 at tp4;
- the screen is PSNR >= min(30, floor - 4), SSIM >= min(0.97, floor - 0.10) and
  LPIPS <= max(0.05, 2 x floor).

`frames.py` adds the per-frame PSNR and each clip's mean absolute frame-to-frame difference (a
flicker or motion signal). The frames are kept under
`/mnt/nvme/difflet-exp/sparse/model/frames/`: a montage of frames 0 / 40 / 80 against bf16, a
strip of frames 38-42, and strips around each arm's frame-difference peaks.

| clip against bf16 | PSNR mean (worst frame) | SSIM | LPIPS | latent cosine / rel L2 | frame-to-frame difference / bf16's (mean, max) |
|---|---|---|---|---|---|
| t6 (0.534) | **11.08** (10.35, frame 0) | 0.273 | 0.597 | 0.596 / 0.919 | 2.29x, 2.97x |
| t10 (0.756) | **12.70** (10.88, frame 80) | 0.326 | 0.478 | 0.747 / 0.723 | 1.58x, 1.82x |
| floor: bf16 tp8 (12c) | 22.75 (18.31) | 0.784 | 0.098 | | |
| screen thresholds | 18.75 | 0.684 | 0.197 | | |
| MXFP8, for scale (12c) | 15.08 (12.96) | 0.391 | 0.351 | | |

**Measured.** Both masks are far below the floor and fail all three screens. They are further
from bf16 than MXFP8, which itself fails. More density helps only a little: +1.6 dB from 0.534
to 0.756.

**What the frames show** (looked at; descriptive):

- **bf16.** One shot: a grey-and-white cat on a brick path beside a rock, yellow flowers and a
  water bowl. The cat stands, then turns toward the camera. Little motion; frame differences
  2-7.
- **t6.**
  - **A different sample from frame 0.** A wide low shot of a lawn and a white porch, a small
    cat far off.
  - **Drift.** The clip then moves, without a hard cut, to a close-up of a black-and-white cat
    at a wooden fence (frame 40). It ends on an oversaturated, high-contrast close-up of a face
    (frame 80).
  - **Broken motion.** Around frames 12-13 (the frame-difference peak, 21.5 against bf16's
    maximum 7.2) the cat's body ghosts: two overlapping bodies, and the coat pattern shifts. A
    second peak at frames 44-48 is a fast head turn.
  - **No flicker.** Within a segment the motion is smooth frame to frame (frames 38-42 are
    coherent).
  - **Reading.** The failure is temporal consistency at long range, plus late colour blow-out.
    It is not frame-to-frame flicker.
- **t10.**
  - **Coherent.** One brown-and-white cat throughout, in profile then turning to the camera, in
    a scene with leafy plants and a stone edge. That is not bf16's composition, but the motion
    arc is similar.
  - **No ghosting.** Its frame-difference peak (frames 48-52, 13.2) is the cat walking.
  - **Late degradation.** The last frames degrade: posterized, oversaturated fur, blur and dark
    streaks at the bottom right. Frame 80 is its worst PSNR.

**Inferred.**

- PSNR against bf16 punishes any other valid sample, which is why the floor itself is only
  22.75 dB. Still, t6 shows real artifacts (ghosting, identity drift, colour blow-out), and t10
  shows late blow-out.
- A fixed +-W-frame window at every step and every layer removes the long-range temporal
  attention Wan uses for global layout and identity. The early, layout-forming steps are the
  likely place it matters; this run cannot separate them.

### 9e. Estimates (inferred, not measured)

- **ScalarE exp at model level.** Section 4's ScalarE lane on device 8 is faster per call than
  its VectorE lane on device 10:
  - dense: 52.33 against 63.73 ms;
  - `temporal:6`: 32.76 against 38.30 ms;
  - `temporal:10`: 41.74 against 50.03 ms.

  With 80 calls per step, that would give, against the measured shipped 10,121 ms:
  - bf16 with the ScalarE exp: ~9.21 s (0.91);
  - t6: ~7.68 s (0.76);
  - t10: ~8.40 s (0.83).

  The VectorE exp was used here, so the mask is the only difference from shipped.
- **Removing the fixed cost** (section 5; not tried in this round, kept as a follow-up). The
  VectorE temporal fit's fixed part is 9.85 ms per call, and it is spread over the 1,024
  (group, section) pairs per head.
  - **Skipping empty pairs** (the bookkeeping and the output round trip; the last section still
    normalizes):
    - t6, 30.7% of pairs empty: ~3.0 ms per call, ~0.24 s per step, step ~7.88 s (0.78).
    - t10, 11.8% empty: ~1.2 ms per call, ~0.09 s per step.
  - **The whole fixed cost**, for example one section per sparse group and no HBM round trip:
    at most ~9.85 ms per call, ~0.79 s per step. At t6 that is a step of ~7.33 s (0.72); with
    the ScalarE exp as well (whose fit's fixed part is 10.75 ms), ~6.8 s (0.67).
  - Neither changes the quality verdict above.

### 9f. Caveats

- **One prompt, one seed, one resolution.** The quality is measured against bf16 at the same
  seed, not against a reference sample (no CPU fp32 reference at this shape).
- **Masks tried.** Only the temporal family reached the model: `temporal:6` and `temporal:10`,
  at every step and every layer. Untested:
  - the STA-like masks;
  - step schedules (dense early, sparse late) and per-layer choices;
  - token reordering into 3D cubes.
- **Timing set.** The timed arms ran back to back on one device and CPU set. t10's compile on
  another device and CPU set overlapped bf16's first request only. The ratios use second
  requests, which nothing of this experiment overlapped.
- **Disk.** The host's NVMe was at 98% (170 GB free) during the runs. Most of that is other
  jobs. This experiment's caches total ~70 GB, with the bf16 copy shared.

### 9g. Paths

- **Code.**
  - `difflet/models/wan/modeling_wan.py` (`DIFFLET_EXP_ATTN_SPARSE`, `_attn_kernel`)
  - `difflet/backends/neuron/ops_impl/block_sparse_attention.py`
  - `difflet/backends/cpu/ops_impl/block_sparse_attention.py`
  - `attn_tile_masks.kv_tiles_for`
  - `tests/unit/models/wan/test_wan_attn_sparse.py`
  - `scripts/block_sparse/wan_arms.sh`
- **Evidence, on the host** under `/mnt/nvme/difflet-exp/sparse/model/`:
  - `runs.log`, `gate_md5.txt`;
  - `<arm>/{compile,gate,timed}.log`, `md5.txt`, the clips, `work_*` (latents);
  - `quality/<arm>.{json,log}`;
  - `frames/` (montages, strips, `*_temporal.json`);
  - `cache/<arm>/`.
  - In `/mnt/nvme/difflet-exp/sparse/`: `op_check.{py,json}`, `compile_all.sh`, `gate.sh`,
    `timed.sh`, `score.sh`, `frames.py`.
