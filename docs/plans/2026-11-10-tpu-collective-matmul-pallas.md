# Pallas collective-matmul kernels for TPU sequence parallelism

Date: 2026-11-10 · Branch: `tpu-ltx2-sp` · Status: proposal, gated on step 1

## Goal

Replace the internals of `difflet/backends/tpu/ops_impl/collective_matmul.py`
(today a JAX `shard_map` ring run through `torch.tpu.pallas.jax_op`) with
hand-written Pallas TPU kernels, behind the same three functions every SP model
already calls:

| function | used by |
|---|---|
| `all_gather_matmul(x, linears)` — AG + column-parallel matmul, 1–4 weights | Wan, Qwen-Image, HunyuanVideo, FLUX, LTX-2 |
| `matmul_reduce_scatter(x, linear)` — row-parallel matmul + RS | same |
| `matmul2_reduce_scatter(x0, l0, x1, l1)` — two pairs, one RS | FLUX single blocks |

No model file changes. Trainium and CUDA paths are untouched (TPU-only module).

## Why

The JAX ring hides about 55% of the communication it replaces
(`sp_pair_probe`, Wan block, 2026-11-03: 6.33 → 4.95 ms per block of 8.8 ms;
the all-reduce was 2.5 ms). XLA decides the overlap of each hop's
`collective-permute` with its einsum, and every hop's partial result goes
through HBM. A kernel controls both: DMA start/wait around the matmul with
double buffering, and accumulators kept in VMEM.

Expected gain, Wan as the example: ~1.1 ms/block still exposed after SP; hiding
85–90% of it is 0.6–0.9 ms/block, **~7–10% of a step** on top of today's SP.
LTX-2 (all-reduce ~33% of a TP step) should gain at least as much; FLUX and
HunyuanVideo less.

Secondary: fusing work the JAX ring cannot (bias, gelu, an RMSNorm prologue,
group-wise int8 dequantization — the MiniMax-H3 probe OOMed at group 128 because
the compiled graph materialized every dequantized weight).

## Lessons from 2026-11-10 that shape the design

LTX-2 SP produced wrong output, hangs and core halts at 480x704x49 (2310 video
rows padded to 2312). Root cause, from `benchmark/tpu_profile/cm_padded_repro.py`,
per-rank HLO dumps and per-rank `jax.profiler` traces:

- torch_tpu compiles **one program per process**, and the ring is **inlined**
  into each (`collective-permute`s, all `channel_id=1`, paired by order only).
- Padding inside the compiled graph with **rank-constant** slice offsets let XLA
  specialize each rank differently (ranks whose shard lies inside the unpadded
  input folded the `concatenate` away); the per-rank programs diverged, a chip
  halted in the ring's first matmul fusion, the others waited in a collective.
- Every unpadded length 1024..6144 (incl. 2312) was right; plain JAX at 2312 was
  right; padding eagerly before the graph was right; padding in the graph with
  the offset as a **runtime tensor** (identical programs on every rank) was right.

Design rules that follow, for the kernels and for their callers:

1. **Rank-dependent values enter compiled graphs as runtime tensors**, never as
   Python constants; the program around an SP collective must be structurally
   identical on every rank. Document this in `collective_matmul.py`.
2. Sequence padding stays **outside** the compiled graph (as LTX-2 now does).
3. A Pallas kernel is an opaque custom call, so XLA cannot fold neighbouring ops
   into it — this removes the failure class above, but not rule 1 (the kernel's
   call order must still match across ranks).

## Design

### AG + matmul

```
in : x_local (B, S/tp, K); W_i (N_i/tp, K), i = 1..n
out: (B, S, N_i/tp)
split x_local into two halves; ring over the 2x2 snake, both directions
for hop in 0..tp-1:
    start remote copy: top half -> cw neighbour, bottom half -> ccw neighbour
    matmul the halves held now with every W_i, tiles [tm x tk] x [tk x tn] in VMEM
    store each result block at its owner's rows (shard_of_pos table, as today)
    wait for this hop's DMA (semaphores)
```

Double-buffered send/compute buffers. n weights share one DMA (ag2/3/4).

### matmul + RS

```
in : x (B, S, K/tp); W (N, K/tp); out: (B, S/tp, N)
for hop in 0..tp-1:
    partial = x[chunk for the rank `hop` steps away] @ W^T
    acc = received accumulator + partial      (VMEM if it fits, else HBM double buffer)
    start remote copy acc -> next rank
final acc = this rank's rows; add bias once
```

`rs2` sums two (x, W) pairs into the same accumulator.

### Shapes

| dimension | range seen today |
|---|---|
| rows per rank | 578 (LTX-2), 1170 (Wan), 1536, 1024 + 128 (FLUX streams) — often not a tile multiple |
| K, N | 1024 – 16384, multiples of 128 |
| weights per AG | 1 – 4 |
| dtype | bf16 (int8 later) |
| topology | v5e-4, 2x2, tp = 4 |

Ragged rows: mask the last tile in-kernel, or require callers to pad (eagerly)
to a multiple of `16 * 2 * tp`. Start with the latter; it keeps the kernel
simpler and matches rule 2.

### Switch

`DIFFLET_TPU_SP_KERNEL=pallas|jax` (default `jax` until step 5 passes).
Anything the kernel does not support (tp != 4, unsupported arity or shape)
falls back to the JAX ring with a one-line notice.

## Step 1 result (2026-11-10): the cost is the AG ring's write-back, not the wait

Rank-0 traces of the natural, warm, compiled denoise (main's bench with
`DIFFLET_BENCH_TRACE_DIR`), split by `benchmark/tpu_profile/exposed_comm.py`
(reads the xplane directly when the JSON export is truncated):

| 20 steps | Wan TP | Wan SP | LTX-2 TP | LTX-2 SP |
|---|---|---|---|---|
| window | 7200.7 ms | 5837.2 ms (-19%) | 4339.8 ms | 3793.5 ms (-12.6%) |
| ICI wait | 1978.6 (27.5%) | 192.9 (3.3%) | 1275.5 (29.4%) | 258.5 (6.8%) |
| compute | 4987.8 | 5398.9 (+411) | 2585.4 | 2993.5 (+408) |
| of which `dynamic-update-slice` | -- | 643.0 (11.0%) | -- | 264.7 (7.0%) |
| idle | 3.1% | 3.6% | 10.5% | 13.2% |

- The rings already hide 80-90% of the all-reduce (the op-level probe's 55%
  understated it); what is left waiting is 3-7% of a step.
- SP adds ~410 ms of compute per 20 steps, mostly the AG ring writing each
  hop's product into the full output with `dynamic_update_slice` (tp=4: eight
  writes per AG). The matmuls themselves get *faster* under SP (LTX-2: 82% ->
  90% of peak).
- LTX-2 also spends 176 ms in small all-reduces (global qk RMS norm, audio TP)
  and idles 13% between steps -- separate, cheaper targets.

Revised order: **P0** -- assemble the AG output once (collect the hops,
concatenate in ring-relative order, one reorder by the runtime ring position)
instead of eight `dynamic_update_slice`s; same math, so outputs must be
bit-identical; measure with `sp_pair_probe.py` and a Wan / LTX-2 bench A/B. If
P0 removes most of the write-back, the kernel's remaining target is the 3-7%
wait and the kernel is likely not worth it; otherwise continue with the Pallas
AG kernel below, which writes each tile of the product in place.

## P0 result (2026-11-10): concat helps Wan, not LTX-2, and is not output-equal

`DIFFLET_TPU_SP_AG_ASSEMBLE=concat` (default `dus`). Same tree (md5 77c0c1ad4ed5),
back to back, main's bench, rank-0 traces:

| | Wan dus | Wan concat | LTX-2 dus | LTX-2 concat |
|---|---|---|---|---|
| step | 292.6 ms | 275.1 ms (-6.0%) | 189.7 ms | 194.2 ms (+2.4%) |
| window (20 steps) | 5848.5 | 5468.8 | 3814.4 | 3806.6 |
| compute | 5399.0 | 5022.4 | 2993.6 | 2870.8 |
| ICI wait | 192.9 | 152.5 | 258.6 | 390.1 |

- `dynamic-update-slice` is gone; the assembly costs instead ~480 ms (Wan:
  copy +130, reshape +105, a `while` from the runtime-indexed `take` +106,
  fusion +110) -- net -380 ms.
- LTX-2 (289-row halves) is ICI-bound: the compute saved (-123 ms) became wait
  (+131 ms). Less HBM traffic alone does not help it.
- Op parity at Wan shapes (1170 rows/rank, hidden 5120): ag1/2/3 vs plain
  gather + linear are 0.0 in **both** modes.
- End to end the outputs differ: Wan dus vs concat 29.06 dB, while TP vs dus is
  43.87 dB and TP vs concat 29.43 dB (sha256 TP 8e646ad9b22e, dus e41edd1e23ab,
  concat 495847095139, each deterministic over 3 runs). LTX-2 dus vs concat
  43.75 dB (ef75047e8a34 / ec137f0d0fec). The Wan difference follows the
  content (largest on the subject), with no band at shard boundaries -- a
  trajectory drift ~5x the TP/SP noise, cause not found (a one-step forward
  comparison is the next check). Not checked: Qwen, FLUX, HunyuanVideo.

Decision: keep `dus` as default; continue with the Pallas AG kernel, which
writes each product tile once at its final rows (removing the remaining
assembly cost on Wan). For LTX-2 the kernel's value is overlap control, not
HBM traffic.

## Kernel result (2026-11-11): Wan -8.3% per step, LTX-2 flat, outputs within the TP/SP spread

`difflet/backends/tpu/ops_impl/collective_matmul_pallas.py`, behind
`DIFFLET_TPU_SP_KERNEL=pallas` (default `jax`), AG only. What got it there,
op-level (plain JAX, v5e-4, `benchmark/tpu_profile/cm_pallas_bench.py`, ms per call):

| version | Wan q/k/v | Wan FFN up | LTX-2 q/k/v | LTX-2 FFN up |
|---|---|---|---|---|
| JAX ring (dus) | 1.259 | 1.050 | 0.445 | 0.500 |
| v1: emit_pipeline per (hop, weight, half) | 1.541 | 1.181 | 0.606 | 0.630 |
| v3: halves resident in VMEM, VMEM->VMEM remote DMA | 1.241 | 1.074 | 0.464 | 0.540 |
| v4: one weight-tile stream over every (hop, weight, tile), runtime loop | **1.185** | **1.056** | **0.406** | **0.517** |

- Ablation: communication is fully hidden from v3 on (kernel == kernel without
  remote copies, within 0.001 ms); what is left is the matmul.
- Dead ends: pre-transposed weights (no change); wider column tiles (worse:
  each pipeline's first tile load is exposed); padding the halves to 16 rows and
  one (2*hrp)-row dot per tile ("v5": +2% on Wan q/k/v, nothing elsewhere, and
  Mosaic only slices tiled dims in whole tiles, so it needs an XLA pad first).
- A plain Pallas matmul equals XLA at the same shape
  (`pallas_matmul_sweep.py`); the kernel's gap to the one-chip XLA einsum is the
  small M (585 / 289 rows per dot), not the kernel.
- Fully unrolled, the stream took 225 s to compile at Wan's FFN shape; a
  `fori_loop` over a hop's tiles: 5-10 s.
- `has_side_effects=True` breaks torch_tpu (SPMD partitioner: "Side-effect HLO
  must have sharding"); the kernel does not set it.

Through torch_tpu (`cm_ops_parity.py`, `DIFFLET_TPU_SP_KERNEL=pallas`, Wan and
LTX-2 shapes, all four ranks): ag1/ag2/ag3 vs plain gather + linear 0.0,
including two calls in one graph. No fallback.

Full models, main's bench, same tree (md5 011bdf616420), back to back:

| | Wan jax | Wan pallas | LTX-2 jax | LTX-2 pallas |
|---|---|---|---|---|
| step | 292.4 ms | **268.0 ms (-8.3%)** | 190.1 ms | 190.5 ms |
| window (20 steps) | 5832.4 | 5350.9 | 3798.4 | 3735.3 |
| `dynamic-update-slice` | 643.1 | 0 | 264.7 | 0 |
| ICI wait | 192.9 | 85.5 | 258.5 | 111.8 |

- Wan pays ~250 ms of relayout copies/reshapes instead: 585-row halves are not
  a multiple of the (8, 128) HBM tile, so `(B, n, 2, 585, N) -> (B, 4680, N)` and
  the input split are real copies. Removing them needs rows per rank padded to a
  multiple of 16 outside the graph (as LTX-2 pads), i.e. a model change.
- LTX-2 is ICI-bound: the compute saved becomes wait, as with concat.

Output (Wan, `sp_step_drift.py`, PSNR of the decoded frames):

| | TP vs jax | jax vs pallas | TP vs pallas |
|---|---|---|---|
| 1 step, seed 42 | 49.88 | 50.47 | 49.88 |
| 4 steps, seed 42 | 49.04 | 49.47 | 49.40 |
| 20 steps, seed 42 | 43.87 | 26.09 | 26.23 |
| 20 steps, seed 7 | 38.87 | 45.85 | 38.38 |
| 20 steps, seed 1234 | 43.46 | 41.14 | 44.75 |

Per step the three agree at the bf16 noise level; after 20 steps which pair
drifts apart depends on the seed (each configuration is deterministic over
three runs). Read: the kernel is another valid rounding, like TP vs SP. Not
checked: more seeds, a seed-independent quality metric (fp32 reference,
perceptual), LTX-2's seed spread, a visual review of the seed-42 output, Qwen /
FLUX / HunyuanVideo. Open risk: consecutive kernels share the barrier semaphore
(`collective_id=0`); no failure seen, not proven safe.

### The other models (2026-11-11, same tree md5 011bdf616420, main's bench)

| model | step: JAX ring -> kernel | PSNR jax vs kernel | PSNR TP vs jax (floor) | note |
|---|---|---|---|---|
| Qwen-Image | 178.0 -> 179.7 ms | inf (bit-identical) | 47.03 | no `dynamic-update-slice` in the jax trace; ICI -140 ms, compute +189 ms |
| FLUX | 156.1 -> 157.3 ms | 43.90 | 41.52 | single blocks' ag4 (768x3 + 3072) fell back to the ring; dual blocks: dus -253 ms, kernel +545 vs fusion -186 |
| HunyuanVideo | 517.8 -> 518.1 ms | 42.52 | 45.55 | the step is bound by the dense single blocks' all-reduce (1.0 s / 20 steps) |

The kernel pays only where the ring's `dynamic-update-slice` is expensive (Wan);
where XLA already fuses the ring's writes (Qwen-Image) or the dus is small,
its matmul is a little slower than XLA's einsum and it loses 1-2%.

### HunyuanVideo single-stream blocks under SP (`DIFFLET_TPU_SP_SINGLE=1`)

`backends/tpu/hunyuan_video/sp.py`: the 40 single-stream blocks run on a
contiguous chunk of the joint [latent | text] sequence (10496 = 4 x 2624; a chunk
may straddle the boundary), ag4 (q|k|v|mlp) + rs2 as FLUX's single blocks; the
first block cuts, the last gathers. Same tree (md5 3d32d564e9bd), back to back:

| | dual-only SP | dual + single SP |
|---|---|---|
| step | 516.1 ms | **502.2 ms (-2.7%; -8.7% vs TP 549.9)** |
| window (20 steps) | 10322.1 | 10003.7 |
| ICI wait | 1042.5 (all-reduce 1025.7) | **141.6** |
| `dynamic-update-slice` | -- | 646.3 (the ag4 rings) |
| PSNR vs TP | 45.55 | 44.35 (dual-only vs single: 45.68) |

First attempt hung in the first forward: the cut was a Python-constant slice of
the in-graph concatenation; ranks 0-2's chunks lie inside the latent, rank 3's
holds the text, so XLA folded the concatenation on 0-2 only and the programs
diverged (rule 1 again). The cut is now `index_select` with a per-rank row
buffer.

The ag4's dus, next (all single-stream SP; tree d9e05f83d1aa for jax / pallas,
the kernel now takes weights of different widths through one common tile,
q|k|v 768 + mlp 3072 -> 384):

| HunyuanVideo | step | window | `dynamic-update-slice` | PSNR vs TP |
|---|---|---|---|---|
| TP | 549.9 ms | | | |
| dual-only SP | 516.1 | 10322.1 | -- | 45.55 |
| + single SP, JAX ring | 500.7 | 10023.6 | 646.3 | 44.35 |
| + single SP, concat assembly | 493.9 | 9846.1 | 0 (copy +142, while +224) | 42.86 |
| + single SP, **Pallas kernel** | **478.0 (-13.1% vs TP)** | **9525.4** | 0 | 42.86 |

The kernel's video is bit-identical to the concat assembly's (sha256
f13f00fb7aba both); the JAX ring's is reproducible across trees (03e10d9b7119).
Each configuration is deterministic over three runs.

### FLUX with the mixed-width kernel (single blocks' ag4 on the kernel)

A first rerun looked identical to the old one -- torch_tpu's compilation cache
keys a `jax_op` by name and signature, so the unchanged op name reused the
ring program (num_compilations=0, 18.8 s warm-up, same sha256). Op names now
carry a hash of the two source files. Fresh compile (78 s, num_compilations=1,
no fallback), tree 077ab253f18d:

| FLUX (28 steps) | TP | SP, JAX ring | SP, Pallas kernel |
|---|---|---|---|
| step | 175.2 ms | 156.5 | **140.6 (-10.2%; -19.7% vs TP)** |
| window | | 4329.5 | 3890.5 |
| `dynamic-update-slice` | | 1171.1 | 0 |
| PSNR vs TP | | 41.52 | 39.52 (jax vs kernel 44.91) |

Not checked: other seeds, a visual review (TP vs kernel max |d| 199/255).

## 2026-11-11 afternoon: cleanup, the RS kernel, Wan padding and RoPE, a GELU epilogue

All on v5e-4 with main's `bench.py`, each pair back to back on one node (granted
16:16 UTC, released 20:19 UTC); traces compared with
`benchmark/tpu_profile/trace_diff.py` (per op kind, or per opcode and output
shape with `--by-shape`).

**AG kernel cleanup.** Only the streamed variant is kept, with full names
(`all_gather_matmul_body`, `unsupported_reason`, `ring_tables`). Against 868401eb:
Wan / FLUX / HunyuanVideo outputs sha256-identical, steps within noise. The
version table above used (K, N) weights and column tiles up to 512; the models
run (N, K) and up to 256 (Wan FFN up then has 128-wide tiles): op-level 1.187 /
1.066 / 0.411 / 0.519 ms (Wan q/k/v, FFN up, LTX-2 q/k/v, FFN up), bit-exact.

**RS kernel** (`DIFFLET_TPU_SP_RS_KERNEL=pallas`, off by default). Tile by tile:
multiply, add the upstream accumulator's tile in the epilogue, send it on;
accumulators VMEM to VMEM; three receive slots per direction. A first version
that added each hop's accumulator in a separate pass was slower everywhere.
Op-level, ms per call:

| shape | JAX ring | kernel | note |
|---|---|---|---|
| Wan attention out | 0.518 | 0.504 | ICI-bound (no compute 0.462) |
| LTX-2 attention out | 0.214 | 0.212 | ICI-bound |
| Wan FFN down, 1170 rows | 0.928 | 1.040 | ~75 us input relayout (rows not a multiple of 8) |
| Wan FFN down, 1168 rows | 0.917 | 0.964 | compute-bound; column tile 512: 0.945 |
| FLUX single pair | 0.627 | 0.663 | compute-bound; column tile 512: 0.649 |

Wan end to end (padded) 262.2 -> 262.7 ms. Unlike the AG ring, the RS ring has no
large assembly cost to remove, and where it is compute-bound the kernel's matmul
is ~5% slower than XLA's einsum. In the model one step differs from the ring by
50.4 dB, four by 48.9 dB (seed 42), as the AG kernel did.

**Wan row padding** (`DIFFLET_TPU_WAN_SP_PAD`, default on): 4680 -> 4736 rows, 1184
per rank, 592-row halves. Step 269.2 -> 262.2 ms, output vs unpadded 44.7 dB.
Trace (20 steps): the rings' input-split and output-assembly relayouts, 137.8 ms,
are gone; cutting the pad keys (4680 is not a tile multiple: copy + slice ~33 ms),
1.2% more data per hop (+13 ms ICI wait) and the final gather over the hidden
size (+8 ms) take ~54 back. The "~250 ms relayout" estimated above counted
every reshape / copy; most of them are attention and RoPE layout changes that
do not depend on the row count.

**Wan split-halves RoPE** (`DIFFLET_TPU_WAN_SPLIT_ROPE`, default on), as FLUX's:
q/k rows and RMSNorm weights permuted per head once, rotation as two halves.
The interleave's reshapes / fusions / copies and, unexpectedly, the 66 ms
relayout of the AG kernel's q/k/v outputs are gone. Permuting the weights on the
device with `copy_` left them in a layout every forward copied back
(`copy bf16[1280,5120]`, 80 per step, 50 ms, plus ~120 ms more device idle);
permuted on the host, step 262.2 -> 249.2 ms. Output vs TP 42.2 dB.

**Wan, the day's net:** 269.2 -> 249.2 ms per step (-7.4%); TP is 360.7.

**Bias + GELU in the AG kernel's epilogue (tried, reverted).** For Wan's FFN
up-projection it removes the separate 60 ms `[1,4736,3456]` GELU fusion, but the
FFN-down RS ring's matmuls then read `h` 10% slower (113 -> 125 us per op,
+65 ms): XLA's memory-space assignment had placed the GELU fusion's output in
`S(1)` (alternate memory, presumably VMEM), while a Pallas output (`pl.ANY`)
lives in plain HBM. Step 250.4 -> 249.4 ms, within noise. A Pallas kernel's
outputs forfeit that placement for their consumers; whether the AG kernel's
q/k/v outputs pay this in attention is not checked.

Not checked: other seeds, FLUX's RoPE permutation (it uses `copy_` on the device
too, so may pay the same per-forward weight copy), padding for the other models.

### Reproducing the afternoon's numbers

Environment: v5e-4 (us-west1-c), torch 2.14.1+cpu, torch_tpu 0.2.0, libtpu
0.0.51, jax 0.12.0.1; Wan 2.1 14B T2V at 480x832x9, 20 steps, seed 42 (main's
`benchmark/models.py` entry `wan_2_1`). Each table row is one
`benchmark/tpu_profile/bench_case.sh` case; a pair differs only in its
environment and runs back to back on the same node and tree:

    W="DIFFLET_TPU_COMPILE=1 DIFFLET_TPU_DEVICE_VAE=1 DIFFLET_TPU_SP=1"
    bench_case.sh wp-pad0 wan_2_1 $W DIFFLET_TPU_WAN_SP_PAD=0 DIFFLET_TPU_WAN_SPLIT_ROPE=0
    bench_case.sh wp-pad1 wan_2_1 $W DIFFLET_TPU_WAN_SP_PAD=1 DIFFLET_TPU_WAN_SPLIT_ROPE=0
    bench_case.sh wr-rope1 wan_2_1 $W                       # both defaults on
    bench_case.sh wp-rs wan_2_1 $W DIFFLET_TPU_SP_RS_KERNEL=pallas
    # + DIFFLET_BENCH_TRACE_DIR=<dir> for the trace rows, then
    python -m benchmark.tpu_profile.trace_diff <dir_a> <dir_b> [--by-shape --only copy,reshape]
    python -m benchmark.tpu_profile.image_psnr <a>/run1_synced_frames_0_4_8.png <b>/...
    python -m benchmark.tpu_profile.cm_rs_pallas_bench --ablate --column-tile-limits 128,256,512,1024
    python -m benchmark.tpu_profile.sp_step_drift --model wan_2_1 --steps 1,4 --out <dir> --tag <t>

The 18:42 UTC measurement of "both on" predates the commit only by comments
(the source hash in the op names differs, so the committed tree compiles anew;
not re-measured). Run-to-run noise of one configuration on one node: Wan
267.7-269.2 ms (four runs), FLUX 141.9-144.7 ms with identical outputs.

Not verified: the "vs TP" PSNRs use a TP run from an earlier tree
(`rp-wan-base`), not one on this tree; split RoPE's 26 dB against the
interleaved layout after 20 steps is attributed to the RMSNorm's summation order
by analogy (the RS kernel's 1- and 4-step check), not by its own 1-step run;
only seed 42; that `S(1)` is VMEM.

## Steps

| # | Work | Validation | Estimate |
|---|---|---|---|
| 1 | **Decision gate.** Profile every model's compiled SP step; measure the communication still exposed (`trace_breakdown.py`, `sp_pair_probe.py`). | Go if exposed comm ≥ 8% of a step on LTX-2 or Wan; stop if < 5%. | 0.5 day |
| 2 | Pallas AG (1 weight) and RS (1 pair) in **plain JAX**, one process owning 4 chips (harness like `cm_pure_jax.py`). | vs numpy fp32 reference; per-op time vs the JAX ring at Wan / LTX-2 / FLUX FFN shapes; overlap ratio. | 2–3 days |
| 3 | Through `jax_op` into torch_tpu behind the switch. | `cm_ops_parity.py`; `cm_padded_repro.py` length sweep and all pad modes; per-rank program fingerprints identical. | 1–2 days |
| 4 | AG with 2/3/4 weights, RS with 2 pairs, bias. | As step 3, plus `sp_pair_probe.py` per block. | 1–2 days |
| 5 | Per-model A/B with main's `bench.py`: Wan, Qwen-Image, HunyuanVideo, FLUX, LTX-2. Same commit, back to back, kernel vs JAX ring. | PSNR within the noise floor of compile-vs-eager; image sha256, versions and commit recorded (reproducibility rule). | 1–2 days |
| 6 (optional) | Fusions: gelu after the up-projection, RMSNorm prologue, group-wise int8 dequant. | FFN micro-benchmark; MiniMax-H3 int8 probe at group 128. | as needed |

Total for steps 1–5: about 1–1.5 weeks.

## Success criteria

- Per-op overlap ≥ 85% (today 55%).
- LTX-2 and Wan step time ≥ 5% below today's SP.
- Output of every model within its compile-vs-eager noise floor; no hang or core
  halt across the length sweep and pad modes.
- One environment variable reverts to the JAX ring.

## Risks

| Risk | Mitigation |
|---|---|
| Pallas TPU remote DMA / semaphore API on jax 0.12 | Step 2 in plain JAX first; start from the collective-matmul example in JAX's "Distributed computing in Pallas for TPUs" guide. |
| Deadlocks and out-of-bounds access are hard to debug | The 2026-11-10 tooling: minimal repro, per-rank HLO fingerprints, timer-stopped per-rank traces. |
| torch_tpu support for Pallas custom calls | The TPU flash attention is already a Pallas kernel on the same path. |
| Tied to v5e-4 (2x2, tp = 4) | Topology and tile sizes as parameters; other configs fall back explicitly. |
| Gain smaller than estimated | Step 1 measures it before any kernel work. |

## Cheaper alternative to try first

Split each shard into 4 or 8 chunks instead of 2 in the existing JAX ring, so
XLA has more hops to overlap. No kernel; a day of work; tells how much of the
gap is granularity rather than control.
