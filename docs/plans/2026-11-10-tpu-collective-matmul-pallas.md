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
