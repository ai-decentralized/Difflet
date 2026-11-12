# Wan self-attention q/k/v in MXFP8: the half-split rotary layout (phase H)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wan 2.1 / 2.2's default MXFP8 set quantizes the self-attention q/k/v too, without the
5x block slowdown that kept them bf16 (`917d087`), by giving the quantized q/k a rotary layout
the compiler moves in whole rows.

**Outcome (2026-11-08):** Task H6 (`90c648f`) reverted the default to the `917d087` set (the
cross-attention q/k/v and the FFN) after the two-expert Wan 2.2 rule failure (-1.44 dB / SSIM
-0.034 over 12 clips); the half-split layout serves `DIFFLET_QUANT_TARGETS` overrides that
quantize a self-attention q or k.

**Architecture:** When any self-attention `to_q` / `to_k` is an FP8 twin, the Wan transformer
application reorders every self-attention head's q/k output channels at load from
`(0, 1, 2, 3, ...)` to `(0, 2, 4, ..., 1, 3, 5, ...)` (weight rows, MX block scales, bias, and
the q/k RMSNorm weights), and the model switches its rotary to the half-split law: pair `i` is
`(x[i], x[i + d/2])`, the tables are `cat(c, c)` / `cat(-s, s)`, and the rotation is
`x * cos + cat(x[d/2:], x[:d/2]) * sin`. q and k get the same reordering, so `q . k` is the same
up to summation order, and nothing after the attention changes (v and the output projection are
untouched). bf16 keeps the interleaved law.

**Tech Stack:** PyTorch-native neuron backend (torch_neuronx 2.14), nkilib `matmul_mxfp8`,
Wan 2.1 / 2.2-A14B, trn3 tp4.

**Spec:** this file; evidence `docs/verification/2026-11-07-mxfp8-video-models.md` (the Wan
section, and a new section for this phase); the shared Decisions block of
`docs/plans/2026-11-07-mxfp8-proj-out-rope.md` (floor-relative screen, `J`, leave-one-out).

## Why (measured 2026-11-07/08, Wan 2.1, tp4, 480x832x9, 50 steps, seed 42)

A device profile of the tp4 block NEFF, replayed with its collectives (`neuron-explorer
capture`, 4 workers), put the whole slowdown in the rotary op
(`difflet/backends/neuron/ops_impl/embeddings.py:39`, `x * cos + (x @ P) * sin`). With q/k from
the MXFP8 kernel its DMAs move 2-byte elements (1480 descriptors, 87.8 ms per block); with q/k
from a bf16 matmul the same op moves 256-byte head rows (240 descriptors, 0.04 ms).

A scratch prototype of this plan (`/home/ubuntu/fp8-bench/wan21_tp4/replay/make_hs_scratch.py`),
same-session A/B, every NEFF cached:

| arm | step median | block (device) | vs bf16 | PSNR / SSIM vs bf16 | rerun bit-identical |
|---|---|---|---|---|---|
| bf16 (interleaved) | 431.9 ms (432.3 rerun) | 10.67 ms | 1.00x | floor 37.0 dB | **no** (37.2 dB apart) |
| bf16, half-split | 483.6 ms | 11.36 ms | 1.12x | 36.4 dB / 0.953 | yes |
| default `917d087` (attn2 + FFN) | 387.3 ms | 9.30 ms | 0.90x | 33.0 dB / 0.944 | yes |
| all q/k/v + FFN, interleaved | 2263 ms | 56.89 ms | 5.2x | - | yes |
| **all q/k/v + FFN, half-split** | **330.7 ms** | **8.05 ms** | **0.77x** | 34.2 dB / 0.946, cos 0.9988 | yes |

On the CPU the reordered half-split toy matches the interleaved one to 2.4e-7 / 3.0e-7 (max rel),
and the control (reordered weights, interleaved rotary) is 2-3e-2 off.

## Global Constraints

- Every device run: LNC=2, 50 steps, seed 42 (user, 2026-11-07).
- The half-split layout is used only when a self-attention `to_q` or `to_k` is quantized
  (bf16 is 12% slower with it); then every self-attention of that module uses it.
- No new backend op: the half-split rotation is plain tensor code in the Wan model module.
- K must be a multiple of 128 per rank (`mx._tile_sizes`); Wan tp8 stays refused.
- CPU fp32 (phase G) stays paused until the user asks (user, 2026-11-07). Phase G is now running
  (2026-11-08).
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; push every commit.

## Review Focus

1. **Wan 2.2's two experts:** each expert's application reorders its own weights; a pipeline
   with one expert quantized and one not must not mix layouts within one module.
2. **A partial override** (`DIFFLET_QUANT_TARGETS` quantizing `to_q` of block 0 only): every
   self-attention of the module still switches together, and the output still matches the
   dequantized reference.
3. **The dequant compute path** (`--quant-compute dequant`): same layout rule (the layout follows
   what is quantized, not the compute path), output matches the reference.
4. **SP / CP / ring / ulysses:** the reordering is within each head and the tables are
   per-position, so CP's sequence scatter of the tables and SP's gather are unchanged; the toy's
   tp2 / sp worker paths (`tests/unit/backends/_neuron_workers.py`) must still match.
5. **A second load** (`load()` twice, or the weight store reused by another application): the
   reorder must apply exactly once per loaded tensor.

---

### Task H1: the half-split rotary layout in the Wan model

**Files:**
- Modify: `difflet/models/wan/modeling_wan.py` (`WanRotaryPosEmbed`, `WanAttention.forward`,
  `WanTransformer3DModel`)
- Create: `tests/unit/models/test_wan_half_split_rope.py`

**Interfaces:**
- Produces, in `difflet/models/wan/modeling_wan.py`:
  - `ROPE_LAYOUTS = ("interleaved", "half_split")`
  - `half_split_index(n: int, head_dim: int) -> torch.Tensor` (int64; per head
    `cat(arange(0, hd, 2), arange(1, hd, 2))` offset by the head; `n % head_dim == 0` else
    `ValueError`)
  - `apply_rotary_emb_half_split(hidden_states, cos, sin) -> torch.Tensor`: `half = d // 2`,
    `(x * cos + cat([x[..., half:], x[..., :half]], -1) * sin).to(x.dtype)`
  - `WanRotaryPosEmbed`: two more non-persistent buffers `freqs_cos_half` (each axis table's
    `[:, 0::2]`, concatenated t|h|w) and `freqs_sin_half` (`[:, 1::2]`), sliced at `__init__` on
    the CPU; `forward(hidden_states, layout="interleaved")` returns the interleaved tables as
    today, or for `"half_split"` the base tables `B` (64 wide for head_dim 128, built by the
    same expand/cat as today on the half buffers) as `(cat(Bc, Bc), cat(-Bs, Bs))`, shape
    `(1, S, 1, head_dim)` like today.
  - `WanAttention.rope_layout: str` (default `"interleaved"`); self-attention rotates with
    `apply_rotary_emb_half_split` when it is `"half_split"`, else as today.
  - `WanTransformer3DModel.rope_layout: str` (default `"interleaved"`), passed to `self.rope`.
  - `WanTransformer3DModel.use_half_split_rope_() -> int`: reorders every `blocks[i].attn1`'s
    `to_q` / `to_k` rows (`weight`; `weight_scale` when the layer has one; `bias` when present)
    and `norm_q` / `norm_k` weights with `half_split_index`, sets every `attn1.rope_layout` and
    the model's to `"half_split"`, returns the number of attentions reordered. A second call is
    a no-op returning 0 (the model's `rope_layout` is already `"half_split"`).
    Tensors are replaced with `index_select` on their own device (`.data =` for parameters,
    attribute assignment for the `weight_scale` buffer), never edited in place.

- [x] **Step 1: Write the failing tests** in `tests/unit/models/test_wan_half_split_rope.py`:
  - `test_half_split_index_reorders_within_each_head`:
    `half_split_index(8, 4).tolist() == [0, 2, 1, 3, 4, 6, 5, 7]`; `half_split_index(6, 4)`
    raises `ValueError`.
  - `test_half_split_tables_are_the_interleaved_tables_reordered`: for a `WanRotaryPosEmbed`
    with head_dim 128 (Wan's t/h/w split 44/42/42) and a `(1, 16, 3, 8, 12)` input,
    `cos_h == cos_il[..., perm]` and `sin_h == sign * sin_il[..., perm]` exactly, with
    `perm = half_split_index(128, 128)` and `sign` = -1 on the first 64 entries, +1 on the rest;
    also for head_dim 32 (the FP8 toy's 12/10/10).
  - `test_half_split_rotation_is_the_interleaved_rotation_reordered`: random fp32
    `x (1, 96, 4, 128)`; `apply_rotary_emb_half_split(x[..., perm], cos_h, sin_h)` equals
    `difflet.ops.apply_rotary_emb(x, cos_il, sin_il)[..., perm]` within `atol=1e-6, rtol=0`.
  - `test_reordered_model_matches_the_interleaved_model`: inside `tp1_mesh()`, for the W1 toy
    and `TINY_FP8_WAN` (`tests/unit/backends/_neuron_wan_toy.py`): the output of a deep copy
    after `use_half_split_rope_()` (returns `num_layers`) matches the original's within max rel
    error `1e-5`; the same copy with `rope_layout` forced back to `"interleaved"` on the model
    and every `attn1` differs by more than `1e-3` (the control); a second
    `use_half_split_rope_()` returns 0 and leaves the output unchanged.
- [x] **Step 2: Run** `pytest tests/unit/models/test_wan_half_split_rope.py -v`: fail on the
  missing names.
- [x] **Step 3: Implement** the interfaces above.
- [x] **Step 4: Run** `pytest tests/unit/models/test_wan_half_split_rope.py tests/unit/backends/test_neuron_wan_fp8.py tests/unit/backends/test_neuron_rotary.py -q` (and every Wan test under `tests/unit/backends` and `tests/unit/models`): pass.
- [x] **Step 5: Commit** `feat(wan): the half-split rotary layout — self-attention q/k reordered per head (0,2,..,1,3,..) with their norms, tables cat(c,c)/cat(-s,s), rotation on contiguous halves; exact to fp32 summation order on the toys`.

**Done** (`73f1722`).

### Task H2: the Wan application uses it for quantized q/k, and the default set quantizes them (reverted by Task H6)

**Files:**
- Modify: `difflet/backends/neuron/wan/transformer.py` (`NeuronWanTransformerApplication`)
- Modify: `difflet/quant/targets.py` (`WAN_TARGETS`, `WAN_KEPT_BF16`, the Wan comment)
- Modify: `tests/unit/backends/test_neuron_wan_fp8.py`, `tests/unit/quant/test_targets_wan.py`
- Modify: `difflet/cli/quant.py` only if its `--quant` help names Wan's self-attention as bf16

**Interfaces:**
- Consumes: H1's `WanTransformer3DModel.use_half_split_rope_()` and `rope_layout`.
- Produces: `NeuronWanTransformerApplication._after_load(module)`: if any `blocks[i].attn1.to_q`
  or `.to_k` is an `_FP8LinearBase` (`difflet/backends/neuron/ops_impl/fp8_linear.py`), call
  `module.use_half_split_rope_()` and log
  `[wan] self-attention q/k reordered for the half-split rotary (<n> attentions): their q/k come from the MXFP8 kernel`
  once; then `super()._after_load(module)` (which packs the fp8 weights, so the reorder
  happens before the packing).
- `WAN_TARGETS` becomes, in this order (the order of the pre-`917d087` default, so the
  directory `mxfp8-e17b42dc8e5dc336` and its NEFF-free weights are reused):
  `attn1.(to_q|to_k|to_v)`, `attn2.(to_q|to_k|to_v)`, `ffn.net.0.proj`, `ffn.net.2`.
  `WAN_KEPT_BF16` loses the `attn1` q/k/v pattern; the comment above `WAN_TARGETS` replaces the
  917d087 measurement with this plan's table (the 330.7 / 387.3 / 431.9 ms row set and the
  profile's two DMA numbers), and says the layout rule.

- [x] **Step 1: Update the tests.**
  - `tests/unit/quant/test_targets_wan.py`: the default-target test's name and sets include the
    self-attention q/k/v (rename it `test_default_targets_cover_every_attention_qkv_and_the_ffn`);
    the kept-bf16 test no longer lists `attn1.to_q`.
  - `tests/unit/backends/test_neuron_wan_fp8.py`: `TARGETS_PER_BLOCK = 8`; `MODULE_TARGETS` and
    `CHECKPOINT_TARGETS` gain `attn1.to_q`, `attn1.to_k`, `attn1.to_v`;
    `test_application_quantizes_through_the_checkpoint_key` asserts `block.attn1.to_q` is an
    `FP8ColumnParallelLinear`, `app.module.rope_layout == "half_split"`, every
    `blocks[i].attn1.rope_layout == "half_split"`, and keeps its `assert_close` against the
    dequantized interleaved reference (`atol=1e-4, rtol=1e-4`).
  - New `test_a_set_without_self_attention_qk_keeps_the_interleaved_rotary`: with the
    `917d087` set as `DIFFLET_QUANT_TARGETS` override (`attn2` q/k/v + FFN), `rope_layout` stays
    `"interleaved"` on the module and every `attn1`, and the output matches the reference.
  - New `test_quantizing_one_block_qk_switches_every_self_attention`: override
    `blocks\.0\.attn1\.to_q`; every `attn1.rope_layout == "half_split"`; output matches the
    reference (Review Focus 2).
  - New `test_the_dequant_path_uses_the_same_layout`: `compute="dequant"` with the default set:
    `rope_layout == "half_split"`, output matches the reference (Review Focus 3).
  - The tp2 / sp worker cases in this file (if any) are unchanged and must still pass with the
    new default; if the file has none, add one tp2 case through the existing toy worker
    harness (`tests/unit/backends/_neuron_workers.py`) asserting the reference match.
- [x] **Step 2: Run** `pytest tests/unit/quant/test_targets_wan.py tests/unit/backends/test_neuron_wan_fp8.py -v`: the new and changed assertions fail.
- [x] **Step 3: Implement.**
- [x] **Step 4: Run** `pytest tests/unit/quant tests/unit/backends/test_neuron_wan_fp8.py tests/unit/models tests/unit/cli -q` plus every Wan test under `tests/unit/backends`: pass.
- [x] **Step 5: Commit** `feat(quant): Wan's default MXFP8 set quantizes the self-attention q/k/v again, in the half-split rotary layout — 330.7 vs 431.9 ms per step at tp4 480x832x9 (0.77x; 387.3 without them), 8.05 vs 9.30 ms per block`.

**Done** (`f0ad1c6`, review fix `65b4ec5`; the TPU sequence-parallel override follows the layout
in `1a8382f`). CPU gate at `65b4ec5`: 2058 passed, 111 skipped, 0 failed.

### Task H3 (device): Wan 2.1 and 2.2 with the new default

Run on a free device, `NEURON_RT_VISIBLE_CORES` pinned; source a snapshot of the commit;
cache `/home/ubuntu/difflet-cache-video`; prompt `"a cat walking through a garden"` (the Wan
section's), 480x832.

**Done 2026-11-08** (evidence section 8, from a snapshot of `1a8382f`). New default (H2's set,
reverted by Task H6 after step 3's two-expert rerun): 0.77x at
480x832x9 (331.6 / 336.4 against 433.4 / 433.9 ms for Wan 2.1, 331.3 / 334.2 against 432.8 /
432.9 for Wan 2.2) and 0.94x at 81 frames, bit-identical across its two generates. Deviations:
- Step 1's block device time is the prototype's capture (8.05 ms); the committed default
  reproduces the prototype's clip bit for bit (`da16a5d3…`), and no new capture ran.
- Step 3 first ran without `--dual-expert` (`wan_h3/h3.sh`), so Wan 2.2 ran its high-noise
  expert for every step: one MXFP8 directory and one reorder line per process, 64 compiled
  blocks. It was rerun with both experts (`wan_h3/wan22d`, `wan_h3/multi/wan22d`; evidence 8i):
  the reorder line twice, 0.76x at 9 frames and 0.93x at 81, and the 12-clip mean 1.44 dB /
  SSIM 0.034 below the old default's, so the rule fails and Task H6 reverted the default.
- Step 5: on single clips the rule fails two cells (Wan 2.1 at 81 frames, SSIM -0.022; Wan 2.2
  at 9 frames, -1.28 / -1.20 dB) while the other two gain 2.06 and 5.56 dB, against a bf16
  run-to-run distance of 36.0 / 36.4 dB: noise-dominated. As FLUX's D5, the rule was applied to
  the mean over 4 prompts x 3 seeds at 9 frames, one batch per arm: -0.46 dB / -0.003 (Wan 2.1)
  and -0.41 dB / -0.003 (Wan 2.2, one expert) against the old default; it passed, and H2's
  `targets.py` stayed until the two-expert rerun: -1.44 dB / -0.034, the rule fails, H6.
- Step 4: the tip's bf16 pair differs and the CPU-op pair is bit-identical (`450d04ef…`, the
  pre-campaign clip), at 438.3 / 439.9 against 433.3 / 433.1 ms: a finding of `fcb0dca`,
  reported to the user, who chose reproducibility (Task H5).

- [x] (partial: no new block capture ran, see Done) **Step 1: Wan 2.1 at 9 frames:** in one session, `bf16` (gen50a, gen50b), the new default
  (compile, gen50a, gen50b). Record step medians, md5s (mp4 and latents), `compare` against
  bf16 gen50a, and the block device time (dump + `capture`, the recipe in the evidence doc).
  Expect about 331 ms and 34 dB; the default must be bit-identical across its two generates.
- [x] **Step 2: Wan 2.1 at 81 frames:** bf16 and the new default, gen50a each; ratio and
  `compare` (the previous default was 0.98x and 28.8 dB against a 37.2 dB floor).
- [x] **Step 3: Wan 2.2-A14B at 9 and 81 frames:** as steps 1-2 (both experts quantized; the
  log must show the reorder line for both experts). Done with both experts (`run_wan22d.sh`,
  evidence 8i): the reorder line twice; bf16 432.7 / 432.2, old default 390.2 / 389.8, new
  default 329.0 / 330.8 ms at 9 frames, 5079.7 / 4984.5 / 4730.3 ms at 81; the first run had
  omitted `--dual-expert` (`wan_h3/h3.sh`, one expert, 8e).
- [x] **Step 4: bf16 reproducibility.** The interleaved bf16 runs at the tip are not
  bit-identical run to run (three runs, three md5s, 37.2 dB apart), while the pre-`fcb0dca`
  bf16 pair was. Run two bf16 generates from a scratch copy whose neuron
  `apply_rotary_emb` is the CPU op (the pre-`fcb0dca` law) and two from the tip. If the CPU-op
  pair is bit-identical and the tip pair is not, record it as a finding of `fcb0dca` for the
  user (bf16 cost of the CPU op: 437.6 vs 431.8 ms); do not change bf16 in this phase.
- [x] **Step 5: Rule.** The new default ships when at 9 frames it is faster than the `917d087`
  default by more than `J` and, at 9 and 81 frames for both models, its PSNR against bf16 is not
  below the `917d087` default's by more than 1 dB and its SSIM not by more than 0.01. Otherwise
  revert H2's `targets.py` change (keep H1 and the application rule: an override still gets the
  fast layout) and record why.

### Task H4: evidence and docs

- [x] Evidence doc: a new section "Wan self-attention q/k/v: the rotary layout" with the
  capture recipe (dump with `neuron-explorer inspect`, replay with `capture
  --collectives-worker-count=4 --collectives-workers-per-node=4`, `view --output-format
  summary-json` / `parquet`), the profile tables, H3's tables and the rule's outcome; correct
  the Wan section's statement that the cause was TP-specific / unknown.
- [x] README and `docs/cli-staged-commands.md`: Wan's MXFP8 lines (set, ratio at 9 and 81
  frames).
- [x] Tick this plan; commit `docs: Wan in MXFP8 with the half-split rotary — <ratios>`; push;
  update the status page.

**Done 2026-11-08** (evidence section 8; section 4's TP-specific / profile-not-obtained
statements carry pointers to it, unrewritten). The status page is updated outside this plan's
commits.

### Task H5: Wan's interleaved rotary goes back to the stride-2 law on the neuron backend (user, 2026-11-08)

H3 step 4: with the neuron backend's `apply_rotary_emb` (`x @ P`, `fcb0dca`) bf16 Wan is not
bit-reproducible run to run (433.3 / 433.1 ms, two different clips); with the CPU backend's
stride-2 law the pair is bit-identical and equals the pre-campaign clip `450d04ef` (438.3 /
439.9 ms). The user chose reproducibility. Scope: Wan only. HunyuanVideo also calls
`difflet.ops.apply_rotary_emb` and its bf16 pairs are reproducible, and every HunyuanVideo number
was measured with the `x @ P` op, so the neuron op itself stays.

**Files:**
- Modify: `difflet/ops/embeddings.py`, `difflet/ops/__init__.py` (if it lists the ops),
  `difflet/backends/neuron/ops_impl/embeddings.py`, `difflet/models/wan/modeling_wan.py`
- Test: `tests/unit/models/test_wan_half_split_rope.py` (or a new
  `tests/unit/backends/test_neuron_rotary.py` case for the neuron op)

**Interfaces:**
- `difflet.ops.apply_rotary_emb_slices(hidden_states, freqs_cos, freqs_sin)`: the interleaved
  law computed with stride-2 slices. Dispatched like the other ops, with one difference: a
  backend that does not define `apply_rotary_emb_slices` gets its own `apply_rotary_emb`
  (so cpu, cuda, tpu and trainium behave exactly as today without a change).
- The neuron backend's `ops_impl/embeddings.py` defines `apply_rotary_emb_slices` as the CPU
  backend's `apply_rotary_emb` (re-exported, the same function object), and keeps its own
  `apply_rotary_emb` (`x @ P`) unchanged for HunyuanVideo.
- `WanAttention.forward`: the interleaved self-attention rotation calls
  `difflet.ops.apply_rotary_emb_slices`; the half-split path is unchanged.

**Done 2026-11-08** (`c7c2346`; evidence 8h). CPU gate at `c7c2346`: 2103 passed, 111 skipped,
0 failed; the full CPU suite there: 5046 passed, 266 skipped, 1 failed
(`test_neuron_hv1_app::test_registry_builds_the_neuron_application`, pre-existing on `main`).
Step 6 on the device (`wan_h3/h5/`, a snapshot of `c7c2346`): bf16 440.0 / 439.6 ms, mp4
`450d04ef…` on both runs (the pre-campaign clip) and latents `bf61d6d7…` on both; the then
default's (H2's set, half-split) clip `da16a5d3…` unchanged at 330.3 ms. Cost: 440 against 433 ms
per bf16 step (1.6%).

- [x] **Step 1: Write the failing tests.**
  - `test_slices_op_falls_back_to_the_backends_rotary`: on the CPU backend,
    `difflet.ops.apply_rotary_emb_slices` returns exactly (`torch.equal`) what
    `difflet.ops.apply_rotary_emb` returns, for random bf16 and fp32 `(1, 24, 4, 128)` inputs
    and Wan-shaped interleaved tables.
  - `test_the_neuron_backend_slices_op_is_the_cpu_law`:
    `difflet.backends.neuron.ops_impl.embeddings.apply_rotary_emb_slices is
    difflet.backends.cpu.ops_impl.embeddings.apply_rotary_emb`.
  - `test_wan_interleaved_self_attention_uses_the_slices_op`: on the W1 toy (tp1 mesh), a spy on
    `difflet.ops` `apply_rotary_emb_slices` is called for the interleaved model's self-attention
    (2 calls per block: q and k) and not for a model after `use_half_split_rope_()`; the toy's
    output is unchanged (`torch.equal`) against the output before this task on the CPU backend.
- [x] **Step 2: Run** them: fail on the missing name.
- [x] **Step 3: Implement.**
- [x] **Step 4: Run** `pytest tests/unit/models tests/unit/backends/test_neuron_rotary.py tests/unit/backends/test_neuron_wan_fp8.py tests/unit/quant tests/unit/cli -q` and every Wan test under `tests/unit/backends`: pass.
- [x] **Step 5: Commit** `fix(wan): bf16 Wan is bit-reproducible again — its interleaved rotary runs the stride-2 law on the neuron backend (difflet.ops.apply_rotary_emb_slices), not fcb0dca's x @ P, which made two bf16 runs differ (433 vs 438 ms per step at tp4 480x832x9); HunyuanVideo keeps x @ P`.
- [x] **Step 6 (device, not the implementer's):** two bf16 generates of Wan 2.1 at tp4 480x832x9,
  50 steps, seed 42, "a cat walking through a garden": bit-identical, and equal to `450d04ef`;
  the new default's 9-frame clip unchanged (`da16a5d3`), since its self-attention is half-split.

### Task H6: the rule reverts H2's default for Wan (the two-expert Wan 2.2 measurement)

H3 step 3's Wan 2.2 rows and the 12-clip test had run one expert (no `--dual-expert`). Rerun with both experts
(`wan_h3/wan22d`, `wan_h3/multi/wan22d`, the same prompts, seeds and arms; both experts reorder, the log prints the
line twice): steps bf16 432.7 / old 390.2 / new 329.0 ms at 9 frames and 5079.7 / 4984.5 / 4730.3 ms at 81 frames;
the 12-clip mean against bf16 is old 29.04 dB / SSIM 0.895 / LPIPS 0.084, new 27.60 / 0.860 / 0.106 (-1.44 dB,
-0.034; new better on 3 of 12 clips; the worst clip, prompt 1 seed 1234, 28.50 -> 17.90 dB / SSIM 0.562). Step 5's
rule on the means fails for Wan 2.2 (both thresholds), so H2's `targets.py` change is reverted for "wan" (one table
serves Wan 2.1 and 2.2; Wan 2.1 alone passed at -0.46 dB). H1 and the application rule stay: an override that
quantizes a self-attention q or k still gets the half-split layout.

**Files:**
- Modify: `difflet/quant/targets.py` (`WAN_TARGETS`, `WAN_KEPT_BF16`, the Wan comment)
- Modify: `tests/unit/quant/test_targets_wan.py`, `tests/unit/backends/test_neuron_wan_fp8.py`

**Interfaces:**
- `WAN_TARGETS` is the `917d087` set again, in its order: `attn2.(to_q|to_k|to_v)`, `ffn.net.0.proj`, `ffn.net.2`
  (so the default spec's digest is the `917d087` one: directory `mxfp8-6d28df7cc78386b5` for Wan 2.1).
- `WAN_KEPT_BF16` lists `attn1.(to_q|to_k|to_v)` again. The comment keeps phase H's cause and fix (the rotary's
  element-wise DMAs; the half-split layout: 8.05 vs 9.30 ms per tp4 block, 0.77x at 9 frames) and records why the
  default does not take it: the two-expert Wan 2.2 12-clip numbers above, with Wan 2.1's (-0.46 dB) beside them.

**Done 2026-11-08** (`90c648f`, review fix `19b0aa9`; evidence 8j). Final-tree reruns from a snapshot of `90c648f`
(evidence 8k, `wan_h3/final/`, tp4 480x832x9, no override, each pair bit-identical): Wan 2.1 default 393.1 / 393.5 ms
against bf16 440.0 at `c7c2346` (0.89x), directory `mxfp8-6d28df7cc78386b5`; Wan 2.2 with both experts bf16 439.6 /
439.4 ms and default 391.8 / 391.0 ms (0.89x), directories `transformer/mxfp8-372c52a68c440756` and
`transformer_2/mxfp8-03f6d63f26aab3cf`; no reorder line in either default. The step-4 pytest counts are not recorded
here. The status page is updated outside this plan's commits.

- [x] **Step 1: Update the tests.** `test_targets_wan.py`: the default-target test covers the cross-attention q/k/v and
  the FFN only again (restore its earlier name), the kept-bf16 test lists `attn1.to_q`. `test_neuron_wan_fp8.py`:
  `TARGETS_PER_BLOCK = 5`, `MODULE_TARGETS` / `CHECKPOINT_TARGETS` without `attn1.*`; the application test asserts
  the default leaves `attn1.to_q` a plain linear and `rope_layout == "interleaved"` on the module and every `attn1`;
  the override tests that quantize `attn1` (H2's) keep asserting `"half_split"` and the reference match.
- [x] **Step 2: Run** them: the changed assertions fail.
- [x] **Step 3: Implement.**
- [x] (partial: the pytest counts are not recorded, see Done) **Step 4: Run** `pytest tests/unit/quant tests/unit/models tests/unit/backends/test_neuron_wan_fp8.py tests/unit/cli -q`: pass.
- [x] **Step 5: Commit** `feat(quant): Wan's default MXFP8 set goes back to the cross-attention q/k/v and the FFN — with both Wan 2.2 experts the self-attention q/k/v cost 1.44 dB / SSIM 0.034 over 12 clips (27.60 vs 29.04 dB from bf16), beyond the rule; the half-split layout stays for overrides (0.77x at 480x832x9)`.
