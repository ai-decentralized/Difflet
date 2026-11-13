# MXFP8 Attention (SageAttention-style QK and PV) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the DiTs' self- and joint-attention core, softmax(QK^T)V, with both matmuls in
MXFP8 on trn3 (`nc_matmul_mx`), switchable per launch, and measure it alone and together with
the linear MXFP8 of `--quant mxfp8`: DiT step time, block profile and quality against the
device's bf16 output.

**Architecture:** A fork of nkilib's `attention_cte` (the kernel every Difflet model calls today)
reduced to the subset Difflet uses — non-causal, optionally bounded (sequence packing for padded
text) — whose first matmul takes Q and K as MXFP8 along the head dim and whose second takes P and
V as MXFP8 along the key axis. Q, K and V are quantized once per call by a pre-pass (K smoothed
first, as SageAttention does); P is quantized inside the kernel's loop. The fork is registered as
the torch op `difflet_neuron::mx_attention` (as `mx_linear` is) and `ops_impl/attention.py`
routes to it when `DIFFLET_ATTN_QUANT=mxfp8`.

**Tech Stack:** NKI 0.7 (`nki.isa.nc_matmul_mx`, `nki.isa.quantize_mx`, `nki.isa.exponential`),
nkilib 2.14 (`core/attention/attention_cte.py`,
`experimental/quantize_mxfp8/quantize_mxfp8.py:quantize_block_mxfp8_kernel`), torch-neuronx
`nki_op` / `wrap_nki`, pytest.

**Spec:** the user's decisions of 2026-11-13 (conversation): a new branch from
`feat/mxfp8-video-models` (this one, `feat/mxfp8-sage-attention`); QK and PV both quantized,
MXFP8 (trn3 has no INT8 matmul, so SageAttention's INT8 QK is not available); fork
`attention_cte`; no CPU fake-quant quality study now — speed first, but every run reports its
quality metrics; test together with linear quantization. Background:
`docs/plans/2026-11-13-dit-block-profile.md` (its results decide where this pays).

## Global Constraints

- trn3 only (NeuronCore-v4: `nc_matmul_mx`); the switch is refused on any other device or exec
  mode with the same check as `--quant` (`core/config_checks.require_fp8_supported`).
- Switch: environment variable `DIFFLET_ATTN_QUANT` in `{"", "mxfp8"}`, read once at import of
  `ops_impl/attention.py`; any other value is an error naming the variable and the allowed values.
  With it unset, every compiled graph is unchanged (bit-identical outputs: Qwen-Image `--quant
  mxfp8` md5 `7737239be780a7877cd8b2ebe32bfb2f`, Wan 2.1 bf16 480x832x9 md5 `450d04ef...`).
- Routed: `attention()`'s non-causal paths (unbounded and bounded) with head dim 128. Not routed
  (stay bf16): `cross_attention`, ring / Ulysses / joint variants, causal calls, head dim != 128
  (LTX-2's audio stream, 64).
- MX blocks: 32 elements, E8M0 scales, OCP `float8_e4m3fn`; Q and K blocked along the head dim,
  P and V blocked along the key axis, in `nc_matmul_mx`'s block membership (`difflet/quant/mx.py`
  `swizzle_perm` gives it for any contraction length).
- K smoothing: `k - k.mean(dim=key axis)` per (batch, head, channel) before quantization
  (softmax-invariant: it shifts every score of a query row by the same constant).
- Device runs: LNC=2, 50 steps, seed 42 for quality runs; the Plan-A shapes and layouts; device
  hygiene and disk rules as in the profile plan.
- Quality: PSNR / SSIM / LPIPS against the device's bf16 output of the same config
  (`scripts/compare_difflet_runs.py`), next to the bf16-vs-bf16 floor where the evidence doc has
  one for that shape. CPU fp32 only if the user asks.
- One commit per task, pushed to `origin/feat/mxfp8-sage-attention`; the artifact page updated
  per model.

## Decision gates (reported to the user, not decided alone)

- **G1 (after Task 3):** PV-only MXFP8 kernel time vs `attention_cte` bf16 at Wan 2.1's per-rank
  shape. Report the ratio, the accuracy and the engine profile.
- **G2 (after Task 4):** QK+PV kernel time vs PV-only and vs bf16, same shape. QK's contraction
  is the head dim (128 = 32 partitions x4 in MX), a quarter of the PE rows unless four matmuls
  share the array by row tiling; if QK in MXFP8 is not faster than QK in bf16, report and ask.
- **G3 (after Task 6):** per model, whether the attention switch should become a default.

## Review Focus

1. Key lengths that are not a multiple of the tile (Wan 2.1 81f, HunyuanVideo-1.5's 49,290 +
   1,985): zero-padded keys score 0, not -inf, and after K smoothing they are not even zero — the
   op pads with bounds so padded keys get exactly zero probability. Test in Task 1 and Task 5.
2. Bounded rows (padded text in HunyuanVideo, HV1.5, Qwen-Image, LTX-2): a 32-key P block that
   straddles a bound must contribute only its in-range keys. Test in Task 1 and the probe.
3. The bounded path needs `scale == 1.0` in the kernel (attention_cte's range_select): q is
   pre-scaled in fp32 before quantization. Test in Task 5.
4. LNC2 launch with an odd B*H per core (e.g. HunyuanVideo-1.5 at tp2: 8 heads per rank, x2
   batch): the grid split must not drop a head. Probe in Task 3.
5. A model whose attention is called through `cross_attention` or SP/CP variants silently
   running bf16 while the user believes it is quantized: the run logs once which calls were
   routed and how many (Task 5).

---

### Task 1: CPU reference of MXFP8 attention

**Files:**
- Create: `difflet/quant/mx_attention.py`
- Test: `tests/unit/quant/test_mx_attention.py`

**Interfaces:**
- Produces: `mx_attention_reference(q, k, v, *, scale: float, bound_min=None, bound_max=None,
  quantize_qk: bool = True, quantize_pv: bool = True, p_mode: str = "dynamic") -> torch.Tensor`
  on `[BH, S, D]` fp32/bf16 tensors (CPU), returning `[BH, S_q, D]` in q's dtype; `p_mode` in
  `{"dynamic", "static"}` ("static": P scaled by 2**8 into one E8M0-identity block scale).

- [ ] **Step 1: Failing tests** — with `quantize_qk=quantize_pv=False` it equals
  `torch.nn.functional.scaled_dot_product_attention` to 1e-5 (fp32); smoothing K alone leaves the
  output unchanged to 1e-5; with bounds, keys outside `[bound_min, bound_max)` change nothing
  (perturb them, output identical); a 32-key block straddling `bound_max` gives the same output
  as the truncated sequence; on Gaussian inputs (BH 2, S 1024, D 128) the quantized output's
  cosine to fp32 is > 0.99 for both p modes.
- [ ] **Step 2: Run** — `DIFFLET_BACKEND=cpu PYTHONPATH=. ~/venvs/native-fp8/bin/python -m pytest tests/unit/quant/test_mx_attention.py -q -p no:cacheprovider`; expected FAIL.
- [ ] **Step 3: Implement** with `difflet.quant.mx` (`quantize_mx` / `dequantize_mx` on the
  tensor transposed so the contraction axis is last: D for Q and K, keys for P and V); the
  softmax in fp32 over the masked scores.
- [ ] **Step 4: Run** — expected PASS. **Step 5: Commit and push** — `feat(quant): CPU reference of MXFP8 attention (QK along D, PV along keys, K smoothing, bounds)`.

### Task 2: Probe harness and the bf16 baseline

**Files:**
- Create: `scripts/mx_attention/probe.py`, `scripts/mx_attention/shapes.json`
- Test: `tests/unit/scripts/test_mx_attention_probe.py`

**Interfaces:**
- Consumes: `mx_attention_reference` (Task 1).
- Produces: `probe.py --shape <name> --impl <bf16|mx> [--p-mode ...] --out <json>` on one core:
  rel L2 and cosine of the device output vs the reference (exact quantized reference for `mx`,
  fp32 for `bf16`), and the kernel's device time (median of 20 calls after 3 warm-ups, from
  `neuron-explorer capture` of the probe NEFF); `shapes.json` = per model the per-rank
  `{bh, s_q, s_k, d, bounded}` of its self/joint attention, read from the Plan-A block parquets'
  attention custom-call operand shapes (one entry per model at the Plan-A shape).

- [ ] **Step 1: Failing test** — the CLI rejects an unknown shape name and an impl outside
  `{bf16, mx}`; `shapes.json` holds all seven models, every `d == 128` except where the test
  names it.
- [ ] **Step 2-4:** run (FAIL), implement, run (PASS).
- [ ] **Step 5: Device baseline** — `--impl bf16` for every shape on one idle core: the bf16
  `attention_cte` time per call is the denominator of every later ratio. Record in the evidence doc.
- [ ] **Step 6: Commit and push** — `scripts(mx-attention): one-core probe and the bf16 attention_cte baseline per model shape`.

### Task 3: Kernel v0 — the fork, PV in MXFP8

**Files:**
- Create: `difflet/backends/neuron/ops_impl/mx_attention_kernel.py` (the fork; keeps
  attention_cte's license header and a line naming the nkilib version it was forked from)
- Test: `tests/unit/backends/test_neuron_mx_attention_kernel.py` (CPU: argument checks only)

**Interfaces:**
- Produces: `mx_attention_fwd(q, k, v, v_scales, *, scale: float, bound_min=None,
  bound_max=None, qk_mx: bool = False, q_scales=None, k_scales=None, p_mode: str = "dynamic")`,
  an `nki.jit` kernel on attention_cte's `tp_q=True, tp_k=True` layout; asserts (with the
  argument named) on anything outside the subset: causal, prior, sink, sliding window, CP flags,
  position bias, cache_softmax, head dim != 128.

- [ ] **Step 1:** Copy `attention_cte.py` (nkilib 2.14, the venv's copy) and delete every branch
  outside the subset; the bf16 subset must still match `attention_cte` bit for bit on the probe
  (Task 2, `--impl` of the fork with both MX flags off) before any MX change.
- [ ] **Step 2:** MM2 on `nc_matmul_mx`: V arrives as MXFP8 along keys (`float8_e4m3fn_x4` +
  E8M0 scales, quantized by the pre-pass); P quantized per 512-key tile — `p_mode="dynamic"`:
  `exponential` to bf16 then `quantize_mx`; `p_mode="static"`: `exponential` writes fp8
  directly with the max offset lowered by ln(2**8) and identity micro-scales (E8M0 127), the
  output divided by 2**8 (or the row sum taken from the same scaled P).
- [ ] **Step 3: Probe** both p modes at every shape: rel L2 vs the Task 1 reference with
  `quantize_qk=False` below 3e-3 (the bf16-output rounding level of `mx_linear`, evidence
  section 1a); device time per call.
- [ ] **Step 4: Gate G1** — report to the user: ratio to bf16 per shape and p mode, accuracy,
  and the engine split of one capture (Plan-A `analyze.py` on the probe NEFF).
- [ ] **Step 5: Commit and push** — `feat(neuron): MXFP8 attention kernel v0, a reduced attention_cte fork with PV on nc_matmul_mx`.

### Task 4: Kernel v1 — QK in MXFP8

**Files:**
- Modify: `difflet/backends/neuron/ops_impl/mx_attention_kernel.py`

- [ ] **Step 1:** MM1 on `nc_matmul_mx` with Q and K as MXFP8 along the head dim (pre-pass
  quantized, K smoothed first); at D = 128 the contraction fills 32 partitions, so issue four
  matmuls per 128x128 PE array with `tile_position` row tiling (four query tiles or four key
  tiles at once); scores stay fp32 in PSUM, so the softmax path is unchanged.
- [ ] **Step 2: Probe** at every shape: rel L2 vs the full Task 1 reference below 3e-3; time.
- [ ] **Step 3: Gate G2** — report QK+PV vs PV-only vs bf16 per shape, with the engine split.
- [ ] **Step 4: Commit and push** — `feat(neuron): MXFP8 attention kernel v1, QK^T on nc_matmul_mx with row tiling`.

### Task 5: The op, the pre-pass and the switch

**Files:**
- Create: `difflet/backends/neuron/ops_impl/mx_attention.py`
- Modify: `difflet/backends/neuron/ops_impl/attention.py` (routing in `attention()`)
- Test: `tests/unit/backends/test_neuron_mx_attention.py`

**Interfaces:**
- Consumes: `mx_attention_fwd` (Task 3/4).
- Produces: torch op `difflet_neuron::mx_attention(q, k, v, bound_min, bound_max, scale) -> bf16`
  on `[BH, S, D]` (bounds may be empty tensors for the unbounded call), registered with
  `nki_op` + `wrap_nki` and a fake, as `fp8_kernels._load_mx_op` does; `ATTN_QUANT: str` (the
  resolved switch) and `mx_attention_calls() -> int` (routed-call counter for the run log).
  The pre-pass in the op: K smoothing, q pre-scaled in fp32 when bounded, pad S_q and S_k to the
  kernel's multiples with bounds covering only real keys, then
  `quantize_block_mxfp8_kernel` (or the kernel's own `quantize_mx` on load, whichever the probe
  measured faster in Task 3) for Q, K, V.

- [ ] **Step 1: Failing tests** (CPU, the op monkeypatched to the Task 1 reference): with the
  switch set, `attention()` routes unbounded and bounded non-causal D=128 calls to the op and
  leaves causal, D=64, `cross_attention`, ring and Ulysses calls on their bf16 paths; an invalid
  value raises naming `DIFFLET_ATTN_QUANT`; padded keys get zero weight (perturb the padding,
  output identical); with the switch unset the traced graph contains no `mx_attention` node.
- [ ] **Step 2-4:** run (FAIL), implement, run (PASS); the unit suite of `tests/unit/backends`
  and `tests/unit/quant` passes.
- [ ] **Step 5: Device identity** — Qwen-Image `--quant mxfp8` with the switch unset: md5
  `7737239be780a7877cd8b2ebe32bfb2f`.
- [ ] **Step 6: Commit and push** — `feat(neuron): DIFFLET_ATTN_QUANT=mxfp8 routes self/joint attention to the MXFP8 kernel`.

### Task 6: Model runs, 2x2 arms

**Files:**
- Create: `scripts/mx_attention/run_arms.sh`
- Modify: `docs/verification/2026-11-13-dit-block-profile.md` (new section) and the artifact page

- [ ] **Step 1:** Per model at its Plan-A shape and layout, four arms — `bf16`, `linear`
  (`--quant mxfp8`), `attn` (`DIFFLET_ATTN_QUANT=mxfp8`), `both` — HunyuanVideo-1.5 `bf16` and
  `attn` only. Order: `wan21`, `hunyuan_video`, `ltx_2`, `qwen_image`, `flux`, `wan22`,
  `hunyuan_video_15` (the order a reader gets results in; the arms themselves run one per idle
  device in parallel, as the profile plan's lanes do). Each arm: one 50-step generate (step
  median from the log, output saved),
  a second generate for the `attn` and `both` arms (determinism: md5 equal), and a Plan-A block
  capture.
- [ ] **Step 2:** Quality: `compare_difflet_runs.py` of `linear`, `attn`, `both` against `bf16`
  (PSNR / SSIM / LPIPS; latents for HunyuanVideo-1.5 if its 45-minute host decode is the
  bottleneck, said so in the table), next to the evidence doc's bf16 floor for the shape.
- [ ] **Step 3:** After each model: commit its numbers, push, update the page (step time per
  arm, block breakdown with the attention component, quality).
- [ ] **Step 4: Gate G3** — the table to the user.
- [ ] **Step 5: Commit and push** — `docs(mx-attention): 2x2 arms per model — step time, block profile, quality`.

### Task 7 (only if Task 6's profile asks for it): V from the linear as MXFP8

Only V can come straight out of the `to_v` MX linear in the attention's MXFP8 layout: Q and K
pass qk-norm and RoPE between the linear and the attention, which need full precision, so their
quantization stays in the pre-pass. This task is written as its own plan if the Task 6 profile
shows the V pre-pass costing at least 5% of the attention component's time.
