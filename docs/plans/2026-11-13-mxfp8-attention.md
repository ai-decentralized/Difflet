# MXFP8 Attention (SageAttention-style QK and PV) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the DiTs' self- and joint-attention core, softmax(QK^T)V, with its matmuls in
MXFP8 on trn3 (`nc_matmul_mx`), switchable per launch, and measure it together with the linear
MXFP8 of `--quant mxfp8` against bf16: DiT step time, block profile, and quality against the
device's bf16 output and the existing CPU fp32 references.

**Architecture:** A fork of nkilib's `attention_cte` (the kernel every Difflet model calls today),
first copied verbatim and guarded to the subset Difflet uses (non-causal, optionally bounded),
then changed so that Q, K and V are quantized to MXFP8 as the fork's loaders bring them into
SBUF (`nisa.quantize_mx`), MM1 (QK^T, contraction = head dim) and MM2 (PV, contraction = keys) run
on `nc_matmul_mx`, and P is formed as today (`exponential` to bf16), transposed and quantized per
512-key tile. The torch side only smooths K, pre-scales q for the bounded path and pads. The fork
is registered as the torch op `difflet_neuron::mx_attention` (as `mx_linear` is) and
`ops_impl/attention.py` routes the self/joint call sites to it under `DIFFLET_ATTN_QUANT`.

**Tech Stack:** NKI 0.7 (`nisa.nc_matmul_mx`, `nisa.quantize_mx`, `nisa.exponential`), nkilib 2.14
(`core/attention/attention_cte.py`; `experimental/attention_mxfp8/attention_mxfp8_tkg.py` as the MX
reference pattern), torch-neuronx `nki_op` / `wrap_nki`, pytest.

**Spec:** the user's decisions of 2026-11-13 (conversation): branch `feat/mxfp8-sage-attention`
from `feat/mxfp8-video-models`; QK and PV both quantized, MXFP8 (trn3 has no INT8 matmul);
fork `attention_cte`; no CPU fake-quant quality study; speed decides every gate; model runs are
bf16 vs both (HV1.5: bf16 vs attn). Background: `docs/plans/2026-11-13-dit-block-profile.md`
(Plan A; its page is the one this plan extends). Reviewed by a Fable panel on 2026-11-13; this is
the revised plan.

## Global Constraints

- trn3 only. The switch is checked where the model and exec mode are known
  (`difflet/backends/neuron/core/application_base.py`, beside the `--quant` check at :194-198):
  with `DIFFLET_ATTN_QUANT=mxfp8` set, `require_fp8_supported` runs even when `quant is None`.
- Switch contract, fixed from Task 5 on: `DIFFLET_ATTN_QUANT` unset, `"off"` or `"mxfp8"`; any
  other value is an error naming the variable and the three values. Resolved once at import into
  `ATTN_QUANT` in `ops_impl/attention.py`. Unset or `"off"`: every compiled graph is unchanged
  (Qwen-Image `--quant mxfp8` md5 `7737239be780a7877cd8b2ebe32bfb2f`; Wan 2.1 bf16 480x832x9 md5
  `450d04ef...`).
- Routed, by call site: the self/joint attention of Qwen-Image, FLUX, Wan (`attn1`), HunyuanVideo,
  HunyuanVideo-1.5 and LTX-2's video `attn1`, head dim 128. Not routed (bf16): every
  cross-attention call site (Wan `attn2`; LTX-2 `attn2`, `audio_*`, `audio_to_video_attn`,
  `video_to_audio_attn`), `cross_attention()`, `_dense_bhsd`, ring / Ulysses / joint ring
  variants, causal calls, head dim != 128 (LTX-2's audio, 64). `attention()` takes a keyword-only
  `mx: bool` that those call sites pass as `False`.
- MX blocks: a block is 8 partitions x 4 packed bytes of an operand in SBUF; which logical indices
  form a block is defined by the fork's loaders, and the CPU reference uses the same mapping,
  passed explicitly (`qk_perm`, `pv_perm`, pure functions exported by the kernel module).
  OCP `float8_e4m3fn`, E8M0 scales.
- Padding: S_k padded to a multiple of 512 and S_q to a multiple of 128 (the fork supports no
  remainder tile). Unbounded calls stay on the unmasked path: MM1, row max and `exponential` run on
  `num_valid_k` as the kernel does today, and the last P tile's tail is zeroed before
  `quantize_mx`; V's padding is zero. Bounds are used only by models that are bounded in bf16
  (HunyuanVideo, HunyuanVideo-1.5, Qwen-Image, LTX-2); the bounded path needs scale 1.0, so q is
  pre-scaled in fp32. K smoothing subtracts the mean over the real keys only.
- Quality: PSNR / SSIM / LPIPS against the device's bf16 output of the same config
  (`scripts/compare_difflet_runs.py`), next to the bf16-vs-bf16 floor where the evidence doc has
  one; where `/home/ubuntu/fp8-bench/fp32_ref/<key>` already has a CPU fp32 reference for the
  shape, also scored with `scripts/fp32_ref/score.py`. No new CPU fp32 runs.
- Steps and flags as the `bench.sh` key sets them (Wan 2.2 official: 40 steps); seed 42; LNC=2.
- Devices and lanes: Plan A's Global Constraints and `scripts/profile_dit/lib.sh`
  (`require_idle`, `BUSY` requeue, 48-thread CPU sets, cold-compile cap, prewarm, disk gate);
  outputs under `/mnt/nvme/difflet-mxattn/<key>/<arm>/`.
- Timing protocol (Task 6): (a) compile and warm every arm in parallel on idle devices, untimed;
  (b) once our own `neuronx-cc` processes have exited, timed runs: one lane per model, bf16 then
  the quantized arm on the same device and CPU set, models in parallel, two lanes per NUMA node;
  `uptime` and `fuser /dev/neuron*` logged at each timed run's start and end. Narrower lanes only
  if Wan 2.1 480x832x9 bf16 lands within 1% of 437-440 ms on them.
- The page is Plan A's English page; this plan adds the G1-G3 tables, bf16 vs both step times,
  the attention phase split, Q/K/V quantization time, and quality.
- One commit per task (Task 3's verbatim fork and its branch pruning are separate commits),
  pushed to `origin/feat/mxfp8-sage-attention`.

## Decision gates (decided by this rule, user 2026-11-13: "只要有速度提升就可以有")

**Faster** = the candidate's median is below the comparison's by more than
`J = max(2%, spread)`, spread = (max - min) / median over the comparison's repeated runs. Quality
is reported every time and never vetoes. **Broken** (a bug, fixed before the gate is read) =
(a) at kernel level, the probe's device output misses Task 3/4's accuracy threshold against the
Task 1 mirrored reference, or (b) at model level, non-finite latents or noise on screen. Model-level
cosine against bf16 is reported only, never a bug signal (HunyuanVideo-1.5's two bf16 runs are
already 0.911 / 0.897 apart, evidence 9d).

- **G1 (Task 3):** PV-only MXFP8 passes if it is faster than `attention_cte` bf16 on any
  `shapes.json` row (the table has a `path: bounded/unbounded` column). If it wins nowhere, run
  Task 4's QK-only variant (`qk_mx=True`, PV bf16) before stopping; stop only if that wins nowhere
  too.
- **G2 (Task 4):** per shape — where PV passed, QK+PV vs PV-only; where it did not, QK-only vs
  bf16. QK code is removed only if it wins on no shape. After G2 each model gets one variant
  (`{pv}`, `{qk}` or `{qk, pv}`), recorded next to `ATTN_QUANT_MODELS`; Task 6's quantized arm runs
  that variant; a model with no winning variant gets no quantized-attention arm. A kept QK is
  provisional until Task 6's block capture shows the attention component below Plan A's bf16 one.
- **G3 (Task 6):** keep = `both / bf16` (this session, same device and CPU set) <
  `linear / bf16` (the evidence row: section, host) - J. One linear-alone run is made for a model
  only when the two are within 2J, or when this session's bf16 step differs from the evidence's by
  more than 1%. HunyuanVideo-1.5: `attn / bf16` < 1 - J. Kept models become the `--quant mxfp8`
  default (Task 7).

## Review Focus

1. Key lengths that are not a multiple of 512 (Wan 81f 32,760; HV1.5 51,275): zero-padded keys
   score 0, not -inf; the unmasked path must zero P's tail before quantizing. Tests in Task 1, probe
   in Task 3.
2. Bounded rows: an MX block (by `pv_perm`) holding in-range and out-of-range keys must contribute
   only the in-range ones. Test in Task 1; probe with `bound_max % 32 != 0` in Task 3.
3. The bounded path's scale == 1.0 rule: q pre-scaled in fp32 before quantization. Test in Task 5.
4. LNC2 with HunyuanVideo-1.5's 8 heads per rank: the grid split must not drop a head; Task 3's
   probe compares head by head.
5. A call site silently staying bf16 while the user believes it is quantized: the run logs once
   the routed-call counts bucketed by `(s_q, s_k, bounded)` (Task 5).

---

### Task 1: CPU reference that mirrors the kernel

**Files:**
- Create: `difflet/quant/mx_attention.py`
- Test: `tests/unit/quant/test_mx_attention.py`

**Interfaces:**
- Consumes: `qk_perm(d: int) -> torch.Tensor`, `pv_perm(tile: int) -> torch.Tensor` (Task 3
  exports them from the kernel module; until then the test uses `difflet.quant.mx.swizzle_perm`).
- Produces: `mx_attention_reference(q, k, v, *, scale, bound_min=None, bound_max=None,
  qk_mx: bool, pv_mx: bool, qk_perm=None, pv_perm=None, section_len: int | None = None) -> Tensor`
  on `[BH, S, D]` CPU tensors, returning `[BH, S_q, D]` in q's dtype. It forms P as the kernel
  does: `E = exp(score - m)` with `m` the row max (or, with `section_len`, each section's running
  max, accumulated with `exp(prev_max - cur_max)` corrections), E rounded to bf16, quantized per
  512-key tile along keys with `pv_perm`, multiplied by V (quantized the same way), divided by the
  row sum the kernel uses (the unquantized fused sum from `exponential`; Task 3 Step 3 confirms it
  and switches the reference if the kernel sums the dequantized P).

- [ ] **Step 1: Failing tests** — with `qk_mx=pv_mx=False`, sectioned (`section_len=8192`) and
  unsectioned both equal `scaled_dot_product_attention` in fp32 to 1e-5; K smoothing alone
  changes nothing to 1e-5; perturbing out-of-bounds keys changes nothing; an MX block (by
  `pv_perm`) holding both in-range and out-of-range keys gives the same output as the truncated
  sequence; unbounded with a key length not a multiple of 512, perturbing the padding changes
  nothing; a non-default perm changes the scales.
- [ ] **Step 2: Run** — `DIFFLET_BACKEND=cpu PYTHONPATH=. ~/venvs/native-fp8/bin/python -m pytest tests/unit/quant/test_mx_attention.py -q -p no:cacheprovider`; expected FAIL.
- [ ] **Step 3: Implement** with `difflet.quant.mx.quantize_mx` / `dequantize_mx` on the operand
  transposed so the contraction axis is last, the perms passed explicitly.
- [ ] **Step 4: Run** — expected PASS. **Step 5: Commit and push** — `feat(quant): CPU reference of MXFP8 attention mirroring the kernel's P formation`.

### Task 2: Probe harness and the bf16 baseline

**Files:**
- Create: `scripts/mx_attention/probe.py`, `scripts/mx_attention/shapes.json`
- Test: `tests/unit/scripts/test_mx_attention_probe.py`

**Interfaces:**
- Consumes: `mx_attention_reference` (Task 1).
- Produces: `probe.py --shape <name> --impl <cte|fork> [--pv-mx] [--qk-mx] --out <json>` on one
  core at LNC=2: rel L2 / cosine of the device output against the reference (the Task 1 mirror
  for `fork`, fp32 for `cte`), per-head for multi-head shapes; time = the compiled region from
  bf16 q/k/v to bf16 out (quantization included), measured as evidence 4c measured kernels — 20
  calls in one process, one sync — in 3 processes per variant, reporting median and spread;
  capture only for the engine split. `shapes.json` rows (per rank; confirmed against the routed-call
  log of Task 5 or Plan A's attention operand shapes, whichever exists first):

| name | B*H | S_q = S_k | D | path |
|---|---|---|---|---|
| `wan_81f` | 10 | 32,760 | 128 | unbounded |
| `hv1_L` | 6 | 67,320 + 256 text | 128 | bounded |
| `hv15_121f` | 8 | 49,290 + 1,985 text | 128 | bounded |
| `ltx2_video` | 8 | 6,144 | 128 | unbounded |
| `flux_1024` | 6 | 4,096 + 512 text | 128 | unbounded |
| `qwen_1024` | 6 x CFG batch | 4,096 + text (read from the run) | 128 | bounded |
| `ltx2_audio` | 8 | 126 | 64 | not routed (listed so the test pins it) |

- [ ] **Step 1: Failing test** — the CLI rejects an unknown shape and an impl outside
  `{cte, fork}`; `shapes.json` has the seven rows; every routed row has `d == 128`, `ltx2_audio`
  has 64 and `routed: false`.
- [ ] **Step 2: Run** (FAIL). **Step 3: Implement.** **Step 4: Run** (PASS).
- [ ] **Step 5: Device baseline** — `--impl cte` for every routed shape: median and spread are
  the denominators of G1/G2. Record in the evidence doc.
- [ ] **Step 6: Commit and push** — `scripts(mx-attention): one-core probe and the attention_cte bf16 baseline per model shape`.

### Task 3: Kernel v0 — the fork, PV in MXFP8

**Files:**
- Create: `difflet/backends/neuron/ops_impl/mx_attention_kernel.py`
- Test: `tests/unit/backends/test_neuron_mx_attention_kernel.py`

**Interfaces:**
- Produces: `mx_attention_fwd(q, k, v, *, scale, bound_min=None, bound_max=None, pv_mx: bool,
  qk_mx: bool)` (`nki.jit`, attention_cte's `tp_q=True, tp_k=True` layout, bf16 inputs);
  `qk_perm`, `pv_perm` (pure functions, the loaders' block membership).

- [ ] **Step 1: Verbatim fork** — copy nkilib 2.14's `attention_cte.py` (keep its license header,
  add a line naming the version), add `kernel_assert`s refusing causal, prior, sink, sliding
  window, CP flags, position bias, `cache_softmax`, head dim != 128; CPU test of the argument
  checks; probe `--impl fork` with both MX flags off is bit-identical to `cte` on every shape.
  Commit and push (`feat(neuron): fork attention_cte verbatim with the subset's guards`).
- [ ] **Step 2: Lock the block membership** — on one 128 x 512 tile, compare the kernel's
  `quantize_mx` scale output with the reference's scales under `pv_perm`, element by element,
  before reading any rel L2.
- [ ] **Step 3: MM2 on `nc_matmul_mx`**, as a checklist:
  (a) V's loader brings each `[128, D]` x4 tile holding 512 keys, quantized on load, with its scales;
  (b) P's layout for MM2 is `src[g, 4m+j] = P[m, 4g+j]`: a micro-probe first checks whether one
  `dma_transpose` access pattern produces it; otherwise use `swizzle_quantize_mx`'s path (fp32 view
  plus stride-2 `nc_transpose` through PSUM), whose TensorE cost is reported in G1;
  (c) one `quantize_mx` per (query group, 512-key tile), after zeroing the tail of the last tile on
  the unmasked path; (d) one `nc_matmul_mx` per 512 keys, adapting `_pv_mm2_tile_selection` and
  relaxing `_V_TILE_SZ == sb_p`; (e) remainders handled by padding to 512.
- [ ] **Step 4: Probe** every routed shape (HV1.5 at LNC=2 head by head; bounded shapes with
  `bound_max % 32 != 0`): rel L2 against the Task 1 mirror below a threshold fixed by the first
  probe at the smallest shape (expected near the 2.3e-3 bf16-output rounding of `mx_linear`,
  evidence 1a; never widened to ~3e-2); time. Record `quantize_mx`'s VectorE time beside the
  current capture's MM2 13.2 ms / 80,730 instructions and `exponential` 3.65 ms, and confirm the
  Wan arm has no `RANGE_SELECT` instructions.
- [ ] **Step 5: Gate G1** — apply the rule; record per shape and path: ratio, accuracy, engine
  split, decision.
- [ ] **Step 6: Prune** the branches outside the subset (separate commit), probe unchanged.
- [ ] **Step 7: Commit and push** — `feat(neuron): MXFP8 attention kernel v0, PV on nc_matmul_mx`.

### Task 4: Kernel v1 — QK in MXFP8

**Files:**
- Modify: `difflet/backends/neuron/ops_impl/mx_attention_kernel.py`,
  `tests/unit/backends/test_neuron_mx_attention_kernel.py` (argument checks for `qk_mx`)

- [ ] **Step 1: Row-tiling feasibility** — a minimal kernel with four `nc_matmul_mx` in the four
  row quadrants compiles and matches the reference. Constraints from the ISA doc: the tile size
  follows the stationary partition size (32 at D = 128); stationary, moving and both scales sit at
  SBUF partition `start_row` (0 / 32 / 64 / 96); the stationary free size must exceed 64 (so S_q
  is a multiple of 128, asserted in the kernel); no tiling with indirection; check the four
  concurrent score tiles against the fork's PSUM bank allocation.
- [ ] **Step 2: MM1 on `nc_matmul_mx`** with Q and K quantized on load along the head dim (K
  smoothed on the torch side); whether four query tiles or four key tiles share the array is
  decided by the probe; scores stay fp32 in PSUM. QK-only (`qk_mx` without `pv_mx`) is supported.
- [ ] **Step 3: Probe** every routed shape, all variants; threshold as Task 3.
- [ ] **Step 4: Gate G2** — apply the rule per shape; assign each model its variant.
- [ ] **Step 5: Commit and push** — `feat(neuron): MXFP8 attention kernel v1, QK^T on nc_matmul_mx with row tiling`.

### Task 5: The op, the routing and the switch

**Files:**
- Create: `difflet/backends/neuron/ops_impl/mx_attention.py`
- Modify: `difflet/backends/neuron/ops_impl/attention.py`, `difflet/models/wan/modeling_wan.py`,
  `difflet/backends/neuron/ltx_2/transformer.py`, `difflet/backends/neuron/core/application_base.py`
- Test: `tests/unit/backends/test_neuron_mx_attention.py`

**Interfaces:**
- Consumes: `mx_attention_fwd` (Tasks 3-4).
- Produces: torch op `difflet_neuron::mx_attention(q, k, v, bound_min, bound_max, scale, pv_mx,
  qk_mx) -> bf16` on `[BH, S, D]` (empty bound tensors for unbounded calls), registered with
  `nki_op` + `wrap_nki` and a fake, as `fp8_kernels._load_mx_op`; `ATTN_QUANT: str`;
  `mx_attention_calls() -> dict[tuple[int, int, bool], int]`. The torch side: K smoothing over real
  keys, q pre-scaled in fp32 when bounded, padding per the Global Constraints.

- [ ] **Step 1: Failing tests** (CPU, op monkeypatched to the Task 1 reference, via
  `monkeypatch.setattr` as `test_neuron_attention_c3.py:244` does): with `mxfp8`, `attention()`
  routes unbounded and bounded non-causal D = 128 calls; causal, D = 64, `mx=False`,
  `cross_attention()`, `_dense_bhsd`, ring and Ulysses (cp > 1: no `mx_attention` node in the
  graph) stay bf16; Wan's `_attn_kernel` passes `mx=False` for cross-attention, LTX-2's shared call
  site likewise; a bounded call with scale != 1 reaches the op with scale 1.0 and q pre-scaled;
  padded keys get zero weight; `mx_attention_calls()` counts the routed calls and the log line
  appears once; an invalid value raises naming `DIFFLET_ATTN_QUANT`; with the switch set on a
  non-trn3 platform or in eager mode, the application raises `ValueError`; unset or `off`: no
  `mx_attention` node.
- [ ] **Step 2: Run** (FAIL). **Step 3: Implement**, including the one-time routed-call log line.
- [ ] **Step 4: Run** (PASS); `tests/unit/backends` and `tests/unit/quant` pass.
- [ ] **Step 5: Device identity** — switch unset: Qwen-Image `--quant mxfp8` md5
  `7737239be780a7877cd8b2ebe32bfb2f` and Wan 2.1 bf16 480x832x9 md5 `450d04ef...`.
- [ ] **Step 6: Commit and push** — `feat(neuron): DIFFLET_ATTN_QUANT=mxfp8 routes self/joint attention to the MXFP8 kernel`.

### Task 6: Model runs, bf16 vs both

**Files:**
- Create: `scripts/mx_attention/run_arms.sh`
- Modify: `scripts/profile_dit/capture.sh` and its test (arms `both` = `--quant mxfp8` with
  `DIFFLET_ATTN_QUANT=mxfp8`, `attn` = the switch alone, `hv15_official` only),
  `docs/verification/2026-11-13-dit-block-profile.md`, the page

- [ ] **Step 1: Arms** — per model at its Plan A key: `bf16` and `both` (HunyuanVideo-1.5: `bf16`
  and `attn`), each with the model's G2 variant. Compile and warm in parallel (untimed), then the
  timed protocol of the Global Constraints: per model one lane, bf16 then the quantized arm, two
  requests each (their step medians give the spread), a second generate of the quantized arm
  (determinism: md5 equal), and a Plan A block capture of the quantized arm. A bf16 output already
  on this host with a matching md5 and thread count may serve for quality, never for timing.
- [ ] **Step 2: Quality** — the quantized arm vs bf16 with `compare_difflet_runs.py` (HV1.5 by
  mp4; latents only if its decode fails); scored against `fp32_ref` where the shape has a
  reference (Qwen-Image, FLUX, LTX-2: their Plan A shape is the phase G shape). Columns: quantized
  arm vs bf16 | linear-alone vs bf16 (evidence 7f / 11c / 12c) | bf16 floor | quantized arm vs
  fp32 (with 11c's score beside it). Rows at Wan 2.2 81f and HV1.5 121f are marked "below the
  bf16 floor's resolution, not a quality reading"; long-shape cosine is reported only.
- [ ] **Step 2b (only if the user approves):** a quality-only pair at the short shapes (Wan 2.1 and
  Wan 2.2 480x832x9, HunyuanVideo 320x512x61, HV1.5 9 frames) with the plain `bench.sh` keys,
  reusing a bf16 clip whose md5 matches section 11c; not used by any gate.
- [ ] **Step 3:** After each model: commit its numbers, push, update the page (step times, G3
  row, attention phase split, Q/K/V quantization time and V's share of the attention component).
- [ ] **Step 4: Gate G3** — apply the rule per model; table and decisions to the page.
- [ ] **Step 5: Commit and push** — `docs(mx-attention): bf16 vs both per model — step time, block profile, quality`.

Budget: one cold compile per quantized arm (HV1 ~40 min, Wan 11-16 min, HV1.5 ~24 min) plus one
bf16 and two quantized timed runs per model: ~1.5-2 h of device time without rework.

### Task 7: G3's decisions as the default

**Files:**
- Modify: `difflet/quant/targets.py` (`ATTN_QUANT_MODELS: dict[str, frozenset[str]]`, model -> its
  G2 variant), `difflet/cli/quant.py` (`prepare_quant`: for a model in the table,
  `os.environ.setdefault("DIFFLET_ATTN_QUANT", "mxfp8")`, as the same file does at :156, so an
  explicit `off` stays)
- Test: `tests/unit/quant/test_attn_quant_models.py`

- [ ] **Step 1: Failing tests** — a model in the table with `--quant mxfp8` gets the switch; with
  `DIFFLET_ATTN_QUANT=off` it does not; a model outside the table does not; no `--quant` does not.
- [ ] **Step 2: Run** (FAIL). **Step 3: Implement.** **Step 4: Run** (PASS).
- [ ] **Step 5:** Confirm that the compile / stage cache keys tell attention on and off apart.
- [ ] **Step 6: Device identity** — one kept model's `--quant mxfp8` run equals its Task 6
  `both` md5.
- [ ] **Step 7: Commit and push** — `feat(quant): --quant mxfp8 runs attention in MXFP8 on the models where it is faster`.

## Deferred (not scheduled)

- **V from the linear as MXFP8.** Only V can leave the `to_v` MX linear already in the attention's
  MXFP8 layout; Q and K pass qk-norm and RoPE first. Written as its own plan if Task 6's capture
  shows V's load-time quantization at 5% or more of the attention component.
- **Static P.** P written as fp8 by `exponential` with identity block scales. As planned before
  review it cannot compile: `exponential` writes only legacy `float8_e4m3` / `float8_e5m2`,
  `nc_matmul_mx` takes only x4 types, legacy e4m3 cannot hold 256, and the DGE transpose needs a
  2-byte dtype. A spike only if G1's engine split shows P's `quantize_mx` significant, and only
  after a one-tile probe shows a dtype `exponential` writes that `nc_matmul_mx` accepts beside V's
  `e4m3fn_x4`, and a 1-byte transpose or a layout needing none; scaling 2**7; it reports rel L2
  against fp32 at real S_k beside its speed.
