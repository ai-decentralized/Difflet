# attention_cte exp on ScalarE by default: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make nkilib `attention_cte`'s `allow_vector_exp=False` (softmax exp on ScalarE instead of VectorE)
the default for every bf16 attention call of Difflet's native neuron backend on trn3, and prove it on device.

**Architecture:** One switch in `difflet/backends/neuron/ops_impl/attention.py`: `attention(..., vector_exp=False)`
by default. The bounded path already has a ScalarE-exp op. The unbounded path and the logsumexp (ring / joint-ring)
path go through torch-neuronx internals that cannot forward the knob, so both get a direct `wrapped_flash_fwd`
call that mirrors `torch_neuronx.neuron_dynamo_backend.decompositions._compute_attention_nki` plus the knob.

**Tech Stack:** torch 2.14 + torch-neuronx 2.14.3, nkilib 2.14 (`/opt/aws_neuronx_venv_torch_neuronx_2_14_0`),
venv `/home/ubuntu/venvs/native-fp8`, trn3.48xlarge (16 devices, LNC=2).

**Spec:** user request 2026-11-16 (this plan); evidence it rests on: `docs/verification/2026-11-13-dit-block-profile.md`
section 9 on branch `feat/mxfp8-sage-attention` (commit 5d5b83c5): kernel 7.5-21% faster on six shapes, rel L2 vs
fp32 unchanged, direct call bit-identical to torch-neuronx's SDPA op at the default knob.

## User decisions (2026-11-16)

- Model level runs every tier (A, B and C) of Task 4.
- Quality is reported, never a gate, except: stop and ask if any model's after-vs-CPU-fp32 PSNR is more
  than 0.5 dB below its before-vs-fp32 PSNR.
- MiniMax-H3's `attention_vector_exp` knob is deleted (Task 2).
- Execution: inline (superpowers:executing-plans), one whole-branch review at the end.

## Global Constraints

- Base: `main` at d70faffc. Branch `feat/attn-scalar-exp-default`; one commit per task, each pushed.
- The knob takes effect only on trn3 B1: nkilib `attention_cte.py:1361-1365` (`ac.allow_vector_exp and is_trn3_b1()`).
  trn2 and trn3pre have no `nisa.exponential`; they always run the exp on ScalarE.
- `DIFFLET_NEURON_ATTENTION=sdpa` stays as is (torch-neuronx's SDPA, VectorE exp on trn3). No new env switch.
- The trainium (NxD / torch-xla) backend is out of scope: torch_xla is not installed in the trn3 venv, so its
  four `attention_cte` calls (`difflet/backends/trainium/ops_impl/attention.py:48,79,314,395`) run only on the
  trn2 stack, where the knob is a no-op.
- The NKI Library ring kernel (`_ring_kernel_op`, `ring_attention_spmd_fwd`) has no such argument: unchanged.
- Device runs: LNC=2, 50 steps, seed 42, bf16 (no `--quant`), the evidence doc's prompts and shapes.
- Before every device launch: `fuser /dev/neuron<d>` empty (every runtime also opens `/dev/neuron0`: check the
  holder's `NEURON_RT_VISIBLE_CORES` before calling device 0 busy) and `mpstat -P ALL 2 1` idle on the device's
  NUMA node (devices 0-7: CPUs 48-95,144-191; 8-15: 0-47,96-143). Pin `NEURON_RT_VISIBLE_CORES=4d..4d+3`,
  `taskset` to idle CPUs of that node, `NEURON_NUMA_PIN=0`; `PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:$PATH`.
- Caches, frozen trees and outputs on `/mnt/nvme/difflet-vexp/` (root disk is 98% full). CPU-only jobs in the venv
  set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` (else they hold `/dev/neuron0`).

## Review Focus

1. An nkilib without `allow_vector_exp` (older trn2 stacks): every attention call must keep working on the
   default op, not raise `TypeError` at trace time. Test in Task 1 (scalar-exp kernel `None` -> default kernel).
2. LSE layout: the kernel returns softmax stats tiled `[B*H, 128, n_tile_q]`; a wrong transpose silently
   corrupts ring merges. Test in Task 1 (`_flash_fwd` LSE vs `_compute_attention_nki` on the same fake stats).
3. GQA-shaped calls (`H_kv != H`): the mirror must reshape k/v by `H_kv` as `_compute_attention_nki` does. Test in Task 1.
4. A NEFF cache or NKI trace cache serving the old VectorE kernel after the change. Checked on device in Task 3
   (default-arm outputs must equal section 9's ScalarE outputs, not the shipped ones).
5. CPU CI has no Neuron SDK: the new torch-neuronx test must skip cleanly and be listed in
   `scripts/neuron-skips.txt`. Checked in Task 1 Step 5 with `/opt/aws/neuron/bin` off `PATH`.

---

### Task 1: ScalarE exp by default in the neuron attention ops

**Files:**
- Modify: `difflet/backends/neuron/ops_impl/attention.py` (`attention` :70-130, `_flash_attention` :152,
  `_load_flash_bounded_op` :161, `_load_flash_kernel` :245, `_load_flash_lse_op` :261, `_flash_kernel` :310, module docstring)
- Test: `tests/unit/backends/test_neuron_attention_c3.py`, `tests/unit/backends/test_neuron_compile_c9.py:790-800`
- Modify: `tests/manual/check_neuron_compile_c9.py:80`, `scripts/neuron-skips.txt`

**Interfaces:**
- Produces: `attention(..., vector_exp: bool = False, ...)` (default flipped; now also governs the unbounded path).
- Produces: `_flash_fwd(q, k, v, *, is_causal: bool, scale: float, cache_softmax: bool) -> Tensor | tuple[Tensor, Tensor]`
  on `[B, H, L, d]`: `_compute_attention_nki`'s body (reshape to `[B*H, L, d]` / `[B*H_kv, S, d]`, grid
  `(get_logical_neuron_cores(),)`, `tp_q=True, tp_k=True, scale, causal_mask=is_causal, cache_softmax`) plus
  `allow_vector_exp=False`; with `cache_softmax` it returns `(out, lse)` with `_compute_attention_nki`'s LSE math
  (`-(neg_max + log(out_sum_recip))`, fp32, reshape `[B, H, p, t]`, `transpose(-2, -1)`, reshape `[B, H, -1]`).
  It reads `wrapped_flash_fwd` and `get_logical_neuron_cores` from the `decompositions` module at call time.
- Produces: `_flash_scalar_exp_kernel()` -> the nki_op `difflet_neuron::flash_attention_scalar_exp` wrapped to the
  SDPA kernel's signature `(q, k, v, *, is_causal, scale, training)`, or `None` (no torch-neuronx, or nkilib's
  `attention_cte` has no `allow_vector_exp` parameter). Resolved once at import, like `_FLASH_KERNEL`.
- `_load_flash_bounded_op(vector_exp=False)` returns `None` under the same no-knob condition (the existing
  `_FLASH_BOUNDED_SCALAR_EXP_OP or _FLASH_BOUNDED_OP` fallback then applies).
- `difflet_neuron::flash_attention_lse` calls `_flash_fwd(..., cache_softmax=True)` instead of `_compute_attention_nki`.

- [ ] **Step 1: Write the failing tests** in `test_neuron_attention_c3.py`
  - `fake_kernel` fixture also patches `_flash_scalar_exp_kernel` to the same recorder (existing asserts unchanged).
  - `test_unbounded_default_takes_the_scalar_exp_kernel`: separate recorders for `_flash_kernel` and
    `_flash_scalar_exp_kernel`; default call -> only the scalar recorder, with
    `dict(shape=(1, BH, SQ, D), is_causal=False, scale=0.2, training=False)`; `vector_exp=True` -> only the plain one;
    both outputs `torch.equal`.
  - `test_unbounded_falls_back_to_the_default_kernel_without_the_knob`: `_flash_scalar_exp_kernel` -> `None`;
    default call goes to `_flash_kernel`'s recorder.
  - Rewrite `test_vector_exp_false_routes_the_bounded_call_to_the_scalar_exp_op` as
    `test_bounded_default_takes_the_scalar_exp_op`: default -> scalar op; `vector_exp=True` -> `_FLASH_BOUNDED_OP`.
  - `test_flash_fwd_mirrors_compute_attention_nki` (`pytest.importorskip("torch_neuronx.neuron_dynamo_backend.decompositions")`),
    parametrized over `cache_softmax` in {False, True} and `(H, H_kv)` in {(4, 4), (4, 2)}: patch
    `decompositions.wrapped_flash_fwd` with a recorder supporting `[grid]` that returns fixed fake tensors
    (out `[B*H, L, d]`; for stats `neg_max`, `out_sum_recip` shaped `[B*H, 128, L // 128]` with distinct values)
    and `get_logical_neuron_cores` -> 2; call `_compute_attention_nki(q, k, v, False, 0.2, cache_softmax)` and
    `_flash_fwd(q, k, v, is_causal=False, scale=0.2, cache_softmax=cache_softmax)`; assert the recorded grids are
    equal, tensor args have equal shapes, kwargs equal except `_flash_fwd`'s extra `allow_vector_exp=False`, and
    outputs (and LSE) are `torch.equal`.
  - In `test_neuron_compile_c9.py::test_kernel_attention_path_traces_as_one_graph`, also patch
    `_flash_scalar_exp_kernel` with `_stub_kernel_getter`.

- [ ] **Step 2: Run them, expect failures**

Run: `DIFFLET_BACKEND=cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=. taskset -c <idle CPUs> /home/ubuntu/venvs/native-fp8/bin/python -m pytest tests/unit/backends/test_neuron_attention_c3.py tests/unit/backends/test_neuron_compile_c9.py -q -p no:cacheprovider`
Expected: the new tests FAIL (`_flash_scalar_exp_kernel` / `_flash_fwd` not defined; default still routes to the VectorE op).

- [ ] **Step 3: Implement** the Interfaces above. `_flash_attention(q, k, v, *, scale, causal, vector_exp)` picks
  `(not vector_exp and _flash_scalar_exp_kernel()) or _flash_kernel()`. Update `attention`'s docstring (the knob now
  covers both paths; trn3 only; numbers from section 9) and the module docstring. The H3-specific wording moves
  to Task 2. `tests/manual/check_neuron_compile_c9.py:80`: the op name the default path now emits
  (confirm the graph's name for the new op on device in Task 3).

- [ ] **Step 4: Run the tests, expect PASS** (same command as Step 2).

- [ ] **Step 5: CI-shaped check and the skip list.** Run the two files again with `/opt/aws/neuron/bin` removed
  from `PATH` and in a venv without torch-neuronx if one exists (else confirm the `importorskip` reason); add
  `tests/unit/backends/test_neuron_attention_c3.py::test_flash_fwd_mirrors_compute_attention_nki[...]` ids to
  `scripts/neuron-skips.txt`.

- [ ] **Step 6: Commit and push**

```bash
git commit -m "perf(neuron): attention_cte exp on ScalarE (allow_vector_exp=False) by default"
git push origin feat/attn-scalar-exp-default
```

### Task 2: MiniMax-H3 drops its own opt-in

**Files:**
- Modify: `difflet/backends/neuron/minimax_h3/transformer.py:136-140` (field `attention_vector_exp`), `:412-422` (binding)
- Test: `tests/unit/backends/test_neuron_minimax_h3_l1.py:160-185`

**Interfaces:**
- Consumes: Task 1's default (`vector_exp=False` everywhere).
- Produces: `NeuronMiniMaxH3Config` without `attention_vector_exp`; H3 processors keep the plain `difflet.ops.attention`.

- [ ] **Step 1: Rewrite the test** as `test_neuron_h3_processors_use_the_plain_attention`: every processor's
  `_attention` is not a `functools.partial`; `NeuronMiniMaxH3Config(**tiny, attention_vector_exp=True)` raises `TypeError`.
- [ ] **Step 2: Run it, expect FAIL.** `pytest tests/unit/backends/test_neuron_minimax_h3_l1.py -q`
- [ ] **Step 3: Delete** the field, its comment and the binding block (H3's 66.4 -> 54.0 ms note moves into
  `attention`'s docstring in Task 1 if not already there). `grep -rn attention_vector_exp` must return nothing.
- [ ] **Step 4: Run it, expect PASS;** then the CPU unit suite pinned to idle CPUs
  (`scripts/cpu-gate.sh -n <N>` or the Global Constraints command over `tests/unit`, `--basetemp=/mnt/nvme/difflet-vexp/pytest`,
  `TMPDIR` on NVMe; delete the basetemp after). Expected: only the two known pre-existing failures
  (`hv1_app registry`, `qwen_image_q4 e2e`), checked against the same suite on d70faffc if they differ.
- [ ] **Step 5: Commit and push** `refactor(neuron/minimax_h3): the ScalarE exp is the backend default now`.

### Task 3: On device, kernel level

**Files:**
- Create: `tests/manual/probe_neuron_attention_scalar_exp.py`

**Interfaces:**
- Consumes: `attention(..., vector_exp=...)`, `_attention_with_lse` (Task 1); `difflet.backends.neuron.compile.compile_region`.
- Produces: per shape and arm a JSON (`median_ms`, `spread`, per-run ms, rel L2 / cosine vs fp32 per head and overall,
  output md5) and `<arm>/<shape>.out.pt`, under `/mnt/nvme/difflet-vexp/kernel/`.

The probe copies `scripts/mx_attention/probe.py`'s method (branch `feat/mxfp8-sage-attention`): inputs
`torch.Generator().manual_seed(0)`, `randn(bh, s, d).bfloat16()` for q, k, v in that order, q and k times `--peak`,
bounded rows `bound_min=0, bound_max=valid_k`; scale `d ** -0.5`; the region
`attention(q, k, v, scale=scale, bound_min=lo, bound_max=hi, tp_q=True, tp_k=True, vector_exp=<arm>)` through
`compile_region(..., backend="neuron")`; one saved first output, one warm call, `--calls` calls and one sync,
in `--runs` worker processes; fp32 reference one head at a time. Arms: `scalar` (default) and `vector`
(`vector_exp=True`). `--lse` times and scores `_attention_with_lse` instead (out and lse vs fp32 logsumexp).
Shapes (section 9's `shapes.json`): flux_1024 6x4608x128, qwen_1024 6x5120 valid 4609, ltx2_video 8x6144,
wan_81f 10x32760, hv15_121f 8x51275 valid 50283, hv1_L 6x67576 valid 67449.

- [ ] **Step 1: Bit identity.** One lane per shape, six lanes in parallel on idle devices. `scalar` outputs ==
  `/mnt/nvme/difflet-mxattn/probe/rebalance/scalar_exp/<shape>.out.pt`, `vector` outputs ==
  `/mnt/nvme/difflet-mxattn/probe/rebalance/shipped/<shape>.out.pt` (`torch.equal`). A mismatch is a bug (stale
  cache, wrong argument): stop and debug.
- [ ] **Step 2: Timing.** Per shape, `vector` then `scalar` on the same device and CPU set, 3 processes, 40 calls
  (flux, qwen) or 20 (rest). Faster = `scalar / vector < 1 - J`, `J = max(2%, vector's spread)`.
  Expected from section 9: 0.79-0.93.
- [ ] **Step 3: Peaked case** (`--peak 2`, scores std ~4, never measured before): rel L2 vs fp32 per head, both arms,
  all six shapes. Report; non-finite or a rel L2 above 2x the vector arm's is a stop.
- [ ] **Step 4: LSE path** (`--lse`) at the shapes the joint ring feeds it: Qwen-Image cp2 text partial and image
  partial (q 12 heads at tp2 x 2048 image rows; k/v text 512, image 2048, d 128; read the real per-rank shapes from a
  `--cp-degree 2 --cp-mode ring` trace first). Both arms: out rel L2 and |lse - fp32 lse| max. Report.
- [ ] **Step 5: Op name.** Run `tests/manual/check_neuron_compile_c9.py` on one device; it must count the new op.
- [ ] **Step 6: Commit and push** the probe: `test(neuron): kernel-level probe of attention's ScalarE exp default`.

### Task 4: On device, model level

**Files:**
- Create: `docs/verification/2026-11-16-attention-scalar-exp.md` (Task 3 and Task 4 evidence)

Driver: `scripts/quant_bench/` (`bench.sh`, `quality.py`, `report.py`, `stamp.py`) and
`scripts/mx_attention/{run_arms.sh,steps.py}` from `feat/mxfp8-sage-attention` at 5d5b83c5, copied unchanged into
both frozen trees: `git archive d70faffc` -> `/mnt/nvme/difflet-vexp/tree_before`, `git archive <Task 2 commit>` ->
`/mnt/nvme/difflet-vexp/tree_after` (the driver derives `REPO` from its own location, so each copy runs its own
tree; only `difflet/` differs). Arm `bf16`, `ROOT=/mnt/nvme/difflet-vexp/<before|after>`; per model and tree:
`warm` (2 steps, compiles), `timed` (one process, two identical requests; DiT step = median of the orchestrator's
per-request step medians, spread = (max - min) / median), `gen2` (new process, determinism).

| tier | key (`bench.sh args_for`) | attention path | why |
|---|---|---|---|
| A | `qwen_image` 1024x1024 tp4 | bounded | text-padded MMDiT; kernel 0.925 |
| A | `flux` 1024x1024 tp4 | unbounded | kernel 0.897 |
| A | `wan21_official` 480x832x81 tp4 | unbounded | the wan_81f shape; kernel 0.816 |
| A | `qwen_image` tp2 cp2 `--cp-mode ring` | joint ring + LSE op | the only Task 1 path the kernel table lacks |
| B | `hv15_official` | bounded | kernel 0.787 |
| B | `ltx_2` 512x768x121 tp4 | unbounded (+ audio d=64, not routed) | kernel 0.857 |
| B | `hunyuan_video_L` 544x960x129 | bounded | kernel 0.789 |
| C | `wan22_official`, MiniMax-H3 768p | as Wan / bounded+unbounded | same kernels; H3 changes only its unbounded and VAE calls |

- [ ] **Step 1: Lanes.** One lane per model (tp4 = one device at LNC=2), as many in parallel as idle devices allow;
  in each lane `before` then `after` on the same device and CPU set, `run_arms.sh` order
  (`warm before`, `warm after`, `timed before`, `timed after`, `gen2 after`).
- [ ] **Step 2: Speed.** after / before DiT step; faster = `< 1 - J`, `J = max(2%, before's spread)`. Also e2e warm.
- [ ] **Step 3: Quality.** after vs before: PSNR / SSIM of images or frames and latent cosine (`quality.py`);
  vs CPU fp32 where a reference for the same config already exists (Qwen-Image, FLUX; no new CPU fp32 runs).
  Non-finite output, noise, or a `gen2` md5 different from `timed_1` is a bug.
- [ ] **Step 4: Evidence doc** with both tables (Task 3, Task 4), devices, CPU sets, trees, md5s, and the
  trn2 note (knob is a no-op there, unverified on trn2 hardware); commit and push
  `docs(verification): attention_cte ScalarE exp default, kernel and model evidence`.

### Task 5: Review and PR

- [ ] **Step 1:** Whole-branch review (`/code-review` or a fresh reviewer) on the diff vs main; fix findings, one commit each.
- [ ] **Step 2:** Draft PR to `main` with the speed and quality tables; the trn2 note; the unchanged sites and why
  (trainium backend, ring kernel, `DIFFLET_NEURON_ATTENTION=sdpa`).
