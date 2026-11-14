# Handover: DiT block profile + MXFP8 attention (paused 2026-11-14)

All work on `feat/mxfp8-sage-attention` was paused at the user's request
("所有Task暂停下，并写好交接文档，这台机器需要关闭维护下，晚点我们再重新跑。") because the host is being
shut down for maintenance. This document is what a resumed session needs. Everything below is
committed and pushed except where a path says otherwise.

## 1. Where things stand

- **Branch:** `feat/mxfp8-sage-attention`, branched from `feat/mxfp8-video-models` at `9fddd88f`.
  The pause point is `c0e4aaff` plus this document's own commit. Everything is on `origin`.
  Worktree: `/home/ubuntu/Difflet/.claude/worktrees/mxfp8-sage-attention`. The tree is clean, and
  no device job of this work is running.
- **Plans:**
  - `docs/plans/2026-11-13-dit-block-profile.md` (Plan A, profile);
  - `docs/plans/2026-11-13-mxfp8-attention.md` (Plan B, MXFP8 attention).
- **Evidence:** `docs/verification/2026-11-13-dit-block-profile.md`. Sections 1-6 cover Plan A;
  sections 7-12 cover Plan B and the bf16 findings.
- **Plan A page:** https://claude.ai/artifact/Em8C2DS8wBHcEPHAfk6rzy (private, "DiT Block Profile").
  Plan B results are not on the page. Plan B's page step belonged to its Task 6, which never ran.
- **Execution ledger (git-ignored, on EBS):**
  `.superpowers/sdd/2026-11-13-mxfp8-attention/`:
  - `progress.md` is the full ledger, with every report, review and ruling in order;
  - `rulings.md` holds rulings BR1-BR27;
  - each task has a brief, a report and a review package.

  Plan A's ledger is `.superpowers/sdd/2026-11-13-dit-block-profile/`. A copy of both is in the
  evidence tarball (section 5).

## 2. Results so far (all reviewed by an independent reviewer unless marked)

| step | result |
|---|---|
| Plan A: 7 models x bf16/linear MXFP8 block profiles | done (doc sections 1-6, page). Linear MXFP8 block 0.74-0.98x bf16; attention-heavy video blocks are VectorE-bound in attention. |
| Plan B Task 1-2: CPU reference of MXFP8 attention, one-core probe, attention_cte bf16 baseline | done (sections 7) |
| Task 3: fork of attention_cte, PV on `nc_matmul_mx` | accurate (rel L2 vs mirror 1.3e-3..2.4e-3) but **G1 fails** on all six shapes (1.05-1.20x shipped bf16), section 8 |
| Task 4-pre: attention_cte `allow_vector_exp=False` (exp on ScalarE) | **bf16 7.5-21% faster on all six shapes**, accuracy unchanged on flat inputs; call-path control clean (section 9). By BR19 this is the baseline every gate must beat. |
| Task 4a: fork with exp on ScalarE, G1 re-read, QK row tiling | G1 fails again (PV 1.08-1.51x); four `nc_matmul_mx` in the four PE row quadrants are feasible (section 10) |
| Task 4b: QK^T on `nc_matmul_mx` with row tiling | **G2 fails**: QK-only 1.03-1.21x, QK+PV 1.17-1.54x (section 11). Device MX matmul error ~2e-4 of score RMS (hardware; BIRSim/simulator show 1e-7). |
| Plan B gate decision (BR23) | **Plan B stops**: Tasks 5-7 (op/switch, model runs, default flip) are not executed. Fork code and probes stay on the branch as evidence. |
| const-max probe (BR24): nkilib `attention_const_max`, bf16, unmodified | **0.617 / 0.651 / 0.701x** of the rebalanced cte on flux_1024 / ltx2_video / wan_81f; error 1.5-2.1x cte's (flat) and 7.5-8x (peaked); x4 inputs overflow. Wan: TensorE 99.4%, VectorE 73.7% of span, so by BR24's rule FP8 attention on trn3 closes; a faster TensorE could cut at most ~26% more. Section 12. **Review paused, partial** (see section 4). |

Why FP8 attention lost on the attention_cte structure: at wan_81f the rebalanced kernel's TensorE
(45.3 ms busy) sits just above two other floors, the DMA P-transpose queue (43.8 ms) and VectorE
(42.6 ms, of which the MM1 PSUM evict is 39.4 ms). Any FP8 matmul can gain at most 0-5% there, and
each FP8 variant adds layout or quantize work to one of those floors. A PE activity throttle
(HAM `activity_1`, PE allowed 4 of 8 cycles) is real but does not cap MX in general: linear MXFP8
spends only 5-9% of its time in throttle windows.

## 3. Decisions

**Made by the user:**
- Gates are decided on speed alone ("只要有速度提升就可以有").
- Runs compare bf16 vs both only; HunyuanVideo-1.5 compares bf16 vs attn.
- **2026-11-14: make `allow_vector_exp=False` Difflet's bf16 attention default ("好的").**
  This is Task VEXP. It is not started (section 4).

**Open, the user's call:**
1. Adopt bf16 `attention_const_max` (30-38% faster kernel) for unbounded models? Recommendation:
   if yes, start with FLUX, whose 57 layers all pass the docstring's bound rule. Wan fails 4/40
   layers and LTX-2 fails 3/48, which need a per-layer fallback to cte. It needs a plan of its
   own with model-level quality evidence, since kernel error is higher and overflow is possible.
2. FP8 on top of const-max (at most ~26% more)? Only meaningful if 1 is adopted.
3. Branch finishing: keep the MX fork code (evidence) or merge only the probes and docs. To be
   decided after the whole-branch final review.
4. Carried from before this branch:
   - fix the evidence doc's 7e VAE misattribution on `feat/mxfp8-video-models`;
   - delete the Qwen schema-2 quantized dirs (216 GB) under `/home/ubuntu/difflet-cache-fp8`;
     the root disk is 97% full.

## 4. Paused work and how to resume, in order

1. **Task VEXP: `allow_vector_exp=False` by default.**
   - Brief: `.superpowers/sdd/2026-11-13-mxfp8-attention/task-vexp-brief.md`. Nothing was changed:
     the implementer was stopped while reading.
   - Its call-site map is in `task-vexp-report.md`:
     - `difflet/backends/neuron/ops_impl/attention.py` needs three changes. The unbounded flash
       path must call `wrapped_flash_fwd` directly, because torch-neuronx's SDPA wrapper cannot
       pass the knob. The other two are the bounded path and the LSE path (also a direct
       `wrapped_flash_fwd` call, with `cache_softmax=True`).
     - `difflet/backends/trainium/ops_impl/attention.py:48, :79, :314, :395` take the knob.
     - Ring attention there uses a different kernel, which has no such option.
     - `attention_base.py` is vendored and imported by nothing, so it stays unchanged (BR27).
   - Risk: an nkilib without `allow_vector_exp` would raise on the keyword. Every venv on this
     host has it.
   - The three attention files are identical on `main` and this branch, so keep the code commit
     cherry-pickable.
   - Its kernel-level bit-identity references are on instance storage
     (`/mnt/nvme/difflet-mxattn/probe/rebalance/scalar_exp/<shape>.out.pt`). If they were wiped,
     regenerate them first with, per shape in `scripts/mx_attention/shapes.json`:

     ```bash
     probe.py --shape <s> --impl cte --cte-no-vector-exp --runs 1 --save-output --out <dir>
     ```

   - Then run the model-level check: two models, before and after, same device and CPU set.
2. **Finish the const-max review (section 12, commit `c0e4aaff`).**
   - Already checked and reproduced: timing, arm parity, `softmax_max`, the BR24 arithmetic.
   - Not checked:
     - the raw engine JSONs and their like-for-like with Task 4b's capture;
     - LTX-2/Wan accuracy, the x4 overflow count and `emulate.py`;
     - the whole real-model bound table (tensors, layer counts 40/57/48, Sk, d, formula,
       all-head norms, sigma);
     - the capture and accuracy logs.
   - Doc fixes it asked for:
     - name the shapes in "5/4/3 heads per core" (wan 5, ltx2 4, flux 3);
     - say the peaked probe used the sigma-1 max, not the per-model max;
     - sigma > 1 is not the docstring's precondition (std(Q), std(K) <= 1), so the layer counts
       overstate it;
     - add the BR26 adoption statement;
     - fix the "~1 ms fixed overhead" line.
3. **Whole-branch final review**, then the branch-finishing decision with the user (section 3,
   item 3).

## 5. Data: what survives a stop/start

`/mnt/nvme` is a RAID of EC2 NVMe **instance storage**. It survives a reboot and is **wiped by a
stop/start**. The EBS root (`/`) survives.

- **In git (safe):** all code, tests, plans, evidence docs, and Plan A's curated assets
  (`docs/verification/assets/2026-11-13-dit-block-profile/`).
- **On EBS (safe across stop/start):**
  - `/home/ubuntu/difflet-mxattn-evidence-2026-11-14/mxattn-evidence.tar.gz` holds 5,263 small
    files: every JSON/log/script from `/mnt/nvme/difflet-mxattn` (probe results, run logs, capture
    summaries, the scratch analysis tools of Tasks 3b-4b and the const-max probe), both SDD
    ledgers, and the adversarial reviews' scripts. `filelist.txt` beside it lists them.
    Extract with `tar xzf ... -P`, which restores the absolute paths.
  - The worktree, including the git-ignored ledger.
  - The compile caches under `/home/ubuntu/difflet-cache-{video,fp8,cold}`.
- **Lost on stop/start (regenerable):**
  - `/mnt/nvme/difflet-mxattn` (26 GB: probe output tensors, NEFFs, capture parquets, the
    probe's NKI/compile cache);
  - `/mnt/nvme/difflet-prof` (108 GB: Plan A's raw captures);
  - `/mnt/nvme/neuron-compile-cache`;
  - `/mnt/nvme/difflet-quant-bench-hub` (292 GB model copy).

  A resumed probe recompiles cold: about 20 min per HunyuanVideo shape, minutes for the others.

## 6. Environment and rules for a resumed session

- **Python and SDK:**

  ```bash
  ~/venvs/native-fp8/bin/python
  export PATH=/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin:/opt/aws/neuron/bin:$PATH
  PYTHONPATH=<worktree>
  ```

  The NKI trace cache may be on (`DIFFLET_NKI_TRACE_CACHE=1`); it re-checks the kernel file's
  contents.
- **Unit tests** (never the full suite):

  ```bash
  DIFFLET_BACKEND=cpu PYTHONPATH=. ~/venvs/native-fp8/bin/python -m pytest <file> -q -p no:cacheprovider
  ```

  Before a commit, run `tests/unit/backends/test_neuron_mx_attention_kernel.py`,
  `tests/unit/quant/` and `tests/unit/scripts/`. One known flaky test,
  `test_mx_attention_probe.py::test_with_an_mx_flag_the_reference_is_the_task1_mirror[tiny-flags0]`,
  fails at 8 torch threads with a one-ulp difference. The VEXP brief fixes it.
- **Probe:**

  ```bash
  scripts/mx_attention/probe.py --shape <s> --impl {cte,fork,constmax} [--pv-mx] [--qk-mx] [--cte-no-vector-exp] [--peak 2] --runs 3 --calls <20|40> --out <dir>
  ```

  Use 40 calls for flux_1024 and qwen_1024, because host dispatch is not hidden at 20.
- **Shared host (BR4):**
  - Check `fuser /dev/neuron<d>` before every device command, and use only idle devices.
  - One logical core per probe: device d = cores 4d..4d+3 at LNC=2.
  - Pin `NEURON_RT_VISIBLE_CORES` and `taskset` to the device's NUMA node: devices 0-7 use
    48-95,144-191; devices 8-15 use 0-47,96-143.
  - Other sessions (e.g. ubuntu-1d, ubuntu-9d) borrow devices 8-15 as a contiguous set for SGLang
    and ask first. Answer, vacate, and wait for their "free".
  - Never delete caches without the user's approval.
- **Process:** subagent-driven development with one implementer per task and an independent
  review per task. Push every commit to `origin/feat/mxfp8-sage-attention`.
