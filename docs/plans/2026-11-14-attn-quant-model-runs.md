# `--attn-quant` and the model runs: Task 5b + Task 6 (plan)

**Goal:** ship attention quantization as an opt-in launch option, `--attn-quant {qk,pv,qk+pv}`,
then measure `bf16` against `both` on every model at its Plan A profile shape. `both` means
`--quant mxfp8 --attn-quant qk+pv`. The measurements are step time, the Plan A block profile, and
quality.

**Spec:** the user's decisions of 2026-11-14 (ledger BR34, BR35 and later):
- QK and PV are both implemented and selected per launch, never as a default.
- `--attn-quant` is independent of `--quant`.
- Task 6 runs on all models of the original plan, and its `both` arm uses `qk+pv`.
- No short-shape pairs: every model, the long videos included, runs at its Plan A profile shape.
- The arms are the original Task 6's.
- Order: Task 5b, then its review, then Task 6, one model at a time, with a commit, a push and a
  page update after each model.

Parent plan: `docs/plans/2026-11-13-mxfp8-attention.md`. Its Global Constraints still apply:
routing set, padding as amended by the parent's addendum 2, switch-off identity, seed 42, LNC=2,
steps and flags from `bench.sh`.

## Task 5b: finish `--attn-quant`

The work is in `.superpowers/sdd/2026-11-13-mxfp8-attention/task-5b-wip.patch` (paused, 14
files). The brief is `task-5b-brief.md`. Already done and checked:
- the flag;
- the variant table: `qk`, `pv` and `qk+pv` map to `(qk_mx, pv_mx)`;
- the env check;
- the export before any backend import;
- forwarding to stage lines and DP workers;
- `attn_quant` in the DiT stage cache keys, with off keeping today's key;
- off md5s: Qwen `7737239b` and Wan `39bac229`;
- routed output equal to the fork bit for bit at qwen_1024 for all three variants and at wan_81f
  for `qk`;
- four smoke generates;
- separate stage directories and NEFFs per variant.

Remaining:
1. Check the routed call against the fork bit for bit at wan_81f for `pv` and `qk+pv`.
2. Run the unit suites (`tests/unit/backends`, `quant`, `scripts`, CLI). Show that any failure
   is one of the three that also fail on the base commit.
3. Fill the Wan rows of evidence section 14.
4. Commit `feat(cli): --attn-quant {qk,pv,qk+pv} selects MXFP8 attention per launch` and push.

**Review:** one task review over `336b8785..HEAD` (Task 5 and 5b together). It covers:
- switch-off parity;
- the routing cases;
- K smoothing, applied only when QK^T is MXFP8, and padding;
- registration;
- validation;
- cache keys;
- forwarding;
- Task 5's two edits outside the brief, in LTX-2's TP processor and the trainium backend.

## Task 6: model runs, `bf16` vs `both`

### Configurations (Plan A keys and shapes, tp4 unless stated)

| key | shape | arms | fp32 reference at this shape |
|---|---|---|---|
| `qwen_image` | 1024x1024 | bf16, both | yes |
| `flux` | 1024x1024 | bf16, both | yes |
| `ltx_2` | 512x768x121 | bf16, both | yes |
| `wan21_official` | 480x832x81, CFG 5.0 | bf16, both | no |
| `wan22_official` | 480x832x81, CFG 4.0/3.0, 40 steps | bf16, both | no |
| `hunyuan_video_L` | 544x960x129 | bf16, both | no |
| `hv15_official` | 480x848x121, CFG 6.0, tp2 x CFG2 | bf16, attn (`--attn-quant qk+pv` alone) | no |

Steps, flags and prompts come from `scripts/quant_bench/bench.sh`; seed 42; LNC=2. Linear-alone
is not rerun. G3 reads it from the evidence doc
(`docs/verification/2026-11-07-mxfp8-video-models.md`, sections 7f / 11c / 12c). There is one
exception: one linear-alone run for a model when `both` and linear-alone are within 2J of each
other, or when this session's bf16 step differs from the evidence's by more than 1%.

### Protocol, per model

Each model gets one lane: one idle device (two for HV1.5), with `NEURON_RT_VISIBLE_CORES` and
`taskset` pinned.
1. Compile and warm both arms. Compiles run in parallel across lanes and are untimed.
2. Timed: bf16, then the quantized arm, back to back on the same device and CPU set. Two requests
   each; the medians of their per-step times give the step time and the spread.
3. A second generate of the quantized arm. Its md5 must equal the first (determinism).
4. A Plan A block capture of the quantized arm (`scripts/profile_dit/capture.sh` with arm
   `both`, or `attn` for HV1.5). It is compared with Plan A's bf16 capture of the same key.

### Metrics

**Speed**
- DiT step time: the warm median ms per step, its spread, and `both / bf16`.
- G3, reported only: `both / bf16 < linear / bf16 (evidence) - J`, with J = max(2%, spread).
  For HV1.5: `attn / bf16 < 1 - J`. Nothing becomes a default.
- Block profile, from Plan A's analysis:
  - per-engine busy ms and share of the block: TensorE, VectorE, ScalarE, GpSimd, HBM DMA,
    PSUM read-out;
  - the binding engine;
  - per-component ms: attention, MLP, norms and modulation, projections;
  - attention phases: MM1 (QK^T), exp, MM2 (PV), and the `quantize_mx` time of Q/K/V/P;
  - V's share of the attention component.

**Quality**, scored with `scripts/compare_difflet_runs.py` on the decoded output (video: mean
over frames plus the worst frame):
- PSNR (dB);
- SSIM;
- LPIPS (AlexNet);
- latents rel L2 and cosine.

Each model gets these columns:
- `both vs bf16`;
- `linear-alone vs bf16` (from the evidence doc);
- the bf16 floor (bf16 vs bf16, where the evidence doc has one);
- `both vs fp32` (`scripts/fp32_ref/score.py`), for Qwen-Image, FLUX and LTX-2 only, with the
  evidence doc's bf16-vs-fp32 and linear-vs-fp32 beside it.

Two rows carry a caveat:
- The Wan 2.2 81f and HV1.5 121f rows are marked "below the bf16 floor's resolution, not a
  quality reading". For example, HV1.5's two bf16 runs already sit 0.911 / 0.897 apart in latent
  cosine.
- Long-shape cosine is reported only.

A model's quantized output is **broken** (a bug, fixed before its numbers count) when its
latents are non-finite or it shows noise on screen.

### Outputs

- After each model: its numbers go to evidence section 15
  (`docs/verification/2026-11-13-dit-block-profile.md`), then a commit and a push. The Plan A
  page (https://claude.ai/artifact/Em8C2DS8wBHcEPHAfk6rzy) gets a dataset with the step times,
  the G3 row, the attention phase split and the quality columns. The controller uploads it.
- At the end: the G3 table, then a whole-branch final review, then the branch-finishing decision
  with the user.

### Budget and risks

- Device time: one cold compile per arm and key. Measured before: HV1 ~40 min, Wan 11-16 min,
  HV1.5 ~24 min; Qwen, FLUX and LTX-2 are shorter. Plus three timed generates and one capture per
  model. With 3-4 lanes in parallel this is about 6-8 h wall clock, plus Task 5b's ~1-2 h. It
  depends on how many devices are idle; devices 0 and 6 are held by orphans of another project.
- Expected result: the kernel-level QK+PV attention is 1.17-1.54x bf16. So `both` is likely
  slower than linear-alone, more so where attention is a larger share of the block (Wan, HV1,
  HV1.5). Quality is the open question this run answers.
- Disk: new caches go on `/mnt/nvme` (instance storage, lost on stop/start). The EBS root has
  ~63 GB free. No cache is deleted without the user.
