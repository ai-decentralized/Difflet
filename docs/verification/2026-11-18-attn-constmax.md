# E1: nkilib `attention_const_max` (bf16) at model level, FLUX.1-dev then LTX-2 (trn3)

Plan: `docs/plans/2026-11-18-attn-constmax-sparse-experiments.md`, section E1. Branch
`exp/attn-constmax-sparse`. Prior evidence: `docs/verification/2026-11-13-dit-block-profile.md`
section 12 on `feat/mxfp8-sage-attention` (kernel level: 0.617x the rebalanced cte at
`flux_1024`, 1.5-2.1x its error on flat inputs, 7.5-8x peaked, overflow at x4).

Every number below is **measured** on this host unless it is marked *inferred* or *evidence*
(quoted from an earlier doc, not re-run).

## 1. Summary

| | FLUX.1-dev 1024x1024, tp4, 50 steps | LTX-2 512x768x121, tp4, 50 steps |
|---|---|---|
| warm DiT step, bf16 (shipped cte) | 220.5 ms | 1282.6 ms |
| bf16 + const-max | **200.0 ms (0.907x)** | **1217.3 ms (0.949x)**, 45 of 48 video attn1 routed |
| `--quant mxfp8` + const-max | **162.2 ms (0.736x)** | not run (plan: two arms) |
| const-max vs bf16 (PSNR / SSIM / LPIPS / latent cos) | 41.40 dB / 0.9931 / 0.0024 / 0.99957 | 27.63 dB / 0.9307 / 0.0292 / 0.99947 (video) |
| const-max vs CPU fp32 (bf16 arm beside it) | 23.31 dB (bf16 23.36) | 16.54 dB (bf16 16.55) |

## 2. Method

**Hook (opt-in, `DIFFLET_EXP_ATTN_CONSTMAX=1`).**

- `difflet/backends/neuron/ops_impl/attention.py`: `attention()` takes `constmax_sigma` (a
  `[128, 1]` tensor, the layer's score sigma). With the env set, a call that carries one and is
  unbounded, non-causal, bf16 (q, k and v), d = 128, without a mask and on the flash-kernel path
  goes to `difflet_neuron::attention_const_max`. That op is registered with `nki_op` like the
  bounded op and wraps nkilib 2.14 `experimental/attention/attention_const_max.py`, unmodified:
  `kernel[(lnc,)](q, k, v, softmax_max=..., softmax_scale=scale, q_layout=SEQ_MAJOR,
  k_layout=SEQ_MAJOR)` on `[B*H, S, d]`. Every other call is unchanged. With the env unset the op
  is not even registered.
- `softmax_max` = 0.88 sigma sqrt(2 ln Sk), the kernel docstring's rule, at the call's own Sk. It
  is computed in the graph from the `[128, 1]` sigma and passed in the kernel's `[128, 1]` fp32
  **tensor** form, not as a Python scalar. Why: a per-layer scalar would be baked into each
  block's trace, and `compile_blocks` shares one graph per block class (Dynamo would recompile
  per layer and hit its limit of 8). As a buffer, sigma is a graph input. The two FLUX block
  NEFFs differ from bf16's (the cache diff: bf16 `40a2c51f…` / `cffdd7f3…`, const-max
  `d73ca1d5…` / `e0de2689…`), and there are still only two.
- **Threading.** FLUX: `NeuronFluxTransformer2DModel.__init__` registers each attention's sigma
  as a non-persistent `[128, 1]` buffer `constmax_sigma`. The non-CP wrapper
  `attention_wrapper_sharded_without_swap` passes it on, and only when it exists. LTX-2: the same
  on `transformer_blocks.{i}.attn1`, passed by the TP processor's self-attention branch.
  Non-persistent buffers are not checkpoint entries, and `build_on_meta` keeps buffers real.
- **Tests (CPU).**
  - `tests/unit/backends/test_neuron_attention_constmax.py`: the route and its softmax_max;
    nothing is routed without a sigma, causal, masked, bounded, fp16, fp32, d = 64, under
    autograd or with the env off; the op is not loaded with the env unset.
  - `tests/unit/models/flux/test_flux_modeling.py::test_constmax_env_threads_the_table_sigma`:
    buffers and kwarg with the env set, neither with it unset.
  - `tests/unit/backends/test_neuron_ltx2_constmax.py`: the LTX-2 allowlist and the env-unset
    case.

**softmax_max table provenance.**

- `scripts/attn_constmax/table.py` is section 12's `bounds.py`
  (`/mnt/nvme/difflet-mxattn/probe/constmax/bounds.py`) with its formulas unchanged:
  - sigma = sqrt(sum(g_q^2 g_k^2) / d);
  - softmax_max = 0.88 sigma sqrt(2 ln Sk);
  - hard_bound = sqrt(d) max|g_q| max|g_k|.
- It reads only the QK-RMSNorm tensors of the hub snapshots: FLUX `3de623fc…`, LTX-2
  `64f65d09…`, the same snapshots `bounds.json` used.
- One change: `bounds.py` reports the worst head of the worst (q norm, k norm) pair. One kernel
  call covers every head and, in FLUX's joint blocks, all four pairs, so the table keeps the max
  over heads and pairs. A layer passes (`ok`) when gap = hard_bound - softmax_max <= 80 at the
  reference Sk (FLUX 4,608, LTX-2 6,144).
- **Cross-check against `bounds.json`.** The script asserts that every one-pair layer matches
  `bounds.json`'s per-call numbers (`sigma_max_over_heads`, `softmax_max_max_over_heads`,
  `call_gap`) to 1e-4: FLUX 38 of 38 single blocks, LTX-2 48 of 48. FLUX's 19 joint layers use
  the max over pairs, which `bounds.json` does not report.
- Committed tables: `difflet/models/flux/attn_constmax.json` and
  `difflet/models/ltx_2/attn_constmax.json`.
  - FLUX: 57 layers, sigma 0.608-2.092 (softmax_max 2.20-7.56 at Sk 4,608), largest gap 42.7
    (`single_transformer_blocks.36`). All 57 pass.
  - LTX-2: 48 layers, sigma 0.288-3.931, and 3 fail: `transformer_blocks.44/45/47.attn1`, gaps
    96.2 / 134.8 / 112.4. These 3 are the allowlist's fallbacks to cte.
- No value was tuned against any output.

**Device rules.**

- Devices 0-7 only, `fuser` idle at every process start (logged in each arm's
  `<mode>.env.txt`).
- One 16-thread CPU slice per process from 48-95,144-191, idle by `mpstat`;
  `NEURON_RT_VISIBLE_CORES`, `taskset` and `NEURON_NUMA_PIN=0`.
- E2 ran block-sparse probes on the host at the same time (load average 19-34). E2's processes
  held `/dev/neuron0`, so every timed arm ran on device 1 (cores 4-7, CPUs 56-63,152-159).

## 3. Gate (before any timing)

### 3a. Kernel probe at the per-rank shape (`scripts/attn_constmax/probe.py`)

- **Shapes.** FLUX: 6 heads x 4,608 x 128 (tp4's per-rank shape), one logical core (device 0,
  core 0). LTX-2: 8 x 6,144 x 128 (device 1, core 4).
- **Inputs.** x ~ N(0, 1), RMS-normalized over d, times the layer's real checkpoint gains:
  - FLUX joint blocks: text rows take `norm_added_q/k`;
  - LTX-2: the 8 heads of the rank that holds the layer's largest-sigma head.
- **Two families.**
  - `indep`: q and k from independent x.
  - `self`: q_i and k_i from the same x_i, so the diagonal score is about
    sum(g_q g_k) / sqrt(d), the peaked case.
- RoPE is not applied. Both routes go through `ops_impl.attention` in a compiled region; the
  reference is fp32 attention of the same bf16 inputs.

| layer | family | sigma | softmax_max | max score | const-max rel L2 (worst head) | cte rel L2 | const-max non-finite |
|---|---|---|---|---|---|---|---|
| FLUX `single.36` (largest gap) | indep | 1.110 | 4.01 | 6.9 | 5.54e-3 (5.62e-3) | 2.30e-3 | 0 |
| | self | | | 16.0 | 3.74e-3 | 1.66e-3 | 0 |
| FLUX `single.25` (largest sigma) | indep | 2.092 | 7.56 | 11.4 | **1.21e-2** (1.23e-2) | 2.02e-3 | 0 |
| | self | | | 25.3 | 2.6e-6 | 2.6e-6 | 0 |
| FLUX `double.18` (joint) | indep | 1.754 | 6.34 | 9.3 | 9.10e-3 | 2.13e-3 | 0 |
| | self | | | 20.9 | 1.42e-3 | 3.88e-4 | 0 |
| FLUX `double.0` (joint) | indep | 0.887 | 3.21 | 3.9 | 2.53e-3 | 2.33e-3 | 0 |
| | self | | | 10.6 | 4.06e-3 | 1.66e-3 | 0 |
| LTX-2 `26.attn1` (largest routed gap) | indep | 2.272 | 8.35 | 12.8 | 1.30e-2 (1.39e-2) | 2.02e-3 | 0 |
| | self | | | 32.1 | 2.04e-3 (1.09e-2) | 1.35e-3 | 0 |
| LTX-2 `46.attn1` (largest routed sigma) | indep | 3.788 | 13.92 | 19.2 | 1.28e-2 (1.38e-2) | 1.84e-3 | 0 |
| | self | | | 45.8 | 6.44e-4 | 6.43e-4 | 0 |
| LTX-2 `45.attn1` (excluded, reference) | indep | 3.882 | 14.27 | 21.5 | 1.35e-2 (1.48e-2) | 1.84e-3 | 0 |
| | self | | | 47.9 | 9.75e-4 | 5.28e-4 | 0 |

- **Gate: passed.** Every output is finite, peaked rows included (max score up to 47.9 against
  a softmax_max of 14.3). At real gains the error grows with sigma, as section 12's bf16-scores
  explanation predicts:
  - about 1.1x cte at sigma 0.89;
  - 2.4x at 1.11;
  - 6x at 2.09 (FLUX's largest);
  - about 7x at LTX-2's 2.3-3.9.
- **Timing.** The probe's per-call times (FLUX ~10 ms, LTX-2 ~23 ms for both routes) are
  dominated by this harness's per-call host overhead, not kernel time. They are not used;
  section 12's duo timings are the kernel-level numbers.
- Raw: `/mnt/nvme/difflet-exp/constmax/probe/probe.json` (FLUX) and `probe_ltx2/probe.json`, with
  `run.log` and `head.txt`.

### 3b. 10-step FLUX smoke (`arms.sh flux cm smoke`, device 2)

- Latents finite: `[1, 4096, 64]` bf16, mean -0.167, std 1.07, |max| 4.53.
- The image is a clean cat on a bench with no noise or artifacts (`flux/cm/smoke.png`, md5
  `2b3b752b…`).
- Step median 199.8 ms (n=9). **Gate: passed.**

### 3c. 10-step LTX-2 smoke (const-max on device 2, bf16 on device 3)

- Latents finite: video `[1, 6144, 128]` (std 1.19, |max| 5.56) and audio `[1, 126, 128]`
  (std 0.98, |max| 5.72).
- The clip is coherent, a grassy field with a dark cat. It looks the same as the bf16 10-step
  clip, frame for frame (`ltx_2/smoke_bf16_top_cm_bottom.png`: frames 0, 60 and 120). The washed-out
  look is LTX-2 at 10 steps, in both arms.
- Step median 1213.4 ms (const-max) against 1284.7 ms (bf16), on different devices.
- **Gate: passed.**

## 4. FLUX.1-dev: runs

- **Settings.** bench.sh's `flux` key: 1024x1024, `--tp-degree 4`, 50 steps, seed 42, prompt
  "a cat sitting on a bench", guidance 3.5.
- **Compile and warm-up**, untimed, one arm per device: bf16 on device 1, cm on 2, mxcm on 3; 5:10
  / 5:10 / 6:38 wall.
- **Timed runs.** One process per arm, two identical requests (`--requests`), on device 1, CPUs
  56-63,152-159, back to back: bf16 08:27:35, cm 08:28:43, mxcm 08:29:48. `fuser` was idle at
  each start.
- **Warm step** = the DiT step median of the second request (report.py's `step_ms`).
- **Tree.** Each arm's head is in its `timed.env.txt`: bf16 at `2c98f795`, cm and mxcm at
  `14e8c9cb`. The only differences between them are E2's commits and E1's `arms.sh` smoke mode,
  none of them on a model path. The uncommitted E2 edits in the worktree (`mx_attention_kernel.py`,
  `attn_tile_masks.py`) are imported by no model.

| arm | request 1 step median | **request 2 (warm) step median** | ratio to bf16 | output md5 (both requests) |
|---|---|---|---|---|
| bf16 (shipped cte) | 220.1 ms | **220.5 ms** | 1 | `67da9e98…` = evidence A1 |
| bf16 + const-max | 199.5 ms | **200.0 ms** | **0.907** | `23f8c63d…` |
| `--quant mxfp8` + const-max | 162.3 ms | **162.2 ms** | **0.736** | `5fd4a295…` |

- **Determinism.** In each arm both requests are bit-identical. The bf16 arm reproduces the
  evidence's A1 image (`docs/verification/2026-11-07-mxfp8-video-models.md` 6a: md5 `67da9e98`,
  221.6 ms).
- *Inferred.*
  - Const-max saves 20.5 ms per step in bf16, about 0.36 ms per call over FLUX's 57 attention
    calls. The kernel-level numbers at this per-rank shape predict 0.41 ms per call, 23.3 ms per
    step: section 9's *shipped* cte (VectorE exp, as here) ran 0.9155 ms and section 12's
    const-max 0.5069 ms. The model keeps about 88% of the kernel-level gain.
  - The mxfp8 arm: evidence B1 (mxfp8 linears only, same targets, another session) ran
    182.9 / 183.3 ms. The ~21 ms gap to 162.2 ms is also the attention, but no same-session
    mxfp8-only arm ran.

## 5. FLUX.1-dev: quality

- Scored by `scripts/attn_constmax/score.sh` (`scripts/compare_difflet_runs.py`) on each arm's
  second request: PNG and the final packed latents. Since this branch, the in-process tp4 decode
  also writes those latents (`difflet/cli/orchestrators/flux.py`).
- Reference: `/home/ubuntu/fp8-bench/fp32_ref/flux` (CPU fp32 diffusers, same bf16 noise).

| pair | PSNR (dB) | SSIM | LPIPS | latent cosine | latent rel L2 |
|---|---|---|---|---|---|
| const-max vs bf16 | **41.40** | 0.9931 | 0.0024 | 0.99957 | 0.029 |
| mxfp8 + const-max vs bf16 | 24.18 | 0.9224 | 0.0683 | 0.97928 | 0.204 |
| bf16 vs fp32 (measured; evidence 23.36) | 23.36 | 0.9170 | 0.0748 | 0.97568 | 0.222 |
| **const-max vs fp32** | **23.31** | 0.9156 | 0.0761 | 0.97532 | 0.223 |
| **mxfp8 + const-max vs fp32** | **22.11** | 0.8918 | 0.1177 | 0.96446 | 0.269 |
| *evidence:* mxfp8 linears only (B1) vs fp32 | 21.30 | 0.885 | 0.139 | - | - |
| floor: bf16 tp2 vs tp4 (evidence 6a, same image) | 29.4 | 0.963 | 0.021 | - | - |
| floor: bf16 tp2 (F2 image) vs fp32 (measured here) | 23.14 | 0.9124 | 0.0829 | - | - |
| floor: CPU bf16 vs fp32 (evidence, `fp32_ref/flux_cpu_bf16`) | 21.47 | 0.890 | 0.135 | 0.95756 | 0.293 |

- **const-max (bf16) holds quality.**
  - It is 41.4 dB from bf16, 12 dB tighter than bf16's own tp2-vs-tp4 floor (29.4 dB): the
    trajectory did not turn.
  - Against fp32 it is 0.05 dB below bf16 (23.31 vs 23.36). That is less than the spread
    between the two bf16 TP degrees against fp32 (23.36 vs 23.14) and well above the CPU bf16
    floor (21.47).
  - Rule applied: hold = no worse than bf16 against fp32 beyond the floor. 0.05 dB < 0.22 dB.
- **mxfp8 + const-max** is 22.11 dB from fp32. That is between the CPU bf16 floor and bf16, and
  above the evidence's mxfp8-linears-only 21.30. Its 24.2 dB from bf16 is the mxfp8 linears'
  usual trajectory drift (evidence 6a/6b: B1 22.6 dB from bf16 on this image). One image cannot
  rank 22.11 against 21.30 (6b).

## 6. LTX-2 (FLUX held, so it ran)

- **Settings.** bench.sh's `ltx_2` key: 512x768x121, `--tp-degree 4`, 50 steps, seed 42, prompt
  "a cat walking through a garden", mp4 output.
- **Arms.** Two, as planned: bf16, and bf16 with const-max.
- **Allowlist.** 45 of the 48 video `attn1` layers go to const-max. `transformer_blocks.44`, `.45`
  and `.47` fail the gap rule and stay on cte. The audio stream and the cross-attentions are not
  routed.
- **Graphs.** The const-max cache holds all 89 of bf16's NEFFs plus 4 new block NEFFs, one per
  rank. So the 3 fallback layers reuse bf16's block graph, and the 45 routed layers share one new
  graph.
- **Compile and warm-up**, untimed: bf16 on device 2, cm on device 3, 12:43 each. Both read
  the checkpoint from the root-disk hub: I/O-bound, D state.
- **Timed runs** on device 1, CPUs 56-63,152-159, back to back: bf16 08:53:24, cm 08:57:49. Tree
  `2984fed6`, clean under `difflet/`.

| arm | request 1 step median | **request 2 (warm) step median** | ratio to bf16 | mp4 md5 (both requests) |
|---|---|---|---|---|
| bf16 (shipped cte) | 1283.1 ms | **1282.6 ms** | 1 | `0702ffd2…` = evidence section 2 (1283.9 ms) |
| bf16 + const-max (45 of 48 attn1) | 1215.7 ms | **1217.3 ms** | **0.949** | `e81078e1…` |

- *Inferred.* Const-max saves 65.3 ms per step, 1.45 ms per routed call. That is more than the
  kernel-level prediction at this per-rank shape: section 9's shipped cte ran 2.0604 ms and
  const-max 1.1495 ms, so 0.91 ms per call, 41 ms per step. One possible reason is the
  docstring's: without the P-transpose DMA, the kernel no longer serializes with the collectives
  around it. Not measured; no profile was taken.

**Quality** (mp4 against mp4, as the evidence scored it; latents from the work dir; reference
`/home/ubuntu/fp8-bench/fp32_ref/ltx_2`, CPU fp32 diffusers with bf16 noise):

| pair | clip PSNR mean (worst frame) | SSIM | LPIPS | video latents cos / rel L2 | audio latents cos / rel L2 |
|---|---|---|---|---|---|
| **const-max vs bf16** | **27.63 (25.33)** | 0.9307 | 0.0292 | 0.99947 / 0.032 | 0.99993 / 0.012 |
| floor: bf16 tp2 vs tp4 (evidence 3a) | 25.9 (23.6) | 0.909 | 0.040 | 0.9975 / 0.071 | 0.9999 / 0.014 |
| bf16 vs fp32 (measured; evidence 16.55) | 16.55 (15.13) | 0.4716 | 0.1948 | 0.93524 / 0.374 | 0.98429 / 0.215 |
| **const-max vs fp32** | **16.54 (15.16)** | 0.4708 | 0.1947 | 0.93524 / 0.374 | 0.98446 / 0.214 |

- **LTX-2 holds quality too.**
  - Const-max is closer to bf16 than bf16 at another TP degree is (27.6 vs 25.9 dB; video
    latents rel L2 0.032 vs 0.071).
  - Against fp32 it is level with bf16 (16.54 vs 16.55, latents cosine equal to 5 digits).
- The fp32 distance itself is dominated by LTX-2's host-side bf16 state (a separate finding,
  2026-11-12: 21.3 dB with fp32 host state). This run does not change that, and it limits what
  the fp32 column can resolve.

## 7. Caveats

- **One image, one clip, one seed.** At 50 steps FLUX quality on one image is trajectory drift
  (evidence 6b). The 41.4 dB const-max-vs-bf16 distance is what makes the FLUX verdict robust.
  A smaller gap would need the twelve-image set.
- **The kernel's error at real gains is up to 6-7x cte's** (3a), from the bf16 scores in PSUM
  (section 12's emulation). The model-level effect on FLUX is small (41.4 dB), but per-layer
  attention fidelity is lower, and other prompts, resolutions or LoRAs were not tested.
- **The softmax_max rule is gamma-only.** It ignores RoPE and the activations' own statistics;
  the probe's inputs are synthetic with real gains. No overflow was seen in any run, but nothing
  proves an upper bound on real scores: the hard bound is the only guarantee, and the gap <= 80
  rule is the docstring's heuristic for it. For LTX-2's across-heads norm the per-head bound is
  optimistic (section 12). LTX-2's routed layers have sigma up to 3.8 > 1, outside the
  docstring's std <= 1 precondition.
- **Tensor-form softmax_max.** It is not the scalar form section 12 timed. The model-level step
  times include it.
- **Scope of the hook.**
  - Routed: FLUX's non-CP path (tp4 here) and LTX-2's TP processor (tp >= 2).
  - Not routed: CP (ring / ulysses / gather_kv), LTX-2 at tp1, the other models, cross-attention
    and the audio stream.
  - The mxfp8 LTX-2 arm was not run (plan: two arms).
- **Shared host.** E2's probes ran on other devices at the same time (load average 19-34). The
  arms ran back to back on one device and one CPU set. Both bf16 arms reproduce their evidence
  bit for bit: FLUX `67da9e98` at 220.5 vs 221.6 ms, LTX-2 `0702ffd2` at 1282.6 vs 1283.9 ms.
- **Results-doc name.** The plan names one doc for E1 and E2
  (`2026-11-18-attn-constmax-sparse.md`). E1 is here, as the task asked; E2 writes
  `2026-11-18-attn-block-sparse.md`.
- **The mxfp8 + const-max ratio against mxfp8 alone is cross-session** (evidence B1), not
  measured here.

## 8. Paths

- Code:
  - `difflet/backends/neuron/ops_impl/attention.py`
  - `difflet/models/flux/modeling_flux.py`
  - `difflet/models/ltx_2/modeling_ltx_2.py`
  - `difflet/cli/orchestrators/flux.py` (latents)
- Tables: `difflet/models/{flux,ltx_2}/attn_constmax.json`, written by
  `scripts/attn_constmax/table.py`.
- Run tools: `scripts/attn_constmax/{probe.py,arms.sh,score.sh}`.
- Outputs, under `/mnt/nvme/difflet-exp/constmax/`:
  - `probe/`, `probe_ltx2/`;
  - `flux/<arm>/` (compile/smoke/timed logs and `*.env.txt` with `fuser` and `mpstat`, `warm_*.png`,
    `work/worker_0/latents_req000{0,1}.pt`);
  - `flux/scores/`, `ltx_2/...`;
  - the per-arm caches `cache/<model>-<arm>`, kept.
