# E1 const-max bf16 attention on all seven DiTs: per-model gain (trn3)

Summary of the run "const-max bf16 attention on all models" (branch `exp/attn-constmax-models`).
It collects the six model lanes, the KERNEL lane and an independent auditor's verdict per lane.
Every auditor correction is applied below; section 6 lists each one and where it went.
HunyuanVideo (HV1) was pending in the first two versions of this summary (two attempts, no arm
timed). A third attempt timed all four arms on 2026-11-19 (lane doc `e9361d0f`). Its auditor
returned CORRECTED with two corrections; both are applied here (section 6). The lane doc itself
is not edited by this update, so it still carries the two uncorrected statements.

Sources, not re-measured here:

- Lane docs (each holds its own method, raw paths and caveats):
  - `docs/verification/2026-11-18-attn-constmax-qwen.md`
  - `docs/verification/2026-11-18-attn-constmax-flux.md`
  - `docs/verification/2026-11-18-attn-constmax-ltx2.md`
  - `docs/verification/2026-11-18-attn-constmax-wan.md` (Wan 2.1 and Wan 2.2)
  - `docs/verification/2026-11-18-attn-constmax-hv1.md` (`e9361d0f`, four arms timed on the third attempt)
  - `docs/verification/2026-11-18-attn-constmax-hv15.md`
  - `docs/verification/2026-11-18-attn-constmax-kernel.md` (KERNEL lane)
- E1: `docs/verification/2026-11-18-attn-constmax.md`.
- Kernel-level const-max: `feat/mxfp8-sage-attention`
  `docs/verification/2026-11-13-dit-block-profile.md`, section 12. There it ran at 0.617x / 0.651x
  / 0.701x of the *rebalanced* cte at `flux_1024` / `ltx2_video` / `wan_81f`.
- MXFP8 linears alone vs bf16: `docs/verification/2026-11-07-mxfp8-video-models.md`, sections
  7f, 11b and 12b.

Labels: **measured** means it was measured in this run, in the lane named. *Inferred* means it
was computed from measured numbers and not measured itself. *Evidence* means it was quoted from an
earlier doc. Section 5 lists every inference.

**Arms**, per model, all on one device and one CPU set, back to back. Each arm is one process
with two identical requests; the warm step is the mean of the two request medians. (E1's
template, `2026-11-18-attn-constmax.md` line 160, uses request 2's DiT step median instead.)

- A: bf16, shipped attention_cte.
- B: bf16 + `DIFFLET_EXP_ATTN_CONSTMAX=1`.
- C: `--quant mxfp8` (the model's shipped target set).
- D: `--quant mxfp8` + const-max.

## 1. Summary

| model | shape, per-rank attention call | routed to const-max | A step (ms) | B/A | C/A | D/A | D/C | B vs A: PSNR, latent cos (worst frame) | vs CPU fp32: B (A) | bf16 floor | auditor | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen-Image | 1024x1024, tp4, 50 steps; 6 x 5,120 x 128, key-bounded | 57 of 60 joint | 304.7 | **0.896** | 0.790 | **0.712** | 0.901 | 48.82 dB, 0.99993 (n/a: image) | 35.91 (36.34) dB | tp2 vs tp4 44.31 dB, but measured on the pre-`c9f4009` unmasked graph. No floor exists for the current graph (corrected). | CORRECTED | DONE |
| FLUX.1-dev | 1024x1024, tp4, 50 steps; 6 x 4,608 x 128, unbounded | 57 of 57 | 222.6 | **0.903** | 0.834 | **0.742** | 0.889 | 41.40 dB, 0.99957 (n/a: image) | 23.31 (23.36) dB | tp2 vs tp4 29.4 dB; CPU bf16 vs fp32 21.47 dB | CORRECTED | DONE |
| LTX-2 | 512x768x121, tp4, 50 steps; 8 x 6,144 x 128 (video attn1), unbounded | 45 of 48 video attn1; audio and cross-attn stay on cte | 1,287.1 | **0.948** | 0.757 | **0.695** | 0.919 | 27.63 dB, 0.99947 (25.33) | 16.54 (16.55) dB | tp2 vs tp4 25.9 dB (worst 23.6) | CORRECTED | DONE |
| Wan 2.1 T2V 14B | 480x832x81, tp4, 50 steps, CFG 5.0 (2 forwards); 10 x 32,760 x 128, unbounded | 36 of 40 attn1, 72 calls per step; attn2 stays on cte | 10,107.5 | **0.813** | 0.981 | **0.793** | 0.809 | 22.21 dB, 0.96906 (21.87) | 20.33 (20.15) dB, section 8 | tp8 vs tp4 22.75 dB (worst 18.31), from host B | CONFIRMED | DONE |
| Wan 2.2 T2V A14B | 480x832x81, tp4, 40 steps, CFG 4.0 / 3.0, two experts; 10 x 32,760 x 128, unbounded | 36 of 40 attn1 per expert, 72 calls per step | 10,115.2 | **0.813** | 0.981 | **0.793** | 0.808 | 12.31 dB, 0.89913 (11.77); fails the SSIM screen | 10.87 (11.00) dB, section 8 | tp8 vs tp4 14.83 dB (worst 13.74), host B, weak | CONFIRMED | DONE |
| HunyuanVideo | 544x960x129, tp4, 50 steps, guidance 6.0; 6 x 67,576 x 128, key-bounded (hi 67,449) | 55 of 60 joint (19 of 20 dual, 36 of 40 single); on device, B's and D's routed NEFFs hold `attention_const_max` (auditor) | 18,485.6 | **0.4975** | 0.971 | **0.468** | 0.482 | 17.28 dB, 0.93291 (11.60) | 17.53 (15.36) dB, section 8 | none at this shape (bf16 tp2 runs out of HBM); 36.64 dB at 320x512x61 cannot resolve it | CORRECTED (3rd attempt) | DONE (3rd attempt; the first two timed no arm) |
| HunyuanVideo-1.5 | 480x848x121, tp2 x CFG2, 50 steps, CFG 6.0; 8 x 51,275 x 128, key-bounded | 52 of 54 joint | 9,917.5 | **0.632** | 1.011 | **0.631** | 0.624 | 14.76 dB, 0.8744 (14.04) | 14.55 (16.09) dB, section 8 | 15.90 (worst 15.31) / 16.94 / 18.41 dB | CORRECTED | DONE |

Notes on the table:

- All timings are **measured**.
- Every C/A agrees with the MXFP8 evidence, which ran in other sessions:
  - Qwen 0.79 and FLUX 0.83 (11b);
  - LTX-2 0.75 (11b);
  - Wan 2.1 and 2.2 0.98 at 81 frames with CFG (12b);
  - HV1 0.97 (7f);
  - HV1.5 1.01 (12b).
- Bit-identical reproductions of the evidence:
  - A's output md5 equals the evidence's bf16 for Qwen, FLUX and LTX-2.
  - C's md5 equals the evidence's MXFP8 for the same three.
  - B and D for FLUX, and B for LTX-2, equal E1's.
- Where A does not reproduce the evidence:
  - Wan 2.1 / 2.2: A equals this host's own earlier bf16 runs, not host B's.
  - HV1.5: A's host text encode ran on 12 threads, the evidence's on 96.
  - HV1: A (`6336311a`) is deterministic (both requests and an earlier A run, `A/timed_pre`,
    give one md5 and equal latents). It differs from 7f's `258cd73e` because of `c13fda35`
    (2026-11-11, fp32 timestep), and equals the block-profile doc's section-15 bf16 run (15d).
  - A's step times still match the evidence within 0.4%.
- Warm end-to-end follows the step (**measured**, request 2; each lane defines e2e its own way,
  see its doc):

  | | Qwen | FLUX | LTX-2 (no host decode) | Wan 2.1 | Wan 2.2 | HV1 | HV1.5 |
  |---|---|---|---|---|---|---|---|
  | A warm e2e | 16.23 s | 11.55 s | 74.81 s | 526.24 s | 425.49 s | 1,003.0 s | 564.2 s |
  | B / A | 0.903 | 0.906 | 0.973 | 0.821 | 0.822 | 0.537 | 0.679 |
  | D / A | 0.733 | 0.749 | 0.741 | 0.802 | 0.802 | 0.510 | 0.680 |

## 2. Per model

What limited the gain is *inferred* from the kernel-level per-call times. The quality reading is
**measured** against this run's A.

Table 2 (*inferred* throughout):

- "kernel cm / cte": the ratio of the per-call times at the model's per-rank shape.
- "predicted": that per-call saving times the routed calls per rank per step.
- "share": the shipped cte's per-call time times all calls of that kind, over A's step. It is a
  rough bound on how much of the step the attention takes.

| model | kernel cm / cte (ms per call) | source | predicted saving / step | measured A - B | kept | attention share of A's step |
|---|---|---|---|---|---|---|
| Qwen-Image | 0.511 (1.202 -> 0.614) | KERNEL lane, same session, bounded | 57 x 0.588 = 33.5 ms | 31.8 ms | 95% | ~24% (60 x 1.202 ms) |
| FLUX.1-dev | 0.554 (0.9155 -> 0.5069) | section 9 shipped cte vs section 12, cross-session | 57 x 0.409 = 23.3 ms | 21.55 ms | 93% | ~23% (57 x 0.9155 ms) |
| LTX-2 | 0.558 (2.0604 -> 1.1495) | section 9 vs 12, cross-session | 45 x 0.911 = 41.0 ms | 67.55 ms | 165% | ~8% (48 video attn1 x 2.06 ms) |
| Wan 2.1 | 0.575 (63.72 -> 36.65) | section 9 vs 12, cross-session | 72 x 27.07 = 1,949 ms | 1,885.6 ms | 97% (approx.) | ~50% (80 x 63.72 ms) |
| Wan 2.2 | same shape | same | 1,949 ms | 1,896.1 ms | 97% (approx.) | ~50% |
| HunyuanVideo | 0.507 (183.80 -> 93.17) | KERNEL lane, bounded; the cte arm includes HV1's 2-call head split | 55 x 90.6 = 4,985 ms | 9,288.1 ms | 186% | ~60% kernel only (60 x 183.8 ms); ~79% with the dual-block glue (+20 x ~180 ms) |
| HunyuanVideo-1.5 | 0.501 (143.26 -> 71.74) | KERNEL lane, bounded, this exact call | 52 x 71.5 = 3,719 ms | 3,645 ms | 98% | ~78% (54 x 143.26 ms) |

The per-call const-max / cte ratio is 0.50-0.58 for every model, so the kernel is not what
separates them. **The attention's share of the step is**, and it grows with the sequence length.
Two models keep more than the kernel predicts: LTX-2 (165%, cause not measured) and HV1 (186%).
At HV1, const-max also skips the shipped bounded route's dual-block glue (fp32 q pre-scale, pad,
head-chunk slices; ~180 ms per dual block in the 11-13 profile, *evidence*), which the kernel
probe ran on contiguous inputs and so did not see. That glue explains 3.4-3.6 s of HV1's 4.3 s
excess (*inferred*; see HV1 below).
The unrouted layers cost little: 0.2-2.5% of the step at most. The 3-5 unrouted layers per model
would save, *inferred* from the kernel times only:

- Qwen: 1.8 ms;
- LTX-2: 2.7 ms;
- Wan: 217 ms;
- HV1.5: 143 ms;
- HV1: 453 ms (plus ~180 ms of glue in the one unrouted dual block, `transformer_blocks.1`).

- **Qwen-Image (B/A 0.896, D/A 0.712).**
  - *Limit.* At 5,120 tokens the joint attention is about a quarter of the step (*inferred*).
    Blocks 0, 52 and 58 fail the gap rule; block 0's image q/k gains reach 63 / 48.5, sigma 270.
    The model keeps 95% of the kernel-level saving.
  - *Correction.* The Qwen lane wrote that no kernel-level timing exists at Qwen's shape. The
    KERNEL lane measured it after the Qwen doc was committed (`47c7df2d`).
  - *Quality.* B is 48.82 dB / cos 0.99993 from A, the closest B of any model. Against fp32 it is
    0.43 dB below A, on one image, which cannot rank it.
  - *Floor (auditor correction).* The only tp2-vs-tp4 floor (44.31 dB) comes from the pre-`c9f4009`
    graph, which had no text mask. So "B tighter than the floor" and "D/C 37.75 dB below the
    floor range" are not established on the current graph. Read them as indicative only.

- **FLUX.1-dev (B/A 0.903, D/A 0.742).**
  - *Limit.* At 4,608 tokens attention is about 23% of the step (*inferred*). All 57 calls are
    routed, so the attention share is the whole limit.
  - *Quality.* B is 41.40 dB from A, 12 dB tighter than the bf16 tp2-vs-tp4 floor (29.4 dB).
    Against fp32 it is 0.05 dB below A. **Holds.**
  - *MXFP8 arms.* D vs C is 23.56 dB: the MXFP8 trajectory drift the evidence already knew.
    Against fp32, D is 0.80 dB above C (corrected from 0.81).

- **LTX-2 (B/A 0.948, D/A 0.695).**
  - *Limit.* Only the video self-attention is routed, and it is about 8% of the step
    (*inferred*). The audio stream and the cross-attentions stay on cte.
  - *Unexplained gain.* The model saves 1.50 ms per routed call (1.75 ms on MXFP8), more than the
    kernel-level 0.91 ms. This is *inferred* and not profiled.
  - *Quality.* B is 27.63 dB (worst frame 25.33) from A, tighter than the 25.9 dB (23.6) floor.
    Against fp32 it is level with A (16.54 vs 16.55). **Holds.**
  - *MXFP8 arms.* D vs C is 25.15 dB (worst 23.28), latent rel L2 0.074. That is at, or slightly
    past, the floor.

- **Wan 2.1 (B/A 0.813, D/A 0.793).**
  - *Limit.* At 32,760 tokens the self-attention is about half the step (*inferred*). The shipped
    cte is VectorE-bound there, at 63.7 ms per call. Blocks 33, 34, 38 and 39 stay on cte, about
    217 ms per step forgone (*inferred*).
  - *Quality.* B is 22.21 dB (worst 21.87) from A, about at the bf16 tp8-vs-tp4 floor of 22.75 dB
    (18.31). It passes the floor screen on all three metrics. There is no fp32 reference.
    Same cat, pose and scene; A's foreground bowl is absent in B in all three sampled frames
    (wording per audit).
  - *MXFP8 arms.* MXFP8 alone buys 2% (C/A 0.981). D/C 0.809: the two gains add in ms
    (1,892 vs 1,886 ms).

- **Wan 2.2 (B/A 0.813, D/A 0.793).**
  - *Limit.* Same as Wan 2.1: the same per-rank call, 36 of 40 routed in each expert. The
    low-noise expert's gains equal Wan 2.1's to within 1.6e-5, so its routed sigma max is 2.56;
    2.55 is the high-noise expert's (wording per audit).
  - *Quality.* B is 12.31 dB from A, past the weak bf16 floor (14.83 dB), and **fails the
    floor screen on SSIM** (0.252 < 0.339), as the shipped MXFP8 arm C does (11.14 dB). B is still
    the arm closest to A, keeps A's composition and shows no noise. Acceptability is open.

- **HunyuanVideo (B/A 0.4975, D/A 0.468): the largest gain of the campaign.**
  - *Status.* The first two attempts timed no arm (the second compiled A in 1:12:03, then was
    killed at 13:24:12). The third re-ran all four compiles on 2026-11-18 (A 8:45 from cache,
    B 1:14:30, C 42:38, D 50:50; exit 0, 60 blocks, no fallbacks), passed the 10-step gate
    (15:03, all latents finite, frames clean) and timed A, B, C, D back to back on device 5 on
    2026-11-19, 03:20-06:17. The second attempt's open issues are closed: every compile re-ran,
    and the `work/worker_0` latents are the timed ones (the auditor checked their mtimes).
  - *Measured.* Const-max halves the step: 9,288 ms off bf16 and 9,303 ms off MXFP8. The warm
    129-frame request falls from 1,003 to 538 s. Each arm's two request medians agree within
    0.016%.
  - *Why B/A is below the kernel ratio (inferred).* B/A 0.4975 is below the routed call's
    kernel-level 0.507, and A - B is 186% of the kernel prediction. Const-max also skips the
    shipped bounded route's dual-block glue: the fp32 q pre-scale, the pad to 67,584 and the two
    3-head chunk slices of the padded tensor. That costs ~180 ms per dual block in the 11-13
    profile (*evidence*), 3,414-3,481 ms per step at 19 routed dual blocks. Kernel saving plus
    glue give 8,400-8,630 ms of the 9,288 ms; 7-10% is unattributed, and B was not profiled.
    With the glue, attention is ~79% of A's step, not the ~60% behind the old prediction
    (~0.73-0.74, now superseded by the measurement).
  - *Recoverable without const-max (inferred).* The pre-scale and the head split are torch-side
    choices of the bounded cte route, so ~3.4 s per step at bf16 looks recoverable with no
    const-max error. No arm tested it.
  - *MXFP8 arms.* C/A 0.971 matches 7f's 0.97. The two gains add in ms (MXFP8 saves 533.5 ms
    without const-max, 548.6 ms with it), so D/A 0.468 is below B/A x C/A 0.483. The same glue
    sum on MXFP8 gives 8,635-8,801 ms against 9,303 ms measured (auditor issue: the lane doc's
    "about 8,635 ms" leaves out the single-block term).
  - *Quality.* B vs A is 17.28 dB (worst 11.60, frame 84), SSIM 0.765, LPIPS 0.336, latent cos
    0.933: **further from A than MXFP8 moved it** (C vs A 21.33 dB, cos 0.964). On top of MXFP8
    the const-max step is the smallest of the three changes: D vs C 30.29 dB (worst 21.01),
    SSIM 0.944, LPIPS 0.071, cos 0.997. B is the outlier: its latent cos to A, C and D is
    0.928-0.933, while A, C and D sit within 0.964-0.997 of each other. Per latent frame, B vs A
    dips to 0.82-0.91 in frames 12-22 (pixel ~45-88); at frame 84 B shows a pale background where
    A has dark trunks. All four clips are the same clean, coherent shot.
  - *Floor.* None at this shape (bf16 tp2 runs out of HBM; no fp32 reference). The only floor,
    7a's 320x512x61 bf16 tp2 vs tp4 (36.64 dB / SSIM 0.957 / LPIPS 0.031; screen 30 dB / 0.857 /
    0.062, *evidence*), cannot resolve this shape. B, C and D vs A fail all three screens; D vs C
    passes PSNR and SSIM and fails LPIPS (0.071 > 0.062). One seed and one prompt cannot separate
    const-max error from a trajectory split. B also drops A's fp32 q pre-scale, which changes q's
    rounding.
  - *Routing (auditor correction).* **Shown directly on device**, not only indirectly as the lane
    doc says. Each NEFF is a 1 KiB header and a gzip stream: B's `d190`/`dfa4` and D's
    `81d5`/`a8c6` contain `attention_const_max`, `attention_const_max_kernel` and
    `attention_const_max_lnc` and no `attention_cte`. A's `03c8`/`df55` and C's `35a3`/`4b3b`
    contain only `attention_cte` (24 against 12 hits: two head-chunk calls against one call).
    The NEFF keys agree: B opens A's 148 keys plus those 2, D opens C's 148 plus 2.
  - *No-head-split risk: did not fire.* B and D compiled and ran the 6 x 67,576 call in one piece
    at model level, with no compile or allocation error. HBM use was not recorded. Compile peak
    host RSS (auditor correction): A 34.4 GB, B 57.3 GB, C 69.4 GB, D 57.3 GB.
  - *md5.* A differs from 7f because of `c13fda35`, as the second attempt's auditor expected; it
    is not a const-max finding (section 1 notes).
  - *Timing caveats (auditor issues).* A and B ran before `corenode.lock` was added to the arm
    runner; C and D ran under it (D waited 7 min 45 s for it). The 5-minute load averages at B's
    and C's starts were 22 and 39. A matches `A/timed_pre` within 0.05% and the block-profile
    15a within 0.02%, and B's spread is 0.001%, so no effect shows; contention could only have
    slowed A or B. The arms are not interleaved, so a drift over the 3 h chain is not excluded.
    The step values are per-request medians (n = 49) and maxes; raw per-step times are not kept.

- **HunyuanVideo-1.5 (B/A 0.632, D/A 0.631): the second-largest gain, after HV1.**
  - *Limit.* At 51,275 tokens the joint attention is about 78% of the step (*inferred*). The
    model keeps 98% of the KERNEL lane's per-call saving at this exact call (52 x 71.5 = 3,719 ms
    predicted vs 3,645 ms measured). The auditor flagged that as the strongest evidence for the
    routing.
  - *MXFP8 arms.* MXFP8 linears alone buy nothing (C/A 1.011), so D is about B.
  - *Cost.* Const-max adds 1.08 GiB of runtime HBM per rank; the peak is 18.76 of 36.0 GiB.
  - *Quality.* B is 14.76 dB (worst 14.04), cos 0.874 from A: **past every bf16 floor pair**
    (15.90-18.41 dB, cos 0.911-0.952). It passes the floor-relative screens, only barely on 12c's
    SSIM (0.268 vs 0.262). The clip is coherent, but the cat is a grey tabby where A's is orange.
  - *Correction.* The const-max route does not pad, so softmax_max is taken at 51,275, not at the
    padded 51,328. The effect is about 0.005%; no number changes.

Quality detail (**measured**; D vs C is the const-max step on top of MXFP8):

| model | B vs A SSIM / LPIPS | D vs C PSNR (worst), cos | vs fp32: C / D | screen (B) |
|---|---|---|---|---|
| Qwen-Image | 0.9973 / 0.0008 | 37.75 dB, 0.99854 | 37.36 / 39.17 dB | n/a (the floor comparison is not established) |
| FLUX.1-dev | 0.9931 / 0.0024 | 23.56 dB, 0.97777 | 21.30 / 22.11 dB | n/a (well inside the 29.4 dB floor) |
| LTX-2 | 0.9307 / 0.0292 | 25.15 (23.28) dB, 0.99723 | 16.85 / 16.68 dB | n/a (inside the 25.9 dB floor) |
| Wan 2.1 | 0.7403 / 0.1467 | 25.61 (23.91) dB, 0.9885 | n/a | passes all three |
| Wan 2.2 | 0.2515 / 0.3671 | 13.56 (13.04) dB, 0.9347 | n/a | fails SSIM |
| HunyuanVideo | 0.7648 / 0.3358 | 30.29 (21.01) dB, 0.99655 | n/a | fails all three (the only floor, 320x512x61, cannot resolve this shape) |
| HunyuanVideo-1.5 | 0.2684 / 0.4221 | 13.66 (13.13) dB, 0.8623 | n/a | passes (barely on 12c SSIM) |

## 3. The KERNEL lane (key-bounded const-max)

Qwen-Image, HV1 and HV1.5 run their joint attention as key-bounded calls (`bound_min = 0`,
`bound_max = hi`). nkilib's `attention_const_max` has no mask. The KERNEL lane wrote a bounded
fork, commit `2d6714cf`, doc `47c7df2d`. FLUX, LTX-2 and Wan use E1's unbounded op.

**Design.**

- The op is `difflet_neuron::attention_const_max_bounded(q, k, v, scale, softmax_max, hi)`, in
  `difflet/backends/neuron/ops_impl/attention_const_max_kernel.py`.
- It forks nkilib 2.14 `experimental/attention/attention_const_max.py`, sha256 `4aa030f0…`, pinned
  and test-checked. Only 4 functions are copied; the other helpers are imported from upstream.
- The new `key_bound` argument is int32 `[N, Sq, 1]`, attention_cte's `bound_max` layout, so
  Sq >= 128.
- MM1 is transposed, so the exp already takes a per-partition max operand. Per head the fork
  builds one `[128, num_k_tiles]` f32 operand: softmax_max + 30000 x (key_index >= hi).
- A masked key's exp is exactly 0, in the numerator and in the `[V | 1]` row sum. No control flow
  depends on the bound. Without a bound the instruction stream is upstream's.
- The route is opt-in (`DIFFLET_EXP_ATTN_CONSTMAX=1`). It takes non-causal bf16, d = 128,
  S_q >= 128 calls that carry a sigma and bounds.
- softmax_max = 0.88 sigma sqrt(2 ln Sk), computed in the graph. The route neither pre-scales q
  nor pads, and it does not chunk heads.

**Correctness (measured).**

- Simulator: 54 cases pass, on the VectorE and the ScalarE exp, at LNC 1 and 2.
  - Without a bound the output is bit-identical to nkilib, and also at hi = Sk.
  - Bounded, rel L2 vs masked fp32 is <= 1.2x the unmodified kernel's own error.
  - Zeroing the mask term fails all 36 masked cases.
- Route tests: 24 pass. Neighbouring attention tests: 105 pass.
- Device gate (device 7, Qwen shape): every output is finite. rel L2 is 0.658 vs unmasked fp32
  and 1.26e-2 vs masked fp32, so the bound is applied on hardware.

**Probe**: device 7, one logical core per rank, synthetic inputs at real QK-norm gains, `indep`
family, the largest-sigma layer.

| per-rank call | layer (sigma) | hi | shipped bounded cte | const-max bounded | ratio | rel L2 vs masked fp32: cte / const-max | finite |
|---|---|---|---|---|---|---|---|
| Qwen-Image 6 x 5,120 x 128 | `transformer_blocks.56` (3.79) | 4,609 | 1.202 ms | 0.614 ms | **0.511** | 4.07e-3 / 1.26e-2 | yes |
| HunyuanVideo 6 x 67,576 x 128 | `single_transformer_blocks.14` (3.95) | 67,449 | 183.8 ms | 93.17 ms | **0.507** | 5.80e-3 / 2.26e-2 | yes |
| HunyuanVideo-1.5 8 x 51,275 x 128 | `transformer_blocks.32` (3.07) | 50,283 | 143.26 ms | 71.74 ms | **0.501** | 5.42e-3 / 1.57e-2 | yes |

**Caveats.**

- The cte arm includes its route's q pre-scale, the pad to 128 and HV1's 2-call head split, on
  contiguous inputs. At model level HV1's dual-block inputs are not contiguous, and that glue
  costs ~180 ms per dual block more (section 2, HV1). The
  Qwen ratio carries about 0.01 ms per call of loop overhead in both arms (*inferred* from E1
  section 12).
- The const-max error is the kernel's own, from bf16 scores and the fixed max; the bound adds no
  error. On flat inputs it is 2.9-3.9x cte's, on peaked inputs 1.1-4.1x.
- The route trusts `bound_min = 0` and `bound_max` constant along queries. This is not checked at
  run time. A future caller with per-query bounds would get silently wrong output.
- The first compile at the video shapes is a long single-threaded NKI trace: 1,235 s (HV1) and
  897 s (HV1.5), about 5 GB RSS, and it looks hung while it runs. With
  `DIFFLET_NKI_TRACE_CACHE=1`, later processes start in 26-35 s.
- The KERNEL lane had no auditor of its own. The Qwen auditor matched its gate quote against
  `KERNEL.json`. The HV1.5 auditor matched its per-call saving against the model-level step.
  The HV1 auditor matched its 0.507 (183.80 / 93.17 ms) against the KERNEL doc.

## 4. What it means

**Ranking.**

Every value in both rankings is **measured** (HV1 included, from its third attempt).

- By B/A (bf16 + const-max vs shipped bf16):
  1. HV1 0.4975;
  2. HV1.5 0.632;
  3. Wan 2.1 0.813 and Wan 2.2 0.813;
  4. Qwen-Image 0.896;
  5. FLUX 0.903;
  6. LTX-2 0.948.
- By D/A (MXFP8 + const-max vs shipped bf16):
  1. HV1 0.468;
  2. HV1.5 0.631;
  3. LTX-2 0.695;
  4. Qwen-Image 0.712;
  5. FLUX 0.742;
  6. Wan 2.1 and Wan 2.2 0.793.
  The order changes because MXFP8 linears help the short-sequence models (C/A 0.76-0.83) and
  barely help Wan, HV1 or HV1.5 (C/A 0.97-1.01). There, const-max is the only one of the two
  that pays.
- **The gain tracks the attention's share of the step**, which tracks sequence length:
  - about 8% (LTX-2's video attn1 at 6.1k tokens, the only part routed) gives 5%;
  - about 23-24% at 4.6-5.1k tokens gives 10%;
  - about 50% at 32.8k tokens gives 19%;
  - about 78% at 51.3k tokens gives 37%;
  - about 79% at 67.6k tokens (HV1: ~60% kernel plus ~19% dual-block glue) gives 50% (shares
    *inferred*, gains **measured**).
  - HV1 gains more than HV1.5 at about the same share because const-max halves the kernel and
    removes the glue outright. Counting the kernel alone, the earlier prediction put HV1's share
    at ~60% and its B/A at ~0.73-0.74; the measured 0.4975 supersedes it. The share predicts the
    gain only when it counts the route's glue (*inferred*).
- The two gains compose. On top of MXFP8, const-max saves about the same ms as on bf16 for FLUX,
  Wan, HV1.5 and HV1 (within 5%), less for Qwen (23.8 vs 31.8 ms) and more for LTX-2 (78.9 vs
  67.6 ms). D/A is within 0.012 of B/A x C/A for every model except LTX-2 (0.695 vs 0.717) and
  HV1 (0.468 vs 0.483, because there the two savings add in ms rather than in proportion).

**Where const-max is worth adopting, and where not.** This is a reading for the user, who makes
the default call. Every verdict rests on one seed and one prompt.

- **Quality holds, modest gain: FLUX (0.903), LTX-2 (0.948), Qwen-Image (0.896).**
  - B is far inside the floor for FLUX (41.4 vs 29.4 dB) and LTX-2 (27.6 vs 25.9 dB), and level
    with A against fp32.
  - Qwen's B is 48.8 dB from A. Its floor needs a tp2-vs-tp4 control on the current masked graph
    before it counts.
  - These are the safest default candidates, but they buy only 5-10%.
- **Large gain, quality open: Wan 2.1 (0.813).**
  - B is about at the floor and passes the screen on one clip.
  - It is the strongest candidate among the big gains. It needs the twelve-clip set before it
    becomes a default.
- **Large gain, quality not established: HV1 (0.4975), HV1.5 (0.632), Wan 2.2 (0.813).**
  - B is past every bf16 floor pair (HV1.5), fails the SSIM screen (Wan 2.2), or moved further
    from A than MXFP8 did with no floor at its shape (HV1: 17.28 vs 21.33 dB).
  - The clips are coherent, but one clip with no fp32 reference cannot separate const-max error
    from trajectory drift. None should become a default on this evidence.
  - They can stay opt-in, for users who trade fidelity to A for 19-50% of the step.
- **HV1: part of the gain may come without const-max.** About 3.4 s of HV1's 9.3 s saving per step is
  the shipped bounded cte route's dual-block glue (fp32 q pre-scale and head split, torch-side
  choices), not the kernel (*inferred*). Removing it from the shipped route would carry no
  const-max error. No arm tested it; it is the cheapest next experiment for HV1.

**Open risks.**

- **One image or clip, one seed, one prompt per model.** At 50 steps (FLUX, Qwen) and at 81-121
  frames with CFG (Wan, HV1.5), one sample moves by several dB under any bit change.
- **There is no CPU fp32 reference** for Wan, HV1 or HV1.5 at these shapes.
- **Floors from other graphs or hosts.** Qwen's floor is from the pre-`c9f4009` graph; Wan's
  floors are host B's, and this host's bf16 md5 differs from host B's.
- **softmax_max comes from the QK-RMSNorm gains only.** It ignores RoPE and the activations'
  statistics.
  - Every model routes layers with sigma > 1, outside the kernel docstring's std <= 1
    precondition. Routed sigma maxima: FLUX 2.09, Wan 2.56, HV1.5 3.07, Qwen 3.79, LTX-2 3.79,
    HV1 3.95.
  - HV1's largest routed gap is 78.2 against the limit of 80.
- **No hard overflow guarantee.**
  - The gap <= 80 rule is the docstring's heuristic. The hard bound is the only guarantee, and
    for Wan's and LTX-2's across-heads RMSNorm the per-head bound is optimistic (section 12: up
    to 40x at Wan).
  - Section 12 saw overflow at x4 inputs. No run here produced a non-finite value.
- **Kernel error grows with the score scale.**
  - At real gains, E1 measured about 6x cte at sigma 2.1 and about 7x at 2.3-3.9.
  - Section 12 measured 7.5-8x on peaked inputs.
  - The bounded fork: 2.9-3.9x (flat), 1.1-4.1x (peaked).
  - The model-level effect is small at short sequences (FLUX 41.4 dB, Qwen 48.8 dB). It is
    unresolved at long ones.
- **The bounded route's input contract** (`bound_min = 0`, constant `bound_max`) is unchecked at
  run time.
- **Not exercised:** CP ring / ulysses (not routed), gather_kv CP, Qwen CFG-parallel batch-2 calls
  with per-row hi, other resolutions, LoRAs.
- **Cost:**
  - first-compile NKI trace of 15-20 min at the video shapes (use `DIFFLET_NKI_TRACE_CACHE=1`);
  - +1.08 GiB runtime HBM per rank on HV1.5; HV1's HBM use was not recorded.

## 5. Measured vs inferred

**Measured** (in this run, recomputed by the lane auditors from the raw logs):

- every A/B/C/D step, ratio, spread and warm e2e in sections 1 and 2;
- every md5;
- every quality number in section 1 and in section 2's quality table;
- the routed-layer tables, recomputed from the checkpoint gains (HV1's on every attempt);
- the KERNEL lane's simulator, gate and probe numbers;
- HV1's compiles, gate, NEFF keys and NEFF contents (the contents checked by its auditor).

*Inferred* (arithmetic on measured numbers, not measured itself):

1. Per-call savings at model level: Qwen 0.56 / 0.42 ms, FLUX 0.38 ms, LTX-2 1.50 / 1.75 ms,
   Wan 26.2 / 26.3 ms, HV1.5 ~70 ms, HV1 168.9 ms. Each is A - B over the routed calls.
2. Table 2: the predicted savings, the "kept" fraction and the attention share of the step.
   - FLUX's, LTX-2's and Wan's per-call cte and const-max times come from two sessions and
     devices (section 9's shipped cte, section 12's const-max), so their "kept" values are
     approximate.
   - The shares use one-core kernel times on synthetic inputs, not a model profile. No model run
     was profiled.
3. HV1's glue breakdown (section 2): kernel saving plus the *evidence* dual-block glue explain
   8,400-8,630 of the 9,288 ms (7-10% unattributed), and 8,635-8,801 of the 9,303 ms on MXFP8;
   the ~79% attention share; and that the shipped route could recover the glue without
   const-max. The earlier prediction (B/A ~0.73-0.74, D/A ~0.70-0.72) is superseded by the
   measured 0.4975 / 0.468.
4. LTX-2's model-level saving exceeds the kernel prediction (165%). The cause is not measured; E1
   suggested less serialization with the collectives.
5. "The two gains compose": B/A x C/A against D/A, from measured ratios. Why Qwen saves less on
   MXFP8 (23.8 vs 31.8 ms) and LTX-2 more (78.9 vs 67.6 ms) is not measured.
6. The ms forgone on unrouted layers (section 2).
7. Quality readings that rank arms against fp32 on one sample are flagged as not rankable where
   they appear.

## 6. Auditor corrections applied

| lane | correction | applied in |
|---|---|---|
| Qwen | The tp2-vs-tp4 floor (44.31 dB; 39.1-45.8 dB over 3 prompts) is from 2026-11-01, on the pre-`c9f4009` unmasked graph. No floor exists for the current graph. | Section 1, floor cell; section 2, Qwen; section 4, open risks |
| Qwen | "B tighter than the floor" and "D/C below the floor range" are not established; the numbers stand. The "27.5-42.6 dB MXFP8-from-bf16 range" in the lane doc is pre-rebase too. | Section 2, Qwen; quality table, screen column |
| Qwen | Tree note: from `2d6714cf` to `b05c71d6` no `difflet/` code changed. A docs file and two lane scripts were added (not "only docs"). | Here; no number affected |
| FLUX | "No timed process compiled anything" is false. A's timed process compiled its two DiT NEFFs before the measured steps. C's compiled ~70 small NEFFs during request 1; that, not one-off work, is C's 7.8 s first-request delay. Step medians are unaffected. | Here; section 1 values unchanged |
| FLUX | D vs C against fp32 is 0.80 dB, not 0.81. | Section 2, FLUX |
| FLUX | The CPU bf16 floor (21.47 dB) is from `/home/ubuntu/fp8-bench/fp32_ref/flux_cpu_bf16/score.json`, not from the 2026-11-07 doc. | Section 7 (source); section 1 value unchanged |
| LTX-2 | A's and B's compile steps did not hit E1's caches; they recompiled their block NEFFs, because `difflet/` changed and with it the HLO hashes, not the outputs. Timed A and B hit those new NEFFs. | Here |
| LTX-2 | C's timed run recompiled 9 NEFFs in process, before its first step delta. C's compile had run at `3a570e2f`, before `2d6714cf`. The step medians are unaffected. | Here |
| LTX-2 | The "5 of 89 NEFFs differ" are stale outputs of C's compile step. Timed D differs from timed C only by the 4 const-max block NEFFs, which strengthens D/C. | Here |
| LTX-2 | D's smoke compiled 4 small NEFFs; D's timed run compiled nothing. | Here |
| Wan | Confirmed, no corrections. Issues carried: | |
| | - "largest gain of E1" is out of date (HV1.5 0.632); | Section 4, ranking |
| | - B lacks the bowl in all three sampled frames; | Section 2, Wan 2.1 |
| | - the routed sigma max is 2.56, from Wan 2.2's low-noise expert; | Section 2, Wan 2.2 |
| | - the 97% is cross-session, so approximate. | Table 2; section 5 |
| HV1 | Nothing to check (no arm timed). Arm A's compile was a NEFF cache miss (neuronx-cc at +34 s), not cache hits. HV1 is pending, not 1.00x. | Section 1, HV1 row; section 2, HV1 |
| HV1, 2nd attempt | Nothing to check again (no arm timed); no corrections. Issues carried: | |
| | - A is not ready for timing. The killed recheck overwrote `A/compile.env.txt` (no `exit 0`), and `timed_chain.sh` would skip A, so all four compiles must re-run. The lane's "A compiled, exit 0" holds only for `compile_1`; | Section 1, HV1 status; section 2, HV1 (Next); section 4 |
| | - A's and C's md5 are expected to differ from the *evidence* (`c13fda35`); a mismatch is not a const-max finding; | Section 2, HV1; section 5 item 3 |
| | - `hv1_collect.py`'s warm step is the mean of the two request medians, not E1's request-2 median; | Header (Arms); section 2, HV1 |
| | - the `work/worker_0` latents-path concern is probably unfounded (the HV1.5 and Wan 2.1 timed arms wrote there); unconfirmed for HV1; | Section 2, HV1 |
| | - device-level routing is open: B's and D's NEFF and HLO keys must differ from A's and C's. | Section 1, HV1 routed cell; section 2, HV1; section 4 |
| | Check note: the 1,200 s `neff_local` wait is the lane's figure. The log shows the stale-lock break, not the wait length. | Superseded by the 3rd attempt |
| HV1, 3rd attempt | Corrected: device routing is shown directly, not only indirectly (the lane doc, sections 4 and 9). Decompressing each NEFF (gzip from byte 1024) finds `attention_const_max` / `_kernel` / `_lnc` and no `attention_cte` in B's `d190`/`dfa4` and D's `81d5`/`a8c6`; A's `03c8`/`df55` and C's `35a3`/`4b3b` hold only `attention_cte` (24 against 12 hits). | Section 1, HV1 routed cell; section 2, HV1 (Routing) |
| HV1, 3rd attempt | Corrected: compile peak host RSS is A 34.4 GB, B 57.3 GB, C 69.4 GB, D 57.3 GB. `/usr/bin/time -v` reports KiB; the lane doc's 33.6 / 55.9 / 67.8 / 55.9 divided by 1e6. Cosmetic; no JSON number changes. | Section 2, HV1 (No-head-split risk) |
| | Issues carried: | |
| | - step values are per-request medians (n = 49) and maxes; raw per-step times are not kept (each max is within 0.3% of its median); | Section 2, HV1 (Timing caveats) |
| | - the lane doc's MXFP8 glue sum (~8,635 ms) leaves out the single-block term: 8,635-8,801 ms against 9,303 ms measured; | Section 2, HV1 (MXFP8 arms); section 5 item 3 |
| | - this summary listed HV1 as not timed (*inferred* ~0.73-0.74) and HV1.5 as the largest measured gain; HV1's measured 0.4975 supersedes both; | Header; sections 1, 2 and 4 |
| | - B's quality question is beyond this protocol (one seed, no fp32 reference, no floor at this shape); | Section 2, HV1 (Floor); section 4 |
| | - A and B ran without `corenode.lock` (load averages 22 and 39 at B's and C's starts); no effect shows, but the arms are not interleaved; | Section 2, HV1 (Timing caveats) |
| | - not verified by the auditor: that the coordinator held `neuron5.lock`, HBM use, SSIM / LPIPS (PSNR and the latent metrics were recomputed), the CPU unit test (not run). | Here |
| | The 2nd attempt's issues are closed, and its HV1 notes are replaced by the measured results: all four compiles re-ran; A's md5 differs from 7f as expected; the `work/worker_0` latents are the timed ones; B's and D's NEFF keys differ from A's and C's. | Section 2, HV1 (Status, md5) |
| HV1.5 | The const-max route does not pad: the call is 8 x 51,275 and softmax_max is taken at 51,275, not at 51,328. | Section 1, shape; section 2, HV1.5 |
| HV1.5 | Issues carried: | |
| | - "largest of the campaign" held among the measured models only; HV1 (0.4975) now supersedes it; | Section 2 header; section 4 |
| | - the KERNEL lane's exact-call probe is the stronger routing evidence; | Section 2, HV1.5 |
| | - "differs only in encode thread count" is an inference (the trees differ too). | Section 1 notes |

## 7. Paths

- Lane evidence: `/mnt/nvme/difflet-exp/constmax-models/`:
  - `{qwen_image,flux,ltx_2,wan21,wan22,hunyuan_video,hunyuan_video_15}/`;
  - `kernel/` and `KERNEL.json`.
- Commits, per lane:
  - KERNEL: `2d6714cf`, `47c7df2d`;
  - Qwen: `3576d32b`, `63b0ae2c`, `409ea6ca`;
  - FLUX: `3a570e2f`, `b05c71d6`;
  - LTX-2: `0c3b0792`;
  - Wan: `7c41de7d`, `d166e799`;
  - HV1: `ec3856c1`, `69908bf9`, `e9361d0f` (the 3rd attempt's timed arms and lane doc);
  - HV1.5: `0a098276`, `306cdaa0`.
- Floors and references:
  - CPU fp32 references: `/home/ubuntu/fp8-bench/fp32_ref/{qwen_image,flux,ltx_2}`;
  - FLUX CPU bf16 floor: `fp32_ref/flux_cpu_bf16/score.json`;
  - the other floors are the *evidence* sections cited in the lane docs.

## 8. Against CPU fp32 at the campaign shapes (added 2026-11-21)

The CPU fp32 references of the four long video models at the shapes and settings of the device arms
(`docs/plans/2026-11-20-cpu-fp32-reference-long.md`: same prompt, negative prompt, CFG, steps, seed
and initial noise, md5-asserted; only the dtype, fp32, and the device, CPU, differ). Scored with
`scripts/attn_constmax/score.sh`: video PSNR (frame mean and worst frame), SSIM, LPIPS and latent
cosine of each arm's warm request against the reference. Measured; one prompt and one seed each.

**Wan 2.1 T2V 14B, 480x832x81, CFG 5.0, 50 steps** (reference wall time 15:57:51, 96 threads, nice 19)

| arm | vs fp32 PSNR (worst frame) | SSIM | LPIPS | latent cos |
|---|---|---|---|---|
| A bf16 | 20.15 (18.25) | 0.675 | 0.187 | 0.9527 |
| B bf16 + const-max | 20.33 (18.34) | 0.658 | 0.161 | 0.9573 |
| C mxfp8 | 17.83 (15.60) | 0.551 | 0.241 | 0.9245 |
| D mxfp8 + const-max | 17.82 (15.73) | 0.555 | 0.242 | 0.9257 |

**Wan 2.2 T2V A14B, 480x832x81, CFG 4.0 / 3.0, 40 steps** (reference wall time 15:19:09, 96 threads, nice 19)

| arm | vs fp32 PSNR (worst frame) | SSIM | LPIPS | latent cos |
|---|---|---|---|---|
| A bf16 | 11.00 (9.89) | 0.185 | 0.460 | 0.8433 |
| B bf16 + const-max | 10.87 (10.06) | 0.177 | 0.477 | 0.8302 |
| C mxfp8 | 12.41 (11.83) | 0.267 | 0.368 | 0.8983 |
| D mxfp8 + const-max | 11.98 (11.64) | 0.235 | 0.390 | 0.8840 |

**HunyuanVideo, 544x960x129, guidance 6.0, 50 steps** (reference wall time 22:49:19, 96 threads, nice 19)

| arm | vs fp32 PSNR (worst frame) | SSIM | LPIPS | latent cos |
|---|---|---|---|---|
| A bf16 | 15.36 (10.73) | 0.714 | 0.424 | 0.8998 |
| B bf16 + const-max | 17.53 (15.66) | 0.818 | 0.300 | 0.9424 |
| C mxfp8 | 15.31 (10.51) | 0.718 | 0.417 | 0.8974 |
| D mxfp8 + const-max | 15.29 (10.49) | 0.718 | 0.417 | 0.8972 |

**HunyuanVideo-1.5, 480x848x121, CFG 6.0, 50 steps** (reference wall time 19:59:29, 96 threads, nice 19)

| arm | vs fp32 PSNR (worst frame) | SSIM | LPIPS | latent cos |
|---|---|---|---|---|
| A bf16 | 16.09 (14.85) | 0.328 | 0.358 | 0.9156 |
| B bf16 + const-max | 14.55 (14.06) | 0.261 | 0.440 | 0.8665 |
| C mxfp8 | 12.52 (11.99) | 0.190 | 0.533 | 0.7653 |
| D mxfp8 + const-max | 12.48 (12.04) | 0.193 | 0.542 | 0.7685 |

Reading (measured numbers above; the readings are inferences from one prompt and one seed):

- **Wan 2.1.** B is as close to fp32 as A (+0.18 dB, latent cos 0.957 vs 0.953). C and D are about
  2.3 dB further: the MXFP8 linears, not const-max, move Wan 2.1 away from fp32. The earlier B vs A
  gap (22.21 dB) is a different but equally close sample, not a loss.
- **Wan 2.2.** Every arm is 10.9-12.4 dB from fp32, bf16 included: the 81-frame, two-expert,
  CFG 4.0 / 3.0 trajectory diverges, so one sample cannot rank the arms (C even lands closer than
  A). B and A agree within 0.13 dB.
- **HunyuanVideo.** B is 2.2 dB closer to fp32 than A (worst frame +4.9 dB, cos 0.942 vs 0.900),
  so the earlier "B is the outlier" against A was A's distance, not B's. D is no closer than C, so
  the gain is not attributable to const-max from one sample. Reading: no loss.
- **HunyuanVideo-1.5.** B is 1.5 dB further from fp32 than A (cos 0.867 vs 0.916), below the bf16
  floor pairs (15.9-18.4 dB between two bf16 runs); C and D are about 3.6 dB further. It is the one
  model with a possible const-max loss against fp32, smaller than MXFP8's, and one sample cannot
  separate it from trajectory spread.
- **All seven models.** Against fp32, const-max (B) is within 0.5 dB of bf16 (A) on Qwen-Image,
  FLUX, LTX-2, Wan 2.1 and Wan 2.2, 2.2 dB closer on HunyuanVideo, and 1.5 dB further on
  HunyuanVideo-1.5. On top of MXFP8 (D vs C) it is within 0.4 dB everywhere. A default change
  would still need a multi-seed, multi-prompt set on the video models, HunyuanVideo-1.5 first.
