# MXFP8 + const-max + ScalarE exp + TeaCache on trn3 (arm E): evidence

Plan: `docs/plans/2026-11-22-mxfp8-constmax-teacache-trn3.md`. Branch `exp/arm-e-mxfp8-constmax-teacache`
(main 5727fe76 + this campaign). Run 2026-11-22 on one trn3.48xlarge, LNC=2, seed 42, one process with two
identical requests per arm (request 2 is the warm one). Every number is **measured** here unless it says
**reused** (the 2026-11-18 const-max campaign, `/mnt/nvme/difflet-exp/constmax-models/`).

## Arms

| arm | precision | attention | TeaCache | layout |
|---|---|---|---|---|
| A (reused) | bf16 | attention_cte, softmax exp on VectorE (the 11-18 trees predate main #164) | off | tp4; HV1.5 tp2 x CFG2 |
| A_cat (HunyuanVideo only) | as A | as A, from a snapshot of 11-18's tree ceab09f4 | off | tp4 |
| D (reused) / B for HV1.5 | mxfp8 (HV1.5: bf16) | const-max | off | as A |
| **E1** | mxfp8 (HV1.5: **no MXFP8**, bf16) | `--attention const-max`; the calls left on attention_cte run ScalarE exp (#164) | `--teacache-online-delta 0.6` | as A |
| **E2** | as E1 | as E1 | `--teacache-cadence 2` | as A |

The const-max kernel's own exp runs on VectorE on trn3 (no engine switch). ScalarE exp applies only to the
calls that stay on attention_cte: the unrouted layers (Qwen-Image 3 of 60, FLUX 0 of 57, LTX-2 3 of 48 video
attn1, Wan 4 of 40 attn1 per expert, HunyuanVideo 5 of 60, HV1.5 2 of 54) and every cross-attention and
audio call. So E differs from 11-18's D (HV1.5: B) by TeaCache and by ScalarE exp in those calls.

Settings (as 11-18): Qwen-Image 1024x1024 50 steps unguided ("a cat sitting on a bench"); FLUX.1-dev
1024x1024 50 steps guidance 3.5 (same prompt); LTX-2 512x768x121 50 steps; Wan 2.1 480x832x81 50 steps CFG 5.0;
Wan 2.2 480x832x81 40 steps CFG 4.0 / 3.0 two experts; HunyuanVideo 544x960x129 50 steps guidance 6.0;
HV1.5 480x848x121 50 steps CFG 6.0 (video prompt "a cat walking through a garden"; HunyuanVideo moved from
11-18's fox prompt to this one, so its baseline is A_cat).

## 1. Speed

DiT step = the **effective** step: (request time - the first-step segment - after the last step) / the steps
it covers (49, Wan 2.2 39; HV1.5 denoise / 50). Full step = the log's step median (a full DiT step; fewer than
half the steps are skipped in every arm). Warm e2e = request 2 by each lane's 11-18 definition (Qwen-Image and
Wan: request + parent decode-and-save; FLUX, HunyuanVideo: request; HV1.5: encode + denoise + decode on one
VAE core like A; LTX-2: request without the host VAE decode, with it in brackets).

| model | A step (ms) | E1 step | **E1/A** | E2 step | **E2/A** | E1 / E2 full step | A warm e2e (s) | E1 e2e | **E1/A** | E2 e2e | **E2/A** | skipped E1 / E2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen-Image | 304.7 | 163.3 | 0.536 | 127.6 | 0.419 | 215.3 / 214.1 | 16.23 | 9.60 | 0.591 | 7.68 | 0.473 | 12 / 20 of 50 |
| FLUX.1-dev | 222.6 | 98.8 | 0.444 | 98.2 | 0.441 | 164.7 / 164.4 | 11.55 | 5.41 | 0.468 | 5.37 | 0.465 | 20 / 20 of 50 |
| LTX-2 | 1,287.1 | 559.2 | 0.434 | 522.7 | 0.406 | 881.9 / 880.0 | 74.81 (113.67) | 40.79 (74.68) | 0.545 (0.657) | 38.62 (73.97) | 0.516 (0.651) | 18 / 20 of 50 |
| Wan 2.1 | 10,107.5 | 5,011.4 | 0.496 | 4,688.4 | 0.464 | 7,892.8 / 7,888.1 | 526.24 | 273.93 | 0.521 | 258.12 | 0.491 | 18 / 20 of 50 |
| Wan 2.2 | 10,115.2 | 7,104.6 | 0.702 | 5,081.0 | 0.502 | 7,901.7 / 7,890.9 | 425.49 | 305.55 | 0.718 | 226.75 | 0.533 | 4 / 14 of 40 (derived) |
| HunyuanVideo (A_cat) | 18,488.0 | 5,022.7 | 0.272 | 5,018.8 | 0.271 | 8,468.5 / 8,465.9 | 1,003.41 | 333.30 | 0.332 | 333.09 | 0.332 | 20 / 20 of 50 |
| HV1.5 (**no MXFP8**) | 9,917.5 | 4,104.0 | 0.414 | 3,730.0 | 0.376 | 6,214.5 / 6,211.0 | 564.2 | 275.3 | 0.488 | 257.0 | 0.456 | 17 / 20 of 50 |

- A_cat (HunyuanVideo, cat prompt, 11-18 tree): step 18,488.4 / 18,487.6 ms, request 2 1,003.41 s; the fox A was
  18,485.6 ms / 1,003.04 s (+0.01%): the prompt does not change the step.
- E's full step against 11-18's D (HV1.5: B): Qwen 215.3 vs 216.8, FLUX 164.7 vs 165.1, LTX-2 881.9 vs 894.8,
  Wan 2.1 7,892.8 vs 8,018.4, Wan 2.2 7,901.7 vs 8,018.6, HunyuanVideo 8,468.5 vs 8,648.9, HV1.5 6,214.5 vs
  6,272.5 (-0.3% to -2.1%; the cte calls on ScalarE).
- Wan 2.2's skip counts are derived from timing (steps - round(loop time / full step)): its controller resets at
  the expert switch, so the stats line covers the low-noise stage only (E1: 27 full / 0 skipped; E2: 17 / 10).
  Online-delta re-takes its baseline at the switch and skips nothing after it, hence E1's 0.70x.

## 2. Quality against CPU fp32 (single prompt, single seed)

PSNR (frame mean; worst frame) / SSIM against the diffusers CPU fp32 reference at the same settings
(`/home/ubuntu/fp8-bench/fp32_ref/{qwen_image,flux,ltx_2}`, `/mnt/nvme/difflet-exp/fp32_ref_long/*_long`),
scorer `scripts/compare_difflet_runs.py` as for 11-18's arms (`/mnt/nvme/difflet-exp/arm-e/score_e.sh`).

| model | 11-18 vs fp32: A / B / C / D (dB, reused) | **E1 vs fp32** | **E2 vs fp32** | E1 / E2 vs A (dB) | E1 / E2 vs D (HV1.5: B) |
|---|---|---|---|---|---|
| Qwen-Image | 36.34 / 35.91 / – / – | 36.41 / SSIM 0.990 | 35.83 / 0.991 | 33.80 / 34.10 | 40.32 / 37.30 |
| FLUX.1-dev | 23.36 / 23.31 / 21.30 / 22.11 | 21.50 / 0.888 | 23.67 / 0.912 | 22.74 / 27.00 | 23.71 / 24.83 |
| LTX-2 | 16.55 / 16.54 / 16.85 (C) / – | 16.61 (15.32) / 0.464 | 16.92 (15.59) / 0.489 | 23.46 / 22.97 | 24.64 / 23.69 |
| Wan 2.1 | 20.15 / 20.33 / 17.83 / 17.82 | 18.09 (17.07) / 0.586 | 18.33 (17.59) / 0.597 | 18.53 / 18.75 | 21.17 / 22.85 |
| Wan 2.2 | 11.00 / 10.87 / 12.41 / 11.98 | 11.85 (11.25) / 0.247 | 12.34 (11.85) / 0.263 | 11.24 / 11.31 | 13.04 / 13.59 |
| HunyuanVideo (cat) | pending: the cat-prompt fp32 reference (~23 h, started 11-22 13:3x) | pending | pending | 21.39 / 23.57 | – (D ran the fox prompt) |
| HV1.5 (no MXFP8) | 16.09 / 14.55 / 12.52 / 12.48 | 15.30 (14.26) / 0.289 | 15.28 (14.40) / 0.297 | 15.27 / 15.29 | 15.86 / 15.02 |

Reading (single seed; the 11-18 A-D range is the yardstick):

- Qwen-Image: E at A's level against fp32; TeaCache costs nothing measurable.
- FLUX: E2 at A's level; E1 1.9 dB below A, at the CPU bf16-vs-fp32 floor (21.47 dB). Both skip 20 steps, different
  ones. One image: within FLUX's bf16 tp2-vs-tp4 floor (29.4 dB), not a ranking.
- LTX-2: every arm sits at the bf16 host-state floor (~16.5 dB; fix 81a6c597 is not on this tree, so A and E are
  comparable). E vs A 23.5 / 23.0 dB is below the bf16 tp2-vs-tp4 floor (25.9 dB) and in line with MXFP8 alone vs
  bf16 (22.5 dB, 11-07).
- Wan 2.1: E at D's level (MXFP8 costs ~2 dB, as on 11-18); TeaCache adds nothing.
- Wan 2.2 and HunyuanVideo-class trajectories are chaotic at these shapes (11-18's arms scatter by ~2 dB, not
  monotonic); differences under 2 dB are not conclusions.
- HV1.5 (no MXFP8): E between A and B, above B.

## 3. Validity checks (all passed, every model)

- Request 1 and 2 outputs identical (md5) in every E arm; all ranks printed identical `[teacache] stats`
  (4 ranks x 2 requests).
- MXFP8 quantized directories reused from 11-18 (digests: Qwen d1231d8ec61363a8, FLUX 809df193b011ba1f,
  LTX-2 916a153a8eeb2fa3, Wan 2.1 6d28df7cc78386b5, Wan 2.2 372c52a68c440756, HunyuanVideo 46e0952c1c5a9f74);
  Wan's routing line "36 of 40 self-attentions routed" for both experts; FLUX's cache holds 6 const-max NEFFs.
- No NEFF added during any timed generate.
- 14-step smoke with cadence 2 (Wan 2.2: 20): every rank 12 full / 2 skipped (Wan 2.2: 10 / 3 in the last stage).
  First TeaCache runs of Qwen-Image and LTX-2 on the neuron backend.
- HV1.5's decode 59.0 s in request 2 (A: 59.2 s): one VAE core, `--vae-cores 1` reaches the router's worker.

## 4. What went wrong and was fixed during the run

- Wan 2.1 / 2.2's first compile failed in 16 s: the runner sent every lane to the NVMe hub, but the 11-18 Wan
  lanes read the default hub, and the bf16 copy's manifest (hence the MXFP8 digest) holds the source path.
  Fixed in 7e5ee706 (Wan lanes use the default hub); the copies and digests then matched 11-18.
- HunyuanVideo's E1 compile waited 20 min on lock files it shared with A_cat's hard-linked (`cp -al`) cache copy.
  Time only; each process wrote its own files.
- HV1.5 rebuilt its bf16 copy from the NVMe hub (same snapshot revision, same bits; HV1.5 does not quantize).

## 5. Provenance

Code snapshots: `src-c46b0c0c` (Qwen-Image, FLUX, LTX-2, HunyuanVideo, HV1.5), `src-7e5ee706` (Wan 2.1, 2.2).
Devices: E1 on each model's 11-18 device (0-6, 12 threads on NUMA node 1, same slices as A); E2 and smokes on
8-15 (NUMA node 0); A_cat on device 7. Runner `scripts/arm_e/run.sh`, collector `scripts/arm_e/collect.py`
(`--self-check` reproduces the seven A rows from the 11-18 logs). md5 of the reused 11-18 results files:
flux 3a84216f, hunyuan_video 5d14e92d, hunyuan_video_15 1c2f08d2, ltx_2 50a2425d, qwen_image a35511fa,
wan21 a91f407a, wan22 461a9923.
