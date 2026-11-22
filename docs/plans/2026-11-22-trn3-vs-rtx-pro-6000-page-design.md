# Trainium3 vs RTX PRO 6000 comparison page — design

Status: draft for review, 2026-11-22.

## Intent

A public-facing web page that puts one Trainium3 chip next to one NVIDIA RTX PRO 6000 for the
diffusion models Difflet runs. It has two jobs: compare fairly, and look good enough that people
outside the team want to share it.

## Goals, non-goals, anti-goals

- **Goals:** per-step DiT latency for every model where both sides have a number; output quality
  against the same fp32 reference; the method on the page, close to the numbers; a striking first
  screen (one big number) and one interactive element (a 50-step denoising race).
- **Non-goals:** re-measuring the RTX PRO 6000 (decided 2026-11-22: use what exists); cost per
  output; end-to-end latency comparison (the two sides measure it differently, see Constraints);
  other GPUs (H100, B200, B300); multi-prompt quality.
- **Anti-goals:** a number that reads stronger than its source. Every RTX figure derived from a
  whole-request time is marked as an upper bound; missing data says "not measured", never a
  stand-in from another GPU.

## Constraints

- trn3 numbers come from two internal reports (November 2026) that are not in the repo:
  "Trainium3 single-chip performance evaluation" and "Trainium3 native backend: output quality
  evaluation". The figures used are copied into the appendix below.
- RTX PRO 6000 speed exists only in `docs/verification/gpu-backend-results.md`,
  `docs/verification/hunyuanvideo-cuda.md` and `docs/verification/hunyuanvideo15-cuda.md`.
  FLUX.1-dev and Qwen-Image have a measured DiT forward; Wan 2.2, HunyuanVideo 1.0 and 1.5 only
  have whole-request times, so their per-step values are upper bounds. LTX-2 and Wan 2.1 have no
  RTX PRO 6000 data.
- trn3 warm e2e is a fresh `generate` process (launch, load, warm-up, encode, 50 steps, decode,
  save); the RTX whole-request times exclude loading. They are not comparable, so the page
  compares per-step only.
- Quality figures are one prompt and one seed. The quality report says such figures detect
  regressions but do not characterise accuracy (Qwen-Image moves from −6.1 dB to +1.5 dB with a
  second prompt). The page shows them with that caveat next to them (user: "有什么给什么").
- Must work at phone width and respect `prefers-reduced-motion`.

## Approaches

1. **Self-contained static HTML in the repo (chosen).** One file with its data inline, vanilla
   JS, no build step and no dependencies beyond Google Fonts. It opens from disk, can be hosted on
   GitHub Pages or S3, and is versioned with the numbers it shows.
2. claude.ai artifact only. Fastest to share, but private by default and tied to one claude.ai
   account (the mockups already hit this twice), so not a public website. Used only as a preview.
3. Generate the page from `benchmark/*.json`. The data is not in that format (it comes from
   reports and verification docs), and a generator for seven rows is more code than the page.

Format, chosen from mockups A–H on the design canvas: mockup H, the keynote skeleton (E) with the
race (D) on top. Canvas: https://claude.ai/artifact/NUfD8XtkjvC5EkJ59KGgqj#artboard-Trn3VsRTX6000

## Design

- **File:** `docs/compare/trn3-vs-rtx-pro-6000.html`, one file. A `DATA` object at the top of
  its script holds every number with its source string; all ratios, chips and the hero number are
  computed from it, never typed into the markup.
- **Sections, top to bottom:**
  1. Hero: FLUX.1-dev per-step ratio, trn3 best layout against RTX compiled (241 / 213.9 =
     1.13×), with the eager ratio (1.42×) underneath; three tiles: 7 models on one chip, 5 of 8
     configurations at or above the RTX bf16 floor, 1.15–1.49× over Trainium2.
  2. Race: pick a model with RTX data (FLUX, Qwen-Image, Wan 2.2, HunyuanVideo 1.0, 1.5), RTX
     compile/eager toggle, two lanes replaying 50 steps at per-step speed (playback scaled so the
     slower lane takes ~8 s), a noise-to-image reveal, the winner and margin; upper-bound rows say
     so under the result.
  3. Per-step table, all 7 models: trn3 TP4, trn3 best layout, RTX eager, RTX compiled,
     "trn3 best vs RTX compiled" chip, RTX source.
  4. Quality: one dumbbell row per configuration on a dB axis, trn3 vs fp32 and RTX bf16 vs fp32,
     gap chip, SSIM; the single-prompt caveat and the LTX-2 fp32-timestep figure below.
  5. Method: hardware, stack, step definition, settings, quality method; a data note (RTX not
     re-measured, upper bounds, missing models, why no e2e).
- **Look:** dark ground (#050608), trn3 orange #FF8A3D, RTX blue #5AA9FF; Space Grotesk numbers,
  Inter / Noto Sans SC text, JetBrains Mono figures; scroll reveal via CSS `animation-timeline`
  behind `@supports` and `prefers-reduced-motion`.
- **Language:** English (my call, for a public page; the mockup copy is Chinese and gets
  translated). Say if it should stay Chinese or be bilingual.
- **Check:** a small script asserts the computed hero ratio, table chips and the 5/8 count against
  the appendix values, so an edited number cannot silently change a claim.

## Left to the builder

Exact spacing and type scale, the race art (abstract placeholder unless real outputs are
supplied), animation timing.

## Appendix: the numbers

Per-step DiT latency, ms (trn3: median of 49 step intervals, 50 steps, seed 42, one chip):

| Model, configuration | trn3 TP4 | trn3 best (layout) | RTX eager | RTX compiled | RTX source |
|---|---:|---:|---:|---:|---|
| FLUX.1-dev, 1024² | 221.4 | 213.9 (TP4+SP) | 303 | 241 | DiT forward, gpu-backend-results.md |
| Qwen-Image, 1024² | 302.5 | 280.4 (TP4+SP) | 323 | 243 | DiT forward, gpu-backend-results.md |
| Wan 2.2, 480×832×9 | 872.2 | 754.7 (TP2×CFG2+SP) | ≈1,072 | ≤820 | 2 × 536 ms forward; 32.8 s ÷ 40 steps |
| HunyuanVideo 1.0, 320×512×61 | 724¹ | 664¹ (TP4+SP) | ≤1,312.5 | ≤1,228.8 | 65.626 / 61.440 s ÷ 50, hunyuanvideo-cuda.md |
| HunyuanVideo 1.5, 480×848×121 | 9,914 | 9,914 (TP2×CFG2) | ≤8,246 | ≤8,159 | 412.3 / 407.95 s ÷ 50, hunyuanvideo15-cuda.md |
| LTX-2, 480×704×49, CFG 3.5 | 557.9 | 495.9 (TP2×CFG2) | — | — | not measured |
| Wan 2.1, 480×832×9, CFG 5.0 | 872.8 | 754.1 (TP2×CFG2+SP) | — | — | never run on a GPU |

¹ (T₅₀ − T₁₀) / 40; that pipeline records no in-run step times.

Trainium2 → Trainium3 per step, same stack and layout: 1.15–1.49× (table in the performance
report).

Quality, PSNR dB / SSIM against diffusers fp32 on the RTX PRO 6000 (TF32 off), one prompt, seed
42, bit-identical initial latents:

| Model | Layout | trn3 vs fp32 | RTX bf16 floor vs fp32 | Gap |
|---|---|---|---|---:|
| FLUX.1-dev | TP4 | 16.04 / 0.773 | 15.96 / 0.772 | at |
| Wan 2.2, high-noise expert | TP4 | 23.88 / 0.858 | 23.22 / 0.848 | +0.7 |
| Wan 2.2, both experts | TP4 | 24.95 / 0.920 | 24.05 / 0.854 | +0.9 |
| Wan 2.1 | TP4 | 24.34 / 0.913 | 18.92 / 0.823 | +5.4 |
| HunyuanVideo 1.0 | TP4 | 21.81 / 0.913 | 19.51 / 0.865 | +2.3 |
| HunyuanVideo 1.5 | TP2×CFG2 | 30.25 / 0.958 | 32.58 / 0.970 | −2.3 |
| Qwen-Image | TP4 | 28.39 / 0.976 | 34.51 / 0.993 | −6.1 |
| LTX-2 | TP4 | 26.84 / 0.928 | 34.12 / 0.964 | −7.3 |

LTX-2 with the fp32 timestep fix: 27.62 / 0.939.

Hardware and stacks: trn3.48xlarge, one Trainium3 chip (4 logical cores at LNC=2, 36 GB HBM per
core), torch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc 2.28. RTX PRO 6000 Blackwell Server
Edition 96 GB, Difflet CUDA backend, torch 2.11.0+cu128 (FLUX, Qwen, Wan) and torch 2.13.0+cu130
(HunyuanVideo).
