# Const-max attention at model level, LTX-2 lane (four arms, one session, trn3)

Campaign: nkilib `attention_const_max` (bf16 attention with a fixed per-layer softmax max) on
every model, branch `exp/attn-constmax-models`. This lane runs LTX-2's four arms back to back.
E1 (`docs/verification/2026-11-18-attn-constmax.md`, section 6) ran two of them (A and B). C and
D are new here.

Every number is **measured** on this host in this run unless it is marked *inferred* or
*evidence* (quoted from an earlier doc, not re-run).

## 1. Summary

LTX-2, 512x768x121, tp4, 50 steps, seed 42, prompt "a cat walking through a garden" (bench.sh's
`ltx_2` key). Device 2 (cores 8-11), CPUs 60-65,156-161. Clean under `difflet/` for all four
timed arms; no `difflet/` change between them (section 4).

| arm | warm DiT step (ms) | ratio | warm request (s) | mp4 md5 (both requests) | vs A: PSNR mean (worst) / SSIM / LPIPS / video latent cos | vs CPU fp32: PSNR mean (worst) / SSIM / LPIPS |
|---|---|---|---|---|---|---|
| A bf16 (shipped `attention_cte`) | **1287.1** | 1 | 74.81 | `0702ffd2…` = evidence section 2 | - | 16.55 (15.13) / 0.4716 / 0.1948 |
| B bf16 + const-max | **1219.6** | **B/A 0.948** | 72.76 | `e81078e1…` = E1 | 27.63 (25.33) / 0.9307 / 0.0292 / 0.99947 | 16.54 (15.16) / 0.4708 / 0.1947 |
| C `--quant mxfp8` | **973.7** | C/A 0.757 | 59.93 | `c1057ed5…` = evidence section 3 | 22.47 (20.85) / 0.8301 / 0.0780 / 0.99093 | 16.85 (15.54) / 0.4854 / 0.1898 |
| D `--quant mxfp8` + const-max | **894.8** | **D/A 0.695, D/C 0.919** | 55.42 | `ce166b9d…` (new) | 23.04 (21.24) / 0.8469 / 0.0721 / 0.99266 | 16.68 (15.29) / 0.4667 / 0.1947 |

- **Gain.** Const-max saves 67.6 ms per step on bf16 (0.948x) and 78.9 ms on top of MXFP8
  (0.919x). MXFP8 linears and const-max together take a step to 0.695x of shipped bf16.
- **Quality.**
  - B holds: 27.6 dB from A, tighter than bf16's own tp2-vs-tp4 floor (25.9 dB, evidence 3a),
    and level with A against fp32 (16.54 vs 16.55).
  - D is 25.15 dB from C (worst frame 23.28, video latents rel L2 0.074). That is at the bf16
    floor (25.9 / 23.6 / 0.071), marginally past it.
  - Against fp32, D (16.68 dB) sits between A (16.55) and C (16.85). Its SSIM (0.4667) is
    0.005 below A's and 0.019 below C's. One clip cannot rank 0.2 dB differences (section 6).
- **Reproducibility.** All four arms are deterministic: in each arm both requests give the same
  mp4 and `torch.equal` latents, video and audio. A, B and C equal their earlier runs' md5s:
  A = evidence section 2 (bf16 `0702ffd2`), B = E1, C = evidence section 3 (mxfp8 `c1057ed5`).
  So A's md5 **equals** the evidence doc's bf16 md5.

## 2. Method

**Hook.** E1's, unchanged (E1 section 2):

- With `DIFFLET_EXP_ATTN_CONSTMAX=1`, `_LTX2TransformerTraceModule` registers a non-persistent
  `[128, 1]` buffer `constmax_sigma` on each video `transformer_blocks.{i}.attn1` whose table row
  passes the gap rule.
- The TP processor's self-attention branch passes it to `ops_impl.attention`.
- Unbounded, non-causal, bf16, d = 128 calls go to `difflet_neuron::attention_const_max`
  (nkilib 2.14 `experimental/attention/attention_const_max.py`, unmodified), with
  softmax_max = 0.88 sigma sqrt(2 ln Sk) computed in-graph as a tensor.
- With `--quant mxfp8` the q/k/v projections are MXFP8 linears whose outputs are bf16, so the
  attention call meets the same conditions. D's routing is shown by its cache: it holds four
  block NEFFs that C's does not, about 4.83 MB each against about 5.59 MB for the cte block
  NEFFs, one per rank. That matches E1's bf16 case: one new block graph shared by the 45
  routed layers.

**Routed.** 45 of the 48 video `attn1` layers. `transformer_blocks.44/45/47.attn1` fail the gap
rule (gaps 96.2 / 134.8 / 112.4 > 80) and stay on `attention_cte`. Not routed: the audio stream,
every cross-attention, and LTX-2 at tp1.

**softmax_max table provenance.** `difflet/models/ltx_2/attn_constmax.json`, written by
`scripts/attn_constmax/table.py` from the QK-RMSNorm gains of hub snapshot `64f65d09…`:

- sigma = sqrt(sum(g_q^2 g_k^2) / d);
- hard_bound = sqrt(d) max|g_q| max|g_k|;
- `ok` when hard_bound - softmax_max <= 80 at the reference Sk 6,144.

Sigma runs from 0.288 to 3.931. The table is E1's, unchanged and cross-checked against
`bounds.json` (48 of 48). No value was tuned against any output.

**Arms.** One runner, `scripts/attn_constmax/ltx2_lane.sh <bf16|cm|mx|mxcm>
<compile|smoke|timed>`. It uses `arms.sh`'s LTX-2 flags and modes, adds arm C (`mx`), and writes
outputs under `/mnt/nvme/difflet-exp/constmax-models/ltx_2/<arm>/`.

- **Caches.** A and B reuse E1's caches (`/mnt/nvme/difflet-exp/constmax/cache/ltx_2-{bf16,cm}`).
  C and D each get their own (`.../constmax-models/ltx_2/cache/{mx,mxcm}`). `--quant` turns on
  the NKI trace cache, and its HLO cache lives under each arm's `--cache-dir`.
- **Env.** A and C run with `DIFFLET_EXP_ATTN_CONSTMAX` unset. B and D run with it set to 1.
- **Locks and checks.** Each step runs under device 2's `flock` (`chain.sh`) and refuses to
  start if `fuser /dev/neuron2` shows a holder. Every step also had
  `NEURON_RT_VISIBLE_CORES=8-11`, `taskset -c 60-65,156-161` and `NEURON_NUMA_PIN=0`.
- **Warm step** = the median, over the two requests, of each request's DiT step median (n = 49
  each, the timing hook in `difflet/cli/orchestrators/torchrun.py`). Spread = (max - min) / min
  of the two.
- **Warm request** = request 2's `request:` time: prompt encode, plus 50 steps, plus the
  scheduler. LTX-2's host VAE decode runs after the stage, outside it (section 6).

## 3. Gate (before timing)

**Compile and warm-up**, untimed, all on device 2:

| arm | wall | load + warm-up | blocks / fallbacks | note |
|---|---|---|---|---|
| C mx | 6:58 | 364.9 s | 96 / none | new; quantized DiT `mxfp8-916a153a8eeb2fa3` |
| D mxcm | 12:02 | 669.4 s | 96 / none | new |
| A bf16 | 2:02 | 113.8 s | 96 / none | E1's cache, hit |
| B cm | 2:51 | 162.2 s | 96 / none | E1's cache, hit |

**10-step smokes.**

- **B** (`cm/smoke.mp4`): latents finite. Video `[1, 6144, 128]` bf16 has std 1.19 and |max|
  5.56; audio `[1, 126, 128]` has std 0.98 and |max| 5.72.
  - The mp4 md5 `24754043…` is **bit-identical** to E1's B smoke.
  - The frame strip (`cm/smoke_strip.png`, frames 0, 60 and 120) shows E1's grassy field with a
    dark cat, and no noise.
  - Step median 1218.3 ms (n = 9). **Gate: passed.**
- **D** (`mxcm/smoke.mp4`, `7fd307c4…`; run as well, since D had never been run): latents finite.
  Video has std 1.17 and |max| 5.41; audio has std 0.99 and |max| 5.69.
  - The strip (`mxcm/smoke_strip.png`) shows the same scene and the cat in the same places,
    with no noise. The washed-out look is LTX-2 at 10 steps, in every arm (E1 3c).
  - Step median 893.5 ms (n = 9). **Gate: passed.**

The 50-step timed outputs of all four arms were also checked (`timed_latents_finite.txt`,
`timed_strips_A_B_C_D.png`: frames 0, 60 and 120 of request 2, rows A to D). The latents are
finite, and every clip shows the same garden path and white cat with no noise. No arm is broken.

## 4. Timing

Back to back on device 2, CPUs 60-65,156-161, from 09:57:00 to 10:16:52. One process per arm,
two identical requests each. `fuser /dev/neuron2` was idle and the CPU slice was 100% idle at
each start (`<arm>/timed.env.txt`).

| arm | request 1 step median (max) | request 2 step median (max) | **warm step** | spread | request 1 / 2 (s) | tree |
|---|---|---|---|---|---|---|
| A bf16 | 1287.1 (1292.1) | 1287.1 (1292.0) | **1287.1** | 0.00% | 75.79 / 74.81 | `63b0ae2c` |
| B cm | 1219.5 (1222.8) | 1219.6 (1225.0) | **1219.6** | 0.01% | 72.94 / 72.76 | `b05c71d6` |
| C mx | 973.4 (979.1) | 974.0 (979.4) | **973.7** | 0.06% | 62.63 / 59.93 | `b05c71d6` |
| D mxcm | 896.0 (900.7) | 893.6 (898.0) | **894.8** | 0.27% | 56.57 / 55.42 | `409ea6ca` |

- **Ratios** (warm step): B/A **0.948**, C/A 0.757, D/A **0.695**, D/C **0.919**. On warm
  request time: B/A 0.973, C/A 0.801, D/A 0.741, D/C 0.925. The request includes a prompt encode
  of 11.6-14.9 s, which varies by up to 3 s between requests.
- **Trees.** Other lanes committed between the arms. `git diff 63b0ae2c..409ea6ca -- difflet` is
  empty: only their docs and scripts changed. `difflet/` was clean at every start.
- **The KERNEL lane's `2d6714cf`** (09:40, `attention.py`: a key-bounded fork for *bounded*
  calls) landed before every timed arm. It changed nothing here:
  - unbounded calls keep E1's op;
  - B's smoke after it is bit-identical to E1's;
  - A, B and C reproduce their earlier md5s;
  - D's smoke and timed runs hit the cache D compiled at 09:34 (no recompile).
- **Cross-checks (evidence).**
  - A ran 1282.6 ms in E1 (device 1) and 1283.9 ms in the evidence's section 2; here 1287.1
    (+0.35%).
  - B ran 1217.3 ms in E1.
  - C ran 966.4 / 968.2 ms in the evidence's section 3; here 973.7 (+0.6%). The quantized
    directory key differs from the evidence's (`916a153a…` against `825b78f4…`), but the output
    md5 is the same.
- *Inferred.* Per routed call (45 per step), const-max saves 1.50 ms in bf16 and 1.75 ms on top
  of MXFP8. The kernel-level prediction at this per-rank shape is 0.91 ms per call (E1 section 6:
  shipped cte 2.0604 ms, const-max 1.1495 ms). The model-level gain exceeds it, as it did in E1,
  and more so with MXFP8. One possible reason is the kernel docstring's: without the P-transpose
  DMA, the kernel no longer serializes with the collectives and DMA around it, and the MXFP8
  linears add DMA. No profile was taken.

## 5. Quality

Scored on each arm's request 2 (`scripts/attn_constmax/score.sh` with
`ROOT=/mnt/nvme/difflet-exp/constmax-models`, plus `compare_difflet_runs.py` for D vs C).
mp4 against mp4: per-frame mean over 121 frames, worst frame, SSIM and LPIPS. The final packed
latents come from the work dirs. The fp32 reference is `/home/ubuntu/fp8-bench/fp32_ref/ltx_2`:
CPU fp32 diffusers, same bf16 noise.

| pair | PSNR mean (worst, frame) | SSIM | LPIPS | video latents cos / rel L2 | audio latents cos / rel L2 |
|---|---|---|---|---|---|
| **B vs A** | **27.63** (25.33, #120) | 0.9307 | 0.0292 | 0.99947 / 0.032 | 0.99993 / 0.012 |
| C vs A | 22.47 (20.85, #92) | 0.8301 | 0.0780 | 0.99093 / 0.134 | 0.99984 / 0.019 |
| **D vs A** | **23.04** (21.24, #78) | 0.8469 | 0.0721 | 0.99266 / 0.121 | 0.99983 / 0.018 |
| **D vs C** | **25.15** (23.28, #120) | 0.8872 | 0.0476 | 0.99723 / 0.074 | 0.99991 / 0.014 |
| floor: bf16 tp2 vs tp4 (evidence 3a) | 25.9 (23.6) | 0.909 | 0.040 | 0.9975 / 0.071 | 0.9999 / 0.014 |
| A vs fp32 (evidence 16.55 (15.13)) | 16.55 (15.13, #118) | 0.4716 | 0.1948 | 0.93524 / 0.374 | 0.98429 / 0.215 |
| **B vs fp32** | **16.54** (15.16) | 0.4708 | 0.1947 | 0.93524 / 0.374 | 0.98446 / 0.214 |
| C vs fp32 (evidence 16.85 (15.54)) | 16.85 (15.54) | 0.4854 | 0.1898 | 0.94082 / 0.354 | 0.98354 / 0.216 |
| **D vs fp32** | **16.68** (15.29) | 0.4667 | 0.1947 | 0.93649 / 0.367 | 0.98314 / 0.220 |

**bf16 floor** (evidence `2026-11-07-mxfp8-video-models.md`): bf16 tp2 vs tp4 is 25.9 dB
(worst 23.6), SSIM 0.909, LPIPS 0.040, video latents rel L2 0.071 (3a). Against fp32, bf16 is
16.55 dB and mxfp8 16.85 (section 10). That distance is dominated by LTX-2's host-side bf16
state (2026-11-12 finding: 21.3 dB with fp32 host state), which limits what the fp32 column can
resolve.

- **B (bf16 + const-max) holds quality.** It is closer to A than bf16 at another TP degree is
  (27.6 against 25.9 dB; latents rel L2 0.032 against 0.071), and level with A against fp32
  (-0.01 dB, video latents cosine equal to 5 digits). Same numbers as E1, bit for bit.
- **D (mxfp8 + const-max) at the floor.**
  - On top of MXFP8, const-max moves the clip by about the bf16 reduction-order floor: 25.15 dB
    against 25.9, worst frame 23.28 against 23.6, latents rel L2 0.074 against 0.071. That is
    a larger move than B's on bf16, slightly past the floor on each measure.
  - D is *closer* to A than C is (23.04 against 22.47 dB, latents cosine 0.99266 against
    0.99093).
  - Against fp32, D gives back about half of C's PSNR edge over A (16.68 against C 16.85 and A
    16.55). Its SSIM (0.4667) is the lowest of the four, 0.005 below A. Its LPIPS equals A's.
  - Reading (*inferred*): D is no further from fp32 than shipped bf16 in PSNR and LPIPS, and
    0.005 lower in SSIM. One clip and one seed cannot separate these differences from
    trajectory drift (FLUX evidence 6b: one image cannot rank 0.8 dB).
- **No arm is broken.** All latents are finite, and every clip shows the same scene and cat
  (section 3).

## 6. Caveats

- **One clip, one seed, one prompt.** D's 0.17 dB below C against fp32 and 0.019 SSIM are
  inside what one clip can resolve. A multi-prompt set was not run.
- **The kernel's error at real gains is up to about 7x cte's** at LTX-2's sigma 2.3-3.9 (E1 3a:
  bf16 scores in PSUM). The routed layers have sigma up to 3.8 > 1, outside the docstring's
  std <= 1 precondition. The softmax_max rule ignores RoPE and the activations, and the hard
  bound plus the gap <= 80 heuristic is the only guarantee. No overflow was seen.
- **Per-step series.** The stage logs only each request's median and max (n = 49). It does not
  log every step, so the raw evidence holds those, not a per-step series.
- **The warm request excludes the host VAE decode.** LTX-2's host decode runs after the stage.
  From the mp4 mtimes it is 37.5-38.9 s per clip in every arm, and the attention route does not
  touch it.
- **Shared host.** Other lanes compiled and ran on other devices throughout (load average
  55-134). Device 2 and its CPU slice were idle at every start. All four timed arms ran back to
  back on one device and one CPU set.
- **NEFF sets.** C's and D's caches were compiled independently. Besides D's four const-max
  block NEFFs, 5 of C's 89 NEFFs have a different hash in D's cache, at the same sizes within
  2 KB. This was not investigated. C's output equals the evidence's MXFP8 md5, so C is the
  shipped MXFP8 path.
- **Disk.** C's and D's caches (46 GB each) are kept under
  `/mnt/nvme/difflet-exp/constmax-models/ltx_2/cache/`. `/mnt/nvme` had 132 GB free after the
  run.

## 7. Paths

- Runner: `scripts/attn_constmax/ltx2_lane.sh`. Scoring: `scripts/attn_constmax/score.sh`,
  `scripts/compare_difflet_runs.py`. Table: `difflet/models/ltx_2/attn_constmax.json`.
- Evidence: `/mnt/nvme/difflet-exp/constmax-models/ltx_2/`.
  - `results.json`: per arm, per request timing, md5s, ratios and every score.
  - `commands.txt`: commands and env. `chain.sh` and `chain.log` hold the step order and times.
  - `collect.py` and `gate.py`.
  - `latents_determinism.txt`, `timed_latents_finite.txt`, `timed_strips_A_B_C_D.png`.
  - `<arm>/{compile,smoke,timed}.{log,env.txt}`, `warm_{1,2}.mp4`, `work/worker_0/latents_req000{0,1}.pt`.
  - `{cm,mxcm}/{smoke.mp4,smoke_strip.png,gate.json}`.
  - `scores/*.json`.
