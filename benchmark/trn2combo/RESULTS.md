# Best feature combination per model — trn2.3xlarge (2026-10-03 → 05)

Goal: for each model, the combination of Difflet's existing features that gives the
lowest **DiT step time / fewest DiT calls** and the lowest **warm e2e**, with compile
time out of scope (compiles are reused wherever the cache key allows). Branch
`campaign/trn2-flux-best-combo-2026-10-03`. Models: FLUX.1-dev, Wan 2.1, Qwen-Image,
HunyuanVideo, LTX-2 (HunyuanVideo-1.5: CLI is a stub; MiniMax-H3: no feature axes —
both out of scope by decision; Wan 2.2: the CLI runs one expert only).

## Answer per model

Each workload is fixed (the benchmark matrix shape, steps and guidance); fewer DiT
calls come only from TeaCache. "Quality" is mean PSNR over 4 holdout prompts vs the
same-layout, same-decoder uncached output, plus a visual check.

| model (workload) | recommended combination | DiT calls | denoise loop | warm e2e (n=5) | quality |
|---|---|---|---|---|---|
| **FLUX.1-dev** (1024², 28 steps, g 3.5) | tp4 + `--teacache-cadence 2` | 28 → **19** | 7.5 → 5.1 s | 40.7 → **38.3 s** | 37.8 dB |
| **Qwen-Image** (1024², 20 steps, g 4.0) | tp4sp + `--teacache-cadence 2` | 20 → **15** | 9.1 → **6.0 s** | 65.5 → **62.9 s** | 41.1 dB |
| ↳ fewest calls | tp4sp + calibrated adaptive, 7-skip budget | 20 → **13** | **4.9 s** | 65.9 s (probe load +2–3 s) | 37.0 dB, visually equal |
| **HunyuanVideo** (320×512×61, 20 steps) | tp4sp + `--teacache-online-delta 0.1` | 20 → **17** | 16.5 → **13.7 s** | 113.7 → **109.3 s** | 37.4 dB |
| **Wan 2.1** (480×832×9, 20 steps, g 1.0, `--host-vae`) | tp4 + `--teacache-cadence 2` | 20 → **15** | 11.6 → **8.8 s** | 86.3 → **83.9 s** | 34.5 dB (visually equal); online-delta 0.4 passes at 35.4 dB with 17 calls |
| ↳ guidance 5.0 (realistic Wan) | tp4 + `--teacache-cadence 2` | 40 → **30** | 23.1 → **17.4 s** | 99.0 → **92.5 s** | visually equal (PSNR ≈ 24 dB is drift) |
| ↳ ring-conforming shape 512×768×9 | tp2cp2 **ring** | — | 11.0 → **9.0 s** (447 vs 537 ms/step) | 93.2 → **90.4 s** | 36.3 dB (benchmark prompt; layout drift) |
| **LTX-2** (480×704×49, 20 steps, g 1.0) | tp4 + `--teacache-cadence 2` | 20 → **15** | 9.2 → **6.9 s** | 57.0 → **55.3 s** | 37.1 dB |

Cross-model findings:

1. **TeaCache cadence 2 is the default winner** (FLUX, Qwen, Wan, LTX-2): ~25–32% fewer
   DiT calls at ≥ 34 dB. HunyuanVideo tolerates fewer skips — online-delta 0.1 (3 skips).
   Cadence 3/4 skip *fewer* steps (cadence N skips every N-th) and are never better.
2. **Calibrated adaptive only pays on Qwen-Image** (fit R² 0.99): 13/20 calls at 37.0 dB.
   On FLUX (R² 0.59), Wan (0.59) and HunyuanVideo (0.77) it gives the same calls as
   cadence 2 at lower quality, or fewer calls at failing quality, plus a probe cost.
3. **Layouts:** sequence parallel wins per-step on Qwen-Image (−12%) and HunyuanVideo
   (−3%); ulysses on FLUX (−2.4%); ring on Wan at a conforming shape (−17%). tp2 layouts
   load more weights per core, so tp4 / tp4sp usually win warm e2e.
4. **Warm e2e is load-bound** (weights 11–60 s of 38–114 s): step savings move it by
   2–7 s. The denoise-loop and resident columns are the clean step-reduction signal.

## Bugs found and fixed / filed

| # | what | status |
|---|---|---|
| 1 | **Neuron Wan VAE corrupts every frame after the first** (frames 1+ at 8–15 dB vs the host VAE on the same latents; eager module bit-identical to diffusers, so the trace / compile is at fault) | **filed: issue #73** (assigned). Wan measured with `--host-vae`. |
| 2 | **Wan calibrated-adaptive TeaCache could never load** — CLI stage cache key omitted the probe NEFF added in 0f9ef0f | **fixed: a330828** (+ test); measured after the fix |
| 3 | **LTX-2 calibrated-adaptive TeaCache could never load** — `compile()` never built the probe artifact the load resolves | **fixed: f43928d** (+ test) |
| 4 | Harness: holdout files keyed by label only (cross-model collision) | fixed: 26e3411 |

Limits recorded (not bugs): FLUX tp2cp2 + probe exceeds 24 GB HBM per core; Wan ring needs
per-rank tokens % 128 (480×832×9 → 2340); Qwen tp2cp2 gather_kv's first cold load takes
13+ min (a 2026-10-04 run stopped at 26 min was most likely this, not a deadlock — the
retry passed).

## Setup

- trn2.3xlarge: 1 Trainium2, 4 NeuronCores (LNC=2), 24 GB HBM per core, 124 GB RAM.
  Toolchain from `requirements-neuron.lock` (torch-neuronx 2.15, NRT 2.34.10, diffusers
  0.38.0) — the same as the 2026-09 campaign.
- Attention: the default `attention_cte` megakernel only (sdpa excluded by request).
- Seed 42; workloads per model as in the table above (`benchmark/models.py` MATRIX).
- FLUX.1-dev @ `3de623fc`, 1024×1024, **28 steps, guidance 3.5, seed 42**, bf16. The
  workload is fixed; fewer DiT calls come only from TeaCache.

### Metrics

| column | meaning | source |
|---|---|---|
| per-step | mean DiT call, device-synced inter-step deltas of a real generate (n = calls − 1) | `benchmark.step_realloop` |
| DiT calls | 28 minus TeaCache skips | TeaCache stats line / call count |
| resident | in-process generate wall with the model loaded (serving steady state), median of 3 | `step_realloop --generates 3` |
| warm e2e | fresh `difflet generate` process, warm OS page cache, median of 3 (finalists: 5) after 1 discarded warm-up | `benchmark.warm_e2e` |
| load | weight load inside that warm e2e | adapter log parse |
| PSNR | this cell's image vs the tp4 image, same prompt and seed | `benchmark.output_parity` |
| mean PSNR (4 prompts) | the same, over the 4 *holdout* prompts of `scripts/data/m9_teacache_prompts.tsv` (neither the benchmark nor a calibration prompt) | `benchmark.combo_quality` |

Quality bar (agreed up front): **PSNR ≥ 35 dB vs the uncached tp4 image**, judged on
the 4-prompt mean, plus a visual check
([holdout grid](../../artifacts/flux-best-combo-2026-10-04/flux_quality_holdout.jpg),
[calibrated-adaptive budgets](../../artifacts/flux-best-combo-2026-10-04/flux_quality_tcad_budgets.jpg)).

## FLUX.1-dev — answer

| goal | best combination | result | vs tp4 baseline | quality |
|---|---|---|---|---|
| fewest DiT calls within the bar | **TeaCache cadence 2** (any layout) | 19 / 28 calls | −32% calls | 37.8 dB mean (tp4 and tp2cp2) |
| lowest per-step | **tp2cp2 ulysses** | 262.8 ms | 269.4 → −2.4% | 42.3 dB mean (layout only) |
| lowest resident generate, full VAE | **tp2cp2 ulysses + cadence 2** | 5.34 s | 7.88 → −32% | 37.8 dB |
| lowest warm e2e, full VAE | **tp4 + cadence 2** | 38.3 s (37.3–40.0, n=5) | 40.7 → −6% | 37.8 dB |
| lowest warm e2e, any decoder | **tp4 + TAEF1 + cadence 2** | 31.6 s (30.6–32.5, n=5); resident 5.25 s | −22% warm, −33% resident | **30.3 dB — below the bar** (TAEF1 decode; visually close) |
| lowest resident, any decoder | tp2cp2 + TAEF1 + cadence 2 | 5.12 s | −35% | 30.7 dB — below the bar |

**Recommendation: tp4 + `--teacache-cadence 2`.** It is the warm-e2e winner and within
0.14 s (2.6%) of the best resident time; tp2cp2 + cadence 2 saves those 0.14 s per image
in a resident server but costs 3.6 s per fresh process (tp2 loads half the transformer
per core: 25 s vs 22 s). If TAEF1's approximate decode is acceptable for the use case,
add `--taef1-path madebyollin/taef1` for a further −6.7 s warm e2e — but it fails the
PSNR bar by design, so that is a product decision, not a measurement one.

## FLUX.1-dev — all cells

Warm e2e is n=3 except the seven finalists (n=5). PSNR is the benchmark prompt.

| label | configuration | per-step (ms) | DiT calls | resident (s) | warm e2e (s) | load (s) | PSNR vs tp4 |
|---|---|---|---|---|---|---|---|
| `tp4` | tp=4 | 269.4 | 28/28 | 7.88 | 40.7 | 22.7 | ref |
| `tp4tc2` | tp=4 + TeaCache cadence 2 | 269.9 | 19/28 | 5.48 | 38.3 | 22.2 | 41.3 dB |
| `tp4tc3` | tp=4 + TeaCache cadence 3 | 269.7 | 22/28 | 6.28 | 39.7 | 23.3 | 38.5 dB |
| `tp4tc4` | tp=4 + TeaCache cadence 4 | 270.0 | 24/28 | 6.83 | 40.3 | 23.9 | 38.2 dB |
| `tp4tcod005` | tp=4 + TeaCache online-delta 0.05 | 269.7 | 26/28 | 7.37 | 41.9 | 24.8 | 48.0 dB |
| `tp4tcod01` | tp=4 + TeaCache online-delta 0.1 | 269.5 | 21/28 | 6.28 | 38.5 | 22.0 | 42.8 dB |
| `tp4tcod02` | tp=4 + TeaCache online-delta 0.2 | 269.6 | 19/28 | 5.74 | 38.6 | 22.8 | 37.6 dB |
| `tp4tcod04` | tp=4 + TeaCache online-delta 0.4 | 269.8 | 19/28 | 5.47 | 38.3 | 22.7 | 39.7 dB |
| `tp4tcad` | tp=4 + TeaCache calibrated adaptive, 9-skip budget | 272.5 | 19/28 | 5.53 | 41.0 | 24.9 | 37.4 dB |
| `tp4tcad12` | tp=4 + calibrated adaptive, 12-skip budget | 272.8 | 16/28 | 4.71 | 40.0 | 24.7 | 27.5 dB ✗ |
| `tp4tcad14` | tp=4 + calibrated adaptive, 14-skip budget | 273.2 | 15/28 | 4.44 | 40.1 | 25.8 | 19.5 dB ✗ |
| `tp4taef1` | tp=4 + TAEF1 | 269.8 | 28/28 | 7.68 | 34.3 | 16.2 | 34.6 dB ✗ |
| `tp4taef1tc2` | tp=4 + TAEF1 + cadence 2 | 270.1 | 19/28 | 5.25 | 31.6 | 17.1 | 33.9 dB ✗ |
| `tp4sp` | tp=4 + sequence parallel | 278.2 | 28/28 | 8.13 | 41.9 | 23.8 | 48.1 dB |
| `tp2cp2` | tp=2 × cp=2 (ulysses) | 262.8 | 28/28 | 7.70 | 44.7 | 26.0 | 43.9 dB |
| `tp2cp2tc2` | ulysses + cadence 2 | 263.0 | 19/28 | 5.34 | 41.9 | 25.1 | 38.9 dB |
| `tp2cp2tc3` | ulysses + cadence 3 | 262.6 | 22/28 | 6.13 | 43.1 | 27.2 | 37.5 dB |
| `tp2cp2tc4` | ulysses + cadence 4 | 263.2 | 24/28 | 6.66 | 43.2 | 25.8 | 37.2 dB |
| `tp2cp2tcod005` | ulysses + online-delta 0.05 | 262.8 | 26/28 | 7.18 | 44.3 | 26.6 | 44.4 dB |
| `tp2cp2tcod01` | ulysses + online-delta 0.1 | 263.3 | 21/28 | 6.14 | 43.5 | 25.5 | 40.7 dB |
| `tp2cp2tcod02` | ulysses + online-delta 0.2 | 263.6 | 19/28 | 5.62 | 43.5 | 27.0 | 36.9 dB |
| `tp2cp2tcod04` | ulysses + online-delta 0.4 | 263.4 | 19/28 | 5.35 | 42.4 | 25.9 | 39.6 dB |
| `tp2cp2tcad` / `tcad12` / `tcad14` | ulysses + calibrated adaptive | **BLOCKED (HBM)** ¹ | | | | | |
| `tp2cp2taef1` | ulysses + TAEF1 | 262.8 | 28/28 | 7.48 | 37.5 | 19.6 | 34.2 dB ✗ |
| `tp2cp2taef1tc2` | ulysses + TAEF1 + cadence 2 | 263.5 | 19/28 | 5.12 | 35.3 | 19.8 | 33.3 dB ✗ |
| `tp2cp2gkv` | tp=2 × cp=2 (gather_kv) | 278.4 | 28/28 | 8.15 | 44.2 | 24.8 | 43.0 dB |
| `tp2cp2ring` | tp=2 × cp=2 (ring) | 267.4 | 28/28 | 7.83 | 44.7 | 26.8 | 41.4 dB |

¹ tp2 × cp2 plus the TeaCache probe NEFF leaves NC 2 at 23.3 GB of tensors (of ~24 GB);
the first generate's 36 MB allocation fails (`TDRV: Failed to allocate DEVICE memory
(37748736 bytes)`). Breakdown: `logs/tp2cp2tcad/neuron_mem_table_nc2.log`. The same
probe fits under tp4, where each core holds a quarter of the transformer. Fix paths
(not done): a larger instance, or T5 off the DiT cores.

### Quality over 4 holdout prompts (mean / worst PSNR vs tp4)

| candidate | mean | worst | per prompt (sailboat, neon alley, books, puppy) |
|---|---|---|---|
| tp2cp2 (layout only) | 42.3 | 36.0 | 49.6, 36.0, 41.8, 41.8 |
| tp4 + cadence 2 | **37.8** | 32.4 | 47.3, 32.4, 38.0, 33.6 |
| tp2cp2 + cadence 2 | **37.8** | 32.5 | 45.2, 32.5, 36.3, 37.2 |
| tp4 + calibrated adaptive (9 skips) | 37.1 | 32.3 | |
| tp4 + online-delta 0.4 | 36.6 | 31.3 | 41.3, 31.3, 35.6, 38.2 |
| tp4 + online-delta 0.1 | 35.9 | 29.2 | |
| tp4 + TAEF1 | 31.6 | 29.0 | 31.7, 29.0, 31.5, 34.3 |
| tp2cp2 + TAEF1 | 31.2 | 28.3 | |
| tp2cp2 + TAEF1 + cadence 2 | 30.7 | 27.6 | |
| tp4 + TAEF1 + cadence 2 | 30.3 | 27.6 | |

Cadence 2 measured against its *own* uncached decoder / layout path costs the same
everywhere: 38.8 dB (tp2cp2), 36.8 dB (tp4 + TAEF1) on the 4-prompt mean. The TAEF1
rows' gap to the bar is the approximate decoder, not TeaCache.

## Findings

1. **FLUX warm e2e is load-bound.** tp4: 40.7 s = 22.7 s weight load + ~7.9 s generate
   + ~10 s process / runtime start. Denoise-only features can move warm e2e by at most
   ~7.5 s; the per-step spread between layouts (263–278 ms) is ≤ 0.4 s per image.
   Resident time is the clean signal for TeaCache; load size decides warm e2e.
2. **Cadence N skips every N-th step** (`difflet/pipeline/teacache.py:241-244`), so
   cadence 2 is the most aggressive fixed cadence (9 skips) and 3 / 4 skip *fewer*
   (6 / 4) — and at no better quality. Online-delta never skips two steps in a row, so
   it saturates at cadence 2's 19 calls (α ≥ 0.2).
3. **Below 19 calls nothing passes.** Calibrated adaptive is the only mode that can skip
   consecutive steps. Re-calibrated on this host (8 prompts, 216 on-device pairs,
   Pearson 0.708, R² 0.59; fits at 9 / 12 / 14 skips), the 12- and 14-skip budgets reach
   16 and 15 calls (resident 4.71 / 4.44 s) but fall to 27.5 / 19.5 dB with visible
   halo and grid artifacts. At the 9-skip budget it matches cadence 2's calls with lower
   quality and +3 ms per step for the probe — cadence 2 dominates it on FLUX.
4. **Layouts.** tp2cp2 ulysses has the fastest DiT step (262.8 ms, −2.4%), ring is next
   (267.4), tp4 is the baseline (269.4); tp4sp (278.2) and tp2cp2 gather_kv (278.4) are
   slower. Every tp2 layout loads 2–4 s more per process.
5. **TAEF1** compiles in 3 min instead of ~16 and cuts the decoder load by 6.5 s, the
   largest single warm-e2e gain (−6.4 s), with resident −0.2 s. It fails the PSNR bar
   on every prompt (29–34 dB vs the full VAE) yet is hard to tell apart at a glance
   (see the holdout grid) — keep it opt-in.
6. **Compile reuse.** The TeaCache cadence / online-delta and calibrated-adaptive budget
   labels never recompiled (8 s manifest hits; only `teacache_probe_enabled` is in the
   key, so the 9 / 12 / 14-skip budgets share one probe artifact). Total device compile
   for the 9 artifacts (5 layouts, 2 TAEF1, 2 probe): 4,300 s ≈ 1.2 h.

### Not run, with the reason

- **TAEF1 + calibrated-adaptive probe** — calibrated adaptive was not better than
  cadence 2, so per the plan the combined artifact was skipped.
- **`difflet serve` check of the finalists** — resident serving rejects
  `--teacache-cadence` / `--teacache-online-delta` (`difflet/serving/options.py`), so
  the finalists cannot be served as is; the in-process resident column is the serving
  steady state (reference: 2026-09 serving tp4 c=1 p50 8.21 s vs resident 7.88 s here).
- **sdpa attention, DP, CFG-parallel** — excluded by request / by design (DP adds
  throughput, FLUX is guidance-distilled).

## Wan 2.1 — answer

480×832×9, 20 steps, seed 42. **Decode with `--host-vae`:** the default Neuron VAE
corrupts frames 1+ (issue #73), so Neuron-VAE outputs are not a quality measure and
every Wan quality / warm-e2e figure below is host-VAE unless the label lacks `hostvae`
(those rows are kept for their DiT timings, which the VAE does not affect).

| goal | best combination | result | quality |
|---|---|---|---|
| fewest calls within the bar, guidance 1.0 | tp4 + online-delta 0.4 | 17/20 calls, denoise 10.1 s | 35.4 dB mean |
| best speed at visually equal quality, guidance 1.0 | **tp4 + cadence 2** | 15/20, denoise 11.65 → 8.80 s, warm 86.3 → 83.9 s (n=5) | 34.5 dB mean, visually identical |
| guidance 5.0 (the realistic setting) | **tp4 + cadence 2** | 30/40 calls, denoise 23.1 → 17.4 s, warm 99.0 → 92.5 s (n=5) | visually equal (24 dB = trajectory drift) |
| lowest per-step | **tp2cp2 ring at 512×768×9** | 447.3 ms vs tp4 537.2 (−17%), denoise 9.0 vs 11.0 s, warm 90.4 vs 93.2 s | 36.3 dB |

Notes: guidance 1.0 (the benchmark workload) makes weak videos (the puppy prompt shows no
puppy); guidance 5.0 videos are real — see [the host-VAE holdout grid](../../artifacts/flux-best-combo-2026-10-04/wan_hostvae_quality_holdout.jpg).
tp4sp ties tp4 (577 vs 575 ms/step); tp2cp2 ulysses equals tp4 per step but loads more;
gather_kv is +3%. Ring at 480×832 is a shape LIMIT (2340 tokens/rank, not % 128).
CFG-parallel (tp2 × 2 branches) trims the guidance-5 denoise 7% but loads 9 s more
(warm 109.2 vs 99.0 s); TeaCache is off under CFG-parallel by design. Calibrated adaptive
(after fix a330828; R² 0.59) gives 15 calls at 33.2 dB / 13 calls at 29.9 dB — below
cadence 2. Host VAE costs ~+4 s warm e2e vs the (broken) Neuron VAE: the 30 s Neuron VAE
load disappears and the CPU decode of 9 frames takes ~35 s.

### All wan_2_1 cells

| label | configuration | per-step (ms) | DiT calls | denoise loop (s) | resident (s) | warm e2e (s) | load (s) | PSNR vs ref |
|---|---|---|---|---|---|---|---|---|
| `tp4` | tp=4, bf16, single-transformer (no MoE), attention_cte, 2-stage (transformer + VAE) subprocess pipeline | 574.4 | 20/20 | 11.63 | — | 82.4 | 50.3 | ref |
| `tp4tc2` | tp=4 + TeaCache fixed cadence 2 (--teacache-cadence 2) | 575.1 | 15/20 | 8.77 | — | 78.2 | 50.3 | 36.7 dB |
| `tp4tcod01` | tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.1 | 574.6 | 20/20 | 11.65 | — | 80.8 | 49.2 | identical |
| `tp4tcod02` | tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.2 | 575.0 | 20/20 | 11.65 | — | 79.8 | 46.9 | identical |
| `tp4tcod04` | tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.4 | 574.5 | 17/20 | 9.91 | — | 79.7 | 48.9 | 36.7 dB |
| `tp4hostvae` | tp=4 + host VAE | 575.0 | 20/20 | 11.65 | — | 86.3 | 18.6 | ref (vs tp4hostvae) |
| `tp4hostvaetc2` | tp=4 + host VAE + TeaCache cadence 2 | 574.7 | 15/20 | 8.80 | — | 83.9 | 17.1 | 34.3 dB (vs tp4hostvae) |
| `tp4hostvaetcod04` | tp=4 + host VAE + TeaCache online-delta 0.4 | 574.3 | 17/20 | 10.13 | — | 84.2 | 16.7 | 33.2 dB (vs tp4hostvae) |
| `tp4hostvaetcad` | tp=4 + host VAE + TeaCache calibrated adaptive | 582.3 | 15/20 | 8.89 | — | 83.4 | 17.2 | 32.2 dB (vs tp4hostvae) |
| `tp4hostvaetcad7` | tp=4 + host VAE + TeaCache calibrated adaptive, 7-skip budget | 584.0 | 13/20 | 7.73 | — | 84.1 | 19.1 | 27.6 dB (vs tp4hostvae) |
| `tp4sp` | tp=4 + sequence parallel | 577.4 | 20/20 | 11.68 | — | 79.8 | 46.8 | 33.3 dB |
| `tp4sptc2` | tp=4 + sequence parallel + TeaCache cadence 2 | 577.3 | 15/20 | 8.76 | — | 78.3 | 46.6 | 33.8 dB |
| `tp4sptcod04` | tp=4 + sequence parallel + TeaCache online-delta 0.4 | 577.8 | 17/20 | 9.93 | — | 78.3 | 48.0 | 34.0 dB |
| `tp4sphostvae` | tp=4 + sequence parallel + host VAE | 577.5 | 20/20 | 11.75 | — | 86.7 | 18.7 | 25.2 dB (vs tp4hostvae) |
| `tp4sphostvaetc2` | tp=4 + sequence parallel + host VAE + TeaCache cadence 2 | 577.9 | 15/20 | 8.77 | — | 83.2 | 17.3 | 25.9 dB (vs tp4hostvae) |
| `tp2cp2` | tp=2 x cp=2 (ulysses) | 574.2 | 20/20 | 11.57 | — | 86.6 | 54.2 | 34.1 dB |
| `tp2cp2gkv` | tp=2 x cp=2 (gather_kv) | 590.9 | 20/20 | 11.92 | — | 87.6 | 53.7 | 34.1 dB |
| `tp2cp2ring` | tp=2 x cp=2 (ring) | **BLOCKED** — LIMIT: ring needs per-rank tokens % 128 == 0 (nkilib ring_attention_spmd_fwd, NCC_INKI016); 480x832x9 gives 2340 per rank, so the stage fails fast ||||||||
| `tp4g5` | tp=4 at guidance 5.0 (two sequential CFG branches) | 574.9 | 40/20 | 23.14 | — | 92.4 | 47.7 | ref (vs tp4g5) |
| `tp4g5tc2` | tp=4 at guidance 5.0 + TeaCache cadence 2 | 575.7 | 30/20 | 17.41 | — | 88.2 | 48.0 | 26.9 dB (vs tp4g5) |
| `tp2cfgg5` | tp=2 x CFG-parallel at guidance 5.0 | 1068.8 | 20/20 | 21.48 | — | 101.2 | 56.6 | 28.1 dB (vs tp4g5) |
| `tp4hostvaeg5` | tp=4 at guidance 5.0 (two sequential CFG branches) + host VAE | 574.8 | 40/20 | 23.14 | — | 99.0 | 17.9 | ref (vs tp4hostvaeg5) |
| `tp4hostvaeg5tc2` | tp=4 at guidance 5.0 + TeaCache cadence 2 + host VAE | 575.3 | 30/20 | 17.40 | — | 92.5 | 17.2 | 20.8 dB (vs tp4hostvaeg5) |
| `tp2cfgg5hostvae` | tp=2 x CFG-parallel at guidance 5.0 + host VAE | 1068.9 | 20/20 | 21.49 | — | 109.2 | 26.6 | 22.3 dB (vs tp4hostvaeg5) |
| `r768tp4` | 512x768x9 (ring-conforming), tp=4 + host VAE | 537.2 | 20/20 | 10.99 | — | 93.2 | 17.4 | ref (vs r768tp4) |
| `r768tp2cp2` | 512x768x9 (ring-conforming), tp=2 x cp=2 (ulysses) + host VAE | 531.6 | 20/20 | 10.72 | — | 92.8 | 22.9 | 36.4 dB (vs r768tp4) |
| `r768tp2cp2ring` | 512x768x9 (ring-conforming), tp=2 x cp=2 (ring) + host VAE | 447.3 | 20/20 | 9.03 | — | 90.4 | 23.2 | 36.3 dB (vs r768tp4) |

## Qwen-Image — answer

1024×1024, 20 steps, guidance 4.0 (single forward, guidance-distilled).

| goal | best combination | result | quality |
|---|---|---|---|
| fewest calls / fastest denoise | **tp4sp + calibrated adaptive, 7-skip** | 13/20 calls, denoise 9.12 → 4.89 s (−46%) | 37.0 dB vs tp4sp, visually equal |
| best warm e2e | **tp4sp + cadence 2** (tp4 + online-delta 0.4 ties) | 62.9 s (62.3) vs 65.5 s; denoise 6.04 s | 41.1 dB |
| lowest per-step | **tp4sp** | 365.3 ms vs 416.0 (−12%) | 46.4 dB vs tp4 |

Calibration fit R² 0.986 (8 prompts) — the only model where adaptive beats cadence 2;
its probe adds ~2–3 s of weight load, so it wins resident / denoise, not warm e2e.
TeaCache is near-lossless here (41–46 dB). tp2 layouts are slower (ulysses 453.8, gather_kv
455.7, ring 492.4 ms) and load ~6 s more. [Holdout grid](../../artifacts/flux-best-combo-2026-10-04/qwen_quality_holdout.jpg).

### All qwen_image cells

| label | configuration | per-step (ms) | DiT calls | denoise loop (s) | resident (s) | warm e2e (s) | load (s) | PSNR vs ref |
|---|---|---|---|---|---|---|---|---|
| `tp4` | tp=4, bf16, joint attention via attention_cte | 416.0 | 20/20 | 9.12 | — | 65.5 | 36.9 | ref |
| `tp4tc2` | tp=4 + TeaCache fixed cadence 2 (--teacache-cadence 2) | 416.2 | 15/20 | 7.05 | — | 63.3 | 33.2 | 45.2 dB |
| `tp4tcod01` | tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.1 | 416.5 | 20/20 | 9.01 | — | 64.2 | 32.8 | identical |
| `tp4tcod02` | tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.2 | 416.6 | 17/20 | 7.90 | — | 64.0 | 33.3 | 46.1 dB |
| `tp4tcod04` | tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.4 | 416.6 | 15/20 | 6.92 | — | 62.3 | 34.2 | 45.8 dB |
| `tp4tcad` | tp=4 + TeaCache calibrated adaptive (--teacache-speedup at cadence 2's skip budget, --teacache-calibration per model) | 418.5 | 15/20 | 6.40 | — | 65.8 | 38.8 | 45.0 dB |
| `tp4tcad7` | tp=4 + TeaCache calibrated adaptive, 7-skip budget | 419.4 | 13/20 | 5.58 | — | 65.4 | 37.2 | 37.2 dB |
| `tp4sp` | tp=4 + sequence parallel | 365.3 | 20/20 | 7.87 | — | 64.1 | 35.2 | 46.4 dB |
| `tp4sptc2` | tp=4 + sequence parallel + TeaCache cadence 2 | 365.0 | 15/20 | 6.04 | — | 62.9 | 36.1 | 43.9 dB |
| `tp4sptcod02` | tp=4 + sequence parallel + TeaCache online-delta 0.2 | 365.1 | 17/20 | 6.78 | — | 63.7 | 36.3 | 45.0 dB |
| `tp4sptcod04` | tp=4 + sequence parallel + TeaCache online-delta 0.4 | 365.2 | 15/20 | 6.06 | — | 63.4 | 33.3 | 44.4 dB |
| `tp4sptcad` | tp=4 + sequence parallel + TeaCache calibrated adaptive | 368.3 | 15/20 | 5.64 | — | 65.9 | 36.5 | 44.7 dB |
| `tp4sptcad7` | tp=4 + sequence parallel + TeaCache calibrated adaptive, 7-skip budget | 367.5 | 13/20 | 4.89 | — | 65.9 | 37.9 | 36.8 dB |
| `tp2cp2` | tp=2 x cp=2 (ulysses) | 453.8 | 20/20 | 10.21 | — | 74.1 | 42.7 | 45.1 dB |
| `tp2cp2gkv` | tp=2 x cp=2 (gather_kv) | 455.7 | 20/20 | 10.29 | — | 76.7 | 40.4 | 46.6 dB |
| `tp2cp2ring` | tp=2 x cp=2 (ring) | 492.4 | 20/20 | 10.85 | — | 75.1 | 42.1 | 47.0 dB |

## HunyuanVideo — answer

320×512×61, 20 steps, guidance 6.0 (distilled), host VAE decode.

| goal | best combination | result | quality |
|---|---|---|---|
| best overall | **tp4sp + online-delta 0.1** | 17/20 calls, denoise 16.5 → 13.66 s, warm 113.7 → 109.3 s (n=5) | 37.4 dB vs tp4sp |
| lowest per-step | **tp4sp** | 787.9 ms vs 812.3 (−3%) | layout drift (frames visually equal) |

HunyuanVideo tolerates few skips: on the benchmark prompt only online-delta 0.1 stays
above 35 dB (cadence 2 31.8, online-delta 0.2 33.2); over the holdout prompts cadence 2 /
0.2 average 36 dB but online-delta 0.1 is the safe choice. Calibrated adaptive is no longer
HBM-blocked (0f9ef0f) but its probe adds 5–8% per call and quality falls to 21–24 dB.
Layout drift in 61-frame video is large in PSNR (tp4sp 24.8 dB vs tp4) while the frames
match visually, so TeaCache is judged per layout. tp2cp2 ulysses is slower (849.9 ms).

### All hunyuan_video cells

| label | configuration | per-step (ms) | DiT calls | denoise loop (s) | resident (s) | warm e2e (s) | load (s) | PSNR vs ref |
|---|---|---|---|---|---|---|---|---|
| `tp4` | tp=4, bf16, attention_cte | 812.3 | 20/20 | 16.50 | — | 113.7 | 60.1 | ref |
| `tp4tc2` | tp=4 + TeaCache fixed cadence 2 (--teacache-cadence 2) | 812.3 | 15/20 | 12.43 | — | 109.8 | 57.1 | 31.8 dB |
| `tp4tcod01` | tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.1 | 812.9 | 17/20 | 14.05 | — | 111.6 | 57.1 | 35.5 dB |
| `tp4tcod02` | tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.2 | 812.8 | 16/20 | 13.24 | — | 108.9 | 57.0 | 33.2 dB |
| `tp4tcod04` | tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.4 | 812.7 | 15/20 | 12.47 | — | 109.7 | 55.7 | 22.5 dB |
| `tp4tcad` | tp=4 + TeaCache calibrated adaptive (--teacache-speedup at cadence 2's skip budget, --teacache-calibration per model) | 854.8 | 15/20 | 12.95 | — | 110.6 | 56.8 | 24.1 dB |
| `tp4tcad7` | tp=4 + TeaCache calibrated adaptive, 7-skip budget | 874.5 | 13/20 | 11.46 | — | 107.3 | 56.9 | 21.3 dB |
| `tp4sp` | tp=4 + sequence parallel | 787.9 | 20/20 | 16.03 | — | 113.1 | 55.2 | 24.8 dB |
| `tp4sptc2` | tp=4 + sequence parallel + TeaCache cadence 2 | 788.0 | 15/20 | 12.03 | — | 109.3 | 56.5 | 23.9 dB |
| `tp4sptcod01` | tp=4 + sequence parallel + TeaCache online-delta 0.1 | 788.8 | 17/20 | 13.66 | — | 109.3 | 55.9 | 24.8 dB |
| `tp4sptcod02` | tp=4 + sequence parallel + TeaCache online-delta 0.2 | 788.4 | 16/20 | 12.85 | — | 108.3 | 54.5 | 24.2 dB |
| `tp2cp2` | tp=2 x cp=2 (ulysses) | 849.9 | 20/20 | 17.20 | — | 117.8 | 58.6 | 28.8 dB |

## LTX-2 — answer

480×704×49, 20 steps, guidance 1.0; tp4 only (no CP / SP path). Text encoder and VAE
run on the host, so the Neuron weight load is only ~11 s and step savings show in e2e.

| goal | best combination | result | quality |
|---|---|---|---|
| fewest calls / best e2e | **tp4 + cadence 2** | 15/20 calls, denoise 9.16 → 6.88 s, warm 57.0 → 55.3 s (n=5) | 37.1 dB mean |

Online-delta 0.1 / 0.2 skip nothing on LTX-2 and 0.4 skips one step. At guidance 1.0 the
subject renders weakly (as on Wan). __LTX_TCAD__

### All ltx_2 cells

| label | configuration | per-step (ms) | DiT calls | denoise loop (s) | resident (s) | warm e2e (s) | load (s) | PSNR vs ref |
|---|---|---|---|---|---|---|---|---|
| `tp4` | tp=4, bf16, TP-sharded transformer + attention_cte self-attn, guidance=1.0 (batch-1 NEFF) | 457.8 | 20/20 | 9.16 | — | 57.0 | 10.8 | ref |
| `tp4tc2` | tp=4 + TeaCache fixed cadence 2 (--teacache-cadence 2) | 458.3 | 15/20 | 6.88 | — | 55.3 | 10.9 | 36.6 dB |
| `tp4tcod01` | tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.1 | 458.3 | 20/20 | 9.17 | — | 57.2 | 10.4 | identical |
| `tp4tcod02` | tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.2 | 458.3 | 20/20 | 9.17 | — | 57.0 | 14.7 | identical |
| `tp4tcod04` | tp=4 + TeaCache online-delta adaptive (--teacache-online-delta 0.4 | 458.6 | 19/20 | 8.72 | — | 56.8 | 11.5 | 36.9 dB |

## Reproduce

```bash
./scripts/setup_env.sh                       # .venv from requirements-neuron.lock
export DIFFLET_BENCH_DEVICE=trn2combo
# any list of combo labels (benchmark/models.py COMBO_LABELS); resumable per metric
bash benchmark/trn2/run_combo.sh /tmp/done flux_1_dev tp4 tp4tc2 tp2cp2 tp4taef1tc2
# calibrated adaptive: probe artifact + 8-prompt calibration, then other budgets
python -m benchmark.tcad_prep --model flux_1_dev --prompts 8
python -m benchmark.teacache_calibrate fit --model flux_1_dev --target-skips 12
# 4-prompt holdout quality
python -m benchmark.combo_quality gen --model flux_1_dev --config tp4tc2
python -m benchmark.combo_quality score --model flux_1_dev --ref tp4 --configs tp4tc2
python -m benchmark.combo_table --model flux_1_dev
```

Per-cell results: `flux_1_dev[_<label>].json` here; calibrations in `teacache_calib/`.
