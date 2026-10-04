# Best feature combination per model — trn2.3xlarge (2026-10-03 → 04)

Goal: for each model, the combination of Difflet's existing features that gives the
lowest **DiT step time / fewest DiT calls** and the lowest **warm e2e**, with compile
time out of scope (compiles are reused wherever the cache key allows). Branch
`campaign/trn2-flux-best-combo-2026-10-03`. Models so far: **FLUX.1-dev**.

## Setup

- trn2.3xlarge: 1 Trainium2, 4 NeuronCores (LNC=2), 24 GB HBM per core, 124 GB RAM.
  Toolchain from `requirements-neuron.lock` (torch-neuronx 2.15, NRT 2.34.10, diffusers
  0.38.0) — the same as the 2026-09 campaign.
- Attention: the default `attention_cte` megakernel only (sdpa excluded by request).
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
