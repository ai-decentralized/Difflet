# FLUX cache experiment protocol v1

This protocol makes FLUX cache A/B evidence comparable across code revisions.
Every prospective hardware collection records a canonical, SHA-256-protected
`difflet-flux-cache-experiment-protocol-v1` object in both output manifests.
Offline scoring independently records a
`difflet-flux-cache-evaluation-protocol-v1` object in `quality-curve-v2`.

The protocol fixes:

- the exact model snapshot and Neuron compile-cache identity;
- Git commit, branch, and dirty-worktree state;
- Python, Neuron, PyTorch, Diffusers, Transformers, NumPy, and Pillow versions;
- hardware/backend, TP degree, shape, scheduler config, step count, dtype, and guidance;
- CPU `torch.Generator` seed reset semantics and the ordered seed list;
- a versioned prompt-suite split, including prompt text, category, stable ID, and digest;
- the timing boundary and included/excluded work;
- cache semantics: complete Transformer noise-prediction forecasting, real-anchor-only
  history, Newton divided differences, and index coordinates.

The independent evaluation protocol fixes the evaluator Git commit and dirty state,
Python/PyTorch/TorchVision/LPIPS/NumPy/Pillow versions, metric configuration, LPIPS
package version (`0.1.4`), calibration version (`0.1`), network choice, and a SHA-256 over
every tensor in the loaded LPIPS model state. A protocol-v1 curve without this second
protocol is rejected by calibration.

## Prompt splits

`benchmark/flux_cache/prompt-suite-v1.json` contains:

- `legacy_parity`: the two prompts found in the historical FLUX TeaCache runner;
- `calibration`: eight stratified prompts recovered from the M9 calibration suite;
- `holdout`: eight disjoint prompts recovered from the same suite.

Calibration and holdout must remain disjoint. Candidate parameters may be selected using
`calibration`; the final automatic acceptance decision must also pass `holdout`.

The frozen selected-split digests are:

- `legacy_parity`: `9b59e23f7d5d6ffbc17ed85e973f5be636878fda6dcc091c2379cd704da15502`;
- `calibration`: `3a2ab8ba25eb6eea3fbaa072a6495c40b960af51b23c480aa98ee33e06b88c9a`;
- `holdout`: `68523dea70f615c04847f38b526aaedc03a31594dc6c40f88b9f0f2915bf7a17`.

The collector defaults to `legacy_parity`. Select another frozen split with
`--prompt-split calibration` or `--prompt-split holdout`. `--prompt-suite`,
`--prompts-json`, and repeated `--prompt` are mutually exclusive. Inline/custom prompts
are digest-protected in their resulting manifest but are labelled `inline-unversioned`.

## Prospective evidence requirements

A protocol-v1 collection fails before its first measured sample if the Git worktree is
dirty, the resolved model is not a 40-character Hugging Face snapshot commit, or the
Neuron compile manifest cannot be read. The evaluator independently binds the protected
protocol to the manifest model, shape, scheduler, step count, guidance, prompt text and
seed matrix. Changing those values and merely recomputing one manifest field is rejected.
Diffusers' internal `_use_default_values` scheduler field is canonicalized as a sorted
set; its process-randomized list order does not create false protocol drift.

The legacy parity command shape is:

```bash
python scripts/collect_flux_cache_ab.py \
  --out-dir /absolute/new/output-directory \
  --prompt-split legacy_parity \
  --seed 0 --seed 1 \
  --num-steps 50 --height 1024 --width 1024 --guidance-scale 3.5 \
  --warmup-steps 14 --anchor-intervals 4 --orders 1 \
  --anchor-phase 1 --cooldown-steps 1 --coord index \
  --tp-degree 4 --dtype bfloat16 \
  --allow-hardware \
  --foreground-ack "I am running FLUX cache A/B in the foreground"
```

## Status of the pre-rebuild numbers

The architecture document retains a summary for `warmup=14`, `interval=4`, `order=1`
(`2.06x`, trajectory cosine `0.996803`, worst PSNR `35.44 dB`, LPIPS `0.0318`). The raw
manifest, model revision, exact prompt binding, metric configuration, and timing boundary
were not committed and could not be recovered from Git objects, reflogs, shell history, or
local experiment artifacts.

Those values are therefore **legacy summary evidence**, not a parity oracle. They must not
be mixed with new `quality-curve-v2` results. New parity starts with the first clean,
digest-valid protocol-v1 hardware run.

The 2026-07-31 controlled audit is recorded in
[`docs/reports/flux-cache-parity-audit-20260731.md`](../reports/flux-cache-parity-audit-20260731.md).
It found that the frozen `legacy_parity` cases span `39.79` down to `25.98 dB`; the
fox-only worst case is `35.59 dB`, close to the legacy `35.44 dB` summary, while the
night-market cases control the current four-sample worst value. This is strong evidence
for case-set sensitivity, but not proof of the unsaved historical prompt binding.

## TaylorSeer naming boundary

Difflet's architecture predicts the complete Transformer noise prediction from real
anchors using Newton divided differences. The upstream TaylorSeer and Cache-DiT projects
forecast internal block features/residuals with derivative ladders. These are distinct
systems and must not share benchmark labels without recording the prediction target.

## Metric naming

The automatic trajectory gate uses the minimum flattened cosine over individual denoise
steps (`minimum-per-step-flattened-v1`). A cosine over the entire stacked trajectory is a
different diagnostic and must never be substituted under the same field name. There is
no fixed conversion between the two. On the same 2026-07-31 index-coordinate artifacts,
the worst minimum-per-step cosine was `0.98307991`, while the worst float64 stacked
cosine was `0.99798978`; the latter still did not reproduce the legacy `0.996803`.
