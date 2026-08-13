# FLUX two-anchor Gram-tap validation (2026-08-11)

## Decision

The scalar Gram identity is correct, and the proposed anchor signal is a
useful proxy for next-segment tensor error for the tested FLUX.1-dev A12
order-1 TaylorSeer cache. Both the clean-trajectory kill test and a real cache
feedback rollout passed by wide margins.

This result does **not** establish a conditional quality bound, adaptive
quality-budget accounting, semantic quality protection, or serving value for
an in-graph Gram implementation.

## Current-system audit

- The registered and qualified runtime is FLUX.1-dev at 1024x1024, 50 steps,
  BF16, TP4. The repository does not currently register or qualify FLUX.2.
- The qualified profile is a 14-static-anchor order-1 TaylorSeer profile with
  a bounded two-step brake. Its measured confirmation speedup is 3.2458x with
  zero quality failures in 32 confirmation prompts.
- The current branch head adds an unqualified globally calibrated linear
  predictor. Its skipped-step weights minimize clean final-latent deviation,
  while its anchor measurement deliberately falls back to two-anchor Taylor
  extrapolation. The experiments below therefore validate the TaylorSeer A12
  mechanism, not that newer calibrated predictor.
- The existing `measure_anchor_error` path already forms the residual and L2
  norm on device tensors and transfers scalar summaries to the host. It does
  not transfer the full 512 KB output tensor. An in-graph Gram tap would extend
  the observable geometry and may reduce host/XLA dispatch overhead, but the
  two-anchor scalar measurement primitive is not new to this codebase.

## Mathematical and hardware corrections

1. With K resident anchors, a new anchor yields a request-specific partial
   Gram slice over those K anchors, not a complete Gram row.
2. TP4 local partial dot products require a cross-rank sum. The existing
   output all-gather does not perform that reduction; a scalar all-reduce,
   gather-plus-sum, or measured fusion is still required.
3. For K=2, the work is three 262144-element dot products: 786432 MACs, or
   about 1.57 MFLOPs when multiply and add are counted separately. The HTML's
   0.26 MFLOP and “two-million-fold” statements are not self-consistent.
4. Proposed pre-gather storage is K x 128 KB/core in BF16. Current gathered
   outputs are replicated and larger. Persistent alias state demonstrates
   device/HBM persistence, not guaranteed residency in the roughly 24 MB
   on-chip scratchpad, so the scratchpad comparison is not yet evidence.
5. Reconstructing a small residual norm by subtracting large Gram terms can
   lose precision. A CPU FP32 simulation over 432 anchor measurements had no
   negative squared residuals and a 95th-percentile relative discrepancy of
   5.63e-5 versus direct residual L2, but compiled Trainium reduction precision
   remains a hardware measurement.
6. For the single two-anchor counterfactual error, direct on-device residual
   L2 is simpler and numerically safer than materializing six Gram entries.
   Gram becomes useful when a policy actually consumes a reusable K-anchor
   geometry rather than only one scalar error.

## Gate A: clean trajectories

- Data: 48 full-DiT calibration trajectories.
- Candidate: static anchors `[0,1,2,3,4,5,9,15,21,31,41,49]`, order-1 index
  TaylorSeer.
- Primary prompt-level association: Spearman rho 0.8979, bootstrap 95% CI
  `[0.7833, 0.9525]`, one-sided permutation p `< 1e-4`.
- All six fixed next-segment correlations were positive; the range was
  `[0.6420, 0.8881]`.
- Decision: pass only to Gate B.

Registered result:
`benchmark/flux_cache/gram-transfer-clean-gate-result.json`.

## Gate B: real cache feedback rollout

- Data: 32 prospectively frozen confirmation prompts with no exact Gate A
  prompt overlap, seed 7.
- At every natural skip, a shadow full-DiT output was evaluated on the exact
  same pre-step cache latent, while the original prediction advanced the
  trajectory.
- Every sample executed 12 real anchors and 38 natural skips; all 1216 shadow
  labels passed step-count checks.
- Primary equal-weight mean of six within-fixed-segment Spearman correlations:
  rho 0.7066, clustered bootstrap 95% CI `[0.6008, 0.7680]`, one-sided
  permutation p `< 1e-4`.
- All six segments were positive; individual rho values ranged from 0.5198 to
  0.8017. The raw absolute-error sensitivity was also positive (rho 0.7393).
- Decision: the proxy is supported for this predictor and schedule; advance
  only to a separately fitted conditional-bound study.

Registered result:
`benchmark/flux_cache/gram-transfer-cache-rollout-gate-result.json`.

## Current-head calibrated-predictor sensitivity

The branch-head `calibrated_linear` predictor creates a signal mismatch: skips
use globally fitted per-step weights, while real-anchor counterfactual probes
fall back to two-anchor Taylor extrapolation. An exploratory in-sample clean
sensitivity over the 48 calibration trajectories found a positive prompt-level
association (rho 0.6903), but the six fixed next-segment correlations were
`[-0.2372, 0.4772, 0.4898, 0.1683, 0.6998, 0.6744]` (mean 0.3787). This is not a
registered or independent gate, but it is enough to show that the strong A12
TaylorSeer result cannot be inherited by the newer calibrated predictor. It
needs its own clean and real-rollout gates, ideally with a predictor-consistent
anchor probe.

## Remaining inference gap

The evidence now supports:

`anchor counterfactual tensor error -> next-segment shadow tensor damage`

It does not yet support:

`anchor signal -> calibrated upper bound -> safe adaptive skip count -> final semantic quality`

A next experiment must fit a monotone conditional upper bound on a disjoint
development cohort and test empirical coverage and saved-anchor efficiency on
a prospectively frozen holdout. Prior work in this repository rejected a
regional anchor-error feature as a direct classifier of VQA failures (ROC AUC
0.463), so tensor-error transfer must not be silently equated with semantic
quality protection.
