# FLUX cache phase-schedule A1 historical retrospective

Status: complete. Machine-readable result SHA-256: `ed0d37e912518bbd329226105fc64413b16407decde41edb4e47a517699a3c5b`.

## Evidence boundary

- This retrospective reads existing JSON only and makes zero hardware calls. Every required artifact passed its file-hash check.
- The historical outcomes were already opened. This report may standardize descriptive evidence only; it may not select a profile, change A2/A3, or support a serving claim.
- The pressure timing and step-29 completion datasets contain VQAScore only. An introduced failure in the latter is a VQA failure, not a full ImageReward-plus-VQA contract failure.
- Semantic tags are post-hoc manual multilabel annotations. They describe four step-29 non-rescues and are forbidden from every gate.

## Per-request rescue curve for the pressure profile

Candidate: `adaptive-vqa-stress-w6-i32-m16-k40-o1-index`; 12 independent source failures.

| terminal step | rescued | R(t) | mean brake benefit |
|---:|---:|---:|---:|
| 7 | 12/12 | 1.0000 | 0.447428 |
| 13 | 12/12 | 1.0000 | 0.449707 |
| 21 | 12/12 | 1.0000 | 0.419759 |
| 29 | 8/12 | 0.6667 | 0.263997 |
| 37 | 1/12 | 0.0833 | -0.018433 |

Step-29 non-rescues: p010-s2, p019-s2, p033-s2, p035-s2. Their shared post-hoc tag is: counting. This says only that all four cases in this failure-enriched queue contain counting constraints. The sample is too small, and the queue is already compositionally enriched, so it does not establish a general semantic-category effect.

The per-request recomputation exactly matches all historical source and rescue counts. Mean brake benefit matches exactly at steps 7, 13, 21, and 29. At step 37 the per-request recomputation is `-0.018432617188`, while the historical result records `-0.018391927083`, a difference of `-0.000040690104`. This report preserves both values, uses the bound per-request semantic artifact as the recomputation source, and does not silently rewrite the historical file.

## Introduced failure

Among the 36 step-29 continue-cache VQA pass controls, terminal braking introduced 2 VQA failures: p013-s2, p039-s2. Because ImageReward was not rescored, the full-contract introduced-failure count is `null`.

The light brake and recovery controls in the historical stage-oil pilot, plus the three terminal-followup controls, have both contract metrics and contain no observed introduced contract failure. Their sample size supports description only, not a safety claim.

## Contradiction adjudication

- The stage-oil family contains 2 independent failure requests repeated across 6 phase targets, with zero rescued targets. Calling these "six failures" is not an independence-correct description.
- The extreme-pressure family rescues 12 of 12 independent failure requests at step 21.
- Profile, prompt/seed population, and intervention grid all change together. The difference cannot be attributed to one factor.

The only allowed conclusion is: Observed rescue horizon depends on the profile family, failure mechanism, and sampled request distribution; neither historical curve is a deployable or model-wide horizon.

Therefore A1 supplies no `t_full` or `t_dead` for mask generation. The registered near-frontier A2 probe remains required with its existing futility rules.
