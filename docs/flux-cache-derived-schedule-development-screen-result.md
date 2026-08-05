# FLUX derived schedule development screen result

Status: stopped with no eligible static or static-plus-brake representative. No
confirmation, holdout, or serving claim is permitted.

## Decision

The registered development screen completed 32 shared full-DiT baselines, the
legacy brake-only comparator, and all four theory-derived candidates: 192 images in
total. All images received both frozen ImageReward and VQAScore evaluations.

Every new candidate passed the registered speed gate, but every new candidate had
at least one ImageReward-only contract failure. The development gate required zero
failures in 32 independent prompt groups. The evaluator therefore returned
`stop_no_eligible_representatives` and selected no family representative.

| Candidate | Full-DiT speedup | Speed vs legacy | Failures | 95% upper bound | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Legacy brake-only comparator | 3.2152x | 1.0000x | 0/32 | 8.94% | Comparator only |
| Static, 12 anchors | 3.9486x | 1.2281x | 2/32 | 18.39% | Reject |
| Static + bounded brake, 12 anchors | 3.9344x | 1.2237x | 2/32 | 18.39% | Reject |
| Static, 13 anchors | 3.6962x | 1.1496x | 1/32 | 13.98% | Reject |
| Static + bounded brake, 13 anchors | 3.6536x | 1.1363x | 1/32 | 13.98% | Reject |

The speed conclusions are robust to the registered paired bootstrap: every new
candidate's 95% lower bound relative to legacy exceeded 1.09. Speed was not the
blocking dimension.

## Failure structure

All four new candidates fail `p013-s3`, the text-rendering prompt `FRESH BREAD
TODAY`. ImageReward harm ranges from `0.9653` to `1.1807`; no VQAScore harm exceeds
the contract margin. Both 12-anchor candidates also fail `p028-s3`, a
scene-composition prompt, with ImageReward harm `0.8120`.

The bounded online brake did not repair either common failure. On `p013-s3`, the
13-anchor combined policy consumed its two-step insertion budget, yet its
ImageReward harm increased from `0.9653` for static-only to `1.0753`. This is
consistent with the prior evidence that the Taylor error is a numerical stability
signal rather than an end-to-end preference signal.

This screen does not isolate whether the ImageReward failures are caused by one
specific tail gap. The candidates are atomic schedules, and opened labels may not
be used to scan replacement masks. It does establish the decision actually needed:
none of the frozen, theoretically derived 12/13-anchor candidates is eligible for
confirmation.

## Consequence

The conditional 64-group confirmation and 32-group final holdout are not run. This
is a completed negative branch of the preregistered pipeline, not missing work.
Running those stages after a zero-representative screen would convert development
labels into an unregistered candidate-selection loop.

The machine-readable summary is
[`benchmark/flux_cache/derived-schedule-development-screen-result.json`](../benchmark/flux_cache/derived-schedule-development-screen-result.json).
