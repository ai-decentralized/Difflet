# FLUX cache phase-schedule A2 result

Status: stopped by the preregistered exact-control rule. No serving claim.

## Decision

A2 collected and scored every registered source request. It observed 12 VQAScore
source failures, so the minimum-positive-count futility rule did not fire. The
frozen selection order then chose six failures from the i16/k20 profile. Three of
those failures were in the `spatial-relation` category, but that profile had only
two unused exact-profile, exact-category pass controls. The registered selector
therefore returned:

```text
status = insufficient_matched_controls
reason = no unused exact profile/category control for profile-1::p013-s2
```

The registration fixes `insufficient_exact_controls_action` to
`stop_insufficient_matched_controls`. No failure was replaced, no cross-profile
control was borrowed, and no prompt, profile, margin, or intervention grid was
changed after labels opened. Terminal-horizon collection and A3 repair-depth
collection were not permitted. At this point in the sequence, brake-only remained
the frozen legacy comparator; its later prospective confirmation was rejected.

## Frozen identity

- Study: `flux-cache-phase-schedule-horizon-development-2026-08-05`
- Git commit: `d391597d8d4ccac1efdb81511158904bc2b45c9e`
- Experiment branch: `experiment/phase-schedule-horizon-20260805`
- Registration content SHA-256:
  `e49062b34084f63e4901dd0c3de2f6f885accf07c81269ebcef99faa1cd1bfc6`
- Registration file SHA-256:
  `cb6e0ab48f776c884af078cc7211d5767c34d75a4257e1f827dad7361427ec74`
- Model: `black-forest-labs/FLUX.1-dev` at
  `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`
- Generation: 50 steps, 1024x1024, guidance 3.5, bfloat16, TP=4, seed 2
- Prompt split: `phase_schedule_horizon_development`, 48 prompt groups
- Source profiles: registered i12/k16 and i16/k20 adaptive profiles
- Quality margins: ImageReward harm `> 0.7824214100837708`; VQAScore harm
  `> 0.25`

## Collection and scoring

Collection ran from `2026-08-05T08:07:38Z` to `2026-08-05T08:26:06Z` and
produced all 144 registered images: 48 shared full-DiT baselines and 48 images for
each source profile. Semantic scoring completed at `2026-08-05T08:33:40Z` with
`complete=true` for all 144 ImageReward and VQAScore records.

The scorer used the frozen checkpoint revisions and hashes. The new machine was
missing two entries in the evaluator-specific Hugging Face cache; those entries
were populated from byte-identical local checkpoints before VQAScore was rerun.
No model or revision changed.

| Profile | Speedup | Skipped steps/request | VQA failures | ImageReward failures | Both-metric passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| i12/k16 | 4.2203x | 39/50 | 2/48 | 15/48 | 32/48 |
| i16/k20 | 4.7697x | 40/50 | 10/48 | 31/48 | 14/48 |

Across both profiles, 12/96 candidate requests failed the VQAScore margin and
46/96 passed both quality margins. These development counts do not certify either
profile for serving.

## Frozen selection outcome

The six selected failures, in preregistered order, were:

| Evaluation id | Category | VQAScore harm | ImageReward harm |
| --- | --- | ---: | ---: |
| `profile-1::p008-s2` | spatial-relation | 0.265625 | 2.120555 |
| `profile-1::p010-s2` | spatial-relation | 0.796875 | 0.717484 |
| `profile-1::p013-s2` | spatial-relation | 0.480469 | 0.556107 |
| `profile-1::p019-s2` | attribute-binding | 0.259766 | 1.384250 |
| `profile-1::p022-s2` | attribute-binding | 0.339844 | 3.002599 |
| `profile-1::p029-s2` | interaction | 0.764648 | 0.423477 |

The i16/k20 pass pool contained two `spatial-relation`, two
`attribute-binding`, and one `interaction` exact-profile controls. The third
spatial failure therefore had no eligible one-to-one control. The selector cleared
the partial control list and emitted zero selected controls, as required by the
registered fail-closed implementation.

## Artifact bindings

Artifacts are under
`/home/ubuntu/difflet-artifacts/flux-cache-phase-schedule-source-20260805`.

| Artifact | SHA-256 |
| --- | --- |
| `quality-input.json` | `086a4cc13de392030e4c573ff1496543754d1444752cabdefb90588ee6dda762` |
| `speedup-candidates-v1.json` | `3af26edcc799b6fc3bd8f82b8f314d4a0684f14bd3606b4b7fb121d458cefbda` |
| `semantic-scores.json` | `e0b10c2b807863c928ab235b1ff8a8b7d5c9cfdd46e0ab87eae8c889a699ccbb` |
| `phase-schedule-selection.json` | `c380f01a6b0fae053622ab690bfc736df44e5524f941a8555ecf8e8e33dec892` |

No terminal or repair artifact directory was created.

## Interpretation

This result does not show that the phase-aware static schedule hypothesis is
false; the causal horizon was never measured. It shows that this registered A2
design could not construct its required balanced failure/control intervention set
near the chosen profile frontier. Any retry must be a new preregistered study with
new data. Changing the selection algorithm on these opened labels would not be a
valid continuation of this A2 run.

A separately registered scheduler-weighted derivation later generated candidates
without reading these semantic labels. Its development screen stopped with no
eligible representative; see
`docs/flux-cache-derived-schedule-development-screen-result.md`. The subsequent
prospective brake-only confirmation also failed, so references above to the
existing brake-only path mean a legacy comparator, not a serving qualification.
