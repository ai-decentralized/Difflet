# FLUX brake-only methodology-v1 confirmation result

Status: rejected by the preregistered confirmation gate. No serving claim.

The frozen 32-group confirmation completed all baseline and candidate images and
scored every image with both registered judges. The brake-only candidate produced
one contract failure. The exact-binomial one-sided 95% upper bound is 13.98%, above
the registered 10% maximum; the registration requires zero failures in 32 groups.

| Quantity | Result |
| --- | ---: |
| Independent prompt groups | 32 |
| Contract failures | 1 |
| Failure-rate upper bound | 13.9849% |
| Measured speedup versus full DiT | 3.3198x |
| Serving qualified | No |

The failure is `p017-s0`. Its ImageReward harm is `0.9166896995`, exceeding the
frozen `0.7824214101` margin. Its VQAScore harm is `0.146484375`, below the frozen
`0.25` margin. The failure is therefore ImageReward-only. The online controller did
not tighten or enter recovery for this request; it ran 13 full Transformer steps,
skipped 37, and ended with anchor error `0.3441113978`.

## Historical-code replay

To test whether recent cache changes caused this result, the exact prompt, seed,
candidate, and generation settings were replayed at historical commit
`64b70abdc7e34058f16f25ec6113829d1c08af99`, predating the recent phase-schedule
work. The replay candidate PNG has the same SHA-256 as the confirmation PNG:

```text
cedf49c31ca666fe49751c8eb0527094b4c0bc1759ffb305d8f5792d693c7bff
```

The baseline PNG also matched byte for byte. Both final latents and every stored
trajectory tensor were elementwise identical; controller statistics were identical
apart from elapsed wall time. The prospective rejection is therefore not explained
by recent code changes. It is a new-prompt quality failure of the frozen candidate.

## Evidence boundary

This result does not erase the historical 1/64 development result. It changes its
status: brake-only remains a useful legacy/development comparator and numerical
fail-closed policy, but it is not prospectively confirmed as a serving-safe base.
No threshold may be retuned on this opened holdout.

The machine-readable summary is
[`benchmark/flux_cache/brake-only-methodology-v1-confirmation-result.json`](../benchmark/flux_cache/brake-only-methodology-v1-confirmation-result.json).
