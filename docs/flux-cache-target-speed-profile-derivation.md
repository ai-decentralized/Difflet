# FLUX cache profiles derived from a hardware speed target

Status: implemented candidate-generation method. The generated profile is a
development artifact and has no serving or quality claim until it passes a frozen
end-to-end quality and measured-speed confirmation.

## Decision rule

The profile generator no longer accepts a manually selected list such as
`anchor_budgets: [12, 13]`. Its external performance input is one minimum hardware
speed target.

The bound hardware timing manifest must contain a shared full-DiT baseline and at
least two static-mask measurements with different real-step counts. The generator
fits the deterministic affine model

```text
aggregate_latency_s = intercept_s
                    + incremental_real_step_s * real_steps
```

For baseline latency `L_full`, target speedup `S`, and a frozen two-step online
brake reserve, it computes

```text
maximum_latency       = L_full / S
total_real_step_budget = floor(
    (maximum_latency - intercept_s) / incremental_real_step_s
)
static_anchor_budget   = total_real_step_budget - 2
```

The rule selects the largest total real-step budget that still meets the requested
minimum speed. This spends all available latency on quality rather than scanning
multiple anchor counts. Invalid numerical state and fail-closed execution are
explicitly outside the speed guarantee.

After the count is fixed, the existing scheduler-weighted dynamic program chooses
the anchor locations. The q95/q99 anchor-error rules then derive the two brake
threshold values. Neither stage reads semantic quality labels.

## Registered 1024x1024 example

The current registration binds the measured Trainium manifest at
`/home/ubuntu/difflet-artifacts/flux-cache-derived-schedule-development-screen-20260805/speedup-candidates-v1.json`
and requests a minimum `3.2x` speedup.

The two measured static points are:

| Real steps | Aggregate latency over 32 requests |
| ---: | ---: |
| 12 | 117.4259116810 s |
| 13 | 125.4431222780 s |

The fitted incremental real-step cost is `8.0172105969 s` over the 32 requests,
with an intercept of `21.2193845178 s`. The shared full-DiT baseline is
`463.6633422929 s`. The formula produces:

```text
maximum latency at 3.2x = 144.8947944665 s
total real-step budget  = 15
online brake reserve    = 2
static anchor budget    = 13
predicted reserved speed = 3.2772928545x
```

The resulting static mask remains

```text
0, 1, 2, 3, 4, 5, 9, 15, 21, 29, 38, 45, 49
```

but `13` is now an output of the registered hardware rule, not an input candidate
chosen beside `12`.

Frozen artifacts:

- registration:
  `benchmark/flux_cache/target-speed-schedule-derivation-registration.json`;
- derivation:
  `benchmark/flux_cache/target-speed-schedule-derivation-result.json`;
- static and static-plus-brake candidates:
  `benchmark/flux_cache/target-speed-candidates/`.

The older 12/13 derivation, candidates, and development screen remain unchanged as
historical evidence.

## Command interface

```bash
python scripts/derive_flux_cache_schedule.py register \
  --quality-input /path/to/quality-input.json \
  --methodology benchmark/flux_cache/offline-gate-methodology-v1.json \
  --hardware-speed-manifest /path/to/speedup-candidates-v1.json \
  --target-speedup 3.2 \
  --study-id flux-cache-target-speed-schedule-derivation \
  --created-at 2026-08-06T00:00:00Z \
  --out /path/to/registration.json
```

Registration freezes the timing manifest hash, hardware identity, fitted model,
target speed, total budget, static budget, and derivation implementation hash.
Loading the registration recomputes the fit and budget and rejects any mismatch.

## Evidence boundary

The affine fit predicts the budget; it does not certify the resulting latency.
This first example has only two calibration counts, so its `R^2=1` is algebraic
and not evidence of general linearity. A fresh backend or resolution should collect
at least three static-mask timing points spanning the intended budget and must
measure the final combined profile end to end. Quality remains governed by the
separate frozen quality contract and independent confirmation.

Unattended hardware authorization and automatic fail-closed image adjudication
are specified in `docs/flux-cache-unattended-offline-policy.md`.

## Simplified build path

The deterministic combined candidate no longer needs a development screen. A
frozen one-command build spec sends that single candidate directly to an
independent end-to-end confirmation, then exports it only if quality and measured
speed both pass. The registered 1024x1024 build uses
`benchmark/flux_cache/target-profile-square-1024-build-spec.json` and the new
`target_profile_confirmation` split; neither contains opened target-profile
labels.
