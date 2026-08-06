# FLUX cache phase schedule: phase-aware static anchors with bounded online braking

Status: closed with a negative development result. A1 completed; A2 stopped by its
preregistered exact-control rule; A3 was not run. A separately registered,
scheduler-weighted derivation produced four static/combined candidates, but all
four failed the frozen development quality screen. No Stage D confirmation or
winner holdout was permitted. This document makes no serving claim.

Every experiment follows the discipline in `docs/flux-cache-offline-closure.md`:
atomic candidates, frozen hashes, no threshold retuning after labels open, and no
serving claim from development data.

Frozen execution identity:

- registration: `benchmark/flux_cache/phase-schedule-horizon-registration.json`;
- registration content SHA-256:
  `e49062b34084f63e4901dd0c3de2f6f885accf07c81269ebcef99faa1cd1bfc6`;
- new development split: 48 prompts at seed 2, two profiles, 96 candidate
  requests, and 48 shared full-DiT baselines;
- A3 is conditional: it may run only if A2 finds at least six source failures and
  produces non-null, correctly ordered `t_full_observed` and `t_dead_observed`.

The first study is fixed to FLUX.1-dev, 50 steps, and 1024x1024. Rescue horizons,
static anchor masks, and candidate certifications are resolution-specific evidence.
They may not be transferred to other resolutions with an existing quality contract.
Each added resolution must remeasure the horizon, rematerialize candidates, and run
an independent confirmation.

## Execution outcome

The original A2 path stopped at `insufficient_matched_controls`; see
`docs/flux-cache-phase-schedule-a2-result.md`. It did not generate a horizon or a
candidate.

The follow-up did not alter that stopped registration. It created a new registered
derivation from the 48 complete full-DiT trajectories already collected in A2. The
deterministic objective combines scheduler step mass with first-order Taylor
prediction error, takes a prompt-wise q95 envelope, and solves an exact-budget
dynamic program under frozen mid/tail gap constraints. It did not read semantic
labels. The resulting 12- and 13-anchor masks, with static-only and bounded-brake
variants, were atomically frozen before a new 32-group screen.

The screen result is `stop_no_eligible_representatives`:

- all four new candidates passed the registered speed gate versus legacy
  brake-only;
- all four failed the zero-failure quality gate on ImageReward-only failures;
- no static or combined representative was selected;
- the conditional 64-group confirmation and 32-group holdout were not run.

See `docs/flux-cache-derived-schedule-development-screen-result.md` and
`benchmark/flux_cache/derived-schedule-development-screen-result.json` for the
frozen outcome.

A successor candidate-generation method now derives one anchor count from a
frozen hardware speed target instead of registering `[12, 13]` manually. It does
not reopen or overwrite this stopped study. See
`docs/flux-cache-target-speed-profile-derivation.md` for the formula, bound timing
evidence, and unqualified generated candidates.

## 1. Motivation and evidence boundary

Existing evidence supports the following conclusions, and no stronger ones:

- Semantic repairability has temporal structure. In the extreme-pressure queue,
  all 12 failures are rescued by terminal braking at steps 7, 13, and 21; eight
  remain rescuable at step 29 and one at step 37
  (`terminal-brake-causal-label-result.json`). This is a rescue curve for one
  pressure profile, not a model-wide point of irreversibility.
- Every tested cheap trajectory signal failed its preregistered usability gate.
  AUC ranged from near random to roughly 0.72--0.78, but rankings did not transfer
  across queues and full recall required an unacceptable false-brake rate. The
  correct conclusion is that no per-request semantic signal is currently
  deliverable, not that the signals contain absolutely no semantic information.
- Historical profile comparisons support one-way authority as the safer design.
  Tightening-only brake-only produced 1/64 failures, a 7.20% upper bound, and
  3.387x speedup. The stage-acceleration profile produced 3/64 failures and an
  11.67% upper bound. Thresholds and maximum intervals also changed, so this is a
  design observation rather than an isolated causal test of `allow_acceleration`.
- Interventions are non-monotone. Mild tightening added roughly two to four real
  steps without damaging the small pass-control set, while terminal braking at
  step 29 added roughly 21 real steps and damaged two of 40 requests that had
  passed under continue-cache. The sample is too small to prove any intervention
  magnitude universally safe, but it is sufficient to require bounded authority
  and end-to-end certification.

Design principles:

1. Put semantic temporal structure into design-time static scheduling instead of
   requiring runtime recovery of unobservable semantic risk.
2. Keep online authority one-way: it may add real steps but may not remove or delay
   static anchors.
3. Bound ordinary online braking by a time window and an extra-real-step budget.
4. Keep the numerical fail-closed fuse globally active after ordinary brake
   authority expires.
5. Certify the complete static-plus-online system atomically. Certificates for its
   pieces do not compose.

## 2. Target architecture

```text
Layer 0  Phase-aware static skeleton, frozen as an explicit anchor mask.
         Generate candidates from R(t) and D(t,k), not from terminal curves alone.
         Preserve warmup, cooldown, and require_final_anchor.
         Static anchors may not be removed or delayed.

Layer 1  Bounded Taylor brake, optional pending a three-arm decision.
         Signal: estimate_relative_error already available at real anchors.
         Authority: insert real steps only between adjacent static anchors.
         allow_acceleration = false.
         Every ordinary tighten or recovery insertion consumes B_max.
         Authority window = [warmup_end, t_dead].

Layer 2  Global numerical fuse.
         Static-only: invalid measurement, NaN/Inf, or corrupted predictor state
         permanently disables caching.
         Combined: the same conditions plus the frozen recovery-count limit may
         permanently disable caching.
         Remaining real steps after fail-closed do not consume B_max.
```

The two independent anchor streams are combined by union:

```text
run_real(step) = static_mask[step]
                 or dynamic_mask[step]
                 or cache_disabled
next_anchor    = min(next_static_anchor, next_dynamic_anchor)
```

Constraints:

- The static stream is read-only and independent of previous dynamic anchors.
- A dynamic anchor may reschedule only the dynamic deadline. It may not reset or
  delay the static deadline.
- A step that is both static and dynamic executes once and consumes no dynamic
  budget.
- Consecutive recovery steps consume the dynamic budget one by one. Ordinary
  recovery stops when the budget is exhausted. If the recovery-count fuse is
  reached, caching is disabled globally instead.
- Every real anchor still computes the numerical measurement. Ordinary tighten
  authority ends outside the plastic window, but invalid-measurement handling and
  the global fuse remain active.

`AdaptiveAnchorPolicy._schedule_after_anchor` in
`difflet/pipeline/cache/policies.py` currently resets `_next_anchor_step` to
`step + interval` after every anchor. A static anchor therefore cannot be injected
into its single movable deadline. The combined implementation must maintain
separate static and dynamic state.

Existing thresholds are not inherited. The brake-only values
`tighten_error=1.19` and `recovery_error=1.50` were selected under an interval
distribution of 4--8. A new skeleton changes Taylor-error sampling times and its
observed distribution. Layer 1 thresholds must be developed as a finite atomic
candidate grid, frozen before confirmation labels open.

## 3. Stage A: prerequisite measurement

Stage A develops candidate-generation evidence and makes no serving claim. Before
execution, registration must freeze the prompt/seed matrix, profile order, budgets,
failure definitions, intervention grids, and stopping rules.

### A1. Zero-cost historical retrospective

Status: complete.

- machine-readable result:
  `benchmark/flux_cache/phase-schedule-a1-retrospective.json`;
- adjudication memo:
  `docs/flux-cache-phase-schedule-a1-retrospective.md`.

A1 did not modify the frozen A2/A3 parameters and supplies no `t_full` or `t_dead`
for mask generation.

Inputs:

- `terminal-brake-causal-label-result.json`;
- `terminal-brake-step21-futility-result.json`;
- `brake-intervention-pilot-result.json`;
- `terminal-brake-followup-result.json`.

Outputs:

1. A per-request rescue matrix plus the aggregate R(t) curve and post-hoc semantic
   diagnosis.
2. A per-request introduced-failure table, including pass controls rather than only
   source failures.
3. A contradiction adjudication. The opened stage-oil pilot contains two
   independent failure requests repeated across six phase targets and rescues no
   target at step 15 or later. The extreme-pressure queue contains 12 independent
   failures and rescues 12/12 at step 21. The only allowed conclusion is that the
   observed horizon depends on profile family, failure mechanism, and request
   distribution. Neither curve is universal.

Frozen interpretation limits:

- Pressure timing and step-29 completion contain VQAScore only. `p013-s2` and
  `p039-s2` are introduced VQA failures, not full two-metric contract failures.
- Semantic tags for the four step-29 non-rescues were assigned manually after
  outcomes opened. They are descriptive and forbidden from every gate.
- Request-level recomputation matches all historical source and rescue counts. The
  step-37 mean brake benefit differs from the historical aggregate by about
  `4.07e-05`; both values remain recorded and the old result is not rewritten.

### A2. Near-frontier horizon probe

Purpose: measure R(t) under a failure mechanism closer to deployment rather than
using only the extreme i32/k40 pressure profile.

1. Use a new development split that was not used for prior signal or profile
   selection. Run `adaptive-vqa-stress-i12-k16-candidate.json` and
   `adaptive-vqa-stress-i16-k20-candidate.json` on all 48 preregistered requests
   each, for `N_collect=96`. Collection may not stop after the sixth failure, and
   the profiles may not be replaced after outcomes are observed.
2. Define a source failure as
   `baseline_vqa - cache_vqa > vqa_margin` under the resolution-specific quality
   contract. Record ImageReward failures but do not include them in the R(t)
   denominator.
3. If fewer than six total VQA source failures are observed, record
   `insufficient_source_failures` and stop phase-schedule candidate generation.
   Do not extend prompts, increase pressure, lower the positive-count requirement,
   or fall back to an extreme-pressure schedule. The legacy brake-only profile
   remains the development comparator; this stop makes no serving claim.
4. Sort source failures by the frozen sample-id and profile order, select at most
   six, and match six continue-cache pass controls by profile and semantic category.
   Run both groups at terminal steps `{7,13,17,21,25,29,37}` with the bit-exact
   prefix-identity gate.
5. Report:

```text
R(t) = rescued_source_failures(t) / source_failures
I(t) = introduced_failures(t)      / matched_pass_controls
t_full_observed = latest tested t where R(t)=1 and I(t)=0, else null
t_dead_observed = earliest tested t where R(t)<=0.2 and every later R(t)<=0.2,
                  else null
```

These are development descriptions, not confidence guarantees, and they do not by
themselves select `G_mid` or `G_tail`.

The intermediate profiles may fail only 2%--10% of requests. Fewer than six VQA
failures among 96 requests is therefore an expected negative outcome, not an
execution failure. The registered futility rule prevents spending more evidence
budget on a horizon that is too weakly represented near the deployment frontier.

Null handling is frozen. The horizon JSON may record null, but the mask generator
does not accept null:

- `source_failure_count < 6` -> `insufficient_source_failures`, stop Stage A;
- `t_full_observed is null` -> `no_observed_full_rescue_window`, stop;
- `t_dead_observed is null` -> `no_observed_tail_relaxation_window`, stop;
- `t_dead_observed <= t_full_observed` -> `invalid_horizon_order`, stop;
- A3 and C1 are permitted only when both bounds are non-null and correctly ordered.

Every stop keeps the existing brake-only profile only as the frozen legacy
comparator. No conservative default mask may be introduced after outcomes are
observed, and no serving qualification is implied.

`phase-schedule-horizon.json` must bind at least:

- model id/revision, scheduler class/config hash, and step count;
- height/width, guidance scale, dtype, TP degree, and AOT graph identity;
- predictor type/order/coordinate and cache granularity;
- both source-profile paths and file hashes;
- prompt split, collector, intervention implementation, quality contract, and
  semantic-judge checkpoints/hashes.

Any change that affects numerical trajectories or label semantics invalidates the
old horizon.

### A3. Repair depth D(t,k)

Terminal braking computes every remaining real step and therefore measures an
upper bound on repairability. It does not establish that one anchor or a periodic
gap `G_mid` can repair the trajectory. A3 is mandatory before static candidate
generation.

Reuse the at-most-six source failures and six matched pass controls frozen by A2:

- `t` in `{13,21}`;
- `k` in `{4,8,16}`;
- run `k` consecutive real steps from `t`, then resume the original cache profile;
- run the complete 2x3 grid on all 12 requests, at most 72 intervention branches;
- require bit-exact latent, cache, and predictor state before `t`.

Definitions:

```text
D_rescue(t,k)    = fraction of source failures that pass VQA after repair
D_introduce(t,k) = fraction of matched controls that fail after repair
```

A3 only generates a small candidate grid. A development cell with full rescue and
zero observed introductions does not certify a static skeleton. Final images still
pass the two-metric Stage D gate. In particular, zero introductions among six
controls has a one-sided exact-binomial 95% upper bound of about 39.3%. Results and
presentations must call this a development diagnostic, never evidence that repair
cannot introduce failures.

### Known blind spot

R(t) and D(t,k) define source failures using VQA, while the quality contract is the
union of independent VQA and ImageReward failures. Tail sparsity may be semantically
safe but still harm texture or preference. Stage A reports ImageReward harm and
introduced failures without fitting a separate ImageReward time curve. Stage D
keeps both metrics as independent vetoes.

## 4. Stage B: implementation

### B1. Combined policy

Implement `PhasedStaticPolicy` (or `ExplicitMaskPolicy`) and
`StaticPlusBrakePolicy`:

- the static stream is read-only and materialized from candidate JSON;
- the dynamic stream implements a tighten deadline and consecutive recovery;
- the first tighten rule is frozen as `bisect_next_static_gap`, which inserts a
  midpoint between the current real anchor and the next static anchor rather than
  maintaining a global interval that can move the skeleton;
- `should_skip` uses the union of static and dynamic masks;
- only dynamic extra steps consume `B_max`; overlap with a static anchor is free;
- ordinary insertion stops outside the plastic window, while global fail-closed
  remains active;
- after disable, every remaining step is real and `require_final_anchor` remains
  satisfied;
- `stats()` exposes static-anchor count, dynamic insertions, overlaps, remaining
  budget, recovery count, disable reason, and trigger step.

### B2. Candidate schema

Use the functional schema `difflet-flux-cache-phased-candidate`, revision 1. Do not
put release aliases in class names, schema names, policy types, statistic fields, or
CLI names. Every field below is part of the atomic frozen candidate:

```text
policy.type                 = "phased_static_plus_brake" | "phased_static"
policy.num_steps            = 50
policy.static_anchor_steps  = [ ... ]
policy.warmup_steps / cooldown_steps
policy.require_final_anchor = true
policy.plastic_window       = [w_end, t_dead]
policy.dynamic_budget       = B_max
policy.tighten_error
policy.tighten_rule         = "bisect_next_static_gap"
policy.recovery_error
policy.recovery_steps
policy.disable_after_recoveries
policy.allow_acceleration   = false
policy.invalid_measurement_fail_closed = true
horizon_ref.path / sha256
quality_contract_ref.path / sha256
```

The static-only form fixes `dynamic_budget=0` and omits the combined-only threshold
fields. Both forms require invalid-measurement fail-closed.

Candidate loading lives in `scripts/flux_cache_phased_candidate.py`; the hardware
entry point is `scripts/collect_flux_cache_phased.py`. Do not change the
`scripts/collect_flux_cache_ab.py` hash frozen by the conservative confirmation.
The phase collector may reuse its A/B execution path, but new registration must bind
the base collector, phase collector, and candidate loader hashes.

### B3. Unit and replay tests

- No dynamic sequence may skip or delay a static anchor.
- A static/dynamic overlap executes once and consumes no dynamic budget.
- Every extra recovery step consumes budget; ordinary insertion stops at zero.
- Ordinary insertion is disabled outside the plastic window, while invalid
  measurement still triggers global disable.
- Disable produces real computation through the final required anchor.
- Static-only `materialize_anchor_mask` matches stepwise decisions.
- Combined `materialize_static_anchor_mask` plus an identical measurement sequence
  reproduces the same dynamic decision trace.
- Reset leaks no state across requests.
- Missing fields, out-of-range or non-increasing steps, window conflicts, and hash
  mismatches reject candidate loading.

### B4. Trainium and AOT boundary

An explicit static mask makes the anchor sequence deterministic and improves replay
and graph-cache behavior. The current system AOT-compiles Transformer and decoder
components, not the entire denoise loop as one static graph. This plan does not
claim that host branches or communication overhead disappear. Stage D measures
real end-to-end hardware wall time for both static and combined policies.

## 5. Stage C: candidate development and within-family selection

Stage C and Stage D use different prompt splits. Stage C may open development labels
to freeze later candidates, but it may not make a serving claim.

### C1. Deterministic finite grid

Generate explicit masks from A2/A3 rather than hand-tuning step numbers afterward:

- at most four static candidates: `G_mid` in `{4,6}` x `G_tail` in `{12,16}`;
- every combined candidate uses exactly the same static mask as its static control;
- `B_max` in `{2,4}`; tighten/recovery thresholds come from a finite grid frozen
  before labels open;
- register every candidate, generator identity, input horizon, and file hash before
  C2 collection.

### C2. Within-family development screen

Run every candidate on one fixed development split. Apply the two-metric contract
and prompt-group upper-bound diagnostic to each candidate, then freeze one
representative per family using this order:

1. prefer candidates that pass the quality diagnostic;
2. among passers, choose the shortest total measured hardware wall time;
3. break exact ties by candidate id;
4. if no candidate in a family passes, that family does not enter Stage D.

Stage D therefore receives at most one static representative and one combined
representative. It never selects the best mask or threshold on confirmation data.
After C2, thresholds, `B_max`, mask, plastic window, and representative identity are
immutable.

Combined-family threshold development must run complete candidates and select on
final quality plus wall time. It may not copy 1.19/1.50 directly or claim quality
safety from anchor-error quantiles alone.

## 6. Stage D: three-arm confirmation

### D0. Registration relationship

`brake-only-methodology-v1-confirmation.json` ran exactly as registered. It observed
one ImageReward-only failure in 32 groups, giving a one-sided 95% upper bound of
13.98%, and was rejected. A historical-code replay reproduced the failing image,
latent, trajectory, and controller decisions exactly, excluding recent code changes
as the cause. See `docs/flux-cache-brake-only-methodology-v1-confirmation-result.md`.

The methodology-v1 quality gate remains unchanged. Three-arm sampling quotas,
speed-advantage rules, and comparison order are new preregistered additions.

### D1. Arms

1. Legacy comparator arm: frozen brake-only; it is not a confirmed serving base.
2. Static arm: the C2 Layer 0 representative plus numerical fail-closed.
3. Combined arm: the same Layer 0 mask plus Layer 1 and Layer 2.

If a new family has no C2 representative, that arm is absent. If neither new family
has a representative, Stage D stops and no comparator-only confirmation is run.

### D2. Data and quality gate

- Register 64 new independent prompt groups with one seed each.
- Allocate at least eight groups to each of counting, spatial relations, attribute
  binding, text rendering, and other. Categories affect sampling and diagnostics,
  not the group-level gate.
- Share the paired full-DiT baseline, prompt, seed, and initial latent across arms.
- ImageReward and VQAScore veto independently and are never averaged.
- Each arm must have a one-sided exact-binomial 95% failure-rate upper bound <=10%.
- The evaluator derives the maximum allowed failures for 64 groups from the frozen
  exact-binomial rule; this document does not hard-code an approximation.

### D3. Speed estimand

After AOT compilation and a fixed number of warmups, record ordinary end-to-end wall
time for each prompt:

```text
relative_speed = sum(wall_time_current_brake_only)
                 / sum(wall_time_new_arm)
```

A new arm has deliverable speed advantage only if both hold:

- `relative_speed >= 1.05`;
- a prompt-group paired bootstrap 95% lower bound is greater than 1.00.

Bootstrap repetitions, random seed, warmup count, timing boundary, and outlier rules
must be in registration. The bootstrap clause may be removed before registration,
but its use may not be decided after timings are observed.

This threshold intentionally allows the negative result "quality passes,
relative_speed=1.03, keep current brake-only." Expected tail-sparsity benefit is only
about 5%--10%, so the gate may sit at the edge of the effect. It may not be lowered
after observing a near miss. The gate asks whether a new profile justifies its
engineering and certification cost, not whether temporal structure exists.

### D4. Independent holdout

D1--D3 select at most one winner. If one exists, confirm that frozen winner on 32
new independent prompt groups with zero failures; the one-sided exact-binomial 95%
upper bound is about 8.94%. Remeasure wall time, with no early stopping, candidate
reselection, or field adjustment. Only a passing winner may enter later
serving/interventional release work.

Registration must recalculate budget from the actual arm count. For three arms and
64 groups, a rough estimate is 64 baselines plus the cached compute for three arms,
or about 130--160 full-DiT equivalents. A2/A3 and Stage C are separate budgets and
must not be counted as confirmation evidence.

## 7. Preregistered decision rules

Apply quality before speed:

| Condition | Decision |
|---|---|
| Static and combined pass quality | Do not claim a quality difference. Only arms meeting the speed gate versus current brake-only can win; with an identical mask, prefer shorter total wall time. |
| Static fails and combined passes on the same mask | Layer 1 has end-to-end value evidence, but combined must still meet the 5% speed gate. |
| Static passes and combined fails | Remove Layer 1; static may win only if it meets the speed gate. |
| New arms pass quality but miss the speed gate | Keep brake-only as the legacy comparator and record "quality feasible, engineering benefit insufficient." |
| Both new arms fail quality | Keep brake-only as the legacy comparator, record the negative result, and close this phase-schedule iteration. |

If multiple new arms pass quality and speed, choose the shortest total wall time and
break exact ties by candidate id. Rare failure-count differences among 64 groups are
qualification gates, not evidence that one arm has significantly better quality.

An ImageReward-only rejection does not prove `G_tail` is the cause. At most one new
preregistered iteration may tighten `G_tail` by a transformation declared in
advance, using new development and confirmation splits. If it fails again, stop.
Do not scan skeletons on opened labels.

Explicitly forbidden on opened Stage D or holdout labels: selecting new Layer 1
thresholds, static steps, window bounds, `B_max`, timing statistics, or family
representatives. Any such change starts a new study.

## 8. Deliverables

- Stage A registration and result. The registered stop means no horizon or R/I/D
  artifact exists;
- deterministic static-mask generator and tests;
- combined policy and state-machine tests in
  `difflet/pipeline/cache/policies.py`;
- scheduler-weighted derivation registrations/result, four atomic candidates, and
  the Stage C screen registration/result;
- no Stage D registration/result or winner holdout, because Stage C selected zero
  representatives;
- updated `docs/flux-cache-offline-closure.md` with the positive or negative final
  conclusion.
