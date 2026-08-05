# FLUX cache offline selection and threshold closure

This document separates three claims that must not be collapsed into one:

1. selecting an atomic cache profile offline;
2. developing online brake/oil score thresholds;
3. qualifying the frozen profile and controller for serving.

The machine-readable policy is
[`benchmark/flux_cache/offline-gate-methodology-v1.json`](../benchmark/flux_cache/offline-gate-methodology-v1.json).
The executable workflow is
[`scripts/flux_cache_offline_gate.py`](../scripts/flux_cache_offline_gate.py).

## Evidence flow

```text
freeze methodology + prompt groups + candidate file hashes
                         |
                         v
             paired baseline/candidate A/B
                         |
              +----------+-----------+
              |                      |
              v                      v
     ImageReward + VQAScore     measured wall time
       fail if either fails       on hardware
              |                      |
              +----------+-----------+
                         v
        offline atomic-profile selection
      quality UCB first, then maximum speedup
                         |
                freeze candidate hash
                         v
           prospective profile confirmation

For an additional semantic risk gate:

failure-enriched development -> prompt-grouped OOF scores
      -> brake/oil threshold selection -> freeze gate
      -> untouched threshold confirmation
      -> interventional controller A/B -> serving claim
```

Development data never makes a serving claim. A positive development result
only freezes the next artifact; an untouched confirmation and an
interventional quality/speed run remain mandatory.

## Offline cache-profile selection

An adaptive candidate JSON is the atomic selection unit. Its anchor intervals,
warmup/cooldown, predictor, `tighten_error`, `recovery_error`,
`acceleration_error`, and `allow_acceleration` are selected together. Individual
controller thresholds must not be changed after quality labels are opened.

The selection order is fixed:

1. Apply the frozen ImageReward and VQAScore contract independently. Do not
   average the metrics.
2. Require the one-sided exact-binomial 95% prompt-group failure-rate upper
   bound to be at most 10%. Seeds from one prompt do not create additional
   independent prompt groups.
3. Among eligible profiles, select maximum measured hardware speedup.
4. Break an exact tie by `candidate_id`.
5. Freeze the selected candidate file hash and evaluate it on a newly
   registered profile holdout.

The historical stage-gate result can be replayed with:

```bash
python scripts/flux_cache_offline_gate.py profile-replay \
  --methodology benchmark/flux_cache/offline-gate-methodology-v1.json \
  --stage-result benchmark/flux_cache/stage-gate-study-result.json \
  --candidate stage_acceleration=benchmark/flux_cache/adaptive-stage-oil-p30-e1p40-k12-candidate.json \
  --candidate brake_only=benchmark/flux_cache/adaptive-brake-candidate.json \
  --out benchmark/flux_cache/profile-selection-replay.json
```

This replay selects the brake-only candidate: interval 4--8,
`tighten_error=1.19`, `recovery_error=1.50`, and
`allow_acceleration=false`. Its cumulative result is 1 failure in 64 prompt
groups, with a one-sided 95% upper bound of 7.20%, and its measured speedup is
3.387x. The stage-acceleration profile is rejected at 3 failures in 64 and an
11.67% upper bound despite its higher 3.741x speedup.

This is a deterministic replay of legacy evidence, not a new serving
qualification: methodology-v1 was written after those data, and one of the two
32-prompt audits was frozen but not contract-allowlisted.

### Prospective brake-only confirmation

The conservative candidate is now prospectively frozen for a new
methodology-v1 confirmation in
[`benchmark/flux_cache/brake-only-methodology-v1-confirmation.json`](../benchmark/flux_cache/brake-only-methodology-v1-confirmation.json).
The registration binds the exact candidate, methodology, collector, semantic
scorer, metric checkpoints, generation settings, and a new 32-prompt split in
[`benchmark/flux_cache/brake-only-confirmation-prompt-suite-v1.json`](../benchmark/flux_cache/brake-only-confirmation-prompt-suite-v1.json).
The new prompt text has zero exact overlap with 266 previously registered prompt
texts. Each prompt uses one seed, so the study has 32 independent prompt groups.

The confirmation must score every registered sample and observe zero failures.
For 0 failures in 32 groups, the one-sided exact-binomial 95% upper bound is
8.94%, below the frozen 10% maximum. One or more failures rejects the
confirmation. Passing permits a new prospective interventional quality/speed
holdout; it does not qualify serving by itself.

Validate the freeze before collection:

```bash
/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python \
  scripts/flux_cache_profile_confirmation.py validate \
  --registration benchmark/flux_cache/brake-only-methodology-v1-confirmation.json
```

The exact collection, scoring, and confirmation-evaluation argument vectors are
stored in the registration's `collection` object. The hardware run has not yet
started, and the registered holdout outcomes remain unopened.

## Online semantic risk thresholds

The online gate uses only features available on a real cache request. It does
not use a shadow or full-DiT teacher at skipped steps.

Model development is leave-one-`prompt_index`-out. Both seeds and all candidate
strengths for one prompt remain in the same fold. Feature standardization is
fit inside each fold. Feature-group selection uses grouped-OOF average
precision; threshold selection uses only target-profile OOF probabilities.

- Brake threshold: the minimum failing target-profile OOF score, used with
  `risk_score >= threshold`. It must recall every observed target failure and
  keep passing-request false brakes at or below 10%.
- Oil threshold: the largest cutoff used with `risk_score <= threshold` whose
  all-clear zone has zero failing prompt groups and a one-sided exact-binomial
  95% upper bound at or below 10%. At least 29 independent zero-failure prompt
  groups are required; repeated seeds do not increase this count.

Run the development closure with:

```bash
python scripts/flux_cache_offline_gate.py develop \
  --registration benchmark/flux_cache/online-signal-failure-enriched-audit.json \
  --methodology benchmark/flux_cache/offline-gate-methodology-v1.json \
  --quality-input /path/to/quality-input-v2.json \
  --semantic-report /path/to/semantic-scores.json \
  --development-out /path/to/gate-development.json \
  --decision-out /path/to/gate-closure.json \
  --gate-out /path/to/frozen-gate.json
```

The command validates prompt/candidate hashes, the complete Cartesian sample
matrix, the semantic-report binding, feature identities, quality margins, and a
clean collection commit before fitting. `--gate-out` is written only if every
development rule passes and the registration binds the exact methodology hash.

## Current closure state

| Component | Current result | Serving status |
|---|---|---|
| Offline adaptive profile | Brake-only profile and a fresh 32-prompt confirmation are frozen; collection has not started | Prospectively registered, not yet confirmed; no serving claim |
| Learned semantic brake threshold | Full target failure recall needs 66.7% false brakes | Rejected |
| Learned semantic oil threshold | Best all-clear zone has 6 independent prompt groups; 95% upper bound 39.3% | Rejected |
| Oil/acceleration profile | Quality upper bound 11.67% exceeds the 10% contract | Rejected |

The current engineering decision is therefore conservative: retain the frozen
brake-only candidate for further confirmation work, do not enable acceleration,
and do not deploy the learned semantic vote. The research decision is closed;
serving qualification is intentionally not claimed.

## Paired brake intervention

The next causal question was whether a cheap anchor-time signal controls final
quality, rather than merely correlating with failures. The frozen development
pilot replays the exact same prompt, seed, latent state, Transformer output, and
cache history to an early, middle, or late real anchor. It then deterministically
randomizes one of three post-anchor actions: continue caching, force the minimum
interval, or enter a short recovery window. It does not run a shadow Transformer
at skipped steps.

The 37-image pilot passed all 27 paired identity gates and exactly reproduced
the prior natural-cache and full-DiT images for both known failures. Neither
brake nor recovery rescued any of the six known-failure phase targets. Brake
added 2.44 full Transformer steps on average; recovery added 4.00. All three
pass controls remained inside the quality contract, so the observed cost of a
wrong brake was extra compute rather than a contract failure in this small
control set.

Anchor error ranked ImageReward treatment gain in this opened sample
(`rho=0.733` for brake), but barely ranked the VQAScore gain that defines these
failures (`rho=0.143`; recovery `rho=-0.051`). Because no action actually
rescued a failure, these correlations do not justify a threshold. The registered
decision is `negative_pilot`: do not retune scalar anchor-error thresholds on
these requests; change the predictor or cache granularity before registering a
new causal holdout.

A follow-up tested the actuation upper bound: from each of the same nine target
anchors, permanently disable cache and compute every remaining Transformer
step. All target-prefix identity gates passed. Terminal brake added 21.33 full
steps on average but still rescued zero of six known-failure targets; all three
controls remained pass. For `p001-s1` it recovered at most 18.6% of the VQA gap,
while every `p014-s0` terminal arm slightly reduced VQA relative to continue.
Thus the registered step-15-or-later anchors are already too late for a reliable
semantic fallback. This does not test the first measurable anchor before the
first long skip interval: the source policy has a post-warmup anchor at step 7,
whereas the registered early target is step 15.

## Pre-skip warmup VQA router

An opened-data metric-specific search found one cheap signal that is available
before the first cache skip: the maximum spatial coefficient of variation of
positive acceleration in the shallow 4x4 block-0 modulation map over steps 2
through 5. Lower values were riskier for the two opened VQAScore-failing prompt
groups. The frozen development trigger is `signal <= 0.04691181821870764` and
the target label remains paired VQAScore harm strictly greater than `0.25`.

Routing triggers to the existing brake-only cache profile was rejected: that
profile still skipped 33 to 37 of 50 Transformer steps and repaired zero of
four failure comparisons. A true terminal action at step 6, before the first
eligible skip, repaired all four opened comparisons with no introduced
contract failure. All four terminal images were byte-identical to their
full-DiT baselines. This is an actuation upper bound, not a speed-optimal brake;
the opened 32-request matrices cost 26.4% to 31.5% more after routing.

The exact signal, threshold, and terminal action were then frozen on a new
48-prompt, one-seed prospective holdout. The stage oil profile produced zero
targeted VQAScore failures; maximum paired VQAScore harm was `0.10546875`.
Therefore AUC and failure recall are undefined and the registered result is
`prospective_inconclusive_too_few_positive_groups`, neither confirmation nor
rejection. The signal terminal-routed five passing requests (10.42% false-route
rate), raising diagnostic total cost by 29.49%. One untriggered ImageReward-only
failure remained outside this VQA-specific router's scope.

The serving decision remains unchanged: do not install this router. Another
same-profile prompt expansion has low value because the positive label is too
rare. If work continues, first register a higher-cache-pressure stress arm
that can produce enough VQAScore positives, while keeping this threshold frozen.

## Terminal-brake causal labels under stress

The registered `i32/k40` stress profile produced 12 VQAScore failures in 48
prompt groups. A timing sweep replaced the observational label “continue-cache
eventually fails” with the intervention label
`brake_benefit(t) = VQA(terminal@t) - VQA(continue-cache)`. Every one of the 96
intervention branches passed a bit-identical prefix-latent gate, and all 48
rescored baseline VQA values exactly matched the source report.

Terminal braking at steps 7, 13, and 21 rescued all 12 source failures. The
rescue count fell to 8 at step 29 and 1 at step 37, establishing a real rescue
horizon. The step-29 intervention was then completed for all 48 requests. It
reduced 12 continue-cache failures to six when applied universally, but four
source failures were already unrescuable and two previously passing requests
became VQA failures. Universal terminal@29 increased measured runtime by
173.1%, so terminal braking is neither free nor quality-monotone from a cached
prefix.

A cheap regional output-change signal selected on the 12 failures improved
from AUC 0.551 against the observational final-failure label to AUC 0.722
against the actionable terminal@29 label. This confirms that causal-target
mismatch was material. It does not close the online gate: full actionable
recall routed 36/48 requests, falsely braked 28/40 negatives, retained six
failures, and increased runtime by 130.0%. The best fully post-hoc feature only
reached AUC 0.781 and still required 21/40 false brakes for full recall.

The registered decision is therefore
`causal_target_alignment_confirmed_gate_rejected`. Do not send the step-29
gate to a serving holdout. If scalar-signal work continues, the remaining
bounded experiment is a pre-registered step-21 intervention with exactly one
signal available by step 20; another failure should end whole-output scalar
search and move to a different predictor or cache granularity.

## Step-21 futility and spatial anchor error

The step-21 follow-up stopped before running 36 additional terminal branches.
All 12 source failures were already known to be rescued at step 21, so the
actionable positive set is exactly the 12 continue-cache failures; terminal
outcomes for passing controls cannot improve the online-signal ranking. Across
the bounded zero-extra-DiT scalar family available by step 20, the best opened
feature reached AUC `0.731` and needed 25 false brakes among 36 passing
requests for full recall. A broad 4x4 spatial-consensus variant reached only
AUC `0.722`. Both exceeded the pre-hardware futility limit of nine false
brakes, so completing the control images would consume full-DiT work without
changing the gate decision.

The final scalar diagnostic measured Taylor prediction error by 4x4 region at
an already-computed real anchor. It added no Transformer calls and was fully
passive: all 48 images were byte-identical, and all 48 final latents and
50-step trajectories were tensor-identical to the source run. The registered
step-7 score—the mean of the four largest regional relative errors—had AUC
`0.463`; full recall triggered 47/48 requests and falsely braked 35/36 passes.
Exploratory aggregation at step 7 peaked at AUC `0.604`. The best later result,
AUC `0.655` at step 39, is after the terminal@21 rescue horizon and still
requires 33 false brakes.

This closes scalar whole-output dynamics and scalar/global-or-regional anchor
error aggregation as the route to an online semantic guarantee. The regional
measurement primitive itself is valid and cheap, but its values do not retain
the missing VQA-risk information. A further signal experiment must change the
predictor or cache granularity; another threshold retune on these opened
trajectories is not registered work.

## Partial-block prompt-token coverage sentinel

The final bounded signal experiment changed the predictor rather than retuning
another Taylor-error aggregation. At step 7, a separate Trainium/AOT component
computes only the FLUX block-0 image queries and image/text keys, pools image
queries to 4x4, and returns a fixed `16 x 512` text-attention probability map.
It omits values, attention output, MLP, and all later Transformer blocks. Thus
it adds one shallow partial-block call but zero full-Transformer calls.

The frozen request score was the mean of the weakest quartile of prompt-token
regional maxima, with lower coverage treated as riskier. On the same 48-request
stress cohort and the 12 terminal@21-actionable VQA failures, its AUC was
`0.505`. Full recall required routing 46/48 requests and falsely braking 34/36
passing requests. The best opened exploratory feature reached only AUC `0.606`
and still required 32 false brakes. All 48 images were byte-identical, and all
final latents and full trajectories were tensor-identical to the source run.
The directly timed probe cost was 0.386 seconds across 48 requests, about 0.27%
of the source candidate's summed runtime; end-to-end timing variance is not
treated as a speedup claim.

This closes early block-0 prompt-token coverage as a brake signal. Together
with the prior scalar and spatial-error negatives, the remaining high-value
direction is no longer another threshold search. It is a changed cache
representation: keep attention residual fresh while caching or predicting the
MLP residual, and test whether that improves quality enough to widen cache
intervals.

## Component-residual proxy result

The component follow-up separated two questions that the whole-output probes
had conflated. First, a `24 x 32` block-input sketch (1,536 BF16 bytes per
block) can live in shared HBM across two Trainium AOT buckets. Its same-step
cross-bucket floor and maximum anchor/online difference were both exactly
zero, with no cache-payload host transfer. On synthetic perturbations the
sketch tracked true block-0 MLP residual error with Spearman `0.994`.

Second, the real 48-request trajectory test did not close the quality chain.
At the step-20 decision point, the preselected AOT-exact raw-sample Taylor
error tracked true MLP Taylor error at Spearman `0.748`, below the registered
`0.8` gate. A normalized-input candidate reached `0.874`, showing that a cheap
component proxy is technically possible. A follow-up two-bucket AOT graph
measured a same-step normalized-signal floor of `0.000985` and a maximum
anchor/online discrepancy of `0.000618`, versus real-trajectory signal values
around `2` to `5`. This is a usable exploratory offline/online numerical
bridge, though unlike the raw signal it is not bit-exact or confirmatory.
More importantly, true offline MLP Taylor error itself had failure AUC only
`0.456` (`0.544` with the direction reversed). The normalized candidate's best
direction reached AUC `0.572`; full recall still falsely braked 33 of 36 passing
requests. The AOT-exact raw signal had AUC `0.509` and 34 false brakes.

Therefore the failed link is no longer merely an offline/online numerical
mismatch: single-block MLP error is not an end-to-end VQA-risk target on this
cohort. No component-error brake threshold may be selected from these opened
rows. If residual splitting continues, the bounded next experiment is to use
it as a fixed cache representation—keep attention fresh, cache or predict the
MLP residual—and measure final quality directly, without adding another
adaptive scalar gate.

## x0 preview and lightweight semantic heads

A no-extra-DiT preview was reconstructed from two adjacent cached trajectory
states with the flow-matching scheduler identity
`x0 = x_t - sigma_t * (x_t - x_prev) / (sigma_t - sigma_prev)`. Decoding that
latent with the already compiled FLUX VAE cost about 0.293 seconds per request.
The same frozen XL VQAScore used offline was informative on the preview: at
step 20, low preview score ranked eventual continue-cache failures at AUC
`0.829`; at step 28 the AUC was `0.874`, and preview-to-final-cache VQAScore
Spearman was about `0.89`. On the independent 32-request oil cohort, the
direction persisted at AUC `0.767` and preview-to-final Spearman `0.876`.

This is the first runtime observation that genuinely maps to the offline
semantic metric, but it is not a cheap serving brake. The XL scorer adds a
large secondary model and roughly 1.2–1.5 seconds per image on the measured CPU
path. Its fixed threshold also did not give full recall: step 28 captured 7/8
terminal@29-actionable samples with nine false brakes; the remaining p030
sample required an opened, brittle second condition.

Three attempts to remove the large judge failed. Frozen CLIP embeddings with
an out-of-fold head did not distill preview VQA. BLIP-VQA base yes/no sequence
likelihood reached actionable AUC `0.734` on the stress cohort and continue-
failure AUC `0.683` on the independent cohort, but its frozen threshold missed
one of two independent failures and its Spearman with XL preview VQA was only
`0.074` and `0.311`. A BLIP COCO image-text-matching head reached stress
failure AUC `0.778`, then fell to `0.533` independently and routed none of the
independent failures at the frozen threshold. Pure x0-latent convergence also
failed: its frozen primary actionable AUC was `0.40`; an opened feature that
reached `0.775` triggered 27/32 requests on the independent cohort.

The serving interpretation is precise: x0-preview plus the offline XL teacher
is an effective but expensive online measurement. No tested off-the-shelf
small head or latent-only proxy preserves that mapping. Do not hide this cost
by calling the VAE or small VLM "free," and do not select another threshold
from the opened rows.

Visual-token compression did not make the XL teacher deployable. A stage-level
CPU profile measured the unmodified judge at a median 1.366 seconds per image:
about 1.014 seconds was the T5 encoder, 0.147 seconds the CLIP vision tower, and
0.155 seconds the two-token T5 decoder. Average-pooling the 24x24 CLIP patch
grid to 12x12 reduced the judge to 0.497 seconds, but actionable AUC collapsed
from 0.791 to 0.586 and full recall falsely braked all 40 negatives. Keeping an
unchanged maximum-norm patch from every 2x2 cell also failed: actionable AUC
was 0.688, the nine-false-brake operating point caught only 3/8 stress
positives, and its frozen threshold caught 1/2 independent failures. Keeping
two patches raised latency to 0.657 seconds and caught 0/2 independent failures
at the frozen threshold. These results reject both averaging and representative
patch selection as serving approximations to preview XL-VQA.

The Trainium placement does not turn this cost into a free cache bubble. The
four-core `trn2.3xlarge` application loads the decoder together with the other
FLUX components at worker startup, so a steady-state request does not pay the
standalone decoder's roughly 16-second cold load. The Transformer is TP=4,
however, and the compiled components use synchronous execution
(`async_mode=False`). The aggressive stress profile performs real Transformer
work only at steps 0--5, 7, 39, and 49. The skipped steps advance on the host
too quickly to hide the resident decoder before a terminal@29 decision. A
single-process, four-core hardware intervention measured the unmodified
step-20-to-terminal@29 interval at 21.4 ms. Launching the already-resident VAE
in a Python thread made the decode finish in 307.9 ms, but inflated the same
deadline interval to 317.6 ms and added 292.7 ms to denoising end to end. The
preview appeared ready in 2/2 runs only because it delayed the deadline that
it was meant to meet; every final latent remained identical. Thus about 95% of
the resident VAE time remained on the critical path. Do not count logical
skipped diffusion steps as wall-clock overlap, and do not introduce request-
level host-to-device weight swapping for a brake signal.

## Deep joint-attention semantic vote

The final semantic-vote experiment separated the earlier shallow-probe failure
from the stronger hypothesis that a real deep FLUX block already contains the
needed semantics. A registered Trainium/AOT graph executed joint blocks 0–17
at step 20, then reduced the deepest joint block 18 image-query/text-key map to
the same fixed 4x4-by-512 coverage tensor. This separate prefix graph is only a
discovery instrument. A positive result would still have required an inline
side reduction inside an already-required anchor forward; the 18-block prefix
graph itself was never a proposed serving implementation.

All 48 images, final latents, and complete trajectories were bit-identical to
the source candidate. The separate prefix call averaged 94 ms after warmup.
Against the eight terminal@29-actionable labels, the frozen weakest-token-
quartile score had AUC `0.253`; full recall triggered 47/48 requests and falsely
braked 39/40 negatives. Reversing the direction reached only `0.747`. Across
all seven prelisted reducers, the opened upper bound remained `0.747`, and even
the reducer with the lowest full-recall cost still falsely braked 31/40.

This closes large-space attention semantic voting for the current whole-
Transformer-output cache representation. The negative applies at block 0,
step 20, and the deepest joint block; it is not evidence that attention is
semantically meaningless. It says these request-level coverage reductions do
not map to the causal end-to-end VQA benefit required for a brake. Continuing
now requires a scope change: alter the cache representation or predictor, or
explicitly accept the measured x0-preview semantic-judge cost.

The workflow is executable with:

```bash
python scripts/flux_cache_brake_intervention.py run \
  --protocol benchmark/flux_cache/brake-intervention-pilot.json \
  --out-dir /path/to/brake-pilot \
  --allow-hardware \
  --foreground-ack "I am running the offline FLUX brake intervention pilot"

python scripts/evaluate_flux_cache_semantics.py \
  --quality-input /path/to/brake-pilot/quality-input.json \
  --out /path/to/brake-pilot/semantic-scores.json \
  --expected-images 37 \
  --metrics image_reward vqa_score

python scripts/flux_cache_brake_intervention.py analyze \
  --protocol benchmark/flux_cache/brake-intervention-pilot.json \
  --run-result /path/to/brake-pilot/brake-intervention-run.json \
  --semantic-scores /path/to/brake-pilot/semantic-scores.json \
  --out /path/to/brake-pilot/brake-intervention-analysis.json
```

Current machine-readable results:

- [`benchmark/flux_cache/cache-system-engineering-closure.json`](../benchmark/flux_cache/cache-system-engineering-closure.json)
- [`benchmark/flux_cache/profile-selection-replay.json`](../benchmark/flux_cache/profile-selection-replay.json)
- [`benchmark/flux_cache/online-signal-failure-enriched-closure.json`](../benchmark/flux_cache/online-signal-failure-enriched-closure.json)
- [`benchmark/flux_cache/brake-intervention-pilot-result.json`](../benchmark/flux_cache/brake-intervention-pilot-result.json)
- [`benchmark/flux_cache/terminal-brake-followup-result.json`](../benchmark/flux_cache/terminal-brake-followup-result.json)
- [`benchmark/flux_cache/warmup-vqa-terminal-router-pilot-result.json`](../benchmark/flux_cache/warmup-vqa-terminal-router-pilot-result.json)
- [`benchmark/flux_cache/warmup-vqa-router-holdout-result.json`](../benchmark/flux_cache/warmup-vqa-router-holdout-result.json)
- [`benchmark/flux_cache/terminal-brake-causal-label-result.json`](../benchmark/flux_cache/terminal-brake-causal-label-result.json)
- [`benchmark/flux_cache/terminal-brake-step21-futility-result.json`](../benchmark/flux_cache/terminal-brake-step21-futility-result.json)
- [`benchmark/flux_cache/spatial-anchor-error-probe-result.json`](../benchmark/flux_cache/spatial-anchor-error-probe-result.json)
- [`benchmark/flux_cache/semantic-coverage-probe-result.json`](../benchmark/flux_cache/semantic-coverage-probe-result.json)
- [`benchmark/flux_cache/component-residual-signal-result.json`](../benchmark/flux_cache/component-residual-signal-result.json)
- [`benchmark/flux_cache/deep-semantic-coverage-block18-step20-result.json`](../benchmark/flux_cache/deep-semantic-coverage-block18-step20-result.json)
