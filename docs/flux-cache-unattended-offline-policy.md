# Unattended FLUX cache-profile execution policy

Status: implemented for baseline calibration and independent profile
confirmation. Exploratory screening collectors are archived at commit
`de2a415`.

## One-time scoped hardware authorization

An offline profile run may bind one canonical execution-policy JSON. The policy
authorizes a finite set of stages and is reused without per-stage acknowledgement.
It freezes:

- exact model id and revision;
- hardware backend, product, and tensor-parallel degree;
- step count, dtype, guidance scale, and allowed resolutions;
- allowed profile stages and a request-count ceiling for each stage;
- one absolute artifact root.

The checked-in 1024x1024 policy is
`benchmark/flux_cache/offline-profile-square-1024-execution-policy.json`.
Its allowed stages are baseline calibration, cost calibration, trajectory
collection, candidate screening, and confirmation. Every stage validates its
complete request before loading the model. Any scope mismatch fails closed.

Create a policy once:

```bash
python scripts/flux_cache_execution_policy.py \
  --policy-id flux-cache-offline-profile-square-1024 \
  --created-at 2026-08-06T00:00:00Z \
  --model-id black-forest-labs/FLUX.1-dev \
  --model-revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
  --resolution 1024x1024 \
  --stage-limit baseline_calibration=256 \
  --stage-limit cost_calibration=256 \
  --stage-limit trajectory_collection=192 \
  --stage-limit candidate_screen=192 \
  --stage-limit confirmation=128 \
  --output-root /home/ubuntu/difflet-artifacts \
  --authorize \
  --out /path/to/execution-policy.json
```

Subsequent hardware stages enter through the scoped wrapper. Independent A/B
confirmation accepts exactly one frozen candidate:

```bash
python scripts/collect_flux_cache_authorized.py ab \
  --execution-policy /path/to/execution-policy.json \
  --execution-stage confirmation \
  --phased-candidate /path/to/frozen-candidate.json \
  --out-dir /home/ubuntu/difflet-artifacts/confirmation \
  --model-revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
  --prompt-suite benchmark/flux_cache/prompt-suite-v1.json \
  --prompt-split holdout \
  --seed 0
```

For registered no-cache calibration:

```bash
python scripts/collect_flux_cache_authorized.py baseline \
  --execution-policy /path/to/execution-policy.json \
  --execution-stage baseline_calibration \
  --registration /path/to/multires-registration.json \
  --bucket-id square-1024 \
  --out-dir /home/ubuntu/difflet-artifacts/baseline-square-1024
```

For a fresh label-free schedule calibration, collect the 48 shared full-DiT
trajectories and two frozen phased-static cost points in one policy-bound run:

```bash
python scripts/collect_flux_cache_authorized.py calibration \
  --execution-policy benchmark/flux_cache/offline-profile-square-1024-execution-policy.json \
  --execution-stage trajectory_collection \
  --phased-candidate benchmark/flux_cache/derived-static-a12-o1-index.json \
  --phased-candidate benchmark/flux_cache/derived-static-a13-o1-index.json \
  --out-dir /home/ubuntu/difflet-artifacts/flux-cache-qualified-calibration-20260809 \
  --model-revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
  --prompt-suite benchmark/flux_cache/phase-schedule-development-prompt-suite.json \
  --prompt-split phase_schedule_horizon_development \
  --seed 2
```

This command is calibration, not a candidate sweep: it accepts exactly two
already-frozen static profiles with distinct real-step counts. Candidate images
are retained only to make the cost run auditable; semantic labels are not read.
The baseline trajectory paths and hashes are recorded in `quality-input-v2.json`,
and the paired hardware timings are recorded in `speedup-candidates-v1.json`.

The wrapper does not accept or request another user acknowledgement. It checks
the request count, exact generation identity, device, stage, output path, and
single frozen candidate before loading the model. Each successful output
directory contains `execution-authorization.json`, binding the policy hash,
stage, actual request count, request ceiling, and output directory.

The policy is an auditable workflow guardrail, not a cryptographic user signature.
Launching the parent workflow with an authorized policy is the single user
authorization event. A production scheduler may additionally control who can
create or replace policy files.

## Automatic fail-closed image adjudication

The unattended path does not emit an ambiguous state and never waits for a human
review. For each registered resolution, it takes the observed maximum absolute
no-cache seed-to-seed difference from the frozen multiresolution quality contract:

```text
harm_metric = baseline_score_metric - candidate_score_metric

harm_metric <= natural_range_max_metric  -> automatic_pass
harm_metric >  natural_range_max_metric  -> automatic_reject
```

ImageReward and VQAScore remain independent vetoes. A candidate passes only when
every evaluated request passes both metrics. Manual inspection may be used for
diagnosis after the fact, but manual override is forbidden.

The evaluator also requires the scored comparison set to match the quality
manifest exactly. Missing, duplicate, or substituted rows are errors. It writes
the audit JSON and exits with status `1` when any candidate is rejected; invalid
or incomplete evidence exits with status `2`. Therefore a normal shell or CI
pipeline stops automatically without an image-review pause.

Run the gate with:

```bash
python scripts/flux_cache_natural_range_gate.py \
  --contract benchmark/flux_cache/multires-quality-contract-v1.json \
  --semantic-report /path/to/semantic-scores.json \
  --bucket-id square-1024 \
  --out /path/to/natural-range-evaluation.json
```

For `square-1024`, the frozen limits are:

```text
ImageReward harm <= 0.9721190482378006
VQAScore harm    <= 0.26171875
```

Historical development-screen labels are not serving evidence. The current
workflow applies this rule once to the independently collected confirmation
pair and exports no profile if any comparison exceeds either bound.

## One-command profile build

`scripts/build_flux_cache_profile.py` closes the orchestration boundary. Its
frozen build spec binds the cached calibration evidence, the natural-range
contract, one new confirmation split, the execution policy, both Python
runtimes, scorer caches, output directory, and a hash manifest of the
first-party implementation files used by the build.

The checked-in revision-1 square-1024 build spec and its qualification remain
historical evidence for commit `e16a718`. The modular implementation rejects
that old spec rather than silently inheriting its qualification. A future run
must first register and commit a new revision-2 build spec with a new
confirmation split.

The user-facing execution is one command:

```bash
/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/python \
  scripts/build_flux_cache_profile.py build \
  --spec /path/to/prospective-build-spec.json
```

The command verifies that every hash-bound implementation file, build spec,
prompt suite, and protocol input is tracked and unchanged from `HEAD` before
model loading. Unrelated working-tree changes do not block confirmation. Commit
the frozen workflow inputs before launching it.

Internally it performs only two logical operations:

```text
deterministically derive one combined profile
-> independently confirm quality and measured speed
```

Calibration is a hash-bound cache lookup, not a user-visible stage. The builder
does not run a development candidate screen or select among candidates. It
derives one static-plus-brake profile from the registered speed target, verifies
it on exactly one prospectively frozen prompt split, and atomically exports
`cache-profile.json` only when both conditions hold:

- every confirmation comparison passes both natural-range metric vetoes;
- measured aggregate hardware speedup meets the registered target.

On success it also writes `profile-qualification.json`. On either quality or
speed failure it writes `rejection-report.json`, returns exit status `1`, and
does not create `cache-profile.json`. Invalid scope, evidence, or execution
returns status `2`. No human review or intermediate candidate choice exists.

Revision-6 build specs also freeze an `evidence_role`. The
`hardware_ladder_smoke` role executes the same derived frontier, hardware
collector, baseline reuse, append-only semantic scoring, and ascending
first-pass decision path, but it can never export a deployable profile or a
serving qualification. Its only terminal artifact is
`hardware-ladder-smoke-report.json`, with `deployable_profile_written=false`.
Revision-5 specs retain their historical implicit `serving_qualification`
role.
