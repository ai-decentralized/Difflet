# Unattended FLUX cache-profile execution policy

Status: implemented for the hardware collectors used by offline profile
calibration and screening. Legacy foreground acknowledgement flags remain only
for replaying older registrations.

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

Subsequent hardware stages enter through the scoped compatibility wrapper. For
an A/B candidate screen:

```bash
python scripts/collect_flux_cache_authorized.py ab \
  --execution-policy /path/to/execution-policy.json \
  --execution-stage candidate_screen \
  --phased-candidate /path/to/frozen-candidate.json \
  --out-dir /home/ubuntu/difflet-artifacts/candidate-screen \
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

The wrapper does not accept or request another user acknowledgement. It checks
the request count, exact generation identity, device, stage, and output path
before loading the model. It then injects the old collector acknowledgement
internally, so evidence-bound legacy collectors remain byte-identical. Each
successful output directory contains `execution-authorization.json`, binding
the policy hash, stage, actual request count, request ceiling, and output
directory.

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

Replaying the 32-group derived-schedule screen under this rule automatically
passes legacy brake-only and static-a13, and automatically rejects static-a12,
static-plus-brake-a12, and static-plus-brake-a13. The three rejections are all
the `p013-s3` ImageReward comparison. No review state is produced.

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

The command performs a clean-worktree preflight before model loading, because
the confirmation protocol rejects mutable source evidence. Commit the frozen
builder, spec, and prompt suite before launching it.

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
