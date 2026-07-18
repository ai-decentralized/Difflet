# Video VAE Placement Rollout Plan

## Goal

Align serving placement with the CLI only where the lower layer is already
implemented and accepted. “Available” means that a serving artifact, resident
runtime path, shape contract, and Trn2 acceptance result all exist; a CLI-only
stage is not yet available to serving. Preserve every currently validated
serving path during the migration.

## Compatibility rule

Do not change existing defaults until the corresponding Neuron profile passes
co-load, memory, latency, and output-correctness gates.

The already validated host/hybrid paths are the MVP baseline and remain the
release path throughout this work. Adding placement metadata, compiling an
experimental VAE, or failing an experiment must not change their artifacts,
stage runners, startup command, request behavior, or recorded acceptance
results. Neuron VAE work is additive until a separate promotion decision is
made after acceptance.

| Model | Current validated default | Initial migration policy |
| --- | --- | --- |
| Wan 2.1 | Host VAE | Keep default; add experimental Neuron VAE profile |
| Wan 2.2 | Explicit experimental single-transformer profile; not an MVP default | Preserve experiments, but do not enable a Neuron VAE profile before dual-transformer qualification |
| HunyuanVideo 1.0 | Host CLIP + host VAE | Keep the MVP default; evaluate Neuron CLIP first, select the faster accepted CLIP baseline, then evaluate Neuron VAE against that frozen baseline |
| LTX-2 | Existing hybrid host path | Host-only: the current CLI and lower layer have no Neuron video-VAE decoder artifact/runtime |

## Profile contract

Retain the existing VAE host override at video serving startup/profile
resolution:

```text
--host-vae present -> host/CPU VAE decode
--host-vae omitted -> the model registry's validated default
```

No public `--vae-placement` selector is added. The model registry resolves its
current validated default (currently host VAE), while `--host-vae` always forces
the rollback host path. A promoted Neuron default is selected by omitting the
flag only after that adapter path passes acceptance. The choice is part of the immutable profile
identity. Between the two profiles, all non-VAE
artifacts and stage bindings must be identical: model revision, shape, dtype,
parallel topology, text/prompt-encoder placement, transformer artifacts, and
latent schema do not change. Only the VAE artifact, decoder runner binding, and
decoder placement may differ. An unavailable placement fails startup before the
worker becomes ready; the server never silently falls back. The existing default
remains the validated host placement until migration gates pass.

This contract applies to startup model/profile resolution, not to a live worker
switch. One serve process still owns one immutable profile. Placement is not a
public Videos API request field. A request cannot unload the host decoder, load
a Neuron decoder, or trigger compilation while that process is running.

## Hunyuan two-phase placement experiments

Hunyuan CLIP and VAE placement are two separate, serial experiments. They must
not change in the same experimental profile.

### Phase A: CLIP placement

- Add a startup/profile selector `--clip-placement host|neuron` for Hunyuan. It is
  not a per-request switch. Omission keeps the current validated host CLIP
  default until Phase A promotion is explicitly approved.
- Keep the currently validated host VAE, Llama encoder, denoiser, shape, dtype,
  topology, latent schema, request contract, and all non-CLIP artifacts fixed.
- Add an experimental Neuron CLIP artifact and prompt-encoder binding for the
  resident world-size/profile. Do not rewrite or remove the host CLIP runner.
- Compare host CLIP and Neuron CLIP using identical prompts, seeds, generation
  settings, warmup, and repeated runs.
- Both profiles must pass prompt-embedding correctness, end-to-end MP4
  correctness, startup/shutdown, sync/async API, and memory gates before speed is
  compared.
- Select the accepted profile with the lower repeatable end-to-end generation
  latency. Record CLIP-stage latency separately so the source of the improvement
  is visible. If the difference is within measurement noise, retain host CLIP as
  the lower-risk MVP baseline.
- Promotion changes only the validated Hunyuan CLIP baseline for newly started
  processes. The original host-CLIP/host-VAE profile remains available for
  rollback.

The Phase A profile-diff invariant permits only the CLIP artifact,
prompt-encoder runner binding, `clip_placement`, and CLIP placement to differ.
VAE placement and artifacts remain host and identical.

### Phase B: VAE placement

- Start only after Phase A has selected and recorded one validated CLIP
  baseline.
- Freeze the selected CLIP artifact, prompt-encoder binding, and placement in
  both VAE candidates.
- Compare host VAE and Neuron VAE with every non-VAE artifact, binding, topology,
  shape, dtype, and latent schema held identical.
- Select the faster accepted end-to-end profile subject to correctness and
  memory admission. Preserve the prior host-VAE profile for rollback.

The Phase B profile-diff invariant permits only the resolved `host_vae`, the VAE
artifact, decoder runner binding, and decoder placement
to differ.

## Implementation status

The additive local implementation now contains both Hunyuan experimental paths:

- host CLIP and host VAE remain the registry default and unchanged MVP fallback;
- Neuron CLIP uses a serving-specific `TP=1/world_size=4` artifact and runner;
- Neuron VAE reuses the existing segmented lower-layer decoder in a decoder-only
  `TP=1/world_size=4` application;
- resident loading initializes the full-world TP4 Llama/denoiser components
  before replicated TP1/W4 CLIP/VAE components;
- all four startup combinations resolve explicit pipeline bindings and artifact
  sets; host stages create no Neuron artifact;
- the shared W4 Llama and denoiser identities are identical across every
  placement combination, while CLIP and decoder identities change only when
  their Neuron placement is selected.

The 2026-07-18 Trn2 run completed both Hunyuan experiments for the fixed
TP4/CP1, 512x320x61 BF16 profile. Neuron CLIP and segmented Neuron VAE co-loaded
with the unchanged Llama/denoiser artifacts; three four-step requests averaged
58.402 seconds of inference time. Peak request HBM was 79.05 GiB, peak Neuron
host allocation was 12.75 GiB, and peak process PSS was 13.45 GiB. Repeated MP4
decode, async lifecycle, S3 publication/presign, and clean shutdown all passed.
The candidate is accepted for this fixed profile but is not promoted by this
document; host CLIP/VAE remains the registry default and rollback path.

The Phase B comparison held Neuron CLIP, model revision, TP4/CP1 topology,
BF16 dtype, 512x320x61 shape, 24 FPS, prompts, four denoise steps, and seeds
fixed. Only the VAE artifact, decoder binding, and decoder placement changed:

| Metric | Host VAE | Neuron VAE | Delta |
| --- | ---: | ---: | ---: |
| Mean HTTP wall time, 3 runs | 444.757 s | 58.408 s | 7.61x faster |
| Mean reported inference time | 444.752 s | 58.402 s | 86.9% lower |
| Peak request HBM | 67.40 GiB | 79.05 GiB | +11.65 GiB |
| Peak Neuron host allocation | 14.74 GiB | 12.75 GiB | -1.99 GiB |
| Peak process-tree PSS | 15.64 GiB | 13.45 GiB | -2.19 GiB |

The Neuron profile leaves approximately 16.95 GiB of the 96 GiB device budget.
Its segmented VAE compilation took approximately 33 minutes and produced a
606,251,540-byte artifact, which was archived outside the Git worktree under
its immutable content identity.

Wan 2.1 has now passed a real Neuron-VAE compile, resident co-load, sync/async
API, output decode, resource, and clean-shutdown run for its fixed 832x480x9
profile. Three controlled 20-step requests averaged 13.265 seconds versus the
prior 57.583-second host-VAE synchronous result. Resident HBM increased from
41.34 GiB to 61.99 GiB. The experimental profile is viable, but host VAE remains
the default until an explicit promotion review. Detailed evidence is recorded
in `09_wan21_trn2_serving_validation.md`.

LTX-2 is intentionally not part of this implementation. Its existing CLI and
serving path compile the TP4 dual-stream transformer on Neuron while text
encoding, connectors, video/audio VAE decode, and the vocoder remain on the
host. There is no repository-provided LTX-2 Neuron video-VAE application,
artifact contract, or resident decoder runner to reuse. Therefore LTX-2 keeps
one opaque hybrid `pipeline`, requires the host decode profile, and rejects an
explicit Neuron VAE placement at startup. Adding a new LTX-2 lower-layer backend
would be a separate model-port project, not a serving-placement rollout.

Recommended remote sequence:

1. Re-run the omitted/default host-host profile as the rollback control.
2. Start `--clip-placement neuron --host-vae`; compile the new CLIP
   artifact, run startup smoke, and execute sync plus async API checks.
3. Select and record one accepted CLIP baseline.
4. In the acceptance build, resolve `host_vae=false` internally while keeping
   the selected CLIP placement; compile the decoder artifact and repeat
   correctness, API, memory, and latency checks. Do not expose a second public
   placement selector solely for this experiment.
5. Re-run the corresponding host-VAE candidate without changing any non-VAE
   startup field, then compare the recorded results.

## Incremental implementation

1. Freeze the current Wan, HunyuanVideo, and LTX-2 defaults and add placement
   metadata without changing their runtime behavior.
2. Implement Hunyuan Neuron CLIP as Phase A while holding host VAE and every
   other stage fixed. Measure both CLIP placements and select the faster accepted
   baseline without removing the original MVP profile.
3. Implement Hunyuan Neuron VAE as Phase B only after the CLIP baseline is
   selected. Hold the selected CLIP and every other non-VAE stage fixed.
4. Implement Wan 2.1 Neuron VAE using the CLI VAE stage as the lower-layer
   reference. Compile a serving-specific artifact for the resident world size;
   do not reuse an incompatible CLI TP=1 artifact. Add a serving VAE artifact,
   resident binding, decoder runner, payload validation, and an experimental
   profile.
5. Validate Wan with both placements: startup, async generation, sync
   generation, `/content`, MP4 decode, HBM steady/peak, Neuron host runtime
   memory, host RSS/PSS, and decode latency.
6. Record LTX-2 as host-only for this rollout. Do not add an experimental
   serving-only decoder when the CLI and lower layer provide no Neuron VAE
   application to reuse.
7. Only after a model passes all gates may its validated-default registry pointer
   move from the host profile to the Neuron profile. Promotion is a deployment
   change for newly started processes, not an in-process profile mutation.

## Promotion and rollback

- Each model registry entry keeps one explicit validated-default VAE placement.
  Experimental profiles are exercised by the acceptance harness and never
  update this pointer during compile, startup, or request handling.
- Promotion changes that single registry default only after all acceptance
  evidence is recorded and reviewed. A process resolves the pointer once at
  startup and keeps the resulting profile immutable.
- Rollback restores the prior validated-default pointer and starts replacement
  processes with the prior profile. Existing processes and in-flight requests
  continue on the profile they resolved at startup; they are drained or stopped
  through the normal serving lifecycle.
- A failed compile, load, smoke, generation, or memory gate cannot change the
  default pointer. Omitted `host_vae` therefore continues to resolve to the MVP
  host path after any failed experiment.

## Acceptance gates

- Resident worker starts and shuts down cleanly.
- Async and sync APIs both generate valid MP4 output.
- `/content`, DELETE, TTL cleanup, and async S3 publication remain correct.
- HBM steady and peak remain below the model's admission threshold.
- Host RSS/PSS and Neuron host runtime memory remain within the Trn2 budget;
  no swap or OOM occurs.
- Neuron and host VAE outputs meet the agreed decode/quality correctness gate.
- Hunyuan Phase A compares host and Neuron CLIP with VAE fixed on host; prompt
  embeddings, CLIP-stage latency, end-to-end latency, HBM, and host memory are
  recorded for both candidates.
- Hunyuan Phase B starts from exactly one accepted CLIP baseline and proves that
  CLIP artifacts, bindings, and placement are identical across both VAE
  candidates.
- Profile switching is explicit and does not silently change placement.
- A profile identity diff test proves that all non-VAE artifacts and stage
  bindings are identical. Only `host_vae`, the VAE artifact identity, decoder
  runner binding, and decoder placement may differ; topology, latent schema,
  transformer artifacts, and text/prompt-encoder artifacts remain identical.
- Startup rejects a VAE artifact with the wrong resident world size and rejects
  a placement not implemented by that model adapter.
- A failed experiment leaves the validated-default registry pointer unchanged;
  an explicit promotion can be atomically rolled back to the prior pointer for
  newly started processes without changing the public API.

## Non-goals

- Do not rewrite the model implementations in `difflet/models`.
- Do not change the shared video FIFO or API contract.
- Do not remove the validated host placement while Neuron placement is still
  experimental.
- Do not combine a Hunyuan CLIP placement change and a VAE placement change in
  one experiment or one profile comparison.
- Do not claim a CLI-only stage is available to serving before its acceptance
  gate passes.
