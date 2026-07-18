# T2V Serving Implementation and Verification Plan

## Current status

The local implementation milestone is endpoint-complete. It includes the six
Videos API routes, shared sync/async scheduling, process-local job state,
file-backed artifact ownership, file-backed worker output, and provisional
adapters for LTX-2, Wan 2.1, and HunyuanVideo 1.0. It does not change the
existing lower-layer model computation.

The video-local queue milestone is complete, but the stronger cross-endpoint
admission requirement is not: before Chat Completions and Videos can coexist on
one resident engine, all generation handlers must use one lifespan-owned global
FIFO, capacity budget, ticket order, and recovery fence. The Round 1 design
review also identified hardening that is now implemented locally: preload every serving
tokenizer before readiness, run per-request tokenization on a bounded CPU
executor, enforce text-only multipart limits before admission, expire terminal
jobs after 25 hours, and narrow public DELETE to queued work only.

LTX-2, Wan 2.1, and HunyuanVideo 1.0 now have hardware-qualified fixed-profile
resident results. Their qualification does not generalize to other shapes,
topologies, or checkpoints, and the remaining timeout/restart and soak items
below still apply. Wan 2.2 and HunyuanVideo 1.5 remain outside MVP HTTP Serving
for different reasons. Wan 2.2 is
listed in the README and its single-transformer offline CLI path has run on a
real Trn2, but current wiring disables `transformer_2`, so dual-transformer
reference correctness and resident memory are not qualified. HunyuanVideo 1.5
is not the HunyuanVideo 1.0 row shown in the README; its offline
compile/generate methods remain scaffolds.

The existing benchmark closes the basic fixed-profile offline generation
question for LTX-2, Wan 2.1, and HunyuanVideo 1.0. Later resident validation
separately closed the fixed-profile co-load, memory, API, and MP4 gates; the
offline benchmark alone still proves none of those properties. See the
[Trn2 benchmark evidence audit](07_trn2_benchmark_evidence.md) and the model
validation reports linked from it.

## Implemented milestones

### Phase 0: protocol and immutable profile contract

- Added video model metadata, fixed `num_frames`, MP4 MIME/format, and
  host/Neuron/hybrid runtime placement without changing image positional
  contracts.
- Added multipart request normalization, model-specific fixed-profile
  validation, and stable Videos API response/error types.
- The target P0 request contract is text-only: zero file parts, at most 32 form
  fields, at most 256 KiB per text part, and at most 1 MiB for the complete body,
  including chunked requests. These bounds are enforced before validation and
  generation admission.
- Added file-backed generated output for parent-owned staging paths.
- Kept video routes gated to video-model servers; image Chat Completions remains
  unchanged.

### Phase 1: process-local job and artifact ownership

- Added an `InMemoryVideoJobRepository` with legal-transition compare-and-swap
  and stable listing/cursors for the current serve-process lifetime.
- Added a confined local MP4 store with unguessable `.part.mp4` paths,
  regular-file/inode checks, atomic commit, streaming leases, and full-root
  purge support.
- Added a nonblocking process/host root lease so two services cannot own or
  sweep one media root at the same time.
- Clean shutdown clears jobs and purges staging/final MP4 files. After a crash,
  the next same-profile lease holder purges residual managed media before
  accepting requests; old IDs are never restored.
- Made MP4 encode/validation failure terminal; serving never substitutes a
  tensor file.
- The target retention policy expires terminal async jobs 25 hours after their
  terminal transition. A periodic sweeper removes metadata and media; disk
  pressure emits warning/error logs, and admission returns
  `507 video_storage_full` when the maximum artifact plus safety reserve cannot
  be guaranteed. This policy is implemented locally.

### Phase 2: Videos-local shared scheduling and recovery

- Added one live-capacity-bounded FIFO `VideoGenerationService` shared by
  process-local async jobs and ephemeral sync requests. Queued cancellation
  physically removes the entry under the admission-state lock.
- Added total deadline, cancellation tombstone, completion-publication, worker
  recovery, startup/shutdown, client-disconnect, and cleanup fencing. The
  public DELETE removes queued work, returns `409 video_in_progress` after
  dispatch, and never preempts a healthy running generation.
- Async publication commits the artifact before the job becomes `completed`.
- Sync streaming returns raw `video/mp4`, creates no job record, and cleans the
  artifact after response completion or abort.

### Phase 2b: local hardening complete; cross-endpoint generalization pending

The bounded validation, multipart, queued-delete, TTL, job-cap, storage-ledger,
and sweeper bullets below are implemented. The remaining work in this phase is
the first group: one common owner and mixed-route ticket order if Chat and
Videos are ever registered together.

- Replace/generalize the video-only admission owner with one
  `GenerationAdmissionService` created by the FastAPI lifespan.
- Route every handler that can reach the resident engine through it, including
  Chat Completions, Videos sync, Videos async, and future generation routes.
- Assign a monotonic ticket and reserve capacity at one post-validation
  linearization point so async job creation cannot reorder requests.
- Keep both live work and the physical queue bounded; queued cancellation must
  remove/compact its entry or retain the slot until dequeue.
- Keep non-generation GET routes outside the queue and send DELETE directly to
  the cancellation control plane.
- Preload all image and video tokenizers before readiness. Run request
  tokenization/validation on a bounded CPU executor so invalid requests neither
  block the event loop nor consume a generation ticket.
- Make validation admission physically bounded, with P0 defaults of four
  synchronous CPU validation workers, 32 waiting entries, and a 30-second
  validation sub-deadline; generation concurrency remains one. A full domain
  returns `429 validation_capacity_exhausted`; timeout returns
  `504 validation_timeout`; the total request deadline includes validation.
  Timeout/disconnect must not release a started validation slot before its
  future ends; unstarted work releases only after atomic physical removal.
  The pool is one dedicated lifespan-owned thread executor in the FastAPI
  parent, not additional Uvicorn/model workers.
- Enforce the exact multipart/body limits from Phase 0 while streaming the body;
  reject file fields as `feature_not_supported` and size overflow as stable
  `413 request_too_large`.
- Make DELETE remove queued entries immediately, return
  `409 video_in_progress` for dispatched work, and delete terminal metadata and
  media. Keep hard deadline/shutdown/worker-failure termination as an internal
  recovery mechanism rather than a public cancellation contract.
- Put dispatcher claim and queued DELETE under the same admission-state lock:
  dequeue + `queued -> in_progress` is one atomic claim, and DELETE either wins
  before it or returns `409`. Reuse that boundary for sync disconnect.
- Add a configurable 4,096-record async-job cap with atomic slot reservation;
  after opportunistic expiry, reject exhaustion as
  `429 video_retention_full`.
- Add one cumulative sync/async storage ledger: reserve `max_artifact_bytes` per
  active request plus one global `max(1 GiB, max_artifact_bytes)` safety margin,
  convert to actual bytes at commit, and release on every fenced cleanup path.
- Serialize publication, content-lease acquisition, terminal DELETE, and TTL
  sweeping with one artifact-lifecycle lock. Delete artifact first and metadata
  second; retain accounting/job state for retry if unlink fails.
- Disable an independent engine waiting queue and prove that only the global
  dispatcher invokes `engine.generate`, with at most one running generation.
- Add mixed Chat/Video FIFO, shared `429`, unified deadline, queued deletion,
  running-delete rejection, recovery, shutdown, tokenizer-executor saturation,
  multipart-limit, TTL, disk-pressure, and repeated enqueue/delete tests.
  Include validation saturation/timeout, record-cap failure storms, cumulative
  storage overcommit, reservation-release, lease-vs-sweep, and atomic
  dispatch-vs-DELETE races.

### Phase 3: all six HTTP routes

- `POST /v1/videos`
- `POST /v1/videos/sync`
- `GET /v1/videos/{video_id}`
- `GET /v1/videos`
- `GET /v1/videos/{video_id}/content`
- `DELETE /v1/videos/{video_id}`

Fake-engine tests cover normal lifecycle, validation, pagination, content
gating, cancellation/delete races, deadline and worker-recovery behavior,
process-restart reset semantics, startup purge of crash residue, streaming
cleanup, repository/storage safety, and shared-root ownership. The
repository-wide final local verification pass is recorded in `tasks/todo.md`.
These tests describe the current implementation, including the local Phase 2b
hardening. Only mixed Chat/Video admission tests remain with the cross-endpoint
generalization follow-up.

### Phase 4: model adapters and fixed-profile qualification

| Model | Implemented fixed-profile plan | Public posture |
| --- | --- | --- |
| LTX-2 | One TP4 hybrid `pipeline`; host text/connectors/decode, Neuron DiT; silent MP4 | Fixed-profile resident API and media validation passed; timeout/restart and long soak remain |
| Wan 2.1 | TP4 Neuron prompt encoder + denoiser; Neuron VAE default with `--host-vae` rollback | Fixed-profile resident API, media, memory, and clean-shutdown validation passed |
| HunyuanVideo 1.0 | Host CLIP default, resident W4 Llama + denoiser; Neuron VAE default with `--host-vae` rollback | Fixed-profile resident API, S3, media, memory, and image-regression validation passed |
| Wan 2.2 | Real Trn2 single-transformer path exists; no serving adapter/allowlist entry | Blocked on enabling/validating the second transformer and then the resident hardware gate |
| HunyuanVideo 1.5 | Not in the README supported-model table; no serving adapter/allowlist entry | Blocked on offline prompt-conditioning/compile/generation |

The capacity assumption is deliberately narrow: one serve process owns one
model and one immutable compiled profile. It does not assume that several
models can share the same four logical NeuronCores.

## Local gate status

- [x] Run the complete Serving unit suite and relevant CLI regressions.
- [x] Run formatting, lint, targeted typing, `compileall`, and `git diff --check`.
- [ ] Resolve the Round 1 material design findings in implementation and tests.
- [x] Preserve a clean validation packet with exact model/profile/artifact identity
  before moving to the target host.

## Trainium acceptance packet per model/profile

Start with the exact revision and fixed profile recorded by
`benchmark/trn2` so the first resident comparison changes the execution model,
not the inputs. Use the intended visible core set. For each fixed profile:

1. Record artifact hashes, compiler/runtime versions, topology, dtype, shape,
   frame count, fps, steps, and all host-placement flags.
2. Record host `MemTotal`/`MemAvailable`, swap, cgroup memory, process RSS/PSS,
   and per-core HBM before load, after load, after readiness smoke, at generation
   peak, after completion, and after a second request.
   Also record effective cgroup CPU count, the four-thread validation pool,
   tokenizer-internal parallelism state, host-VAE intra-op thread budget, and
   validation/control/decode latency while validation overlaps host decode.
3. Start the resident worker, pass real readiness inference, and verify every
   required compiled payload is used.
4. Exercise async create/poll/list/content/delete and sync raw-video generation.
5. Validate the MP4 container/codec, dimensions, frame count, fps, duration, and
   silent/audio declaration with PyAV or ffprobe.
6. Exercise queued DELETE, `409 video_in_progress` after dispatch, deadline expiration, worker
   recovery, process-restart reset (`404` for old IDs and an empty list),
   crash-residue startup purge, client disconnect, and a successful second
   generation after each worker-recovery path.
7. Verify the 25-hour terminal TTL with queued/running `expires_at=null`, an
   accelerated test clock, periodic sweeping, the 4,096-record cap,
   warning/error disk-pressure logs, cumulative sync/async reservation, and the
   hard storage reserve.
8. Run repeated requests long enough to expose RSS/HBM growth and profile reuse
   failures, then shut down and verify that no worker, in-memory job, staging
   file, or final artifact remains.
9. Mix Chat Completions and Videos generation whenever both are registered and
   prove strict global ticket order, shared capacity/deadlines, physical queue
   bounds after queued deletion, and maximum engine concurrency of one. Prove
   tokenizer validation completes before ticket assignment and cannot exhaust
   an unbounded executor. Saturation must return the stable validation 429 and
   validation time must count against the total deadline. Repeated timeout and
   disconnect must prove actual CPU validation running <= 4 and waiting <= 32,
   while generation running remains <= 1.

The benchmark's existing compile/generate result is the lower-layer baseline
for the first three candidates, not a replacement for steps 2-8.

AWS documents 128 GB host memory and 96 GB accelerator HBM for
`trn2.3xlarge`; prior observation is approximately 124 GB Linux-visible host
memory. Neither number proves a T2V profile fits. Peak RSS/PSS/HBM and startup
transients must be captured from the actual serving artifacts. Host placement is
explicit model behavior, not a generic CPU-offload mechanism.

## Explicit non-goals

- Dynamic resolution or frame-count compilation/routing in one server.
- Multiple resident models sharing the same four cores.
- A generic CPU-offload/spill system.
- Job persistence across serve-process restarts, cross-host job coordination,
  or remote artifact storage in this milestone. The 25-hour TTL is process-local
  retention, not persistence.
- I2V, V2V, S2V, LoRA, interpolation, or generated audio.
- Wan 2.2 before correct dual-transformer execution.
- HunyuanVideo 1.5 before its offline end-to-end path works.

### Public video response contract

The public video job response follows the OpenAI Video API object: `id`,
`object`, `model`, `status`, `progress`, timestamps, `prompt`, `size`,
`seconds`, `quality`, `error`, `expires_at`, and remix metadata. Resolved
dimensions, frame rate, artifact paths, file size, stage timings, peak memory,
and other serving diagnostics remain internal job metadata and logs; they are
not serialized in the default user response. The local `/v1/videos/sync`
extension still returns raw MP4 bytes.
