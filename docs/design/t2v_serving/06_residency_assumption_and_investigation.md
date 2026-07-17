# T2V Residency Assumption and Investigation Register

## Decision record

- Date: 2026-07-16
- Owner: Difflet T2V serving implementation
- Scope: one fixed model/profile per `trn2.3xlarge` serving process
- Constraint: reuse the current model, Trainium backend, and CLI/model primitives;
  changes are limited to Serving/API/adapters plus narrow CLI/dependency wiring.

### D6: implement against a provisional single-model residency assumption

For implementation planning, treat each supported T2V model profile as capable
of residing in one worker on the four-core target. Memory uncertainty did not
block implementation of the shared Videos API, job/storage layer, file-backed
output contract, or model adapters.

This is a **capacity assumption**, not a release claim. It does not waive model
correctness, compatible Neuron process-world requirements, valid MP4 output, or
the hardware acceptance suite. One instance still serves only one model and one
compiled profile; this decision does not assume that multiple models can share
the same four logical NeuronCores.

## Current memory facts

- [AWS lists `trn2.3xlarge`](https://aws.amazon.com/ec2/instance-types/trn2/)
  with 128 GB host memory and 96 GB accelerator memory.
- The repository's latest benchmark harness records about 124 GB Linux-visible
  host RAM and about 100 GB available for a warm page cache.
- The target host could not be re-read on 2026-07-15 because SSH timed out during
  banner exchange. A fresh `free -h`/`/proc/meminfo` capture remains required.
- The T2V benchmark JSON files currently record `peak_device_mem_gb: null`; no
  retained T2V peak RSS/PSS measurement was found.
- The LTX-2 86 GB checkpoint/page-cache note is startup-cache evidence, not an
  out-of-memory result. Page cache is reclaimable and must not be treated as
  process RSS.

## Real Trn2 offline benchmark interpretation

The repository does contain successful real `trn2.3xlarge` T2V results. This
corrects any earlier wording that could be read as "the models have not run on
Trn2":

- LTX-2 completed TP4/CP1 generation at 480x704x49.
- Wan 2.1 completed TP4/CP1 generation at 480x832x9.
- HunyuanVideo 1.0 completed its staged TP4/CP1 path at 320x512x61.
- Wan 2.2 completed the current single-transformer path at 480x832x9; the run
  did not enable or validate the second expert.
- HunyuanVideo 1.5 has no successful compile/generate benchmark and remains
  pending.

These are finite-tensor offline CLI results. Each warm sample starts a separate
`difflet generate` process and reloads weights with a warm OS page cache. The
successful result records point to `.pt` outputs, not retained media-validation
records. They therefore support the provisional residency assumption but do not
measure resident co-load, same-worker reuse, MP4 validity, or the six HTTP
flows. The full evidence audit is in
[Trn2 benchmark evidence audit](07_trn2_benchmark_evidence.md).

## Evidence and implementation posture

| Model | Existing evidence | Implementation posture | Remaining non-memory blocker |
| --- | --- | --- | --- |
| LTX-2 | Real Trn2 TP4 480x704x49 offline generation, finite tensor; separate-process warm samples | Hybrid resident pipeline adapter implemented | Measure same-worker repeated-request RSS/HBM and validate silent MP4 |
| Wan 2.1 | Real Trn2 TP4 480x832x9 offline staged generation, finite tensor; UMT5+DiT share the transformer stage | Resident W4 prompt/DiT plus host decode adapter implemented | Validate long-lived reuse and MP4 media contract |
| Wan 2.2 | Real Trn2 single-transformer 480x832x9 path, but historical cache reused the 2.1 NEFF and CLI disables `transformer_2` | Assume capacity is provisionally available, but do not advertise full Wan 2.2 Serving yet | Enable/validate dual-expert switching and reference correctness, then measure the larger resident set |
| HunyuanVideo 1.0 | Real Trn2 TP4 320x512x61 staged offline generation, finite tensor | Host CLIP/VAE plus resident W4 Llama/DiT adapter implemented | Full prompt-to-MP4 co-load/correctness test |
| HunyuanVideo 1.5 | Benchmark pending with no compile/e2e/output; offline CLI compile/generate is unimplemented | Keep protocol/registry capability explicit, but do not advertise a working adapter until the existing prompt-conditioning path exists | Missing Qwen2.5-VL/ByT5/image-semantic end-to-end orchestration |

## Existing host placement that may be reused

- LTX-2: text encoder, connectors, video/audio VAE, and vocoder are already
  host-side; only the transformer is a Neuron load.
- Wan: `--host-vae` provides CPU VAE decode; UMT5 and DiT remain on Neuron.
- HunyuanVideo 1.0: the pipeline already supports lazy host VAE decode. The
  serving adapter will use host CLIP/host VAE so resident Neuron components share
  the W4 world.
- Experimental segmented/block streaming exists for LTX-2 and HunyuanVideo 1.5,
  but it is not the initial serving baseline.

Host DRAM is not a transparent extension of Trainium HBM. Every host placement
or file handoff must be explicit; the implementation must not introduce an
imaginary generic CPU-offload switch.

Host VAE decode is CPU work in the resident model-worker process; it is not one
of the four parent-process tokenizer threads. The domains have independent
slots but share host CPU/DRAM bandwidth. Disable tokenizer-internal fan-out,
configure host-VAE native threads independently, and measure their overlap
before keeping the four-thread validation default.

## Frozen Serving decisions

1. The current six `/v1/videos` routes are registered only for video-model
   servers, while image servers retain `/v1/chat/completions`. This mutual
   exclusion prevents a present cross-route race but is not the long-term
   admission guarantee.
2. One lifespan-owned global generation admission service must own the single
   ticket FIFO and capacity budget for every endpoint that can reach the same
   resident engine, including Chat Completions and both Videos create styles.
   The engine has no independent waiting queue and handlers cannot call it
   directly. The current `VideoGenerationService` implements only the
   Videos-local subset, so cross-endpoint generalization remains pending.
3. Public job states are `queued`, `in_progress`, `completed`, and `failed`.
   DELETE removes queued and terminal resources; after dispatch it returns
   `409 video_in_progress`. `deleted` is not a retrievable state.
4. `InMemoryVideoJobRepository` owns process-local job metadata and transition
   compare-and-swap. A local, confined filesystem store owns MP4 staging and
   committed artifacts only for the current serve-process lifetime.
5. The parent creates an unguessable `staging/*.part.mp4` target. The worker
   returns a small file-backed output descriptor; the parent revalidates the
   exact path, file type, byte size, and media metadata before atomic publish.
6. MP4 encoding failure is terminal. Serving must not copy the CLI behavior that
   silently falls back to a `.pt` tensor.
7. Queued DELETE guarantees that the ticket never executes. Once the dispatcher
   takes the ticket, public DELETE returns `409 video_in_progress` and sends no
   cancellation signal. Hard timeout, shutdown, and worker failure may still
   terminate/restart the worker behind an internal recovery fence; that path is
   not part of the public cancellation contract.
   Dispatcher claim and DELETE share one admission-state lock: dequeue,
   ownership claim, and `queued -> in_progress` are atomic with respect to
   queued removal. Sync disconnect uses the same boundary.
8. Artifact commit happens before the job transitions to `completed`.
9. Chat, sync Video, and async Video requests share admission whenever they
   target one engine. Sync creates no job record and deletes its temporary
   artifact after the response closes. A disconnect after dispatch does not
   cancel healthy sync generation; completion discards and cleans its output.
   Model/health/job reads bypass generation admission; DELETE uses the queued
   deletion control plane.
10. Host-only stages are represented honestly in the runtime plan with host
    placement and no fabricated Neuron topology/artifact identity.
11. The initial hardware-validation allowlist is LTX-2, Wan 2.1, and
    HunyuanVideo 1.0. Wan 2.2 remains absent from Serving despite README/offline
    artifact support until its disabled second-transformer wiring and reference
    gate are closed. HunyuanVideo 1.5 remains absent because offline
    compile/generate is still a scaffold.
12. One nonblocking process/host root lease protects the media root. A second
    service fails startup before it can sweep another service's
    staging/artifacts.
13. Clean shutdown clears all jobs and purges staging/final MP4 files. A crash
    may leave media residue, which the next same-profile lease holder purges
    before accepting work. Jobs are never recovered: after restart, old IDs
    return `404` and list is empty.
14. P0 is text-to-video only: zero file parts, at most 32 fields, at most
    256 KiB per text part, and at most 1 MiB total body including chunked input.
    File fields return `feature_not_supported`; limit overflow returns stable
    `413 request_too_large` before admission.
15. All image and video tokenizers preload on CPU before readiness. Per-request
    tokenization/model validation runs on a bounded CPU executor and completes
    before ticket assignment. P0 defaults to four synchronous CPU validation
    workers/32 waiting entries and a 30-second sub-deadline, while generation
    remains one-at-a-time; full/timeout responses are stable
    `429 validation_capacity_exhausted`/`504 validation_timeout`, and validation
    counts against the total request deadline. Timeout/disconnect retains a
    started validation slot until its future ends; unstarted work releases only
    after successful atomic queue removal.
    They are threads in one dedicated parent-process executor, not model workers
    or Uvicorn processes.
16. Terminal jobs expire after 25 hours within the current process. A periodic
    sweeper removes metadata/media, disk pressure emits structured warning/error
    logs, and admission returns `507 video_storage_full` when the maximum
    artifact plus safety reserve cannot be guaranteed.
17. Queued/in-progress `expires_at` is null. Async admission atomically reserves
    one of 4,096 configurable job-record slots; exhaustion after opportunistic
    sweep returns `429 video_retention_full`. The slot remains through failure
    and is released by queued/terminal DELETE, TTL, or shutdown.
18. One process-wide ledger reserves `max_artifact_bytes` for every active
    sync/async request plus one global `max(1 GiB, max_artifact_bytes)` margin.
    Publication, content-lease acquisition, terminal deletion, and sweeping
    share one lifecycle lock. Deletion is artifact-first/metadata-second;
    unlink failure retains metadata/accounting/slot for retry.

## Local implementation verification

- All six routes, sync raw-byte streaming, process-local async jobs, the
  Videos-local shared FIFO, timeout/cancellation recovery, in-memory CAS,
  media-root lease, atomic MP4 publication, lifecycle purge, and the three
  provisional adapters are implemented. Application-wide Chat/Videos admission
  is a documented follow-up, not a completed claim.
- The Round 1 review requirements for tokenizer lifecycle, bounded text-only
  multipart parsing, 25-hour retention/storage pressure, and queued-only DELETE
  are implemented and covered locally. Running public DELETE no longer sends a
  cancellation signal after dispatcher claim.
- The updated Serving suite passed `435` tests with `5` skipped. Formatting,
  changed/new-file lint, targeted typing, compile/import, CLI regression, and
  diff checks passed.
- Independent final review found no remaining P0/P1 issue within the completed
  Videos-local endpoint milestone; the later global Chat/Videos admission
  requirement remains pending by decision.
- These local/fake-engine and static Serving results are separate from the
  successful real Trainium offline benchmarks. The benchmarks cover
  compile/generate for fixed profiles; they do not replace the resident
  load/reuse/media/API/recovery run.

## Investigation checklist

### Capacity and residency

- [ ] Capture `MemTotal`, `MemAvailable`, swap, process RSS/PSS, and cgroup memory
  before load, after load, after smoke, at peak generation, and after a second
  request.
- [ ] Capture per-logical-core HBM categories with `neuron-monitor`, `neuron-top`,
  and per-NEFF INFO logs at the same checkpoints.
- [ ] Verify the exact serving profile stays below the provisional admission
  targets: approximately 85% peak HBM and 80% host RAM, with no swap.
- [ ] Confirm clean shutdown releases every Neuron runtime process and staging
  file.
- [ ] Record startup transient peak separately from steady-state resident use;
  checkpoint/page-cache occupancy must not be counted as process RSS.

### Correctness and media

- [ ] Validate fixed-seed MP4 codec/container, width, height, frame count, fps,
  duration, and silent/audio declaration with PyAV/ffprobe.
- [ ] Run startup smoke, normal request, queued deletion, in-progress DELETE
  rejection, timeout/recovery, and a second normal request for each
  model/profile.
- [ ] Run a long repeated-request soak for every fixed profile and confirm RSS,
  PSS, HBM, latency, and artifact counts return to a stable range.
- [ ] Compare Wan 2.2 dual-expert scheduler switching with a trusted reference.
- [ ] Compare HunyuanVideo 1.0 host-CLIP/host-VAE output with the existing staged
  reference path.
- [ ] Complete HunyuanVideo 1.5 offline prompt-conditioning and generation before
  enabling its public serving checkpoint.

### API lifecycle and cleanup

- [ ] Generalize Videos-local admission into one global ticket FIFO for Chat
  Completions, Videos sync/async, and every future generation endpoint that
  targets the same resident engine.
- [ ] Prove mixed-route FIFO order at the admission-ticket linearization point,
  one shared capacity/deadline budget, maximum engine concurrency of one, and a
  physically bounded queue after repeated queued deletion/disconnect.
- [ ] Exercise async create/poll/list/content/delete and sync raw-video response
  on Trainium.
- [x] Exercise queued DELETE, in-progress `409`, completion-vs-terminal-delete
  race, client
  disconnect, worker recovery, process-restart reset, and startup purge of
  crash-left staging/final media.
- [x] Verify tokenizer preload before readiness and bounded validation-executor
  behavior for both image and video requests; invalid requests receive no ticket;
  4-CPU-worker/32-waiting saturation and the 30-second sub-deadline use stable
  errors and count against the total deadline. Repeated timeout/disconnect must
  keep physical validation within 4/32 while model generation remains at most one.
- [x] Verify zero-file/32-field/256-KiB-part/1-MiB-total request limits,
  including chunked bodies and stable 413 responses.
- [x] Verify accelerated 25-hour expiry, periodic sweeping, structured disk
  pressure logs, 4,096-record failure-storm bounds, cumulative sync/async
  reservation/release, lease-safe artifact-first deletion, retry after unlink
  failure, and hard-reserve 507 behavior.
- [x] Prove dispatcher-vs-DELETE and sync-disconnect-vs-dispatch races use one
  atomic claim: successful queued deletion never executes, and a claimed ticket
  always reads as `in_progress` before `409` is returned.
- [x] Verify clean shutdown empties job state and removes every managed staging
  and final MP4; after restart, verify old IDs return `404` and list is empty.
- [x] Verify failed-job retrieval returns HTTP 200 with its stored structured
  error; pending content returns 409 and failed content returns 422.
- [x] Verify list order is stable by `(created_at, id)` descending with bounded
  cursor pagination.

## Release interpretation

Code presence and production enablement are separate. LTX-2, Wan 2.1, and
HunyuanVideo 1.0 are wired for provisional hardware validation under this
decision. A checkpoint is production-supported only after its correctness,
co-residency, recovery, media, and memory record is attached to this register.
Wan 2.2 remains excluded until its existing second-transformer wiring and
reference-correctness gate are closed; this does not retract its README/offline
artifact support. HunyuanVideo 1.5 remains excluded until its offline
end-to-end path exists. Capacity is not the active blocker for either decision.
