# T2V API and Data Contract

## Status

The earlier proposal in this document used `/v1/chat/completions` with a
Difflet-specific `video_url` response. That proposal is superseded. The
implemented six-route wire surface is the dedicated Videos API documented in
[Videos API review and implementation plan](05_videos_api_review_and_plan.md).
Image-model servers retain their existing Chat Completions behavior.

> **Local hardening implemented:** tokenizer admission bounds, streaming
> multipart limits, queued-only DELETE, 25-hour TTL/sweeper, job-count cap, and
> the storage reservation ledger are covered locally. The remaining admission
> follow-up is a single owner if Chat and Videos routes are ever co-registered.

The target admission contract is application-wide rather than video-route
specific. Every endpoint that can invoke the same resident engine, including
Chat Completions and both Video create styles, must submit through one global
generation FIFO. The current route set registers Chat and Videos mutually
exclusively; cross-endpoint admission is therefore a documented implementation
follow-up before those routes may coexist.

The routes are registered only when the selected serving model has video output
metadata:

| Method | Path | Result |
| --- | --- | --- |
| `POST` | `/v1/videos` | Create an asynchronous job with a 25-hour terminal TTL, bounded by the current serve-process lifetime |
| `POST` | `/v1/videos/sync` | Generate synchronously and stream raw `video/mp4` bytes |
| `GET` | `/v1/videos/{video_id}` | Retrieve job status and metadata |
| `GET` | `/v1/videos` | List current-process jobs with bounded cursor pagination |
| `GET` | `/v1/videos/{video_id}/content` | Stream a completed MP4 |
| `DELETE` | `/v1/videos/{video_id}` | Cancel queued work or delete terminal work; reject `in_progress` |

## Request contract

Both create endpoints accept `multipart/form-data`. The provider normalizes
`prompt`, optional matching `model`, fixed shape/frame fields, seed, inference
steps, and the model-supported guidance/prompt options. Width, height, frame
count, fps, dtype, and topology remain constrained by the immutable compiled
serving profile; an HTTP request cannot trigger dynamic compilation.

P0 is text-to-video only. The parser uses `max_files=0`, at most 32 form
fields, a 256 KiB text-part limit, and a 1 MiB total request-body limit. File
parts are rejected as `feature_not_supported` while oversized text/total bodies
return `413 request_too_large`. The total byte limit must also cover chunked
requests rather than trusting `Content-Length` alone. These limits deliberately
do not copy vLLM-Omni's I2V/V2V upload support.

Inputs such as I2V/V2V references, audio generation, LoRA, interpolation, and
arbitrary extra parameters are rejected until the lower model path and their
security/validation contracts exist. The Wan 2.2 checkpoint is accepted only for
an explicit experimental single-transformer serving profile. Wan 2.2-only
dual-expert controls are not accepted, and this profile is not an MVP default:
the current path produces an artifact while disabling `transformer_2`, so
dual-transformer correctness remains unproven.

## Global admission and deadline contract

Validation has a separate bounded pre-admission domain; this is not model
generation concurrency. Its configurable P0 defaults are four synchronous CPU
validation workers, 32 physically queued validation entries, and a 30-second
validation timeout. Model generation remains limited to exactly one running
request. Each synchronous validator call runs in an executor thread and
FastAPI awaits it asynchronously. Submitting when all 36 validation positions are occupied is
rejected immediately as `429 validation_capacity_exhausted`; the implementation
must not use an executor with an unbounded submission queue. The validation
deadline starts when bounded body parsing finishes and the request first seeks a
validation position; it covers validation wait and execution only. Validation
timeout returns stable `504 validation_timeout`. After validation, Video queue
wait and generation execution use the independent clocks described below. A
generation ticket is assigned only after successful validation, so FIFO order
is among validated requests.

Validation capacity belongs to the underlying executor work, not to the HTTP
waiter. If timeout or disconnect occurs after a validation thread starts, that
work keeps its running-validation slot until its future actually finishes; its
late result is discarded. Work that has not started releases its waiting slot
only after it is atomically removed from the physical queue. If removal loses a
race, the entry continues to own a waiting/running slot until executor
completion. Repeated timeout/disconnect therefore cannot exceed four actual CPU
validation or 32 actual waiting entries.
The four workers are threads in one dedicated FastAPI-lifespan executor, not
additional Uvicorn processes or resident model workers.

- `POST /v1/chat/completions`, `POST /v1/videos`, and
  `POST /v1/videos/sync` share one process-local capacity budget whenever they
  target the same resident engine. Future generation endpoints join the same
  domain.
- Capacity includes tickets reserved while async metadata is created, queued
  requests, and the one running request. A route cannot call `engine.generate`
  directly or use a private waiting queue.
- FIFO order is the monotonic ticket order assigned after structural and
  bounded CPU token validation,
  not whichever handler finishes its later asynchronous preparation first.
- The physical queue as well as the live-work count must remain bounded after
  repeated queued cancellation or client disconnects.
- Immediate capacity rejection uses the same behavior for both Video create
  endpoints. Video queue wait and execution have separate clocks: the bounded
  FIFO defaults to 86,400 seconds, while `request_timeout` starts only after
  the dispatcher claims the item and changes it to `in_progress`. An explicit
  `--queue-timeout` overrides the modality default.
- Queue expiration removes the physical FIFO entry and releases its live-work,
  job-slot, and storage reservations, so it is never submitted for generation.
  Async metadata remains as a terminal `failed` job with error code
  `queue_timeout` until DELETE or TTL cleanup; synchronous callers receive
  `429 queue_timeout`.
- Model/health reads and job list/status/content reads do not consume a
  generation slot. DELETE is control traffic and must not wait behind
  generation tickets: it cancels queued work, rejects running work, and deletes
  terminal work.
- Async admission also reserves one process-local job-record slot. The P0 cap
  is configurable and defaults to 4,096 live or retained async records. If an
  opportunistic expiry sweep cannot free a slot, create returns
  `429 video_retention_full`. Sync requests do not consume job-record slots.

## Runtime and output types

`ServingProfile` and model registry metadata carry video-specific identity:

- `num_frames`
- `output_modality="video"`
- `output_mime_type="video/mp4"`
- immutable model, revision, dtype, shape, topology, and compiler identity

`DiffletGenerateRequest` carries the normalized fixed-profile request and a
parent-owned staging target. The worker returns a file-backed generated-output
descriptor rather than normal MP4 bytes over multiprocessing IPC. The parent
validates the descriptor and atomically publishes the file.

The async endpoint returns process-local job metadata. The sync endpoint creates
no job record and returns raw MP4 bytes with request/model/timing headers;
its temporary committed artifact is removed after streaming completes or
the stream closes. Client disconnect does not cancel healthy model execution
after dispatch; the eventual sync output is discarded and cleaned.

## Validation ownership

Validation remains split across two layers:

1. The HTTP provider performs cheap structural checks before admission:
   required/nonempty bounded prompt, allowed fields, numeric finiteness, model
   equality, fixed profile equality, and unsupported-feature rejection.
2. The selected adapter validates model-specific token buckets, legal frame
   congruence, guidance/parallelism support, and fixed dtype/placement rules.
   Each image/video validator preloads its CPU tokenizer before readiness and
   runs request tokenization on a bounded validation executor. This loads
   vocabulary/rules only, not model weights or Neuron artifacts.

Model validation runs in the parent without loading model weights. Compiled
artifact readiness is checked separately before the resident worker starts.
Invalid requests never receive a generation ticket; FIFO order applies among
requests that finish bounded validation successfully.

## Error mapping

| Condition | HTTP result |
| --- | --- |
| Invalid form, shape/profile conflict, or unsupported field | `400` |
| Multipart text/total body exceeds T2V limits | `413 request_too_large` |
| Queue/admission capacity exhausted | `429` |
| Validation domain is physically full | `429 validation_capacity_exhausted` |
| Async retained-job slot unavailable | `429 video_retention_full` |
| Unknown job | `404` |
| Content requested before completion | `409` |
| DELETE targets an `in_progress` job | `409 video_in_progress` |
| Content requested for a failed job | `422` |
| Request deadline exceeded | `504` plus internal terminate/restart recovery fencing |
| Validation exceeds its 30-second sub-deadline | `504 validation_timeout` |
| Worker unavailable or recovering | `503` |
| Disk safety reserve cannot admit another artifact | `507 video_storage_full` |
| Generation, encoding, repository, or storage failure | stable `5xx` mapping |

Terminal async job metadata remains retrievable with its structured error or
artifact until its 25-hour TTL expires or the serve process stops, whichever
happens first.
Successful DELETE removes the resource; there is no retrievable `deleted`
state.

## DELETE and cancellation contract

The public vLLM-Omni endpoint documentation describes DELETE only as deleting a
job and stored output. Its current source implementation, pinned during this
review at
[`238fc0a`](https://github.com/vllm-project/vllm-omni/blob/238fc0a609311235a671940cf209a7eb72c1dc29/vllm_omni/entrypoints/openai/api_server.py#L3223-L3272),
attempts cancellation for both `queued` and `in_progress`, waits up to two
seconds, and returns `409` when cancellation is still in progress. Its
`AsyncOmni.generate` cancellation path
[aborts internal engine requests](https://github.com/vllm-project/vllm-omni/blob/238fc0a609311235a671940cf209a7eb72c1dc29/vllm_omni/entrypoints/async_omni.py#L438-L449).
This is observed reference behavior, not an explicit stability guarantee in the
public API prose.

Difflet intentionally narrows the public cancellation contract:

- queued DELETE removes the ticket/job and guarantees it never executes;
- running DELETE returns `409 video_in_progress`, does not signal cancellation,
  and leaves the job running; the client may retry after it becomes terminal;
- completion that has already claimed publication is serialized with DELETE,
  after which DELETE removes the terminal artifact and metadata.

The dispatcher and queued DELETE share one admission-state lock and one
linearization point. Under that lock, the dispatcher dequeues the physical
entry, claims it, and performs the `queued -> in_progress` CAS before releasing
the lock. DELETE under the same lock either removes the queued entry/job and
releases its ticket, job slot, and storage reservation first, or observes the
dispatch claim and returns `409 video_in_progress`. It is never valid for DELETE
to return success and for that ticket to execute later. Sync disconnect uses the
same rule: remove before the dispatch claim; after the claim, set
discard-on-completion without signalling cancellation.

Hard deadlines, shutdown, and worker failure may still terminate/restart the
worker as internal safety/recovery behavior. They are not public running-job
cancellation guarantees.

## Process-lifetime state and media ownership

- `InMemoryVideoJobRepository` stores jobs and enforces state changes with
  compare-and-swap for the current serve-process lifetime.
- Queued and in-progress jobs expose `expires_at: null`. Completed and failed
  jobs receive `expires_at = terminal_time + 25 hours` in the same terminal CAS.
  A periodic sweeper removes expired metadata and artifacts; effective
  retention never exceeds the serve-process lifetime.
- A process/host root lease prevents two services from owning the same media
  root concurrently.
- The media store confines staging and final files below private server-owned
  directories and rejects symlink/non-regular-file substitutions.
- MP4 commit is atomic and precedes the async `completed` transition.
- Open content uses an inode-validated file-descriptor lease so concurrent
  DELETE cannot substitute the streamed file.
- One process-level storage ledger covers both sync and async active work. Each
  accepted request atomically reserves `max_artifact_bytes`; one global safety
  margin, defaulting to `max(1 GiB, max_artifact_bytes)`, is kept outside those
  per-request reservations. Admission conservatively requires filesystem free
  bytes to cover all outstanding reservations, the new reservation, and that
  margin; otherwise it returns `507 video_storage_full`.
- On async commit, the active maximum reservation becomes the actual retained
  artifact size and the unused difference is released. Failed generation and
  queued DELETE release the full reservation. Terminal DELETE/TTL release the
  retained bytes after unlink succeeds. Sync commit converts to actual temporary
  bytes and releases them after stream close/disconnect cleanup. Shutdown clears
  every reservation after fenced purge. Warning is logged below twice the safety
  margin; error is logged at or below the margin.
- Artifact publication, terminal DELETE, TTL sweeping, and content-lease
  acquisition serialize through the same artifact-lifecycle lock. Content GET
  opens its inode-validated descriptor while holding that lock, then streams
  without it; a later unlink cannot invalidate the open descriptor. DELETE and
  sweeper remove artifact first and metadata second. If unlink fails, metadata,
  retained-byte accounting, and the job-record slot remain for retry; the API
  must not expose `artifact_missing` as an internal 500 for a valid lease race.
- The 4,096 async record cap includes queued, in-progress, completed, and failed
  jobs. Admission reserves a record slot atomically before job creation;
  queued/terminal DELETE, TTL, and shutdown release it. This bounds a rapid
  failed-job storm even when no artifact bytes are retained.
- Clean shutdown clears in-memory jobs and purges staging and final MP4 files.
- If the process crashes, the next same-profile process that acquires the
  media-root lease purges all residual managed media before accepting work.
- A restart never restores jobs: old IDs return `404`, and the initial list is
  empty.
- MP4 encode/validation failure is terminal; there is no tensor-file fallback.

These are process-lifetime consistency and media-ownership guarantees, not
evidence that a model/profile has passed the required real `trn2.3xlarge`
acceptance run.

## S3 video artifacts and `url`

With complete `DIFFLET_S3_*` configuration, asynchronous videos use local
temporary staging and upload to S3 after validation. Synchronous `/sync` results
use only a local temporary file and are cleaned up after the response; they do
not upload to S3. Without S3, asynchronous jobs use the local artifact store;
workers do not write directly to remote object storage.

After a successful upload, asynchronous status and list responses include the
single Difflet extension field `url`, containing a short-lived presigned GET URL.
It is `null` before completion, without S3, or after upload failure. The API does
not expose `artifact_url` or `presigned_url` aliases, and this does not change
the OpenAI-standard fields.

`GET /content` remains OpenAI-compatible: the server reads from the local file or
S3 SDK and proxies/streams `video/mp4` bytes instead of redirecting the client.
DELETE and the TTL sweeper remove both S3 objects and local staging remnants.

## Video VAE host override

The current MVP selects the accepted Neuron VAE for Wan 2.1 and HunyuanVideo
1.0 while retaining the existing startup-only host override. LTX-2 keeps its
required host/hybrid path. This does not change the six-route lifecycle contract:

```text
--host-vae present -> host/CPU VAE decode
--host-vae omitted -> the model registry's validated default
```

No public `--vae-placement` selector is added. For the accepted Wan 2.1 and
Hunyuan profiles, omission selects Neuron and `--host-vae` is the rollback
command. Host-only adapters such as LTX-2 keep a host registry default.
Placement is not accepted in an HTTP request.

The option is part of the immutable serving profile. All non-VAE artifacts and
stage bindings must remain identical between the two placements: model
revision, shape, dtype, parallel topology, text/prompt-encoder placement,
transformer artifacts, and latent schema do not change. Only the VAE artifact,
decoder runner binding, and decoder placement may differ. The choice is
recorded in internal job/stage metadata, while the OpenAI-compatible public
video response remains unchanged. One running serve process never switches
placement or compiles a new VAE in response to an HTTP request.

For Hunyuan, `--clip-placement host|neuron` is also startup-only. The VAE
comparison begins only after a separate CLIP-placement
experiment has selected one accepted CLIP baseline. That selected CLIP artifact,
binding, and placement are then frozen across both VAE candidates. The earlier
CLIP experiment keeps host VAE fixed and permits only the CLIP artifact,
prompt-encoder binding, and placement to differ. The two changes are never
evaluated together.
