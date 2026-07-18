# Videos API

Difflet exposes a vLLM-Omni-style Videos API for fixed-profile text-to-video
models on Trainium. It supports process-local asynchronous jobs through
`/v1/videos` and direct synchronous generation through `/v1/videos/sync`.

This is the caller-facing API reference. Architecture decisions, implementation
tradeoffs, and Trn2 acceptance evidence live separately under
[`docs/design/t2v_serving`](../design/t2v_serving/00_summary.md).

## MVP scope

- One `difflet serve` process owns one model and one immutable compiled profile.
- Requests may use only the server's compiled width, height, frame count, FPS,
  dtype, and parallel topology. HTTP requests never trigger dynamic compilation.
- The current MVP is text-to-video. Image, video, and audio reference uploads,
  I2V, V2V, S2V, LoRA, interpolation, and generated audio are rejected.
- Synchronous and asynchronous generation share one bounded FIFO and one
  resident worker. The MVP runs one generation at a time.
- Async job metadata is in memory. Restarting the server removes all job IDs and
  starts with an empty list.
- Wan 2.1 and HunyuanVideo 1.0 default to their accepted fixed-profile Neuron
  VAE decoders. `--host-vae` retains the validated host rollback. LTX-2 remains
  host/hybrid because its current lower layer has no Neuron video-VAE decoder.
  Placement is fixed at startup and is not selected by an HTTP request field.

Validated Trn2 serving evidence currently exists for Wan 2.1, HunyuanVideo 1.0,
and LTX-2 at their recorded fixed profiles. The Wan 2.2 checkpoint can be
started explicitly for experiments, but it is not MVP-qualified because its
dual-transformer path still needs correctness and resident-serving acceptance.
HunyuanVideo 1.5 is not registered for serving.

### Security boundary

Difflet serving does not provide built-in authentication, tenant isolation, or
per-user job ownership. The `user` form field is opaque metadata, not an access
control. Any caller that can reach the server can list, retrieve, download, and
delete every job owned by that process. Bind to `127.0.0.1` for local use; when
binding to `0.0.0.0`, place the service on a trusted network or behind an
authenticating, authorizing reverse proxy.

## Quick start

### Start a server

The following example starts the validated HunyuanVideo 1.0 profile. The first
start may download weights and compile missing fixed-shape artifacts before the
resident worker becomes ready.

```bash
TOKENIZERS_PARALLELISM=false difflet serve \
  --model-id hunyuanvideo-community/HunyuanVideo \
  --cache-dir /mnt/model-cache/difflet \
  --tp-degree 4 \
  --cp-degree 1 \
  --height 320 \
  --width 512 \
  --num-frames 61 \
  --clip-placement neuron \
  --host 0.0.0.0 \
  --port 8092 \
  --request-timeout 1800 \
  --worker-restart-timeout 2400
```

Omitting `--host-vae` selects the accepted Neuron VAE default. This example also
selects the measured Neuron CLIP profile; omitting `--clip-placement` keeps CLIP
on the host. `--host-vae` is the only public VAE override and explicitly selects
the host decoder rollback.

The equivalent Wan 2.1 Neuron VAE profile is:

```bash
TOKENIZERS_PARALLELISM=false difflet serve \
  --model-id Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --cache-dir /mnt/model-cache/difflet \
  --tp-degree 4 --cp-degree 1 \
  --height 480 --width 832 --num-frames 9 \
  --host 0.0.0.0 --port 8092 \
  --request-timeout 1800 --worker-restart-timeout 2400
```

Add `--host-vae` to either startup command to use the host decoder instead.

### Hunyuan placement variants

Both decoder paths are release-qualified for the recorded fixed profile, with
Neuron VAE selected by default and host VAE retained for rollback. CLIP remains
a separate startup choice. Placement is fixed for the lifetime of the serve
process; HTTP requests cannot switch it or trigger compilation.

Phase A keeps VAE decode on the host and tests the serving-specific replicated
CLIP artifact (`TP=1`, resident `world_size=4`):

```bash
TOKENIZERS_PARALLELISM=false difflet serve \
  --model-id hunyuanvideo-community/HunyuanVideo \
  --cache-dir /mnt/model-cache/difflet \
  --tp-degree 4 --cp-degree 1 \
  --height 320 --width 512 --num-frames 61 \
  --clip-placement neuron \
  --host-vae \
  --host 0.0.0.0 --port 8092 \
  --request-timeout 1800 --worker-restart-timeout 2400
```

Phase B froze the accepted CLIP placement and changed only the decoder. The
Neuron VAE candidate passed Trn2 acceptance and is now the omitted-flag registry
default. There is intentionally no public `--vae-placement` selector:
`--host-vae` keeps the existing CLI-compatible host override without changing
the HTTP contract.

Readiness opens only after the resident worker has loaded and passed startup
checks:

```bash
curl -sS http://127.0.0.1:8092/health
curl -sS http://127.0.0.1:8092/ready
curl -sS http://127.0.0.1:8092/v1/models
```

### Create an asynchronous job

```bash
create_response=$(curl -sS -X POST http://127.0.0.1:8092/v1/videos \
  -F 'model=hunyuanvideo-community/HunyuanVideo' \
  -F 'prompt=a cinematic mountain landscape at sunrise' \
  -F 'size=512x320' \
  -F 'num_frames=61' \
  -F 'fps=24' \
  -F 'num_inference_steps=4' \
  -F 'guidance_scale=6.0')

video_id=$(printf '%s' "$create_response" | jq -r '.id')
printf '%s\n' "$create_response" | jq .
```

### Poll, download, and delete

```bash
curl -sS "http://127.0.0.1:8092/v1/videos/${video_id}" | jq .
curl -fL "http://127.0.0.1:8092/v1/videos/${video_id}/content" \
  -o "${video_id}.mp4"
curl -sS -X DELETE \
  "http://127.0.0.1:8092/v1/videos/${video_id}" | jq .
```

Content is available only after the job reaches `completed`. Deleting a queued
job cancels it. Deleting an `in_progress` job returns
`409 video_in_progress`; running generation is not publicly cancellable.

### Generate synchronously

```bash
curl -fL --silent --show-error \
  -X POST http://127.0.0.1:8092/v1/videos/sync \
  -F 'model=hunyuanvideo-community/HunyuanVideo' \
  -F 'prompt=a cinematic mountain landscape at sunrise' \
  -F 'size=512x320' \
  -F 'num_frames=61' \
  -F 'fps=24' \
  -F 'num_inference_steps=4' \
  -F 'guidance_scale=6.0' \
  -o hunyuan-sync.mp4
```

The synchronous endpoint waits for FIFO admission and generation, then streams
raw `video/mp4` bytes. It creates no async job record and does not upload the
result to S3.

## API reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/v1/videos` | Create an asynchronous video generation job |
| `POST` | `/v1/videos/sync` | Generate synchronously and return raw video bytes |
| `GET` | `/v1/videos/{video_id}` | Retrieve job status and metadata |
| `GET` | `/v1/videos` | List current-process video jobs |
| `GET` | `/v1/videos/{video_id}/content` | Download completed video content |
| `DELETE` | `/v1/videos/{video_id}` | Cancel queued work or delete a terminal job and output |

### Create request

Both create endpoints accept `multipart/form-data`.

Request support has two layers. The multipart parser first checks syntax and
global bounds. The selected model adapter then decides whether a parsed
model-specific field is supported. A field being parseable does not mean every
model, or any currently qualified model, accepts it.

| Field | Type | Default | Contract | Current model support |
| --- | --- | --- | --- | --- |
| `prompt` | string | required | Trimmed, non-empty text prompt, at most 32,768 characters | All video adapters; model token buckets may impose a smaller effective limit |
| `model` | string | server model | If supplied, must equal the server model | All |
| `seconds` | positive integer string | omitted | Compatibility assertion; `seconds * fps` must equal the compiled frame count | No current Wan, Hunyuan, or LTX-2 profile can satisfy it; omit this field |
| `size` | `WIDTHxHEIGHT` string | compiled profile | Must equal the compiled profile | All |
| `width` | integer | compiled profile | Must agree with `size` and the compiled profile | All |
| `height` | integer | compiled profile | Must agree with `size` and the compiled profile | All |
| `num_frames` | integer | compiled profile | Must equal the compiled profile | All |
| `fps` | integer | serving profile | Must equal the serving profile | All |
| `num_inference_steps` | integer | model default | From 1 through 200 globally; LTX-2 requires at least 2 | All, within the model-specific bound |
| `guidance_scale` | number | model default | Finite value from 0 through 20 | All |
| `guidance_scale_2` | number | null | Parsed from 0 through 20 before model validation | No current adapter; reserved for a future verified Wan 2.2 dual-transformer path |
| `boundary_ratio` | number | null | Parsed from 0 through 1 before model validation | No current adapter; reserved for a future verified Wan 2.2 dual-transformer path |
| `flow_shift` | number | null | Parsed as a finite number before model validation | No current adapter; request-time scheduler override is intentionally disabled |
| `negative_prompt` | string | null | Trimmed text at most 32,768 characters; empty text becomes null | Wan and LTX-2; HunyuanVideo 1.0 rejects non-empty values |
| `seed` | integer | `42` | From 0 through `2^63 - 1` | All |
| `user` | string | null | Optional trimmed opaque value, at most 256 characters; retained only in async process-local metadata and not returned | All; a sync request creates no lasting job metadata |

The current adapter matrix is:

| Field group | Wan 2.1 | Wan 2.2 experimental | HunyuanVideo 1.0 | LTX-2 |
| --- | --- | --- | --- | --- |
| Fixed shape and FPS fields | Must match profile | Must match profile | Must match profile | Must match profile |
| `num_inference_steps` | 1–200 | 1–200 | 1–200 | 2–200 |
| `guidance_scale` | 0–20 | 0–20 | 0–20 | 0–20 |
| `negative_prompt` | Supported | Supported | Non-empty values rejected | Supported |
| `guidance_scale_2` | Rejected | Rejected | Rejected | Rejected |
| `boundary_ratio` | Rejected | Rejected | Rejected | Rejected |
| `flow_shift` | Rejected | Rejected | Rejected | Rejected |
| `seconds` | Omit | Omit | Omit | Omit |

Wan 2.2 uses the current experimental single-transformer Wan adapter, so the
dual-transformer fields remain rejected. The registered default profiles are:

| Model | Default size | Frames / FPS | Default steps | Default guidance |
| --- | --- | --- | --- | --- |
| Wan 2.1 and experimental Wan 2.2 | `832x480` | `9 / 16` | 2 | 1.0 |
| HunyuanVideo 1.0 | `512x320` | `61 / 24` | 4 | 6.0 |
| LTX-2 | `768x512` | `121 / 24` | 40 | 3.5 |

Character limits are transport bounds, not a promise that every prompt fits the
compiled token bucket. Wan validates prompt and negative prompt at 512 tokens;
Hunyuan validates the templated Llama input at 351 tokens and uses a 77-token
CLIP bucket; LTX-2 validates prompt and negative prompt at 1,024 tokens.

The active resolved profile is not currently discoverable through `/v1/models`
or the job response, and startup flags may override the registered shape
defaults. Operators must communicate the active profile out of band. Generic
clients should omit `size`, `width`, `height`, `num_frames`, and `fps`; the server
fills them from its immutable profile. Supply them only as assertions when the
deployment profile is already known.

`seconds` is recognized for wire compatibility but is not a free-form duration
control. Current video profiles require frame-count congruences that cannot equal
an integral number of their 16- or 24-FPS frames, so callers must omit it. For
example, the 61-frame/24-FPS Hunyuan profile has an actual duration of `61 / 24`
seconds. Omitting `seconds` does not cause the server to derive or return an
integer duration; the public job response correctly contains `"seconds": null`.

The request parser allows at most 32 text fields, 256 KiB per text part, and
1 MiB for the complete multipart body. Integer components are rejected before
conversion when they exceed 19 digits, including the two components of `size`;
this prevents oversized numeric strings from blocking request handling or
surfacing as internal errors. File parts and reference fields such as
`input_reference`, `image_reference`, `video_reference`, and `audio_reference`
return `feature_not_supported` in the T2V MVP.

### Async job response

`POST /v1/videos` returns the created job immediately. `GET` and `LIST` use the
same public job shape:

```json
{
  "id": "video_gen_0123456789abcdef0123456789abcdef",
  "object": "video",
  "status": "queued",
  "model": "hunyuanvideo-community/HunyuanVideo",
  "prompt": "a cinematic mountain landscape at sunrise",
  "size": "512x320",
  "seconds": null,
  "progress": 0,
  "quality": "default",
  "created_at": 1784282483,
  "completed_at": null,
  "remixed_from_video_id": null,
  "error": null,
  "url": null,
  "expires_at": null
}
```

Statuses are `queued`, `in_progress`, `completed`, and `failed`. Timestamps are
Unix seconds.

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Process-local identifier generated as `video_gen_` plus 32-character UUID hex |
| `object` | string | Always `video` |
| `status` | string | `queued`, `in_progress`, `completed`, or `failed` |
| `model` | string | Model loaded by this serve process |
| `prompt` | string | Normalized prompt accepted for the job |
| `size` | string | Resolved compiled dimensions as `WIDTHxHEIGHT` |
| `seconds` | string or null | Echoes a supplied and validated integral `seconds` assertion; null when omitted. It is not the computed media duration |
| `progress` | integer | Current MVP reports 0 before terminal success and 100 when completed; it is not continuous stage progress |
| `quality` | string | Always `default` in the T2V MVP; there is no request-time quality selector |
| `created_at` | integer | Job creation time as Unix seconds |
| `completed_at` | integer or null | Terminal transition time for completed or failed jobs; null while active |
| `remixed_from_video_id` | null | Reserved compatibility field; remix is unsupported in the T2V MVP |
| `error` | object or null | Failed-job error as `code`, `message`, and nullable `error_type`; null otherwise |
| `url` | string or null | Difflet extension containing the presigned S3 GET URL after successful async S3 publication; null without remote publication |
| `expires_at` | integer or null | Terminal job cleanup-eligibility time as Unix seconds; null for queued and running jobs |

Field values by state are:

| `status` | `progress` | `completed_at` | `expires_at` | `error` | `url` |
| --- | ---: | --- | --- | --- | --- |
| `queued` | 0 | null | null | null | null |
| `in_progress` | 0 | null | null | null | null |
| `completed` | 100 | set | set | null | Set only for successful S3 publication; otherwise null |
| `failed` | 0 | set | set | set | null |

The public job object intentionally excludes internal resolved fields and
telemetry such as `width`, `height`, `num_frames`, `fps`, actual `duration_s`,
file name and size, inference time, stage timings, and memory measurements.
Those values remain available to internal storage and logs without expanding the
caller contract. To compute the actual media duration, clients need the compiled
profile's `num_frames / fps`; it is not currently present in the public job
object.

The locally committed MP4 is the source of truth for async completion. S3 upload
and URL signing are best effort: if either fails, the job remains `completed`
with `url: null`, the failure is logged, and `/content` remains available. When
S3 publication succeeds, the completed job includes the presigned `url`. Media
validation is the final request-deadline boundary: after the service claims a
validated result for local publication, a slow S3 upload cannot reverse it into
a failed job or delete its local MP4. “Best effort” describes the publication
outcome, not latency isolation: upload and URL signing are awaited in the same
resident-generation FIFO slot, so slow S3 can keep the current job
`in_progress` and delay later work.

### Synchronous response

`POST /v1/videos/sync` returns no JSON job object. It waits for the same FIFO as
async work and then streams raw `video/mp4` bytes with these response headers:

| Header | Meaning |
| --- | --- |
| `Content-Type` | `video/mp4` |
| `Content-Length` | Generated MP4 size in bytes |
| `X-Request-Id` | Ephemeral `video_sync_<hex>` request ID; it is not an async job ID |
| `X-Model` | Model loaded by the serve process |
| `X-Inference-Time-S` | Server-side generation and publication time in seconds |

The synchronous result is deleted from managed local storage after streaming and
is never uploaded to S3.

### List jobs

```bash
curl -sS 'http://127.0.0.1:8092/v1/videos?limit=20' | jq .
curl -sS \
  'http://127.0.0.1:8092/v1/videos?limit=20&after=video_gen_0123456789abcdef0123456789abcdef' \
  | jq .
```

`limit` defaults to 20 and must be an ASCII integer between 1 and 100. Invalid
or overlong numeric text returns `400 invalid_request` without integer
conversion. `after` is an optional job ID cursor. Jobs are ordered by
`created_at DESC, id DESC`.

| Field | Type | Meaning |
| --- | --- | --- |
| `object` | string | Always `list` |
| `data` | array | Public async job objects described above |
| `first_id` | string or null | First returned job ID, or null for an empty page |
| `last_id` | string or null | Last returned job ID, or null for an empty page |
| `has_more` | boolean | Whether another page exists after `last_id` |

An unknown `after` ID returns an empty page. The list is process-local and is
empty after server restart.

### Content response

`GET /v1/videos/{video_id}/content` succeeds only for a completed job. It streams
`video/mp4` bytes with `Content-Length` and
`Content-Disposition: attachment; filename="<video_id>.mp4"`. With S3 enabled,
the server still reads the managed artifact and proxies the content; clients may
instead use the response's presigned `url`.

An already-open content stream owns a safe file descriptor. DELETE unlinks the
managed name but does not revoke bytes already being streamed through that open
descriptor.

### Delete response

Successful deletion removes both the process-local job and its managed output:

```json
{
  "id": "video_gen_0123456789abcdef0123456789abcdef",
  "deleted": true,
  "object": "video.deleted"
}
```

A second lookup or delete returns `404 video_not_found`.

Artifact cleanup happens before job-row removal. If cleanup fails, DELETE
returns an error and retains the job row plus the authoritative local MP4 so the
caller can retry; the background orphan sweeper does not take ownership from
that live job transaction. The optional S3 mirror may already have been removed
if a later local unlink fails, so `/content` remains the portable retry path.

### Retry and disconnect behavior

Async create has no client-supplied idempotency key. Each accepted
`POST /v1/videos` receives a new server-generated ID, so retrying after losing a
successful response can create duplicate, billable generation work. Callers
should retain the first response and avoid automatic POST retries.

Queued sync work is removed when the client disconnects. Once either sync or
async work has been dispatched to the resident worker, public cancellation does
not stop it immediately: the worker/recovery fence continues to own the model
slot until execution terminates. A disconnected sync result is discarded after
that fence; DELETE of an `in_progress` async job continues to return
`409 video_in_progress`.

## Storage and retention

By default, async jobs become eligible for cleanup 25 hours after entering
`completed` or `failed`, or disappear when the serve process stops, whichever
happens first. Both retention and sweep interval are startup-configurable.
`expires_at` is the eligibility timestamp, not a hard read cutoff: cleanup may
occur after that time at the next periodic sweep or async-create-triggered
sweep. This lifetime applies to process-local metadata and managed local
artifacts. Queued and running jobs have no expiry time.

Without S3 configuration, completed async artifacts stay in the managed local
video root until DELETE, expiry, or shutdown. With complete `DIFFLET_S3_*`
configuration, Difflet also uploads completed async MP4 files and includes a
presigned `url` whose requested lifetime matches the video retention period.
The effective URL lifetime may be shorter because of S3 signing limits or
temporary credential expiry. Keep using `GET /content` when the client should
download through the Difflet server.
Process shutdown clears local state but does not currently enumerate and delete
S3 objects or revoke already issued URLs; use DELETE/TTL cleanup before shutdown
or an S3 lifecycle policy when remote cleanup is required.

## Errors

Request-time HTTP failures use the OpenAI-style envelope:

```json
{
  "error": {
    "message": "Video generation queue is full",
    "type": "invalid_request_error",
    "code": "queue_full"
  }
}
```

These codes can be returned directly by a create, retrieve, content, list, or
delete HTTP request:

| HTTP | Code | Meaning |
| --- | --- | --- |
| `400` | `invalid_request` | Malformed form, duplicate/unknown field, or bad list query |
| `400` | `invalid_extra_body` | A parsed field violates its global or model-specific bound |
| `400` | `invalid_prompt` / `prompt_too_long` | Prompt is empty, too large, or exceeds the model token bucket |
| `400` | `model_not_served` | Request `model` does not match the server model |
| `400` | `profile_mismatch` | Shape, frame count, FPS, or duration differs from the compiled profile |
| `400` | `feature_not_supported` | File/reference or model feature is outside the T2V MVP |
| `404` | `video_not_found` | Job ID does not exist in this serve process |
| `409` | `video_in_progress` | DELETE targeted a running generation |
| `409` | `video_not_ready` | Content was requested before generation completed |
| `413` | `request_too_large` | Multipart limits were exceeded |
| `422` | `video_generation_failed` | Content was requested for a failed job; inspect job `error` |
| `429` | `validation_capacity_exhausted` | All validation running and waiting slots are occupied |
| `429` | `queue_full` | Generation FIFO is full before the request is admitted |
| `429` | `queue_timeout` | A synchronous request exceeded its allowed FIFO wait; async jobs report this code in their failed job object instead |
| `429` | `video_retention_full` | The process-local retained-job limit is full |
| `503` | `validation_unavailable` / `video_service_unavailable` | The server is starting, stopping, or not accepting work |
| `504` | `validation_timeout` / `request_timeout` | Validation or a synchronous/pre-create request deadline expired |
| `500` | `video_artifact_missing` / `internal_error` | Completed content is missing or an internal failure occurred |
| `507` | `video_storage_full` | Local storage safety reservation failed |

After async `POST /v1/videos` has returned a queued job, later failures are not a
second HTTP response. Polling returns `status: failed` with the job-level shape:

```json
{
  "error": {
    "code": "request_timeout",
    "message": "Video request timed out",
    "error_type": "server_error"
  }
}
```

Background job codes include `queue_timeout`, `request_timeout`,
`server_shutdown`, `request_cancelled`, and sanitized generation failures such
as `internal_error`. The job's `completed_at` and cleanup-eligibility
`expires_at` are set when it enters `failed`.

## Relationship to vLLM-Omni

The endpoint layout and documentation organization follow the
[vLLM-Omni Videos API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/videos_api/):
quick start, endpoint table, request fields, response behavior, examples, and
storage. Difflet intentionally narrows the MVP to fixed-profile T2V on Trainium
and documents its process-local lifecycle, queued-only cancellation, bounded
multipart input, and optional S3 URL extension explicitly.
