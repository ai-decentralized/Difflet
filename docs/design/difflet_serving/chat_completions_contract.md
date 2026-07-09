# Difflet Chat Completions Contract

Date: 2026-07-07

This document defines the first serving API contract for Difflet. The public
entrypoint is OpenAI-style `POST /v1/chat/completions`; image/video-specific
routes are not part of the first milestone.

This file, together with `architecture.md` and `engine.md`, is the authoritative
P0 serving design. The older
`docs/plans/2026-07-06-difflet-serving-engine.md` keeps broader future design
notes, including rotating, subprocess, and multi-plan ideas, but those are not
part of the P0 implementation scope unless repeated here.

## Route

```text
POST /v1/chat/completions
```

Compatibility category:

- OpenAI-compatible wrapper for request/response envelope.
- Difflet-specific extension through `extra_body`.
- P0 output is generated images only.

The route must not hard-code a model family. It calls the resolved
`DiffletServingEngine`, then formats the engine output according to the model's
declared `output_modalities`.

P0 scope:

- Only image-output serving models are registered.
- `modalities` may be omitted or set to `["image"]`.
- `["video"]`, `["text"]`, `["audio"]`, and mixed unsupported modalities return
  `400 unsupported_modality`.
- Video request/response shapes in this document are future P1+ reference only,
  not part of the first implementation.

## Request Shape

Minimal request:

```json
{
  "model": "Qwen/Qwen-Image",
  "messages": [
    {
      "role": "user",
      "content": "A beautiful landscape painting"
    }
  ],
  "extra_body": {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 4.0,
    "seed": 42
  }
}
```

`model` handling:

- If the server is started with one model, `model` may be omitted. The server
  uses the startup model.
- If `model` is provided, it must match `ServingProfile.accepted_model_ids`.
  This allowlist contains the exact startup model id plus explicit
  same-checkpoint serving aliases for the same loaded profile.
- Do not include every `ModelEntry.hf_paths` value by default. A registry entry
  may group sibling checkpoints, such as Flux dev and schnell, that are not
  aliases for the same loaded artifact.
- Startup model resolution may use `difflet.registry` detector functions, but
  request-time model matching must not. Broad detectors such as substring
  matches are too permissive for a single-model server.
- A mismatch returns `400 model_not_served`.
- A future multi-model server may accept multiple registered model ids, but the
  first implementation is one fixed `ServingProfile` per server process.

Prompt extraction:

- Use the last user message as the prompt source.
- If `content` is a string, use it directly.
- If `content` is an array, concatenate supported text items in order. P0
  accepts only content items with this exact shape:

```json
{"type": "text", "text": "prompt text"}
```

- For P0, reject any non-text content item in the selected user message with
  `400 unsupported_input_modality`. Do not silently drop image, audio, file, or
  other input parts.
- Reject malformed text items, missing `text`, non-string `text`, or unknown
  keys in a content item with `400 unsupported_input_modality`.
- If no non-empty prompt can be extracted, return `400 invalid_prompt`.
- After model-specific prompt templating/tokenization, the prompt must fit the
  selected adapter/profile text bucket. P0 must not silently truncate prompts.
  If the tokenized prompt exceeds the profile limit, return
  `400 prompt_too_long`.
- Serving validation must perform this length check without tokenizer
  truncation. After the prompt is proven to fit, the adapter may pad to the
  fixed execution bucket required by the compiled artifact.

Supported top-level fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `model` | No for single-model server | Requested model id or alias. |
| `messages` | Yes | Chat messages. Last user text becomes the prompt. |
| `modalities` | No | Optional desired output modalities. Omitted means the served model's default output modality. |
| `extra_body` | No | Difflet generation and request-facing shape parameters. |

P0 top-level field handling:

- Allowed fields: `model`, `messages`, `modalities`, and `extra_body`.
- Ignored compatibility fields: `response_format` and
  `artifact_ttl_seconds`. P0 response policy is deployment-owned; these fields
  have no effect whether sent top-level or inside `extra_body`.
- Known Difflet request fields sent at top level return
  `400 invalid_extra_body`, not `400 feature_not_supported`. This includes
  generation fields such as `height`, `width`, `num_frames`,
  `num_inference_steps`, `steps`, `guidance_scale`, `true_cfg_scale`, `seed`,
  `negative_prompt`, and `output_format`; startup/runtime fields such as
  `tp_degree`, `cp_degree`, `cp_mode`, `cfg_parallel`, and `sp_enabled`; and
  TeaCache or other advanced runtime knobs. P0 generation and shape fields must
  live under `extra_body`.
- Explicitly rejected fields: `stream`, `tools`, `tool_choice`, `functions`,
  `function_call`, `temperature`, `top_p`, `n`, `stop`, `max_tokens`,
  `metadata`, and any field that implies text generation, streaming, tool use,
  or other unsupported chat behavior.
- Unknown top-level fields return `400 feature_not_supported` unless they are
  added to this allowlist in a later revision.
- Difflet generation and request-facing shape fields must live inside
  `extra_body`. P0 should not accept flattened generation fields at the top
  level; their presence there is invalid placement and returns
  `400 invalid_extra_body`.

`modalities` compatibility:

- OpenAI chat `modalities` is primarily a text/audio output selector. Difflet
  uses the same top-level field as a serving extension for generated media.
- For P0 text-to-image, omit `modalities` or pass `["image"]`.
- `["video"]`, `["text"]`, `["audio"]`, or mixed unsupported modalities return
  `400 unsupported_modality` unless a future model adapter explicitly supports
  them.

## `extra_body` Parameters

The serving handler normalizes generation fields into `DiffletGenerateRequest`.
P0 response policy is deployment-owned: the handler always returns an artifact
URL and uses the server-configured artifact TTL. Request response-policy fields
such as `response_format` or `artifact_ttl_seconds` are ignored.

Common generation fields:

| Field | Type | Applies to | Meaning |
| --- | --- | --- | --- |
| `height` | int | image | Requested output height. Must match server `ServingProfile` in P0. |
| `width` | int | image | Requested output width. Must match server `ServingProfile` in P0. |
| `num_frames` | int | future video | Requested frame count. Qwen/Flux image adapters reject non-null values with `400 invalid_extra_body`; `null` is treated as absent. |
| `num_inference_steps` | int | image | Preferred request name for denoising steps. |
| `steps` | int | image | Alias for `num_inference_steps`; reject if both disagree. |
| `guidance_scale` | float | image | Classifier-free guidance or model-specific guidance scale. |
| `true_cfg_scale` | float | model-specific | True-CFG guidance scale only for models that explicitly expose a true two-pass CFG path. Qwen/Flux MVP adapters reject it. |
| `seed` | int | image | Random seed. Default comes from the model adapter. |
| `negative_prompt` | string | model-specific | Negative prompt if the model adapter supports it. |
| `output_format` | string | image | Optional output format, e.g. `png`; defaults to the model's configured output format. Selected model adapter validates support and MIME mapping. |

Startup-only serving/runtime fields are not accepted in request `extra_body`:
`tp_degree`, `cp_degree`, `cp_mode`, `cfg_parallel`, and `sp_enabled`. They are
valid only as `difflet serve` startup overrides and as internal
`ServingProfile` identity fields for cache, core placement, and diagnostics. If
a request includes any of these fields, return `400 invalid_extra_body`.

Response-policy fields are ignored in request `extra_body`: `response_format`
and `artifact_ttl_seconds`. P0 always stores generated media through
`ArtifactStore` and returns a URL in `image_url.url`; artifact TTL comes from
server startup configuration.

P0 rejects TeaCache and advanced runtime fields with
`400 invalid_extra_body`. A future API revision may add explicit,
adapter-declared runtime fields, but P0 must not pass them through implicitly.
Unknown model-specific fields should return `400` by default; silently ignoring
generation knobs makes benchmarking and debugging unreliable.

`extra_body` validation rules:

- Unknown fields return `400 invalid_extra_body`.
- Alias pairs are deterministic. For P0, `steps` is an alias for
  `num_inference_steps`; if both are present and values differ, return
  `400 invalid_extra_body`. If both are equal, normalize to
  `num_inference_steps`.
- `height` and `width` are the P0 image request-facing shape/profile fields.
  Values must match the loaded serving profile in P0. Mismatch returns
  `400 profile_mismatch`.
- `num_frames` is not a P0 image field. For Qwen/Flux image adapters,
  `extra_body.num_frames` with any non-null value returns
  `400 invalid_extra_body`; `null` is treated as absent. Reserve
  `num_frames` profile matching for future video adapters.
- `tp_degree`, `cp_degree`, `cp_mode`, `cfg_parallel`, and `sp_enabled` are
  startup-only runtime/profile fields. If they appear in request `extra_body`,
  return `400 invalid_extra_body`.
- `response_format` and `artifact_ttl_seconds` are deployment response-policy
  fields. If they appear top-level or inside request `extra_body`, ignore them.
  They must not affect generation, artifact TTL, output URL behavior, or worker
  input.
- Model-specific fields, such as `true_cfg_scale` or `negative_prompt`, are
  accepted only when the selected model adapter declares support.
- Numeric values must be finite and in the selected adapter's declared range;
  invalid values return `400 invalid_extra_body` before a worker request is
  admitted.

Qwen-Image P0 value limits:

| Field | P0 rule |
| --- | --- |
| prompt length | Reject after Qwen prompt templating/tokenization if it exceeds the configured encoder bucket. Default Qwen serving profile uses the existing `enc_seq=256` bucket unless the adapter declares another value. |
| `num_inference_steps` / `steps` | Positive integer. Adapter default is `4`; server may set an upper bound such as `--max-inference-steps`, and values above it return `400 invalid_extra_body`. |
| `guidance_scale` | Finite non-negative float. Adapter default is `4.0`. |
| `seed` | Integer in the range accepted by PyTorch manual seeding; P0 should accept `0 <= seed <= 2**63 - 1`. |
| `output_format` | `png` only for Qwen P0. |

Flux P0 value limits:

| Field | P0 rule |
| --- | --- |
| prompt length | Reject after Flux tokenizer/template handling if it exceeds the configured Flux text bucket. Default Flux serving profile uses the existing `max_sequence_length=512` path unless the adapter declares another value. No silent truncation. |
| `num_inference_steps` / `steps` | Positive integer. Adapter default is `28`; server may set an upper bound such as `--max-inference-steps`, and values above it return `400 invalid_extra_body`. |
| `guidance_scale` | Finite non-negative float. Adapter default is `3.5`. |
| `seed` | Integer in the range accepted by PyTorch manual seeding; P0 should accept `0 <= seed <= 2**63 - 1`. |
| `output_format` | `png` only for Flux P0. |
| `true_cfg_scale` / `negative_prompt` | Rejected in Flux P0 until the serving adapter explicitly exposes a true-CFG path. |

Request shape/profile fields:

- `height`
- `width`

These are the only image request-time profile matching fields in P0. The first
serving implementation fixes them at startup through `ServingProfile`. If a
request provides a different value from the active profile, return
`400 Bad Request` with a profile mismatch error. Missing optional shape fields
are filled from the active profile.

`num_frames` is reserved for future video adapters. Qwen/Flux image adapters do
not profile-match it in P0; non-null `extra_body.num_frames` returns
`400 invalid_extra_body`.

Startup-only profile identity fields:

- `tp_degree`
- `cp_degree`
- `cp_mode`
- `cfg_parallel`
- `sp_enabled`

The active profile is built from the registered model defaults plus explicit
`difflet serve` startup overrides. For example, Flux keeps its registry
parallel default when `--tp-degree` is omitted, while Qwen uses its registry
default `tp_degree=4`, `cp_degree=1`. These fields affect AOT cache identity,
core placement, and worker loading, but clients cannot override or assert them
per request.

For Qwen P0, `cp_degree > 1` is not a supported active profile even though some
compiled directory names include `cp`. The server must reject that startup
profile until the Qwen text encoder implementation passes context-parallel
configuration into its Neuron model and the shared-worker smoke passes.

P0 loads exactly one `ServingProfile`. Future multi-profile serving may match a
request against several configured loaded profiles, but that is not part of the
first contract. P0 must reject multi-profile startup flags and must not compile,
load, or switch profiles because of request fields.

Per-model input differences:

| Model type | Output | Relevant request fields | Notes |
| --- | --- | --- | --- |
| `qwen_image` | image | `height`, `width`, `num_inference_steps`, `guidance_scale`, `seed` | P0 target gated by shared-worker load/smoke. `height` and `width` must match the startup profile. Reject non-null `num_frames` and `true_cfg_scale`; Qwen uses guidance-distilled single-pass guidance. |
| `flux` | image | `height`, `width`, `num_inference_steps`, `guidance_scale`, `seed` | MVP target. Single-pipeline image model. Reject non-null `num_frames`. |
| `wan` | video | `height`, `width`, `num_frames`, `num_inference_steps`, `guidance_scale`, `seed` | Future P1+ only. Video output must use `ArtifactStore`. |
| `hunyuan_video` | video | `height`, `width`, `num_frames`, `num_inference_steps`, `guidance_scale`, `seed` | Future P1+ only. Video output must use `ArtifactStore`. |
| `ltx_2` | video | `height`, `width`, `num_frames`, `num_inference_steps`, `guidance_scale`, `seed` | Future P1+ only. |

The handler should not assume every model accepts every field. It should ask
the selected model adapter/spec to validate `extra_body`.

## Output Modality Mapping

Each P0 serving model spec must declare image output:

```python
output_modalities=("image",)
```

The chat handler maps engine outputs to chat content parts:

| Engine output modality | Chat content part type | URL field | Default payload policy |
| --- | --- | --- | --- |
| `image` | `image_url` | `image_url.url` | R2 artifact URL in MVP deployment. |

Future P1+ video serving may add a `video_url` content part. It should be a
Difflet extension to the OpenAI-style chat response envelope rather than
overloading `image_url`.

If a future model returns multiple outputs, return multiple content parts in
the same assistant message, ordered by model spec. P0 should reject
multi-modality requests unless the model explicitly declares support.

## Image Response

For image models, MVP deployment returns an artifact-backed URL by default:

```json
{
  "id": "chatcmpl-difflet-...",
  "object": "chat.completion",
  "created": 1783420000,
  "model": "Qwen/Qwen-Image",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example-r2-url/generated/file_abc.png"
            }
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

Deployment URLs must come from `ArtifactStore` backed by R2; they must not
expose arbitrary local filesystem paths. Requesting `response_format` is ignored
in P0; it must not switch the response to data URLs or any local path mode.

Engine/output boundary:

- `DiffletServingEngine.generate(...)` returns bytes plus MIME metadata.
- `difflet/serving/openai/serving_chat.py` stores bytes through `ArtifactStore`
  using the server-configured artifact TTL.
- The handler must call `ref = await ArtifactStore.put_bytes(...)`, then
  `url = await ArtifactStore.get_url(ref)`, and return only `url` in
  `image_url.url`.
- `ArtifactRef.uri` is an internal storage URI or backend locator. The handler
  must never return it directly; it returns only the value from
  `ArtifactStore.get_url(ref)`.
- `ArtifactStore.put_bytes(...)` and `ArtifactStore.get_url(...)` are bounded by
  `artifact_store_timeout`, default 60s. The artifact backend must configure
  SDK/client timeouts so upload or presign cannot hang the HTTP handler after
  worker generation succeeds.

## Future P1+ Video Response

Future P1+ video models should return an artifact-backed content part:

```json
{
  "id": "chatcmpl-difflet-...",
  "object": "chat.completion",
  "created": 1783420000,
  "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": [
          {
            "type": "video_url",
            "video_url": {
              "url": "https://example-r2-url/generated/file_xyz.mp4",
              "file_id": "file_xyz",
              "mime_type": "video/mp4",
              "expires_at": 1783423600
            }
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

Video is not part of P0. When enabled later, generated videos should use
object-storage artifact URLs rather than base64.

## Model Support Matrix

The chat route is the single public generation route for every model that the
serving registry exposes. The first implementation should only serve explicitly
registered, hard-coded models. A model is not automatically supported just
because it exists in Hugging Face or Difflet's CLI.

Serving startup may reuse `difflet.registry.resolve_model(...)` for model family
resolution, default shape, default parallel config, backend support, and download
patterns. The serving registry must still declare which exact checkpoint ids are
enabled for serving. A base registry entry may group sibling checkpoints that
are not interchangeable for a loaded worker.

MVP is text-to-image only. Qwen-Image and Flux are both MVP serving targets and
use the same image response contract. Each server process serves one model and
one output modality.

For Flux P0, serving enables only `black-forest-labs/FLUX.1-dev`. Other Flux
checkpoints in the base registry, such as `black-forest-labs/FLUX.1-schnell`,
must be rejected at startup until the Flux serving orchestrator is parameterized
and verified for that exact checkpoint.

| Model id | Model type | Default output | Chat content type | Min cores for full resident | P0 serving status | Notes |
| --- | --- | --- | --- | ---: | --- | --- |
| `Qwen/Qwen-Image` | `qwen_image` | image/png | `image_url` | `max(stage_cores)` | P0 target, gated | Shared-process 3-stage `prompt_encoder -> denoiser -> decoder`; enabled only after shared-worker co-load and smoke pass for the active profile. CFG-parallel disabled. Per-stage resident core sums are future-only capacity planning. |
| `black-forest-labs/FLUX.1-dev` | `flux` | image/png | `image_url` | `tp*cp` | P0 target | Single pipeline path through `DiffletPipeline`; CFG-parallel disabled. |
| `black-forest-labs/FLUX.1-schnell` | `flux` | image/png | `image_url` | `tp*cp` | Not enabled in P0 | Same base registry family as Flux dev, but not a same-checkpoint alias. Enable only after the Flux serving/common orchestrator is parameterized and tested for this checkpoint. |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `wan` | video/mp4 | `video_url` | `tp*cp + 1` | P1/P2 | Resident 2-stage `denoiser -> decoder`; video artifact output. |
| `Wan-AI/Wan2.1-T2V-14B-Diffusers` | `wan` | video/mp4 | `video_url` | `tp*cp + 1` | P1/P2 | Same contract as Wan2.2 if adapter supports it. |
| `hunyuanvideo-community/HunyuanVideo` | `hunyuan_video` | video/mp4 | `video_url` | `1 + tp*cp + tp*cp` | P2 | Multi-stage video; artifact output. |
| `Lightricks/LTX-2` | `ltx_2` | video/mp4 | `video_url` | `tp*cp` | P2 | Pipeline-style video if serving adapter is added. |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` | `hunyuan_video_15` | video/mp4 | `video_url` | unknown | Not enabled | Current CLI orchestrator reports compile/generate not implemented. |

For Wan, `cfg_parallel` doubles the denoiser core term, so
`min_cores_for_full_resident` becomes `tp*cp*2 + 1` when enabled.

Every serving-enabled model must provide:

- model id and aliases
- output modality
- default MIME type
- supported `extra_body` fields
- fixed `ServingProfile`
- adapter-specific validation
- artifact policy for final output
- default `output_format` and MIME type

## Error Contract

Errors should use the same envelope shape across model types:

```json
{
  "error": {
    "message": "request shape does not match serving profile",
    "type": "invalid_request_error",
    "code": "profile_mismatch"
  }
}
```

Required status codes:

| Condition | HTTP status | Error code |
| --- | ---: | --- |
| Unsupported model id | 400 | `model_not_served` |
| Unsupported output modality | 400 | `unsupported_modality` |
| Unsupported input content item | 400 | `unsupported_input_modality` |
| Unsupported chat feature such as `stream`, `tools`, or function calling | 400 | `feature_not_supported` |
| Unsupported or invalid `extra_body` field | 400 | `invalid_extra_body` |
| Missing, empty, or unextractable prompt | 400 | `invalid_prompt` |
| Prompt exceeds adapter/profile text bucket | 400 | `prompt_too_long` |
| Shape/profile mismatch | 400 | `profile_mismatch` |
| Queue full | 429 | `queue_full` |
| Queue wait exceeds `queue_timeout` | 429 | `queue_timeout` |
| Server is shutting down/draining | 503 | `engine_draining` |
| Worker recovering after request timeout | 503 | `engine_recovering` |
| Worker dead or engine unhealthy | 503 | `engine_unavailable` |
| Artifact store unavailable or misconfigured | 503 | `artifact_store_unavailable` |
| Artifact upload or presign failed after generation | 502 | `artifact_upload_failed` |
| Request exceeds `request_timeout` | 504 | `request_timeout` |

While the engine is recovering a worker after a timed-out request, new
generation requests return `503 engine_recovering`. `/ready` returns 503 during
this state. `/health` may remain 200 if the FastAPI process and recovery task are
alive; it should return 503 only when recovery fails, the worker is dead without
a restart path, or the engine marks itself unrecoverably unhealthy.

If generation succeeds but `await ArtifactStore.put_bytes(...)` or
`await ArtifactStore.get_url(ref)` fails, the handler must return one of the
artifact errors above, clean any request-local temporary data, and must not fall
back to `data_url`, local file URLs, raw filesystem paths, or inline bytes.

## Implementation Notes

- Chat response formatting belongs in `difflet/serving/openai/serving_chat.py`.
- Request/response Pydantic models belong in
  `difflet/serving/openai/protocol.py`.
- Model-specific input validation belongs in the serving model registry or
  adapter, not in the OpenAI route.
- The engine returns `DiffletGenerateOutput`; it does not know about
  OpenAI-style `choices` or R2 credentials.
- `ArtifactStore` owns local paths, file ids, TTL, and presigned URLs.
