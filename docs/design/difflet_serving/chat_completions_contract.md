# Difflet Chat Completions Contract

Date: 2026-07-07

This document defines the first serving API contract for Difflet. The public
entrypoint is OpenAI-style `POST /v1/chat/completions`; image/video-specific
routes are not part of the first milestone.

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
- If `model` is provided, it must match the startup model id or an alias
  registered for the same serving profile.
- A mismatch returns `400 Bad Request`.
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
- If no non-empty prompt can be extracted, return `400 Bad Request`.
- After model-specific prompt templating/tokenization, the prompt must fit the
  selected adapter/profile text bucket. P0 must not silently truncate prompts.
  If the tokenized prompt exceeds the profile limit, return
  `400 prompt_too_long`.

Supported top-level fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `model` | No for single-model server | Requested model id or alias. |
| `messages` | Yes | Chat messages. Last user text becomes the prompt. |
| `modalities` | No | Optional desired output modalities. Omitted means the served model's default output modality. |
| `extra_body` | No | Difflet runtime parameters. |

P0 top-level field handling:

- Allowed fields: `model`, `messages`, `modalities`, and `extra_body`.
- Explicitly rejected fields: `stream`, `tools`, `tool_choice`, `functions`,
  `function_call`, `response_format`, `temperature`, `top_p`, `n`, `stop`,
  `max_tokens`, `metadata`, and any field that implies text generation,
  streaming, tool use, or other unsupported chat behavior.
- Unknown top-level fields return `400 feature_not_supported` unless they are
  added to this allowlist in a later revision.
- Difflet generation fields must live inside `extra_body`. P0 should not accept
  flattened generation fields at the top level. If the same generation field is
  supplied both top-level and inside `extra_body`, return
  `400 invalid_extra_body` rather than choosing one silently.

`modalities` compatibility:

- OpenAI chat `modalities` is primarily a text/audio output selector. Difflet
  uses the same top-level field as a serving extension for generated media.
- For P0 text-to-image, omit `modalities` or pass `["image"]`.
- `["video"]`, `["text"]`, `["audio"]`, or mixed unsupported modalities return
  `400 unsupported_modality` unless a future model adapter explicitly supports
  them.

## `extra_body` Parameters

The serving handler normalizes these fields into `DiffletGenerateRequest`.

Common generation fields:

| Field | Type | Applies to | Meaning |
| --- | --- | --- | --- |
| `height` | int | image | Requested output height. Must match server `ServingProfile` in P0. |
| `width` | int | image | Requested output width. Must match server `ServingProfile` in P0. |
| `num_frames` | int | future video | Requested frame count. P0 rejects video requests. |
| `num_inference_steps` | int | image | Preferred request name for denoising steps. |
| `steps` | int | image | Alias for `num_inference_steps`; reject if both disagree. |
| `guidance_scale` | float | image | Classifier-free guidance or model-specific guidance scale. |
| `true_cfg_scale` | float | model-specific | True-CFG guidance scale only for models that explicitly expose a true two-pass CFG path. Qwen/Flux MVP adapters reject it. |
| `seed` | int | image | Random seed. Default comes from the model adapter. |
| `negative_prompt` | string | model-specific | Negative prompt if the model adapter supports it. |
| `output_format` | string | image | Optional output format, e.g. `png`; defaults to the model's configured output format. Selected model adapter validates support and MIME mapping. |

Serving/runtime fields:

| Field | Type | Applies to | Meaning |
| --- | --- | --- | --- |
| `response_format` | string | image | `url` or `auto`. MVP resolves `auto` to `url` and rejects `data_url`. |
| `artifact_ttl_seconds` | int | image | Optional expiration hint for artifact-backed outputs. Defaults to server `--artifact-ttl-seconds`. |

TeaCache and advanced runtime fields may pass through `extra_body` only if the
selected model adapter declares support. Unknown model-specific fields should
return `400` by default in P0; silently ignoring generation knobs makes
benchmarking and debugging unreliable.

`extra_body` validation rules:

- Unknown fields return `400 invalid_extra_body`.
- Alias pairs are deterministic. For P0, `steps` is an alias for
  `num_inference_steps`; if both are present and values differ, return
  `400 invalid_extra_body`. If both are equal, normalize to
  `num_inference_steps`.
- `height`, `width`, `tp_degree`, `cp_degree`, `cp_mode`, and `cfg_parallel`
  are profile-bound. Values must match exactly one loaded serving profile.
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
| `artifact_ttl_seconds` | Positive integer no greater than server `--max-artifact-ttl-seconds`. |
| `output_format` | `png` only for Qwen P0. |
| `response_format` | `url` or `auto`; `auto` resolves to `url`. |

Profile-bound fields:

- `height`
- `width`
- `num_frames`
- `tp_degree`
- `cp_degree`
- `cp_mode`
- `cfg_parallel`

The first serving implementation fixes these at startup through
`ServingProfile`. If a request provides a different value, return
`400 Bad Request` with a profile mismatch error.

P0 loads exactly one `ServingProfile`. Future multi-profile serving may match a
request against several configured loaded profiles, but that is not part of the
first contract. P0 must reject multi-profile startup flags and must not compile,
load, or switch profiles because of request fields.

Per-model input differences:

| Model type | Output | Relevant request fields | Notes |
| --- | --- | --- | --- |
| `qwen_image` | image | `height`, `width`, `num_inference_steps`, `guidance_scale`, `seed` | P0 target. `height` and `width` must match the startup profile. Reject `true_cfg_scale`; Qwen uses guidance-distilled single-pass guidance. |
| `flux` | image | `height`, `width`, `num_inference_steps`, `guidance_scale`, `seed` | MVP target. Single-pipeline image model. No `num_frames`. |
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
expose arbitrary local filesystem paths. `response_format=data_url` is not part
of P0 and should return `400 invalid_extra_body`.

Engine/output boundary:

- `DiffletServingEngine.generate(...)` returns bytes plus MIME metadata.
- `difflet/serving/openai/serving_chat.py` applies `response_format`.
- For `response_format=url`, the handler calls `ArtifactStore.put_bytes(...)`
  and returns the resulting URL in `image_url.url`.
- For `response_format=auto`, the handler treats it as `url`.
- The handler computes the artifact TTL from
  `extra_body.artifact_ttl_seconds` or the server default and passes it to
  `ArtifactStore`.

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

Serving model resolution should reuse `difflet.registry.resolve_model(...)` for
model id matching, default shape, default parallel config, backend support, and
download patterns. The serving registry only adds topology/runtime-plan metadata
for model types that are explicitly enabled for serving.

MVP is text-to-image only. Qwen-Image and Flux are both MVP serving targets and
use the same image response contract. Each server process serves one model and
one output modality.

| Model id | Model type | Default output | Chat content type | Min cores for full resident | P0 serving status | Notes |
| --- | --- | --- | --- | ---: | --- | --- |
| `Qwen/Qwen-Image` | `qwen_image` | image/png | `image_url` | `tp*cp + tp*cp + 1` for per-stage; `max(stage_cores)` for shared-process | P0 target | Shared-process 3-stage `prompt_encoder -> denoiser -> decoder`; CFG-parallel disabled. |
| `black-forest-labs/FLUX.1-dev` | `flux` | image/png | `image_url` | `tp*cp` | P0 target | Single pipeline path through `DiffletPipeline`; CFG-parallel disabled. |
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
| Prompt exceeds adapter/profile text bucket | 400 | `prompt_too_long` |
| Shape/profile mismatch | 400 | `profile_mismatch` |
| Queue full | 429 | `queue_full` |
| Queue wait exceeds `queue_timeout` | 429 | `queue_timeout` |
| Server is shutting down/draining | 503 | `engine_draining` |
| Worker dead or engine unhealthy | 503 | `engine_unavailable` |
| Artifact store unavailable or misconfigured | 503 | `artifact_store_unavailable` |
| Artifact upload or presign failed after generation | 502 | `artifact_upload_failed` |
| Request exceeds `request_timeout` | 504 | `request_timeout` |

If generation succeeds but `ArtifactStore.put_bytes(...)` or URL generation
fails, the handler must return one of the artifact errors above, clean any
request-local temporary data, and must not fall back to `data_url`, local file
URLs, raw filesystem paths, or inline bytes.

## Implementation Notes

- Chat response formatting belongs in `difflet/serving/openai/serving_chat.py`.
- Request/response Pydantic models belong in
  `difflet/serving/openai/protocol.py`.
- Model-specific input validation belongs in the serving model registry or
  adapter, not in the OpenAI route.
- The engine returns `DiffletGenerateOutput`; it does not know about
  OpenAI-style `choices` or R2 credentials.
- `ArtifactStore` owns local paths, file ids, TTL, and presigned URLs.
