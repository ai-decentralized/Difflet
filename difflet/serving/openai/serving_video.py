"""Transport normalization and response mapping for the Videos API.

The HTTP router and generation service intentionally live elsewhere.  This
module is CPU-only: it validates multipart fields against one immutable serving
profile and produces the existing resident-worker request contract without
importing any model or Trainium implementation.
"""

from __future__ import annotations

import math
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from difflet.serving.errors import (
    DiffletServingError,
    feature_not_supported,
    invalid_extra_body,
    invalid_prompt,
    profile_mismatch,
)
from difflet.serving.openai.protocol.videos import (
    VideoDeleteResponse,
    VideoError,
    VideoGenerationRequest,
    VideoGenerationStatus,
    VideoListResponse,
    VideoResponse,
)
from difflet.serving.types import DiffletGenerateRequest, VideoGenerateOptions

if TYPE_CHECKING:
    from difflet.serving.model_registry import ResolvedServingModel


_MAX_PROMPT_CHARACTERS = 32_768
_MAX_USER_CHARACTERS = 256
_MAX_SEED = 2**63 - 1
MAX_VIDEO_FORM_FIELDS = 32
MAX_VIDEO_FORM_PART_BYTES = 256 * 1024
MAX_VIDEO_FORM_BODY_BYTES = 1024 * 1024
_INTEGER_RE = re.compile(r"^\d+$")
_POSITIVE_INTEGER_RE = re.compile(r"^[1-9]\d*$")
_SIZE_RE = re.compile(r"^([1-9]\d*)x([1-9]\d*)$")

ALLOWED_VIDEO_FORM_FIELDS = frozenset(
    {
        "model",
        "prompt",
        "seconds",
        "size",
        "width",
        "height",
        "num_frames",
        "fps",
        "num_inference_steps",
        "guidance_scale",
        "guidance_scale_2",
        "boundary_ratio",
        "flow_shift",
        "negative_prompt",
        "seed",
        "user",
    }
)

UNSUPPORTED_VIDEO_FORM_FIELDS = frozenset(
    {
        "input_reference",
        "image_reference",
        "video_reference",
        "audio_reference",
        "generate_sound",
        "sound_duration",
        "true_cfg_scale",
        "enable_frame_interpolation",
        "frame_interpolation_exp",
        "frame_interpolation_scale",
        "frame_interpolation_model_path",
        "lora",
        "extra_params",
        "video_params",
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedVideoFacts:
    width: int
    height: int
    num_frames: int
    fps: int
    seconds: str | None
    duration_s: float

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"


async def normalize_video_multipart_request(
    raw_request: Any,
    *,
    resolved_model: "ResolvedServingModel",
    request_id: str | None = None,
) -> DiffletGenerateRequest:
    """Parse a FastAPI/Starlette multipart request without ``Form`` injection.

    Manual parsing keeps malformed multipart errors out of the application's
    JSON-specific ``RequestValidationError`` handler and also lets us reject
    duplicate fields and file uploads before Pydantic coercion.
    """

    content_type = str(getattr(raw_request, "headers", {}).get("content-type", ""))
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type != "multipart/form-data":
        raise DiffletServingError(
            400,
            "invalid_request",
            "video generation requires multipart/form-data",
        )
    try:
        form_request = await _bounded_form_request(raw_request)
        if getattr(form_request, "scope", None) is None:
            form = await form_request.form()
        else:
            form = await form_request.form(
                max_files=0,
                max_fields=MAX_VIDEO_FORM_FIELDS,
                max_part_size=MAX_VIDEO_FORM_PART_BYTES,
            )
    except DiffletServingError:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if "too many files" in message:
            raise feature_not_supported("video file uploads are not supported") from exc
        if any(marker in message for marker in ("too many fields", "part exceeded maximum size")):
            raise _request_too_large("video multipart form exceeds its configured limits") from exc
        raise DiffletServingError(400, "invalid_request", "request form is not valid") from exc
    return normalize_video_request(
        form,
        resolved_model=resolved_model,
        request_id=request_id,
    )


async def _bounded_form_request(raw_request: Any) -> Any:
    """Return a Request-like object whose body was read under the hard total cap."""

    headers = getattr(raw_request, "headers", {})
    raw_length = headers.get("content-length") if hasattr(headers, "get") else None
    if raw_length is not None:
        try:
            content_length = int(raw_length)
        except (TypeError, ValueError) as exc:
            raise DiffletServingError(
                400, "invalid_request", "content-length must be an integer"
            ) from exc
        if content_length < 0:
            raise DiffletServingError(400, "invalid_request", "content-length must be nonnegative")
        if content_length > MAX_VIDEO_FORM_BODY_BYTES:
            raise _request_too_large("video multipart body exceeds 1 MiB")

    stream = getattr(raw_request, "stream", None)
    scope = getattr(raw_request, "scope", None)
    if not callable(stream) or scope is None:
        # Unit-level request doubles may expose only ``form``.  Production ASGI
        # requests always take the bounded streaming path above.
        return raw_request

    body = bytearray()
    async for chunk in stream():
        body.extend(chunk)
        if len(body) > MAX_VIDEO_FORM_BODY_BYTES:
            raise _request_too_large("video multipart body exceeds 1 MiB")

    from starlette.requests import Request

    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": bytes(body), "more_body": False}

    return Request(scope, receive)


def _request_too_large(message: str) -> DiffletServingError:
    return DiffletServingError(413, "request_too_large", message)


def normalize_video_request(
    form: Any,
    *,
    resolved_model: "ResolvedServingModel",
    request_id: str | None = None,
) -> DiffletGenerateRequest:
    """Normalize one strict multipart form against an immutable video profile."""

    values = form_to_mapping(form)
    metadata = resolved_model.metadata
    profile = resolved_model.profile
    if metadata.output_modality != "video" or profile.output_modality != "video":
        raise feature_not_supported("the configured model does not expose video generation")

    prompt = _required_text(values, "prompt", maximum=_MAX_PROMPT_CHARACTERS)
    requested_model = _optional_text(values, "model")
    if requested_model is not None and requested_model != resolved_model.model_id:
        raise DiffletServingError(
            400,
            "model_not_served",
            "request model does not match server model",
        )

    profile_width = _positive_profile_int(profile.width, "width")
    profile_height = _positive_profile_int(profile.height, "height")
    profile_frames = _positive_profile_int(profile.num_frames, "num_frames")
    profile_fps = _positive_profile_int(profile.output_fps, "output_fps")

    size_width: int | None = None
    size_height: int | None = None
    if "size" in values:
        size_match = _SIZE_RE.fullmatch(values["size"])
        if size_match is None:
            raise invalid_extra_body("size must use WIDTHxHEIGHT with positive integers")
        size_width, size_height = (int(size_match.group(1)), int(size_match.group(2)))

    explicit_width = _optional_positive_int(values, "width")
    explicit_height = _optional_positive_int(values, "height")
    if size_width is not None and explicit_width is not None and size_width != explicit_width:
        raise profile_mismatch("size and width disagree")
    if size_height is not None and explicit_height is not None and size_height != explicit_height:
        raise profile_mismatch("size and height disagree")
    width = explicit_width if explicit_width is not None else size_width
    height = explicit_height if explicit_height is not None else size_height
    width = profile_width if width is None else width
    height = profile_height if height is None else height
    if (width, height) != (profile_width, profile_height):
        raise profile_mismatch("request width and height do not match serving profile")

    requested_fps = _optional_positive_int(values, "fps")
    fps = profile_fps if requested_fps is None else requested_fps
    if fps != profile_fps:
        raise profile_mismatch("request fps does not match serving profile")

    requested_seconds = _optional_seconds(values)
    explicit_frames = _optional_positive_int(values, "num_frames")
    derived_frames = int(requested_seconds) * fps if requested_seconds is not None else None
    if (
        explicit_frames is not None
        and derived_frames is not None
        and explicit_frames != derived_frames
    ):
        raise profile_mismatch("seconds, fps, and num_frames disagree")
    num_frames = (
        explicit_frames
        if explicit_frames is not None
        else derived_frames if derived_frames is not None else profile_frames
    )
    if num_frames != profile_frames:
        raise profile_mismatch("request num_frames does not match serving profile")

    steps = _optional_bounded_int(
        values,
        "num_inference_steps",
        minimum=1,
        maximum=200,
    )
    if steps is None:
        steps = _bounded_default_int(metadata.default_steps, "default_steps", 1, 200)

    guidance_scale = _optional_bounded_float(
        values,
        "guidance_scale",
        minimum=0.0,
        maximum=20.0,
    )
    if guidance_scale is None:
        guidance_scale = _bounded_default_float(
            metadata.default_guidance_scale,
            "default_guidance_scale",
            0.0,
            20.0,
        )
    guidance_scale_2 = _optional_bounded_float(
        values,
        "guidance_scale_2",
        minimum=0.0,
        maximum=20.0,
    )
    boundary_ratio = _optional_bounded_float(
        values,
        "boundary_ratio",
        minimum=0.0,
        maximum=1.0,
    )
    flow_shift = _optional_finite_float(values, "flow_shift")
    seed = _optional_bounded_int(values, "seed", minimum=0, maximum=_MAX_SEED)
    if seed is None:
        seed = 42

    negative_prompt = _optional_text(
        values,
        "negative_prompt",
        maximum=_MAX_PROMPT_CHARACTERS,
        empty_is_none=True,
    )
    user = _optional_text(
        values,
        "user",
        maximum=_MAX_USER_CHARACTERS,
        empty_is_none=True,
    )

    # Constructing the public schema after strict parsing provides one final
    # consistency check without allowing Pydantic to coerce arbitrary form data.
    VideoGenerationRequest(
        model=requested_model,
        prompt=prompt,
        seconds=requested_seconds,
        size=f"{width}x{height}",
        width=width,
        height=height,
        fps=fps,
        num_frames=num_frames,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        guidance_scale_2=guidance_scale_2,
        boundary_ratio=boundary_ratio,
        flow_shift=flow_shift,
        seed=seed,
        user=user,
    )

    return DiffletGenerateRequest(
        request_id=request_id or str(uuid.uuid4()),
        model=resolved_model.model_id,
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        output_format="mp4",
        video=VideoGenerateOptions(
            num_frames=num_frames,
            fps=fps,
            negative_prompt=negative_prompt,
            guidance_scale_2=guidance_scale_2,
            boundary_ratio=boundary_ratio,
            flow_shift=flow_shift,
            requested_seconds=requested_seconds,
            user=user,
        ),
    )


def resolved_video_facts(request: DiffletGenerateRequest) -> ResolvedVideoFacts:
    """Extract truthful, response-safe media facts from a normalized request."""

    if request.video is None:
        raise ValueError("video request options are required")
    fps = _positive_profile_int(request.video.fps, "fps")
    num_frames = _positive_profile_int(request.video.num_frames, "num_frames")
    width = _positive_profile_int(request.width, "width")
    height = _positive_profile_int(request.height, "height")
    return ResolvedVideoFacts(
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        seconds=request.video.requested_seconds,
        duration_s=num_frames / fps,
    )


def video_request_to_job_fields(request: DiffletGenerateRequest) -> dict[str, Any]:
    """Return the process-local metadata fields that define one normalized job."""

    facts = resolved_video_facts(request)
    assert request.video is not None
    return {
        "request_id": request.request_id,
        "model": request.model,
        "prompt": request.prompt,
        "width": facts.width,
        "height": facts.height,
        "num_frames": facts.num_frames,
        "fps": facts.fps,
        "seconds": facts.seconds,
        "duration_s": facts.duration_s,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "seed": request.seed,
        "negative_prompt": request.video.negative_prompt,
        "guidance_scale_2": request.video.guidance_scale_2,
        "boundary_ratio": request.video.boundary_ratio,
        "flow_shift": request.video.flow_shift,
        "user": request.video.user,
    }


def video_job_to_response(job: Any) -> VideoResponse:
    """Map a repository record (mapping or dataclass-like object) to the wire model.

    The mapper accepts a small amount of naming variation so the HTTP contract is
    decoupled from the in-memory record implementation. It never exposes a local
    artifact path: completed resources always advertise ``<id>.mp4``.
    """

    request = _read(job, "request", None)
    output = _read(job, "output", None)
    media_metadata = _read(job, "media_metadata", None)
    video_options = _read(request, "video", None)

    video_id = _required_job_value(job, ("id", "video_id"))
    status = VideoGenerationStatus(_required_job_value(job, ("status",)))
    model = _first_value(job, ("model",), fallback=_read(request, "model", None))
    prompt = _first_value(job, ("prompt",), fallback=_read(request, "prompt", None))
    if not isinstance(model, str) or not model:
        raise ValueError("video job is missing model")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("video job is missing prompt")

    width = _first_value(
        job,
        ("width",),
        fallback=_first_value(
            media_metadata,
            ("width",),
            fallback=_first_value(output, ("width",), fallback=_read(request, "width", None)),
        ),
    )
    height = _first_value(
        job,
        ("height",),
        fallback=_first_value(
            media_metadata,
            ("height",),
            fallback=_first_value(output, ("height",), fallback=_read(request, "height", None)),
        ),
    )
    num_frames = _first_value(
        job,
        ("num_frames",),
        fallback=_first_value(
            media_metadata,
            ("num_frames",),
            fallback=_first_value(
                output,
                ("num_frames",),
                fallback=_first_value(
                    request,
                    ("num_frames",),
                    fallback=_read(video_options, "num_frames", None),
                ),
            ),
        ),
    )
    fps = _first_value(
        job,
        ("fps",),
        fallback=_first_value(
            media_metadata,
            ("fps",),
            fallback=_first_value(
                output,
                ("fps",),
                fallback=_first_value(
                    request,
                    ("fps",),
                    fallback=_read(video_options, "fps", None),
                ),
            ),
        ),
    )
    width = _positive_media_int(width, "width")
    height = _positive_media_int(height, "height")
    num_frames = _positive_media_int(num_frames, "num_frames")
    fps = _positive_media_int(fps, "fps")

    seconds = _first_value(
        job,
        ("seconds", "requested_seconds"),
        fallback=_first_value(
            request,
            ("seconds", "requested_seconds"),
            fallback=_read(video_options, "requested_seconds", None),
        ),
    )
    if seconds is not None:
        seconds = str(seconds)

    created_at = _as_unix_seconds(_read(job, "created_at", int(time.time())))
    completed_at = _first_value(job, ("completed_at", "finished_at"), fallback=None)
    if completed_at is None and status in {
        VideoGenerationStatus.COMPLETED,
        VideoGenerationStatus.FAILED,
    }:
        completed_at = _read(job, "updated_at", created_at)
    if completed_at is not None:
        completed_at = _as_unix_seconds(completed_at)

    progress = _read(job, "progress", None)
    if progress is None:
        progress = 100 if status is VideoGenerationStatus.COMPLETED else 0

    error = _map_video_error(job)
    file_size_bytes = _first_value(
        job,
        ("artifact_size_bytes", "file_size_bytes", "size_bytes"),
        fallback=_read(output, "size_bytes", None),
    )
    stage_durations = dict(_read(job, "stage_durations", {}) or {})
    peak_memory_mb = float(_read(job, "peak_memory_mb", 0.0) or 0.0)
    inference_time_s = _first_value(
        job,
        ("inference_time_s",),
        fallback=_read(media_metadata, "inference_time_s", None),
    )
    expires_at = _read(job, "expires_at", None)
    artifact_url = _first_value(job, ("url",), fallback=_read(media_metadata, "url", None))
    raw_file_name = _first_value(
        job,
        ("file_name",),
        fallback=_read(media_metadata, "file_name", None),
    )
    if status is VideoGenerationStatus.COMPLETED:
        file_name = Path(str(raw_file_name)).name if raw_file_name else f"{video_id}.mp4"
    else:
        file_name = None

    return VideoResponse(
        id=str(video_id),
        status=status,
        model=model,
        prompt=prompt,
        size=f"{width}x{height}",
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        seconds=seconds,
        duration_s=num_frames / fps,
        progress=int(progress),
        quality=str(_read(job, "quality", "default")),
        created_at=created_at,
        completed_at=completed_at,
        error=error,
        url=str(artifact_url) if artifact_url else None,
        expires_at=_as_unix_seconds(expires_at) if expires_at is not None else None,
        file_name=file_name,
        file_size_bytes=int(file_size_bytes) if file_size_bytes is not None else None,
        inference_time_s=float(inference_time_s) if inference_time_s is not None else None,
        stage_durations={str(key): float(value) for key, value in stage_durations.items()},
        peak_memory_mb=peak_memory_mb,
        action=_read(job, "action", None),
    )


def video_jobs_to_list_response(
    jobs: list[Any] | tuple[Any, ...],
    *,
    has_more: bool,
) -> VideoListResponse:
    data = [video_job_to_response(job) for job in jobs]
    return VideoListResponse(
        first_id=data[0].id if data else None,
        last_id=data[-1].id if data else None,
        has_more=bool(has_more),
        data=data,
    )


def video_delete_response(video_id: str, *, deleted: bool = True) -> VideoDeleteResponse:
    return VideoDeleteResponse(id=video_id, deleted=deleted)


def form_to_mapping(form: Any) -> dict[str, str]:
    """Convert Starlette ``FormData`` or a mapping to unique text fields."""

    if hasattr(form, "multi_items"):
        items = list(form.multi_items())
    elif isinstance(form, Mapping):
        items = list(form.items())
    else:
        raise DiffletServingError(400, "invalid_request", "request form must be a mapping")

    values: dict[str, str] = {}
    for raw_key, raw_value in items:
        if not isinstance(raw_key, str):
            raise DiffletServingError(400, "invalid_request", "form field names must be strings")
        key = raw_key.strip()
        if key in values:
            raise DiffletServingError(400, "invalid_request", f"form field {key!r} is repeated")
        if key in UNSUPPORTED_VIDEO_FORM_FIELDS:
            raise feature_not_supported(f"video form field {key!r} is not supported")
        if key not in ALLOWED_VIDEO_FORM_FIELDS:
            raise DiffletServingError(
                400,
                "invalid_request",
                f"unknown video form field {key!r}",
            )
        if not isinstance(raw_value, str):
            raise feature_not_supported(
                f"video form field {key!r} must be a text value; file uploads are not supported"
            )
        values[key] = raw_value
    return values


def _required_text(values: Mapping[str, str], field: str, *, maximum: int) -> str:
    if field not in values:
        raise invalid_prompt(f"{field} is required")
    value = values[field].strip()
    if not value:
        raise invalid_prompt(f"{field} must not be empty")
    if len(value) > maximum:
        raise invalid_prompt(f"{field} must not exceed {maximum} characters")
    return value


def _optional_text(
    values: Mapping[str, str],
    field: str,
    *,
    maximum: int | None = None,
    empty_is_none: bool = False,
) -> str | None:
    if field not in values:
        return None
    value = values[field].strip()
    if not value and empty_is_none:
        return None
    if not value:
        raise invalid_extra_body(f"{field} must not be empty")
    if maximum is not None and len(value) > maximum:
        raise invalid_extra_body(f"{field} must not exceed {maximum} characters")
    return value


def _optional_seconds(values: Mapping[str, str]) -> str | None:
    if "seconds" not in values:
        return None
    value = values["seconds"].strip()
    if _POSITIVE_INTEGER_RE.fullmatch(value) is None:
        raise invalid_extra_body("seconds must be a positive integer string")
    return value


def _optional_positive_int(values: Mapping[str, str], field: str) -> int | None:
    return _optional_bounded_int(values, field, minimum=1, maximum=None)


def _optional_bounded_int(
    values: Mapping[str, str],
    field: str,
    *,
    minimum: int,
    maximum: int | None,
) -> int | None:
    if field not in values:
        return None
    raw = values[field].strip()
    if _INTEGER_RE.fullmatch(raw) is None:
        raise invalid_extra_body(f"{field} must be an integer")
    value = int(raw)
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            requirement = f">= {minimum}"
        else:
            requirement = f"between {minimum} and {maximum} inclusive"
        raise invalid_extra_body(f"{field} must be {requirement}")
    return value


def _optional_finite_float(values: Mapping[str, str], field: str) -> float | None:
    if field not in values:
        return None
    raw = values[field].strip()
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise invalid_extra_body(f"{field} must be a finite number") from exc
    if not math.isfinite(value):
        raise invalid_extra_body(f"{field} must be a finite number")
    return value


def _optional_bounded_float(
    values: Mapping[str, str],
    field: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    value = _optional_finite_float(values, field)
    if value is None:
        return None
    if value < minimum or value > maximum:
        raise invalid_extra_body(f"{field} must be between {minimum:g} and {maximum:g} inclusive")
    return value


def _positive_profile_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"video serving profile requires positive {field}")
    return int(value)


def _positive_media_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"video job requires positive integral {field}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        raise ValueError(f"video job requires positive integral {field}")
    return int(numeric)


def _bounded_default_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"video serving metadata has invalid {field}")
    return int(value)


def _bounded_default_float(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"video serving metadata has invalid {field}")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"video serving metadata has invalid {field}")
    return value


_MISSING = object()


def _read(value: Any, field: str, default: Any = _MISSING) -> Any:
    if value is None:
        if default is _MISSING:
            raise KeyError(field)
        return default
    if isinstance(value, Mapping):
        if field in value:
            return value[field]
    elif hasattr(value, field):
        return getattr(value, field)
    if default is _MISSING:
        raise KeyError(field)
    return default


def _first_value(value: Any, fields: tuple[str, ...], *, fallback: Any) -> Any:
    for field in fields:
        try:
            found = _read(value, field)
        except KeyError:
            continue
        if found is not None:
            return found
    return fallback


def _required_job_value(job: Any, fields: tuple[str, ...]) -> Any:
    value = _first_value(job, fields, fallback=_MISSING)
    if value is _MISSING:
        raise ValueError(f"video job is missing {'/'.join(fields)}")
    return value


def _map_video_error(job: Any) -> VideoError | None:
    raw = _read(job, "error", None)
    if isinstance(raw, VideoError):
        return raw
    if isinstance(raw, Mapping):
        code = raw.get("code", "generation_failed")
        message = raw.get("message")
        if message is not None:
            return VideoError(
                code=code,
                message=str(message),
                error_type=raw.get("error_type") or raw.get("type"),
            )
    if raw is not None:
        raw_message = _read(raw, "message", None)
        if raw_message is not None:
            return VideoError(
                code=_read(raw, "code", "generation_failed"),
                message=str(raw_message),
                error_type=_read(raw, "error_type", None),
            )
    code = _read(job, "error_code", None)
    message = _read(job, "error_message", None)
    if message is not None:
        return VideoError(
            code=code or "generation_failed",
            message=str(message),
            error_type=_read(job, "error_type", None),
        )
    if isinstance(raw, str) and raw:
        return VideoError(code=code or "generation_failed", message=raw)
    return None


def _as_unix_seconds(value: Any) -> int:
    if isinstance(value, datetime):
        return int(value.timestamp())
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timestamp must be an integer, float, or datetime")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError("timestamp must be finite and nonnegative")
    return int(value)
