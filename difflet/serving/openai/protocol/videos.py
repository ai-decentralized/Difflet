"""vLLM-Omni-style wire models for Difflet video generation.

The response intentionally carries the resolved media facts instead of inventing
an OpenAI-style ``seconds=\"4\"`` default.  Difflet profiles may represent a
non-integral duration (for example 49 frames at 24 fps), so ``seconds`` is only
echoed when the caller supplied and validated it while ``duration_s`` always
describes the actual output.
"""

from __future__ import annotations

import math
import mimetypes
import time
import uuid
from enum import Enum
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class VideoGenerationStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


SizeStr = Annotated[str, StringConstraints(pattern=r"^[1-9]\d*x[1-9]\d*$")]
SecondStr = Annotated[str, StringConstraints(pattern=r"^[1-9]\d*$")]


@lru_cache
def file_extension(media_type: str) -> str:
    """Return a stable extension for a recognized media type."""

    normalized = str(media_type).split(";", 1)[0].strip().lower()
    extension = mimetypes.guess_extension(normalized, strict=False)
    if extension is None:
        raise ValueError(f"No recognized file extension for media_type {media_type}")
    return extension.lstrip(".")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class VideoParams(_StrictModel):
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    num_frames: int | None = Field(default=None, ge=1)
    fps: int | None = Field(default=None, ge=1)

    @property
    def size(self) -> str | None:
        if self.width is None or self.height is None:
            return None
        return f"{self.width}x{self.height}"


class FileImageReference(_StrictModel):
    file_id: str


class UrlImageReference(_StrictModel):
    image_url: str


ImageReference = UrlImageReference | FileImageReference


class FileVideoReference(_StrictModel):
    file_id: str


class UrlVideoReference(_StrictModel):
    video_url: str


VideoReference = UrlVideoReference | FileVideoReference


class UrlAudioReference(_StrictModel):
    audio_url: str


AudioReference = UrlAudioReference


class VideoGenerationRequest(_StrictModel):
    """OpenAI-style request shape accepted through multipart normalization.

    Several upstream fields remain represented for schema compatibility but are
    deliberately rejected by Difflet's P0 transport whenever present.  The
    normalizer performs strict string parsing before constructing this model so
    Pydantic's permissive form-string coercion is never a security boundary.
    """

    model: str | None = None
    prompt: str = Field(min_length=1, max_length=32_768)
    seconds: SecondStr | None = None
    size: SizeStr | None = None
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    fps: int | None = Field(default=None, ge=1)
    num_frames: int | None = Field(default=None, ge=1)
    negative_prompt: str | None = Field(default=None, max_length=32_768)
    num_inference_steps: int | None = Field(default=None, ge=1, le=200)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    guidance_scale_2: float | None = Field(default=None, ge=0.0, le=20.0)
    boundary_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    flow_shift: float | None = None
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    user: str | None = Field(default=None, max_length=256)

    # Schema-compatible but unsupported in Difflet P0.  ``None`` defaults let
    # the multipart normalizer distinguish omission from an explicitly supplied
    # false/empty value before model construction.
    input_reference: Any | None = None
    image_reference: ImageReference | None = None
    video_reference: VideoReference | None = None
    audio_reference: AudioReference | None = None
    generate_sound: bool | None = None
    sound_duration: float | None = None
    true_cfg_scale: float | None = None
    enable_frame_interpolation: bool | None = None
    frame_interpolation_exp: int | None = None
    frame_interpolation_scale: float | None = None
    frame_interpolation_model_path: str | None = None
    lora: dict[str, Any] | None = None
    extra_params: dict[str, Any] | None = None
    video_params: VideoParams | None = None


class VideoAction(_StrictModel):
    data: list[Any]
    shape: list[int]
    dtype: str | None = None
    raw_action_dim: int | None = None
    action_mode: str | None = None
    domain_id: int | None = None


class VideoData(_StrictModel):
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None
    action: VideoAction | None = None


class VideoGenerationResponse(_StrictModel):
    created: int
    data: list[VideoData]
    stage_durations: dict[str, float] = Field(default_factory=dict)
    peak_memory_mb: float = Field(default=0.0, ge=0.0)


class VideoError(_StrictModel):
    code: int | str
    message: str
    error_type: str | None = None


class VideoResponse(_StrictModel):
    """Stored metadata for an asynchronous video generation job."""

    id: str = Field(default_factory=lambda: f"video_gen_{uuid.uuid4().hex}")
    object: Literal["video"] = "video"
    status: VideoGenerationStatus = VideoGenerationStatus.QUEUED
    model: str
    prompt: str
    size: SizeStr
    width: int = Field(ge=1, exclude=True)
    height: int = Field(ge=1, exclude=True)
    num_frames: int = Field(ge=1, exclude=True)
    fps: int = Field(ge=1, exclude=True)
    seconds: SecondStr | None = None
    duration_s: float = Field(gt=0.0, exclude=True)
    progress: int = Field(default=0, ge=0, le=100)
    quality: str = "default"
    created_at: int = Field(default_factory=lambda: int(time.time()), ge=0)
    completed_at: int | None = Field(default=None, ge=0)
    remixed_from_video_id: None = None
    error: VideoError | None = None
    # Difflet extension: short-lived direct-download URL when remote artifact
    # storage is enabled. OpenAI clients can continue using /content.
    url: str | None = None
    media_type: Literal["video/mp4"] = Field("video/mp4", exclude=True)
    expires_at: int | None = Field(default=None, ge=0)
    file_name: str | None = Field(default=None, exclude=True)
    file_size_bytes: int | None = Field(default=None, ge=0, exclude=True)
    inference_time_s: float | None = Field(default=None, ge=0.0, exclude=True)
    stage_durations: dict[str, float] = Field(default_factory=dict, exclude=True)
    peak_memory_mb: float = Field(default=0.0, ge=0.0, exclude=True)
    action: VideoAction | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _validate_resolved_media_facts(self) -> "VideoResponse":
        expected_size = f"{self.width}x{self.height}"
        if self.size != expected_size:
            raise ValueError(f"size must equal resolved dimensions {expected_size}")
        expected_duration = self.num_frames / self.fps
        if not math.isclose(self.duration_s, expected_duration, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("duration_s must equal num_frames / fps")
        if any(value < 0 or not math.isfinite(value) for value in self.stage_durations.values()):
            raise ValueError("stage_durations values must be finite and nonnegative")
        return self

    @property
    def file_extension(self) -> str:
        return file_extension(self.media_type)


class VideoDeleteResponse(_StrictModel):
    id: str
    deleted: bool
    object: Literal["video.deleted"] = "video.deleted"


class VideoListResponse(_StrictModel):
    first_id: str | None
    last_id: str | None
    has_more: bool
    data: list[VideoResponse]
    object: Literal["list"] = "list"
