"""Process-local metadata repository for video generation jobs.

Job metadata intentionally lives only for the lifetime of the serving process.
Video files are managed separately by :mod:`difflet.serving.video_storage`.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

VideoJobStatus = Literal["queued", "in_progress", "completed", "failed"]

ACTIVE_VIDEO_JOB_STATUSES: frozenset[VideoJobStatus] = frozenset({"queued", "in_progress"})
TERMINAL_VIDEO_JOB_STATUSES: frozenset[VideoJobStatus] = frozenset({"completed", "failed"})
DEFAULT_VIDEO_JOB_RETENTION_SECONDS = 25 * 60 * 60

_ALL_VIDEO_JOB_STATUSES: frozenset[str] = frozenset(
    ACTIVE_VIDEO_JOB_STATUSES | TERMINAL_VIDEO_JOB_STATUSES
)
_ALLOWED_TRANSITIONS: dict[VideoJobStatus, frozenset[VideoJobStatus]] = {
    "queued": frozenset({"in_progress", "failed"}),
    "in_progress": frozenset({"completed", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
}


class VideoJobRepositoryError(RuntimeError):
    """Base error for repository contract violations."""


class VideoJobNotFound(VideoJobRepositoryError):
    """The requested video job does not exist."""


class VideoJobStateConflict(VideoJobRepositoryError):
    """A compare-and-set expected a different current state/version."""


class InvalidVideoJobTransition(VideoJobRepositoryError):
    """The requested public state transition is not allowed."""


@dataclass(frozen=True, slots=True)
class VideoJobError:
    code: str
    message: str
    error_type: str = "server_error"

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("video job error code must not be empty")
        if not self.message:
            raise ValueError("video job error message must not be empty")
        if not self.error_type:
            raise ValueError("video job error type must not be empty")


@dataclass(frozen=True, slots=True)
class VideoJobCreate:
    id: str
    model: str
    prompt: str
    request: Mapping[str, Any] = field(default_factory=dict)
    user: str | None = None
    created_at: int | None = None

    def __post_init__(self) -> None:
        _validate_text_identifier(self.id, "video job id")
        _validate_text_identifier(self.model, "video job model")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("video job prompt must be a non-empty string")
        if not isinstance(self.request, Mapping):
            raise TypeError("video job request must be a mapping")
        if self.user is not None and not isinstance(self.user, str):
            raise TypeError("video job user must be a string or None")
        if self.created_at is not None and (
            isinstance(self.created_at, bool) or not isinstance(self.created_at, int)
        ):
            raise TypeError("video job created_at must be an integer epoch timestamp")
        if self.created_at is not None and self.created_at < 0:
            raise ValueError("video job created_at must be nonnegative")


@dataclass(frozen=True, slots=True)
class VideoJob:
    id: str
    model: str
    prompt: str
    status: VideoJobStatus
    request: dict[str, Any]
    user: str | None
    created_at: int
    updated_at: int
    started_at: int | None
    completed_at: int | None
    expires_at: int | None
    artifact_key: str | None
    artifact_size_bytes: int | None
    media_metadata: dict[str, Any] | None
    error: VideoJobError | None
    version: int


@dataclass(frozen=True, slots=True)
class VideoJobPage:
    data: tuple[VideoJob, ...]
    has_more: bool

    @property
    def first_id(self) -> str | None:
        return self.data[0].id if self.data else None

    @property
    def last_id(self) -> str | None:
        return self.data[-1].id if self.data else None

    @property
    def next_cursor(self) -> str | None:
        """Compatibility helper; the public API exposes ``last_id`` instead."""

        return self.last_id if self.has_more else None


def new_video_job_id() -> str:
    """Return a server-owned, URL-safe public video job identifier."""

    return f"video_gen_{uuid.uuid4().hex}"


class InMemoryVideoJobRepository:
    """Thread-safe, process-local repository with transition-checked CAS writes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._closed = False
        self._jobs: dict[str, VideoJob] = {}

    def __enter__(self) -> "InMemoryVideoJobRepository":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Clear all process-local jobs and permanently close this repository."""

        with self._lock:
            if self._closed:
                return
            self._jobs.clear()
            self._closed = True

    def clear(self) -> None:
        """Remove every job while keeping the repository open."""

        with self._lock:
            self._ensure_open()
            self._jobs.clear()

    def create(self, create: VideoJobCreate) -> VideoJob:
        if not isinstance(create, VideoJobCreate):
            raise TypeError("create must be a VideoJobCreate")
        now = int(time.time()) if create.created_at is None else int(create.created_at)
        request = _copy_json_mapping(create.request, field_name="request")
        with self._lock:
            self._ensure_open()
            if create.id in self._jobs:
                raise VideoJobStateConflict(f"video job {create.id!r} already exists")
            job = VideoJob(
                id=create.id,
                model=create.model,
                prompt=create.prompt,
                status="queued",
                request=request,
                user=create.user,
                created_at=now,
                updated_at=now,
                started_at=None,
                completed_at=None,
                expires_at=None,
                artifact_key=None,
                artifact_size_bytes=None,
                media_metadata=None,
                error=None,
                version=1,
            )
            self._jobs[job.id] = job
            return _copy_job(job)

    def get(self, video_id: str) -> VideoJob | None:
        _validate_text_identifier(video_id, "video job id")
        with self._lock:
            self._ensure_open()
            job = self._jobs.get(video_id)
            return _copy_job(job) if job is not None else None

    def require(self, video_id: str) -> VideoJob:
        job = self.get(video_id)
        if job is None:
            raise VideoJobNotFound(f"video job {video_id!r} was not found")
        return job

    def compare_and_set(
        self,
        video_id: str,
        expected_statuses: VideoJobStatus | Iterable[VideoJobStatus],
        new_status: VideoJobStatus,
        *,
        artifact_key: str | None = None,
        artifact_size_bytes: int | None = None,
        media_metadata: Mapping[str, Any] | None = None,
        error: VideoJobError | None = None,
        expires_at: int | None = None,
        now: int | None = None,
    ) -> VideoJob:
        """Transition one job iff its current state matches the expectation."""

        _validate_text_identifier(video_id, "video job id")
        expected = _normalize_statuses(expected_statuses)
        new_status = _validate_status(new_status)
        timestamp = int(time.time()) if now is None else _validate_timestamp(now, "now")

        if new_status == "completed":
            _validate_text_identifier(artifact_key, "artifact key")
            if (
                isinstance(artifact_size_bytes, bool)
                or not isinstance(artifact_size_bytes, int)
                or artifact_size_bytes <= 0
            ):
                raise ValueError("completed video job requires positive artifact_size_bytes")
            if not isinstance(media_metadata, Mapping) or not media_metadata:
                raise ValueError("completed video job requires media_metadata")
            if error is not None:
                raise ValueError("completed video job must not contain an error")
        elif new_status == "failed":
            if not isinstance(error, VideoJobError):
                raise ValueError("failed video job requires a VideoJobError")
            if any(
                value is not None for value in (artifact_key, artifact_size_bytes, media_metadata)
            ):
                raise ValueError("failed video job must not publish artifact/media data")
        elif any(
            value is not None
            for value in (artifact_key, artifact_size_bytes, media_metadata, error)
        ):
            raise ValueError(f"{new_status} video job cannot publish terminal data")
        if new_status in TERMINAL_VIDEO_JOB_STATUSES:
            if expires_at is None:
                expires_at = timestamp + DEFAULT_VIDEO_JOB_RETENTION_SECONDS
            expires_at = _validate_timestamp(expires_at, "expires_at")
            if expires_at <= timestamp:
                raise ValueError("terminal video job expires_at must be after completion")
        elif expires_at is not None:
            raise ValueError("active video job cannot define expires_at")

        copied_media = (
            _copy_json_mapping(media_metadata, field_name="media_metadata")
            if media_metadata is not None
            else None
        )
        with self._lock:
            self._ensure_open()
            current = self._jobs.get(video_id)
            if current is None:
                raise VideoJobNotFound(f"video job {video_id!r} was not found")
            if current.status not in expected:
                raise VideoJobStateConflict(
                    f"video job {video_id!r} is {current.status!r}, "
                    f"expected {sorted(expected)!r}"
                )
            if new_status not in _ALLOWED_TRANSITIONS[current.status]:
                raise InvalidVideoJobTransition(
                    f"video job cannot transition from {current.status!r} to {new_status!r}"
                )

            updated = replace(
                current,
                status=new_status,
                updated_at=timestamp,
                started_at=(timestamp if new_status == "in_progress" else current.started_at),
                completed_at=(timestamp if new_status in TERMINAL_VIDEO_JOB_STATUSES else None),
                expires_at=expires_at,
                artifact_key=artifact_key,
                artifact_size_bytes=artifact_size_bytes,
                media_metadata=copied_media,
                error=error,
                version=current.version + 1,
            )
            self._jobs[video_id] = updated
            return _copy_job(updated)

    def mark_in_progress(
        self,
        video_id: str,
        *,
        expected_statuses: VideoJobStatus | Iterable[VideoJobStatus] = "queued",
        now: int | None = None,
    ) -> VideoJob:
        return self.compare_and_set(
            video_id,
            expected_statuses,
            "in_progress",
            now=now,
        )

    def mark_completed(
        self,
        video_id: str,
        *,
        artifact_key: str,
        artifact_size_bytes: int,
        media_metadata: Mapping[str, Any],
        expires_at: int | None = None,
        expected_statuses: VideoJobStatus | Iterable[VideoJobStatus] = "in_progress",
        now: int | None = None,
    ) -> VideoJob:
        return self.compare_and_set(
            video_id,
            expected_statuses,
            "completed",
            artifact_key=artifact_key,
            artifact_size_bytes=artifact_size_bytes,
            media_metadata=media_metadata,
            expires_at=expires_at,
            now=now,
        )

    def mark_failed(
        self,
        video_id: str,
        *,
        error: VideoJobError,
        expires_at: int | None = None,
        expected_statuses: VideoJobStatus | Iterable[VideoJobStatus] = (
            "queued",
            "in_progress",
        ),
        now: int | None = None,
    ) -> VideoJob:
        return self.compare_and_set(
            video_id,
            expected_statuses,
            "failed",
            error=error,
            expires_at=expires_at,
            now=now,
        )

    def list(
        self,
        *,
        limit: int = 20,
        after: str | None = None,
        status: VideoJobStatus | None = None,
    ) -> VideoJobPage:
        """List jobs in stable ``created_at DESC, id DESC`` order."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("video job list limit must satisfy 1 <= limit <= 100")
        selected_status = _validate_status(status) if status is not None else None
        with self._lock:
            self._ensure_open()
            jobs = sorted(
                self._jobs.values(),
                key=lambda job: (job.created_at, job.id),
                reverse=True,
            )
            if after is not None:
                after_id = _validate_text_identifier(after, "after video job id")
                anchor = self._jobs.get(after_id)
                if anchor is None:
                    return VideoJobPage(data=(), has_more=False)
                anchor_key = (anchor.created_at, anchor.id)
                jobs = [job for job in jobs if (job.created_at, job.id) < anchor_key]
            if selected_status is not None:
                jobs = [job for job in jobs if job.status == selected_status]
            has_more = len(jobs) > limit
            return VideoJobPage(
                data=tuple(_copy_job(job) for job in jobs[:limit]),
                has_more=has_more,
            )

    def delete(
        self,
        video_id: str,
        *,
        expected_statuses: VideoJobStatus | Iterable[VideoJobStatus] | None = None,
    ) -> VideoJob | None:
        """Physically remove metadata; deletion is intentionally not a state."""

        _validate_text_identifier(video_id, "video job id")
        expected = _normalize_statuses(expected_statuses) if expected_statuses is not None else None
        with self._lock:
            self._ensure_open()
            job = self._jobs.get(video_id)
            if job is None:
                return None
            if expected is not None and job.status not in expected:
                raise VideoJobStateConflict(
                    f"video job {video_id!r} is {job.status!r}, " f"expected {sorted(expected)!r}"
                )
            del self._jobs[video_id]
            return _copy_job(job)

    def count(self, *, status: VideoJobStatus | None = None) -> int:
        selected_status = _validate_status(status) if status is not None else None
        with self._lock:
            self._ensure_open()
            if selected_status is None:
                return len(self._jobs)
            return sum(job.status == selected_status for job in self._jobs.values())

    def list_expired(self, *, now: int | None = None) -> tuple[VideoJob, ...]:
        timestamp = int(time.time()) if now is None else _validate_timestamp(now, "now")
        with self._lock:
            self._ensure_open()
            return tuple(
                _copy_job(job)
                for job in self._jobs.values()
                if job.expires_at is not None and job.expires_at <= timestamp
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise VideoJobRepositoryError("video job repository is closed")


def _copy_job(job: VideoJob) -> VideoJob:
    return replace(
        job,
        request=_copy_json_mapping(job.request, field_name="request"),
        media_metadata=(
            _copy_json_mapping(job.media_metadata, field_name="media_metadata")
            if job.media_metadata is not None
            else None
        ),
    )


def _validate_status(value: str) -> VideoJobStatus:
    if value not in _ALL_VIDEO_JOB_STATUSES:
        raise ValueError(f"unsupported video job status {value!r}")
    return value  # type: ignore[return-value]


def _normalize_statuses(
    statuses: VideoJobStatus | Iterable[VideoJobStatus],
) -> frozenset[VideoJobStatus]:
    raw = (statuses,) if isinstance(statuses, str) else tuple(statuses)
    if not raw:
        raise ValueError("expected_statuses must not be empty")
    return frozenset(_validate_status(status) for status in raw)


def _validate_text_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(value) > 512 or "\x00" in value:
        raise ValueError(f"{field_name} is invalid")
    return value


def _validate_timestamp(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer epoch timestamp")
    return int(value)


def _copy_json_mapping(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"video job {field_name} must be JSON serializable") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError(f"video job {field_name} must be a JSON object")
    return decoded


__all__ = [
    "ACTIVE_VIDEO_JOB_STATUSES",
    "InMemoryVideoJobRepository",
    "InvalidVideoJobTransition",
    "TERMINAL_VIDEO_JOB_STATUSES",
    "VideoJob",
    "VideoJobCreate",
    "VideoJobError",
    "VideoJobNotFound",
    "VideoJobPage",
    "VideoJobRepositoryError",
    "VideoJobStateConflict",
    "VideoJobStatus",
    "new_video_job_id",
]
