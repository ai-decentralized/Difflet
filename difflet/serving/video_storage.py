"""Confined local-file storage for generated MP4 artifacts.

The parent process allocates every staging target.  A resident worker may only
write the pre-created ``*.part.mp4`` file; it never chooses a filename.  The
parent validates the same inode, fsyncs it, and atomically renames it into the
published directory after media validation succeeds.
"""

from __future__ import annotations

import logging
import math
import os
import re
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from difflet.serving.artifact_store import S3ArtifactStore

logger = logging.getLogger(__name__)

DEFAULT_MAX_VIDEO_ARTIFACT_BYTES = 8 * 1024**3
DEFAULT_STREAM_CHUNK_BYTES = 1024 * 1024

_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\Z")
_STAGING_SUFFIX = ".part.mp4"
_FINAL_SUFFIX = ".mp4"
_STAGING_TOKEN_LENGTH = 32
_MAX_VIDEO_ID_LENGTH = 255 - 1 - _STAGING_TOKEN_LENGTH - len(_STAGING_SUFFIX)


class VideoArtifactStorageError(RuntimeError):
    """Base class for local video artifact storage failures."""


class VideoArtifactPathViolation(VideoArtifactStorageError):
    """A path escaped the managed root or resolved through an unsafe entry."""


class VideoArtifactNotFound(VideoArtifactStorageError):
    """A requested committed artifact does not exist."""


class VideoArtifactTooLarge(VideoArtifactStorageError):
    """A staging or committed artifact exceeds the configured maximum size."""


class InvalidVideoArtifact(VideoArtifactStorageError):
    """A managed path is empty, non-regular, replaced, or otherwise invalid."""


@dataclass(frozen=True, slots=True)
class VideoArtifactTarget:
    video_id: str
    staging_path: Path
    artifact_key: str
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class VideoArtifact:
    key: str
    path: Path
    size_bytes: int
    modified_time_ns: int


@dataclass(frozen=True, slots=True)
class VideoArtifactSweep:
    staging_removed: tuple[Path, ...]
    artifacts_removed: tuple[str, ...]


class VideoArtifactLease:
    """An already-open descriptor that remains readable across unlink/delete.

    Starlette's ``FileResponse`` opens its path after the route returns, which
    races a concurrent DELETE.  Returning an open lease lets a streaming
    response consume this descriptor while DELETE safely unlinks the directory
    entry.  ``iter_chunks`` closes the descriptor when iteration ends or is
    cancelled.
    """

    def __init__(self, artifact: VideoArtifact, file: BinaryIO) -> None:
        self.artifact = artifact
        self.file = file
        self._closed = False

    def __enter__(self) -> "VideoArtifactLease":
        if self._closed:
            raise ValueError("video artifact lease is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def fileno(self) -> int:
        if self._closed:
            raise ValueError("video artifact lease is closed")
        return self.file.fileno()

    def close(self) -> None:
        if self._closed:
            return
        self.file.close()
        self._closed = True

    def iter_chunks(self, chunk_size: int = DEFAULT_STREAM_CHUNK_BYTES):
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if self._closed:
            raise ValueError("video artifact lease is closed")
        try:
            while True:
                chunk = self.file.read(chunk_size)
                if not chunk:
                    return
                yield chunk
        finally:
            self.close()


class LocalVideoArtifactStore:
    """Single-host staging and committed artifact store."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_artifact_bytes: int = DEFAULT_MAX_VIDEO_ARTIFACT_BYTES,
    ) -> None:
        if (
            isinstance(max_artifact_bytes, bool)
            or not isinstance(max_artifact_bytes, int)
            or max_artifact_bytes <= 0
        ):
            raise ValueError("max_artifact_bytes must be a positive integer")
        self.max_artifact_bytes = int(max_artifact_bytes)
        self._lock = threading.RLock()
        self._pending_artifact_deletes: dict[str, VideoArtifact] = {}
        requested_root = Path(root).expanduser()
        _mkdir_private_and_reject_symlink(requested_root)
        self.root = requested_root.resolve(strict=True)
        self.staging_root = self.root / "staging"
        self.artifact_root = self.root / "artifacts"
        _mkdir_private_and_reject_symlink(self.staging_root)
        _mkdir_private_and_reject_symlink(self.artifact_root)
        self.staging_root = self.staging_root.resolve(strict=True)
        self.artifact_root = self.artifact_root.resolve(strict=True)
        _require_direct_child(self.staging_root, self.root)
        _require_direct_child(self.artifact_root, self.root)

    def allocate_staging(self, video_id: str) -> VideoArtifactTarget:
        """Create and reserve a parent-owned writable ``*.part.mp4`` inode."""

        video_id = _validate_safe_name(video_id, "video id", suffix=None)
        artifact_key = f"{video_id}{_FINAL_SUFFIX}"
        with self._lock:
            self._ensure_managed_directories()
            final_path = self.artifact_root / artifact_key
            if _lexists(final_path):
                raise InvalidVideoArtifact(
                    f"committed video artifact already exists for {video_id!r}"
                )
            for _attempt in range(16):
                token = uuid.uuid4().hex
                staging_path = self.staging_root / f"{video_id}.{token}{_STAGING_SUFFIX}"
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                flags |= getattr(os, "O_NOFOLLOW", 0)
                try:
                    fd = os.open(staging_path, flags, 0o600)
                except FileExistsError:
                    continue
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode):
                        raise InvalidVideoArtifact("allocated staging target is not a regular file")
                    os.fsync(fd)
                finally:
                    os.close(fd)
                _fsync_directory(self.staging_root)
                return VideoArtifactTarget(
                    video_id=video_id,
                    staging_path=staging_path,
                    artifact_key=artifact_key,
                    device=int(info.st_dev),
                    inode=int(info.st_ino),
                )
        raise VideoArtifactStorageError("could not allocate a unique video staging target")

    def available_bytes(self) -> int:
        """Return currently available bytes on the filesystem owning the store."""

        with self._lock:
            self._ensure_managed_directories()
            stats = os.statvfs(self.root)
            return int(stats.f_bavail) * int(stats.f_frsize)

    def validate_staging(
        self,
        target: VideoArtifactTarget,
        *,
        reported_path: str | Path | None = None,
        expected_size_bytes: int | None = None,
    ) -> int:
        """Validate path confinement, inode identity, type, links, and size."""

        self._validate_target(target)
        if reported_path is not None and not _same_path(reported_path, target.staging_path):
            raise VideoArtifactPathViolation(
                "worker-reported path does not match the parent-owned staging target"
            )
        if expected_size_bytes is not None and (
            isinstance(expected_size_bytes, bool)
            or not isinstance(expected_size_bytes, int)
            or expected_size_bytes <= 0
        ):
            raise ValueError("expected_size_bytes must be a positive integer")
        with self._lock:
            self._ensure_managed_directories()
            info = _lstat_regular_file(target.staging_path, parent=self.staging_root)
            if (int(info.st_dev), int(info.st_ino)) != (target.device, target.inode):
                raise VideoArtifactPathViolation("video staging inode was replaced")
            if int(info.st_nlink) != 1:
                raise VideoArtifactPathViolation("video staging file must have exactly one link")
            size_bytes = int(info.st_size)
            if size_bytes <= 0:
                raise InvalidVideoArtifact("video staging file is empty")
            if size_bytes > self.max_artifact_bytes:
                raise VideoArtifactTooLarge(
                    f"video artifact has {size_bytes} bytes; maximum is "
                    f"{self.max_artifact_bytes}"
                )
            if expected_size_bytes is not None and size_bytes != expected_size_bytes:
                raise InvalidVideoArtifact(
                    "worker-reported video size does not match the staging file: "
                    f"reported={expected_size_bytes}, actual={size_bytes}"
                )
            return size_bytes

    def commit(
        self,
        target: VideoArtifactTarget,
        *,
        reported_path: str | Path | None = None,
        expected_size_bytes: int | None = None,
    ) -> VideoArtifact:
        """Fsync and atomically publish one validated staging artifact."""

        with self._lock:
            size_bytes = self.validate_staging(
                target,
                reported_path=reported_path,
                expected_size_bytes=expected_size_bytes,
            )
            final_path = self._artifact_path(target.artifact_key)
            if _lexists(final_path):
                raise InvalidVideoArtifact(
                    f"committed artifact {target.artifact_key!r} already exists"
                )

            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(target.staging_path, flags)
            try:
                info = os.fstat(fd)
                if (int(info.st_dev), int(info.st_ino)) != (target.device, target.inode):
                    raise VideoArtifactPathViolation("video staging inode changed before commit")
                if int(info.st_size) != size_bytes:
                    raise InvalidVideoArtifact("video staging size changed before commit")
                os.fsync(fd)
            finally:
                os.close(fd)

            os.replace(target.staging_path, final_path)
            try:
                os.chmod(final_path, 0o600, follow_symlinks=False)
                _fsync_directory(self.staging_root)
                _fsync_directory(self.artifact_root)
                final_info = _lstat_regular_file(final_path, parent=self.artifact_root)
                if (int(final_info.st_dev), int(final_info.st_ino)) != (
                    target.device,
                    target.inode,
                ):
                    raise VideoArtifactPathViolation("committed video inode does not match staging")
                final_size = self._validate_committed_size(final_info)
                if final_size != size_bytes:
                    raise InvalidVideoArtifact(
                        "video artifact size changed while it was being committed"
                    )
            except BaseException:
                _unlink_non_directory(final_path)
                _fsync_directory(self.artifact_root)
                raise
            return VideoArtifact(
                key=target.artifact_key,
                path=final_path,
                size_bytes=final_size,
                modified_time_ns=int(final_info.st_mtime_ns),
            )

    def get(self, artifact_key: str) -> VideoArtifact | None:
        path = self._artifact_path(artifact_key)
        with self._lock:
            self._ensure_managed_directories()
            if not _lexists(path):
                return None
            info = _lstat_regular_file(path, parent=self.artifact_root)
            size_bytes = self._validate_committed_size(info)
            return VideoArtifact(
                key=artifact_key,
                path=path,
                size_bytes=size_bytes,
                modified_time_ns=int(info.st_mtime_ns),
            )

    def require(self, artifact_key: str) -> VideoArtifact:
        artifact = self.get(artifact_key)
        if artifact is None:
            raise VideoArtifactNotFound(f"video artifact {artifact_key!r} was not found")
        return artifact

    def open(self, artifact_key: str) -> VideoArtifactLease:
        """Open and validate a committed file before returning a streaming lease."""

        path = self._artifact_path(artifact_key)
        with self._lock:
            self._ensure_managed_directories()
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                path_info = path.lstat()
                _validate_regular_stat(path_info, path)
                fd = os.open(path, flags)
            except FileNotFoundError as exc:
                raise VideoArtifactNotFound(
                    f"video artifact {artifact_key!r} was not found"
                ) from exc
            try:
                info = os.fstat(fd)
                _validate_regular_stat(info, path)
                if (int(info.st_dev), int(info.st_ino)) != (
                    int(path_info.st_dev),
                    int(path_info.st_ino),
                ):
                    raise VideoArtifactPathViolation("video artifact inode changed while opening")
                if int(info.st_nlink) < 1:
                    raise InvalidVideoArtifact("video artifact was unlinked before open")
                size_bytes = self._validate_committed_size(info)
                file = os.fdopen(fd, "rb", closefd=True)
                fd = -1
                artifact = VideoArtifact(
                    key=artifact_key,
                    path=path,
                    size_bytes=size_bytes,
                    modified_time_ns=int(info.st_mtime_ns),
                )
                return VideoArtifactLease(artifact, file)
            finally:
                if fd >= 0:
                    os.close(fd)

    def delete(self, artifact_key: str) -> bool:
        path = self._artifact_path(artifact_key)
        with self._lock:
            self._ensure_managed_directories()
            if not _lexists(path):
                self._pending_artifact_deletes.pop(artifact_key, None)
                return False
            _lstat_regular_file(path, parent=self.artifact_root)
            os.unlink(path)
            _fsync_directory(self.artifact_root)
            self._pending_artifact_deletes.pop(artifact_key, None)
            return True

    def schedule_artifact_delete(self, artifact: VideoArtifact) -> None:
        """Retain cleanup and accounting ownership for an unreachable artifact."""

        if not isinstance(artifact, VideoArtifact):
            raise TypeError("artifact must be a VideoArtifact")
        expected_path = self._artifact_path(artifact.key)
        if not _same_path(artifact.path, expected_path):
            raise VideoArtifactPathViolation(
                "scheduled video artifact path does not match its managed key"
            )
        if artifact.size_bytes <= 0 or artifact.size_bytes > self.max_artifact_bytes:
            raise InvalidVideoArtifact("scheduled video artifact size is invalid")
        with self._lock:
            existing = self._pending_artifact_deletes.get(artifact.key)
            if existing is not None and existing != artifact:
                raise InvalidVideoArtifact(
                    f"conflicting cleanup token for video artifact {artifact.key!r}"
                )
            self._pending_artifact_deletes[artifact.key] = artifact

    def retry_pending_artifact_deletes(self) -> tuple[VideoArtifact, ...]:
        """Retry local orphan deletion and return accounting tokens exactly once."""

        with self._lock:
            pending = tuple(self._pending_artifact_deletes.values())
        removed: list[VideoArtifact] = []
        for artifact in pending:
            try:
                self.delete(artifact.key)
            except Exception:
                logger.exception(
                    "video.local_pending_delete_failed artifact_key=%s",
                    artifact.key,
                )
                continue
            with self._lock:
                still_pending = artifact.key in self._pending_artifact_deletes
            if not still_pending:
                removed.append(artifact)
        return tuple(removed)

    def delete_staging(self, target: VideoArtifactTarget) -> bool:
        self._validate_target(target)
        with self._lock:
            self._ensure_managed_directories()
            if not _lexists(target.staging_path):
                return False
            info = _lstat_regular_file(target.staging_path, parent=self.staging_root)
            if (int(info.st_dev), int(info.st_ino)) != (target.device, target.inode):
                raise VideoArtifactPathViolation("refusing to delete replaced staging inode")
            os.unlink(target.staging_path)
            _fsync_directory(self.staging_root)
            return True

    def sweep_orphans(
        self,
        *,
        active_staging_paths: Iterable[str | Path] = (),
        referenced_artifact_keys: Iterable[str] | None = None,
        older_than_seconds: float = 0.0,
        now: float | None = None,
    ) -> VideoArtifactSweep:
        """Remove stale staging files and optionally unreferenced final files.

        Directory entries are inspected with ``lstat`` and never followed.
        Symlink entries matching the managed filename pattern are unlinked as
        unsafe orphans.  Directories and unrelated files are left untouched.
        """

        if isinstance(older_than_seconds, bool) or not isinstance(older_than_seconds, (int, float)):
            raise TypeError("older_than_seconds must be a nonnegative number")
        if float(older_than_seconds) < 0:
            raise ValueError("older_than_seconds must be nonnegative")
        current_time = time.time() if now is None else float(now)
        if not math.isfinite(current_time) or not math.isfinite(float(older_than_seconds)):
            raise ValueError("orphan sweep times must be finite")
        cutoff_ns = int((current_time - float(older_than_seconds)) * 1_000_000_000)
        active = {str(self._validated_staging_path(path)) for path in tuple(active_staging_paths)}
        referenced = (
            {
                _validate_safe_name(key, "artifact key", suffix=_FINAL_SUFFIX)
                for key in referenced_artifact_keys
            }
            if referenced_artifact_keys is not None
            else None
        )
        removed_staging: list[Path] = []
        removed_artifacts: list[str] = []

        with self._lock:
            self._ensure_managed_directories()
            for path in sorted(self.staging_root.iterdir(), key=lambda item: item.name):
                if not path.name.endswith(_STAGING_SUFFIX) or str(path) in active:
                    continue
                info = path.lstat()
                if int(info.st_mtime_ns) > cutoff_ns:
                    continue
                if stat.S_ISDIR(info.st_mode):
                    continue
                if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    os.unlink(path)
                    removed_staging.append(path)
            if removed_staging:
                _fsync_directory(self.staging_root)

            if referenced is not None:
                for path in sorted(self.artifact_root.iterdir(), key=lambda item: item.name):
                    if not path.name.endswith(_FINAL_SUFFIX) or path.name in referenced:
                        continue
                    info = path.lstat()
                    if int(info.st_mtime_ns) > cutoff_ns:
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        continue
                    if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                        os.unlink(path)
                        removed_artifacts.append(path.name)
                if removed_artifacts:
                    _fsync_directory(self.artifact_root)

        return VideoArtifactSweep(
            staging_removed=tuple(removed_staging),
            artifacts_removed=tuple(removed_artifacts),
        )

    def purge_all_managed(self) -> VideoArtifactSweep:
        """Remove every managed staging/final file regardless of timestamp.

        Callers must hold the service's exclusive media-root lease and fence all
        workers first. Unrelated entries and directories are deliberately left
        untouched.
        """

        removed_staging: list[Path] = []
        removed_artifacts: list[str] = []
        with self._lock:
            self._ensure_managed_directories()
            for path in sorted(self.staging_root.iterdir(), key=lambda item: item.name):
                if not path.name.endswith(_STAGING_SUFFIX):
                    continue
                info = path.lstat()
                if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    os.unlink(path)
                    removed_staging.append(path)
            if removed_staging:
                _fsync_directory(self.staging_root)

            for path in sorted(self.artifact_root.iterdir(), key=lambda item: item.name):
                if not path.name.endswith(_FINAL_SUFFIX):
                    continue
                info = path.lstat()
                if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    os.unlink(path)
                    removed_artifacts.append(path.name)
            if removed_artifacts:
                _fsync_directory(self.artifact_root)

        return VideoArtifactSweep(
            staging_removed=tuple(removed_staging),
            artifacts_removed=tuple(removed_artifacts),
        )

    def _validate_target(self, target: VideoArtifactTarget) -> None:
        if not isinstance(target, VideoArtifactTarget):
            raise TypeError("target must be a VideoArtifactTarget")
        _validate_safe_name(target.video_id, "video id", suffix=None)
        _validate_safe_name(target.artifact_key, "artifact key", suffix=_FINAL_SUFFIX)
        expected_key = f"{target.video_id}{_FINAL_SUFFIX}"
        if target.artifact_key != expected_key:
            raise VideoArtifactPathViolation("target artifact key does not match video id")
        path = self._validated_staging_path(target.staging_path)
        expected_prefix = f"{target.video_id}."
        if not path.name.startswith(expected_prefix) or not path.name.endswith(_STAGING_SUFFIX):
            raise VideoArtifactPathViolation("staging target name does not match video id")
        token = path.name[len(expected_prefix) : -len(_STAGING_SUFFIX)]
        if len(token) != _STAGING_TOKEN_LENGTH or any(
            character not in "0123456789abcdef" for character in token
        ):
            raise VideoArtifactPathViolation("staging target token is invalid")
        if target.device < 0 or target.inode <= 0:
            raise VideoArtifactPathViolation("staging target inode identity is invalid")

    def _validated_staging_path(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            raise VideoArtifactPathViolation("staging path must be absolute")
        if path.parent.resolve(strict=True) != self.staging_root:
            raise VideoArtifactPathViolation("staging path escapes the managed staging root")
        return path

    def _artifact_path(self, artifact_key: str) -> Path:
        key = _validate_safe_name(artifact_key, "artifact key", suffix=_FINAL_SUFFIX)
        path = self.artifact_root / key
        _require_direct_child(path, self.artifact_root, resolve_child=False)
        return path

    def _validate_committed_size(self, info: os.stat_result) -> int:
        if int(info.st_nlink) != 1:
            raise VideoArtifactPathViolation("committed video artifact must have exactly one link")
        size_bytes = int(info.st_size)
        if size_bytes <= 0:
            raise InvalidVideoArtifact("committed video artifact is empty")
        if size_bytes > self.max_artifact_bytes:
            raise VideoArtifactTooLarge(
                f"video artifact has {size_bytes} bytes; maximum is " f"{self.max_artifact_bytes}"
            )
        return size_bytes

    def _ensure_managed_directories(self) -> None:
        for path in (self.root, self.staging_root, self.artifact_root):
            try:
                info = path.lstat()
            except FileNotFoundError as exc:
                raise VideoArtifactPathViolation(
                    f"managed video directory disappeared: {path}"
                ) from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise VideoArtifactPathViolation(
                    f"managed video path is not a real directory: {path}"
                )


class S3VideoArtifactStore(LocalVideoArtifactStore):
    """Local source-of-truth store with best-effort S3 publication.

    The local committed copy is retained until normal DELETE/TTL cleanup so the
    existing inode-validated content lease remains safe. S3 supplies an optional
    direct-download copy and presigned URL; publication failure never invalidates
    a successfully committed local MP4.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        s3: S3ArtifactStore,
        retention_seconds: int = 25 * 60 * 60,
        max_artifact_bytes: int = DEFAULT_MAX_VIDEO_ARTIFACT_BYTES,
    ) -> None:
        super().__init__(root, max_artifact_bytes=max_artifact_bytes)
        self.s3 = s3
        self.retention_seconds = int(retention_seconds)
        self._urls: dict[str, str] = {}
        self._remote_keys: set[str] = set()
        self._pending_remote_deletes: set[str] = set()

    def commit(
        self,
        target: VideoArtifactTarget,
        *,
        reported_path: str | Path | None = None,
        expected_size_bytes: int | None = None,
    ) -> VideoArtifact:
        """Commit locally, then publish remotely for compatibility callers.

        The serving lifecycle uses :meth:`commit_local` and
        :meth:`publish_remote` separately so it can establish its terminal
        publication claim before entering an unbounded network operation.
        """

        artifact = self.commit_local(
            target,
            reported_path=reported_path,
            expected_size_bytes=expected_size_bytes,
        )
        self.publish_remote(artifact)
        return artifact

    def publish_remote(self, artifact: VideoArtifact) -> str | None:
        """Best-effort upload and signing for an already committed local MP4.

        Publication failure never removes or invalidates the local artifact.
        If rollback cannot confirm remote deletion, the key is retained in a
        store-owned retry set that does not depend on job metadata persistence.
        """

        if not isinstance(artifact, VideoArtifact):
            raise TypeError("artifact must be a VideoArtifact")
        current = self.require(artifact.key)
        if current != artifact:
            raise InvalidVideoArtifact(
                f"committed artifact {artifact.key!r} changed before S3 publication"
            )
        key = f"{self.s3.prefix}/{artifact.key}" if self.s3.prefix else artifact.key
        client = None
        try:
            client = self.s3._client()
            client.upload_file(
                str(artifact.path),
                self.s3.bucket,
                key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
            # Keep the requested download lifetime aligned with video retention.
            # S3/credential policy may shorten the effective lifetime.
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.s3.bucket, "Key": key},
                ExpiresIn=self.retention_seconds,
            )
        except Exception:
            logger.exception(
                "video.s3_publish_failed artifact_key=%s fallback=local",
                artifact.key,
            )
            with self._lock:
                self._urls.pop(artifact.key, None)
            if client is None:
                return None
            try:
                client.delete_object(Bucket=self.s3.bucket, Key=key)
            except Exception:
                # The upload may have completed before signing failed. Preserve a
                # conservative marker independent of job metadata so the
                # periodic sweeper can retry remote cleanup.
                with self._lock:
                    self._remote_keys.add(artifact.key)
                    self._pending_remote_deletes.add(artifact.key)
                logger.exception(
                    "video.s3_publish_rollback_failed artifact_key=%s",
                    artifact.key,
                )
            else:
                with self._lock:
                    self._remote_keys.discard(artifact.key)
                    self._pending_remote_deletes.discard(artifact.key)
            return None

        with self._lock:
            self._remote_keys.add(artifact.key)
            self._pending_remote_deletes.discard(artifact.key)
            self._urls[artifact.key] = url
        return url

    def commit_local(
        self,
        target: VideoArtifactTarget,
        *,
        reported_path: str | Path | None = None,
        expected_size_bytes: int | None = None,
    ) -> VideoArtifact:
        """Commit a synchronous result locally without publishing to S3."""
        return super().commit(
            target,
            reported_path=reported_path,
            expected_size_bytes=expected_size_bytes,
        )

    def url(self, artifact_key: str) -> str | None:
        with self._lock:
            return self._urls.get(artifact_key)

    def delete(self, artifact_key: str) -> bool:
        key = f"{self.s3.prefix}/{artifact_key}" if self.s3.prefix else artifact_key
        with self._lock:
            remote_may_exist = artifact_key in self._remote_keys
        if remote_may_exist:
            self.s3._client().delete_object(Bucket=self.s3.bucket, Key=key)
            with self._lock:
                self._remote_keys.discard(artifact_key)
                self._pending_remote_deletes.discard(artifact_key)
                self._urls.pop(artifact_key, None)
        deleted = super().delete(artifact_key)
        return deleted

    def retry_pending_remote_deletes(self) -> int:
        """Retry orphan cleanup that no longer belongs to a job transaction."""

        with self._lock:
            pending = tuple(sorted(self._pending_remote_deletes))
        removed = 0
        for artifact_key in pending:
            key = f"{self.s3.prefix}/{artifact_key}" if self.s3.prefix else artifact_key
            try:
                self.s3._client().delete_object(Bucket=self.s3.bucket, Key=key)
            except Exception:
                logger.exception(
                    "video.s3_pending_delete_failed artifact_key=%s",
                    artifact_key,
                )
                continue
            with self._lock:
                self._remote_keys.discard(artifact_key)
                self._pending_remote_deletes.discard(artifact_key)
                self._urls.pop(artifact_key, None)
            removed += 1
        return removed

    def purge_all_managed(self) -> VideoArtifactSweep:
        # Shutdown cannot enumerate arbitrary bucket contents, but it can make
        # one final attempt for remote keys conservatively retained by this
        # process after a failed publish rollback.
        self.retry_pending_remote_deletes()
        self.retry_pending_artifact_deletes()
        return super().purge_all_managed()


def _mkdir_private_and_reject_symlink(path: Path) -> None:
    if _lexists(path):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise VideoArtifactPathViolation(f"video storage path is not a real directory: {path}")
    else:
        path.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(path, 0o700, follow_symlinks=False)


def _lstat_regular_file(path: Path, *, parent: Path) -> os.stat_result:
    _require_direct_child(path, parent, resolve_child=False)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise VideoArtifactNotFound(f"video artifact path does not exist: {path}") from exc
    _validate_regular_stat(info, path)
    return info


def _validate_regular_stat(info: os.stat_result, path: Path) -> None:
    if stat.S_ISLNK(info.st_mode):
        raise VideoArtifactPathViolation(f"video artifact path is a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise InvalidVideoArtifact(f"video artifact path is not a regular file: {path}")


def _validate_safe_name(value: object, field_name: str, *, suffix: str | None) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise VideoArtifactPathViolation(f"{field_name} is not a safe filename")
    if ".." in value:
        raise VideoArtifactPathViolation(f"{field_name} contains an unsafe path segment")
    if suffix is None and len(value) > _MAX_VIDEO_ID_LENGTH:
        raise VideoArtifactPathViolation(f"{field_name} is too long")
    if suffix is not None and not value.endswith(suffix):
        raise VideoArtifactPathViolation(f"{field_name} must end with {suffix}")
    return value


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.abspath(os.fspath(left)) == os.path.abspath(os.fspath(right))


def _require_direct_child(
    child: Path,
    parent: Path,
    *,
    resolve_child: bool = True,
) -> None:
    parent_resolved = parent.resolve(strict=True)
    child_parent = (
        child.resolve(strict=True).parent if resolve_child else child.parent.resolve(strict=True)
    )
    if child_parent != parent_resolved:
        raise VideoArtifactPathViolation(f"path {child} escapes managed root {parent}")


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _unlink_non_directory(path: Path) -> None:
    if not _lexists(path):
        return
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        os.unlink(path)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = [
    "DEFAULT_MAX_VIDEO_ARTIFACT_BYTES",
    "DEFAULT_STREAM_CHUNK_BYTES",
    "InvalidVideoArtifact",
    "LocalVideoArtifactStore",
    "S3VideoArtifactStore",
    "VideoArtifact",
    "VideoArtifactLease",
    "VideoArtifactNotFound",
    "VideoArtifactPathViolation",
    "VideoArtifactStorageError",
    "VideoArtifactSweep",
    "VideoArtifactTarget",
    "VideoArtifactTooLarge",
]
