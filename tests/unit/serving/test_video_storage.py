from __future__ import annotations

import os
import stat
from dataclasses import replace

import pytest

import difflet.serving.video_storage as video_storage
from difflet.serving.video_storage import (
    InvalidVideoArtifact,
    LocalVideoArtifactStore,
    S3VideoArtifactStore,
    VideoArtifactNotFound,
    VideoArtifactPathViolation,
    VideoArtifactTooLarge,
)


class _FakeS3Client:
    def __init__(
        self,
        *,
        upload_error: Exception | None = None,
        presign_error: Exception | None = None,
        delete_errors: list[Exception] | None = None,
    ) -> None:
        self.upload_error = upload_error
        self.presign_error = presign_error
        self.delete_errors = list(delete_errors or [])
        self.uploads: list[tuple[str, str, str, dict[str, str]]] = []
        self.deletes: list[tuple[str, str]] = []

    def upload_file(self, filename, bucket, key, *, ExtraArgs):
        self.uploads.append((filename, bucket, key, dict(ExtraArgs)))
        if self.upload_error is not None:
            raise self.upload_error

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        if self.presign_error is not None:
            raise self.presign_error
        assert operation == "get_object"
        return f"https://example.test/{Params['Bucket']}/{Params['Key']}?ttl={ExpiresIn}"

    def delete_object(self, *, Bucket, Key):
        self.deletes.append((Bucket, Key))
        if self.delete_errors:
            raise self.delete_errors.pop(0)


class _FakeS3Store:
    bucket = "video-bucket"
    prefix = "difflet"

    def __init__(self, client: _FakeS3Client) -> None:
        self.client = client

    def _client(self):
        return self.client


class _FailingClientS3Store(_FakeS3Store):
    def _client(self):
        raise OSError("client unavailable")


def _write(target, payload: bytes = b"fake mp4 bytes") -> None:
    target.staging_path.write_bytes(payload)


def test_allocate_staging_uses_real_mp4_suffix_and_private_inode(tmp_path):
    store = LocalVideoArtifactStore(tmp_path / "videos")

    target = store.allocate_staging("video_gen_1")
    info = target.staging_path.lstat()

    assert target.staging_path.parent == store.staging_root
    assert target.staging_path.name.endswith(".part.mp4")
    assert target.artifact_key == "video_gen_1.mp4"
    assert (info.st_dev, info.st_ino) == (target.device, target.inode)
    assert stat.S_IMODE(info.st_mode) == 0o600
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700


def test_validate_commit_open_and_delete_have_expected_lifecycle(tmp_path):
    store = LocalVideoArtifactStore(tmp_path / "videos")
    target = store.allocate_staging("video_gen_1")
    payload = b"0123456789"
    _write(target, payload)

    assert store.validate_staging(
        target,
        reported_path=target.staging_path,
        expected_size_bytes=len(payload),
    ) == len(payload)
    artifact = store.commit(
        target,
        reported_path=str(target.staging_path),
        expected_size_bytes=len(payload),
    )

    assert not target.staging_path.exists()
    assert artifact.key == "video_gen_1.mp4"
    assert artifact.path.read_bytes() == payload
    assert artifact.size_bytes == len(payload)
    assert stat.S_IMODE(artifact.path.stat().st_mode) == 0o600
    assert store.require(artifact.key) == artifact

    lease = store.open(artifact.key)
    assert store.delete(artifact.key) is True
    assert store.get(artifact.key) is None
    # An already-open descriptor remains safe to stream after DELETE unlinks the
    # directory entry.
    assert b"".join(lease.iter_chunks(chunk_size=3)) == payload
    assert lease.closed is True
    assert store.delete(artifact.key) is False


def test_pending_local_cleanup_releases_token_after_unlink_fsync_failure(
    tmp_path,
    monkeypatch,
):
    store = LocalVideoArtifactStore(tmp_path / "videos")
    target = store.allocate_staging("video_gen_fsync_failure")
    _write(target)
    artifact = store.commit(target)
    real_fsync_directory = video_storage._fsync_directory

    def fail_artifact_directory_fsync(path):
        if path == store.artifact_root:
            raise OSError("simulated directory fsync failure")
        return real_fsync_directory(path)

    monkeypatch.setattr(video_storage, "_fsync_directory", fail_artifact_directory_fsync)
    with pytest.raises(OSError, match="directory fsync failure"):
        store.delete(artifact.key)
    assert artifact.path.exists() is False

    store.schedule_artifact_delete(artifact)
    monkeypatch.setattr(video_storage, "_fsync_directory", real_fsync_directory)
    assert store.retry_pending_artifact_deletes() == (artifact,)
    assert store.retry_pending_artifact_deletes() == ()


def test_s3_store_streams_upload_and_deletes_remote_and_local(tmp_path):
    client = _FakeS3Client()
    store = S3VideoArtifactStore(
        tmp_path / "videos",
        s3=_FakeS3Store(client),  # type: ignore[arg-type]
        retention_seconds=90_000,
    )
    target = store.allocate_staging("video_gen_1")
    payload = b"streamed mp4"
    _write(target, payload)

    artifact = store.commit(target, expected_size_bytes=len(payload))

    assert artifact.path.read_bytes() == payload
    assert client.uploads == [
        (
            str(artifact.path),
            "video-bucket",
            "difflet/video_gen_1.mp4",
            {"ContentType": "video/mp4"},
        )
    ]
    assert store.url(artifact.key) == (
        "https://example.test/video-bucket/difflet/video_gen_1.mp4?ttl=90000"
    )

    assert store.delete(artifact.key) is True
    assert client.deletes == [("video-bucket", "difflet/video_gen_1.mp4")]
    assert artifact.path.exists() is False


def test_s3_upload_failure_keeps_local_artifact_and_content_available(tmp_path):
    client = _FakeS3Client(upload_error=OSError("upload unavailable"))
    store = S3VideoArtifactStore(
        tmp_path / "videos",
        s3=_FakeS3Store(client),  # type: ignore[arg-type]
    )
    target = store.allocate_staging("video_gen_fallback")
    payload = b"local fallback"
    _write(target, payload)

    artifact = store.commit(target, expected_size_bytes=len(payload))

    assert store.url(artifact.key) is None
    lease = store.open(artifact.key)
    assert b"".join(lease.iter_chunks()) == payload
    # Failed publication performs a best-effort remote rollback. Because that
    # rollback succeeded, normal local DELETE does not call S3 again.
    assert client.deletes == [("video-bucket", "difflet/video_gen_fallback.mp4")]
    assert store.delete(artifact.key) is True
    assert client.deletes == [("video-bucket", "difflet/video_gen_fallback.mp4")]


def test_s3_client_failure_keeps_local_artifact_available(tmp_path):
    client = _FakeS3Client()
    store = S3VideoArtifactStore(
        tmp_path / "videos",
        s3=_FailingClientS3Store(client),  # type: ignore[arg-type]
    )
    target = store.allocate_staging("video_gen_client_fallback")
    _write(target, b"local only")

    artifact = store.commit(target)

    assert store.url(artifact.key) is None
    assert artifact.path.read_bytes() == b"local only"
    assert client.uploads == []
    assert client.deletes == []
    assert store.delete(artifact.key) is True
    assert artifact.path.exists() is False


def test_s3_presign_and_rollback_failure_has_independent_cleanup_retry(tmp_path):
    client = _FakeS3Client(
        presign_error=OSError("signing unavailable"),
        delete_errors=[
            OSError("rollback unavailable"),
            OSError("first sweep unavailable"),
        ],
    )
    store = S3VideoArtifactStore(
        tmp_path / "videos",
        s3=_FakeS3Store(client),  # type: ignore[arg-type]
    )
    target = store.allocate_staging("video_gen_retry")
    _write(target)

    artifact = store.commit(target)

    assert store.url(artifact.key) is None
    assert artifact.path.exists()
    assert client.deletes == [("video-bucket", "difflet/video_gen_retry.mp4")]

    assert store.retry_pending_remote_deletes() == 0
    assert store.retry_pending_remote_deletes() == 1
    assert store.retry_pending_artifact_deletes() == ()
    assert client.deletes == [
        ("video-bucket", "difflet/video_gen_retry.mp4"),
        ("video-bucket", "difflet/video_gen_retry.mp4"),
        ("video-bucket", "difflet/video_gen_retry.mp4"),
    ]

    # Remote ownership was cleared independently, so local deletion no longer
    # needs a job row or another S3 request.
    assert store.delete(artifact.key) is True
    assert client.deletes == [
        ("video-bucket", "difflet/video_gen_retry.mp4"),
        ("video-bucket", "difflet/video_gen_retry.mp4"),
        ("video-bucket", "difflet/video_gen_retry.mp4"),
    ]
    assert artifact.path.exists() is False


def test_s3_delete_exhaustion_remains_job_owned_until_explicitly_orphaned(tmp_path):
    client = _FakeS3Client(
        delete_errors=[OSError(f"delete attempt {attempt}") for attempt in range(3)]
    )
    store = S3VideoArtifactStore(
        tmp_path / "videos",
        s3=_FakeS3Store(client),  # type: ignore[arg-type]
    )
    target = store.allocate_staging("video_gen_delete_retry")
    _write(target)
    artifact = store.commit(target)

    for _attempt in range(3):
        with pytest.raises(OSError, match="delete attempt"):
            store.delete(artifact.key)
    assert artifact.path.exists()

    # A failed job-backed DELETE must not let the background orphan sweep remove
    # content while its completed job row is still retained.
    assert store.retry_pending_remote_deletes() == 0
    assert store.retry_pending_artifact_deletes() == ()
    assert artifact.path.exists()

    # Only the service cleanup path for an artifact with no owning job may
    # transfer full-delete ownership to the store.
    store.schedule_artifact_delete(artifact)
    assert store.retry_pending_remote_deletes() == 0
    assert store.retry_pending_artifact_deletes() == (artifact,)
    assert artifact.path.exists() is False
    assert len(client.deletes) == 4


def test_open_missing_artifact_raises_typed_error(tmp_path):
    store = LocalVideoArtifactStore(tmp_path / "videos")

    with pytest.raises(VideoArtifactNotFound, match="not found"):
        store.open("video_gen_missing.mp4")


def test_staging_validation_rejects_path_mismatch_replacement_and_symlink(tmp_path):
    store = LocalVideoArtifactStore(tmp_path / "videos")
    target = store.allocate_staging("video_gen_1")
    _write(target)

    with pytest.raises(VideoArtifactPathViolation, match="reported path"):
        store.validate_staging(target, reported_path=tmp_path / "elsewhere.mp4")

    target.staging_path.unlink()
    target.staging_path.write_bytes(b"replacement")
    with pytest.raises(VideoArtifactPathViolation, match="inode was replaced"):
        store.validate_staging(target)

    symlink_target = store.allocate_staging("video_gen_2")
    symlink_target.staging_path.unlink()
    os.symlink(tmp_path / "outside.mp4", symlink_target.staging_path)
    with pytest.raises(VideoArtifactPathViolation, match="symlink"):
        store.validate_staging(symlink_target)


def test_staging_validation_rejects_hard_links_and_unsafe_target_names(tmp_path):
    store = LocalVideoArtifactStore(tmp_path / "videos")
    target = store.allocate_staging("video_gen_1")
    _write(target)
    os.link(target.staging_path, tmp_path / "second-link.mp4")

    with pytest.raises(VideoArtifactPathViolation, match="exactly one link"):
        store.validate_staging(target)

    bad_name = replace(target, staging_path=store.staging_root / "other.part.mp4")
    with pytest.raises(VideoArtifactPathViolation, match="name does not match"):
        store.validate_staging(bad_name)

    with pytest.raises(VideoArtifactPathViolation, match="safe filename"):
        store.allocate_staging("../escape")
    with pytest.raises(VideoArtifactPathViolation, match="unsafe path segment"):
        store.allocate_staging("video..escape")
    with pytest.raises(VideoArtifactPathViolation, match="too long"):
        store.allocate_staging("v" * 214)


def test_staging_size_is_nonempty_bounded_and_exact(tmp_path):
    store = LocalVideoArtifactStore(tmp_path / "videos", max_artifact_bytes=4)
    target = store.allocate_staging("video_gen_empty")

    with pytest.raises(InvalidVideoArtifact, match="empty"):
        store.validate_staging(target)

    _write(target, b"12345")
    with pytest.raises(VideoArtifactTooLarge, match="maximum"):
        store.validate_staging(target)

    target.staging_path.write_bytes(b"1234")
    with pytest.raises(InvalidVideoArtifact, match="reported.*actual"):
        store.validate_staging(target, expected_size_bytes=3)
    assert store.validate_staging(target, expected_size_bytes=4) == 4


def test_delete_staging_refuses_replaced_inode(tmp_path):
    store = LocalVideoArtifactStore(tmp_path / "videos")
    target = store.allocate_staging("video_gen_1")
    target.staging_path.unlink()
    target.staging_path.write_bytes(b"replacement")

    with pytest.raises(VideoArtifactPathViolation, match="replaced staging inode"):
        store.delete_staging(target)


def test_commit_revalidates_and_removes_output_changed_during_publish(
    tmp_path,
    monkeypatch,
):
    store = LocalVideoArtifactStore(tmp_path / "videos")
    target = store.allocate_staging("video_gen_1")
    _write(target, b"original")
    real_replace = os.replace

    def _replace_then_mutate(source, destination):
        real_replace(source, destination)
        with open(destination, "ab") as file:
            file.write(b"changed")

    monkeypatch.setattr(video_storage.os, "replace", _replace_then_mutate)

    with pytest.raises(InvalidVideoArtifact, match="size changed"):
        store.commit(target)
    assert not os.path.lexists(store.artifact_root / target.artifact_key)


def test_orphan_sweep_respects_active_staging_and_referenced_artifacts(tmp_path):
    store = LocalVideoArtifactStore(tmp_path / "videos")
    active = store.allocate_staging("video_gen_active")
    orphan = store.allocate_staging("video_gen_orphan")
    _write(active)
    _write(orphan)
    old = 10.0
    os.utime(active.staging_path, (old, old))
    os.utime(orphan.staging_path, (old, old))

    unsafe_link = store.staging_root / "unsafe.part.mp4"
    os.symlink(tmp_path / "outside", unsafe_link)
    os.utime(unsafe_link, (old, old), follow_symlinks=False)
    unrelated = store.staging_root / "keep.txt"
    unrelated.write_text("keep")
    os.utime(unrelated, (old, old))

    keep_target = store.allocate_staging("video_gen_keep")
    drop_target = store.allocate_staging("video_gen_drop")
    _write(keep_target, b"keep")
    _write(drop_target, b"drop")
    keep = store.commit(keep_target)
    drop = store.commit(drop_target)
    os.utime(keep.path, (old, old))
    os.utime(drop.path, (old, old))

    sweep = store.sweep_orphans(
        active_staging_paths=[active.staging_path],
        referenced_artifact_keys=[keep.key],
        older_than_seconds=5,
        now=20,
    )

    assert active.staging_path.exists()
    assert not orphan.staging_path.exists()
    assert not os.path.lexists(unsafe_link)
    assert unrelated.exists()
    assert set(sweep.staging_removed) == {orphan.staging_path, unsafe_link}
    assert keep.path.exists()
    assert not drop.path.exists()
    assert sweep.artifacts_removed == (drop.key,)


def test_purge_all_managed_ignores_future_mtime_and_keeps_unrelated_entries(tmp_path):
    store = LocalVideoArtifactStore(tmp_path / "videos")
    staging = store.allocate_staging("video_gen_staging")
    final_target = store.allocate_staging("video_gen_final")
    _write(staging, b"staging")
    _write(final_target, b"final")
    final = store.commit(final_target)
    future = 10_000_000_000
    os.utime(staging.staging_path, (future, future))
    os.utime(final.path, (future, future))
    unrelated_staging = store.staging_root / "keep.txt"
    unrelated_artifact = store.artifact_root / "keep.txt"
    unrelated_staging.write_text("keep")
    unrelated_artifact.write_text("keep")

    purged = store.purge_all_managed()

    assert purged.staging_removed == (staging.staging_path,)
    assert purged.artifacts_removed == (final.key,)
    assert staging.staging_path.exists() is False
    assert final.path.exists() is False
    assert unrelated_staging.exists()
    assert unrelated_artifact.exists()


def test_storage_root_must_not_be_a_symlink(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(VideoArtifactPathViolation, match="real directory"):
        LocalVideoArtifactStore(linked_root)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_orphan_sweep_rejects_invalid_age(tmp_path, value):
    store = LocalVideoArtifactStore(tmp_path / "videos")

    with pytest.raises((TypeError, ValueError)):
        store.sweep_orphans(older_than_seconds=value)
