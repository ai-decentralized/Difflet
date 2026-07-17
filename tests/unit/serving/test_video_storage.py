from __future__ import annotations

import os
import stat
from dataclasses import replace

import pytest

import difflet.serving.video_storage as video_storage
from difflet.serving.video_storage import (
    InvalidVideoArtifact,
    LocalVideoArtifactStore,
    VideoArtifactNotFound,
    VideoArtifactPathViolation,
    VideoArtifactTooLarge,
)


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
