from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType

import pytest

from difflet.serving.video_jobs import (
    InMemoryVideoJobRepository,
    InvalidVideoJobTransition,
    VideoJobCreate,
    VideoJobError,
    VideoJobRepositoryError,
    VideoJobStateConflict,
)


def _create(
    repository: InMemoryVideoJobRepository,
    video_id: str,
    *,
    created_at: int = 100,
):
    return repository.create(
        VideoJobCreate(
            id=video_id,
            model="test/video",
            prompt=f"prompt for {video_id}",
            request={"width": 16, "height": 16},
            user="caller-1",
            created_at=created_at,
        )
    )


def _complete(
    repository: InMemoryVideoJobRepository,
    video_id: str,
    *,
    now: int = 102,
):
    repository.mark_in_progress(video_id, now=now - 1)
    return repository.mark_completed(
        video_id,
        artifact_key=f"{video_id}.mp4",
        artifact_size_bytes=123,
        media_metadata={
            "width": 16,
            "height": 16,
            "num_frames": 2,
            "fps": 2.0,
            "duration_s": 1.0,
        },
        now=now,
    )


def test_close_clears_process_local_jobs_and_fresh_repository_starts_empty():
    repository = InMemoryVideoJobRepository()
    created = _create(repository, "video_gen_1")

    assert created.status == "queued"
    assert created.version == 1
    repository.close()

    with pytest.raises(VideoJobRepositoryError, match="closed"):
        repository.get("video_gen_1")

    fresh = InMemoryVideoJobRepository()
    assert fresh.get("video_gen_1") is None


def test_clear_removes_jobs_but_repository_remains_open():
    repository = InMemoryVideoJobRepository()
    _create(repository, "video_gen_1")

    repository.clear()

    assert repository.count() == 0
    assert _create(repository, "video_gen_2").status == "queued"


def test_request_and_media_metadata_are_isolated_from_callers():
    repository = InMemoryVideoJobRepository()
    request_payload = {"nested": {"values": [1, 2]}}
    request = MappingProxyType(request_payload)
    created = repository.create(
        VideoJobCreate(
            id="video_gen_1",
            model="test/video",
            prompt="prompt",
            request=request,
            created_at=100,
        )
    )
    request_payload["nested"]["values"].append(3)
    created.request["nested"]["values"].append(4)

    assert repository.require("video_gen_1").request == {"nested": {"values": [1, 2]}}

    repository.mark_in_progress("video_gen_1", now=101)
    media = {"dimensions": {"width": 16, "height": 16}}
    completed = repository.mark_completed(
        "video_gen_1",
        artifact_key="video_gen_1.mp4",
        artifact_size_bytes=123,
        media_metadata=media,
        now=102,
    )
    media["dimensions"]["width"] = 32
    assert completed.media_metadata is not None
    completed.media_metadata["dimensions"]["height"] = 32

    assert repository.require("video_gen_1").media_metadata == {
        "dimensions": {"width": 16, "height": 16}
    }


def test_transition_invariants_and_compare_and_set_conflicts():
    repository = InMemoryVideoJobRepository()
    _create(repository, "video_gen_1")

    running = repository.mark_in_progress("video_gen_1", now=101)
    assert running.status == "in_progress"
    assert running.started_at == 101
    assert running.version == 2

    with pytest.raises(VideoJobStateConflict, match="expected"):
        repository.mark_in_progress("video_gen_1", now=101)
    with pytest.raises(InvalidVideoJobTransition, match="in_progress.*in_progress"):
        repository.compare_and_set(
            "video_gen_1",
            "in_progress",
            "in_progress",
            now=102,
        )
    with pytest.raises(ValueError, match="media_metadata"):
        repository.mark_completed(
            "video_gen_1",
            artifact_key="video_gen_1.mp4",
            artifact_size_bytes=1,
            media_metadata={},
            now=102,
        )

    completed = repository.mark_completed(
        "video_gen_1",
        artifact_key="video_gen_1.mp4",
        artifact_size_bytes=123,
        media_metadata={"width": 16, "height": 16, "num_frames": 2, "fps": 2},
        now=102,
    )
    assert completed.status == "completed"
    assert completed.completed_at == 102
    assert completed.artifact_key == "video_gen_1.mp4"
    assert completed.error is None
    assert completed.version == 3

    with pytest.raises(InvalidVideoJobTransition, match="completed.*failed"):
        repository.mark_failed(
            "video_gen_1",
            expected_statuses="completed",
            error=VideoJobError(code="late_error", message="too late"),
            now=103,
        )


def test_failed_jobs_store_structured_error_without_artifact():
    repository = InMemoryVideoJobRepository()
    _create(repository, "video_gen_1")

    failed = repository.mark_failed(
        "video_gen_1",
        error=VideoJobError(
            code="generation_failed",
            message="model execution failed",
            error_type="server_error",
        ),
        now=101,
    )

    assert failed.status == "failed"
    assert failed.completed_at == 101
    assert failed.error is not None
    assert failed.error.code == "generation_failed"
    assert failed.artifact_key is None
    assert failed.media_metadata is None


def test_id_anchored_list_is_stable_and_unknown_after_is_empty():
    repository = InMemoryVideoJobRepository()
    for video_id in ("video_gen_a", "video_gen_b", "video_gen_c", "video_gen_d"):
        _create(repository, video_id, created_at=100)

    first = repository.list(limit=2)
    assert [job.id for job in first.data] == ["video_gen_d", "video_gen_c"]
    assert first.first_id == "video_gen_d"
    assert first.last_id == "video_gen_c"
    assert first.has_more is True
    assert first.next_cursor == first.last_id

    _create(repository, "video_gen_cz", created_at=100)
    second = repository.list(limit=2, after=first.last_id)
    assert [job.id for job in second.data] == ["video_gen_b", "video_gen_a"]
    assert second.has_more is False
    assert second.next_cursor is None

    unknown = repository.list(limit=20, after="video_gen_unknown")
    assert unknown.data == ()
    assert unknown.has_more is False


def test_list_status_filter_uses_unfiltered_after_anchor():
    repository = InMemoryVideoJobRepository()
    for video_id in ("video_gen_a", "video_gen_b", "video_gen_c"):
        _create(repository, video_id, created_at=100)
    repository.mark_failed(
        "video_gen_b",
        error=VideoJobError(code="failed", message="failed"),
        now=101,
    )

    page = repository.list(limit=10, after="video_gen_c", status="queued")

    assert [job.id for job in page.data] == ["video_gen_a"]


def test_delete_physically_removes_metadata_without_deleted_state():
    repository = InMemoryVideoJobRepository()
    created = _create(repository, "video_gen_1")

    with pytest.raises(VideoJobStateConflict, match="expected"):
        repository.delete("video_gen_1", expected_statuses="completed")
    assert repository.delete("video_gen_1") == created
    assert repository.get("video_gen_1") is None
    assert repository.delete("video_gen_1") is None


@pytest.mark.parametrize("limit", [0, 101, True, 1.5])
def test_list_rejects_invalid_limits(limit):
    repository = InMemoryVideoJobRepository()

    with pytest.raises(ValueError, match="limit"):
        repository.list(limit=limit)


def test_create_rejects_duplicate_ids_and_non_json_request():
    repository = InMemoryVideoJobRepository()
    _create(repository, "video_gen_1")

    with pytest.raises(VideoJobStateConflict, match="already exists"):
        _create(repository, "video_gen_1")
    with pytest.raises(ValueError, match="JSON serializable"):
        repository.create(
            VideoJobCreate(
                id="video_gen_2",
                model="test/video",
                prompt="prompt",
                request={"invalid": object()},
            )
        )


def test_concurrent_creates_and_compare_and_set_are_atomic():
    repository = InMemoryVideoJobRepository()
    worker_count = 12
    create_barrier = threading.Barrier(worker_count)

    def create_job(index: int):
        create_barrier.wait()
        return _create(repository, f"video_gen_{index}")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        created = list(executor.map(create_job, range(worker_count)))

    assert len({job.id for job in created}) == worker_count
    assert repository.count(status="queued") == worker_count

    race_barrier = threading.Barrier(worker_count)

    def claim_job(_index: int) -> bool:
        race_barrier.wait()
        try:
            repository.mark_in_progress("video_gen_0", now=101)
        except VideoJobStateConflict:
            return False
        return True

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        claimed = list(executor.map(claim_job, range(worker_count)))

    assert sum(claimed) == 1
    assert repository.require("video_gen_0").version == 2


def test_terminal_jobs_receive_expiry_and_can_be_selected_for_sweeping():
    repository = InMemoryVideoJobRepository()
    _create(repository, "video_completed")
    repository.mark_in_progress("video_completed", now=100)
    completed = repository.mark_completed(
        "video_completed",
        artifact_key="video_completed.mp4",
        artifact_size_bytes=10,
        media_metadata={"width": 16},
        expires_at=200,
        now=101,
    )
    _create(repository, "video_failed")
    failed = repository.mark_failed(
        "video_failed",
        error=VideoJobError(code="failed", message="failed"),
        expires_at=300,
        now=102,
    )

    assert completed.expires_at == 200
    assert failed.expires_at == 300
    assert repository.list_expired(now=199) == ()
    assert repository.list_expired(now=200) == (completed,)
    assert {job.id for job in repository.list_expired(now=300)} == {
        completed.id,
        failed.id,
    }
