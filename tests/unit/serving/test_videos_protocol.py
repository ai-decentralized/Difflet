from __future__ import annotations

import pytest
from pydantic import ValidationError

from difflet.serving.openai.protocol.videos import (
    VideoDeleteResponse,
    VideoGenerationRequest,
    VideoGenerationStatus,
    VideoListResponse,
    VideoResponse,
    file_extension,
)


def _response(**overrides) -> VideoResponse:
    values = {
        "id": "video_gen_1",
        "status": VideoGenerationStatus.QUEUED,
        "model": "test/video",
        "prompt": "a paper boat",
        "size": "832x480",
        "width": 832,
        "height": 480,
        "num_frames": 49,
        "fps": 24,
        "seconds": None,
        "duration_s": 49 / 24,
        "created_at": 100,
    }
    values.update(overrides)
    return VideoResponse(**values)


def test_video_response_reports_real_media_facts_without_fake_seconds():
    response = _response()

    assert response.object == "video"
    assert response.seconds is None
    assert response.size == "832x480"
    assert (response.width, response.height, response.num_frames, response.fps) == (
        832,
        480,
        49,
        24,
    )
    assert response.duration_s == 49 / 24
    assert response.file_extension == "mp4"


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"size": "480x832"}, "size must equal"),
        ({"duration_s": 2.0}, "duration_s must equal"),
        ({"duration_s": float("nan")}, "finite"),
        ({"stage_durations": {"denoise": -1.0}}, "stage_durations"),
    ],
)
def test_video_response_rejects_false_or_nonfinite_media_metadata(overrides, match):
    with pytest.raises(ValidationError, match=match):
        _response(**overrides)


def test_video_generation_status_does_not_expose_deleted_state():
    assert {status.value for status in VideoGenerationStatus} == {
        "queued",
        "in_progress",
        "completed",
        "failed",
    }


def test_video_request_schema_is_strict_and_bounded():
    request = VideoGenerationRequest(
        prompt="a cat",
        seconds="2",
        size="832x480",
        width=832,
        height=480,
        num_frames=48,
        fps=24,
        num_inference_steps=20,
        guidance_scale=5.0,
        seed=0,
    )

    assert request.seconds == "2"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        VideoGenerationRequest(prompt="a cat", unknown="value")
    with pytest.raises(ValidationError):
        VideoGenerationRequest(prompt="a cat", guidance_scale=float("inf"))
    with pytest.raises(ValidationError):
        VideoGenerationRequest(prompt="a cat", seed=2**63)


def test_video_list_and_delete_responses_follow_vllm_shape():
    item = _response(status="completed", progress=100, file_name="video_gen_1.mp4")
    listing = VideoListResponse(
        first_id=item.id,
        last_id=item.id,
        has_more=False,
        data=[item],
    )
    deleted = VideoDeleteResponse(id=item.id, deleted=True)

    assert listing.model_dump(mode="json")["object"] == "list"
    assert listing.first_id == listing.last_id == "video_gen_1"
    assert deleted.model_dump() == {
        "id": "video_gen_1",
        "deleted": True,
        "object": "video.deleted",
    }


def test_file_extension_rejects_unknown_media_type():
    assert file_extension("video/mp4; charset=binary") == "mp4"
    with pytest.raises(ValueError, match="No recognized"):
        file_extension("application/x-difflet-unknown")
