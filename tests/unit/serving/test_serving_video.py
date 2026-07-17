from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from difflet.serving.errors import DiffletServingError
from difflet.serving.openai.protocol.videos import VideoGenerationStatus
from difflet.serving.openai.serving_video import (
    form_to_mapping,
    normalize_video_multipart_request,
    normalize_video_request,
    resolved_video_facts,
    video_delete_response,
    video_job_to_response,
    video_jobs_to_list_response,
    video_request_to_job_fields,
)
from difflet.serving.types import FileBackedGenerateOutput


def _resolved_model(*, num_frames: int = 48, fps: int = 24):
    return SimpleNamespace(
        model_id="test/video",
        metadata=SimpleNamespace(
            output_modality="video",
            default_steps=30,
            default_guidance_scale=5.0,
        ),
        profile=SimpleNamespace(
            output_modality="video",
            width=832,
            height=480,
            num_frames=num_frames,
            output_fps=fps,
        ),
    )


def test_normalize_video_request_resolves_profile_and_model_specific_fields():
    request = normalize_video_request(
        {
            "model": "test/video",
            "prompt": "  a paper boat  ",
            "seconds": "2",
            "size": "832x480",
            "width": "832",
            "height": "480",
            "fps": "24",
            "num_frames": "48",
            "num_inference_steps": "40",
            "guidance_scale": "6.5",
            "guidance_scale_2": "4",
            "boundary_ratio": "0.5",
            "flow_shift": "3.0",
            "negative_prompt": " blur ",
            "seed": str(2**63 - 1),
            "user": "caller-1",
        },
        resolved_model=_resolved_model(),
        request_id="request-1",
    )

    assert request.request_id == "request-1"
    assert request.model == "test/video"
    assert request.prompt == "a paper boat"
    assert request.output_format == "mp4"
    assert (request.width, request.height, request.num_inference_steps) == (832, 480, 40)
    assert request.guidance_scale == 6.5
    assert request.seed == 2**63 - 1
    assert request.video is not None
    assert (request.video.num_frames, request.video.fps) == (48, 24)
    assert request.video.requested_seconds == "2"
    assert request.video.negative_prompt == "blur"
    assert request.video.guidance_scale_2 == 4.0
    assert request.video.boundary_ratio == 0.5
    assert request.video.flow_shift == 3.0
    assert request.video.user == "caller-1"


def test_omitted_seconds_remains_null_for_non_integral_profile_duration():
    request = normalize_video_request(
        {"prompt": "waves"},
        resolved_model=_resolved_model(num_frames=49, fps=24),
    )

    facts = resolved_video_facts(request)
    assert facts.seconds is None
    assert facts.duration_s == 49 / 24
    assert video_request_to_job_fields(request)["seconds"] is None


@pytest.mark.parametrize(
    "form,code",
    [
        ({"prompt": "x", "model": "other/video"}, "model_not_served"),
        ({"prompt": "x", "size": "480x832"}, "profile_mismatch"),
        ({"prompt": "x", "size": "832x480", "width": "800"}, "profile_mismatch"),
        ({"prompt": "x", "fps": "25"}, "profile_mismatch"),
        ({"prompt": "x", "num_frames": "47"}, "profile_mismatch"),
        ({"prompt": "x", "seconds": "1"}, "profile_mismatch"),
        (
            {"prompt": "x", "seconds": "2", "num_frames": "47"},
            "profile_mismatch",
        ),
    ],
)
def test_fixed_model_and_profile_conflicts_are_rejected(form, code):
    with pytest.raises(DiffletServingError) as exc:
        normalize_video_request(form, resolved_model=_resolved_model())

    assert exc.value.code == code


@pytest.mark.parametrize(
    "field,value",
    [
        ("guidance_scale", "nan"),
        ("guidance_scale_2", "inf"),
        ("boundary_ratio", "1.1"),
        ("flow_shift", "-inf"),
        ("num_inference_steps", "201"),
        ("seed", str(2**63)),
        ("seconds", "0"),
        ("width", "1.5"),
    ],
)
def test_numeric_fields_are_strict_finite_and_bounded(field, value):
    with pytest.raises(DiffletServingError) as exc:
        normalize_video_request(
            {"prompt": "x", field: value},
            resolved_model=_resolved_model(),
        )

    assert exc.value.code == "invalid_extra_body"


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_reference", "file-1"),
        ("generate_sound", "false"),
        ("true_cfg_scale", "0"),
        ("enable_frame_interpolation", "false"),
        ("lora", "{}"),
        ("extra_params", "{}"),
        ("video_params", "{}"),
    ],
)
def test_known_unsupported_fields_are_rejected_even_when_false_or_empty(field, value):
    with pytest.raises(DiffletServingError) as exc:
        normalize_video_request(
            {"prompt": "x", field: value},
            resolved_model=_resolved_model(),
        )

    assert exc.value.code == "feature_not_supported"


def test_unknown_duplicate_and_file_form_fields_are_rejected():
    class _MultiForm:
        def __init__(self, items):
            self._items = items

        def multi_items(self):
            return self._items

    with pytest.raises(DiffletServingError) as unknown:
        form_to_mapping({"prompt": "x", "surprise": "value"})
    assert unknown.value.code == "invalid_request"

    with pytest.raises(DiffletServingError) as duplicate:
        form_to_mapping(_MultiForm([("prompt", "one"), ("prompt", "two")]))
    assert duplicate.value.code == "invalid_request"

    with pytest.raises(DiffletServingError) as upload:
        form_to_mapping({"prompt": object()})
    assert upload.value.code == "feature_not_supported"


@pytest.mark.parametrize("form", [{}, {"prompt": "   "}])
def test_prompt_is_required_and_nonempty(form):
    with pytest.raises(DiffletServingError) as exc:
        normalize_video_request(form, resolved_model=_resolved_model())

    assert exc.value.code == "invalid_prompt"


def test_prompt_negative_prompt_and_user_limits_are_enforced():
    for field, value, code in (
        ("prompt", "x" * 32_769, "invalid_prompt"),
        ("negative_prompt", "x" * 32_769, "invalid_extra_body"),
        ("user", "x" * 257, "invalid_extra_body"),
    ):
        with pytest.raises(DiffletServingError) as exc:
            normalize_video_request(
                {"prompt": "ok", field: value},
                resolved_model=_resolved_model(),
            )
        assert exc.value.code == code


def test_manual_multipart_parser_avoids_fastapi_form_injection():
    class _Request:
        headers = {"content-type": "multipart/form-data; boundary=test"}

        async def form(self):
            return {"prompt": "clouds"}

    request = asyncio.run(
        normalize_video_multipart_request(
            _Request(),
            resolved_model=_resolved_model(),
            request_id="multipart-1",
        )
    )

    assert request.request_id == "multipart-1"
    assert request.prompt == "clouds"


def test_manual_multipart_parser_rejects_wrong_content_type_and_parse_failure():
    class _JsonRequest:
        headers = {"content-type": "application/json"}

    class _BadRequest:
        headers = {"content-type": "multipart/form-data; boundary=test"}

        async def form(self):
            raise ValueError("private parser detail")

    for request in (_JsonRequest(), _BadRequest()):
        with pytest.raises(DiffletServingError) as exc:
            asyncio.run(
                normalize_video_multipart_request(
                    request,
                    resolved_model=_resolved_model(),
                )
            )
        assert exc.value.code == "invalid_request"
        assert "private parser detail" not in exc.value.message


@dataclass(frozen=True)
class _Job:
    video_id: str
    status: str
    request: object
    created_at: int
    completed_at: int | None = None
    output: object | None = None
    error_code: str | None = None
    error_message: str | None = None


def test_job_mapper_uses_real_request_and_file_metadata_without_leaking_path():
    request = normalize_video_request(
        {"prompt": "rain", "seconds": "2"},
        resolved_model=_resolved_model(),
        request_id="job-request",
    )
    output = FileBackedGenerateOutput(
        path="/private/server/output/video.part",
        mime_type="video/mp4",
        output_format="mp4",
        size_bytes=1234,
        width=832,
        height=480,
        num_frames=48,
        fps=24,
        duration_s=2.0,
    )
    job = _Job(
        video_id="video_gen_abc",
        status="completed",
        request=request,
        created_at=100,
        completed_at=120,
        output=output,
    )

    response = video_job_to_response(job)

    assert response.status is VideoGenerationStatus.COMPLETED
    assert response.progress == 100
    assert response.seconds == "2"
    assert response.duration_s == 2.0
    assert response.file_size_bytes == 1234
    assert response.file_name == "video_gen_abc.mp4"
    assert "/private/" not in response.model_dump_json()


def test_failed_job_mapper_sets_terminal_time_and_structured_error():
    job = {
        "id": "video_gen_failed",
        "status": "failed",
        "model": "test/video",
        "prompt": "rain",
        "width": 832,
        "height": 480,
        "num_frames": 49,
        "fps": 24,
        "created_at": 100,
        "updated_at": 130,
        "error": {"code": "generation_failed", "message": "worker failed"},
    }

    response = video_job_to_response(job)

    assert response.completed_at == 130
    assert response.file_name is None
    assert response.error is not None
    assert response.error.code == "generation_failed"


def test_job_mapper_supports_repository_flat_request_and_media_metadata_contract():
    @dataclass(frozen=True)
    class _StoredError:
        code: str
        message: str
        error_type: str

    job = {
        "video_id": "video_gen_flat",
        "status": "failed",
        "request": {
            "model": "test/video",
            "prompt": "rain",
            "width": 832,
            "height": 480,
            "num_frames": 49,
            "fps": 24.0,
            "seconds": None,
        },
        "created_at": 100,
        "updated_at": 130,
        "artifact_size_bytes": 4096,
        "media_metadata": {
            "width": 832,
            "height": 480,
            "num_frames": 49,
            "fps": 24.0,
            "duration_s": 49 / 24,
            "inference_time_s": 12.5,
            "file_name": "/private/store/should-not-leak.mp4",
        },
        "error": _StoredError("worker_failed", "worker failed", "server_error"),
    }

    response = video_job_to_response(job)

    assert (response.width, response.height, response.num_frames, response.fps) == (
        832,
        480,
        49,
        24,
    )
    assert response.seconds is None
    assert response.duration_s == 49 / 24
    assert response.file_size_bytes == 4096
    assert response.inference_time_s == 12.5
    assert response.file_name is None  # failed artifacts are never exposed
    assert response.error is not None
    assert response.error.model_dump() == {
        "code": "worker_failed",
        "message": "worker failed",
        "error_type": "server_error",
    }
    assert "/private/store" not in response.model_dump_json()

    completed = dict(job)
    completed.update(status="completed", error=None, completed_at=140)
    completed_response = video_job_to_response(completed)
    assert completed_response.file_name == "should-not-leak.mp4"
    assert "/private/store" not in completed_response.model_dump_json()


def test_list_and_delete_mappers_have_stable_empty_and_nonempty_shapes():
    request = normalize_video_request(
        {"prompt": "rain"},
        resolved_model=_resolved_model(),
    )
    job = _Job("video_gen_1", "queued", request, 100)

    listing = video_jobs_to_list_response([job], has_more=True)
    empty = video_jobs_to_list_response([], has_more=False)
    deleted = video_delete_response("video_gen_1")

    assert (listing.first_id, listing.last_id, listing.has_more) == (
        "video_gen_1",
        "video_gen_1",
        True,
    )
    assert (empty.first_id, empty.last_id, empty.data) == (None, None, [])
    assert deleted.deleted is True
