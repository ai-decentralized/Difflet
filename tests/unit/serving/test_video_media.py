from __future__ import annotations

import math
import os

import pytest

import difflet.serving.video_media as video_media
from difflet.serving.video_media import (
    VideoMediaDependencyError,
    VideoMediaEncodingError,
    VideoMediaMetadata,
    VideoMediaValidationError,
    encode_tensor_to_mp4,
    normalize_video_tensor,
    validate_mp4,
)


def test_media_metadata_serializes_trusted_fields():
    metadata = VideoMediaMetadata(
        width=16,
        height=8,
        num_frames=4,
        fps=2.0,
        duration_s=2.0,
        codec_name="h264",
        pixel_format="yuv420p",
        size_bytes=123,
    )

    assert metadata.to_dict() == {
        "width": 16,
        "height": 8,
        "num_frames": 4,
        "fps": 2.0,
        "duration_s": 2.0,
        "codec_name": "h264",
        "pixel_format": "yuv420p",
        "size_bytes": 123,
    }


def test_encode_requires_absolute_real_mp4_target(tmp_path):
    with pytest.raises(ValueError, match="end with .mp4"):
        encode_tensor_to_mp4(
            object(),
            tmp_path / "clip.mp4.part",
            fps=24,
            layout="BCTHW",
            value_range="minus_one_to_one",
        )
    with pytest.raises(ValueError, match="absolute"):
        encode_tensor_to_mp4(
            object(),
            "clip.mp4",
            fps=24,
            layout="BCTHW",
            value_range="minus_one_to_one",
        )

    outside = tmp_path / "outside.mp4"
    output = tmp_path / "clip.mp4"
    output.symlink_to(outside)
    with pytest.raises(VideoMediaEncodingError, match="regular file"):
        encode_tensor_to_mp4(
            object(),
            output,
            fps=24,
            layout="BCTHW",
            value_range="minus_one_to_one",
        )


def test_missing_numpy_is_a_typed_dependency_error(tmp_path, monkeypatch):
    output = tmp_path / "clip.part.mp4"
    output.touch()

    def _missing_numpy():
        raise VideoMediaDependencyError("numpy unavailable")

    monkeypatch.setattr(video_media, "_import_numpy", _missing_numpy)

    with pytest.raises(VideoMediaDependencyError, match="numpy unavailable"):
        encode_tensor_to_mp4(
            object(),
            output,
            fps=24,
            layout="BCTHW",
            value_range="minus_one_to_one",
        )


def test_normalize_supported_layouts_and_ranges():
    np = pytest.importorskip("numpy")
    zero_one = np.zeros((1, 2, 3, 2, 4), dtype=np.float32)
    zero_one[:, 1] = 1.0

    bfchw = normalize_video_tensor(
        zero_one,
        layout="BFCHW",
        value_range="zero_to_one",
    )
    assert bfchw.shape == (2, 2, 4, 3)
    assert bfchw.dtype == np.uint8
    assert set(np.unique(bfchw).tolist()) == {0, 255}

    minus_one = np.full((1, 3, 2, 2, 4), -1.0, dtype=np.float32)
    minus_one[:, :, 1] = 1.0
    bcthw = normalize_video_tensor(
        minus_one,
        layout="BCTHW",
        value_range="minus_one_to_one",
    )
    assert bcthw.shape == (2, 2, 4, 3)
    assert set(np.unique(bcthw).tolist()) == {0, 255}

    uint8 = np.zeros((2, 2, 4, 3), dtype=np.uint8)
    assert normalize_video_tensor(
        uint8,
        layout="FHWC",
        value_range="uint8",
    ).flags.c_contiguous


def test_normalize_rejects_invalid_batch_channels_dimensions_and_values():
    np = pytest.importorskip("numpy")

    with pytest.raises(VideoMediaEncodingError, match="batch size 1"):
        normalize_video_tensor(
            np.zeros((2, 3, 1, 2, 2), dtype=np.float32),
            layout="BCTHW",
            value_range="minus_one_to_one",
        )
    with pytest.raises(VideoMediaEncodingError, match="3 channels"):
        normalize_video_tensor(
            np.zeros((1, 4, 1, 2, 2), dtype=np.float32),
            layout="BCTHW",
            value_range="minus_one_to_one",
        )
    with pytest.raises(VideoMediaEncodingError, match="dimensions must be positive"):
        normalize_video_tensor(
            np.zeros((1, 3, 0, 2, 2), dtype=np.float32),
            layout="BCTHW",
            value_range="minus_one_to_one",
        )
    with pytest.raises(VideoMediaEncodingError, match="range"):
        normalize_video_tensor(
            np.full((1, 3, 1, 2, 2), 2.0, dtype=np.float32),
            layout="BCTHW",
            value_range="minus_one_to_one",
        )
    invalid = np.zeros((1, 3, 1, 2, 2), dtype=np.float32)
    invalid[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(VideoMediaEncodingError, match="NaN or infinity"):
        normalize_video_tensor(
            invalid,
            layout="BCTHW",
            value_range="minus_one_to_one",
        )


@pytest.mark.parametrize("layout", ["BCHW", "", None])
def test_normalize_rejects_unknown_layout(layout):
    with pytest.raises(ValueError, match="unsupported.*layout"):
        normalize_video_tensor(object(), layout=layout, value_range="zero_to_one")


@pytest.mark.parametrize("value_range", ["zero_one", "minus_one_one", "", None])
def test_normalize_rejects_unknown_value_range(value_range):
    with pytest.raises(ValueError, match="unsupported.*range"):
        normalize_video_tensor(object(), layout="BCTHW", value_range=value_range)


def test_validate_rejects_missing_empty_symlink_and_oversized_files(tmp_path):
    missing = tmp_path / "missing.mp4"
    with pytest.raises(VideoMediaValidationError, match="does not exist"):
        validate_mp4(missing)

    empty = tmp_path / "empty.mp4"
    empty.touch()
    with pytest.raises(VideoMediaValidationError, match="empty"):
        validate_mp4(empty)

    symlink = tmp_path / "linked.mp4"
    symlink.symlink_to(missing)
    with pytest.raises(VideoMediaValidationError, match="non-symlink"):
        validate_mp4(symlink)

    oversized = tmp_path / "oversized.mp4"
    oversized.write_bytes(b"12345")
    with pytest.raises(VideoMediaValidationError, match="maximum size"):
        validate_mp4(oversized, max_size_bytes=4)


def test_encode_and_reopen_validate_silent_mp4_on_cpu(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    output = tmp_path / "clip.part.mp4"
    values = np.linspace(-1.0, 1.0, num=1 * 3 * 4 * 16 * 16, dtype=np.float32)
    tensor = values.reshape(1, 3, 4, 16, 16)

    metadata = encode_tensor_to_mp4(
        tensor,
        output,
        fps=4,
        layout="BCTHW",
        value_range="minus_one_to_one",
    )

    assert output.stat().st_size == metadata.size_bytes > 0
    assert (metadata.width, metadata.height, metadata.num_frames) == (16, 16, 4)
    assert math.isclose(metadata.fps, 4.0)
    assert math.isclose(metadata.duration_s, 1.0)
    reopened = validate_mp4(
        output,
        expected_width=16,
        expected_height=16,
        expected_num_frames=4,
        expected_fps=4,
    )
    assert reopened == metadata


def test_encoder_failure_raises_and_leaves_no_unvalidated_bytes(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    output = tmp_path / "clip.part.mp4"
    output.write_bytes(b"old untrusted bytes")
    tensor = np.zeros((1, 3, 1, 16, 16), dtype=np.float32)

    with pytest.raises(VideoMediaEncodingError, match="failed to encode"):
        encode_tensor_to_mp4(
            tensor,
            output,
            fps=4,
            layout="BCTHW",
            value_range="minus_one_to_one",
            codec="definitely-not-a-real-codec",
        )

    assert output.read_bytes() == b""


def test_validate_corrupt_mp4_raises_typed_error(tmp_path):
    pytest.importorskip("av")
    output = tmp_path / "corrupt.mp4"
    output.write_bytes(os.urandom(128))

    with pytest.raises(VideoMediaValidationError, match="decodable MP4"):
        validate_mp4(output)
