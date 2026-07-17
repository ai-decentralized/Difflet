"""CPU-only MP4 encoding and validation helpers for T2V serving."""

from __future__ import annotations

import math
import os
import stat
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

VideoTensorLayout = Literal["BFCHW", "BCTHW", "FCHW", "CTHW", "FHWC"]
VideoValueRange = Literal["zero_to_one", "minus_one_to_one", "uint8"]

_SUPPORTED_LAYOUTS: frozenset[str] = frozenset({"BFCHW", "BCTHW", "FCHW", "CTHW", "FHWC"})
_SUPPORTED_VALUE_RANGES: frozenset[str] = frozenset({"zero_to_one", "minus_one_to_one", "uint8"})


class VideoMediaError(RuntimeError):
    """Base error for generated video encoding/validation failures."""


class VideoMediaDependencyError(VideoMediaError):
    """PyAV or NumPy is unavailable in the serving environment."""


class VideoMediaEncodingError(VideoMediaError):
    """A frame tensor could not be encoded as MP4."""


class VideoMediaValidationError(VideoMediaError):
    """A generated file is not the expected silent MP4."""


@dataclass(frozen=True, slots=True)
class VideoMediaMetadata:
    width: int
    height: int
    num_frames: int
    fps: float
    duration_s: float
    codec_name: str
    pixel_format: str | None
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def encode_tensor_to_mp4(
    tensor: Any,
    path: str | Path,
    *,
    fps: int,
    layout: VideoTensorLayout,
    value_range: VideoValueRange,
    codec: str = "libx264",
    pixel_format: str = "yuv420p",
    crf: int | None = 18,
    preset: str | None = "medium",
) -> VideoMediaMetadata:
    """Encode one batch/clip tensor to a silent MP4 and reopen-validate it.

    Supported model layouts intentionally cover the current T2V adapters:

    - LTX-2: ``BFCHW`` in ``zero_to_one`` range.
    - Wan/Hunyuan: ``BCTHW`` in ``minus_one_to_one`` range.

    The destination must retain a real ``.mp4`` suffix.  A staging filename such
    as ``<token>.part.mp4`` is valid; ``<token>.mp4.part`` is rejected because
    FFmpeg commonly infers its container from the final suffix.
    """

    output_path = _validate_output_path(path)
    fps = _validate_positive_int(fps, "fps")
    if not isinstance(codec, str) or not codec:
        raise ValueError("codec must be a non-empty string")
    if not isinstance(pixel_format, str) or not pixel_format:
        raise ValueError("pixel_format must be a non-empty string")
    if crf is not None:
        if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 63:
            raise ValueError("crf must be an integer between 0 and 63")
    if preset is not None and (not isinstance(preset, str) or not preset):
        raise ValueError("preset must be a non-empty string or None")

    frames = normalize_video_tensor(tensor, layout=layout, value_range=value_range)
    num_frames, height, width, channels = tuple(int(value) for value in frames.shape)
    if channels != 3:
        raise VideoMediaEncodingError(
            f"video tensor must contain exactly 3 RGB channels; got {channels}"
        )
    if num_frames <= 0 or height <= 0 or width <= 0:
        raise VideoMediaEncodingError("video tensor dimensions must be positive")
    if pixel_format == "yuv420p" and (width % 2 or height % 2):
        raise VideoMediaEncodingError("yuv420p encoding requires even width and height")

    av = _import_av()
    try:
        with av.open(str(output_path), mode="w", format="mp4") as container:
            stream = container.add_stream(codec, rate=fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = pixel_format
            options: dict[str, str] = {}
            if crf is not None:
                options["crf"] = str(crf)
            if preset is not None:
                options["preset"] = preset
            if options:
                stream.options = options
            time_base = Fraction(1, fps)
            for index, array in enumerate(frames):
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                frame.pts = index
                frame.time_base = time_base
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception as exc:
        _truncate_failed_output(output_path)
        raise VideoMediaEncodingError(f"failed to encode silent MP4 with codec {codec!r}") from exc

    return validate_mp4(
        output_path,
        expected_width=width,
        expected_height=height,
        expected_num_frames=num_frames,
        expected_fps=fps,
        require_silent=True,
    )


def normalize_video_tensor(
    tensor: Any,
    *,
    layout: VideoTensorLayout,
    value_range: VideoValueRange,
):
    """Return a contiguous ``uint8`` NumPy array shaped ``(F,H,W,3)``."""

    if layout not in _SUPPORTED_LAYOUTS:
        raise ValueError(f"unsupported video tensor layout {layout!r}")
    if value_range not in _SUPPORTED_VALUE_RANGES:
        raise ValueError(f"unsupported video value range {value_range!r}")
    np = _import_numpy()
    try:
        if hasattr(tensor, "detach"):
            value = tensor.detach()
            if hasattr(value, "to"):
                value = value.to(device="cpu")
            elif hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "float") and value_range != "uint8":
                value = value.float()
            value = value.numpy() if hasattr(value, "numpy") else value
        else:
            value = tensor
        array = np.asarray(value)
    except Exception as exc:
        raise VideoMediaEncodingError("could not materialize video tensor on CPU") from exc

    expected_ndim = 5 if layout in {"BFCHW", "BCTHW"} else 4
    if int(array.ndim) != expected_ndim:
        raise VideoMediaEncodingError(
            f"layout {layout} requires {expected_ndim} dimensions; got {array.ndim}"
        )
    if layout in {"BFCHW", "BCTHW"}:
        if int(array.shape[0]) != 1:
            raise VideoMediaEncodingError(
                f"video serving supports batch size 1; got {array.shape[0]}"
            )
        array = array[0]
    if layout == "BFCHW" or layout == "FCHW":
        array = np.transpose(array, (0, 2, 3, 1))
    elif layout == "BCTHW" or layout == "CTHW":
        array = np.transpose(array, (1, 2, 3, 0))
    elif layout != "FHWC":  # Defensive for future edits after validation above.
        raise ValueError(f"unsupported video tensor layout {layout!r}")

    if any(int(dimension) <= 0 for dimension in array.shape):
        raise VideoMediaEncodingError("video tensor dimensions must be positive")
    if int(array.shape[-1]) != 3:
        raise VideoMediaEncodingError(
            f"video tensor must contain exactly 3 channels; got {array.shape[-1]}"
        )
    if value_range == "uint8":
        if array.dtype != np.uint8:
            raise VideoMediaEncodingError("uint8 video range requires a uint8 tensor")
        normalized = array
    else:
        array = array.astype(np.float32, copy=False)
        if not bool(np.isfinite(array).all()):
            raise VideoMediaEncodingError("video tensor contains NaN or infinity")
        minimum = float(array.min())
        maximum = float(array.max())
        tolerance = 1e-4
        if value_range == "zero_to_one":
            if minimum < -tolerance or maximum > 1.0 + tolerance:
                raise VideoMediaEncodingError(
                    f"zero_to_one video tensor range is [{minimum}, {maximum}]"
                )
            scaled = np.clip(array, 0.0, 1.0) * 255.0
        else:
            if minimum < -1.0 - tolerance or maximum > 1.0 + tolerance:
                raise VideoMediaEncodingError(
                    f"minus_one_to_one video tensor range is [{minimum}, {maximum}]"
                )
            scaled = (np.clip(array, -1.0, 1.0) + 1.0) * 127.5
        normalized = np.rint(scaled).astype(np.uint8)
    return np.ascontiguousarray(normalized)


def validate_mp4(
    path: str | Path,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_num_frames: int | None = None,
    expected_fps: int | float | None = None,
    require_silent: bool = True,
    max_size_bytes: int | None = None,
) -> VideoMediaMetadata:
    """Reopen and fully decode an MP4, returning trusted media metadata."""

    media_path = _validate_existing_mp4(path, max_size_bytes=max_size_bytes)
    for field_name, value in (
        ("expected_width", expected_width),
        ("expected_height", expected_height),
        ("expected_num_frames", expected_num_frames),
    ):
        if value is not None:
            _validate_positive_int(value, field_name)
    if expected_fps is not None:
        _validate_positive_finite_number(expected_fps, "expected_fps")
    if type(require_silent) is not bool:
        raise TypeError("require_silent must be a boolean")

    av = _import_av()
    try:
        with av.open(str(media_path), mode="r", format="mp4") as container:
            video_streams = tuple(container.streams.video)
            if len(video_streams) != 1:
                raise VideoMediaValidationError(
                    f"generated MP4 must contain exactly one video stream; got {len(video_streams)}"
                )
            audio_streams = tuple(container.streams.audio)
            if require_silent and audio_streams:
                raise VideoMediaValidationError("generated MP4 must not contain an audio stream")
            stream = video_streams[0]
            codec_context = stream.codec_context
            width = int(getattr(codec_context, "width", 0) or getattr(stream, "width", 0) or 0)
            height = int(getattr(codec_context, "height", 0) or getattr(stream, "height", 0) or 0)
            if width <= 0 or height <= 0:
                raise VideoMediaValidationError("generated MP4 has invalid dimensions")
            rate = (
                getattr(stream, "average_rate", None)
                or getattr(stream, "guessed_rate", None)
                or getattr(stream, "base_rate", None)
            )
            if rate is None:
                raise VideoMediaValidationError("generated MP4 does not declare a frame rate")
            fps = float(rate)
            if not math.isfinite(fps) or fps <= 0:
                raise VideoMediaValidationError("generated MP4 has invalid frame rate")

            decoded_count = 0
            for frame in container.decode(stream):
                if int(frame.width) != width or int(frame.height) != height:
                    raise VideoMediaValidationError(
                        "generated MP4 contains frames with inconsistent dimensions"
                    )
                decoded_count += 1
            if decoded_count <= 0:
                raise VideoMediaValidationError("generated MP4 contains no decodable frames")
            stream_codec = getattr(stream, "codec", None)
            codec_name = str(
                getattr(codec_context, "name", None)
                or getattr(stream_codec, "name", None)
                or "unknown"
            )
            pixel_format = (
                str(codec_context.format.name)
                if getattr(codec_context, "format", None) is not None
                else None
            )
    except VideoMediaValidationError:
        raise
    except Exception as exc:
        raise VideoMediaValidationError("generated file is not a decodable MP4") from exc

    _require_equal("width", width, expected_width)
    _require_equal("height", height, expected_height)
    _require_equal("num_frames", decoded_count, expected_num_frames)
    if expected_fps is not None and not math.isclose(
        fps,
        float(expected_fps),
        rel_tol=1e-4,
        abs_tol=1e-3,
    ):
        raise VideoMediaValidationError(
            f"generated MP4 fps mismatch: expected {expected_fps}, got {fps}"
        )

    size_bytes = int(media_path.stat(follow_symlinks=False).st_size)
    return VideoMediaMetadata(
        width=width,
        height=height,
        num_frames=decoded_count,
        fps=fps,
        duration_s=float(decoded_count) / fps,
        codec_name=codec_name,
        pixel_format=pixel_format,
        size_bytes=size_bytes,
    )


def _validate_output_path(path: str | Path) -> Path:
    output = Path(path)
    if output.suffix.lower() != ".mp4":
        raise ValueError("video output path must end with .mp4")
    if not output.is_absolute():
        raise ValueError("video output path must be absolute")
    try:
        parent_info = output.parent.lstat()
    except FileNotFoundError as exc:
        raise ValueError("video output parent directory does not exist") from exc
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise ValueError("video output parent must be a real directory")
    if os.path.lexists(output):
        info = output.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise VideoMediaEncodingError("video output target must be a regular file")
    return output


def _validate_existing_mp4(
    path: str | Path,
    *,
    max_size_bytes: int | None,
) -> Path:
    media_path = Path(path)
    if media_path.suffix.lower() != ".mp4":
        raise VideoMediaValidationError("generated video path must end with .mp4")
    if not media_path.is_absolute():
        raise VideoMediaValidationError("generated video path must be absolute")
    try:
        info = media_path.lstat()
    except FileNotFoundError as exc:
        raise VideoMediaValidationError("generated MP4 does not exist") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise VideoMediaValidationError("generated MP4 must be a regular non-symlink file")
    if int(info.st_size) <= 0:
        raise VideoMediaValidationError("generated MP4 is empty")
    if max_size_bytes is not None:
        maximum = _validate_positive_int(max_size_bytes, "max_size_bytes")
        if int(info.st_size) > maximum:
            raise VideoMediaValidationError(f"generated MP4 exceeds maximum size {maximum} bytes")
    return media_path


def _require_equal(field_name: str, actual: int, expected: int | None) -> None:
    if expected is not None and actual != int(expected):
        raise VideoMediaValidationError(
            f"generated MP4 {field_name} mismatch: expected {expected}, got {actual}"
        )


def _validate_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _validate_positive_finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return result


def _truncate_failed_output(path: Path) -> None:
    try:
        if os.path.lexists(path) and stat.S_ISREG(path.lstat().st_mode):
            with path.open("wb"):
                pass
    except OSError:
        # Preserve the original codec/container exception.  The artifact store
        # owns final staging cleanup and will validate the path again.
        pass


def _import_av():
    try:
        import av
    except ImportError as exc:  # pragma: no cover - environment guard
        raise VideoMediaDependencyError(
            "PyAV is required for video encoding; install the project runtime dependencies"
        ) from exc
    return av


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment guard
        raise VideoMediaDependencyError(
            "NumPy is required for video encoding; install the project runtime dependencies"
        ) from exc
    return np


__all__ = [
    "VideoMediaDependencyError",
    "VideoMediaEncodingError",
    "VideoMediaError",
    "VideoMediaMetadata",
    "VideoMediaValidationError",
    "VideoTensorLayout",
    "VideoValueRange",
    "encode_tensor_to_mp4",
    "normalize_video_tensor",
    "validate_mp4",
]
