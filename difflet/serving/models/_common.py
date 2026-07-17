"""Small shared helpers for resident video model adapters."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from difflet.serving.video_media import VideoTensorLayout, VideoValueRange

from difflet.common.neuron_cores import (
    resolve_available_neuron_core_ids,
    select_neuron_core_ids,
)
from difflet.serving.types import (
    DiffletGenerateRequest,
    DistributedProcessEnvironment,
    FileBackedGenerateOutput,
    FileOutputTarget,
    RuntimeEnvironment,
    ServingProfile,
    WorkerAllocationSpec,
)


def require_video_target(request: DiffletGenerateRequest) -> FileOutputTarget:
    if request.output_format != "mp4":
        raise ValueError("video worker requires MP4 output")
    if request.video is None or request.video.output_target is None:
        raise ValueError("video request requires a parent-owned output target")
    target = request.video.output_target
    if not isinstance(target, FileOutputTarget):
        raise TypeError("video output target must be FileOutputTarget")
    if target.mime_type != "video/mp4" or target.output_format != "mp4":
        raise ValueError("video output target must be MP4")
    path = Path(target.staging_path)
    if not path.is_absolute() or not path.name.endswith(".part.mp4"):
        raise ValueError("video output target must be an absolute .part.mp4 path")
    try:
        parent_info = path.parent.lstat()
        target_info = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError("video output target must be pre-created by the parent process") from exc
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise ValueError("video output target parent must be a real directory")
    if stat.S_ISLNK(target_info.st_mode) or not stat.S_ISREG(target_info.st_mode):
        raise ValueError("video output target must be a regular non-symlink file")
    return target


def encode_video_tensor(
    tensor,
    request: DiffletGenerateRequest,
    *,
    layout: VideoTensorLayout,
    value_range: VideoValueRange,
) -> FileBackedGenerateOutput:
    from difflet.serving.video_media import encode_tensor_to_mp4

    target = require_video_target(request)
    assert request.video is not None
    metadata = encode_tensor_to_mp4(
        tensor,
        target.staging_path,
        fps=request.video.fps,
        layout=layout,
        value_range=value_range,
    )
    return FileBackedGenerateOutput(
        path=target.staging_path,
        mime_type="video/mp4",
        output_format="mp4",
        size_bytes=metadata.size_bytes,
        width=metadata.width,
        height=metadata.height,
        num_frames=metadata.num_frames,
        fps=metadata.fps,
        duration_s=metadata.duration_s,
    )


class StartupSmokeTarget:
    """Worker-local output target used only by StagePipelineEngine startup smoke."""

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="difflet-video-smoke-"))
        self.path = self.root / "startup-smoke.part.mp4"
        self.path.touch(mode=0o600)

    def file_target(self) -> FileOutputTarget:
        return FileOutputTarget(staging_path=str(self.path))

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def validate_smoke_output(
    output: object,
    *,
    profile: ServingProfile,
    expected_path: Path,
) -> None:
    from difflet.serving.video_media import validate_mp4

    if profile.num_frames is None or profile.output_fps is None:
        raise RuntimeError("video profile is incomplete")
    if not isinstance(output, FileBackedGenerateOutput):
        raise RuntimeError("startup smoke did not produce file-backed output")
    if output.path != str(expected_path):
        raise RuntimeError("startup smoke wrote outside its temporary target")
    if output.mime_type != "video/mp4" or output.output_format != "mp4":
        raise RuntimeError("startup smoke output is not MP4")
    if output.size_bytes <= 0:
        raise RuntimeError("startup smoke MP4 is empty")
    expected_descriptor = (
        profile.width,
        profile.height,
        profile.num_frames,
        float(profile.output_fps),
    )
    actual_descriptor = (
        output.width,
        output.height,
        output.num_frames,
        float(output.fps),
    )
    if actual_descriptor != expected_descriptor:
        raise RuntimeError("startup smoke output metadata does not match the serving profile")
    expected_duration = float(profile.num_frames) / float(profile.output_fps)
    if not math.isclose(output.duration_s, expected_duration, rel_tol=1e-4, abs_tol=1e-3):
        raise RuntimeError("startup smoke duration metadata does not match the serving profile")
    metadata = validate_mp4(
        expected_path,
        expected_width=profile.width,
        expected_height=profile.height,
        expected_num_frames=profile.num_frames,
        expected_fps=profile.output_fps,
    )
    if metadata.size_bytes != output.size_bytes:
        raise RuntimeError("startup smoke MP4 size metadata mismatch")
    if not math.isclose(metadata.duration_s, output.duration_s, rel_tol=1e-4, abs_tol=1e-3):
        raise RuntimeError("startup smoke MP4 duration metadata mismatch")


def resident_environment(
    profile: ServingProfile,
    *,
    allocation_id: str,
    virtual_core_size: int | None = None,
) -> tuple[RuntimeEnvironment, WorkerAllocationSpec]:
    world_size = profile.world_size
    distributed = DistributedProcessEnvironment(1, 1, 0, 0)
    environment = RuntimeEnvironment(
        available_core_ids=resolve_available_neuron_core_ids(required_num_cores=world_size),
        num_cores_override=None,
        virtual_core_size_override=virtual_core_size,
        logical_nc_config_override=None,
        inherited_distributed=distributed,
        child_distributed=distributed,
    )
    allocation = WorkerAllocationSpec(
        allocation_id=allocation_id,
        requested_num_cores=world_size,
        effective_num_cores=world_size,
        world_size=world_size,
        requested_virtual_core_size=virtual_core_size,
        effective_virtual_core_size=virtual_core_size,
    )
    return environment, allocation


def combined_profile_identity(*digests: str) -> str:
    return hashlib.sha256("\0".join(digests).encode("ascii")).hexdigest()


def compiled_model_payloads_ready(application, artifact_root: str | Path) -> bool:
    """Fail closed when a lower artifact check accepts empty or special files."""

    components = getattr(application, "components", None)
    if not callable(components):
        return False
    specs = tuple(components())
    if not specs:
        return False
    root = Path(artifact_root)
    for spec in specs:
        component_root = root / (getattr(spec, "artifact_name", None) or spec.name)
        for name in ("model.pt", "neuron_config.json"):
            path = component_root / name
            try:
                info = path.lstat()
            except OSError:
                return False
            if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                return False
    return True


@contextmanager
def serving_compile_environment(
    world_size: int,
    *,
    virtual_core_size: int | None = None,
):
    names = (
        "NEURON_RT_VISIBLE_CORES",
        "NEURON_RT_NUM_CORES",
        "NEURON_RT_VIRTUAL_CORE_SIZE",
        "NEURON_LOGICAL_NC_CONFIG",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
    )
    original = {name: os.environ.get(name) for name in names}
    try:
        core_ids = select_neuron_core_ids(required_num_cores=world_size)
        os.environ["NEURON_RT_VISIBLE_CORES"] = ",".join(str(value) for value in core_ids)
        os.environ["NEURON_RT_NUM_CORES"] = str(world_size)
        if virtual_core_size is None:
            os.environ.pop("NEURON_RT_VIRTUAL_CORE_SIZE", None)
        else:
            os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = str(virtual_core_size)
        os.environ.pop("NEURON_LOGICAL_NC_CONFIG", None)
        os.environ.update(
            {"WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}
        )
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


__all__ = [
    "StartupSmokeTarget",
    "combined_profile_identity",
    "compiled_model_payloads_ready",
    "encode_video_tensor",
    "require_video_target",
    "resident_environment",
    "serving_compile_environment",
    "validate_smoke_output",
]
