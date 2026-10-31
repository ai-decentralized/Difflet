"""Policy shared by the GPU application, CLI and serving: which platform the flow runs on and
which requests it refuses.

Importing this module must not import torch: ``difflet serve``'s parent process refuses
unsupported profiles before any worker starts.
"""

from __future__ import annotations

from typing import Any
import operator

from difflet.models.spec import ModelSpec
from difflet.platforms import current_platform, get_platform
from difflet.platforms.base import PARALLEL_MODE_FLAGS, Platform


def gpu_platform() -> Platform:
    """The GPU platform this flow runs on: the current one, or CUDA when the flow is exercised
    from another platform (unit tests and tooling on a host without a GPU)."""
    platform = current_platform()
    return platform if platform.capabilities.flow == "gpu" else get_platform("cuda")


def is_set(value: Any) -> bool:
    """An option carries a request: not None, not False, not an empty sequence."""
    if value is None or value is False:
        return False
    if isinstance(value, (list, tuple, str)) and not value:
        return False
    return True


def unsupported_parallel(spec: ModelSpec, parallel: Any, platform: Platform) -> list[str]:
    """The parallel modes ``parallel`` asks for that the platform does not implement, plus
    CFG-parallel for a model with no conditional/unconditional pair to split."""
    modes = platform.unsupported_parallel_modes(parallel)
    if (
        getattr(parallel, "cfg_parallel_enabled", False)
        and not spec.accepts_cfg_parallel
        and "cfg" not in modes
    ):
        modes.append("cfg")
    return modes


def cfg_parallel_does_not_apply(spec: ModelSpec) -> str:
    return (
        f"--cfg-parallel does not apply to {spec.display_name}: it runs one transformer pass per "
        "step, with no conditional/unconditional pair to split"
    )


def validate_gpu_steps(value) -> int:
    """Require a positive integral request without silently coercing or defaulting it."""
    try:
        steps = operator.index(value)
    except TypeError as exc:
        raise ValueError("steps must be a positive integer") from exc
    if isinstance(value, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    return steps


def validate_gpu_shape(spec: ModelSpec, *, height, width, num_frames=None) -> None:
    """Validate declared geometry without torch, before downloads or worker loading."""
    for name, value, multiple in (
        ("height", height, spec.height_multiple),
        ("width", width, spec.width_multiple),
    ):
        try:
            size = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{spec.display_name} {name} must be an integer") from exc
        if isinstance(value, bool) or size <= 0 or size % multiple:
            raise ValueError(
                f"{spec.display_name} {name} must be a positive multiple of {multiple}"
            )
    if spec.output == "video":
        try:
            frames = operator.index(num_frames)
        except TypeError as exc:
            raise ValueError(f"{spec.display_name} num_frames must be an integer") from exc
        if isinstance(num_frames, bool) or frames <= 0 or (frames - 1) % spec.frame_stride:
            raise ValueError(
                f"{spec.display_name} num_frames must be positive and n*{spec.frame_stride}+1"
            )


def _refuse_parallel(spec: ModelSpec, parallel: Any, platform: Platform) -> None:
    modes = unsupported_parallel(spec, parallel, platform)
    if not modes:
        return
    mode = modes[0]
    if mode == "cfg":
        raise NotImplementedError(cfg_parallel_does_not_apply(spec))
    raise NotImplementedError(
        f"{PARALLEL_MODE_FLAGS[mode]} is not supported for {spec.display_name} on the "
        f"{platform.display_name} backend yet"
    )


def _refuse_kwargs(spec: ModelSpec, kwargs: dict[str, Any], platform: Platform) -> None:
    """Refuse an application keyword the model does not take. A key is refused only when it
    carries a request (not None / False): the CLI and DiffletPipeline pass several of them with
    their "off" value."""
    for key, value in kwargs.items():
        if value is None or value is False:
            continue
        if key in spec.refused_application_kwargs or key.startswith(
            spec.refused_application_kwarg_prefixes
        ):
            raise NotImplementedError(
                f"{key} is not supported for {spec.display_name} on the "
                f"{platform.display_name} backend yet ({spec.refused_application_kwargs_reason})"
            )


def validate_gpu_application(spec, *, shape, parallel, kwargs, platform):
    """The same pre-I/O request policy for public factories and direct applications."""
    _refuse_kwargs(spec, kwargs, platform)
    _refuse_parallel(spec, parallel, platform)
    resolved = {
        "height": spec.default_height if shape.get("height") is None else shape["height"],
        "width": spec.default_width if shape.get("width") is None else shape["width"],
        "num_frames": (
            (spec.default_num_frames if shape.get("num_frames") is None else shape["num_frames"])
            if spec.output == "video"
            else None
        ),
    }
    validate_gpu_shape(spec, **resolved)
    return resolved


__all__ = [
    "cfg_parallel_does_not_apply",
    "gpu_platform",
    "is_set",
    "unsupported_parallel",
    "validate_gpu_shape",
    "validate_gpu_application",
]
