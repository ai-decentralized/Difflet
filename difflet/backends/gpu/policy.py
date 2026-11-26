"""Policy shared by the GPU application, CLI and serving: which platform the flow runs on and
which requests it refuses.

Importing this module must not import torch: ``difflet serve``'s parent process refuses
unsupported profiles before any worker starts.
"""

from __future__ import annotations

from typing import Any

from difflet.models.spec import ModelSpec
from difflet.platforms import current_platform, get_platform
from difflet.platforms.base import Platform


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
    CFG-parallel for a model that does not accept it."""
    modes = platform.unsupported_parallel_modes(parallel)
    if (
        getattr(parallel, "cfg_parallel_enabled", False)
        and not spec.accepts_cfg_parallel
        and "cfg" not in modes
    ):
        modes.append("cfg")
    return modes


def cfg_parallel_refusal(spec: ModelSpec) -> str:
    """Why ``--cfg-parallel`` is refused for a model that does not accept it: a true-CFG model
    has the pair but the split is not implemented yet; any other model has no pair at all."""
    if spec.guidance == "cfg":
        return f"--cfg-parallel is not implemented for {spec.display_name} on GPUs yet"
    return (
        f"--cfg-parallel does not apply to {spec.display_name}: it runs one transformer pass per "
        "step, with no conditional/unconditional pair to split"
    )


def guidance_scale_2_refusal(spec: ModelSpec, model_id: str) -> str | None:
    """Why ``guidance_scale_2`` is refused for ``model_id``, or ``None`` when its application
    takes it (``spec.guidance_scale_2_checkpoint_ids``): the one wording ``difflet run`` and
    ``difflet serve`` give before ``generate`` (for ``difflet run``, before any download or
    load). A trailing slash is ignored, as the serving registry ignores it."""
    if model_id.rstrip("/") in spec.guidance_scale_2_checkpoint_ids:
        return None
    return (
        f"guidance_scale_2 has no effect on {model_id}: it has one denoising stage, so every "
        "step uses guidance_scale"
    )


__all__ = [
    "cfg_parallel_refusal",
    "gpu_platform",
    "guidance_scale_2_refusal",
    "is_set",
    "unsupported_parallel",
]
