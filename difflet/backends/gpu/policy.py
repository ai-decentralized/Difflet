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


__all__ = ["cfg_parallel_does_not_apply", "gpu_platform", "is_set", "unsupported_parallel"]
