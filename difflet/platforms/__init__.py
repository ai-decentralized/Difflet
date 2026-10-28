"""Platform selection: ``DIFFLET_BACKEND`` when set, otherwise auto-detection.

``current_platform()`` is the one place that knows which backend is running; everything else
reads its ``capabilities`` or calls its methods.
"""

from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache

from difflet import envs
from difflet.platforms.base import (
    PARALLEL_MODE_FLAGS,
    PARALLEL_MODES,
    Platform,
    PlatformCapabilities,
)

PLATFORM_FACTORIES = {
    "cpu": "difflet.platforms.cpu:create_platform",
    "neuron": "difflet.platforms.neuron:create_platform",
    "trainium": "difflet.platforms.trainium:create_platform",
    "tpu": "difflet.platforms.tpu:create_platform",
    "cuda": "difflet.platforms.cuda:create_platform",
    "rocm": "difflet.platforms.rocm:create_platform",
}

#: Auto-detection order. Trainium stays first: Neuron venvs also ship torch_xla, and a host with
#: torch_neuronx is a Trainium host whatever else it has.
#: ``neuron`` (torch-neuronx without torch_xla) before ``trainium`` (with it), as the backend registry.
_DETECTION_ORDER = ("neuron", "trainium", "tpu", "cuda", "rocm")


def resolve_platform_name(name: str | None = None) -> str:
    value = name or envs.DIFFLET_BACKEND
    if value:
        normalized = value.strip().lower()
        if normalized in PLATFORM_FACTORIES:
            return normalized
        known = ", ".join(sorted(PLATFORM_FACTORIES))
        raise ValueError(f"unknown Difflet backend {value!r}; known backends: {known}")
    return detect_platform_name()


def get_platform(name: str | None = None) -> Platform:
    return _platform_by_name(resolve_platform_name(name))


def current_platform() -> Platform:
    return get_platform()


def platform_names(*, flow: str) -> tuple[str, ...]:
    """The platforms that run ``flow`` (for example every GPU platform)."""
    return tuple(
        name for name in PLATFORM_FACTORIES if _platform_class(name).capabilities.flow == flow
    )


@lru_cache(maxsize=None)
def _platform_by_name(name: str) -> Platform:
    module_name, _, attr_name = PLATFORM_FACTORIES[name].partition(":")
    platform = getattr(importlib.import_module(module_name), attr_name)()
    if platform.name != name:
        raise RuntimeError(f"platform factory for {name!r} returned platform {platform.name!r}")
    return platform


def _platform_class(name: str) -> type[Platform]:
    return type(_platform_by_name(name))


def detect_platform_name() -> str:
    for name in _DETECTION_ORDER[:3]:
        if _platform_class(name).detect():
            return name
    try:
        import torch  # noqa: F401
    except ImportError:
        # Difflet is Trainium-first. Falling back to Trainium keeps unit tests and local
        # metadata paths usable on hosts without torch.
        return "trainium"
    for name in _DETECTION_ORDER[3:]:
        if _platform_class(name).detect():
            return name
    # No accelerator found: keep the historical Trainium default rather than failing before
    # model resolution.
    return "trainium"


__all__ = [
    "PARALLEL_MODES",
    "PARALLEL_MODE_FLAGS",
    "PLATFORM_FACTORIES",
    "Platform",
    "PlatformCapabilities",
    "current_platform",
    "detect_platform_name",
    "get_platform",
    "platform_names",
    "resolve_platform_name",
]
