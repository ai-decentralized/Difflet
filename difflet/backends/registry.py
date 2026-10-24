"""Backend selection: the platform registry under its historical names."""

from __future__ import annotations

import importlib
import importlib.util

from difflet.platforms import (
    PLATFORM_FACTORIES as _BACKEND_FACTORIES,
)
from difflet.platforms import (
    detect_platform_name as _auto_detect_backend,
)
from difflet.platforms import (
    _platform_by_name as _get_backend_by_name,
)
from difflet.platforms import (
    get_platform as get_backend,
)
from difflet.platforms import (
    resolve_platform_name as resolve_backend_name,
)


def current_backend() -> str:
    return resolve_backend_name()


__all__ = [
    "_BACKEND_FACTORIES",
    "_auto_detect_backend",
    "_get_backend_by_name",
    "current_backend",
    "get_backend",
    "importlib",
    "resolve_backend_name",
]
