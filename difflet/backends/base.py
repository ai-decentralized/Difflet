"""Backend runtime contracts: kept as aliases of the platform contract in
``difflet.platforms.base``."""

from __future__ import annotations

from difflet.platforms.base import Platform as BackendRuntime
from difflet.platforms.base import PlatformCapabilities as BackendCapabilities

__all__ = ["BackendCapabilities", "BackendRuntime"]
