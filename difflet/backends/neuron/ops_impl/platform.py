"""Neuron platform target ("trn1", "trn2", ...), in the form the Trainium backend returns it."""

from __future__ import annotations

import os
import re
from enum import Enum
from functools import lru_cache

PLATFORM_OVERRIDE_ENV = "NEURON_PLATFORM_TARGET_OVERRIDE"
_PRODUCT_NAME = "/sys/devices/virtual/dmi/id/product_name"


class hardware(Enum):
    TRN1 = "trn1"
    TRN2 = "trn2"
    TRN3 = "trn3"


@lru_cache(maxsize=None)
def get_platform_target() -> str:
    """Return the platform target string; the override env var wins, then torch_neuronx, then the instance type."""
    override = os.environ.get(PLATFORM_OVERRIDE_ENV)
    if override:
        return override.strip().lower()
    try:
        from torch_neuronx.utils import get_platform_target as _torch_neuronx_target

        return str(_torch_neuronx_target()).strip().lower()
    except (ImportError, AttributeError):
        pass
    target = _target_from_instance_type(_read_product_name())
    if target is None:
        raise RuntimeError(
            f"cannot determine the Neuron platform target; set {PLATFORM_OVERRIDE_ENV} (e.g. trn2)"
        )
    return target


def _target_from_instance_type(product_name: str | None) -> str | None:
    match = re.match(r"\s*(trn\d+)", product_name or "")
    return match.group(1) if match else None


def _read_product_name() -> str | None:
    try:
        with open(_PRODUCT_NAME, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


__all__ = ["get_platform_target", "hardware"]
