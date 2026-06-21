"""Lazy dispatch from difflet.ops to the selected backend implementation."""

from __future__ import annotations

import importlib

from difflet.backends import get_backend


def load_backend_attr(module_name: str, attr_name: str):
    backend = get_backend()
    module = importlib.import_module(
        f"difflet.backends.{backend.name}.ops_impl.{module_name}"
    )
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise NotImplementedError(
            f"backend {backend.name!r} does not implement difflet.ops.{module_name}.{attr_name}"
        ) from exc
