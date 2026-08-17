"""Backend discovery and runtime registry."""

from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache

from difflet import envs
from difflet.backends.base import BackendRuntime

_BACKEND_FACTORIES = {
    "cpu": "difflet.backends.cpu.runtime:create_backend",
    "trainium": "difflet.backends.trainium.runtime:create_backend",
    "tpu": "difflet.backends.tpu.runtime:create_backend",
    "cuda": "difflet.backends.cuda.runtime:create_backend",
    "rocm": "difflet.backends.rocm.runtime:create_backend",
}


def resolve_backend_name(name: str | None = None) -> str:
    value = name or envs.DIFFLET_BACKEND
    if value:
        normalized = value.strip().lower()
        if normalized in _BACKEND_FACTORIES:
            return normalized
        known = ", ".join(sorted(_BACKEND_FACTORIES))
        raise ValueError(f"unknown Difflet backend {value!r}; known backends: {known}")
    return _auto_detect_backend()


def current_backend() -> str:
    return resolve_backend_name()


def get_backend(name: str | None = None) -> BackendRuntime:
    backend_name = resolve_backend_name(name)
    return _get_backend_by_name(backend_name)


@lru_cache(maxsize=None)
def _get_backend_by_name(backend_name: str) -> BackendRuntime:
    module_name, _, attr_name = _BACKEND_FACTORIES[backend_name].partition(":")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    runtime = factory()
    if runtime.name != backend_name:
        raise RuntimeError(
            f"backend factory for {backend_name!r} returned runtime {runtime.name!r}"
        )
    return runtime


def _auto_detect_backend() -> str:
    if importlib.util.find_spec("torch_neuronx") is not None:
        return "trainium"

    # TPU hosts: torch_xla plus the TPU runtime library. The torch_neuronx
    # check above must stay first — Neuron venvs also ship torch_xla but
    # never libtpu, so a Trainium host cannot mis-detect as TPU.
    if (
        importlib.util.find_spec("torch_xla") is not None
        and importlib.util.find_spec("libtpu") is not None
    ):
        return "tpu"

    try:
        import torch
    except ImportError:
        # Difflet is Trainium-first today. Falling back to Trainium keeps unit
        # tests and local metadata paths usable even on hosts without torch.
        return "trainium"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.version, "hip", None):
        return "rocm"

    # Until CUDA/ROCm are implemented, preserve the historical Trainium
    # default instead of failing before model resolution.
    return "trainium"
