"""Persistent compile-cache locations for the neuron backend; imports no torch.

torch-neuronx defaults its caches to ``/tmp``, lost on reboot. These helpers move
them under ``<DIFFLET_COMPILE_CACHE>/_neuron`` so they persist and every rank and
process on the host shares them.

``import torch`` autoloads torch_neuronx as a device backend, and torch_neuronx
reads its HLO cache location at import. The locations must therefore be set
before torch is first imported: by a launcher for the processes it starts, or as
the first statement of a script. This module stays importable without torch for
that reason.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

from difflet import envs

CACHE_SUBDIR = "_neuron"
# torch-neuronx cache variables and their subdirectories under the cache root.
CACHE_ENV_DIRS = {
    "TORCH_NEURONX_NEFF_CACHE_DIR": "neff",
    "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": "neff_local",
    "TORCH_NEURONX_HLO_CACHE_DIR": "hlo",
}


def neuron_cache_env(root: str | os.PathLike | None = None) -> dict[str, str]:
    """Cache locations under ``<root>/_neuron`` (root: DIFFLET_COMPILE_CACHE); variables already set win."""
    base = Path(os.path.expanduser(str(root or envs.DIFFLET_COMPILE_CACHE))) / CACHE_SUBDIR
    return {var: os.environ.get(var) or str(base / sub) for var, sub in CACHE_ENV_DIRS.items()}


def apply_neuron_cache_env(root: str | os.PathLike | None = None) -> dict[str, str]:
    """Export the cache locations and create the directories; call before ``import torch``."""
    env = neuron_cache_env(root)
    for var, path in env.items():
        os.environ[var] = path
        Path(path).mkdir(parents=True, exist_ok=True)
    if "torch_neuronx" in sys.modules or "torch" in sys.modules:
        warnings.warn(
            "torch was imported before the neuron cache locations were set (it autoloads "
            "torch_neuronx); TORCH_NEURONX_HLO_CACHE_DIR takes effect only in processes "
            "started afterwards",
            stacklevel=2,
        )
    return env


__all__ = ["CACHE_ENV_DIRS", "CACHE_SUBDIR", "apply_neuron_cache_env", "neuron_cache_env"]
