"""Opt-in dump of denoise-loop latents for offline quality comparison.

Set ``DIFFLET_DUMP_LATENTS_DIR`` to a directory and the denoise loops write
their initial and final latents there as ``<name>.pt`` (CPU tensors, model
dtype preserved). An environment variable rather than a CLI flag so it reaches
staged subprocesses unchanged. Unset, it is a no-op.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

DUMP_LATENTS_ENV = "DIFFLET_DUMP_LATENTS_DIR"


def dump_latents(name: str, tensor: torch.Tensor) -> None:
    out_dir = os.environ.get(DUMP_LATENTS_ENV)
    if not out_dir:
        return
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path / f"{name}.pt")
