"""The software build a measurement ran on, recorded beside every parity and benchmark result."""

from __future__ import annotations

import os
import platform
import subprocess
from importlib import metadata

_PACKAGES = ("torch", "torch-neuronx", "nki", "neuronx-cc", "diffusers", "transformers")
IMAGE_DIGEST_ENV = "DIFFLET_NEURON_IMAGE_DIGEST"


def build_info() -> dict[str, str]:
    """Python and package versions, the Difflet commit, and the image digest if exported."""
    info = {"python": platform.python_version()}
    for name in _PACKAGES:
        try:
            info[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            info[name] = "absent"
    info["difflet_commit"] = _difflet_commit()
    info["image_digest"] = os.environ.get(IMAGE_DIGEST_ENV, "unknown")
    return info


def _difflet_commit() -> str:
    """Short commit of the Difflet checkout, suffixed ``-dirty`` when tracked files are modified."""
    root = os.path.dirname(os.path.abspath(__file__))  # git finds the checkout from any subdirectory
    try:
        head = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        changed = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not head:
        return "unknown"
    return f"{head}-dirty" if changed else head


__all__ = ["IMAGE_DIGEST_ENV", "build_info"]
