"""Replica HBM fit check: weights-only, header-only safetensors scan (spec §CLI).

The limit is per platform: one Trainium2 chip, or the GPU's memory on a GPU platform.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

HBM_LIMIT_BYTES = 96_000_000_000  # one Trainium2 chip


def _file_param_count(path: Path) -> int:
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    count = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        count += math.prod(info["shape"]) if info["shape"] else 1
    return count


def component_weight_bytes(model_path: str | Path, dtype_bytes: int = 2) -> dict[str, int]:
    model_path = Path(model_path)
    sizes: dict[str, int] = {}
    for st_file in sorted(model_path.rglob("*.safetensors")):
        component = st_file.relative_to(model_path).parts[0]
        sizes[component] = sizes.get(component, 0) + dtype_bytes * _file_param_count(st_file)
    if not sizes:
        raise RuntimeError(f"no safetensors found under {model_path}")
    return sizes


def replica_limit_bytes() -> int | None:
    """The per-replica device memory for the current platform, or ``None`` when unknown.

    Trainium: one Trainium2 chip (``HBM_LIMIT_BYTES``). A GPU platform: device 0's total memory
    (a replica's weights live on each of its GPUs), unknown when no GPU is visible.
    """
    from difflet.platforms import current_platform

    platform = current_platform()
    if platform.capabilities.flow != "gpu":
        return HBM_LIMIT_BYTES
    return platform.device_memory_bytes()


def assert_replica_fits(
    model_path: str | Path, *, dtype_bytes: int = 2, limit_bytes: int | None = None
) -> None:
    """Raise when the replica's weights exceed ``limit_bytes`` (default: per backend)."""
    if limit_bytes is None:
        limit_bytes = replica_limit_bytes()
        if limit_bytes is None:
            print("[dp-router] no CUDA device visible; skipping HBM fit check", flush=True)
            return
    sizes = component_weight_bytes(model_path, dtype_bytes=dtype_bytes)
    total = sum(sizes.values())
    if total > limit_bytes:
        breakdown = ", ".join(f"{k}={v / 1e9:.1f}GB" for k, v in sorted(sizes.items()))
        raise RuntimeError(
            f"replica weights {total / 1e9:.1f}GB exceed the per-replica HBM "
            f"limit {limit_bytes / 1e9:.0f}GB ({breakdown})"
        )
