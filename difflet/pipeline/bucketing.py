"""Backend-neutral shape canonicalization for multi-shape compiles.

A request's shape set is normalized, deduped and sorted largest-first here, once,
so the compile cache key, the CLI, serving and every model entry agree on one
deterministic order regardless of backend. The backend-specific half of
bucketing — turning the canonical shapes into example inputs for an AOT
compiler — stays with each backend (``difflet/backends/trainium/core/bucketing.py``
re-exports these names for its callers).

Moved out of the Trainium package by the CUDA backend plan (Phase 1, task 1.5)
so a host without the Neuron SDK can import the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Optional

# (height, width, num_frames); num_frames is None for image models.
CompileShape = tuple[int, int, Optional[int]]


def _normalize_shape(shape) -> CompileShape:
    if isinstance(shape, Mapping):
        height = shape.get("height")
        width = shape.get("width")
        frames = shape.get("num_frames")
    elif isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        if len(shape) == 2:
            height, width = shape
            frames = None
        elif len(shape) == 3:
            height, width, frames = shape
        else:
            raise ValueError(
                f"compile shape must have 2 (h, w) or 3 (h, w, frames) entries, got {shape!r}"
            )
    else:
        raise TypeError(f"compile shape must be a mapping or sequence, got {type(shape)!r}")

    if height is None or width is None:
        raise ValueError(f"compile shape is missing height/width: {shape!r}")
    return (int(height), int(width), None if frames is None else int(frames))


def _shape_sort_key(shape: CompileShape) -> tuple[int, int, int, int]:
    height, width, frames = shape
    volume = height * width * (frames if frames is not None else 1)
    return (volume, height, width, frames if frames is not None else 0)


def canonicalize_shapes(shapes: Iterable) -> tuple[CompileShape, ...]:
    """Normalize, dedupe, and sort a shape collection largest-first."""

    normalized = [_normalize_shape(shape) for shape in shapes]
    if not normalized:
        raise ValueError("compile shape list must not be empty")
    frame_kinds = {shape[2] is None for shape in normalized}
    if len(frame_kinds) > 1:
        raise ValueError(
            "compile shapes must be uniformly (h, w) or uniformly (h, w, frames); "
            f"got a mix: {normalized!r}"
        )
    deduped = tuple(dict.fromkeys(normalized))
    return tuple(sorted(deduped, key=_shape_sort_key, reverse=True))
