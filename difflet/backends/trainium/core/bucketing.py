"""Shape-set bucketing: compile K request shapes into ONE artifact.

The NxD v1 ``ModelBuilder`` derives ``bucket_degree`` from the number of
example-input sets passed to ``ModelBuilder.add`` and automatically preserves
one shared weight residency across every bucket NEFF. The entire compile-side
mechanism therefore lives in ``input_generator()``: return K example tuples
instead of one and the builder does the rest. At runtime the NxD router
dispatches by the exact shape signature of the flattened input list, so two
buckets must never share a signature (``dedupe_example_inputs``).

Canonical ordering: shapes are deduped and sorted descending by
``(volume, height, width, frames)`` — largest first — so a wrapper's
``priority_model_idx=0`` always anchors weight-layout optimization to the
largest shape, and manifests/hashes see one deterministic order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Optional

import torch

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


def resolve_compile_shapes(config) -> tuple[CompileShape, ...]:
    """Shape set for a component config: ``compile_shapes`` or the single h/w/f."""

    shapes = getattr(config, "compile_shapes", None)
    if shapes:
        return canonicalize_shapes(shapes)
    num_frames = getattr(config, "num_frames", None)
    return (
        (
            int(config.height),
            int(config.width),
            None if num_frames is None else int(num_frames),
        ),
    )


def example_signature(example: Sequence[torch.Tensor]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(tensor.shape) for tensor in example)


def dedupe_example_inputs(
    examples: list[tuple[torch.Tensor, ...]],
) -> list[tuple[torch.Tensor, ...]]:
    """Drop examples whose full shape signature repeats.

    The NxD router keys on the shapes of the whole flattened input list; two
    buckets with identical signatures would be unreachable/ambiguous. Wrappers
    whose inputs do not depend on the request shape (e.g. a fixed-tile VAE
    decoder) legitimately collapse to a single bucket here.
    """

    seen: set[tuple[tuple[int, ...], ...]] = set()
    unique: list[tuple[torch.Tensor, ...]] = []
    for example in examples:
        signature = example_signature(example)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(example)
    return unique


class ShapeBucketedInputGenerator:
    """Mixin for ModelWrapper subclasses that compile one bucket per shape.

    Subclasses implement ``example_inputs_for_shape`` and inherit an
    ``input_generator`` that expands ``config.compile_shapes`` (falling back to
    the config's single height/width/num_frames) into deduped example inputs,
    largest shape first.
    """

    def example_inputs_for_shape(self, shape: CompileShape) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def input_generator(self) -> list[tuple[torch.Tensor, ...]]:
        return dedupe_example_inputs(
            [self.example_inputs_for_shape(shape) for shape in resolve_compile_shapes(self.config)]
        )
