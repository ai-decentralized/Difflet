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

import torch

# The backend-neutral half (CompileShape, canonicalize_shapes) lives in
# difflet.pipeline.bucketing since the CUDA backend plan (Phase 1, task 1.5);
# re-exported here so on-device scripts and serving code keep their import path.
from difflet.pipeline.bucketing import (  # noqa: F401  (re-export)
    CompileShape,
    _normalize_shape,
    _shape_sort_key,
    canonicalize_shapes,
)


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
