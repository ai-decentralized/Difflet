"""What is quantized and how: the spec every MXFP8 path shares. Torch-free.

A :class:`QuantSpec` names the target layers, the compute path and the TP degree the
row-parallel weights were quantized for. The format is fixed: OCP MXFP8 (``float8_e4m3fn``
elements, one E8M0 scale per 32 values along K, for weights offline and for activations inside
the kernel; ``mx.py``). It is JSON-serialisable, so it travels as an application kwarg, into the
quantized checkpoint's manifest and into every cache key (:meth:`QuantSpec.cache_fields`): a
different spec is a different artifact, and no spec adds nothing, so bf16 keys stay
byte-identical.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

#: Bump when the quantized checkpoint layout or the scale law changes: old quantized directories
#: and cache entries are then never matched. 2: mxfp8 only (the per-tensor static ``fp8`` recipe
#: and its calibration are gone). 3: a K remainder of 384 per rank is laid out as the kernel's
#: fast-DMA loader reads it, one partial tile, not the legacy 256 + 128 sub-tiles (``mx.py``;
#: every directory with K per rank a multiple of 512 is byte-identical to schema 2's).
QUANT_SCHEMA = 3

#: The one format: OCP microscaling, 32-value E8M0 block scales for weights (offline) and
#: activations (in the kernel, no calibration), in Trainium3's own block membership (``mx.py``).
FORMAT = "mxfp8"
#: ``dequant``: fp8 weights, bf16 matmul (memory only). ``nki``: the MXFP8 tensor-engine kernel.
COMPUTE_MODES = ("nki", "dequant")
DEFAULT_COMPUTE = "nki"
#: One E8M0 scale per ``BLOCK`` values along K, and the smallest K the swizzle lays out (one
#: 128-element sub-tile, ``mx._tile_sizes``): a weight's K per rank must be a multiple of
#: ``LAYOUT_K``, not merely of ``BLOCK``, or it has no MX layout at all. Defined here, torch-free,
#: so the driver can hold a spec against a checkpoint's shapes without importing ``mx``.
BLOCK = 32
LAYOUT_K = 128
#: The trace-module prefix the neuron Qwen-Image and LTX-2 applications put before diffusers'
#: own module names; target patterns are written against the diffusers names.
STRIPPED_PREFIXES = ("transformer.",)


@dataclasses.dataclass(frozen=True)
class QuantSpec:
    """One quantization recipe for one DiT."""

    targets: tuple[str, ...]  # regex patterns, full-matched against the module name
    compute: str = DEFAULT_COMPUTE
    #: The TP degree the row-parallel weights were quantized for. A row-parallel layer's rank
    #: holds a K range, and MX blocks follow each rank's own K layout, so the quantized weights
    #: of those layers depend on tp (``checkpoint.py``).
    tp_degree: int = 1
    #: Patterns (as ``targets``) of the targets whose weight the TP layer shards along K; at
    #: ``tp_degree > 1`` those are quantized per rank K range.
    row_parallel: tuple[str, ...] = ()
    schema: int = QUANT_SCHEMA

    def __post_init__(self) -> None:
        if int(self.tp_degree) < 1:
            raise ValueError(f"tp_degree must be >= 1, got {self.tp_degree}")
        if self.compute not in COMPUTE_MODES:
            raise ValueError(f"compute must be one of {COMPUTE_MODES}, got {self.compute!r}")
        if not self.targets or any(not t for t in self.targets):
            raise ValueError("a QuantSpec needs at least one non-empty target pattern")
        for pattern in (*self.targets, *self.row_parallel):
            re.compile(pattern)

    @property
    def format(self) -> str:
        return FORMAT

    # ------------------------------------------------------------------ matching

    def matches(self, name: str) -> bool:
        """Whether the module at ``name`` (a dotted path) is quantized. A trace-module prefix
        (``transformer.``) is stripped first, so the same spec serves the diffusers model and the
        wrappers around it."""
        bare = strip_prefix(name)
        return any(re.fullmatch(p, bare) for p in self.targets)

    def unmatched_targets(self, names: Iterable[str]) -> list[str]:
        """The target patterns that match none of ``names`` (as :meth:`matches`, prefix
        stripped). A pattern that matches nothing would silently quantize less than the spec
        claims, and the layers it missed would surface only at the load, as a dtype mismatch
        that names no pattern; so every site that applies a spec refuses when this is
        non-empty, naming the patterns. Torch-free, so the driver can hold a spec against a
        checkpoint's key names before any rank builds a module."""
        bare = [strip_prefix(name) for name in names]
        return [p for p in self.targets if not any(re.fullmatch(p, name) for name in bare)]

    def is_row_parallel(self, name: str) -> bool:
        bare = strip_prefix(name)
        return any(re.fullmatch(p, bare) for p in self.row_parallel)

    def k_shards(self, name: str) -> int:
        """How many K ranges the weight at ``name`` is quantized in: the TP degree for a
        row-parallel target, else 1."""
        return int(self.tp_degree) if self.is_row_parallel(name) else 1

    # ------------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        body = dataclasses.asdict(self)
        body["targets"] = list(self.targets)
        body["row_parallel"] = list(self.row_parallel)
        body["format"] = FORMAT
        return body

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> QuantSpec:
        body = dict(body)
        fmt = body.pop("format", FORMAT)
        if fmt != FORMAT:
            raise ValueError(f"quant format must be {FORMAT!r}, got {fmt!r}")
        body["targets"] = tuple(body["targets"])
        body["row_parallel"] = tuple(body.get("row_parallel", ()))
        return cls(**body)

    def cache_fields(self) -> dict[str, Any]:
        """The additive cache-key fields for an artifact built with this spec: everything that
        changes the compiled graph or the weights. Absent when quantization is off (the caller
        adds nothing), so pre-FP8 keys are unchanged."""
        return {
            "quant": FORMAT,
            "quant_schema": self.schema,
            "quant_compute": self.compute,
            "quant_targets_sha256": self.targets_sha256(),
            "quant_tp_degree": int(self.tp_degree),
        }

    def targets_sha256(self) -> str:
        return hashlib.sha256(json.dumps(list(self.targets)).encode()).hexdigest()[:16]

    def weights_identity(self) -> dict[str, Any]:
        """The fields that decide the quantized checkpoint's bytes (not the compute path): the
        quantized directory is keyed by these, so ``dequant`` and ``nki`` share one directory."""
        return {
            "quant": FORMAT,
            "quant_schema": self.schema,
            "quant_targets": list(self.targets),
            "quant_row_parallel": list(self.row_parallel),
            "quant_tp_degree": int(self.tp_degree),
        }


def strip_prefix(name: str) -> str:
    for prefix in STRIPPED_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def quant_cache_fields(spec: QuantSpec | None) -> dict[str, Any]:
    """``{}`` when quantization is off, else :meth:`QuantSpec.cache_fields`. The one helper every
    key site calls, so a bf16 artifact's key never gains a field."""
    return {} if spec is None else spec.cache_fields()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "BLOCK",
    "COMPUTE_MODES",
    "DEFAULT_COMPUTE",
    "FORMAT",
    "LAYOUT_K",
    "QUANT_SCHEMA",
    "QuantSpec",
    "file_sha256",
    "quant_cache_fields",
    "strip_prefix",
]
