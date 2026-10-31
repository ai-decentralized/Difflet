"""What is quantized and how: the spec every FP8 path shares. Torch-free.

A :class:`QuantSpec` names the format (OCP ``float8_e4m3fn``, max 448), the weight scale
granularity, how activations are scaled, the compute path and the target layers. It is
JSON-serialisable, so it travels as an application kwarg, into the quantized checkpoint's
manifest and into every cache key (:meth:`QuantSpec.cache_fields`): a different spec is a
different artifact, and no spec adds nothing, so bf16 keys stay byte-identical.

Activation scaling is static in v1: ``input_scale = amax * margin / fp8_max`` from a
calibration file (``calibration.py``), because the dynamic per-tensor variant measured far worse
on the Qwen-Image DiT (about 13 dB against 33 dB latent SNR on the AoT backend) and because a
static scale keeps the compiled block graph free of a per-call reduction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any

#: Bump when the quantized checkpoint layout or the scale law changes: old quantized directories
#: and cache entries are then never matched.
QUANT_SCHEMA = 1

FORMATS = ("fp8",)
WEIGHT_GRANULARITIES = ("tensor",)
ACTIVATION_MODES = ("static",)
#: ``dequant``: fp8 weights, bf16 matmul (memory only). ``nki``: the FP8 tensor-engine kernel.
COMPUTE_MODES = ("nki", "dequant")
DEFAULT_COMPUTE = "nki"
#: OCP E4M3FN: the format Trainium3 computes natively. (Trainium2's legacy E4M3 stops at 240.)
FP8_MAX = 448.0
#: Head-room over the calibrated activation amax before the clamp bites.
DEFAULT_ACTIVATION_MARGIN = 1.25
#: The trace-module prefix the neuron Qwen-Image and LTX-2 applications put before diffusers'
#: own module names; target patterns are written against the diffusers names.
STRIPPED_PREFIXES = ("transformer.",)


@dataclasses.dataclass(frozen=True)
class QuantSpec:
    """One quantization recipe for one DiT."""

    targets: tuple[str, ...]  # regex patterns, full-matched against the module name
    format: str = "fp8"
    fp8_max: float = FP8_MAX
    weight_granularity: str = "tensor"
    activation: str = "static"
    activation_margin: float = DEFAULT_ACTIVATION_MARGIN
    compute: str = DEFAULT_COMPUTE
    #: sha256 of the calibration file's bytes: the activation scales are part of the identity.
    calibration_sha256: str | None = None
    schema: int = QUANT_SCHEMA

    def __post_init__(self) -> None:
        if self.format not in FORMATS:
            raise ValueError(f"quant format must be one of {FORMATS}, got {self.format!r}")
        if self.weight_granularity not in WEIGHT_GRANULARITIES:
            raise ValueError(
                f"weight_granularity must be one of {WEIGHT_GRANULARITIES}, "
                f"got {self.weight_granularity!r}"
            )
        if self.activation not in ACTIVATION_MODES:
            raise ValueError(
                f"activation must be one of {ACTIVATION_MODES}, got {self.activation!r}"
            )
        if self.compute not in COMPUTE_MODES:
            raise ValueError(f"compute must be one of {COMPUTE_MODES}, got {self.compute!r}")
        if not self.targets or any(not t for t in self.targets):
            raise ValueError("a QuantSpec needs at least one non-empty target pattern")
        if not float(self.fp8_max) > 0:
            raise ValueError(f"fp8_max must be positive, got {self.fp8_max}")
        if not float(self.activation_margin) >= 1.0:
            raise ValueError(f"activation_margin must be >= 1, got {self.activation_margin}")
        for pattern in self.targets:
            re.compile(pattern)

    # ------------------------------------------------------------------ matching

    def matches(self, name: str) -> bool:
        """Whether the module at ``name`` (a dotted path) is quantized. A trace-module prefix
        (``transformer.``) is stripped first, so the same spec serves the diffusers model and the
        wrappers around it."""
        bare = strip_prefix(name)
        return any(re.fullmatch(p, bare) for p in self.targets)

    # ------------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        body = dataclasses.asdict(self)
        body["targets"] = list(self.targets)
        return body

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> QuantSpec:
        body = dict(body)
        body["targets"] = tuple(body["targets"])
        return cls(**body)

    def cache_fields(self) -> dict[str, Any]:
        """The additive cache-key fields for an artifact built with this spec: everything that
        changes the compiled graph or the weights. Absent when quantization is off (the caller
        adds nothing), so pre-FP8 keys are unchanged."""
        return {
            "quant": self.format,
            "quant_schema": self.schema,
            "quant_fp8_max": self.fp8_max,
            "quant_weight_granularity": self.weight_granularity,
            "quant_activation": self.activation,
            "quant_activation_margin": self.activation_margin,
            "quant_compute": self.compute,
            "quant_targets_sha256": self.targets_sha256(),
            "quant_calibration_sha256": self.calibration_sha256,
        }

    def targets_sha256(self) -> str:
        return hashlib.sha256(json.dumps(list(self.targets)).encode()).hexdigest()[:16]

    def weights_identity(self) -> dict[str, Any]:
        """The fields that decide the quantized checkpoint's bytes (not the compute path): the
        quantized directory is keyed by these, so ``dequant`` and ``nki`` share one directory."""
        return {
            "quant": self.format,
            "quant_schema": self.schema,
            "quant_fp8_max": self.fp8_max,
            "quant_weight_granularity": self.weight_granularity,
            "quant_activation": self.activation,
            "quant_activation_margin": self.activation_margin,
            "quant_targets": list(self.targets),
            "quant_calibration_sha256": self.calibration_sha256,
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
    "ACTIVATION_MODES",
    "COMPUTE_MODES",
    "DEFAULT_ACTIVATION_MARGIN",
    "DEFAULT_COMPUTE",
    "FORMATS",
    "FP8_MAX",
    "QUANT_SCHEMA",
    "QuantSpec",
    "WEIGHT_GRANULARITIES",
    "file_sha256",
    "quant_cache_fields",
    "strip_prefix",
]
