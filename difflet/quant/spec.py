"""Quantization spec: what is quantized and how. Backend-neutral, hashable.

Mirrors FastVideo's ``FP8Config`` (``fastvideo/layers/quantization/fp8_config.py``):
FP8 e4m3 absmax scales, no calibration set, weights per-tensor (default) or
per-output-channel, activations quantized dynamically per call or left in
bf16 (weight-only). Only the attention q/k/v/out projections and the FFN
up/down projections are targets; patch embedding, time/text embedders, adaLN
modulation, norms and ``proj_out`` stay bf16.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

FORMAT_FP8_E4M3 = "fp8_e4m3"
FORMATS = (FORMAT_FP8_E4M3,)
GRANULARITIES = ("tensor", "channel")
ACTIVATIONS = ("dynamic", "none")

# The Wan target set is the default (the first wired model); the per-model
# sets live in difflet.quant.targets. ``to_out.0`` is the diffusers/Difflet
# attention output projection (index 1 is the dropout). A target is a dotted
# suffix or, when it contains ``*``, a glob over the full module name.
from difflet.quant.targets import WAN_TARGETS as DEFAULT_TARGETS  # noqa: E402

# CLI spelling -> spec value.
CLI_FORMATS = {"fp8": FORMAT_FP8_E4M3}
_FORMAT_TO_CLI = {value: key for key, value in CLI_FORMATS.items()}


@dataclass(frozen=True)
class QuantSpec:
    format: str = FORMAT_FP8_E4M3
    weight_granularity: str = "tensor"
    activation: str = "dynamic"
    targets: tuple[str, ...] = DEFAULT_TARGETS

    def __post_init__(self) -> None:
        if self.format not in FORMATS:
            raise ValueError(f"unsupported quant format {self.format!r}; known: {FORMATS}")
        if self.weight_granularity not in GRANULARITIES:
            raise ValueError(
                f"unsupported weight granularity {self.weight_granularity!r}; "
                f"known: {GRANULARITIES}"
            )
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"unsupported activation mode {self.activation!r}; known: {ACTIVATIONS}"
            )
        targets = tuple(str(t) for t in self.targets)
        if not targets or any(not t for t in targets):
            raise ValueError("quant targets must be a non-empty tuple of module-name suffixes")
        object.__setattr__(self, "targets", targets)

    # ---------------------------------------------------------------- matching

    def matches(self, module_name: str) -> bool:
        """True when ``module_name`` (dotted, no trailing ``.weight``) is a target.

        Globs (``*`` in the pattern) match the whole name; because the Qwen /
        LTX-2 device names carry a ``transformer.`` prefix, a glob also
        matches with any dotted prefix in front of it.
        """
        for t in self.targets:
            if "*" in t:
                if fnmatchcase(module_name, t) or fnmatchcase(module_name, "*." + t):
                    return True
            elif module_name == t or module_name.endswith("." + t):
                return True
        return False

    @classmethod
    def for_model(cls, model_type: str, **fields: Any) -> "QuantSpec":
        """A spec with ``model_type``'s target set (``ValueError`` if not wired)."""
        from difflet.quant.targets import targets_for

        return cls(targets=targets_for(model_type), **fields)

    # ------------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "weight_granularity": self.weight_granularity,
            "activation": self.activation,
            "targets": list(self.targets),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantSpec":
        return cls(
            format=str(data.get("format", FORMAT_FP8_E4M3)),
            weight_granularity=str(data.get("weight_granularity", "tensor")),
            activation=str(data.get("activation", "dynamic")),
            targets=tuple(data.get("targets") or DEFAULT_TARGETS),
        )

    @classmethod
    def coerce(cls, value: "QuantSpec | dict[str, Any] | None") -> "QuantSpec | None":
        if value is None:
            return None
        if isinstance(value, QuantSpec):
            return value
        if isinstance(value, dict):
            return cls.from_dict(value)
        raise TypeError(f"quant must be a QuantSpec, dict or None, got {type(value).__name__}")

    # ------------------------------------------------------------------ identity

    def label(self) -> str:
        """Short human label, e.g. ``fp8-tensor-dyn`` / ``fp8-channel-wo``."""
        act = "dyn" if self.activation == "dynamic" else "wo"
        return f"{_FORMAT_TO_CLI[self.format]}-{self.weight_granularity}-{act}"

    def checkpoint_identity(self) -> dict[str, Any]:
        """The part of the spec that changes the quantized *weights* on disk.

        The activation mode is a graph-time choice, so one checkpoint serves
        both ``dynamic`` and ``none``.
        """
        from difflet.quant.fp8 import FP8_MAX

        return {
            "format": self.format,
            # The saturation range is baked into the stored weights: a checkpoint
            # quantized against 448 (torch's e4m3fn max) is NaN on Trainium.
            "fp8_max": FP8_MAX,
            "weight_granularity": self.weight_granularity,
            "targets": list(self.targets),
        }

    def checkpoint_label(self) -> str:
        return f"{_FORMAT_TO_CLI[self.format]}-{self.weight_granularity}"

    def checkpoint_hash(self, source: str | None = None) -> str:
        payload = dict(self.checkpoint_identity())
        if source is not None:
            payload["source"] = source
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]

    # ----------------------------------------------------------------------- CLI

    @classmethod
    def from_args(
        cls, args: argparse.Namespace, model_type: str | None = None
    ) -> "QuantSpec | None":
        """Build from ``--quant/--quant-granularity/--quant-act``; None when unset.

        ``model_type`` selects that model's target set; without it the default
        (Wan) targets apply.
        """
        fmt = getattr(args, "quant", None)
        if not fmt:
            return None
        if fmt not in CLI_FORMATS:
            raise ValueError(f"--quant must be one of {sorted(CLI_FORMATS)}, got {fmt!r}")
        fields = dict(
            format=CLI_FORMATS[fmt],
            weight_granularity=getattr(args, "quant_granularity", None) or "tensor",
            activation=getattr(args, "quant_act", None) or "dynamic",
        )
        return cls.for_model(model_type, **fields) if model_type else cls(**fields)

    def cli_args(self) -> list[str]:
        return [
            "--quant",
            _FORMAT_TO_CLI[self.format],
            "--quant-granularity",
            self.weight_granularity,
            "--quant-act",
            self.activation,
        ]


__all__ = [
    "ACTIVATIONS",
    "CLI_FORMATS",
    "DEFAULT_TARGETS",
    "FORMATS",
    "FORMAT_FP8_E4M3",
    "GRANULARITIES",
    "QuantSpec",
]
