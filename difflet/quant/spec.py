"""Quantization spec: what is quantized and how. Backend-neutral, hashable.

Mirrors FastVideo's ``FP8Config`` (``fastvideo/layers/quantization/fp8_config.py``):
FP8 e4m3 absmax scales, no calibration set, weights per-tensor (default) or
per-output-channel, activations quantized dynamically per call (W8A8). Only the
attention q/k/v/out projections and the FFN up/down projections are targets;
patch embedding, time/text embedders, adaLN modulation, norms and ``proj_out``
stay bf16.

Weight-only FP8 (``activation="none"``, fp8 weights dequantized to bf16 at run
time) existed until 2026-10-03 and was removed: FP8 PTQ is always dynamic now,
and legacy dicts / flags asking for weight-only are rejected rather than
silently run as dynamic.
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

# The Wan target set is the default (the first wired model); the per-model
# sets live in difflet.quant.targets. ``to_out.0`` is the diffusers/Difflet
# attention output projection (index 1 is the dropout). A target is a dotted
# suffix or, when it contains ``*``, a glob over the full module name.
from difflet.quant.targets import WAN_TARGETS as DEFAULT_TARGETS  # noqa: E402

# CLI spelling -> spec value.
CLI_FORMATS = {"fp8": FORMAT_FP8_E4M3}
_FORMAT_TO_CLI = {value: key for key, value in CLI_FORMATS.items()}

_WEIGHT_ONLY_REMOVED = (
    "weight-only FP8 (activation 'none' / --quant-act none) was removed on 2026-10-03; "
    "FP8 PTQ is always dynamic (W8A8)"
)


def _reject_weight_only(activation: Any) -> None:
    if activation in (None, "dynamic"):
        return
    raise ValueError(f"{_WEIGHT_ONLY_REMOVED}; got activation={activation!r}")


@dataclass(frozen=True)
class QuantSpec:
    format: str = FORMAT_FP8_E4M3
    weight_granularity: str = "tensor"
    targets: tuple[str, ...] = DEFAULT_TARGETS
    # Path of a calibration JSON (``scripts/ptq_calibrate_activations.py``): per
    # target linear, the input absmax over a real denoising run. When set the
    # activation scales are static per layer (checkpoint ``input_scale`` tensors,
    # no absmax reductions on the device); when None they are dynamic per call.
    calibration: str | None = None

    def __post_init__(self) -> None:
        if self.format not in FORMATS:
            raise ValueError(f"unsupported quant format {self.format!r}; known: {FORMATS}")
        if self.weight_granularity not in GRANULARITIES:
            raise ValueError(
                f"unsupported weight granularity {self.weight_granularity!r}; "
                f"known: {GRANULARITIES}"
            )
        if self.calibration is not None and self.weight_granularity != "tensor":
            # NxD creates the static input_scale parameter only for per-tensor weights.
            raise ValueError("static activation scales (calibration) require weight_granularity='tensor'")
        targets = tuple(str(t) for t in self.targets)
        if not targets or any(not t for t in targets):
            raise ValueError("quant targets must be a non-empty tuple of module-name suffixes")
        object.__setattr__(self, "targets", targets)

    # ---------------------------------------------------------------- activations

    @property
    def activation_scales(self) -> str:
        """``"static"`` (calibrated per-layer constants) or ``"dynamic"`` (per call)."""
        return "static" if self.calibration else "dynamic"

    def calibration_sha(self) -> str:
        """Content hash of the calibration file (16 hex chars); ``FileNotFoundError`` if missing."""
        if not self.calibration:
            raise ValueError("spec has no calibration file")
        with open(self.calibration, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:16]

    def calibration_layers(self) -> dict[str, float]:
        """``{layer name: input absmax}`` from the calibration file."""
        with open(self.calibration or "", "r", encoding="utf-8") as handle:
            data = json.load(handle)
        layers = data.get("layers", data)
        return {name: float(entry["amax"] if isinstance(entry, dict) else entry) for name, entry in layers.items()}

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
        data: dict[str, Any] = {
            "format": self.format,
            "weight_granularity": self.weight_granularity,
            "targets": list(self.targets),
        }
        if self.calibration:
            # The content hash is what identifies the artifact; the path locates the file.
            data["calibration"] = self.calibration
            data["calibration_sha"] = self.calibration_sha()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuantSpec":
        _reject_weight_only(data.get("activation"))  # legacy key: "dynamic" tolerated
        return cls(
            format=str(data.get("format", FORMAT_FP8_E4M3)),
            weight_granularity=str(data.get("weight_granularity", "tensor")),
            targets=tuple(data.get("targets") or DEFAULT_TARGETS),
            calibration=data.get("calibration") or None,
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
        """Short human label, e.g. ``fp8-tensor`` / ``fp8-channel`` / ``fp8-tensor-static``."""
        suffix = "-static" if self.calibration else ""
        return f"{_FORMAT_TO_CLI[self.format]}-{self.weight_granularity}{suffix}"

    def checkpoint_identity(self) -> dict[str, Any]:
        """The part of the spec that changes the quantized checkpoint on disk."""
        from difflet.quant.fp8 import FP8_MAX, STATIC_ACT_MARGIN

        identity: dict[str, Any] = {
            "format": self.format,
            # The saturation range is baked into the stored weights: a checkpoint
            # quantized against 448 (torch's e4m3fn max) is NaN on Trainium.
            "fp8_max": FP8_MAX,
            "weight_granularity": self.weight_granularity,
            "targets": list(self.targets),
        }
        if self.calibration:
            identity["calibration_sha"] = self.calibration_sha()
            identity["static_margin"] = STATIC_ACT_MARGIN
        return identity

    def checkpoint_label(self) -> str:
        return self.label()

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
        """Build from ``--quant/--quant-granularity``; None when unset.

        ``model_type`` selects that model's target set; without it the default
        (Wan) targets apply.
        """
        fmt = getattr(args, "quant", None)
        if not fmt:
            return None
        if fmt not in CLI_FORMATS:
            raise ValueError(f"--quant must be one of {sorted(CLI_FORMATS)}, got {fmt!r}")
        _reject_weight_only(getattr(args, "quant_act", None))
        fields = dict(
            format=CLI_FORMATS[fmt],
            weight_granularity=getattr(args, "quant_granularity", None) or "tensor",
            calibration=getattr(args, "quant_calibration", None) or None,
        )
        return cls.for_model(model_type, **fields) if model_type else cls(**fields)

    def cli_args(self) -> list[str]:
        out = [
            "--quant",
            _FORMAT_TO_CLI[self.format],
            "--quant-granularity",
            self.weight_granularity,
        ]
        if self.calibration:
            out += ["--quant-calibration", self.calibration]
        return out


__all__ = [
    "CLI_FORMATS",
    "DEFAULT_TARGETS",
    "FORMATS",
    "FORMAT_FP8_E4M3",
    "GRANULARITIES",
    "QuantSpec",
]
