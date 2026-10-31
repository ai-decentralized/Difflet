"""Static activation calibration: the absmax each quantized linear's input reaches.

:class:`ActivationCalibrator` hooks the target linears of a bf16 (or fp32) model and keeps, per
module, the largest ``|x|`` seen over every forward of a calibration run (several prompts, every
denoising step: a DiT's activation range moves with the timestep, so one step is not enough).
:func:`write_calibration` stores the result as JSON beside its provenance (what model, which
checkpoint, which prompts, how many steps, at what shape), and :class:`Calibration` reads it
back and turns an amax into the static ``input_scale`` the spec asks for.

The file's sha256 is part of the spec (``QuantSpec.calibration_sha256``) and so of every cache
key: two calibrations are two sets of weights-plus-scales, even for the same checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from difflet.quant.fp8 import activation_scale
from difflet.quant.spec import QuantSpec, file_sha256, strip_prefix

CALIBRATION_SCHEMA = 1


class ActivationCalibrator:
    """Forward pre-hooks on every module of ``model`` the ``spec`` targets, recording the
    running absmax of the first positional input. Use as a context manager, then
    :meth:`amax` (keyed by the module name with the trace prefix stripped)."""

    def __init__(self, model: nn.Module, spec: QuantSpec) -> None:
        self.spec = spec
        self._amax: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._handles: list[Any] = []
        self._targets = [
            (strip_prefix(name), module) for name, module in model.named_modules()
            if spec.matches(name) and isinstance(module, nn.Linear)
        ]
        if not self._targets:
            raise ValueError("the spec's target patterns match no nn.Linear in the model")

    def __enter__(self) -> ActivationCalibrator:
        for name, module in self._targets:
            self._handles.append(module.register_forward_pre_hook(self._hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, name: str):
        def record(module, args):
            x = args[0]
            value = float(x.detach().abs().amax().float().cpu())
            if value > self._amax.get(name, 0.0):
                self._amax[name] = value
            self._calls[name] = self._calls.get(name, 0) + 1
        return record

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self._targets]

    def amax(self) -> dict[str, float]:
        missing = [n for n in self.names if n not in self._amax]
        if missing:
            raise ValueError(f"{len(missing)} target linears saw no forward, e.g. {missing[:3]}")
        return dict(sorted(self._amax.items()))

    def calls(self) -> dict[str, int]:
        return dict(sorted(self._calls.items()))


@dataclasses.dataclass(frozen=True)
class Calibration:
    """A calibration file, read."""

    path: Path
    amax: dict[str, float]
    provenance: dict[str, Any]
    sha256: str

    def input_scale(self, name: str, spec: QuantSpec) -> torch.Tensor:
        """fp32 ``[1]`` static scale for the linear at ``name`` (prefix stripped)."""
        key = strip_prefix(name)
        try:
            amax = self.amax[key]
        except KeyError:
            raise KeyError(
                f"calibration {self.path} has no activation amax for {key!r}; it was "
                "collected on another model or another target set"
            ) from None
        return activation_scale(amax, margin=spec.activation_margin, fp8_max=spec.fp8_max)

    def covers(self, names) -> list[str]:
        """The names (prefix stripped) that the file has no amax for."""
        return [n for n in (strip_prefix(x) for x in names) if n not in self.amax]


def write_calibration(path: str | Path, *, amax: dict[str, float], model: str,
                      checkpoint: dict[str, Any], prompts: list[str], steps: int,
                      shape: dict[str, Any], calls: dict[str, int] | None = None,
                      extra: dict[str, Any] | None = None) -> Path:
    """Write the calibration JSON. ``checkpoint`` identifies the weights the amax were collected
    on (model id, snapshot or path) so a mismatch can be refused at load."""
    path = Path(path)
    body = {
        "schema": CALIBRATION_SCHEMA,
        "model": model,
        "checkpoint": checkpoint,
        "prompts": list(prompts),
        "steps": int(steps),
        "shape": dict(shape),
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(extra or {}),
        "calls": dict(sorted((calls or {}).items())),
        "amax": dict(sorted(amax.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_calibration(path: str | Path) -> Calibration:
    path = Path(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(
            f"{path}: calibration schema {body.get('schema')!r}, expected {CALIBRATION_SCHEMA}"
        )
    amax = body.get("amax")
    if not isinstance(amax, dict) or not amax:
        raise ValueError(f"{path}: no 'amax' table")
    bad = [k for k, v in amax.items() if not (isinstance(v, (int, float)) and v >= 0)]
    if bad:
        raise ValueError(f"{path}: non-numeric or negative amax for {bad[:3]}")
    provenance = {k: v for k, v in body.items() if k not in ("amax", "calls")}
    return Calibration(path=path, amax={k: float(v) for k, v in amax.items()},
                       provenance=provenance, sha256=file_sha256(path))


__all__ = [
    "CALIBRATION_SCHEMA",
    "ActivationCalibrator",
    "Calibration",
    "read_calibration",
    "write_calibration",
]
