"""The FP8-PTQ members every multi-component application shares.

Lifted from the Wan application so FLUX, Qwen-Image, HunyuanVideo and LTX-2
carry one implementation: the spec (with the model's own target set), the
quantized-checkpoint directory per transformer subfolder (memoized), and the
"make sure the fp8 copy exists" step that ``compile`` runs before tracing and
``generate`` / ``serve`` run before loading.
"""

from __future__ import annotations

import os
from typing import Any

from difflet.quant.checkpoint import ensure_quantized_checkpoint, quantized_checkpoint_dir
from difflet.quant.spec import QuantSpec


def resolve_quant(
    model_path: str,
    subfolder: str,
    quant: "QuantSpec | dict[str, Any] | None",
    quant_cache_dir: "str | os.PathLike[str] | None",
    *,
    model_type: str,
) -> "tuple[QuantSpec | None, str | None]":
    """``(spec with the model's targets, quantized checkpoint dir)`` or ``(None, None)``.

    For applications that build their backbone config before the application
    object exists (FLUX, LTX-2 factories): the same resolution
    ``QuantApplicationMixin._quant_checkpoint_dir`` performs, as a function.
    """
    spec = QuantSpec.coerce(quant)
    if spec is None:
        return None, None
    spec = QuantSpec.for_model(
        model_type, format=spec.format, weight_granularity=spec.weight_granularity
    )
    source = os.path.join(model_path, subfolder)
    return spec, str(quantized_checkpoint_dir(quant_cache_dir, source, spec))


class QuantApplicationMixin:
    quant_tag: str = "quant"  # print prefix, e.g. "wan", "flux"
    model_path: str

    def _init_quant(self, kwargs: dict[str, Any], *, model_type: str) -> None:
        """Read ``quant`` / ``quant_cache_dir`` from the application kwargs.

        ``quant`` may be a QuantSpec or its dict; the dict from the CLI /
        serving may carry the default (Wan) targets, so the model's own target
        set replaces them. ``quant_cache_dir`` is runtime-only (never hashed).
        """
        spec = QuantSpec.coerce(kwargs.get("quant"))
        self.quant_spec = (
            None
            if spec is None
            else QuantSpec.for_model(
                model_type, format=spec.format, weight_granularity=spec.weight_granularity
            )
        )
        self._quant_cache_dir = kwargs.get("quant_cache_dir")
        self.quant_checkpoint_dirs: dict[str, str] = {}

    def _quant_checkpoint_dir(self, subfolder: str) -> str | None:
        """Where the quantized copy of ``<model_path>/<subfolder>`` lives (None = bf16)."""
        if self.quant_spec is None:
            return None
        if subfolder not in self.quant_checkpoint_dirs:
            source = os.path.join(self.model_path, subfolder)
            self.quant_checkpoint_dirs[subfolder] = str(
                quantized_checkpoint_dir(self._quant_cache_dir, source, self.quant_spec)
            )
        return self.quant_checkpoint_dirs[subfolder]

    def ensure_quantized_checkpoints(self, *, create: bool, force: bool = False) -> dict[str, str]:
        """Make sure every quantized transformer checkpoint exists.

        ``create=True`` (compile, ``difflet quantize``) builds missing copies on
        the CPU; ``create=False`` (generate / serve load) raises with the
        command to run. No-op for a bf16 application.
        """
        if self.quant_spec is None:
            return {}
        resolved: dict[str, str] = {}
        for subfolder, dest in self.quant_checkpoint_dirs.items():
            source = os.path.join(self.model_path, subfolder)
            path = ensure_quantized_checkpoint(
                source, dest, self.quant_spec, create=create, force=force
            )
            resolved[subfolder] = str(path)
            print(
                f"[{self.quant_tag}] quantized checkpoint ({self.quant_spec.checkpoint_label()}) "
                f"for {subfolder}: {path}",
                flush=True,
            )
        return resolved

    def compile(self, compiled_model_path: str, debug: bool = False, select=None) -> None:
        """The trace-time checkpoint loader reads the quantized copy, so it must
        exist before the first component compiles (cooperative: the mixin comes
        first in the MRO, the multi-component base does the actual compile)."""
        self.ensure_quantized_checkpoints(create=True)
        super().compile(compiled_model_path, debug=debug, select=select)  # type: ignore[misc]


__all__ = ["QuantApplicationMixin"]
