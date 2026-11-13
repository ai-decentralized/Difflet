"""The pipeline applications' ``quant`` option, resolved to a spec and the DiT's directory.

Why one function: Qwen-Image and FLUX each resolved ``quant`` in a method of their own, and
every model wired after them would have copied it. The option has three spellings (``"mxfp8"``,
a ``QuantSpec``, its dict) and the directory two sources (the compile cache, an explicit
``quant_checkpoint``), bound by rules that must not drift between models: the default spec is
built at the launch's tp, because a row-parallel weight is quantized per rank K range; the
directory is never written here (every rank of a ``generate`` resolves, ``difflet compile``
writes on the driver before torchrun starts, ``difflet/cli/quant.py``); an explicit directory is
checked against the spec and the source checkpoint; a directory without a spec is an error.
"""

from __future__ import annotations

from difflet.quant.checkpoint import check_quantized_dir, prepare_quantized_checkpoint
from difflet.quant.spec import FORMAT, QuantSpec
from difflet.quant.targets import default_spec


def resolve_quant(model: str, quant, quant_checkpoint, transformer_path: str, *,
                  tp_degree: int) -> tuple[QuantSpec | None, str]:
    """``(spec, DiT directory)`` for a pipeline application's ``quant`` and ``quant_checkpoint``
    options, or ``(None, transformer_path)`` when quantization is off.

    ``quant`` is ``"mxfp8"`` (``model``'s default target table, ``difflet.quant.targets``, at
    ``tp_degree``), a ``QuantSpec`` or its dict. ``quant_checkpoint`` names the quantized
    directory (``check_quantized_dir``); without it the directory is the one under the compile
    cache for ``transformer_path`` and the spec, which ``difflet compile`` must have written."""
    if quant is None:
        if quant_checkpoint is not None:
            raise ValueError("quant_checkpoint needs quant ('mxfp8' or a QuantSpec)")
        return None, transformer_path
    if isinstance(quant, str):
        if quant != FORMAT:
            raise ValueError(f"quant must be {FORMAT!r}, a QuantSpec or its dict, got {quant!r}")
        spec = default_spec(model, tp_degree=int(tp_degree))
    elif isinstance(quant, dict):
        spec = QuantSpec.from_dict(quant)
    elif isinstance(quant, QuantSpec):
        spec = quant
    else:
        raise TypeError(f"quant must be {FORMAT!r}, a QuantSpec or its dict, got {type(quant)}")
    if quant_checkpoint is None:
        quant_checkpoint = prepare_quantized_checkpoint(transformer_path, spec, create=False)
    else:
        check_quantized_dir(quant_checkpoint, transformer_path, spec)
    return spec, str(quant_checkpoint)


__all__ = ["resolve_quant"]
