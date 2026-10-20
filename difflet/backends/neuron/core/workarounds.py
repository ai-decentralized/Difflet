"""Workarounds for torch-neuronx eager-kernel bugs, installed by the backend's prepare_runtime.

Boolean bitwise ops on non-contiguous operands. Eager ``bitwise_and`` on ``bool`` neuron tensors
returns wrong values when an operand is a non-contiguous view, at sizes such as 256 x 256 (a
64 x 64 case was correct; ``uint8`` and ``int64`` are correct; ``.contiguous()`` on the strided
operand fixes it). diffusers builds HunyuanVideo 1.5's token-refiner mask exactly so,
``m & m.transpose(2, 3)`` with ``m`` a row-repeated padding mask; on the device the valid tokens
then attended to all padding (12,800 mask entries set instead of 2,500) and the output was wrong.
The guard makes the non-contiguous operand of ``&``, ``|`` and ``^`` contiguous when both are
``bool`` and one is on a neuron device (``|`` and ``^`` as a precaution); every other case runs
the original operator unchanged.
"""

from __future__ import annotations

import torch

_OPERATORS = ("__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__",
              "bitwise_and", "bitwise_or", "bitwise_xor")
_FUNCTIONS = ("bitwise_and", "bitwise_or", "bitwise_xor")
_originals: dict[tuple[str, str], object] = {}


def _on_neuron(t) -> bool:
    return isinstance(t, torch.Tensor) and t.device.type == "neuron"


def _contiguous_bool_operands(a, b):
    if (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
            and a.dtype == torch.bool and b.dtype == torch.bool and (_on_neuron(a) or _on_neuron(b))):
        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()
    return a, b


def install_bool_bitwise_guard() -> None:
    """Wrap the boolean bitwise operators (idempotent)."""
    for name in _OPERATORS:
        if ("Tensor", name) in _originals:
            continue
        original = getattr(torch.Tensor, name)

        def method(self, other, *args, _original=original, **kwargs):
            self, other = _contiguous_bool_operands(self, other)
            return _original(self, other, *args, **kwargs)

        _originals[("Tensor", name)] = original
        setattr(torch.Tensor, name, method)
    for name in _FUNCTIONS:
        if ("torch", name) in _originals:
            continue
        original = getattr(torch, name)

        def function(input, other, *args, _original=original, **kwargs):
            input, other = _contiguous_bool_operands(input, other)
            return _original(input, other, *args, **kwargs)

        _originals[("torch", name)] = original
        setattr(torch, name, function)


def uninstall_bool_bitwise_guard() -> None:
    """Restore the original operators (tests)."""
    for (owner, name), original in list(_originals.items()):
        setattr(torch.Tensor if owner == "Tensor" else torch, name, original)
        del _originals[(owner, name)]


__all__ = ["install_bool_bitwise_guard", "uninstall_bool_bitwise_guard"]
