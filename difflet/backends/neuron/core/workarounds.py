"""Workarounds for torch-neuronx kernel issues, installed by the backend's prepare_runtime.

LayerNorm in bf16. The device's bf16 LayerNorm is less accurate than the CPU's, whose kernel
accumulates the statistics in fp32: over HunyuanVideo 1.5's 54 blocks the drift compounded to
1.21x the bf16 CPU control against fp32 (blocks 0-3 at 1.00x, block 26 at 1.04x, block 53 at
1.16x). Computing it in fp32 for bf16 inputs (upcast input, weight and bias; normalize; cast back)
brings the model to 0.92x. ``F.layer_norm`` is wrapped (``nn.LayerNorm`` calls it); other dtypes
and devices run unchanged.

GroupNorm in bf16. The same failure, much larger, since a group's statistics span channels times
the spatial size: ``F.group_norm`` in bf16 on the device against fp32, 32 groups, relative L1
1.8e-2 at 262,144 elements per group, 0.32 at 1,048,576, 0.71 at 2,097,152 and 1.35 at 4,194,304
(the CPU's bf16 stays at 1.4e-3 throughout). Flux's VAE decoder in bf16 came out at 0.287 against
fp32 (68x the bf16 CPU control). ``F.group_norm`` is wrapped (``nn.GroupNorm`` calls it) exactly as
``F.layer_norm`` is.

HunyuanVideo's VAE decoder hit the same failure: on one tile, each GroupNorm run alone on the fp32
input cast to bf16 was off by 6e-3 to 2e-2 rel-L1 at 81,920 elements per group, 3e-2 to 4e-2 at
327,680, 0.2 to 0.38 at 1.2M to 2.4M, and 1.3 to 2.1 at 4.5M to 8.9M (the CPU's bf16: 2e-3 to
4e-3 at every size); the decoder's other ops (Conv3d, SiLU, the attention's Linear layers, the
upsampling) matched the CPU's bf16 error. The whole bf16 decode came out at 20.8 dB PSNR against
the fp32 decode (55.9 dB in bf16 on the host) before the wrap. No DiT on this backend uses
GroupNorm.

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


def install_layer_norm_fp32() -> None:
    """Wrap ``torch.nn.functional.layer_norm`` (idempotent)."""
    import torch.nn.functional as F

    if ("F", "layer_norm") in _originals:
        return
    original = F.layer_norm

    def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        if _on_neuron(input) and input.dtype == torch.bfloat16:
            return original(input.float(), normalized_shape,
                            None if weight is None else weight.float(),
                            None if bias is None else bias.float(), eps).to(input.dtype)
        return original(input, normalized_shape, weight, bias, eps)

    _originals[("F", "layer_norm")] = original
    F.layer_norm = layer_norm


def install_group_norm_fp32() -> None:
    """Wrap ``torch.nn.functional.group_norm`` (idempotent)."""
    import torch.nn.functional as F

    if ("F", "group_norm") in _originals:
        return
    original = F.group_norm

    def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
        if _on_neuron(input) and input.dtype == torch.bfloat16:
            return original(input.float(), num_groups,
                            None if weight is None else weight.float(),
                            None if bias is None else bias.float(), eps).to(input.dtype)
        return original(input, num_groups, weight, bias, eps)

    _originals[("F", "group_norm")] = original
    F.group_norm = group_norm


def install_neuron_workarounds() -> None:
    """Every workaround (idempotent): what prepare_runtime installs."""
    install_bool_bitwise_guard()
    install_layer_norm_fp32()
    install_group_norm_fp32()


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
    import torch.nn.functional as F

    owners = {"Tensor": torch.Tensor, "torch": torch, "F": F}
    for (owner, name), original in list(_originals.items()):
        setattr(owners[owner], name, original)
        del _originals[(owner, name)]


uninstall_neuron_workarounds = uninstall_bool_bitwise_guard

__all__ = ["install_bool_bitwise_guard", "install_group_norm_fp32", "install_layer_norm_fp32",
           "install_neuron_workarounds", "uninstall_bool_bitwise_guard",
           "uninstall_neuron_workarounds"]
