"""MX (microscaling) ops — not available on the neuron backend.

MX is a Trainium hardware format: FP8 data under E8M0 block scales. The
trainium backend runs it through ``nkilib``-backed NKI kernels launched as
``TorchXlaKernel``, which needs torch_xla, and TorchNeuron has no FP8 support
yet. Silently falling back to an unquantized path (or to the CPU reference)
would make a model that *asks* for MX quietly run at a different precision and
memory footprint than requested. These raise instead.

If MX-quantized weights ever need to run on TorchNeuron, the honest options are
to dequantize at load time or to call NKI MX kernels directly, the way
``attention.py`` calls the flash kernel — both deliberate decisions, not
defaults.

Phase 1 of the neuron backend (component C6).
"""

from __future__ import annotations

_UNSUPPORTED = (
    "MX (microscaling) quantization is not supported on the neuron backend (TorchNeuron); "
    "run without MX quantization or use DIFFLET_BACKEND=trainium"
)


def quantize_mx(*args, **kwargs):
    raise NotImplementedError(f"quantize_mx: {_UNSUPPORTED}")


def dequantize_mx(*args, **kwargs):
    raise NotImplementedError(f"dequantize_mx: {_UNSUPPORTED}")


def linear_mx(*args, **kwargs):
    raise NotImplementedError(f"linear_mx: {_UNSUPPORTED}")


def matmul_mx(*args, **kwargs):
    raise NotImplementedError(f"matmul_mx: {_UNSUPPORTED}")


__all__ = ["dequantize_mx", "linear_mx", "matmul_mx", "quantize_mx"]
