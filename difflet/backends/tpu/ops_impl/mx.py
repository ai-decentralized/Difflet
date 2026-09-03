"""MX (microscaling) ops — not available on TPU.

MX is a Trainium hardware format (``nkilib``-backed on that backend). There is
no TPU equivalent, and silently falling back to an unquantized path would make
a model that *asks* for MX quietly run at a different precision and memory
footprint than requested. These raise instead.

If MX-quantized weights ever need to run on TPU, the honest options are to
dequantize at load time or to implement an emulated path — both deliberate
decisions, not defaults.

Phase 2 of docs/plans/2026-08-16-tpu-backend-support.md.
"""

from __future__ import annotations

_UNSUPPORTED = (
    "MX (microscaling) is a Trainium hardware format with no TPU equivalent; "
    "run this model without MX quantization on the tpu backend"
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
