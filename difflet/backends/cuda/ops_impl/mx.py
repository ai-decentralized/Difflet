"""MX (microscaling) ops on the CUDA backend: deliberately unimplemented (plan D9).

MX is a Trainium hardware format (``nkilib``-backed there). Release 1 of the CUDA
backend runs bf16 only; silently dequantizing or falling back would make a model
that *asks* for MX run at a different precision than requested. FP8/FP4 on
Blackwell is follow-up F4, decided by measurement. These raise instead.
"""

_UNSUPPORTED = "MX is not implemented on the CUDA backend (plan D9; FP8/FP4 is follow-up F4)"


def quantize_mx(*args, **kwargs):
    raise NotImplementedError(f"quantize_mx: {_UNSUPPORTED}")


def dequantize_mx(*args, **kwargs):
    raise NotImplementedError(f"dequantize_mx: {_UNSUPPORTED}")


def linear_mx(*args, **kwargs):
    raise NotImplementedError(f"linear_mx: {_UNSUPPORTED}")


def matmul_mx(*args, **kwargs):
    raise NotImplementedError(f"matmul_mx: {_UNSUPPORTED}")


__all__ = ["dequantize_mx", "linear_mx", "matmul_mx", "quantize_mx"]
