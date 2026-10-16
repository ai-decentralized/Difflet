"""Platform facts for the CUDA backend.

Everything CUDA-specific the rest of the backend may need lives here, so
the ops and application code stay generic torch-accelerator code and a ROCm
backend can later register the same modules with its own platform file.
"""

from __future__ import annotations

from enum import Enum

import torch


class hardware(Enum):
    CPU = "cpu"
    TRN1 = "trn1"
    TRN2 = "trn2"
    TPU = "tpu"
    CUDA = "cuda"


def get_platform_target():
    return hardware.CUDA


def distributed_backend_name() -> str:
    """The torch.distributed backend for the CUDA backend's process groups."""
    return "nccl"


def sdpa_kernel_report(q, k, v) -> str:
    """Which SDPA backend torch picks for these tensors, as a short string.

    Recorded for Qwen-Image's shapes when the backend was brought up. Probing is by
    elimination with ``torch.nn.attention.sdpa_kernel``: the first backend that can
    run the call is the one the default dispatcher would choose first.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # very old torch
        return "unknown (torch.nn.attention unavailable)"
    order = [
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("cudnn", getattr(SDPBackend, "CUDNN_ATTENTION", None)),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ]
    for name, backend in order:
        if backend is None:
            continue
        try:
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return f"{name} (device={q.device.type}, dtype={q.dtype}, shape={tuple(q.shape)})"
        except Exception:  # noqa: BLE001 — probing by elimination
            continue
    return "none (no SDPA backend accepted the call)"


__all__ = [
    "distributed_backend_name",
    "get_platform_target",
    "hardware",
    "sdpa_kernel_report",
]
