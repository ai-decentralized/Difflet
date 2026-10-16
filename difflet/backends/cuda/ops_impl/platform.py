"""Platform facts for the CUDA backend.

Everything CUDA-specific the rest of the backend may need lives here (plan D11), so
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
    """Process-group backend for follow-up F2."""
    return "nccl"


def compute_capability() -> tuple[int, int] | None:
    """``(major, minor)`` of the current device, or None without CUDA."""
    if not torch.cuda.is_available():
        return None
    return tuple(torch.cuda.get_device_capability())


def sdpa_kernel_report(q, k, v) -> str:
    """Which SDPA backend torch picks for these tensors, as a short string.

    Phase 0 of the plan records this for Qwen-Image's shapes (D7). Probing is by
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
    if q.dim() == 3:  # the backend lifts (B*H, S, D) to 4-D before SDPA; probe the same way
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
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
    "compute_capability",
    "distributed_backend_name",
    "get_platform_target",
    "hardware",
    "sdpa_kernel_report",
]
