"""Platform facts for the CUDA backend.

Everything CUDA-specific the rest of the backend may need lives here, so
the ops and application code stay generic torch-accelerator code and a ROCm
backend can later register the same modules with its own platform file.
"""

from __future__ import annotations

from enum import Enum


class hardware(Enum):
    CPU = "cpu"
    TRN1 = "trn1"
    TRN2 = "trn2"
    TPU = "tpu"
    CUDA = "cuda"


def get_platform_target():
    return hardware.CUDA


__all__ = [
    "get_platform_target",
    "hardware",
]
