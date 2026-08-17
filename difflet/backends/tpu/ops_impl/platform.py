"""TPU platform helpers.

Mirrors the CPU backend's shape (a ``hardware`` enum plus
``get_platform_target()``) rather than Trainium's, which re-exports
``torch_neuronx``/NxD functions that do not exist here.

Phase 2d of docs/plans/2026-08-16-tpu-backend-support.md.
"""

from __future__ import annotations

from enum import Enum


class hardware(Enum):
    CPU = "cpu"
    TRN1 = "trn1"
    TRN2 = "trn2"
    TPU = "tpu"


def get_platform_target():
    return hardware.TPU


#: XLA's default matmul precision on TPU is NOT fp32. The MXU multiplies
#: fp32 inputs at reduced precision unless told otherwise, which measured as
#: ~1e-2 absolute error per matmul on v5e — against ~2e-6 at "highest".
#: Compounded through a DiT that makes the Phase 5 target (cosine >= 0.9995
#: vs. the diffusers reference) unreachable, and the failure is silent: every
#: shape is right and nothing errors. Any TPU entry point must set this
#: before running real work.
DEFAULT_MATMUL_PRECISION = "highest"


def configure_matmul_precision(precision: str = DEFAULT_MATMUL_PRECISION) -> str:
    """Set XLA's matmul precision, returning the previous value.

    Deliberately explicit rather than done at import time: it is process-wide
    global state, and silently mutating it on import would be worse than the
    bug it prevents.
    """
    import torch_xla.backends as backends

    previous = backends.get_mat_mul_precision()
    backends.set_mat_mul_precision(precision)
    return previous


def tpu_device_kind() -> str:
    """Concrete chip generation (e.g. ``TPU v5 lite``), or ``unknown``.

    Used for capacity decisions — v5e has 16 GB HBM per chip while v5p has
    ~95 GB, which is the difference between a model fitting at tp=1 and not.
    """
    try:
        import torch_xla.core.xla_model as xm

        return str(xm.xla_device_hw(xm.xla_device()) or "unknown")
    except Exception:  # noqa: BLE001 — a probe must never break import
        return "unknown"


__all__ = [
    "configure_matmul_precision",
    "DEFAULT_MATMUL_PRECISION",
    "get_platform_target",
    "hardware",
    "tpu_device_kind",
]
