"""Automatic parallelism planning: hardware + model -> parallel config.

The planner enumerates every parallel configuration the runtime would accept for
a given (hardware, model, shape), scores them, and ranks them. Every entry point
here is read-only: nothing in this package compiles, loads weights, or touches a
NeuronCore, so ``difflet plan`` is safe to run on a laptop.

Ranking is two-stage (AoiZora port): an analytic placement-oblivious prune, then
topology-aware placement ranking over the Trainium core/chip hierarchy.

Design notes live in ``docs/plans/2026-07-27-parallelism-planner.md`` and
``docs/plans/2026-08-20-planner-aoizora-topology-prototype.md``.
"""

from __future__ import annotations

__all__ = [
    "HardwareProfile",
    "detect_hardware",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name in __all__:
        from difflet.planner import hardware

        return getattr(hardware, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
