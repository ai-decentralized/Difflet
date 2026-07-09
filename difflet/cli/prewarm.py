"""Overlap the one-time Neuron runtime init with host-side model loading.

The first device allocation in a process lazily triggers Neuron runtime init /
NeuronCore bring-up (``nrt_init`` -> ``tdrv_init`` -> core reset + ready-wait),
which measures ~6.7s and otherwise sits serially in front of the first weight
tensor's allocation. Firing a throwaway device allocation on a background thread
at generate/stage start lets that bring-up run concurrently with the host-side
weight load instead of blocking it.

Best-effort only: every failure is swallowed so correctness never depends on it.
The main thread's own first device op remains the source of truth — if prewarm
races ahead, the main thread finds the runtime already up; if it fails or lags,
the main thread pays the init as before. Set ``DIFFLET_DISABLE_PREWARM`` to a
non-empty value to opt out.
"""
from __future__ import annotations

import os
import threading


def prewarm_neuron_runtime(num_devices: int) -> threading.Thread | None:
    """Kick off Neuron runtime init in the background so it overlaps host load.

    Args:
        num_devices: number of local NeuronCores the process will use (e.g.
            ``DiffletParallelConfig.world_size``, or ``NEURON_RT_NUM_CORES`` in a
            stage subprocess). Touching one core triggers the process-wide
            bring-up; the rest are touched too so every core's context is warm.

    Returns the started daemon thread, or ``None`` if prewarm is disabled or
    ``num_devices`` is non-positive.
    """
    if os.environ.get("DIFFLET_DISABLE_PREWARM"):
        return None
    if num_devices < 1:
        return None

    def _touch() -> None:
        try:
            import torch
            import torch_neuronx  # noqa: F401  registers the privateuseone backend
        except Exception:
            return
        for i in range(num_devices):
            try:
                torch.empty(1, device=f"privateuseone:{i}")
            except Exception:
                # A missing/busy core must never surface here — prewarm is a
                # latency optimization, not a correctness dependency.
                pass

    thread = threading.Thread(target=_touch, name="neuron-prewarm", daemon=True)
    thread.start()
    return thread
