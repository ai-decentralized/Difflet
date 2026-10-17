"""Per-core HBM budget of the neuron backend (TorchNeuron, one process per NeuronCore). Torch-free.

Why it exists: a rank whose weights exceed its core's HBM does not fail when it starts loading.
The device allocator retries every shard copy until it gives up, so the error comes long after
the launch. The Qwen-Image tp1 60-layer load (38.05 GiB per rank against 24 GiB) raised
``NRT_RESOURCE`` only after about 16 minutes of retries
(``docs/verification/qwen-image-neuron-evidence.md``, V2k LIMIT). If the estimate is compared
with the core's HBM before any weight is materialised, the same mistake fails in seconds and the
message says which tp would fit.

The budget is ``FIT_FRACTION`` of one core's HBM, for weights only. There is no activation
estimate: the check rejects a load that clearly cannot fit and lets a marginal one go to the
device. ``TorchNeuronApplicationBase.load`` compares the exact bytes of this rank's meta-built
module with it. The TP layers size their shards in ``__init__``, so the meta shapes are already
this rank's slices. Every neuron model goes through it (Wan, Qwen-Image, their VAEs and umT5).

One core's HBM comes from ``DIFFLET_NEURON_HBM_GIB`` if set, else from ``neuron-ls``
(``memory_size`` over the logical cores of a device), else from trn2's numbers: 96 GiB per
device over 8 physical cores, times the LNC (``NEURON_LOGICAL_NC_CONFIG``, default 2), which is
24 GiB. The 24 GiB matches the V2k probe, whose allocator stopped short of 25,769,803,776 B.

Torch-free, like ``difflet/cli/dp/hbm_check.py``'s header scan: the CLI driver imports this
before torchrun starts any rank.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

from difflet import envs

GIB = 1024**3
#: Weights above this fraction of one core's HBM are rejected. The rest is left for the
#: activations, the compiled NEFFs and the runtime, which this check does not estimate.
FIT_FRACTION = 0.95
HBM_ENV = "DIFFLET_NEURON_HBM_GIB"
#: trn2: one device has 96 GiB of HBM shared by 8 physical NeuronCores; LNC=2 pairs them into
#: 4 logical cores of 24 GiB each.
_TRN2_DEVICE_HBM = 96 * GIB
_TRN2_PHYSICAL_CORES = 8
_TRN2_DEFAULT_LNC = 2


class HbmBudgetError(ValueError):
    """The weights of one rank cannot fit in one NeuronCore's HBM at this topology.

    A ``ValueError`` because the parallel configuration is wrong (tp too small, or a component
    in the wrong place), as the Wan application's text encoder placement check always reported.
    """


def per_core_hbm_bytes() -> tuple[int, str]:
    """One logical NeuronCore's HBM in bytes, and where the number came from."""
    raw = (envs.DIFFLET_NEURON_HBM_GIB or "").strip()
    if raw:
        try:
            gib = float(raw)
        except ValueError:
            gib = -1.0
        if not gib > 0:
            raise ValueError(f"{HBM_ENV}={raw!r} is not a positive number of GiB")
        return int(gib * GIB), f"{HBM_ENV}={raw}"
    from difflet.planner.hardware import detected_hbm_per_core

    detected = detected_hbm_per_core()
    if detected is not None:
        return detected
    lnc = int(envs.NEURON_LOGICAL_NC_CONFIG)
    lnc = lnc if lnc > 0 else _TRN2_DEFAULT_LNC
    per_core = _TRN2_DEVICE_HBM * lnc // _TRN2_PHYSICAL_CORES
    return per_core, f"trn2 default: 96 GiB per device / 8 physical cores x LNC={lnc}"


def budget_enforced(device_type: str) -> bool:
    """Whether a load on ``device_type`` is checked. The neuron device is always checked. Any
    other device (the CPU tests) is checked only when ``DIFFLET_NEURON_HBM_GIB`` gives it a
    budget, because it has no HBM of its own."""
    return device_type == "neuron" or bool((envs.DIFFLET_NEURON_HBM_GIB or "").strip())


def safetensors_numel(directory: str | Path) -> int:
    """Total element count of the ``*.safetensors`` files directly in ``directory``. Only the
    headers are read. Returns 0 when the directory has none (a component the checkpoint lacks)."""
    total = 0
    for path in sorted(Path(directory).glob("*.safetensors")):
        with path.open("rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_len))
        for name, info in header.items():
            if name != "__metadata__":
                total += math.prod(info["shape"]) if info["shape"] else 1
    return total


def suggested_tp(estimate_bytes: int, tp_degree: int, limit_bytes: int) -> int:
    """The smallest power-of-two multiple of ``tp_degree`` at which ``estimate_bytes`` (measured at
    ``tp_degree``) would fit under ``limit_bytes`` if all of it were sharded. This is a lower
    bound, since replicated tensors do not shrink."""
    tp = max(int(tp_degree), 1)
    while estimate_bytes * int(tp_degree) / tp > limit_bytes and tp < 1024:
        tp *= 2
    return tp


def check_fits(estimate_bytes: int, *, what: str, tp_degree: int, advice: str = "") -> None:
    """Raise ``HbmBudgetError`` when ``estimate_bytes`` of weights per rank exceed
    ``FIT_FRACTION`` of one core's HBM. ``what`` names the load in the message. ``advice``
    replaces the default "raise tp" guidance; a ``{tp}`` in it becomes ``suggested_tp``."""
    hbm, origin = per_core_hbm_bytes()
    limit = int(hbm * FIT_FRACTION)
    if estimate_bytes <= limit:
        return
    tp = suggested_tp(estimate_bytes, tp_degree, limit)
    if not advice:
        advice = (
            "shard it over more cores: tp_degree={tp} or more (--tp-degree), or fewer CP/CFG/DP "
            "lanes to free the cores for TP"
        )
    advice = advice.replace("{tp}", str(tp))
    raise HbmBudgetError(
        f"{what}: {estimate_bytes / GIB:.2f} GiB of weights per rank at tp={tp_degree} do not "
        f"fit one NeuronCore's {hbm / GIB:.1f} GiB of HBM ({origin}; weights may use at most "
        f"{FIT_FRACTION:.0%}, {limit / GIB:.2f} GiB). Nothing was loaded; {advice}"
    )


__all__ = [
    "FIT_FRACTION",
    "GIB",
    "HBM_ENV",
    "HbmBudgetError",
    "budget_enforced",
    "check_fits",
    "per_core_hbm_bytes",
    "safetensors_numel",
    "suggested_tp",
]
