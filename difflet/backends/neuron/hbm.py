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
this rank's slices. Every neuron model goes through it (Wan, Qwen-Image, LTX-2, their VAEs and
text encoders).

Before anything is built, the estimates from the checkpoint headers answer the same question
for components that share a core: ``Component`` names a checkpoint subdirectory, the dtype it
holds on the device and the tensors it skips, and ``components_fit`` adds the components up over
the TP group (a device text encoder beside the DiT, which the per-component check at load would
only catch once the DiT had loaded). Each model's ``hbm.py`` declares its components. A
quantized DiT (``--quant``) is counted as its directory stores it, fp8 weights and uint8 block
scales rather than the bf16 source (``quantized_dir_fits``, ``Component.stored_in``).

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
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
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


def safetensors_numel(
    directory: str | Path,
    keep: Callable[[str], bool] | None = None,
    *,
    index: str | None = None,
) -> int:
    """Total element count of the ``*.safetensors`` files directly in ``directory``, or of the
    tensors whose name ``keep`` accepts (a component that loads only part of its checkpoint, as
    Qwen-Image's text encoder skips the vision tower). With ``index``, the name of a safetensors
    index in ``directory`` that exists, only the shards its weight map lists are read (LTX-2's
    text encoder directory also lists a diffusers index of another checkpoint). Only the headers
    are read. Returns 0 when the directory has none (a component the checkpoint lacks)."""
    directory = Path(directory)
    index_path = directory / index if index else None
    if index_path is not None and index_path.is_file():
        names = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
        paths = [directory / name for name in names]
    else:
        paths = sorted(directory.glob("*.safetensors"))
    total = 0
    for path in paths:
        with path.open("rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_len))
        for name, info in header.items():
            if name != "__metadata__" and (keep is None or keep(name)):
                total += math.prod(info["shape"]) if info["shape"] else 1
    return total


def safetensors_bytes(directory: str | Path, *, index: str | None = None) -> int:
    """Total data bytes of the ``*.safetensors`` files directly in ``directory`` (or of the
    shards ``index`` lists), from the headers' ``data_offsets``: the bytes as stored, whatever
    their dtypes (a quantized checkpoint mixes fp8 weights, uint8 block scales and bf16
    tensors)."""
    directory = Path(directory)
    index_path = directory / index if index else None
    if index_path is not None and index_path.is_file():
        names = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
        paths = [directory / name for name in names]
    else:
        paths = sorted(directory.glob("*.safetensors"))
    total = 0
    for path in paths:
        with path.open("rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_len))
        for name, info in header.items():
            if name != "__metadata__":
                lo, hi = info["data_offsets"]
                total += int(hi) - int(lo)
    return total


def _against_budget(estimate: int) -> tuple[bool, int, int]:
    """``(fits, estimate, limit)``: ``estimate`` against ``FIT_FRACTION`` of one core's HBM."""
    limit = int(per_core_hbm_bytes()[0] * FIT_FRACTION)
    return estimate <= limit, estimate, limit


def quantized_dir_bytes(dit_dir: str | Path, *, tp_degree: int) -> int:
    """Lower bound of one rank's bytes of a quantized DiT directory
    (``difflet/quant/checkpoint.py``, the ``--quant`` launches): the directory's safetensors as
    stored (fp8 weights, uint8 block scales, the bf16 rest) over tp. The bf16 estimate of the
    source checkpoint would count every fp8 weight at twice its size and refuse a tp the
    quantized DiT fits."""
    return safetensors_bytes(dit_dir) // max(int(tp_degree), 1)


def quantized_dir_fits(dit_dir: str | Path, *, tp_degree: int) -> tuple[bool, int, int]:
    """``(fits, estimate, limit)``: ``quantized_dir_bytes`` against ``FIT_FRACTION`` of one
    core's HBM. Torch-free, like ``components_fit``."""
    return _against_budget(quantized_dir_bytes(dit_dir, tp_degree=tp_degree))


@dataclass(frozen=True)
class Component:
    """One checkpoint subdirectory a device component loads, for the estimates below.

    ``subdir`` is relative to the snapshot root; ``dtype_bytes`` is the element size the
    component holds on the device (not the checkpoint's); tensors whose names start with one of
    ``skipped_prefixes`` are never loaded (a vision tower, an ``lm_head``); ``index`` names the
    safetensors index whose shards are the component's (``safetensors_numel``). A component
    ``stored_in`` a directory (a quantized DiT, ``quantized_dir_bytes``) loads that directory
    and is counted as it stores it, whatever ``dtype_bytes`` says."""

    subdir: str
    dtype_bytes: int
    skipped_prefixes: tuple[str, ...] = ()
    index: str | None = None
    stored_dir: str | None = None

    def numel(self, model_dir: str | Path) -> int:
        """The element count this component loads from ``<model_dir>/<subdir>``, or from
        ``stored_dir``."""
        skipped = self.skipped_prefixes
        keep = (lambda name: not name.startswith(skipped)) if skipped else None
        directory = Path(model_dir) / self.subdir
        if self.stored_dir is not None:
            directory = Path(self.stored_dir)
        return safetensors_numel(directory, keep, index=self.index)

    def nbytes(self, model_dir: str | Path) -> int:
        """The bytes this component holds on the device: its elements in ``dtype_bytes``, or,
        ``stored_in`` a directory, that directory's bytes as stored (``safetensors_bytes``)."""
        if self.stored_dir is not None:
            return safetensors_bytes(self.stored_dir, index=self.index)
        return self.numel(model_dir) * int(self.dtype_bytes)

    def in_dtype(self, dtype_bytes: int) -> Component:
        """The same tensors held in another element size."""
        return replace(self, dtype_bytes=int(dtype_bytes))

    def stored_in(self, directory: str | Path) -> Component:
        """The same component loaded from ``directory`` and counted as it is stored there."""
        return replace(self, stored_dir=str(directory))


def components_bytes(model_dir: str | Path, components: Iterable[Component], *,
                     tp_degree: int) -> int:
    """Lower bound of one rank's weight bytes when ``components`` share every core of a TP group
    of ``tp_degree``: their elements times their element sizes (a component ``stored_in`` a
    directory: its bytes as stored), all counted as sharded over tp (replicated tensors do not
    shrink, so a rejection means the weights clearly cannot fit)."""
    total = sum(c.nbytes(model_dir) for c in components)
    return total // max(int(tp_degree), 1)


def components_fit(model_dir: str | Path, components: Iterable[Component], *,
                   tp_degree: int) -> tuple[bool, int, int]:
    """``(fits, estimate, limit)``: ``components_bytes`` against ``FIT_FRACTION`` of one core's
    HBM. Torch-free, for the CLI drivers' checks before torchrun starts any rank."""
    return _against_budget(components_bytes(model_dir, components, tp_degree=tp_degree))


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
    "Component",
    "FIT_FRACTION",
    "safetensors_bytes",
    "GIB",
    "HBM_ENV",
    "HbmBudgetError",
    "budget_enforced",
    "check_fits",
    "components_bytes",
    "components_fit",
    "per_core_hbm_bytes",
    "quantized_dir_bytes",
    "quantized_dir_fits",
    "safetensors_numel",
    "suggested_tp",
]
