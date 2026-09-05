"""Neuron hardware discovery.

Before this module the only "detection" Difflet did was
``difflet/common/neuron_cores.py``'s ``DEFAULT_NEURON_CORE_IDS = (0, 1, 2, 3)``
-- a literal that happens to be right on ``trn2.3xlarge`` and wrong everywhere
else. On a 64-core ``trn2.48xlarge`` it silently hands back four cores, so a run
uses 1/16 of the machine and nothing reports it.

``neuron-ls -j`` is the ground truth. It needs no Neuron runtime, no device
claim, and no torch import, so it is cheap enough to call from the CLI:

    [{"instance_type": "trn2.3xlarge", "neuron_device": 0, "nc_count": 4,
      "logical_neuroncore_config": 2, "memory_size": 103079215104,
      "neuroncore_ids": [0, 1, 2, 3], "neuron_processes": []}]

Two quantities are deliberately kept apart:

- **machine cores** -- what the box physically has (``num_devices *
  cores_per_device``).
- **allocated cores** -- what *this process* may spend, which is what a planner
  must budget against. A DP worker launched by ``difflet/cli/dp/router.py`` sees
  ``NEURON_RT_VISIBLE_CORES=2-3`` and has two cores regardless of the box.

Multi-device support is deferred (see the plan's Part 6), but the *shape* of
this profile is already multi-device: ``num_devices`` and ``cores_per_device``
are separate fields and the ``neuron-ls`` parse handles the full device array,
so the second version extends rather than rewrites.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

# Where ``neuron-ls`` lives when the Neuron tools are not on PATH.
_NEURON_LS_FALLBACK_PATH = "/opt/aws/neuron/bin/neuron-ls"

# ``neuron-ls`` is a driver query, not a device claim, but bound the wait so a
# wedged driver cannot hang ``difflet plan``.
_PROBE_TIMEOUT_SECONDS = 30.0

# Per-platform defaults for when ``neuron-ls`` is unavailable (no Neuron tools
# installed, container without /dev/neuron*, developer laptop). Cores are the
# *logical* NeuronCore count per device at that platform's default LNC, which is
# the unit every Difflet degree is expressed in.
_PLATFORM_DEFAULTS: dict[str, tuple[int, int, int]] = {
    # platform: (cores_per_device, hbm_bytes_per_device, lnc)
    "trn1": (2, 32 * 1024**3, 1),
    "trn2": (4, 96 * 1024**3, 2),
    "trn3": (4, 96 * 1024**3, 2),
}
_UNKNOWN_PLATFORM_DEFAULT = _PLATFORM_DEFAULTS["trn2"]


@dataclass(frozen=True)
class HardwareProfile:
    """What the planner is allowed to spend, and on what.

    ``source`` records provenance so ``difflet plan`` can say whether it is
    reasoning about a real box or a guess -- a predicted plan built on fallback
    constants is much weaker evidence than one built on ``neuron-ls`` output,
    and the user deserves to see which they got.
    """

    instance_type: str
    platform_target: str
    num_devices: int
    cores_per_device: int
    hbm_bytes_per_device: int
    lnc: int
    allocated_cores: int
    busy_cores: tuple[int, ...] = ()
    source: str = "fallback"

    def __post_init__(self) -> None:
        for field_name in ("num_devices", "cores_per_device", "allocated_cores"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field_name} must be an int >= 1, got {value!r}")
        if self.hbm_bytes_per_device < 1:
            raise ValueError("hbm_bytes_per_device must be >= 1")

    @property
    def machine_cores(self) -> int:
        """Every logical NeuronCore on the box, ignoring this process's slice."""

        return self.num_devices * self.cores_per_device

    @property
    def is_multi_device(self) -> bool:
        return self.num_devices > 1

    @property
    def hbm_bytes_per_core(self) -> int:
        return self.hbm_bytes_per_device // self.cores_per_device

    def describe(self) -> str:
        gib = self.hbm_bytes_per_device / 1024**3
        devices = f"{self.num_devices} device" + ("s" if self.num_devices != 1 else "")
        detail = (
            f"{self.instance_type} ({devices}, {self.cores_per_device} cores/device, "
            f"{gib:.0f} GiB/device, LNC={self.lnc})"
        )
        if self.allocated_cores != self.machine_cores:
            detail += f" -- {self.allocated_cores} of {self.machine_cores} cores allocated"
        return detail


def detect_hardware(
    *,
    allocated_cores_override: int | None = None,
    use_cache: bool = True,
) -> HardwareProfile:
    """Build a :class:`HardwareProfile` for the current host.

    ``allocated_cores_override`` is the CLI's ``--total-cores``. It wins over
    everything else so a user can plan for a box they are not sitting on.

    Allocation precedence below the override: ``NEURON_RT_VISIBLE_CORES`` (the
    exact core list a DP worker or serving process inherits), then
    ``NEURON_RT_NUM_CORES``, then the machine's full core count.
    """

    devices = _probe_neuron_ls(use_cache=use_cache)
    if devices:
        base = _profile_from_neuron_ls(devices)
    else:
        base = _fallback_profile()

    allocated, allocation_source = _resolve_allocated_cores(
        machine_cores=base.machine_cores,
        override=allocated_cores_override,
    )
    source = base.source if allocation_source is None else f"{base.source}+{allocation_source}"
    return HardwareProfile(
        instance_type=base.instance_type,
        platform_target=base.platform_target,
        num_devices=base.num_devices,
        cores_per_device=base.cores_per_device,
        hbm_bytes_per_device=base.hbm_bytes_per_device,
        lnc=base.lnc,
        allocated_cores=allocated,
        busy_cores=base.busy_cores,
        source=source,
    )


def detected_core_count() -> int | None:
    """The machine's logical NeuronCore count, or ``None`` if undetectable.

    Deliberately narrow: ``difflet/common/neuron_cores.py`` needs exactly this
    and must not pay for the rest of the profile, nor raise when the Neuron
    tools are absent.
    """

    devices = _probe_neuron_ls()
    if not devices:
        return None
    total = sum(_device_core_count(device) for device in devices)
    return total or None


def _resolve_allocated_cores(
    *, machine_cores: int, override: int | None
) -> tuple[int, str | None]:
    if override is not None:
        if override < 1:
            raise ValueError(f"allocated core override must be >= 1, got {override}")
        return override, "override"

    visible = os.environ.get("NEURON_RT_VISIBLE_CORES", "").strip()
    if visible:
        count = len(_parse_core_list(visible))
        if count:
            return count, "NEURON_RT_VISIBLE_CORES"

    num_cores = os.environ.get("NEURON_RT_NUM_CORES", "").strip()
    if num_cores:
        try:
            parsed = int(num_cores)
        except ValueError:
            parsed = 0
        if parsed >= 1:
            return parsed, "NEURON_RT_NUM_CORES"

    return machine_cores, None


def _parse_core_list(raw: str) -> tuple[int, ...]:
    """Parse ``NEURON_RT_VISIBLE_CORES`` syntax: ``0,1`` or ``0-3`` or a mix.

    Tolerant by design -- this is used to *size* an allocation, and
    ``difflet.common.neuron_cores`` remains the strict validator that rejects
    malformed values on the runtime path.
    """

    cores: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start_text, _, end_text = token.partition("-")
            try:
                start, end = int(start_text), int(end_text)
            except ValueError:
                continue
            if end >= start:
                cores.extend(range(start, end + 1))
        else:
            try:
                cores.append(int(token))
            except ValueError:
                continue
    return tuple(dict.fromkeys(cores))


@lru_cache(maxsize=1)
def _probe_neuron_ls_cached() -> tuple[dict, ...]:
    return tuple(_run_neuron_ls())


def _probe_neuron_ls(*, use_cache: bool = True) -> tuple[dict, ...]:
    if use_cache:
        return _probe_neuron_ls_cached()
    return tuple(_run_neuron_ls())


def _run_neuron_ls() -> list[dict]:
    binary = shutil.which("neuron-ls")
    if binary is None and os.path.exists(_NEURON_LS_FALLBACK_PATH):
        binary = _NEURON_LS_FALLBACK_PATH
    if binary is None:
        return []
    try:
        completed = subprocess.run(
            [binary, "-j"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def _profile_from_neuron_ls(devices: tuple[dict, ...]) -> HardwareProfile:
    first = devices[0]
    instance_type = str(first.get("instance_type") or "unknown")
    lnc = _coerce_int(first.get("logical_neuroncore_config"), default=1)

    core_counts = [_device_core_count(device) for device in devices]
    cores_per_device = max(core_counts) if core_counts else 0
    if cores_per_device < 1:
        cores_per_device = _UNKNOWN_PLATFORM_DEFAULT[0]

    memory_sizes = [_coerce_int(device.get("memory_size"), default=0) for device in devices]
    hbm = max(memory_sizes) if memory_sizes else 0
    if hbm < 1:
        hbm = _UNKNOWN_PLATFORM_DEFAULT[1]

    busy: list[int] = []
    for device in devices:
        processes = device.get("neuron_processes")
        if not isinstance(processes, list):
            continue
        for process in processes:
            if isinstance(process, dict):
                busy.extend(_process_core_ids(process))
    return HardwareProfile(
        instance_type=instance_type,
        platform_target=_platform_from_instance_type(instance_type),
        num_devices=len(devices),
        cores_per_device=cores_per_device,
        hbm_bytes_per_device=hbm,
        lnc=lnc,
        allocated_cores=len(devices) * cores_per_device,
        busy_cores=tuple(sorted(dict.fromkeys(busy))),
        source="neuron-ls",
    )


def _device_core_count(device: dict) -> int:
    ids = device.get("neuroncore_ids")
    if isinstance(ids, list) and ids:
        return len(ids)
    return _coerce_int(device.get("nc_count"), default=0)


def _process_core_ids(process: dict) -> list[int]:
    """Best-effort extraction of the cores a foreign process holds.

    ``neuron-ls`` has spelled this field several ways across SDK versions and
    the format is not contractual, so accept whatever is recognizable and treat
    an unparseable entry as "no known cores" rather than failing the probe.
    """

    for key in ("neuroncore_ids", "used_neuroncore_ids", "nc_ids", "cores"):
        value = process.get(key)
        if isinstance(value, list):
            coerced = (_coerce_int(item, default=-1) for item in value)
            return [core_id for core_id in coerced if core_id >= 0]
        if isinstance(value, str):
            return list(_parse_core_list(value))
    return []


def _platform_from_instance_type(instance_type: str) -> str:
    prefix = instance_type.split(".", 1)[0].lower()
    for platform in _PLATFORM_DEFAULTS:
        if prefix.startswith(platform):
            return platform
    if prefix.startswith("inf2"):
        return "inf2"
    return prefix or "unknown"


def _fallback_profile() -> HardwareProfile:
    platform = _platform_target_via_torch() or "trn2"
    cores, hbm, lnc = _PLATFORM_DEFAULTS.get(platform, _UNKNOWN_PLATFORM_DEFAULT)
    return HardwareProfile(
        instance_type=f"{platform}.unknown",
        platform_target=platform,
        num_devices=1,
        cores_per_device=cores,
        hbm_bytes_per_device=hbm,
        lnc=lnc,
        allocated_cores=cores,
        source="fallback",
    )


def _platform_target_via_torch() -> str | None:
    """Ask ``torch_neuronx`` for the platform, if it is importable.

    Only reached when ``neuron-ls`` is missing. Guarded because the planner must
    stay usable without a Neuron install, and because importing torch_neuronx is
    expensive enough that we never want it on the fast path.
    """

    try:  # pragma: no cover - depends on the Neuron toolchain being present
        from difflet.ops.platform import get_platform_target

        target = get_platform_target()
    except Exception:
        return None
    return str(target).lower() if target else None


def _coerce_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default
