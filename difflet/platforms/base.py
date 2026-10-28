"""The platform contract: one ``Platform`` per backend.

Code outside ``difflet/platforms`` and ``difflet/backends`` asks the current platform what it
can do (``capabilities``) or asks it to do something (the methods below) instead of comparing
backend names, so a new backend is a new platform module rather than a hunt through the CLI,
serving and pipeline code.

Importing this module, or any platform module, must not import torch: the serving parent
process reads capabilities without loading it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Every parallel mode Difflet knows. A platform lists the ones its collectives and attention
#: implement; ``unsupported_parallel_modes`` checks a config against that list.
PARALLEL_MODES = frozenset({"tp", "sp", "cp", "cfg", "dp"})


@dataclass(frozen=True)
class PlatformCapabilities:
    requires_aot: bool
    single_process_multi_core: bool
    supports_torchrun_mpmd: bool
    #: Which application, CLI and serving path the platform runs: ``"neuron"`` (staged NEFFs),
    #: ``"tpu"``, ``"gpu"`` (the generic eager flow in ``difflet.backends.gpu``) or ``"cpu"``.
    flow: str = "neuron"
    #: The parallel modes this platform implements.
    parallel_modes: frozenset[str] = PARALLEL_MODES
    #: ``torch.compile`` of the DiT blocks is an option on this platform.
    supports_torch_compile: bool = False
    #: What one device is called in messages.
    device_unit: str = "NeuronCores"


class Platform:
    """Runtime policy for one backend. Subclasses set ``name`` and ``capabilities``."""

    name: str
    #: The name users see in messages ("CUDA", "Trainium").
    display_name: str = ""
    capabilities: PlatformCapabilities

    @classmethod
    def detect(cls) -> bool:
        """True when this host can run the platform. Must stay cheap and must not import torch
        unless the platform needs torch to answer."""
        return False

    def prepare_runtime(self, parallel) -> None:
        """Initialize runtime state before application construction."""

    def resolve_load_rank_range(
        self,
        *,
        start_rank_id: int | None,
        local_ranks_size: int | None,
    ) -> tuple[int | None, int | None]:
        return start_rank_id, local_ranks_size

    def default_tp_degree(self) -> int | None:
        """The tensor-parallel degree a run uses when the user sets none, or ``None`` for the
        model registry's default."""
        return None

    def toolchain_versions(self) -> dict[str, str | None]:
        """Versions hashed into the compile cache key beyond the shared toolchain."""
        return {}

    def layer_markers(self) -> tuple[type, type] | None:
        """The (start, end) layer-boundary marker modules, or ``None`` for identity markers."""
        return None

    def unsupported_parallel_modes(self, parallel: Any) -> list[str]:
        """The modes ``parallel`` asks for that this platform does not implement, in a fixed
        order. ``parallel`` is any config with the usual degree attributes; missing ones count
        as off."""
        requested = []
        if int(getattr(parallel, "tp_degree", 1) or 1) > 1:
            requested.append("tp")
        if getattr(parallel, "sp_enabled", False):
            requested.append("sp")
        if int(getattr(parallel, "cp_degree", 1) or 1) > 1:
            requested.append("cp")
        if getattr(parallel, "cfg_parallel_enabled", False):
            requested.append("cfg")
        if int(getattr(parallel, "dp_degree", 1) or 1) > 1:
            requested.append("dp")
        return [mode for mode in requested if mode not in self.capabilities.parallel_modes]


#: How each parallel mode is named in messages, by the flag that requests it.
PARALLEL_MODE_FLAGS = {
    "tp": "--tp-degree > 1 (tensor parallelism)",
    "sp": "--sp (sequence parallelism)",
    "cp": "--cp-degree > 1 (context parallelism)",
    "cfg": "--cfg-parallel (one CFG branch per device)",
    "dp": "--dp-degree > 1 (data-parallel replicas)",
}


__all__ = [
    "PARALLEL_MODES",
    "PARALLEL_MODE_FLAGS",
    "Platform",
    "PlatformCapabilities",
]
