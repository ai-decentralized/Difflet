"""The GPU platform family: eager torch on ``cuda``-type devices, one process per GPU.

CUDA and ROCm share it: torch on ROCm exposes its devices under the ``cuda`` device type, so
the generic GPU flow (``difflet.backends.gpu``) runs on either; a subclass supplies detection,
runtime checks and its toolchain versions.
"""

from __future__ import annotations

import atexit
import os
from typing import Mapping

from difflet.platforms.base import Platform, PlatformCapabilities

#: What a GPU platform implements today: CFG branches on two GPUs under ``torchrun``, and
#: data-parallel replicas (one worker per GPU). TP, SP and CP need collectives and ring /
#: Ulysses attention in the backend's ``ops_impl`` first.
GPU_PARALLEL_MODES = frozenset({"cfg", "dp"})


class GpuPlatform(Platform):
    capabilities = PlatformCapabilities(
        requires_aot=False,
        single_process_multi_core=False,
        supports_torchrun_mpmd=True,
        flow="gpu",
        parallel_modes=GPU_PARALLEL_MODES,
        supports_torch_compile=True,
        device_unit="GPUs",
    )
    #: The environment variable that restricts a process to some of the host's GPUs.
    visible_devices_var = "CUDA_VISIBLE_DEVICES"
    #: ``torch.distributed`` backend for multi-process runs.
    distributed_backend = "nccl"

    def _require_device(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"DIFFLET_BACKEND={self.name} selected but torch.cuda.is_available() is False: "
                f"no {self.display_name} device, driver, or matching torch build on this host."
            )

    def prepare_runtime(self, parallel) -> None:
        import torch

        self._require_device()
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        # Bind the device first so the communicator is created on this rank's GPU.
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if world_size > 1:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    f"WORLD_SIZE={world_size} but this torch build has no torch.distributed"
                )
            if not torch.distributed.is_initialized():
                # Env rendezvous: torchrun sets MASTER_ADDR/PORT, RANK and WORLD_SIZE.
                torch.distributed.init_process_group(self.distributed_backend)
                # The group this call created is torn down at exit, so a rank does not
                # leave the communicator to the interpreter's finalisers (warnings, a hung exit).
                atexit.register(_destroy_process_group_at_exit)

    def device_count(self) -> int | None:
        import torch

        return int(torch.cuda.device_count()) if torch.cuda.is_available() else None

    def device_memory_bytes(self) -> int | None:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.get_device_properties(0).total_memory)

    def default_tp_degree(self) -> int:
        # The registry's default degrees describe Trainium topologies; a GPU run is one card
        # per tensor-parallel group unless the user asks for more.
        return 1

    def visible_devices(
        self, inherited: str | None, device_indices: list[int] | tuple[int, ...]
    ) -> str:
        """The ``visible_devices_var`` value for a worker owning ``device_indices``, counted
        from this process's visible devices (so a parent already restricted to some GPUs hands
        out its own, not physical GPUs 0 and 1)."""
        parent = [d.strip() for d in (inherited or "").split(",") if d.strip()]
        if not parent:
            return ",".join(str(i) for i in device_indices)
        if max(device_indices) >= len(parent):
            raise ValueError(
                f"device ids {list(device_indices)} are outside the parent's "
                f"{self.visible_devices_var}={inherited!r}"
            )
        return ",".join(parent[i] for i in device_indices)

    def worker_env(self, base_env: Mapping[str, str], device_indices: list[int]) -> dict[str, str]:
        """A worker's environment: its GPUs in ``visible_devices_var``, no ``NEURON_RT_*``."""
        env = {k: v for k, v in base_env.items() if not k.startswith("NEURON_RT_")}
        env[self.visible_devices_var] = self.visible_devices(
            base_env.get(self.visible_devices_var), device_indices
        )
        return env


def _destroy_process_group_at_exit() -> None:
    """Destroy the default process group if it is still initialised (no-op otherwise)."""
    import torch

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
