"""CUDA backend runtime.

Release 1 of docs/plans/2026-10-16-cuda-backend-support.md: one process, one
device, no ``torch.distributed``. Under ``torchrun`` (``WORLD_SIZE > 1``) each process binds its
``LOCAL_RANK`` GPU and joins the NCCL
process group from the launcher's env rendezvous; the parallel mesh then builds
its axis groups from it.
"""

from __future__ import annotations

import atexit
import os

import torch

from difflet.backends.base import BackendCapabilities, BackendRuntime
from difflet.backends.cuda.ops_impl.platform import distributed_backend_name


class CudaBackend(BackendRuntime):
    name = "cuda"
    capabilities = BackendCapabilities(
        requires_aot=False,
        single_process_multi_core=False,
        supports_torchrun_mpmd=True,
    )

    def prepare_runtime(self, parallel) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DIFFLET_BACKEND=cuda selected but torch.cuda.is_available() is False: "
                "no CUDA device, driver, or CUDA-enabled torch build on this host."
            )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        # Bind the device first so the NCCL communicator is created on this rank's GPU.
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if world_size > 1:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    f"WORLD_SIZE={world_size} but this torch build has no torch.distributed"
                )
            if not torch.distributed.is_initialized():
                # Env rendezvous: torchrun sets MASTER_ADDR/PORT, RANK and WORLD_SIZE.
                torch.distributed.init_process_group(distributed_backend_name())
                # The group this call created is torn down at exit, so a rank does not
                # leave NCCL to the interpreter's finalisers (warnings, a hung exit).
                atexit.register(_destroy_process_group_at_exit)


def _destroy_process_group_at_exit() -> None:
    """Destroy the default process group if it is still initialised (no-op otherwise)."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def create_backend() -> CudaBackend:
    return CudaBackend()
