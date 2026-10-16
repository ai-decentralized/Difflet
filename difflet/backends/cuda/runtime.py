"""CUDA backend runtime.

Release 1 of docs/plans/2026-10-16-cuda-backend-support.md: one process, one
device, no ``torch.distributed`` (decision D5). Multi-process execution under
``torchrun`` is follow-up F2 and raises until it lands, so that a ``torchrun``
launch cannot silently run every rank on GPU 0.
"""

from __future__ import annotations

import os

import torch

from difflet.backends.base import BackendCapabilities, BackendRuntime


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
        if world_size > 1:
            raise NotImplementedError(
                f"WORLD_SIZE={world_size}: multi-process CUDA execution is follow-up F2 "
                "of the CUDA backend plan; Release 1 runs one process on one device."
            )
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def create_backend() -> CudaBackend:
    return CudaBackend()
