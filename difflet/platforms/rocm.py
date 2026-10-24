"""The ROCm platform (reserved): the generic GPU flow on AMD GPUs once ``ops_impl`` exists."""

from __future__ import annotations

from difflet.platforms.gpu import GpuPlatform


class RocmPlatform(GpuPlatform):
    name = "rocm"
    display_name = "ROCm"
    visible_devices_var = "HIP_VISIBLE_DEVICES"

    @classmethod
    def detect(cls) -> bool:
        import torch

        return bool(getattr(torch.version, "hip", None))

    def prepare_runtime(self, parallel) -> None:
        raise NotImplementedError(
            "DIFFLET_BACKEND=rocm is reserved but not implemented yet. "
            "Use DIFFLET_BACKEND=trainium for the current Flux path."
        )


def create_platform() -> RocmPlatform:
    return RocmPlatform()
