"""The CUDA platform: the generic GPU flow on NVIDIA GPUs."""

from __future__ import annotations

from importlib import metadata

from difflet.platforms.gpu import GpuPlatform


class CudaPlatform(GpuPlatform):
    name = "cuda"
    display_name = "CUDA"

    @classmethod
    def detect(cls) -> bool:
        import torch

        return bool(torch.cuda.is_available())

    def toolchain_versions(self) -> dict[str, str | None]:
        # No AOT artifact today, but the manifest and any torch.compile cache still key on the
        # kernels torch was built against. torch.version.cuda is the CUDA runtime the wheel was
        # built for (None on a CPU-only build); cudnn.version() needs a CUDA build too.
        import torch

        versions: dict[str, str | None] = {}
        try:
            versions["triton"] = metadata.version("triton")
        except metadata.PackageNotFoundError:
            versions["triton"] = None
        versions["cuda"] = torch.version.cuda
        cudnn = torch.backends.cudnn.version() if torch.version.cuda else None
        versions["cudnn"] = None if cudnn is None else str(cudnn)
        return versions


def create_platform() -> CudaPlatform:
    return CudaPlatform()
