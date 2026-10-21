"""The CPU platform: the numerical reference the other backends are checked against."""

from __future__ import annotations

from difflet.platforms.base import Platform, PlatformCapabilities


class CpuPlatform(Platform):
    name = "cpu"
    display_name = "CPU"
    capabilities = PlatformCapabilities(
        requires_aot=False,
        single_process_multi_core=False,
        supports_torchrun_mpmd=False,
        flow="cpu",
    )

    def prepare_runtime(self, parallel) -> None:
        return None


def create_platform() -> CpuPlatform:
    return CpuPlatform()
