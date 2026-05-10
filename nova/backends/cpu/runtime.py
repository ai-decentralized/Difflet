"""CPU backend runtime used by numerical-alignment scripts."""

from nova.backends.base import BackendCapabilities, BackendRuntime


class CpuBackend(BackendRuntime):
    name = "cpu"
    capabilities = BackendCapabilities(
        requires_aot=False,
        single_process_multi_core=False,
        supports_torchrun_mpmd=False,
    )

    def prepare_runtime(self, parallel) -> None:
        return None


def create_backend() -> CpuBackend:
    return CpuBackend()
