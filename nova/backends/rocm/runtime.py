"""ROCm backend placeholder runtime."""

from nova.backends.base import BackendCapabilities, BackendRuntime


class RocmBackend(BackendRuntime):
    name = "rocm"
    capabilities = BackendCapabilities(
        requires_aot=False,
        single_process_multi_core=False,
        supports_torchrun_mpmd=True,
    )

    def prepare_runtime(self, parallel) -> None:
        raise NotImplementedError(
            "NOVA_BACKEND=rocm is reserved but not implemented yet. "
            "Use NOVA_BACKEND=trainium for the current Flux path."
        )


def create_backend() -> RocmBackend:
    return RocmBackend()
