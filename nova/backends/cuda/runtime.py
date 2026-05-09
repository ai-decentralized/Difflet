"""CUDA backend placeholder runtime."""

from nova.backends.base import BackendCapabilities, BackendRuntime


class CudaBackend(BackendRuntime):
    name = "cuda"
    capabilities = BackendCapabilities(
        requires_aot=False,
        single_process_multi_core=False,
        supports_torchrun_mpmd=True,
    )

    def prepare_runtime(self, parallel) -> None:
        raise NotImplementedError(
            "NOVA_BACKEND=cuda is reserved but not implemented yet. "
            "Use NOVA_BACKEND=trainium for the current Flux path."
        )


def create_backend() -> CudaBackend:
    return CudaBackend()
