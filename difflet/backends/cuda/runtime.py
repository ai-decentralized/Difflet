"""CUDA backend placeholder runtime."""

from difflet.backends.base import BackendCapabilities, BackendRuntime


class CudaBackend(BackendRuntime):
    name = "cuda"
    capabilities = BackendCapabilities(
        requires_aot=False,
        single_process_multi_core=False,
        supports_torchrun_mpmd=True,
    )

    def prepare_runtime(self, parallel) -> None:
        raise NotImplementedError(
            "DIFFLET_BACKEND=cuda is reserved but not implemented yet. "
            "Use DIFFLET_BACKEND=trainium for the current Flux path."
        )


def create_backend() -> CudaBackend:
    return CudaBackend()
