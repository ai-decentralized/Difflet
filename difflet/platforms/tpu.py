"""The TPU platform (scaffolding: ops are implemented, the compile/load lifecycle is not)."""

from __future__ import annotations

import importlib.util

from difflet.platforms.base import Platform, PlatformCapabilities


class TpuPlatform(Platform):
    name = "tpu"
    display_name = "TPU"
    # Settled by a spike on a v5litepod-4:
    # - requires_aot: torch.export -> StableHLO. A lazy + persistent cache cannot
    #   deserialize executables on torch_xla, so there is no lazy-compile fallback.
    # - supports_torchrun_mpmd: True, the opposite of Trainium's rule — the spike
    #   ran 4 processes via torch_xla.launch, one per chip, exporting and executing
    #   a real 4-replica collective.
    capabilities = PlatformCapabilities(
        requires_aot=True,
        single_process_multi_core=True,
        supports_torchrun_mpmd=True,
        flow="tpu",
    )

    @classmethod
    def detect(cls) -> bool:
        # torch_xla plus the TPU runtime library. Neuron venvs also ship torch_xla but never
        # libtpu, and Trainium is detected first, so a Trainium host cannot mis-detect as TPU.
        return (
            importlib.util.find_spec("torch_xla") is not None
            and importlib.util.find_spec("libtpu") is not None
        )

    def prepare_runtime(self, parallel) -> None:
        # The compile/load lifecycle lands here. It MUST call
        # ops_impl.platform.configure_matmul_precision() before any real work:
        # XLA's default TPU matmul precision is not fp32 (~1e-2 error per matmul),
        # which silently destroys numerical parity.
        raise NotImplementedError(
            "DIFFLET_BACKEND=tpu has ops_impl but no compile/load lifecycle yet. "
            "Use DIFFLET_BACKEND=trainium or DIFFLET_BACKEND=cpu."
        )


def create_platform() -> TpuPlatform:
    return TpuPlatform()
