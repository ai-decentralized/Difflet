"""TPU backend runtime scaffolding.

Phase 1 of docs/plans/2026-08-16-tpu-backend-support.md: registers the
backend so ``DIFFLET_BACKEND=tpu`` resolves, model entries can opt in via
their ``backends`` tuple, and the compile cache can key on the backend.
``prepare_runtime`` intentionally raises until the Phase 0 spike settles the
compile/load model (AOT StableHLO export vs. lazy + persistent cache vs.
torchax) and Phase 2 lands ``ops_impl``.
"""

from __future__ import annotations

from difflet.backends.base import BackendCapabilities, BackendRuntime

_PLAN = "docs/plans/2026-08-16-tpu-backend-support.md"


class TpuBackend(BackendRuntime):
    name = "tpu"
    # Settled by the Phase 0 spike on a v5litepod-4 (see the plan's status log):
    # - requires_aot: Direction A (torch.export -> StableHLO) won. Direction B
    #   (lazy + persistent cache) cannot deserialize executables on torch_xla,
    #   so there is no lazy-compile fallback to fall back to.
    # - supports_torchrun_mpmd: True, and genuinely the opposite of Trainium's
    #   rule — the spike ran 4 processes via torch_xla.launch, one per chip,
    #   exporting and executing a real 4-replica collective.
    capabilities = BackendCapabilities(
        requires_aot=True,
        single_process_multi_core=True,
        supports_torchrun_mpmd=True,
    )

    def prepare_runtime(self, parallel) -> None:
        # Phase 3 lands the compile/load lifecycle here. When it does, it MUST
        # call ops_impl.platform.configure_matmul_precision() before any real
        # work: XLA's default TPU matmul precision is not fp32 (~1e-2 error
        # per matmul, measured), which silently destroys numerical parity.
        raise NotImplementedError(
            "DIFFLET_BACKEND=tpu has ops_impl (Phase 2) but no compile/load "
            f"lifecycle yet — that is Phase 3 (see {_PLAN}). "
            "Use DIFFLET_BACKEND=trainium or DIFFLET_BACKEND=cpu."
        )


def create_backend() -> TpuBackend:
    return TpuBackend()
