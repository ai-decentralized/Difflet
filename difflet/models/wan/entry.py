"""Wan registry entry helpers."""

from __future__ import annotations

from typing import Any

from difflet.pipeline.parallel_config import DiffletParallelConfig


def create_wan_application(
    *,
    model_path: str,
    parallel: DiffletParallelConfig,
    dtype: Any,
    shape: dict[str, int | None],
    backend: str = "trainium",
    **kwargs: Any,
) -> Any:
    if backend == "tpu":
        from difflet.models.wan.tpu_application import TpuWanApplication

        return TpuWanApplication(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            shape=shape,
            **kwargs,
        )

    if backend == "cuda":
        # Release 1 of the CUDA backend is one process on one device (plan D5, D6).
        # Refuse here, before two 14B experts are loaded, rather than at the first
        # collective. Two parallel-config types reach this factory (DiffletParallelConfig
        # and serving's ParallelTopology): ask, do not assume.
        tp = int(getattr(parallel, "tp_degree", 1) or 1)
        cp = int(getattr(parallel, "cp_degree", 1) or 1)
        dp = int(getattr(parallel, "dp_degree", 1) or 1)
        cfg_parallel = bool(getattr(parallel, "cfg_parallel_enabled", False))
        if tp > 1:
            raise NotImplementedError(
                f"Wan on CUDA runs tp=1 on one GPU in Release 1 (got tp={tp}); "
                "tensor parallel across PCIe is follow-up F6 of the CUDA backend plan"
            )
        if getattr(parallel, "sp_enabled", False):
            raise NotImplementedError(
                "Wan on CUDA has no sequence parallelism in Release 1 (it needs tp > 1); "
                "follow-up F6 of the CUDA backend plan"
            )
        if cp > 1 or dp > 1 or cfg_parallel:
            raise NotImplementedError(
                f"Wan on CUDA is one GPU in Release 1 (got cp={cp}, dp={dp}, "
                f"cfg_parallel={cfg_parallel}); CFG runs as sequential cond/uncond passes, "
                "and the second card is follow-up F2 of the CUDA backend plan"
            )
        from difflet.models.wan.cuda_application import CudaWanApplication

        return CudaWanApplication(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            shape=shape,
            **kwargs,
        )

    if backend != "trainium":
        raise NotImplementedError(
            f"Wan supports the trainium, tpu and cuda backends, got {backend!r}"
        )

    from difflet.models.wan.application import NeuronWanApplication

    return NeuronWanApplication(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        shape=shape,
        **kwargs,
    )
