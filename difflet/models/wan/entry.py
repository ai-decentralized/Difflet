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
        # The CUDA backend runs one expert pair per process at tp=1;
        # CFG-parallel runs the cond and uncond branches in two such processes
        # under torchrun. Refuse the rest here, before two 14B experts are loaded,
        # rather than at the first collective. Two parallel-config types reach this
        # factory (DiffletParallelConfig and serving's ParallelTopology): ask, do not
        # assume.
        tp = int(getattr(parallel, "tp_degree", 1) or 1)
        cp = int(getattr(parallel, "cp_degree", 1) or 1)
        dp = int(getattr(parallel, "dp_degree", 1) or 1)
        if tp > 1:
            raise NotImplementedError(
                f"Wan on CUDA runs tp=1 on one GPU (got tp={tp}); "
                "tensor parallelism (tp > 1) is not supported on the CUDA backend yet"
            )
        if getattr(parallel, "sp_enabled", False):
            raise NotImplementedError(
                "sequence parallelism is not supported on the CUDA backend yet (it needs tp > 1)"
            )
        if cp > 1:
            raise NotImplementedError(
                f"context parallelism (cp > 1) is not supported on the CUDA backend yet "
                f"(got cp={cp})"
            )
        if dp > 1:
            # Data-parallel replicas are separate single-GPU processes; the router
            # never hands one application a dp > 1 config.
            raise NotImplementedError(
                f"Wan on CUDA is one GPU per application (got dp={dp}); "
                "the second card is not supported on the CUDA backend yet"
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
