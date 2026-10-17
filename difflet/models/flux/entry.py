"""Flux registry entry helpers."""

from __future__ import annotations

from typing import Any

from difflet.pipeline.parallel_config import DiffletParallelConfig


def create_flux_application(
    *,
    model_path: str,
    parallel: DiffletParallelConfig,
    dtype: Any,
    shape: dict[str, int | None],
    backend: str = "trainium",
    **kwargs: Any,
) -> Any:
    if backend not in {"trainium", "cuda"}:
        raise NotImplementedError(f"Flux supports the trainium and cuda backends, got {backend!r}")

    if backend == "cuda":
        # Release 1 of the CUDA backend is one process on one device.
        # Refuse here, before the 24 GB DiT is loaded, rather than at the first collective.
        # Two parallel-config types reach this factory (DiffletParallelConfig and serving's
        # ParallelTopology): ask, do not assume.
        tp = int(getattr(parallel, "tp_degree", 1) or 1)
        cp = int(getattr(parallel, "cp_degree", 1) or 1)
        dp = int(getattr(parallel, "dp_degree", 1) or 1)
        if tp > 1:
            raise NotImplementedError(
                f"Flux on CUDA runs tp=1 on one GPU (got tp={tp}); "
                "tensor parallelism (tp > 1) is not supported on the CUDA backend yet"
            )
        if getattr(parallel, "sp_enabled", False):
            raise NotImplementedError(
                "sequence parallelism is not supported on the CUDA backend yet (it needs tp > 1)"
            )
        if cp > 1 or dp > 1 or getattr(parallel, "cfg_parallel_enabled", False):
            raise NotImplementedError(
                f"Flux on CUDA runs on one GPU (got cp={cp}, dp={dp}, "
                f"cfg_parallel={bool(getattr(parallel, 'cfg_parallel_enabled', False))}); "
                "context, data and CFG parallelism are not supported for Flux on CUDA yet"
            )
        from difflet.models.flux.cuda_application import CudaFluxApplication

        return CudaFluxApplication(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            shape=shape,
            **kwargs,
        )

    from difflet.models.flux.application import (
        NeuronFluxApplication,
        create_flux_config,
        get_flux_parallelism_config,
    )

    height = int(shape.get("height") or 1024)
    width = int(shape.get("width") or 1024)
    taef1 = bool(kwargs.pop("taef1", False))
    taef1_path = kwargs.pop("taef1_path", None)
    compile_shapes = kwargs.pop("shapes", None)
    if compile_shapes:
        from difflet.pipeline.bucketing import canonicalize_shapes

        compile_shapes = canonicalize_shapes(compile_shapes)
    world_size = get_flux_parallelism_config(
        backbone_tp_degree=parallel.tp_degree,
        cp_degree=parallel.cp_degree,
        cfg_parallel_enabled=parallel.cfg_parallel_enabled,
    )
    configs = create_flux_config(
        model_path=model_path,
        world_size=world_size,
        backbone_tp_degree=parallel.tp_degree,
        dtype=dtype,
        height=height,
        width=width,
        cfg_parallel_enabled=parallel.cfg_parallel_enabled,
        context_parallel_enabled=parallel.cp_degree > 1,
        cp_mode=parallel.cp_mode,
        sp_enabled=getattr(parallel, "sp_enabled", False),
        taef1=taef1,
        taef1_path=taef1_path,
        compile_shapes=compile_shapes,
    )
    return NeuronFluxApplication(
        model_path,
        *configs,
        height=height,
        width=width,
        taef1=taef1,
        taef1_path=taef1_path,
        **kwargs,
    )
