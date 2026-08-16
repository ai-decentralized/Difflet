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
    if backend != "trainium":
        raise NotImplementedError(
            f"Flux currently supports only the trainium backend, got {backend!r}"
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
        from difflet.backends.trainium.core.bucketing import canonicalize_shapes

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
