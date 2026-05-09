"""Flux registry entry helpers."""

from __future__ import annotations

from typing import Any

from nova.pipeline.parallel_config import NovaParallelConfig


def create_flux_application(
    *,
    model_path: str,
    parallel: NovaParallelConfig,
    dtype: Any,
    shape: dict[str, int | None],
    backend: str = "trainium",
    **kwargs: Any,
) -> Any:
    if backend != "trainium":
        raise NotImplementedError(
            f"Flux currently supports only the trainium backend, got {backend!r}"
        )

    from nova.models.flux.application import (
        NeuronFluxApplication,
        create_flux_config,
        get_flux_parallelism_config,
    )

    height = int(shape.get("height") or 1024)
    width = int(shape.get("width") or 1024)
    world_size = get_flux_parallelism_config(
        backbone_tp_degree=parallel.tp_degree,
        context_parallel_enabled=parallel.cp_enabled,
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
        context_parallel_enabled=parallel.cp_enabled,
    )
    return NeuronFluxApplication(
        model_path,
        *configs,
        height=height,
        width=width,
        **kwargs,
    )
