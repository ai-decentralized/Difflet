"""MiniMax-H3 registry entry helpers."""

from __future__ import annotations

from typing import Any

from difflet.pipeline.parallel_config import DiffletParallelConfig


def create_minimax_h3_application(
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
            f"MiniMax-H3 currently supports only the trainium backend, got {backend!r}"
        )
    if parallel.tp_degree != 4:
        raise NotImplementedError(
            f"MiniMax-H3's first fixed graph is TP4, got TP{parallel.tp_degree}."
        )
    if parallel.cp_degree != 1:
        raise NotImplementedError("MiniMax-H3 context parallelism is not yet qualified.")
    if parallel.cfg_parallel_enabled:
        raise NotImplementedError(
            "MiniMax-H3 is guidance-distilled and has no CFG branch to parallelize."
        )
    if getattr(parallel, "sp_enabled", False):
        raise NotImplementedError("MiniMax-H3 sequence parallelism is not yet qualified.")

    from difflet.models.minimax_h3.application import NeuronMiniMaxH3Application

    return NeuronMiniMaxH3Application(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        shape=shape,
        **kwargs,
    )
