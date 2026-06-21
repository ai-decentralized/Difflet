"""LTX-2 registry entry helpers."""

from __future__ import annotations

from typing import Any

from difflet.pipeline.parallel_config import DiffletParallelConfig


def create_ltx_2_application(
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
            f"LTX-2 currently supports only the trainium backend, got {backend!r}"
        )
    if parallel.cp_degree > 1:
        raise NotImplementedError("LTX-2 CP is deferred until the M4c transformer spike.")
    if parallel.cfg_parallel_enabled:
        raise NotImplementedError(
            "LTX-2 CFG-parallel is deferred until the M4c transformer spike."
        )

    from difflet.models.ltx_2.application import NeuronLTX2Application

    return NeuronLTX2Application(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        shape=shape,
        **kwargs,
    )
