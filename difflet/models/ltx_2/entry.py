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
    if parallel.cp_degree > 1:
        raise NotImplementedError("LTX-2 CP is deferred until the M4c transformer spike.")
    if backend == "tpu":
        from difflet.models.ltx_2.tpu_application import TpuLTX2Application

        return TpuLTX2Application(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            shape=shape,
            **kwargs,
        )
    if backend != "trainium":
        raise NotImplementedError(
            f"LTX-2 supports the trainium and tpu backends, got {backend!r}"
        )

    from difflet.models.ltx_2.application import NeuronLTX2Application

    return NeuronLTX2Application(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        shape=shape,
        **kwargs,
    )
