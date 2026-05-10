"""Wan registry entry helpers."""

from __future__ import annotations

from typing import Any

from nova.pipeline.parallel_config import NovaParallelConfig


def create_wan_application(
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
            f"Wan currently supports only the trainium backend, got {backend!r}"
        )

    from nova.models.wan.application import NeuronWanApplication

    return NeuronWanApplication(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        shape=shape,
        **kwargs,
    )
