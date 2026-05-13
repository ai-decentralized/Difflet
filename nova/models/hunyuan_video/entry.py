"""HunyuanVideo registry entry helpers."""

from __future__ import annotations

from typing import Any

from nova.pipeline.parallel_config import NovaParallelConfig


def create_hunyuan_video_application(
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
            f"HunyuanVideo currently supports only the trainium backend, got {backend!r}"
        )
    if parallel.cp_enabled:
        raise NotImplementedError("HunyuanVideo CP is deferred until M3 polish")
    if parallel.cfg_parallel_enabled:
        raise NotImplementedError("HunyuanVideo CFG-parallel is deferred until M3 polish")

    from nova.models.hunyuan_video.application import NeuronHunyuanVideoApplication

    return NeuronHunyuanVideoApplication(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        shape=shape,
        **kwargs,
    )


def create_hunyuan_video15_application(
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
            f"HunyuanVideo 1.5 currently supports only the trainium backend, got {backend!r}"
        )
    if parallel.cp_enabled:
        raise NotImplementedError("HunyuanVideo 1.5 CP is deferred until the transformer port")
    if parallel.cfg_parallel_enabled:
        raise NotImplementedError(
            "HunyuanVideo 1.5 CFG-parallel is deferred until the transformer port"
        )

    from nova.models.hunyuan_video.application import NeuronHunyuanVideoApplication

    return NeuronHunyuanVideoApplication(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        shape=shape,
        model_version="1.5",
        **kwargs,
    )
