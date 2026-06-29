"""HunyuanVideo registry entry helpers."""

from __future__ import annotations

from typing import Any

from difflet.pipeline.parallel_config import DiffletParallelConfig


def create_hunyuan_video_application(
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
            f"HunyuanVideo currently supports only the trainium backend, got {backend!r}"
        )
    if parallel.cfg_parallel_enabled:
        raise NotImplementedError(
            "HunyuanVideo is guidance-distilled (single forward pass with the "
            "guidance scale baked into the timestep embedding); CFG-parallel "
            "requires true two-pass classifier-free guidance and does not apply."
        )

    from difflet.models.hunyuan_video.application import NeuronHunyuanVideoApplication

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
    parallel: DiffletParallelConfig,
    dtype: Any,
    shape: dict[str, int | None],
    backend: str = "trainium",
    **kwargs: Any,
) -> Any:
    if backend != "trainium":
        raise NotImplementedError(
            f"HunyuanVideo 1.5 currently supports only the trainium backend, got {backend!r}"
        )
    if parallel.cp_degree > 1:
        raise NotImplementedError("HunyuanVideo 1.5 CP is deferred until the transformer port")
    if parallel.cfg_parallel_enabled:
        raise NotImplementedError(
            "HunyuanVideo 1.5 is guidance-distilled (single forward pass with the "
            "guidance scale baked into the timestep embedding); CFG-parallel "
            "requires true two-pass classifier-free guidance and does not apply."
        )

    from difflet.models.hunyuan_video.application import NeuronHunyuanVideoApplication

    return NeuronHunyuanVideoApplication(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        shape=shape,
        model_version="1.5",
        **kwargs,
    )
