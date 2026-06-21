"""Qwen-Image registry entry helpers."""

from __future__ import annotations

from typing import Any

from difflet.pipeline.parallel_config import DiffletParallelConfig


def create_qwen_image_application(
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
            f"Qwen-Image currently supports only the trainium backend, got {backend!r}"
        )
    if parallel.cp_degree > 1:
        raise NotImplementedError("Qwen-Image CP is deferred until the M4a transformer spike.")
    if parallel.cfg_parallel_enabled:
        raise NotImplementedError(
            "Qwen-Image CFG-parallel is deferred until the M4a transformer spike."
        )

    from difflet.models.qwen_image.application import NeuronQwenImageApplication

    return NeuronQwenImageApplication(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        shape=shape,
        **kwargs,
    )

