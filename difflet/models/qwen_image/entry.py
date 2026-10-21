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
    if backend not in {"trainium", "tpu"}:
        raise NotImplementedError(
            f"Qwen-Image supports the trainium and tpu backends, got {backend!r}"
        )
    # Two different parallel-config types reach this factory: the pipeline's
    # DiffletParallelConfig and serving's ParallelTopology, which carries no
    # cfg flag. Ask, do not assume.
    if getattr(parallel, "cfg_parallel_enabled", False):
        raise NotImplementedError(
            "Qwen-Image is guidance-distilled (single forward pass with the "
            "guidance scale baked into the timestep embedding); CFG-parallel "
            "requires true two-pass classifier-free guidance and does not apply."
        )

    if backend == "tpu":
        from difflet.models.qwen_image.tpu_application import TpuQwenImageApplication

        return TpuQwenImageApplication(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            shape=shape,
            **kwargs,
        )

    from difflet.models.qwen_image.application import NeuronQwenImageApplication

    return NeuronQwenImageApplication(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        shape=shape,
        **kwargs,
    )

