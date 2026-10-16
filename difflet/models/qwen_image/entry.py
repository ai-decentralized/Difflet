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
    if backend not in {"trainium", "tpu", "cuda"}:
        raise NotImplementedError(
            f"Qwen-Image supports the trainium, tpu and cuda backends, got {backend!r}"
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

    if backend == "cuda":
        # Release 1 of the CUDA backend is one process on one device (plan D5, D6).
        # Refuse here, before 41 GB of DiT are loaded, rather than at the first collective.
        tp = int(getattr(parallel, "tp_degree", 1) or 1)
        cp = int(getattr(parallel, "cp_degree", 1) or 1)
        dp = int(getattr(parallel, "dp_degree", 1) or 1)
        if tp > 1 or dp > 1:
            raise NotImplementedError(
                f"Qwen-Image on CUDA runs tp=1 on one GPU in Release 1 (got tp={tp}, dp={dp}); "
                "multi-GPU is follow-up F2 of the CUDA backend plan"
            )
        if cp > 1:
            raise NotImplementedError(
                f"Qwen-Image on CUDA has no context parallelism in Release 1 (got cp={cp}); "
                "follow-up F2/F6 of the CUDA backend plan"
            )
        from difflet.models.qwen_image.cuda_application import CudaQwenImageApplication

        return CudaQwenImageApplication(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            shape=shape,
            **kwargs,
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
