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
    if backend not in {"trainium", "neuron"}:
        raise NotImplementedError(
            f"HunyuanVideo supports the trainium and neuron backends, got {backend!r}"
        )
    if parallel.cfg_parallel_enabled:
        raise NotImplementedError(
            "HunyuanVideo is guidance-distilled (single forward pass with the "
            "guidance scale baked into the timestep embedding); CFG-parallel "
            "requires true two-pass classifier-free guidance and does not apply."
        )

    if backend == "neuron":
        # TorchNeuron, one process per NeuronCore (torchrun): the DiT on the device, the
        # Llama/CLIP encode on rank 0's host. Keyword arguments and their defaults:
        # neuron_application._KWARG_DEFAULTS.
        from difflet.models.hunyuan_video.neuron_application import (
            TorchNeuronHunyuanVideoApplication,
        )

        return TorchNeuronHunyuanVideoApplication(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            shape=shape,
            **kwargs,
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
    if parallel.cp_degree > 1 and parallel.cp_mode == "ring":
        # Same as HV1.0: the registry keeps ring for the native backend; only AOT rejects.
        raise NotImplementedError(
            "HunyuanVideo 1.5 on the trainium backend has no ring context parallelism: "
            "the Trainium joint ring-attention kernel has no text-mask path for the "
            "padded text stream. Use --cp-mode ulysses (or gather_kv)."
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
