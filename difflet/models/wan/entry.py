"""Wan registry entry helpers."""

from __future__ import annotations

from typing import Any

from difflet.pipeline.parallel_config import DiffletParallelConfig


def create_wan_application(
    *,
    model_path: str,
    parallel: DiffletParallelConfig,
    dtype: Any,
    shape: dict[str, int | None],
    backend: str = "trainium",
    **kwargs: Any,
) -> Any:
    if backend == "tpu":
        from difflet.models.wan.tpu_application import TpuWanApplication

        return TpuWanApplication(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            shape=shape,
            **kwargs,
        )

    if backend == "neuron":
        # TorchNeuron, one process per NeuronCore (torchrun): the DiT on the device (TP, SP,
        # CFG, CP), umT5 on rank 0's host by default or on the device
        # (text_encoder_device="neuron", the CLI's default), the VAE on the device
        # (enable_vae_decoder=True) or the host. Keyword arguments and their defaults:
        # neuron_application._KWARG_DEFAULTS.
        from difflet.models.wan.neuron_application import TorchNeuronWanApplication

        return TorchNeuronWanApplication(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            shape=shape,
            **kwargs,
        )

    if backend != "trainium":
        raise NotImplementedError(
            f"Wan supports the trainium, tpu and neuron backends, got {backend!r}"
        )

    from difflet.models.wan.application import NeuronWanApplication

    return NeuronWanApplication(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        shape=shape,
        **kwargs,
    )
