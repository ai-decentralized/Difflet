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
    if backend not in {"trainium", "neuron"}:
        raise NotImplementedError(
            f"LTX-2 supports the trainium and neuron backends, got {backend!r}"
        )
    if parallel.cp_degree > 1:
        raise NotImplementedError("LTX-2 CP is deferred until the M4c transformer spike.")

    if backend == "neuron":
        # TorchNeuron, one process per NeuronCore (torchrun): the DiT on the device, the
        # Gemma-3 text path on rank 0's host. Keyword arguments and their defaults:
        # neuron_application._KWARG_DEFAULTS.
        from difflet.models.ltx_2.neuron_application import TorchNeuronLtx2Application

        return TorchNeuronLtx2Application(
            model_path=model_path,
            parallel=parallel,
            dtype=dtype,
            shape=shape,
            **kwargs,
        )

    from difflet.models.ltx_2.application import NeuronLTX2Application

    return NeuronLTX2Application(
        model_path=model_path,
        parallel=parallel,
        dtype=dtype,
        shape=shape,
        **kwargs,
    )
