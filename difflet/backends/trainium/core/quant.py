"""FP8 PTQ on Trainium: NxD's quantized parallel linears, driven by NeuronConfig.

The vendored NxDI plumbing already carries the quantization fields on
``NeuronConfig`` (``quantized``, ``quantized_checkpoints_path``,
``quantization_type``, ``quantization_dtype``, ``activation_quantization_type``,
``quantize_clamp_bound``, ``modules_to_not_convert``), the checkpoint loader
already renames ``weight_scale`` -> ``scale`` and leaves fp8 tensors uncast, and
``ModelWrapper`` adds the ``--experimental-unsafe-fp8e4m3fn-as-fp8e4m3``
compiler flag. What Difflet's own model instances never did is call the
conversion; :func:`quantize_traced_model_` is that call, mirrored from
``DecoderModelInstance.load_module`` in ``model_wrapper.py`` so the q-config is
built exactly the way NxDI builds it. No NKI kernel is involved: the FP8 GEMM
is whatever neuronx-cc lowers for ``QuantizedColumnParallel`` /
``QuantizedRowParallel``.
"""

from __future__ import annotations

import os
from typing import Any

from difflet.quant.spec import QuantSpec

_NXD_QUANTIZATION_TYPE = {
    "tensor": "per_tensor_symmetric",
    "channel": "per_channel_symmetric",
}
_NXD_ACTIVATION_TYPE = {"dynamic": "DYNAMIC", "none": None}
_NXD_QUANTIZED_DTYPE = {"fp8_e4m3": "f8e4m3"}
FP8_HLO2TENSORIZER_FLAG = "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"


def neuron_config_kwargs(spec: QuantSpec, quantized_checkpoints_path: str | os.PathLike[str]) -> dict[str, Any]:
    """``NeuronConfig(**kwargs)`` fields that turn a backbone into its FP8 form."""
    kwargs: dict[str, Any] = {
        "quantized": True,
        "quantized_checkpoints_path": str(quantized_checkpoints_path),
        "quantization_type": _NXD_QUANTIZATION_TYPE[spec.weight_granularity],
        "quantization_dtype": _NXD_QUANTIZED_DTYPE[spec.format],
    }
    activation = _NXD_ACTIVATION_TYPE[spec.activation]
    if activation is not None:
        kwargs["activation_quantization_type"] = activation
    return kwargs


def is_fp8_quantized(neuron_config: Any) -> bool:
    return bool(getattr(neuron_config, "quantized", False)) and (
        getattr(neuron_config, "quantization_dtype", None) == "f8e4m3"
    )


def fp8_hlo2tensorizer_options(neuron_config: Any) -> str:
    """Extra ``--internal-hlo2tensorizer-options`` tokens (with trailing space)
    a model's explicit compiler args must carry when it is FP8-quantized.

    ``ModelWrapper`` appends its own ``--internal-hlo2tensorizer-options`` with
    this flag, but a model that passes explicit compiler args ends up with two
    such options on the command line; carrying the flag in both makes the
    outcome independent of which one neuronx-cc honours.
    """
    return f"{FP8_HLO2TENSORIZER_FLAG} " if is_fp8_quantized(neuron_config) else ""


def build_q_config(neuron_config: Any) -> dict[str, Any]:
    """The NxD q-config NxDI builds for ``neuron_config`` (see model_wrapper.py)."""
    from neuronx_distributed.quantization.quantization_config import (
        ActivationQuantizationType,
        QuantizationType,
        QuantizedDtype,
        get_default_custom_qconfig_dict,
        get_default_per_channel_custom_qconfig_dict,
    )

    quantization_type = QuantizationType(neuron_config.quantization_type)
    if quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
        q_config = get_default_per_channel_custom_qconfig_dict()
    elif quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
        q_config = get_default_custom_qconfig_dict()
    else:
        raise RuntimeError(f"{neuron_config.quantization_type} is not supported for FP8 PTQ")
    quantized_dtype = neuron_config.quantization_dtype
    q_config["quantized_dtype"] = (
        QuantizedDtype.get_dtype(quantized_dtype) if isinstance(quantized_dtype, str) else quantized_dtype
    )
    activation = getattr(neuron_config, "activation_quantization_type", None)
    if activation is not None:
        q_config["activation_quantization_type"] = ActivationQuantizationType(activation)
    else:
        # NxDI passes the raw None through the enum; keep the vendored behaviour
        # when the enum has a NONE member, else leave the q-config default.
        try:
            q_config["activation_quantization_type"] = ActivationQuantizationType(None)
        except ValueError:
            pass
    q_config["clamp_bound"] = getattr(neuron_config, "quantize_clamp_bound", float("inf"))
    return q_config


def quantize_traced_model_(model: Any, neuron_config: Any) -> Any:
    """In place: swap the model's NxD parallel linears for their quantized forms.

    No-op unless ``neuron_config.quantized``. Called from a backbone's
    ``_create_model`` after the bf16 cast, i.e. at trace time and at every
    load, so the traced graph and the weight loader agree on the layer set.
    """
    if not getattr(neuron_config, "quantized", False):
        return model
    from neuronx_distributed.quantization.quantize import convert

    convert(
        model,
        q_config=build_q_config(neuron_config),
        inplace=True,
        mapping=None,
        modules_to_not_convert=getattr(neuron_config, "modules_to_not_convert", None),
    )
    return model


__all__ = [
    "FP8_HLO2TENSORIZER_FLAG",
    "build_q_config",
    "fp8_hlo2tensorizer_options",
    "is_fp8_quantized",
    "neuron_config_kwargs",
    "quantize_traced_model_",
]
