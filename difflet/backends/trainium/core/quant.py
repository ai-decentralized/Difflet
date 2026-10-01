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
# NxD's ActivationQuantizationType enum *values* are lowercase ("dynamic",
# "static"); NeuronConfig validates the raw string against those values, so the
# member name "DYNAMIC" is rejected ("Unsupported activation quantization type").
_NXD_ACTIVATION_TYPE = {"dynamic": "dynamic", "none": None}
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


# ------------------------------------------------------------- activations
#
# NxD's own DYNAMIC path (QuantizedColumnParallel / QuantizedRowParallel.forward)
# quantizes the activation *per channel along axis 1* and then dequantizes the
# output with scale_dequantize, which unsqueezes the scale past the output's
# rank: on the 3-D [batch, seq, hidden] activations a DiT feeds it the trace
# fails with "Check failed: input_sizes.size() <= output_sizes.size() (4 vs. 3)"
# (trn2, 2026-10-01). NxDI only consumes DYNAMIC through its NKI MLP kernels.
# Difflet therefore owns the activation path: per-tensor absmax scales, the
# same law as the CPU reference, in subclasses of the NxD layers that fall back
# to NxD's forward for weight-only quantization.


def quantize_activation_per_tensor(
    x: "torch.Tensor", clamp_bound: float = float("inf")
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Dynamic per-tensor absmax fp8 quantization of an activation of any rank.

    Returns ``(fp8 tensor, float32 0-D scale)``; the scale broadcasts over any
    output rank. Same law as ``difflet.quant.fp8.quantize_activation`` (so
    device-fp8 and CPU-fp8 agree in value); ``clamp_bound`` is NxD's
    ``quantize_clamp_bound`` applied to the absmax before scaling.
    """
    import torch

    from difflet.quant.fp8 import FP8_DTYPE, FP8_MAX, FP8_MIN_SCALE

    x32 = x.to(torch.float32)
    # Explicit dims: on XLA, amax() with no dims traced as a no-op "reduction"
    # (trn2, 2026-10-01: the scale came back input-shaped, f32[1,16,128]).
    amax = x32.abs().amax(dim=tuple(range(x32.ndim)))
    if clamp_bound != float("inf"):
        amax = amax.clamp(max=clamp_bound)
    scale = (amax / FP8_MAX).clamp_min(FP8_MIN_SCALE)
    quantized = (x32 / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return quantized, scale


def dynamic_fp8_linear(layer: Any, input_parallel: "torch.Tensor", **impl_kwargs: Any) -> "torch.Tensor":
    """``input @ W^T`` with both operands fp8 and a per-tensor dynamic input scale.

    ``layer`` is an NxD quantized parallel linear: ``weight`` (fp8), ``scale``
    (float32, ``[1]`` per-tensor or ``[out, 1]`` per-channel), ``_forward_impl``
    and ``clamp_bound``. The output is dequantized by ``input_scale *
    weight_scale`` and cast back to the input dtype; the weight scale is
    flattened so it broadcasts over the output's last dim for any rank.
    """
    import torch

    original_dtype = input_parallel.dtype
    quantized, input_scale = quantize_activation_per_tensor(
        input_parallel, getattr(layer, "clamp_bound", float("inf"))
    )
    output = layer._forward_impl(input=quantized, weight=layer.weight, bias=None, **impl_kwargs)
    output = output.to(torch.float32) * input_scale * layer.scale.to(torch.float32).reshape(-1)
    return output.to(original_dtype)


_MAPPING: dict[Any, Any] | None = None


def quant_module_mapping() -> dict[Any, Any]:
    """``convert()`` mapping: NxD parallel linears -> Difflet's quantized forms.

    The classes are NxD's ``QuantizedColumnParallel`` / ``QuantizedRowParallel``
    (same parameters, same checkpoint adaptors, same weight-only forward) with
    the DYNAMIC activation branch replaced by :func:`dynamic_fp8_linear`.
    """
    global _MAPPING
    if _MAPPING is not None:
        return _MAPPING
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
    from neuronx_distributed.quantization import quantization_layers as ql
    from neuronx_distributed.quantization.quantization_config import ActivationQuantizationType

    def _adopt(cls, mod, new_mod):
        new_mod.__class__ = cls  # NxD's from_float instantiates its own class by name
        # NxD types the bias and the dequantized dtype from mod.dtype — the
        # construction-time dtype. Difflet builds the model and then casts it
        # (model.to(bf16)), so mod.dtype is still float32 and every biased
        # quantized linear promoted its output to fp32 (trn2, 2026-10-01: the
        # weight-only HLO ran 316/400 dots as F32 x F32). Use the live dtype.
        dtype = mod.weight.dtype
        new_mod.dtype = dtype
        new_mod.dequantized_dtype = dtype
        if getattr(new_mod, "bias", None) is not None:
            new_mod.bias.data = new_mod.bias.data.to(dtype)
        return new_mod

    class PerTensorDynamicColumnParallel(ql.QuantizedColumnParallel):
        @classmethod
        def from_float(cls, mod, q_config=ql._DEFAULT_CUSTOM_QCONFIG_DICT):
            return _adopt(cls, mod, super().from_float(mod, q_config))

        def forward(self, input, *args, **kwargs):
            if self.activation_quantization_type != ActivationQuantizationType.DYNAMIC:
                return super().forward(input, *args, **kwargs)
            if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
                input_parallel = input
            else:
                input_parallel = ql.copy_to_tensor_model_parallel_region(
                    input, process_group=self.tensor_parallel_group
                )
            output_parallel = dynamic_fp8_linear(
                self, input_parallel,
                async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                autograd_func_class=self.autograd_func_class,
                save_for_backward=False,
                process_group=self.tensor_parallel_group,
            )
            if self.gather_output:
                assert not self.sequence_parallel_enabled
                output = ql.gather_from_tensor_model_parallel_region(
                    output_parallel, process_group=self.tensor_parallel_group
                )
            else:
                output = output_parallel
            return (output + self.bias) if self.bias is not None else output

    class PerTensorDynamicRowParallel(ql.QuantizedRowParallel):
        @classmethod
        def from_float(cls, mod, q_config=ql._DEFAULT_CUSTOM_QCONFIG_DICT):
            return _adopt(cls, mod, super().from_float(mod, q_config))

        def forward(self, input_, *args, **kwargs):
            if self.activation_quantization_type != ActivationQuantizationType.DYNAMIC:
                return super().forward(input_, *args, **kwargs)
            if self.input_is_parallel:
                input_parallel = input_
            else:
                assert not self.sequence_parallel_enabled
                input_parallel = ql.scatter_to_tensor_model_parallel_region(
                    input_, process_group=self.tensor_parallel_group
                )
            # Under TP each rank scales its own input slice: the partial products
            # are dequantized exactly before the all-reduce.
            output_ = dynamic_fp8_linear(
                self, input_parallel,
                async_grad_allreduce=False,
                sequence_parallel_enabled=False,
                sequence_dimension=self.sequence_dimension,
                autograd_func_class=self.autograd_func_class,
                save_for_backward=False,
                process_group=self.tensor_parallel_group,
            )
            if self.reduce_output:
                if self.sequence_parallel_enabled:
                    output_ = ql.reduce_scatter_to_sequence_parallel_region(
                        output_, self.sequence_dimension, process_group=self.tensor_parallel_group
                    )
                else:
                    output_ = ql.reduce_from_tensor_model_parallel_region(
                        output_, process_group=self.tensor_parallel_group
                    )
            return (output_ + self.bias) if self.bias is not None else output_

    _MAPPING = {
        ColumnParallelLinear: PerTensorDynamicColumnParallel,
        RowParallelLinear: PerTensorDynamicRowParallel,
    }
    return _MAPPING


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
        mapping=quant_module_mapping(),
        modules_to_not_convert=getattr(neuron_config, "modules_to_not_convert", None),
    )
    return model


__all__ = [
    "FP8_HLO2TENSORIZER_FLAG",
    "build_q_config",
    "dynamic_fp8_linear",
    "fp8_hlo2tensorizer_options",
    "is_fp8_quantized",
    "neuron_config_kwargs",
    "quant_module_mapping",
    "quantize_activation_per_tensor",
    "quantize_traced_model_",
]
