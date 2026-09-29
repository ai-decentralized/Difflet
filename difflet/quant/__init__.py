"""Post-training FP8 quantization of DiT linear layers (backend-neutral core).

Device execution lives in ``difflet.backends.trainium.core.quant`` (NxD
quantized parallel linears); this package holds the spec, the fp8 math, the
offline checkpoint quantizer, the CPU emulation and the comparison metrics.
"""

from difflet.quant.checkpoint import (
    ensure_quantized_checkpoint,
    is_valid_quantized_checkpoint,
    quantize_checkpoint_dir,
    quantize_state_dict,
    quantized_checkpoint_dir,
    read_manifest,
)
from difflet.quant.fake_linear import FakeQuantLinear, quantize_module_
from difflet.quant.fp8 import (
    FP8_DTYPE,
    FP8_MAX,
    dequantize,
    fp8_linear_reference,
    quantize_activation,
    quantize_weight,
)
from difflet.quant.spec import DEFAULT_TARGETS, QuantSpec

__all__ = [
    "DEFAULT_TARGETS",
    "FP8_DTYPE",
    "FP8_MAX",
    "FakeQuantLinear",
    "QuantSpec",
    "dequantize",
    "ensure_quantized_checkpoint",
    "fp8_linear_reference",
    "is_valid_quantized_checkpoint",
    "quantize_activation",
    "quantize_checkpoint_dir",
    "quantize_module_",
    "quantize_state_dict",
    "quantize_weight",
    "quantized_checkpoint_dir",
    "read_manifest",
]
