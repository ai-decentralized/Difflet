"""The MXFP8 linear kernel of the neuron backend: nkilib's ``matmul_mxfp8`` on Trainium3's MX
tensor-engine path, called from the native backend as a custom op (the way ``attention.py``
calls the flash kernel).

Why a kernel: Neuron SDK 2.33's TorchNeuron has no FP8 matmul reachable from PyTorch
(``torch._scaled_mm`` does not lower; TorchAO/FP8 is a private beta). What the hardware has is
``nisa.nc_matmul_mx``: fp8 x fp8 with E8M0 block scales at 4x the bf16 rate, NeuronCore-v4
(trn3) only. nkilib's kernel takes the bf16 activation straight from the producing op and
block-quantizes it inside (which keeps the graph out of the compiler's slow path for in-graph
quantize chains, see ``difflet/quant/mx.py``); the weight arrives pre-quantized in the swizzled
x4 layout (``mx.kernel_layout``), laid out on the host once after the load.
"""

from __future__ import annotations

import torch

from difflet.quant.mx import FP8_DTYPE  # noqa: F401  (re-exported for the layer module)


def logical_cores() -> int:
    try:
        from torch_neuronx.neuron_dynamo_backend.decompositions import get_logical_neuron_cores

        return int(get_logical_neuron_cores())
    except ImportError:
        return 1


def _load_mx_op():
    """The custom op ``difflet_neuron::mx_linear(x2d, data, scales) -> bf16 [M, N]``, registered
    once (``nki_op``, as the attention ops are), or None without the Neuron SDK."""
    try:
        return torch.ops.difflet_neuron.mx_linear  # already registered (module reload)
    except (AttributeError, RuntimeError):
        pass
    try:
        import nki
        import nki.language as nl
        from nkilib.experimental.matmul_mxfp8.matmul_mxfp8_generic_kernel import matmul_mxfp8
        from torch_neuronx import wrap_nki
        from torch_neuronx.nki_hop import nki_op
    except ImportError:
        return None

    @nki.jit
    def mx_linear_kernel(x, w_data, w_scales):
        w_x4 = w_data.view(nl.float8_e4m3fn_x4)
        # fast_dma_transpose: the activation's gather-transpose load addresses the source with
        # a direct 4-D pattern instead of per-element offset buffers in SBUF. Measured on one
        # trn3 core at Qwen's tp4 MLP shape (4096 x 3072 -> 3072): 0.339 -> 0.282 ms, the same
        # numbers; on the device 260 -> 239 ms per step (2026-11-01). The other loader options
        # (PE swizzle on a transposed activation, block/tile sizes, LNC2 sharding on M) were
        # not better end to end; ``quant_scheme="1x32"`` gives wrong numbers.
        return matmul_mxfp8(x, w_x4, rhs_scales=w_scales, lhs_is_swizzled=False,
                            float8_dtype="float8_e4m3fn", output_dtype=nl.bfloat16,
                            run_with_lnc2=True, fast_dma_transpose=True)

    # wrap_nki: the traceable callable whose ``[grid]`` sets the LNC launch (as the attention
    # kernels are called); a raw nki.jit kernel rejects the grid.
    kernel = wrap_nki(mx_linear_kernel)

    def fake(x, w_data, w_scales):
        return torch.empty((x.shape[0], w_data.shape[1]), dtype=torch.bfloat16, device=x.device)

    @nki_op("difflet_neuron::mx_linear", mutates_args={}, fake_fn=fake)
    def mx_linear(x: torch.Tensor, w_data: torch.Tensor, w_scales: torch.Tensor) -> torch.Tensor:
        grid = (logical_cores(),)
        return kernel[grid](x, w_data, w_scales)

    return mx_linear


# Resolved once at import: no import or registration inside traced (compiled) code.
_MX_OP = _load_mx_op()


def mx_available() -> bool:
    return _MX_OP is not None


#: The kernel block-quantizes the activation along its row (free) dim and refuses a row count
#: that is not a multiple of 8 (``NCC_INKI016 ... F dim must be divisible by 8``, one-core
#: probe, 2026-11-07). Every token count of Qwen-Image, FLUX and LTX-2's video and text streams
#: is one; LTX-2's audio stream (126 tokens at the default shape, 51 at 480x704x49),
#: HunyuanVideo-1.5 (49,290 image and 1,985 text tokens) and Wan under CP2 (2,340) are not.
#: Under LNC2 the kernel gives each physical core ``M // 2`` rows (a pre-quantized weight
#: shards the activation, ``resolve_lnc2_sharding``) and asserts the rule on the full ``M``
#: only; that half need not be a multiple of 8: 996 and 1172 rows per core (1,985 -> 1,992 and
#: 2,340 -> 2,344, a partial last 128-row tile) came back correct on the device
#: (``probe_mx_shapes_lnc2.json``), so 8 is the whole rule, not 16.
M_MULTIPLE = 8


def padded_rows(m: int) -> int:
    """The row count the kernel is called with for ``m`` activation rows: ``m`` rounded up to
    a multiple of :data:`M_MULTIPLE` (``m`` itself when it already is one)."""
    return -(-m // M_MULTIPLE) * M_MULTIPLE


def mx_linear(x: torch.Tensor, weight_packed: torch.Tensor, weight_scales: torch.Tensor
              ) -> torch.Tensor:
    """``x @ W^T`` on the MX path: ``x`` ``[..., K]`` bf16 (quantized in the kernel),
    ``weight_packed`` int32 ``[K/4, N]`` and ``weight_scales`` uint8 ``[K/4, N]`` from
    ``prepare_mx_weight``; returns ``[..., N]`` in ``x``'s dtype. The rows are zero-padded to
    :func:`padded_rows` for the kernel and the padding's outputs dropped; a count that already
    is a multiple adds no op (the compiled graph of every current model is unchanged)."""
    if _MX_OP is None:
        raise RuntimeError("the mxfp8 kernel needs torch_neuronx and nkilib (the Neuron SDK); "
                           "use compute='dequant' elsewhere")
    k = x.shape[-1]
    n = weight_packed.shape[1]
    x2 = x.reshape(-1, k)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    m = x2.shape[0]
    pad = padded_rows(m) - m
    if pad:
        # Zero rows: each row's block scales are its own, so the padding changes no real row's
        # quantization, and its rows of the product are sliced off below.
        x2 = torch.nn.functional.pad(x2, (0, 0, 0, pad))
    out = _MX_OP(x2.contiguous(), weight_packed, weight_scales)
    if pad:
        out = out[:m]
    return out.reshape(*x.shape[:-1], n).to(x.dtype)


def prepare_mx_weight(weight_q: torch.Tensor, weight_scale: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(fp8 [N, K], uint8 [N, K/32])`` of this rank's shard -> the kernel's ``(data, scales)``
    (``difflet/quant/mx.kernel_layout``), on the weight's device."""
    from difflet.quant import mx

    device = weight_q.device
    if device.type != "cpu":
        # The layout is a permutation gather plus a scatter: eager on the device that is a
        # chain of small launches per layer (45 s of start-up for Qwen-Image's 720 layers);
        # on the host it is milliseconds, and the result moves once.
        data, scales = mx.kernel_layout(weight_q.cpu(), weight_scale.cpu())
        return data.to(device), scales.to(device)
    return mx.kernel_layout(weight_q, weight_scale)


__all__ = [
    "FP8_DTYPE",
    "M_MULTIPLE",
    "logical_cores",
    "mx_available",
    "mx_linear",
    "padded_rows",
    "prepare_mx_weight",
]
