"""The FP8 linear kernel of the neuron backend: an NKI matmul on Trainium3's MX tensor-engine
path, called from the native backend as a custom op (the way ``attention.py`` calls the flash
kernel), with static per-tensor scales.

Why this kernel exists: Neuron SDK 2.33's TorchNeuron has no FP8 matmul reachable from PyTorch
(``torch._scaled_mm`` does not lower; TorchAO/FP8 is a private beta). What the hardware has is
``nisa.nc_matmul_mx``: fp8 x fp8 with E8M0 block scales, 4x the bf16 rate ("4x perf mode" in
nkilib). Per-tensor FP8 rides on it by setting every block scale to 1.0 (E8M0 127) and applying
the one fp32 scale (``input_scale * weight_scale``) on the way out of PSUM, so the arithmetic is
exactly ``FakeQuantLinear``'s: fp8 operands, fp32 accumulation, one output scale. Measured on a
trn3 core at 4096x3072 @ 3072x3072: 0.178 ms against 0.607 ms for bf16 ``F.linear`` (with the
activation quantized in the graph, 0.229 ms).

Layout: the kernel contracts over the partition dimension, 4 fp8 values packed per int32, so
both operands arrive as int32 ``[K/4, rows]``: ``pack_x4`` turns an fp8 ``[rows, K]`` tensor
into that (``[k // 4, row]`` holds ``t[row, 4(k//4) .. 4(k//4)+3]``, byte 0 the lowest k). The
weight is packed once after the load (``prepare_fp8_kernels_``); the activation is quantized
and packed in the compiled graph each call. Rows (tokens) are padded to a multiple of
``128 * LNC`` (the M tiles are sharded over the two LNC programs) and the result is cut back.

Shapes: K/4 must tile into 128-partition tiles with at most one trailing 64- or 32-partition
tile (``nc_matmul_mx`` takes 32, 64 or 128); every Qwen-Image shard shape does (K 3072 ->
6 x 128; the tp4 row-parallel K 768 -> 128 + 64). N is taken in weight chunks that stay under
the SBUF budget, each in PSUM tiles of up to 512 columns.
"""

from __future__ import annotations

import math

import torch

from difflet.quant.fp8 import FP8_DTYPE

P = 128
#: One PSUM bank: 512 fp32 columns.
BANK = 512
#: Bytes per partition the resident weight chunk may take (trn3 SBUF: 224 KiB per partition;
#: the activation tile, the output tile and the scales take a few KiB more).
WEIGHT_SBUF_BYTES = 160 * 1024
OP_NAME = "difflet_neuron::fp8_linear_smx"
#: E8M0 exponent of 1.0: the block scales are all unity on the per-tensor path.
E8M0_ONE = 127


# --------------------------------------------------------------------------- host helpers


def pack_x4(t: torch.Tensor) -> torch.Tensor:
    """fp8 ``[rows, K]`` -> int32 ``[K/4, rows]`` for the kernel (K a multiple of 4)."""
    rows, k = t.shape
    if k % 4:
        raise ValueError(f"K={k} is not a multiple of 4")
    if t.dtype != FP8_DTYPE:
        raise TypeError(f"expected {FP8_DTYPE}, got {t.dtype}")
    # Four adjacent fp8 bytes along K are already one int32 in a contiguous [rows, K] tensor,
    # so the pack is a view and the only data movement is a 4-byte-element transpose (a
    # quarter of the elements a byte-wise permute would move).
    return t.contiguous().view(torch.int32).t().contiguous()


def unpack_x4(packed: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`pack_x4`: int32 ``[K/4, rows]`` -> fp8 ``[rows, K]``."""
    k4, rows = packed.shape
    return packed.unsqueeze(-1).view(FP8_DTYPE).permute(1, 0, 2).reshape(rows, k4 * 4)


def k_tiles(k: int) -> tuple[int, int]:
    """``(full 128-partition tiles, trailing tile partitions)`` of ``K/4``; the trailing tile is
    0, 32 or 64, else the shape is unsupported."""
    if k % 4:
        raise ValueError(f"K={k} is not a multiple of 4")
    k4 = k // 4
    full, rem = divmod(k4, P)
    if rem not in (0, 32, 64):
        raise ValueError(
            f"K={k} (K/4={k4}) does not tile into 128-partition tiles plus a 32/64 remainder; "
            "nc_matmul_mx takes 32, 64 or 128 partitions")
    return full, rem


def weight_chunk(n: int, k: int) -> int:
    """Columns of the weight kept resident in SBUF at a time: ``n`` when it fits
    ``WEIGHT_SBUF_BYTES`` per partition, else the largest multiple of ``BANK`` dividing ``n``
    that does."""
    full, rem = k_tiles(k)
    tiles = full + (1 if rem else 0)
    per_column = tiles * 4  # int32 per K tile per column
    if n * per_column <= WEIGHT_SBUF_BYTES:
        return n
    chunk = (WEIGHT_SBUF_BYTES // per_column) // BANK * BANK
    while chunk >= BANK and n % chunk:
        chunk -= BANK
    if chunk < BANK:
        raise ValueError(f"no SBUF-resident weight chunk for N={n}, K={k}")
    return chunk


def logical_cores() -> int:
    try:
        from torch_neuronx.neuron_dynamo_backend.decompositions import get_logical_neuron_cores

        return int(get_logical_neuron_cores())
    except ImportError:
        return 1


def padded_rows(rows: int, lnc: int) -> int:
    unit = P * lnc
    return math.ceil(rows / unit) * unit


# ------------------------------------------------------------------------------ the kernel


def _build_kernel():
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def fp8_mm_static_mx(x4, w4, scale, n_chunk: int, k_full: int, k_rem: int):
        """``out[M, N] = (x @ w^T) * scale``: ``x4`` int32 ``[K/4, M]``, ``w4`` int32
        ``[K/4, N]`` (both fp8 e4m3fn packed by fours along K), ``scale`` fp32 ``[128, 1]``
        (the one output scale, broadcast over the partition rows); bf16 out. ``M`` a multiple
        of ``128 * num_programs``; ``N`` a multiple of ``n_chunk``."""
        K4, M = x4.shape
        N = w4.shape[1]
        nkt = k_full + (1 if k_rem else 0)
        nmt = M // P
        nprg, prg = nl.num_programs(0), nl.program_id(0)
        m_per_prg = nmt // nprg
        n_chunks = N // n_chunk
        n_banks = (n_chunk + BANK - 1) // BANK
        out = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        scale_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=scale_sb, src=scale)
        # Unity block scales (E8M0 127): the per-tensor scale is applied after accumulation.
        x_mxs = nl.ndarray((P, P), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(dst=x_mxs, value=E8M0_ONE, engine=nisa.gpsimd_engine)
        w_mxs = nl.ndarray((P, BANK), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(dst=w_mxs, value=E8M0_ONE, engine=nisa.gpsimd_engine)
        for nc in range(n_chunks):
            c0 = nc * n_chunk
            w_sb = nl.ndarray((P, nkt, n_chunk), dtype=nl.int32, buffer=nl.sbuf)
            for kt in range(k_full):
                nisa.dma_copy(dst=w_sb[:, kt, :], src=w4[kt * P:(kt + 1) * P, c0:c0 + n_chunk])
            if k_rem:
                nisa.dma_copy(dst=w_sb[:k_rem, k_full, :],
                              src=w4[k_full * P:k_full * P + k_rem, c0:c0 + n_chunk])
            w_x4 = w_sb.view(nl.float8_e4m3fn_x4)
            for mi in nl.affine_range(m_per_prg):
                mt = prg * m_per_prg + mi
                x_sb = nl.ndarray((P, nkt, P), dtype=nl.int32, buffer=nl.sbuf)
                for kt in range(k_full):
                    nisa.dma_copy(dst=x_sb[:, kt, :],
                                  src=x4[kt * P:(kt + 1) * P, mt * P:(mt + 1) * P])
                if k_rem:
                    nisa.dma_copy(dst=x_sb[:k_rem, k_full, :],
                                  src=x4[k_full * P:k_full * P + k_rem, mt * P:(mt + 1) * P])
                x_x4 = x_sb.view(nl.float8_e4m3fn_x4)
                o_sb = nl.ndarray((P, n_chunk), dtype=nl.bfloat16, buffer=nl.sbuf)
                for b in range(n_banks):
                    b0 = b * BANK
                    width = min(BANK, n_chunk - b0)
                    ps = nl.ndarray((P, width), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(nkt):
                        kp = P if kt < k_full else k_rem
                        nisa.nc_matmul_mx(
                            dst=ps, stationary=x_x4[:kp, kt, :],
                            moving=w_x4[:kp, kt, b0:b0 + width],
                            stationary_scale=x_mxs[:kp, :], moving_scale=w_mxs[:kp, :width],
                            accumulate=(kt > 0))
                    nisa.tensor_scalar(dst=o_sb[:, b0:b0 + width], data=ps, op0=nl.multiply,
                                       operand0=scale_sb)
                nisa.dma_copy(dst=out[mt * P:(mt + 1) * P, c0:c0 + n_chunk], src=o_sb)
        return out

    return fp8_mm_static_mx


def _load_op():
    """The custom op ``difflet_neuron::fp8_linear_smx(x4, w4, scale, n_chunk, k_full, k_rem)``,
    registered once (``nki_op``, as the attention ops are), or None without the Neuron SDK."""
    try:
        return torch.ops.difflet_neuron.fp8_linear_smx  # already registered (module reload)
    except (AttributeError, RuntimeError):
        pass
    try:
        from torch_neuronx import wrap_nki
        from torch_neuronx.nki_hop import nki_op
    except ImportError:
        return None
    # wrap_nki: the traceable callable whose ``[grid]`` sets the LNC launch (as the attention
    # kernels are called); a raw nki.jit kernel rejects the grid.
    kernel = wrap_nki(_build_kernel())

    def fake(x4, w4, scale, n_chunk, k_full, k_rem):
        return torch.empty((x4.shape[1], w4.shape[1]), dtype=torch.bfloat16, device=x4.device)

    @nki_op(OP_NAME, mutates_args={}, fake_fn=fake)
    def fp8_linear_smx(x4: torch.Tensor, w4: torch.Tensor, scale: torch.Tensor, n_chunk: int,
                       k_full: int, k_rem: int) -> torch.Tensor:
        grid = (logical_cores(),)
        return kernel[grid](x4, w4, scale, n_chunk, k_full, k_rem)

    return fp8_linear_smx


def _load_mx_op():
    """The ``mxfp8`` linear: nkilib's ``matmul_mxfp8`` with the bf16 activation as the kernel
    input (quantized inside, MX block scales) and the weight pre-quantized in the swizzled x4
    layout (``difflet/quant/mx.kernel_layout``). ``difflet_neuron::mx_linear(x2d, data,
    scales) -> bf16 [M, N]``. The kernel takes the activation straight from the producing op,
    which is what keeps it out of the compiler's slow path (see ``mx.py``)."""
    try:
        return torch.ops.difflet_neuron.mx_linear
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
        return matmul_mxfp8(x, w_x4, rhs_scales=w_scales, lhs_is_swizzled=False,
                            float8_dtype="float8_e4m3fn", output_dtype=nl.bfloat16,
                            run_with_lnc2=True)

    kernel = wrap_nki(mx_linear_kernel)

    def fake(x, w_data, w_scales):
        return torch.empty((x.shape[0], w_data.shape[1]), dtype=torch.bfloat16, device=x.device)

    @nki_op("difflet_neuron::mx_linear", mutates_args={}, fake_fn=fake)
    def mx_linear(x: torch.Tensor, w_data: torch.Tensor, w_scales: torch.Tensor) -> torch.Tensor:
        grid = (logical_cores(),)
        return kernel[grid](x, w_data, w_scales)

    return mx_linear


# Resolved once at import: no import or registration inside traced (compiled) code.
_FP8_OP = _load_op()
_MX_OP = _load_mx_op()


def available() -> bool:
    return _FP8_OP is not None


def mx_available() -> bool:
    return _MX_OP is not None


def mx_linear(x: torch.Tensor, weight_packed: torch.Tensor, weight_scales: torch.Tensor
              ) -> torch.Tensor:
    """``x @ W^T`` on the MX path: ``x`` ``[..., K]`` bf16 (quantized in the kernel),
    ``weight_packed`` int32 ``[K/4, N]`` and ``weight_scales`` uint8 ``[K/4, N]`` from
    ``prepare_mx_weight``; returns ``[..., N]`` in ``x``'s dtype."""
    if _MX_OP is None:
        raise RuntimeError("the mxfp8 kernel needs torch_neuronx and nkilib (the Neuron SDK); "
                           "use compute='dequant' elsewhere")
    k = x.shape[-1]
    n = weight_packed.shape[1]
    x2 = x.reshape(-1, k)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    out = _MX_OP(x2.contiguous(), weight_packed, weight_scales)
    return out.reshape(*x.shape[:-1], n).to(x.dtype)


def prepare_mx_weight(weight_q: torch.Tensor, weight_scale: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(fp8 [N, K], uint8 [N, K/32])`` of this rank's shard -> the kernel's ``(data, scales)``
    (``difflet/quant/mx.kernel_layout``), on the weight's device."""
    from difflet.quant import mx

    return mx.kernel_layout(weight_q, weight_scale)


# ------------------------------------------------------------------------- the torch side


def prepare_weight(weight: torch.Tensor, weight_scale: torch.Tensor, input_scale: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """What the kernel path keeps per layer after the load: the packed weight ``[K/4, N]``
    int32, the output scale ``[128, 1]`` fp32 (``input_scale * weight_scale``), the weight
    chunk and the K tiling. Checks the shape once, here, not per call."""
    n, k = weight.shape
    k_full, k_rem = k_tiles(k)
    chunk = weight_chunk(n, k)
    packed = pack_x4(weight.detach())
    out_scale = (input_scale.float() * weight_scale.float()).reshape(1, 1).expand(P, 1).contiguous()
    return packed, out_scale, chunk, k_full, k_rem


def fp8_linear(x: torch.Tensor, weight_packed: torch.Tensor, out_scale: torch.Tensor,
               input_scale: torch.Tensor, *, fp8_max: float, n_chunk: int, k_full: int,
               k_rem: int, lnc: int) -> torch.Tensor:
    """``x @ W^T`` with ``x`` quantized by the static ``input_scale`` in the graph; returns
    ``[..., N]`` in ``x``'s dtype."""
    if _FP8_OP is None:
        raise RuntimeError("the FP8 kernel needs torch_neuronx (the Neuron SDK); use "
                           "compute='dequant' elsewhere")
    k = x.shape[-1]
    n = weight_packed.shape[1]
    x2 = x.reshape(-1, k)
    rows = x2.shape[0]
    xq = (x2.float() * (1.0 / input_scale.float())).clamp(-float(fp8_max), float(fp8_max))
    xq = xq.to(FP8_DTYPE)
    padded = padded_rows(rows, lnc)
    if padded != rows:
        xq = torch.nn.functional.pad(xq, (0, 0, 0, padded - rows))
    x4 = pack_x4(xq)
    out = _FP8_OP(x4, weight_packed, out_scale, int(n_chunk), int(k_full), int(k_rem))
    if padded != rows:
        out = out[:rows]
    return out.reshape(*x.shape[:-1], n).to(x.dtype)


def reference(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
              input_scale: torch.Tensor, *, fp8_max: float) -> torch.Tensor:
    """The kernel's arithmetic in plain torch (fp8 operands, fp32 accumulation, one output
    scale), for the tests and the device checks."""
    from difflet.quant.fp8 import dequantize, fake_quantize

    xq = fake_quantize(x, input_scale, fp8_max=fp8_max)
    return torch.nn.functional.linear(xq, dequantize(weight, weight_scale)).to(x.dtype)


__all__ = [
    "BANK",
    "OP_NAME",
    "P",
    "available",
    "fp8_linear",
    "k_tiles",
    "logical_cores",
    "mx_available",
    "mx_linear",
    "pack_x4",
    "padded_rows",
    "prepare_mx_weight",
    "prepare_weight",
    "reference",
    "unpack_x4",
    "weight_chunk",
]
