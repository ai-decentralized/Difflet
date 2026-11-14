#!/usr/bin/env python3
"""Lock the MX block membership of the attention fork on one tile, on the device (MXFP8 attention
plan, ``docs/plans/2026-11-13-mxfp8-attention.md``, Task 3 Step 2).

usage: membership.py --out FILE      (NEURON_RT_VISIBLE_CORES = one idle core; taskset around it)

``nisa.quantize_mx`` quantizes the operands in the layout the fork's matmuls read them: the
group-of-4 layout of ``nc_matmul_mx``, partition ``g`` byte ``j`` of the x4 operand holding
contraction index ``4g + j``:

- P ``[128 queries, 512 keys]`` along keys, through nkilib's ``swizzle_quantize_mx`` (stride-2
  ``nc_transpose`` of the fp32 view, MM2's stationary operand, plan Task 3 Step 3(b));
- V ``[512 keys, 128 d]`` along keys, loaded by one DMA access pattern straight into the layout
  (MM2's moving operand);
- Q ``[128 rows, 128 d]`` along the head dim, through ``swizzle_quantize_mx`` (MM1's, as K).

Each one's scales and fp8 bytes are compared element by element with
``difflet.quant.mx.quantize_mx`` of the logical operand (rows x contraction) under the kernel
module's ``pv_perm`` / ``qk_perm``, and under ``swizzle_perm`` to show the comparison tells the two
memberships apart. The scale tile is zeroed first, so the rows ``quantize_mx`` writes show.
Inputs are bf16 with magnitudes ``2^U{-16..15} * U[1, 2)``: block maxima depend on membership.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # the checkout, for difflet

import torch  # noqa: E402

from difflet.quant.mx import BLOCK, quantize_mx, swizzle_perm  # noqa: E402

SEED = 0
SHAPES = {"p": (128, 512), "v": (512, 128), "q": (128, 128)}  # as the kernel reads them


def make_inputs() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for name, shape in SHAPES.items():
        e = torch.randint(-16, 16, shape, generator=g).float()
        sign = torch.randint(0, 2, shape, generator=g).float() * 2 - 1
        out[name] = (sign * (1 + torch.rand(shape, generator=g)) * torch.exp2(e)).bfloat16()
    return out


def logical(name: str, x: torch.Tensor) -> torch.Tensor:
    """``[rows, contraction]``: V contracts over its keys, its first axis."""
    return x.t() if name == "v" else x


def scale_rows(k: int) -> torch.Tensor:
    """The scale tile row of each block: an 8-partition group's scale sits in its 32-partition
    quadrant's first four rows (``difflet.quant.mx.kernel_layout``), group ``a`` at row
    ``(a // 4) * 32 + a % 4``."""
    a = torch.arange(k // BLOCK)
    return (a // 4) * 32 + a % 4


def decode(data_i32: torch.Tensor, scales_u8: torch.Tensor):
    """The device's ``[K/4, M]`` x4 data and scale tiles -> fp8 ``[M, K]`` in logical order (byte
    ``j`` of row ``g`` is index ``4g + j``) and scales ``[M, K/32]`` in block order."""
    k4, m = data_i32.shape
    fp8 = data_i32.contiguous().view(torch.uint8).view(k4, m, 4).permute(1, 0, 2).reshape(m, 4 * k4)
    return fp8.view(torch.float8_e4m3fn), scales_u8[scale_rows(4 * k4)].t()


def compare(x: torch.Tensor, data_i32: torch.Tensor, scales_u8: torch.Tensor,
            perm: torch.Tensor) -> dict:
    """Mismatches of the device's fp8 and scales with ``quantize_mx(x, perm)``, element by element,
    and the scale tile's rows outside :func:`scale_rows` that are not zero."""
    fp8, scales = decode(data_i32, scales_u8)
    ref_fp8, ref_scales = quantize_mx(x, perm)
    other = torch.ones(scales_u8.shape[0], dtype=torch.bool)
    other[scale_rows(x.shape[1])] = False
    return {"scales": ref_scales.numel(),
            "scale_mismatches": int((scales != ref_scales).sum()),
            "fp8": ref_fp8.numel(),
            "fp8_mismatches": int((fp8.view(torch.uint8) != ref_fp8.view(torch.uint8)).sum()),
            "nonzero_scale_rows_outside": int((scales_u8[other] != 0).any(1).sum())}


def device_quantize(inputs: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, ...]]:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nkilib.core.utils.allocator import create_auto_alloc_manager
    from nkilib.experimental.attention_mxfp8.attention_mxfp8_tkg_utils import swizzle_quantize_mx
    from torch_neuronx import wrap_nki

    from difflet.backends import get_backend
    from difflet.backends.neuron.ops_impl.fp8_kernels import logical_cores
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    @nki.jit
    def quantize_operands(p, v, q):
        sbm = create_auto_alloc_manager()
        sbm.open_scope(name="membership")
        outs = []
        for name, x in (("p", p), ("v", v), ("q", q)):
            if name == "v":
                k, m = x.shape
                src = sbm.alloc_stack((k // 4, 4 * m), dtype=nl.bfloat16)
                # src[g, 4n + j] = v[4g + j, n]
                nisa.dma_copy(dst=src.ap(pattern=[[4 * m, k // 4], [4, m], [1, 4]], offset=0),
                              src=x.ap(pattern=[[4 * m, k // 4], [1, m], [m, 4]], offset=0))
            else:
                m, k = x.shape
                src = sbm.alloc_stack((m, k), dtype=nl.bfloat16)
                nisa.dma_copy(dst=src, src=x)
            data = sbm.alloc_stack((k // 4, m), dtype=nl.float8_e4m3fn_x4)
            scale = sbm.alloc_stack((k // 4, m), dtype=nl.uint8)
            nisa.memset(scale, value=0)
            if name == "v":
                nisa.quantize_mx(dst=data, src=src, dst_scale=scale)
            else:
                swizzle_quantize_mx(src, data, scale, sbm)
            data_out = nl.ndarray((k // 4, m), dtype=nl.int32, buffer=nl.shared_hbm)
            scale_out = nl.ndarray((k // 4, m), dtype=nl.uint8, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=data_out, src=data.view(nl.int32))
            nisa.dma_copy(dst=scale_out, src=scale)
            outs += [data_out, scale_out]
        sbm.close_scope()
        return tuple(outs)

    get_backend("neuron").prepare_runtime(DiffletParallelConfig())
    # the compiler runs at LNC=2, so the grid must name both cores; each writes the same tiles
    grid = (logical_cores(),)
    outs = wrap_nki(quantize_operands)[grid](*(inputs[n].to("neuron") for n in SHAPES))
    outs = [t.cpu() for t in outs]
    return {n: (outs[2 * i], outs[2 * i + 1]) for i, n in enumerate(SHAPES)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    sys.path.insert(0, str(HERE))
    from probe import require_idle_core

    from difflet.backends.neuron.ops_impl.mx_attention_kernel import pv_perm, qk_perm

    inputs = make_inputs()
    fuser = require_idle_core()  # just before the device opens
    got = device_quantize(inputs)
    result = {"inputs": f"bf16, sign * U[1, 2) * 2^U{{-16..15}}, seed {SEED}", "fuser": fuser}
    for name, (data, scales) in got.items():
        x = logical(name, inputs[name])
        kernel_perm = (qk_perm if name == "q" else pv_perm)(x.shape[1])
        result[name] = {"logical": list(x.shape),
                        "kernel_perm": compare(x, data, scales, kernel_perm),
                        "swizzle_perm": compare(x, data, scales, swizzle_perm(x.shape[1]))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    torch.save(got, args.out.with_suffix(".pt"))
    print(json.dumps({n: result[n]["kernel_perm"] for n in SHAPES}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
