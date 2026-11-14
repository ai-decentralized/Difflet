#!/usr/bin/env python3
"""Lock the MX block membership of the attention fork on one tile, on the device (MXFP8 attention
plan, ``docs/plans/2026-11-13-mxfp8-attention.md``, Task 3 Step 2; ruling BR16 (1): rerun on the
loaders MM2 really uses; Task 4: and on MM1's).

usage: membership.py --out FILE      (NEURON_RT_VISIBLE_CORES = one idle core; taskset around it)

``nisa.quantize_mx`` quantizes the operands in the layout the fork's matmuls read them: the
group-of-4 layout of ``nc_matmul_mx``, partition ``g`` byte ``j`` of the x4 operand holding
contraction index ``4g + j``:

- P ``[128 queries, 512 keys]`` along keys by the fork's own ``_mx_quantize_p`` (stride-2 TensorE
  transposes of the fp32 view out of a wider exp tile, at a 512 offset as the kernel's later tiles
  are, then ``quantize_mx``), as ``_exp_impl`` does for MM2's stationary operand; ``p_tail``: only
  500 keys valid, the tail zeroed by the helper, as on the last tile of an unmasked call;
- V ``[512 keys, 128 d]`` along keys by the fork's ``_mx_load_v`` (one contiguous DMA, a GpSimd
  regroup, ``quantize_mx``), MM2's moving operand; ``v_tail``: 500 rows loaded, the rest zero;
- Q ``[128 rows, 128 d]`` along the head dim, through nkilib's ``swizzle_quantize_mx`` (3a's lock);
- MM1's own loaders (``qk_mx``), along the head dim: ``q_mm1`` one query group by ``_mx_load_q``,
  repeated in the four row quadrants, ``q_mm1_tail`` 100 rows valid; ``k_mm1`` one 2048-key large
  tile by ``_mx_load_k``, its 512-key tile t in quadrant t, ``k_mm1_tail`` 1500 keys valid (tiles of
  512, 512, 476 and 0). Each quadrant (the tile one ``nc_matmul_mx`` reads) is compared on its own.
  Their input pairs +0, -0 and bf16 subnormals (odd head-dim indices) with normal values, so the
  fp32 view the loaders transpose and copy holds zero-exponent words (subnormal fp32).

Each one's scales and fp8 bytes are compared element by element with
``difflet.quant.mx.quantize_mx`` of the logical operand (rows x contraction, keys past the valid
count zero) under the kernel module's ``pv_perm`` / ``qk_perm``, and under ``swizzle_perm`` to
show the comparison tells the two memberships apart. The scale tiles are zeroed first, so the rows
``quantize_mx`` writes show. Inputs are bf16 with magnitudes ``2^U{-16..15} * U[1, 2)``: block
maxima depend on membership.
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
SHAPES = {"p": (128, 512), "v": (512, 128), "q": (128, 128), "kq": (2048, 128)}  # as the kernel reads them
VALID = 500  # keys of the partial tiles, not a multiple of 32
#: operand -> (input, valid contraction indices; for MM1's, valid rows); the kernel's output order
OPERANDS = {"p": ("p", 512), "p_tail": ("p", VALID), "v": ("v", 512), "v_tail": ("v", VALID),
            "q": ("q", 128), "q_mm1": ("kq", 128), "q_mm1_tail": ("kq", 100),
            "k_mm1": ("kq", 2048), "k_mm1_tail": ("kq", 1500)}
MM1 = ("q_mm1", "q_mm1_tail", "k_mm1", "k_mm1_tail")


def make_inputs() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(SEED)
    out = {}
    for name, shape in SHAPES.items():
        e = torch.randint(-16, 16, shape, generator=g).float()
        sign = torch.randint(0, 2, shape, generator=g).float() * 2 - 1
        out[name] = (sign * (1 + torch.rand(shape, generator=g)) * torch.exp2(e)).bfloat16()
    kq = out["kq"]  # zero-exponent high halves in the fp32 view: +0, -0, a bf16 subnormal
    kq[0::3, 1::2] = 0.0
    kq[1::3, 1::4] = -0.0
    kq[2::3, 1::2] = 2.0 ** -130
    return out


def logical(name: str, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """``[rows, contraction]`` (V contracts over its keys, its first axis), indices past the
    operand's valid count zero; MM1's: the rows its loader reads (a query group, a large tile),
    rows past the valid count zero."""
    src, valid = OPERANDS[name]
    if name in MM1:
        x = inputs[src][:128 if name.startswith("q") else 2048].clone()
        x[valid:] = 0
        return x
    x = inputs[src].t() if src == "v" else inputs[src]
    x = x.clone()
    x[:, valid:] = 0
    return x


def tiles(name: str, inputs: dict[str, torch.Tensor], data: torch.Tensor, scales: torch.Tensor):
    """``(logical, data, scales)`` per tile a matmul reads: the whole operand, or for MM1's each
    row quadrant ``32 t ..`` (Q's group in every quadrant; K's 512-key tile t in quadrant t)."""
    x = logical(name, inputs)
    if name not in MM1:
        return [(x, data, scales)]
    q = name.startswith("q")
    return [(x if q else x[512 * t:512 * (t + 1)], data[32 * t:32 * (t + 1)], scales[32 * t:32 * (t + 1)])
            for t in range(4)]


def scale_rows(k: int) -> torch.Tensor:
    """The scale tile row of each block: an 8-partition group's scale sits in its 32-partition
    quadrant's first four rows (``difflet.quant.mx.kernel_layout``), group ``a`` at row
    ``(a // 4) * 32 + a % 4``."""
    a = torch.arange(k // BLOCK)
    return (a // 4) * 32 + a % 4


def decode(data_i32: torch.Tensor, scales_u8: torch.Tensor, perm: torch.Tensor):
    """The device's ``[K/4, M]`` x4 data and scale tiles -> fp8 ``[M, K]`` in logical order (byte
    ``j`` of row ``r`` is index ``perm[4r + j]``) and scales ``[M, K/32]`` in block order."""
    k4, m = data_i32.shape
    fp8_sw = data_i32.contiguous().view(torch.uint8).view(k4, m, 4).permute(1, 0, 2).reshape(m, 4 * k4)
    fp8 = torch.empty_like(fp8_sw)
    fp8[:, perm] = fp8_sw
    return fp8.view(torch.float8_e4m3fn), scales_u8[scale_rows(4 * k4)].t()


def compare(x: torch.Tensor, data_i32: torch.Tensor, scales_u8: torch.Tensor,
            perm: torch.Tensor) -> dict:
    """Mismatches of the device's fp8 and scales with ``quantize_mx(x, perm)``, element by element
    (the device bytes placed by the same ``perm``), and the scale tile's rows outside
    :func:`scale_rows` that are not zero. ``zero_block_scale_mismatches``: the scale mismatches
    whose block is all zero (its data bytes are zero, so its scale multiplies nothing; the device
    writes 2^-126 there, the CPU quantizer 2^-127: Task 4b)."""
    fp8, scales = decode(data_i32, scales_u8, perm)
    ref_fp8, ref_scales = quantize_mx(x, perm)
    other = torch.ones(scales_u8.shape[0], dtype=torch.bool)
    other[scale_rows(x.shape[1])] = False
    zero = (x[:, perm].reshape(x.shape[0], -1, BLOCK) == 0).all(-1)
    return {"scales": ref_scales.numel(),
            "scale_mismatches": int((scales != ref_scales).sum()),
            "zero_block_scale_mismatches": int(((scales != ref_scales) & zero).sum()),
            "fp8": ref_fp8.numel(),
            "fp8_mismatches": int((fp8.view(torch.uint8) != ref_fp8.view(torch.uint8)).sum()),
            "nonzero_scale_rows_outside": int((scales_u8[other] != 0).any(1).sum())}


def to_numpy(t: torch.Tensor):
    import ml_dtypes

    return t.float().numpy().astype(ml_dtypes.bfloat16)


def unpack(outs) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """The kernel's flat outputs -> ``{operand: (data int32, scales uint8)}``."""
    return {n: (outs[2 * i], outs[2 * i + 1]) for i, n in enumerate(OPERANDS)}


def _build_kernel():
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nkilib.core.utils.allocator import create_auto_alloc_manager
    from nkilib.experimental.attention_mxfp8.attention_mxfp8_tkg_utils import swizzle_quantize_mx

    from difflet.backends.neuron.ops_impl.mx_attention_kernel import (
        _mx_load_k,
        _mx_load_q,
        _mx_load_v,
        _mx_quantize_p,
    )

    @nki.jit
    def quantize_operands(p, v, q, kq):
        sbm = create_auto_alloc_manager()
        sbm.open_scope(name="membership")
        outs = []
        for name, (src_name, valid) in OPERANDS.items():
            x = {"p": p, "v": v, "q": q, "kq": kq}[src_name]
            m, k = (x.shape[1], x.shape[0]) if src_name == "v" else x.shape
            if name in MM1:  # [128, F] x4: four quadrants of 32 partitions
                k, m = 512, 128 if name.startswith("q") else 512
            data = sbm.alloc_stack((k // 4, m), dtype=nl.float8_e4m3fn_x4)
            scale = sbm.alloc_stack((k // 4, m), dtype=nl.uint8)
            nisa.memset(scale, value=0)
            if name in MM1:
                psum = [nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.psum) for _ in range(2)]
                if name.startswith("q"):
                    nat = sbm.alloc_stack((128, 4 * x.shape[1]), dtype=nl.bfloat16)
                    stage = sbm.alloc_stack((128, 256), dtype=nl.float32)
                    _mx_load_q(x, data, scale, psum[0], stage, nat, row_offset=0, num_rows=valid)
                else:
                    nat = [sbm.alloc_stack((128, 4 * x.shape[1]), dtype=nl.bfloat16) for _ in range(4)]
                    stage = [sbm.alloc_stack((128, 256), dtype=nl.float32) for _ in range(4)]
                    _mx_load_k(x, data, scale, psum, stage, nat, row_offset=0, num_keys=valid)
            elif src_name == "v":
                stage = sbm.alloc_stack((k // 4, 4 * m), dtype=nl.bfloat16)
                packed = sbm.alloc_stack((k // 4, 4 * m), dtype=nl.bfloat16)
                _mx_load_v(x, data, scale, stage, packed, row_offset=0, num_keys=valid)
            elif src_name == "p":
                wide = sbm.alloc_stack((m, 2 * k), dtype=nl.bfloat16)  # the tile at offset k
                nisa.memset(wide, value=1.0e30)  # stale exp: must not reach the tail
                nisa.dma_copy(dst=wide[:, k:2 * k], src=x)
                psum = nl.ndarray((k // 4, 2 * m), dtype=nl.float32, buffer=nl.psum)
                stage = sbm.alloc_stack((k // 4, 2 * m), dtype=nl.float32)
                _mx_quantize_p(data, scale, psum, stage, wide, src_offset=k, num_q=m,
                               num_keys=valid)
            else:
                src = sbm.alloc_stack((m, k), dtype=nl.bfloat16)
                nisa.dma_copy(dst=src, src=x)
                swizzle_quantize_mx(src, data, scale, sbm)
            data_out = nl.ndarray((k // 4, m), dtype=nl.int32, buffer=nl.shared_hbm)
            scale_out = nl.ndarray((k // 4, m), dtype=nl.uint8, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=data_out, src=data.view(nl.int32))
            nisa.dma_copy(dst=scale_out, src=scale)
            outs += [data_out, scale_out]
        sbm.close_scope()
        return tuple(outs)

    return quantize_operands


def __getattr__(name):  # the kernel needs the Neuron SDK; built on first use
    if name == "quantize_operands":
        return _build_kernel()
    raise AttributeError(name)


def device_quantize(inputs: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, ...]]:
    from torch_neuronx import wrap_nki

    from difflet.backends import get_backend
    from difflet.backends.neuron.ops_impl.fp8_kernels import logical_cores
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    get_backend("neuron").prepare_runtime(DiffletParallelConfig())
    # the compiler runs at LNC=2, so the grid must name both cores; each writes the same tiles
    grid = (logical_cores(),)
    outs = wrap_nki(_build_kernel())[grid](*(inputs[n].to("neuron") for n in SHAPES))
    return unpack([t.cpu() for t in outs])


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
        perm = qk_perm if name == "q" or name in MM1 else pv_perm
        per_tile = [{"logical": list(x.shape), "kernel_perm": compare(x, d, s, perm(x.shape[1])),
                     "swizzle_perm": compare(x, d, s, swizzle_perm(x.shape[1]))}
                    for x, d, s in tiles(name, inputs, data, scales)]
        result[name] = ({"valid": OPERANDS[name][1], "quadrants": per_tile} if name in MM1
                        else {"valid": OPERANDS[name][1], **per_tile[0]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    torch.save(got, args.out.with_suffix(".pt"))
    print(json.dumps({n: [t["kernel_perm"] for t in result[n]["quadrants"]] if n in MM1
                      else result[n]["kernel_perm"] for n in OPERANDS}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
