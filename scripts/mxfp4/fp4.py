#!/usr/bin/env python3
"""The MXFP4 weight law of the phase-0 estimate: what an fp32 weight dequantizes to as MXFP4
(``float4_e2m1fn`` elements: magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6; one E8M0 scale per 32 values
in the device's swizzled block membership, ``difflet/quant/mx.py``). CPU only. torch has no fp4
arithmetic, so the rounding is written out here and checked against ml_dtypes and nkilib.

    python scripts/mxfp4/fp4.py             # self_check
    python scripts/mxfp4/fp4.py --weights   # and the weight rel L2 of real Qwen-Image / FLUX layers
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from difflet.quant import mx

E2M1_MAX = 6.0
#: e2m1's largest exponent, 6 = 1.5 * 2^2 (e4m3's is 7, ``mx.E4M3_MAX_EXP``).
E2M1_MAX_EXP = 2
LAWS = ("ocp", "noclip", "mse")


def round_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round to nearest, ties to even, onto the e2m1 grid, saturating at +-6."""
    a = v.abs().clamp(max=E2M1_MAX)
    q = torch.where(a < 2.0, torch.round(a * 2) / 2,
                    torch.where(a < 4.0, torch.round(a), torch.round(a / 2) * 2))
    return torch.copysign(q, v)


def _dequant(blocks: torch.Tensor, scale_exp: torch.Tensor) -> torch.Tensor:
    """``blocks`` ``[..., 32]`` through e2m1 with the biased E8M0 exponent ``scale_exp``."""
    scale = torch.exp2((scale_exp - mx.E8M0_BIAS).float()).unsqueeze(-1)
    # nkilib divides by the unclamped scale and stores the byte clamped at 0: the two differ
    # only in a block whose largest value is below 2^-125.
    stored = torch.exp2((scale_exp.clamp(min=0) - mx.E8M0_BIAS).float()).unsqueeze(-1)
    return round_e2m1(blocks / scale) * stored


def fake_quantize_fp4(t: torch.Tensor, law: str = "ocp") -> torch.Tensor:
    """``[R, K]`` -> fp32 ``[R, K]``: what its MXFP4 weight dequantizes to. The block scale's
    exponent, with ``e_max`` the block's largest fp32 exponent: ``ocp`` ``e_max - 2`` (nkilib's
    ``quantize_to_mx``, bit-exact: the block maximum lands in [4, 8) and saturates above 6),
    ``noclip`` one higher (maximum in [2, 4), never saturates), ``mse`` the one of the two with
    the smaller squared error, per block (a weight is quantized offline, so its exponent is
    free)."""
    if law not in LAWS:
        raise ValueError(f"law must be one of {LAWS}, not {law!r}")
    perm = mx.swizzle_perm(t.shape[-1])
    sw = t.float()[..., perm]
    blocks = sw.reshape(*sw.shape[:-1], -1, mx.BLOCK)
    exp = ((blocks.view(torch.int32) >> 23) & 0xFF).amax(dim=-1) - E2M1_MAX_EXP
    if law == "ocp":
        q = _dequant(blocks, exp)
    elif law == "noclip":
        q = _dequant(blocks, exp + 1)
    else:
        a, b = _dequant(blocks, exp), _dequant(blocks, exp + 1)
        # float64: an fp32 error squared can leave fp32, and inf <= inf would always take ocp
        err = [((x - blocks).double() ** 2).sum(-1, keepdim=True) for x in (a, b)]
        q = torch.where(err[0] <= err[1], a, b)
    out = torch.empty_like(sw)
    out[..., perm] = q.reshape(sw.shape)
    return out


def fake_quantize_fp4_sharded(t: torch.Tensor, k_shards: int, law: str = "ocp") -> torch.Tensor:
    """:func:`fake_quantize_fp4` in ``k_shards`` equal K ranges, each with its own block layout:
    a row-parallel weight under TP, as ``mx.fake_quantize_mx_sharded``."""
    k = t.shape[-1]
    if k % k_shards:
        raise ValueError(f"K={k} does not split into {k_shards} shards")
    return torch.cat([fake_quantize_fp4(c, law) for c in t.split(k // k_shards, dim=-1)], dim=-1)


def _nkilib_fp4(w: torch.Tensor) -> torch.Tensor:
    """One K range ``[N, K]`` through nkilib's own offline FP4 quantizer on the kernel layout
    (the swizzled ``[K/4, N*4]`` of ``mx.kernel_layout``), dequantized back to logical order."""
    import nki.language as nl
    from nkilib.core.utils.mx_torch_common import quantize_to_mx, unpack_float4_x4

    n, k = w.shape
    perm = mx.swizzle_perm(k)
    rows = w[:, perm].reshape(n, k // 4, 4).permute(1, 0, 2).reshape(k // 4, n * 4)
    packed, scale = quantize_to_mx(rows.contiguous().numpy(), nl.float4_e2m1fn_x4)
    value = torch.exp2(torch.from_numpy(scale.astype(np.float32)) - mx.E8M0_BIAS)
    deq = unpack_float4_x4(packed) * value.repeat_interleave(8, 0).repeat_interleave(4, 1)
    out = torch.empty(n, k)
    out[:, perm] = deq.reshape(k // 4, n, 4).permute(1, 0, 2).reshape(n, k)
    return out


#: ``(K, k_shards)`` of every default target of Qwen-Image and FLUX at tp4, and the partial tile
#: no default target has. A block lies within one row, so a few rows stand for any N.
CHECK_SHAPES = (
    (3072, 1),    # q/k/v, MLP up, FLUX proj_mlp
    (12288, 4),   # MLP down, FLUX proj_out_mlp: 3072 per rank
    (3072, 4),    # FLUX proj_out_attn: 768 per rank = 512 + a 256 partial tile
    (1408, 1),    # 1024 + a 384 partial tile
)


def self_check() -> None:
    import ml_dtypes

    torch.manual_seed(0)
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    v = torch.cat([torch.linspace(-6, 6, 96001), torch.randn(200000) * 2, ties, -ties])
    v = v.clamp(-6, 6)
    ref = torch.from_numpy(v.numpy().astype(ml_dtypes.float4_e2m1fn).astype(np.float32))
    assert torch.equal(round_e2m1(v), ref), "round_e2m1 differs from ml_dtypes float4_e2m1fn"
    assert sorted(set(round_e2m1(v).abs().tolist())) == [0, .5, 1, 1.5, 2, 3, 4, 6]
    assert torch.equal(round_e2m1(torch.tensor([7.9, -100.0])), torch.tensor([6.0, -6.0]))
    print(f"round_e2m1 == ml_dtypes float4_e2m1fn on {v.numel():,} values")
    # 7.5 scales: ocp saturates to 6, noclip rounds 3.75 to 4; at 2^70 the errors square past fp32
    for big in (torch.full((1, 128), 7.5), torch.full((1, 128), 7.5 * 2.0 ** 70)):
        assert torch.equal(fake_quantize_fp4(big, "mse"), fake_quantize_fp4(big, "noclip"))
    try:
        import nkilib.core.utils.mx_torch_common  # noqa: F401
        nkilib = True
    except ImportError as e:
        nkilib = False
        print(f"SKIPPED the bit equality with nkilib's quantize_to_mx: {e}")
    for k, ks in CHECK_SHAPES:
        w = torch.randn(16, k) * 0.02
        w[0] = 0.0                      # zero blocks
        w[1, 5] = 3.0                   # an outlier: the rest of its block rounds to 0
        w[2] = torch.randn(k) * 1e-38   # block maxima on both sides of 2^-125 (the clamp)
        q = {law: fake_quantize_fp4_sharded(w, ks, law) for law in LAWS}
        assert all(torch.isfinite(x).all() for x in q.values())
        assert torch.equal(fake_quantize_fp4_sharded(q["ocp"][3:], ks, "ocp"), q["ocp"][3:]), \
            "an ocp grid point is not a fixed point of ocp"
        err = {law: float((x - w).norm() / w.norm()) for law, x in q.items()}
        assert err["mse"] <= min(err["ocp"], err["noclip"]) + 1e-7, err
        equal = ""
        if nkilib:
            ref = torch.cat([_nkilib_fp4(c) for c in w.split(k // ks, dim=-1)], dim=-1)
            assert torch.equal(q["ocp"], ref), f"ocp differs from nkilib at K={k} k_shards={ks}"
            equal = "ocp == nkilib quantize_to_mx(float4_e2m1fn_x4), "
        fp8 = float((mx.fake_quantize_mx_sharded(w, ks) - w).norm() / w.norm())
        print(f"[16, {k}] k_shards={ks} ({k // ks} per rank): {equal}rel L2 ocp {err['ocp']:.4f} "
              f"noclip {err['noclip']:.4f} mse {err['mse']:.4f} mxfp8 {fp8:.4f}")
    print("fp4 self-check ok")


#: ``(model, module name)``: a few default targets of each kind, mid-depth.
LAYERS = (
    ("qwen_image", "transformer_blocks.30.attn.to_q"),
    ("qwen_image", "transformer_blocks.30.attn.add_k_proj"),
    ("qwen_image", "transformer_blocks.30.img_mlp.net.0.proj"),
    ("qwen_image", "transformer_blocks.30.img_mlp.net.2"),
    ("qwen_image", "transformer_blocks.30.txt_mlp.net.2"),
    ("flux", "transformer_blocks.9.attn.to_q"),
    ("flux", "transformer_blocks.9.ff.net.2"),
    ("flux", "single_transformer_blocks.19.proj_mlp"),
    ("flux", "single_transformer_blocks.19.attn.to_v"),
)


def _load(root, key: str) -> torch.Tensor:
    from safetensors import safe_open

    for shard in sorted(root.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            if key in f.keys():
                return f.get_tensor(key)
    raise KeyError(f"{key} is in no safetensors file of {root}")


def weights_table(tp: int = 4) -> None:
    """Weight rel L2 of :data:`LAYERS` on the reference snapshots: mxfp8, fp4-ocp, fp4-mse."""
    import importlib

    from difflet.quant.targets import default_spec

    print(f"{'layer':52s} {'[N, K]':>14s} ks  mxfp8    fp4-ocp  fp4-mse   (weight rel L2)")
    for model, name in LAYERS:
        root = importlib.import_module(f"scripts.fp32_ref.{model}").SNAPSHOT / "transformer"
        w = _load(root, f"{name}.weight").float()
        ks = default_spec(model, tp_degree=tp).k_shards(name)
        err = [float((q - w).norm() / w.norm()) for q in (
            mx.fake_quantize_mx_sharded(w, ks), fake_quantize_fp4_sharded(w, ks, "ocp"),
            fake_quantize_fp4_sharded(w, ks, "mse"))]
        print(f"{model + ' ' + name:52s} {str(list(w.shape)):>14s} {ks}   {err[0]:.5f}  "
              f"{err[1]:.5f}  {err[2]:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", action="store_true",
                        help="also print the weight rel L2 table of the real checkpoints")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    self_check()
    if args.weights:
        weights_table()
