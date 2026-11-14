#!/usr/bin/env python3
"""QK^T on ``nc_matmul_mx`` with row tiling: feasibility micro-kernel (MXFP8 attention plan,
``docs/plans/2026-11-13-mxfp8-attention.md``, Task 4 Step 1; ruling BR20 (1)).

At head dim 128 an x4 MX operand of MM1 spans 32 partitions (128 d / 4 packed per byte group), so
one ``nc_matmul_mx`` uses a quarter of the PE rows. Row tiling runs four of them in the four row
quadrants at once. Variants (bf16 q [Sq, 128], k [Sk, 128] in HBM, scores fp32 [Sq, Sk]):

- ``keys``: one 128-query tile (stationary, the same in every quadrant) x four 512-key tiles
  (moving, tile t in quadrant t): the four score tiles of one of the fork's 2048-key large tiles.
- ``queries``: four 128-query tiles (stationary, tile t in quadrant t) x one 512-key tile (moving,
  the same in every quadrant).
- ``bf16``: the fork's MM1 today on the ``keys`` problem: four bf16 ``nc_matmul``
  ``[128 d, 128 q] x [128 d, 512 k]``.
- ``untiled``: control, ``keys``' operand bytes in one full-array ``nc_matmul_mx`` (no tiling,
  contraction over all 128 partitions): the sum over t of the four score tiles, ``[128, 512]``.

ISA constraints applied as the plan's brief states them: ``tile_size=(32, 128)`` (the stationary's
32 partitions set the row tile); stationary, moving and both scales at SBUF partition
``start_row = 32 t``, ``tile_position=(32 t, 0)``; the stationary free size is 128 (> 64); no
``.indirect()``; each score tile ``[128, 512]`` fp32 fills one PSUM bank, and the four sit in
four of the fork's MM1 banks (``check_psum``). Operands reach SBUF in the group-of-4 layout the
fork's ``qk_perm`` describes (partition g byte j = head-dim index 4g + j; the host lays the bf16
out, ``quantize_mx`` quantizes on chip). Each matmul group is issued ``--reps`` times,
accumulating, so the output is ``reps`` x the scores and the TensorE time is matmul-dominated.

``--reload`` alternates two copies of the stationary per rep, so it is loaded again every rep.

Reference: ``reps`` x the fp64 product of the CPU-quantized operands
(``difflet.quant.mx.fake_quantize_mx`` along the head dim, ``qk_perm``); for ``bf16``, of the
bf16 operands. Reported overall and per score tile.

usage: row_tiling.py --mode {sim,birsim,device} --variant VARIANT --out DIR [--reps R] [--reload]

``sim`` is the Python NKI simulator (it accepts some patterns neuronx-cc refuses: not compile
evidence); ``birsim`` compiles with neuronx-cc (LNC=2) and runs BIRSim on the CPU, leaving the NEFF
in ``DIR/<name>_birsim/kernel.neff`` for an engine capture on the device; ``device`` compiles
through torch-neuronx (``wrap_nki``, as ``membership.py``) and runs on the NeuronCore
``NEURON_RT_VISIBLE_CORES`` names (one logical core of an idle device, ruling BR4; pin the CPUs
with ``taskset``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # the checkout, for difflet

import numpy as np  # noqa: E402
import torch  # noqa: E402

D, Q_TILE, K_TILE, QUADRANT = 128, 128, 512, 32
SHAPES = {"keys": (Q_TILE, 4 * K_TILE), "queries": (4 * Q_TILE, K_TILE), "bf16": (Q_TILE, 4 * K_TILE),
          "untiled": (Q_TILE, K_TILE)}
SEED = 0


def check_psum() -> dict:
    """The four concurrent score tiles against the fork's PSUM bank policy
    (``mx_attention_kernel``: MM1 rotates its ``[128, 512]`` fp32 score tiles through
    ``_MM1_PSUM_BANKS_WIDE`` (or banks 0-3), bank ``(large_tile * 4 + k_tile) % len(banks)``, the
    same bank for every query group; MM2 owns bank 4 (accumulating in PSUM, one d-tile); pv_mx's P
    transposes bank 7)."""
    from difflet.backends.neuron.ops_impl import mx_attention_kernel as fork

    def bank(banks, grp, lt, kt):  # the fork's mm1_psum address: the group is not in it
        return banks[(lt * 4 + kt) % len(banks)]

    out = {}
    for name, banks in (("wide", fork._MM1_PSUM_BANKS_WIDE), ("narrow", (0, 1, 2, 3))):
        lts = range(2 * len(banks))
        keys = [{bank(banks, 0, lt, kt) for kt in range(4)} for lt in lts]  # one group, 4 k-tiles
        queries = [{bank(banks, g, lt, kt) for g in range(4)} for lt in lts for kt in range(4)]
        out[name] = {
            "keys_distinct": all(len(b) == 4 for b in keys),
            "keys_clear_of_mm2_and_p": all(not {4, 7} & b for b in keys),
            "queries_distinct": all(len(b) == 4 for b in queries),
            "banks_by_large_tile": [sorted(b) for b in keys],
        }
    return out


def make_inputs():
    """bf16 q [Sq, 128], k [Sk, 128] ~ N(0, 1), seed 0 (keys/bf16 share one problem)."""
    g = torch.Generator().manual_seed(SEED)
    q = torch.randn(4 * Q_TILE, D, generator=g).bfloat16()
    k = torch.randn(4 * K_TILE, D, generator=g).bfloat16()
    return q, k


def operands(variant: str, q: torch.Tensor, k: torch.Tensor):
    """The kernel's two bf16 inputs, laid out as SBUF wants them."""

    def g4(x):  # [rows, 128] -> [32, 4 rows]: partition g, element 4 m + j = x[m, 4 g + j]
        return x.reshape(x.shape[0], QUADRANT, 4).transpose(0, 1).reshape(QUADRANT, -1)

    sq, sk = SHAPES["keys" if variant == "untiled" else variant]
    q, k = q[:sq], k[:sk]
    if variant == "bf16":
        qs, ks = q.T, k.T  # [d, q], [d, k]: the fork's q_sb / k_sb
    elif variant in ("keys", "untiled"):
        qs = torch.cat([g4(q)] * 4)
        ks = torch.cat([g4(k[t * K_TILE:(t + 1) * K_TILE]) for t in range(4)])
    else:
        qs = torch.cat([g4(q[t * Q_TILE:(t + 1) * Q_TILE]) for t in range(4)])
        ks = torch.cat([g4(k)] * 4)
    return qs.contiguous(), ks.contiguous()


def reference(variant: str, q: torch.Tensor, k: torch.Tensor, reps: int, quantized: bool = True
              ) -> torch.Tensor:
    """``reps`` x the variant's scores in fp64, of the MX-quantized operands (unless ``bf16`` or
    not ``quantized``)."""
    from difflet.quant.mx import fake_quantize_mx

    sq, sk = SHAPES["keys" if variant == "untiled" else variant]
    q, k = q[:sq].double(), k[:sk].double()
    if quantized and variant != "bf16":
        perm = torch.arange(D)  # the fork's qk_perm
        q, k = (fake_quantize_mx(x.float(), perm).double() for x in (q, k))
    s = q @ k.T
    if variant == "untiled":  # one contraction over the four quadrants = the four tiles summed
        s = s.reshape(Q_TILE, 4, K_TILE).sum(1)
    return reps * s


def tiles(variant: str, s: torch.Tensor) -> list[torch.Tensor]:
    """The score tiles one matmul group writes (one per quadrant / bf16 matmul)."""
    if variant == "untiled":
        return [s]
    return list(s.split(Q_TILE, 0) if variant == "queries" else s.split(K_TILE, 1))


def build_kernel():
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nkilib.core.utils.kernel_helpers import PSUM_BANK_SIZE

    @nki.jit
    def row_tiled_scores(qs, ks, variant, reps, reload):
        sq, sk = SHAPES[variant]
        out = nl.ndarray((sq, sk), dtype=nl.float32, buffer=nl.shared_hbm)
        # reload: two copies of the stationary, alternated per rep, so the compiler cannot keep
        # it loaded across reps (as in the fork, where MM2 and transposes run between large tiles)
        q_sb = [nl.ndarray(qs.shape, dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1 + reload)]
        k_sb = nl.ndarray(ks.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
        for q_copy in q_sb:
            nisa.dma_copy(dst=q_copy, src=qs)
        nisa.dma_copy(dst=k_sb, src=ks)
        # four score tiles, one PSUM bank each: banks 0-3, the fork's first MM1 large tile
        psum = [nl.ndarray((Q_TILE, K_TILE), dtype=nl.float32, buffer=nl.psum,
                           address=(0, b * PSUM_BANK_SIZE)) for b in range(4)]
        n_tiles = 1 if variant == "untiled" else 4
        if variant == "bf16":
            for r in range(reps):
                for t in range(4):
                    nisa.nc_matmul(psum[t], q_sb[r % len(q_sb)], k_sb[:, nl.ds(t * K_TILE, K_TILE)],
                                   accumulate=r > 0)
        else:
            q_mx, q_sc = [], []
            for q_copy in q_sb:
                q_mx.append(nl.ndarray((128, qs.shape[1] // 4), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf))
                q_sc.append(nl.ndarray((128, qs.shape[1] // 4), dtype=nl.uint8, buffer=nl.sbuf))
                nisa.quantize_mx(dst=q_mx[-1], src=q_copy, dst_scale=q_sc[-1])
            k_mx = nl.ndarray((128, ks.shape[1] // 4), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
            k_sc = nl.ndarray((128, ks.shape[1] // 4), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.quantize_mx(dst=k_mx, src=k_sb, dst_scale=k_sc)
            n_scale = QUADRANT // 8  # a quadrant's scales: its first 4 partitions
            for r in range(reps):
                if variant == "untiled":
                    nisa.nc_matmul_mx(dst=psum[0], stationary=q_mx[r % len(q_mx)], moving=k_mx,
                                      stationary_scale=q_sc[r % len(q_sc)], moving_scale=k_sc,
                                      accumulate=r > 0)
                    continue
                for t in range(4):
                    p = t * QUADRANT
                    nisa.nc_matmul_mx(
                        dst=psum[t],
                        stationary=q_mx[r % len(q_mx)][nl.ds(p, QUADRANT), :],
                        moving=k_mx[nl.ds(p, QUADRANT), :],
                        stationary_scale=q_sc[r % len(q_sc)][nl.ds(p, n_scale), :],
                        moving_scale=k_sc[nl.ds(p, n_scale), :],
                        tile_position=(p, 0),
                        tile_size=(QUADRANT, 128),
                        accumulate=r > 0,
                    )
        o_sb = nl.ndarray((Q_TILE, n_tiles * K_TILE), dtype=nl.float32, buffer=nl.sbuf)
        for t in range(n_tiles):
            nisa.tensor_copy(dst=o_sb[:, nl.ds(t * K_TILE, K_TILE)], src=psum[t])
            if variant == "queries":  # tile t = queries 128 t.., all 512 keys
                nisa.dma_copy(dst=out[nl.ds(t * Q_TILE, Q_TILE), :], src=o_sb[:, nl.ds(t * K_TILE, K_TILE)])
        if variant != "queries":
            nisa.dma_copy(dst=out, src=o_sb)
        return out

    return row_tiled_scores


def run(mode: str, variant: str, reps: int, reload: bool, art: Path) -> np.ndarray:
    import nki

    kernel = build_kernel()
    q, k = make_inputs()
    qs, ks = operands(variant, q, k)
    kw = dict(variant=variant, reps=reps, reload=int(reload))
    if mode == "device":  # torch-neuronx, as membership.py (nki.runtime's nrtpy is not installed)
        from torch_neuronx import wrap_nki

        from difflet.backends import get_backend
        from difflet.backends.neuron.ops_impl.fp8_kernels import logical_cores
        from difflet.pipeline.parallel_config import DiffletParallelConfig

        get_backend("neuron").prepare_runtime(DiffletParallelConfig())
        # LNC=2 as Difflet runs: both cores of the logical core run the same tiles, write the same out
        out = wrap_nki(kernel)[(logical_cores(),)](qs.to("neuron"), ks.to("neuron"), **kw)
        return out.cpu().numpy()
    import ml_dtypes

    qs, ks = (x.float().numpy().astype(ml_dtypes.bfloat16) for x in (qs, ks))
    if mode == "sim":
        return np.asarray(nki.simulate(kernel)(qs, ks, **kw))
    from nki.framework.compiled import StandaloneKernel

    # BIRSim; its kernel.neff (LNC=2, as the device runs) is what an engine capture replays
    standalone = kernel[2]._to_subclass(StandaloneKernel, _enable_simulation=True, artifacts_dir=str(art))
    return np.asarray(standalone(qs, ks, **kw))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mode", required=True, choices=("sim", "birsim", "device"))
    p.add_argument("--variant", required=True, choices=sorted(SHAPES))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--reps", type=int, default=16)
    p.add_argument("--reload", action="store_true", help="a fresh stationary every rep")
    args = p.parse_args(argv)
    name = f"{args.variant}{'_reload' if args.reload else ''}_{args.mode}"
    art = args.out / name
    art.mkdir(parents=True, exist_ok=True)
    fuser = None
    if args.mode == "device":
        from probe import require_idle_core

        fuser = require_idle_core()
    got = torch.from_numpy(run(args.mode, args.variant, args.reps, args.reload, art).astype(np.float64))
    q, k = make_inputs()
    ref = reference(args.variant, q, k, args.reps)
    exact = reference(args.variant, q, k, args.reps, quantized=False)

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    result = {"mode": args.mode, "variant": args.variant, "reps": args.reps, "reload": args.reload,
              "shape": list(got.shape),
              "rel_l2_vs_reference": rel(got, ref), "max_abs_vs_reference": (got - ref).abs().max().item(),
              "rel_l2_per_tile": [rel(a, b) for a, b in zip(tiles(args.variant, got), tiles(args.variant, ref))],
              "rel_l2_vs_bf16_exact": rel(got, exact), "finite": bool(torch.isfinite(got).all()),
              "psum": check_psum(), "fuser": fuser, "artifacts": str(art)}
    (args.out / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("mode", "variant", "reload", "rel_l2_vs_reference",
                                             "rel_l2_per_tile", "rel_l2_vs_bf16_exact")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
