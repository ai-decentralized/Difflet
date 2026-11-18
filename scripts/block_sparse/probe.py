#!/usr/bin/env python3
"""E2 (``docs/plans/2026-11-18-attn-constmax-sparse-experiments.md``): one call of the attention
fork (``ops_impl/mx_attention_kernel.py``) with a static K/V-tile mask at ``wan_81f`` (10 heads x
32,760 tokens x 128, unbounded) on one logical core: time and accuracy.

usage: probe.py --mask {none,full,temporal:W,sta:WT,WH} [--vector-exp] --out FILE [--runs N]
                [--calls N] [--score-only]

``--mask``: ``none`` passes no ``kv_tiles`` (the fork as feat/mxfp8-sage-attention timed it),
``full`` every tile, ``temporal:W`` key frames within +-W, ``sta:WT,WH`` +-WT frames x +-WH rows
(``attn_tile_masks.window_tiles``). ``--vector-exp``: the exp on VectorE (shipped attention_cte's
``allow_vector_exp=True``); default ScalarE.

Time: the method of scripts/mx_attention/probe.py on feat/mxfp8-sage-attention (evidence
2026-11-13 section 4c): a compiled region from bf16 q, k, v to the bf16 output, one warm call, then
``--calls`` calls and one sync, in each of ``--runs`` worker processes; median and (max - min) /
median. The region holds the kernel's torch op alone (no ``ops_impl.attention`` around it: the
unbounded path only reshapes there).

Accuracy (the first worker's output): rel L2 and cosine (float64) against dense fp32 attention
(the approximation the mask makes) and against fp32 attention with the skipped tiles at -inf (the
kernel's own error), per head and overall; references cached per head under ``--ref-dir``.
Inputs q, k, v ~ N(0, 1) bf16 from seed 0 (the fork probe's), scale 128^-0.5.

Device rule: ``NEURON_RT_VISIBLE_CORES`` names one logical core; each worker checks its device
(core // 4) is idle by ``fuser`` before opening it. Pin CPUs with ``taskset`` around the command.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # the checkout, for difflet

# The parent only spawns workers and scores on the CPU: autoloaded torch-neuronx would hold a device.
if "--worker" not in sys.argv:
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch  # noqa: E402

from difflet.backends.neuron.ops_impl.attn_tile_masks import (  # noqa: E402
    density,
    masked_attention_reference,
    window_tiles,
)

BH, S, D, SEED = 10, 32760, 128, 0
WAN_81F = (21, 30, 52)  # latent frames, rows, cols
SCALE = D ** -0.5
NUM_K_TILES = -(-S // 512)


def mask_tiles(spec: str):
    """``kv_tiles`` for a ``--mask`` spec (None for ``none``)."""
    kind, _, arg = spec.partition(":")
    if kind == "none":
        return None
    if kind == "full":
        return (tuple(range(NUM_K_TILES)),) * -(-S // 128)
    if kind == "temporal":
        return window_tiles(*WAN_81F, wt=int(arg))
    if kind == "sta":
        wt, wh = map(int, arg.split(","))
        return window_tiles(*WAN_81F, wt=wt, wh=wh)
    raise SystemExit(f"unknown --mask {spec!r}")


def make_inputs():
    g = torch.Generator().manual_seed(SEED)
    return tuple(torch.randn(BH, S, D, generator=g).bfloat16() for _ in range(3))


def require_idle_core() -> dict:
    cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    if not cores.isdigit():
        raise SystemExit(f"NEURON_RT_VISIBLE_CORES must name one logical core, got {cores!r}")
    device = int(cores) // 4
    done = subprocess.run(["fuser", f"/dev/neuron{device}"], capture_output=True)
    said = (done.stdout + done.stderr).decode(errors="replace").strip()
    if done.returncode != 1 or said:
        raise SystemExit(f"/dev/neuron{device} is not idle (fuser exit {done.returncode} {said})")
    return {"device": device, "core": int(cores), "at": datetime.datetime.now().isoformat(timespec="seconds")}


def worker(args) -> None:
    os.environ.setdefault("DIFFLET_COMPILE_CACHE", "/mnt/nvme/difflet-exp/sparse/cache")
    # kv_tiles is a static argument the trace-cache key holds by value (nki CompileKernel
    # _hashable_arg), and the key's dep check retraces after an edit to the kernel file
    os.environ.setdefault("DIFFLET_NKI_TRACE_CACHE", "1")
    from difflet.backends import get_backend
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    get_backend("neuron").prepare_runtime(DiffletParallelConfig())
    from torch_neuronx import wrap_nki
    from torch_neuronx.nki_hop import nki_op

    from difflet.backends.neuron.compile import compile_region
    from difflet.backends.neuron.ops_impl.fp8_kernels import logical_cores
    from difflet.backends.neuron.ops_impl.mx_attention_kernel import mx_attention_fwd

    kernel = wrap_nki(mx_attention_fwd)
    kw = dict(scale=SCALE, pv_mx=False, qk_mx=False, allow_vector_exp=args.vector_exp)
    tiles = mask_tiles(args.mask)
    if tiles is not None:
        kw["kv_tiles"] = tiles

    @nki_op("difflet_probe::block_sparse_attention", mutates_args={})
    def op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return kernel[(logical_cores(),)](q, k, v, **kw)

    class Region(torch.nn.Module):
        def forward(self, q, k, v):
            return op(q, k, v)

    region = compile_region(Region(), backend="neuron")
    fuser = require_idle_core()
    dev = [t.to("neuron") for t in make_inputs()]
    t0 = time.perf_counter()
    out = region(*dev).cpu()
    first = time.perf_counter() - t0
    if args.save_output:
        torch.save(out, args.out.with_suffix(".out.pt"))
    region(*dev)[0, 0, 0].item()  # warm
    t0 = time.perf_counter()
    for _ in range(args.calls):
        out = region(*dev)
    out[0, 0, 0].item()  # one sync
    ms = 1000 * (time.perf_counter() - t0) / args.calls
    args.worker.write_text(json.dumps({"ms": ms, "first_call_s": first, "fuser": fuser}))


def reference(ref_dir: Path, name: str, tiles, q, k, v, h: int) -> torch.Tensor:
    """Head ``h``'s fp32 reference (``name``: ``dense`` or the mask spec), cached."""
    path = ref_dir / f"{name.replace(':', '_').replace(',', '_')}_h{h}.pt"
    if path.exists():
        return torch.load(path)
    ref = masked_attention_reference(q[h:h + 1], k[h:h + 1], v[h:h + 1], scale=SCALE,
                                     kv_tiles=tiles)[0]
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    torch.save(ref, tmp)
    tmp.rename(path)
    return ref


def compare(out: torch.Tensor, ref_of_head) -> dict:
    total, heads = torch.zeros(4, dtype=torch.float64), []

    def metrics(s):
        return {"rel_l2": (s[0] / s[1]).sqrt().item(), "cosine": (s[2] / (s[1] * s[3]).sqrt()).item()}

    for h in range(BH):
        o, r = out[h].double().flatten(), ref_of_head(h).double().flatten()
        s = torch.stack([(o - r).square().sum(), r.square().sum(), o @ r, o.square().sum()])
        heads.append(metrics(s)["rel_l2"])
        total += s
    return {**metrics(total), "per_head_rel_l2": heads}


def checkout() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(HERE.parents[1]), *a], capture_output=True, text=True).stdout.strip()
    return {"path": str(HERE.parents[1]), "head": git("rev-parse", "--short", "HEAD"),
            "kernel_dirty": bool(git("status", "--porcelain", "--", "difflet/backends/neuron/ops_impl/"))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mask", required=True)
    p.add_argument("--vector-exp", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--ref-dir", type=Path, default=Path("/mnt/nvme/difflet-exp/sparse/ref"))
    p.add_argument("--score-only", action="store_true", help="rescore the saved runs and output")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--save-output", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    tiles = mask_tiles(args.mask)
    if args.worker:
        worker(args)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    passed = [a for a in (argv if argv is not None else sys.argv[1:]) if a != "--score-only"]
    runs = []
    for i in range(args.runs):
        run = Path(f"{stem}.run{i}.json")
        if not args.score_only:
            env = {k: v for k, v in os.environ.items() if k != "TORCH_DEVICE_BACKEND_AUTOLOAD"}
            with Path(f"{stem}.run{i}.log").open("w") as f:
                subprocess.run([sys.executable, __file__, *passed, "--worker", str(run),
                                *(["--save-output"] if i == 0 else [])],
                               stdout=f, stderr=subprocess.STDOUT, check=True, env=env)
        runs.append(json.loads(run.read_text()))

    torch.set_num_threads(len(os.sched_getaffinity(0)))
    args.ref_dir.mkdir(parents=True, exist_ok=True)
    q, k, v = make_inputs()
    out = torch.load(args.out.with_suffix(".out.pt")).float()
    dense = compare(out, lambda h: reference(args.ref_dir, "dense", None, q, k, v, h))
    masked = dense if tiles is None or args.mask == "full" else compare(
        out, lambda h: reference(args.ref_dir, args.mask, tiles, q, k, v, h))
    ms = [r["ms"] for r in runs]
    med = statistics.median(ms)
    result = {"mask": args.mask, "density": 1.0 if tiles is None else density(tiles, NUM_K_TILES),
              "vector_exp": args.vector_exp, "calls": args.calls, "runs_ms": ms, "median_ms": med,
              "spread": (max(ms) - min(ms)) / med, "first_call_s": [r["first_call_s"] for r in runs],
              "vs_dense_fp32": dense, "vs_masked_fp32": masked,
              "nonfinite": int((~out.isfinite()).sum()),
              "inputs": f"q, k, v ~ N(0, 1) bf16 [{BH}, {S}, {D}], seed {SEED}, scale d^-0.5",
              "cpus": sorted(os.sched_getaffinity(0)), "fuser": [r["fuser"] for r in runs],
              "tree": checkout()}
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("mask", "vector_exp", "density", "median_ms", "spread")}
                     | {"rel_dense": dense["rel_l2"], "rel_masked": masked["rel_l2"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
