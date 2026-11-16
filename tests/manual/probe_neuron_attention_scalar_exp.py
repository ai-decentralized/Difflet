"""attention_cte's softmax exp on ScalarE (Difflet's default) against VectorE, one core, one shape.

usage: probe_neuron_attention_scalar_exp.py --shape NAME --arm {scalar,vector,default}
           --out FILE.json [--peak F] [--runs N] [--calls N] [--expect FILE.out.pt]

The region is ``ops_impl.attention.attention`` as the models call it (``tp_q``/``tp_k``, the
bounded rows' key bounds), compiled with ``compile_region``; ``--arm vector`` passes
``vector_exp=True`` (the kernels before the default changed), ``scalar`` the default. LSE shapes
(``lse_*``, ``--arm default``) run ``_attention_with_lse``, the ring / joint-ring partial, which
has no switch: its VectorE baseline is a copy of this script run in the parent tree.

Method (scripts/mx_attention/probe.py on feat/mxfp8-sage-attention, whose saved outputs
``--expect`` compares bit for bit): q, k, v ~ N(0, 1) in bf16 from seed 0 in that order, q and k
times ``--peak``; bounded rows attend keys ``[0, valid_k)``; scale ``d ** -0.5``. ``--runs``
worker processes (a fresh runtime each); each compiles, the first saves its output, then one warm
call, ``--calls`` calls and one sync. The parent scores the output against fp32 attention per
head (query rows in chunks) and writes the JSON: ms per call per run, median, spread
(max - min) / median, rel L2 and cosine overall and per head, and the ``--expect`` verdict.

Device rule: ``NEURON_RT_VISIBLE_CORES`` names one logical core; the worker checks its device
(core // 4) with ``fuser`` just before the first transfer. Pin CPUs with ``taskset``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The parent only spawns workers and scores on the CPU: autoloaded torch-neuronx would hold
# /dev/neuron0 and make device 0 read busy to every other session's fuser.
_PARENT = "--worker" not in sys.argv and "TORCH_DEVICE_BACKEND_AUTOLOAD" not in os.environ
if _PARENT:
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch  # noqa: E402

SEED = 0
SHAPES = {  # section 9's shapes.json, plus the logsumexp partials of the joint ring
    "flux_1024": dict(bh=6, s=4608, d=128),
    "qwen_1024": dict(bh=6, s=5120, d=128, valid_k=4609),
    "ltx2_video": dict(bh=8, s=6144, d=128),
    "wan_81f": dict(bh=10, s=32760, d=128),
    "hv15_121f": dict(bh=8, s=51275, d=128, valid_k=50283),
    "hv1_L": dict(bh=6, s=67576, d=128, valid_k=67449),
    # Qwen-Image tp2 cp2 image hop: 12 heads, 1024 text + 2048 image queries, 2048 image keys.
    "lse_qwen_cp2": dict(bh=12, s=3072, s_k=2048, d=128, lse=True),
    # A long logsumexp partial at a Wan 81f cp2 hop's size (10 heads, 16380 x 16380). Compiled
    # Wan cp2 itself takes the NKI ring kernel here (16380 <= _RING_KERNEL_MAX_SEQ); the op runs
    # at this size for the permute-loop ring (DIFFLET_NEURON_RING_KERNEL=0) and longer shards.
    "lse_wan_cp2": dict(bh=10, s=16380, s_k=16380, d=128, lse=True),
}


def make_inputs(row: dict, peak: float):
    g = torch.Generator().manual_seed(SEED)
    s_k = row.get("s_k", row["s"])
    shapes = ((row["bh"], row["s"], row["d"]), (row["bh"], s_k, row["d"]), (row["bh"], s_k, row["d"]))
    q, k, v = (torch.randn(*shp, generator=g).bfloat16() for shp in shapes)
    q, k = q * peak, k * peak
    if "valid_k" not in row:
        return q, k, v, ()
    hi = torch.full((row["bh"], row["s"], 1), row["valid_k"], dtype=torch.int32)
    return q, k, v, (torch.zeros_like(hi), hi)


def require_idle_core() -> int:
    cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    if not cores.isdigit():
        raise SystemExit(f"NEURON_RT_VISIBLE_CORES must name one logical core, got {cores!r}")
    device = int(cores) // 4
    done = subprocess.run(["fuser", f"/dev/neuron{device}"], capture_output=True)
    said = (done.stdout + done.stderr).decode(errors="replace").strip()
    if done.returncode != 1 or said:
        raise SystemExit(f"/dev/neuron{device} is not idle ({said or done.returncode}); pick another")
    return device


def worker(args, row: dict) -> None:
    os.environ.setdefault("DIFFLET_COMPILE_CACHE", "/mnt/nvme/difflet-vexp/kernel/cache")
    os.environ.setdefault("DIFFLET_NKI_TRACE_CACHE", "1")  # every kernel input is an argument
    from difflet.backends import get_backend
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    get_backend("neuron").prepare_runtime(DiffletParallelConfig())
    from difflet.backends.neuron.compile import compile_region
    from difflet.backends.neuron.ops_impl import attention as attn

    scale, vector_exp = row["d"] ** -0.5, args.arm == "vector"

    class Region(torch.nn.Module):
        def forward(self, q, k, v, *bounds):
            if row.get("lse"):  # [B*H, S, d] -> the op's [B, H, S, d]
                out, lse = attn._attention_with_lse(q[None], k[None], v[None], scale=scale)
                return out[0], lse[0]
            lo, hi = bounds or (None, None)
            return attn.attention(q, k, v, scale=scale, bound_min=lo, bound_max=hi, tp_q=True,
                                  tp_k=True, vector_exp=vector_exp)

    region = compile_region(Region(), backend="neuron")
    q, k, v, bounds = make_inputs(row, args.peak)
    device = require_idle_core()
    dev = [t.to("neuron") for t in (q, k, v, *bounds)]
    out = region(*dev)
    out = tuple(t.cpu() for t in out) if isinstance(out, tuple) else out.cpu()
    if args.save_output:
        torch.save(out, args.out.with_suffix(".out.pt"))
    first = region(*dev)
    (first[0] if isinstance(first, tuple) else first)[0, 0, 0].item()  # warm
    t0 = time.perf_counter()
    for _ in range(args.calls):
        last = region(*dev)
    (last[0] if isinstance(last, tuple) else last)[0, 0, 0].item()  # one sync
    ms = 1000 * (time.perf_counter() - t0) / args.calls
    print(json.dumps({"ms": ms, "device": device}), flush=True)


def reference_head(row, q, k, v, bounds, h, rows=4096):
    """fp32 attention of head ``h`` (and its logsumexp), query rows in chunks."""
    scale = row["d"] ** -0.5
    qh, kh, vh = (t[h].float() for t in (q, k, v))
    if bounds:  # the bounded path pre-scales q in fp32 and rounds it to bf16
        qh, scale = (qh * scale).bfloat16().float(), 1.0
        kh, vh = kh[: row["valid_k"]], vh[: row["valid_k"]]
    outs, lses = [], []
    for i in range(0, qh.shape[0], rows):
        s = qh[i:i + rows] @ kh.T * scale
        lses.append(torch.logsumexp(s, -1))
        outs.append(torch.softmax(s, -1) @ vh)
    return torch.cat(outs), torch.cat(lses)


def score(row, out, q, k, v, bounds) -> dict:
    dev_out, dev_lse = out if isinstance(out, tuple) else (out, None)
    total, heads, lse_err = torch.zeros(4, dtype=torch.float64), [], 0.0
    for h in range(row["bh"]):
        r, r_lse = reference_head(row, q, k, v, bounds, h)
        o, r = dev_out[h].double().flatten(), r.double().flatten()
        s = torch.stack([(o - r).square().sum(), r.square().sum(), o @ r, o.square().sum()])
        heads.append({"rel_l2": (s[0] / s[1]).sqrt().item(), "cosine": (s[2] / (s[1] * s[3]).sqrt()).item()})
        total += s
        if dev_lse is not None:
            lse_err = max(lse_err, (dev_lse[h].double() - r_lse.double()).abs().max().item())
    overall = {"rel_l2": (total[0] / total[1]).sqrt().item(),
               "cosine": (total[2] / (total[1] * total[3]).sqrt()).item(),
               "nonfinite": int((~torch.isfinite(dev_out)).sum())}
    if dev_lse is not None:
        overall["lse_max_abs_err"] = lse_err
    return {"overall": overall, "per_head": heads}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", required=True, choices=sorted(SHAPES))
    p.add_argument("--arm", required=True, choices=("scalar", "vector", "default"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--peak", type=float, default=1.0)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--calls", type=int, default=20)
    p.add_argument("--expect", type=Path, default=None, help="a saved .out.pt to compare bit for bit")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--save-output", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    row = SHAPES[args.shape]
    if bool(row.get("lse")) != (args.arm == "default"):
        p.error("lse_* shapes take --arm default (the tree's own logsumexp op, no switch: run a copy "
                "of this script in the tree to compare); the others take scalar or vector")
    if args.worker:
        return worker(args, row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not (_PARENT and k == "TORCH_DEVICE_BACKEND_AUTOLOAD")}
    base = [sys.executable, __file__, "--shape", args.shape, "--arm", args.arm, "--out", str(args.out),
            "--peak", str(args.peak), "--calls", str(args.calls), "--worker"]
    runs = []
    for i in range(args.runs):
        log = args.out.with_suffix(f".run{i}.log")
        with log.open("w") as f:
            subprocess.run(base + (["--save-output"] if i == 0 else []), stdout=f,
                           stderr=subprocess.STDOUT, env=env, check=True)
        runs.append(json.loads(log.read_text().strip().splitlines()[-1]))
    ms = [r["ms"] for r in runs]
    med = statistics.median(ms)
    out = torch.load(args.out.with_suffix(".out.pt"))
    q, k, v, bounds = make_inputs(row, args.peak)
    first = out[0] if isinstance(out, tuple) else out
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or None
    result = {"shape": args.shape, "row": row, "arm": args.arm, "peak": args.peak, "calls": args.calls,
              "tree": str(ROOT), "head": head,
              "device": runs[0]["device"], "cpus": sorted(os.sched_getaffinity(0)),
              "ms": ms, "median_ms": med, "spread": (max(ms) - min(ms)) / med,
              "out_md5": hashlib.md5(first.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
              "accuracy": score(row, out, q, k, v, bounds)}
    if args.expect is not None:
        ref = torch.load(args.expect)
        result["expect"] = {"file": str(args.expect),
                            "bit_identical": bool(first.shape == ref.shape and torch.equal(first, ref))}
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("shape", "arm", "median_ms", "spread")}
                     | {"rel_l2": result["accuracy"]["overall"]["rel_l2"],
                        "bit_identical": result.get("expect", {}).get("bit_identical")}))


if __name__ == "__main__":
    main()
