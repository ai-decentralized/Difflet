#!/usr/bin/env python3
"""One attention call on one NeuronCore (LNC=2) at a model's per-rank shape: accuracy and time
(MXFP8 attention plan, ``docs/plans/2026-11-13-mxfp8-attention.md``, Task 2).

usage: probe.py --shape NAME --impl {cte,fork} [--qk-mx] [--pv-mx] --out FILE [--runs N]

``--impl cte`` is Difflet's attention as the models call it today, ``ops_impl.attention.attention``
in the ``tp_q``/``tp_k`` layout: nkilib's ``attention_cte`` in bf16, bounded rows through its
range-select op (q pre-scaled in fp32, lengths padded to 128, HV1 split over heads), scored
against fp32 attention. ``--impl fork`` is the MXFP8 fork of plan Task 3 (``--qk-mx`` /
``--pv-mx`` select its variant); until Task 3 lands it stops with a message (ruling BR6).

Time: the compiled region from bf16 q, k, v (and the int32 bounds) to the bf16 output, measured
as evidence section 4c timed kernels: one warm call, then ``CALLS`` calls and one sync, in each of
``--runs`` worker processes (a fresh runtime each). The JSON has every run's ms per call, their
median and the spread (max - min) / median: the denominators of the plan's gates G1/G2.

Accuracy: the first worker saves the device output (``<out>.out.pt``); the host evaluates the
reference one head at a time (ruling BR8: a head of ``wan_81f`` is ~6 section-sized fp32
temporaries, ~6.4 GB) and reports rel L2 and cosine (float64) per head and overall.

Inputs: q, k, v ~ N(0, 1) in bf16 from seed 0, scale ``d ** -0.5``. A bounded row attends keys
``[0, valid_k)`` from every query, the models' key-padding bounds; ``valid_k`` (shapes.json) is
the image tokens plus half the text plus one, never a multiple of 32. attention_cte masks with
range-select, never skipping a tile, so the bound values do not change its time.

Device rule (ruling BR4): ``NEURON_RT_VISIBLE_CORES`` must name one logical core; before each
worker its device (core // 4) must be idle by ``fuser``, else the probe stops. Pin the CPU set
with ``taskset`` around the whole command.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # the checkout, for difflet

import torch  # noqa: E402

from difflet.quant.mx_attention import mx_attention_reference  # noqa: E402

FORK_MODULE = "difflet.backends.neuron.ops_impl.mx_attention_kernel"
CALLS = 20
SEED = 0


def load_shapes() -> dict:
    return json.loads((HERE / "shapes.json").read_text())


def output_path(out: Path) -> Path:
    return out.with_suffix(".out.pt")


def make_inputs(row: dict):
    """bf16 q, k, v ``[bh, s, d]`` and, for a bounded row, the int32 bounds ``[bh, s, 1]``."""
    g = torch.Generator().manual_seed(SEED)
    q, k, v = (torch.randn(row["bh"], row["s"], row["d"], generator=g).bfloat16() for _ in range(3))
    if row["path"] != "bounded":
        return q, k, v, ()
    hi = torch.full((row["bh"], row["s"], 1), row["valid_k"], dtype=torch.int32)
    return q, k, v, (torch.zeros_like(hi), hi)


def reference(row: dict, q, k, v, bounds, h: int) -> torch.Tensor:
    """fp32 attention of head ``h``, ``[s, d]``; the bounded path takes scale 1.0, so q is scaled
    in fp32 first."""
    scale = row["d"] ** -0.5
    qh, kh, vh = q[h:h + 1].float(), k[h:h + 1].float(), v[h:h + 1].float()
    kw = dict(scale=scale)
    if bounds:
        qh, kw = qh * scale, dict(scale=1.0, bound_min=bounds[0][h:h + 1],
                                  bound_max=bounds[1][h:h + 1])
    return mx_attention_reference(qh, kh, vh, qk_mx=False, pv_mx=False, **kw)[0]


def _metrics(s: torch.Tensor) -> dict:
    return {"rel_l2": (s[0] / s[1]).sqrt().item(), "cosine": (s[2] / (s[1] * s[3]).sqrt()).item()}


def compare(device: torch.Tensor, ref_of_head, bh: int):
    """rel L2 and cosine (float64) of ``device[h]`` against ``ref_of_head(h)``, per head and over
    all heads."""
    total, heads = torch.zeros(4, dtype=torch.float64), []
    for h in range(bh):
        o, r = device[h].double().flatten(), ref_of_head(h).double().flatten()
        s = torch.stack([(o - r).square().sum(), r.square().sum(), o @ r, o.square().sum()])
        heads.append(_metrics(s))
        total += s
    return _metrics(total), heads


def summarize(ms: list[float]) -> dict:
    med = statistics.median(ms)
    return {"median_ms": med, "spread": (max(ms) - min(ms)) / med}


def require_idle_core() -> dict:
    """The one visible logical core's device must be held by no process (``fuser`` exit 1)."""
    cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
    if not cores.isdigit():
        raise SystemExit(f"NEURON_RT_VISIBLE_CORES must name one logical core, got {cores!r}")
    device = int(cores) // 4
    rc = subprocess.run(["fuser", f"/dev/neuron{device}"], capture_output=True).returncode
    if rc != 1:
        raise SystemExit(f"/dev/neuron{device} is not idle (fuser exit {rc}); pick another device")
    return {"device": device, "core": int(cores), "idle": True,
            "at": datetime.datetime.now().isoformat(timespec="seconds")}


def run_worker(argv: list[str], log: Path) -> None:
    with log.open("w") as f:
        subprocess.run([sys.executable, __file__, *argv], stdout=f, stderr=subprocess.STDOUT,
                       check=True)


def worker(args, row: dict) -> None:
    """Device side: compile, save the first output if asked, time CALLS calls with one sync."""
    os.environ.setdefault("DIFFLET_COMPILE_CACHE", "/mnt/nvme/difflet-mxattn/probe/cache")
    # every kernel input is a call argument, so the NKI trace cache is safe (compile.py)
    os.environ.setdefault("DIFFLET_NKI_TRACE_CACHE", "1")
    from difflet.backends import get_backend
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    get_backend("neuron").prepare_runtime(DiffletParallelConfig())
    from difflet.backends.neuron.compile import compile_region
    from difflet.backends.neuron.ops_impl.attention import attention

    scale = row["d"] ** -0.5

    class Region(torch.nn.Module):
        def forward(self, q, k, v, *bounds):
            lo, hi = bounds or (None, None)
            return attention(q, k, v, scale=scale, bound_min=lo, bound_max=hi, tp_q=True,
                             tp_k=True)

    region = compile_region(Region(), backend="neuron")
    q, k, v, bounds = make_inputs(row)
    dev = [t.to("neuron") for t in (q, k, v, *bounds)]
    t0 = time.perf_counter()
    out = region(*dev).cpu()
    first = time.perf_counter() - t0
    if args.save_output:
        torch.save(out, output_path(args.out))
    region(*dev)[0, 0, 0].item()  # warm
    t0 = time.perf_counter()
    for _ in range(CALLS):
        out = region(*dev)
    out[0, 0, 0].item()  # one sync, no host copy of the whole output
    ms = 1000 * (time.perf_counter() - t0) / CALLS
    args.worker.write_text(json.dumps({"ms": ms, "first_call_s": first}))


def main(argv: list[str] | None = None) -> int:
    shapes = load_shapes()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shape", required=True, choices=sorted(shapes))
    p.add_argument("--impl", required=True, choices=("cte", "fork"))
    p.add_argument("--qk-mx", action="store_true")
    p.add_argument("--pv-mx", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--save-output", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if (args.qk_mx or args.pv_mx) and args.impl != "fork":
        p.error("--qk-mx and --pv-mx need --impl fork")
    row = {"name": args.shape, **shapes[args.shape]}
    if args.impl == "fork":
        try:
            importlib.import_module(FORK_MODULE)
        except ModuleNotFoundError as exc:
            if exc.name != FORK_MODULE:
                raise
            raise SystemExit(f"--impl fork needs {FORK_MODULE}, the attention_cte fork of plan "
                             "Task 3, which does not exist yet") from None
        raise SystemExit(f"--impl fork: {FORK_MODULE} exists; its device call and Task 1 mirror "
                         "reference are wired into this probe by plan Task 3")
    if args.worker:
        worker(args, row)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    passed = list(argv if argv is not None else sys.argv[1:])
    runs, fuser = [], []
    for i in range(args.runs):
        fuser.append(require_idle_core())
        run = Path(f"{stem}.run{i}.json")
        run_worker([*passed, "--worker", str(run), *(["--save-output"] if i == 0 else [])],
                   Path(f"{stem}.run{i}.log"))
        runs.append(json.loads(run.read_text()))

    torch.set_num_threads(len(os.sched_getaffinity(0)))
    q, k, v, bounds = make_inputs(row)
    device = torch.load(output_path(args.out)).float()
    overall, heads = compare(device, lambda h: reference(row, q, k, v, bounds, h), row["bh"])
    ms = [r["ms"] for r in runs]
    result = {"shape": row, "impl": args.impl, "qk_mx": args.qk_mx, "pv_mx": args.pv_mx,
              "calls": CALLS, "runs_ms": ms, "first_call_s": [r["first_call_s"] for r in runs],
              **summarize(ms), "reference": "fp32", **overall, "per_head": heads,
              "inputs": f"q, k, v ~ N(0, 1) bf16, seed {SEED}, scale d^-0.5",
              "cpus": sorted(os.sched_getaffinity(0)), "fuser": fuser}
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("impl", "median_ms", "spread", "rel_l2", "cosine")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
