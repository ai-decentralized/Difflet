#!/usr/bin/env python3
"""One attention call on one NeuronCore (LNC=2) at a model's per-rank shape: accuracy and time
(MXFP8 attention plan, ``docs/plans/2026-11-13-mxfp8-attention.md``, Task 2).

usage: probe.py --shape NAME --impl {cte,fork} [--qk-mx] [--pv-mx] --out FILE [--runs N]
                [--calls N] [--peak F]

``--impl cte`` is Difflet's attention as the models call it today, ``ops_impl.attention.attention``
in the ``tp_q``/``tp_k`` layout: nkilib's ``attention_cte`` in bf16, bounded rows through its
range-select op (q pre-scaled in fp32, lengths padded to 128, HV1 split over heads), scored
against fp32 attention. ``--impl fork`` is the same call with the kernel swapped for Difflet's
fork (``ops_impl/mx_attention_kernel.py``, plan Task 3; ``--qk-mx`` / ``--pv-mx`` select its
variant), scored against fp32 attention with both flags off and against the Task 1 mirror
(``mx_attention_reference`` on the bf16 inputs, the kernel's block membership) otherwise.

Time: the compiled region from bf16 q, k, v (and the int32 bounds) to the bf16 output, measured
as evidence section 4c timed kernels: one warm call, then ``--calls`` (default 20) calls and one
sync, in each of ``--runs`` worker processes (a fresh runtime each); 20 vs 40 calls shows whether
the host's dispatch is hidden at the ~1 ms rows (ruling BR13). The JSON has every run's ms per call, their
median and the spread (max - min) / median: the denominators of the plan's gates G1/G2.

Accuracy: the first worker saves the device output (``<out>.out.pt``); the host evaluates the
reference one head at a time (ruling BR8: a head of ``wan_81f`` is ~6 section-sized fp32
temporaries, ~6.4 GB) and reports rel L2 and cosine (float64) per head and overall.

Inputs: q, k, v ~ N(0, 1) in bf16 from seed 0, scale ``d ** -0.5`` (the flat case: scores std ~1);
``--peak F`` multiplies q and k by F (2: scores std ~4, the peaked case of ruling BR11). A bounded row attends keys
``[0, valid_k)`` from every query, the models' key-padding bounds; ``valid_k`` (shapes.json) is
the image tokens plus half the text plus one, never a multiple of 32. attention_cte masks with
range-select, never skipping a tile, so the bound values do not change its time.

Device rule (ruling BR4): ``NEURON_RT_VISIBLE_CORES`` must name one logical core; each worker
checks its device (core // 4) is idle by ``fuser`` just before opening it, else stops. Pin the
CPU set with ``taskset`` around the whole command.
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

import torch  # noqa: E402

from difflet.quant.mx_attention import mx_attention_reference  # noqa: E402

CALLS = 20
SEED = 0


def load_shapes() -> dict:
    return json.loads((HERE / "shapes.json").read_text())


def output_path(out: Path) -> Path:
    return out.with_suffix(".out.pt")


def make_inputs(row: dict, peak: float = 1.0):
    """bf16 q, k, v ``[bh, s, d]`` (q and k times ``peak``) and, for a bounded row, the int32
    bounds ``[bh, s, 1]``."""
    g = torch.Generator().manual_seed(SEED)
    q, k, v = (torch.randn(row["bh"], row["s"], row["d"], generator=g).bfloat16() for _ in range(3))
    q, k = q * peak, k * peak
    if row["path"] != "bounded":
        return q, k, v, ()
    hi = torch.full((row["bh"], row["s"], 1), row["valid_k"], dtype=torch.int32)
    return q, k, v, (torch.zeros_like(hi), hi)


def reference(row: dict, q, k, v, bounds, h: int, *, qk_mx: bool = False,
              pv_mx: bool = False) -> torch.Tensor:
    """Head ``h``'s attention, ``[s, d]``: fp32 with both flags off, else the Task 1 mirror of the
    fork (bf16 inputs as the kernel loads them, its MX block membership). The bounded path takes
    scale 1.0, so q is scaled in fp32 first (and, mirroring, rounded to bf16 as
    ``_flash_attention_bounded`` does)."""
    scale = row["d"] ** -0.5
    mirror = qk_mx or pv_mx
    qh, kh, vh = (t[h:h + 1] if mirror else t[h:h + 1].float() for t in (q, k, v))
    kw = dict(scale=scale)
    if bounds:
        qh = (qh.float() * scale).to(qh.dtype)
        kw = dict(scale=1.0, bound_min=bounds[0][h:h + 1], bound_max=bounds[1][h:h + 1])
    if mirror:
        from difflet.backends.neuron.ops_impl.mx_attention_kernel import pv_perm, qk_perm
        kw.update(qk_perm=qk_perm, pv_perm=pv_perm)
    return mx_attention_reference(qh, kh, vh, qk_mx=qk_mx, pv_mx=pv_mx, **kw)[0]


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


def use_fork(*, qk_mx: bool, pv_mx: bool) -> None:
    """Swap the kernel under ``ops_impl.attention``'s two attention_cte calls for the fork's
    ``mx_attention_fwd``: the layout, bounded pre-scale, padding and head chunking stay theirs."""
    from torch_neuronx import wrap_nki
    from torch_neuronx.nki_hop import nki_op

    from difflet.backends.neuron.ops_impl import attention as attn
    from difflet.backends.neuron.ops_impl.fp8_kernels import logical_cores
    from difflet.backends.neuron.ops_impl.mx_attention_kernel import mx_attention_fwd

    kernel = wrap_nki(mx_attention_fwd)

    @nki_op("difflet_probe::mx_attention_fork", mutates_args={})
    def fork(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bound_min: torch.Tensor | None,
             bound_max: torch.Tensor | None, scale: float) -> torch.Tensor:
        bounds = {} if bound_min is None else {"bound_min": bound_min, "bound_max": bound_max}
        return kernel[(logical_cores(),)](q, k, v, scale=scale, pv_mx=pv_mx, qk_mx=qk_mx, **bounds)

    def unbounded(q, k, v, *, is_causal, scale, training):  # torch-neuronx's SDPA kernel
        assert not is_causal and not training
        return fork(q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), None, None,
                    scale).unflatten(0, q.shape[:2])

    attn._FLASH_KERNEL, attn._FLASH_BOUNDED_OP = unbounded, fork


def worker(args, row: dict) -> None:
    """Device side: compile, save the first output if asked, time CALLS calls with one sync."""
    os.environ.setdefault("DIFFLET_COMPILE_CACHE", "/mnt/nvme/difflet-mxattn/probe/cache")
    # every kernel input is a call argument, so the NKI trace cache is safe for nkilib's kernel
    # (compile.py); the fork compiles uncached unless DIFFLET_NKI_TRACE_CACHE=1 (ruling BR12;
    # the cache does retrace after an edit to the fork's file: task-3a report)
    os.environ.setdefault("DIFFLET_NKI_TRACE_CACHE", "1" if args.impl == "cte" else "0")
    from difflet.backends import get_backend
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    get_backend("neuron").prepare_runtime(DiffletParallelConfig())
    from difflet.backends.neuron.compile import compile_region
    from difflet.backends.neuron.ops_impl.attention import attention

    if args.impl == "fork":
        use_fork(qk_mx=args.qk_mx, pv_mx=args.pv_mx)

    scale = row["d"] ** -0.5

    class Region(torch.nn.Module):
        def forward(self, q, k, v, *bounds):
            lo, hi = bounds or (None, None)
            return attention(q, k, v, scale=scale, bound_min=lo, bound_max=hi, tp_q=True,
                             tp_k=True)

    region = compile_region(Region(), backend="neuron")
    q, k, v, bounds = make_inputs(row, args.peak)
    fuser = require_idle_core()  # the first transfer opens the device
    dev = [t.to("neuron") for t in (q, k, v, *bounds)]
    t0 = time.perf_counter()
    out = region(*dev).cpu()
    first = time.perf_counter() - t0
    if args.save_output:
        torch.save(out, output_path(args.out))
    region(*dev)[0, 0, 0].item()  # warm
    t0 = time.perf_counter()
    for _ in range(args.calls):
        out = region(*dev)
    out[0, 0, 0].item()  # one sync, no host copy of the whole output
    ms = 1000 * (time.perf_counter() - t0) / args.calls
    args.worker.write_text(json.dumps({"ms": ms, "first_call_s": first, "fuser": fuser}))


def main(argv: list[str] | None = None) -> int:
    shapes = load_shapes()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shape", required=True, choices=sorted(shapes))
    p.add_argument("--impl", required=True, choices=("cte", "fork"))
    p.add_argument("--qk-mx", action="store_true")
    p.add_argument("--pv-mx", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--calls", type=int, default=CALLS, help="timed calls per process")
    p.add_argument("--peak", type=float, default=1.0, help="q and k times this (BR11: 2)")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--save-output", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if (args.qk_mx or args.pv_mx) and args.impl != "fork":
        p.error("--qk-mx and --pv-mx need --impl fork")
    row = {"name": args.shape, **shapes[args.shape]}
    if args.worker:
        worker(args, row)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = args.out.with_suffix("")
    passed = list(argv if argv is not None else sys.argv[1:])
    runs = []
    for i in range(args.runs):
        run = Path(f"{stem}.run{i}.json")
        run_worker([*passed, "--worker", str(run), *(["--save-output"] if i == 0 else [])],
                   Path(f"{stem}.run{i}.log"))
        runs.append(json.loads(run.read_text()))

    torch.set_num_threads(len(os.sched_getaffinity(0)))
    q, k, v, bounds = make_inputs(row, args.peak)
    device = torch.load(output_path(args.out)).float()
    flags = dict(qk_mx=args.qk_mx, pv_mx=args.pv_mx)
    overall, heads = compare(device, lambda h: reference(row, q, k, v, bounds, h, **flags),
                             row["bh"])
    ms = [r["ms"] for r in runs]
    result = {"shape": row, "impl": args.impl, **flags,
              "calls": args.calls, "peak": args.peak, "runs_ms": ms, "first_call_s": [r["first_call_s"] for r in runs],
              **summarize(ms), "reference": "mirror" if any(flags.values()) else "fp32",
              **overall, "per_head": heads,
              "inputs": f"q, k, v ~ N(0, 1) bf16, seed {SEED}, q and k x {args.peak}, scale d^-0.5",
              "cpus": sorted(os.sched_getaffinity(0)), "fuser": [r["fuser"] for r in runs]}
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("impl", "median_ms", "spread", "rel_l2", "cosine")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
