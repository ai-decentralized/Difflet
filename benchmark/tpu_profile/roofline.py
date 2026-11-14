"""How far a traced TPU denoise is from its roofline, and where the gap is.

    python -m benchmark.tpu_profile.roofline <trace_dir> --steps 28 [--top 25]

Reads rank 0's ``jax.profiler`` trace (``DIFFLET_BENCH_TRACE_DIR``, as
trace_diff / trace_breakdown) and, for every device op, takes the compiler's own
cost numbers from the trace -- ``model_flops`` and ``bytes_accessed`` -- and its
roofline time ``max(flops / peak, bytes / HBM bandwidth)``. The op's *gap* is its
measured time minus that bound. Ops are put in buckets that point at different
fixes:

  matmul       XLA convolution fusions (the MXU work)
  attention    the flash-attention custom call
  cm-kernel    our Pallas collective-matmul kernels (AG / RS rings)
  comm         collectives and the exposed part of async ones (``*-done``)
  relayout     copy / transpose / reshape / bitcast and their async forms
  elementwise  every other fusion or op (norms, modulation, activations, adds)
  idle         the window minus the union of op intervals (host gaps)

Custom calls carry no ``model_flops``; their FLOPs are estimated from the
operand and output shapes (attention: 4 * B * H * Sq * Sk * D; a custom call with 2-D [N, K] weights
matching its first operand's last dim is taken as one of our collective-matmul
kernels: 2 * rows * K * N, from whichever of its input or output side covers
all rows). ``bytes_accessed`` counts every operand, also ones XLA keeps in
VMEM (``S(1)``), so an op's bound is clipped to its measured time. Comm and
relayout ops get a bound of 0: their ideal is to be hidden or removed.

Prints, per denoise step: window / busy / idle, the FLOPs and the step's compute
lower bound, then the per-bucket table (ms, achieved TFLOP/s, roofline ms, gap
ms), and the ops with the largest gap keyed by opcode, output shape and source
line, which is where to look first.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import os
import re

PEAK_TFLOPS = 197.0  # v5e, bf16, per chip
HBM_GBPS = 819.0  # v5e, per chip

#: "%name = <shape> <opcode>(<operands>)..." -- an HLO instruction line.
HLO_LINE = re.compile(r"^%?[\w.-]+ = (.*?)\s([a-z][a-z0-9-]*)\((.*)$", re.DOTALL)
ARRAY = re.compile(r"\b(?:bf16|f32|f16|s32|u32|s8|u8|f8e4m3fn|f8e5m2|pred)\[([\d,]*)\]")
COMM = ("all-reduce", "all-gather", "reduce-scatter", "collective-permute", "all-to-all",
        "send", "recv")
RELAYOUT = ("copy", "transpose", "reshape", "bitcast", "slice", "dynamic-slice",
            "dynamic-update-slice", "concatenate", "pad")


def _shapes(text: str) -> list[list[int]]:
    return [[int(d) for d in m.group(1).split(",") if d] for m in ARRAY.finditer(text)]


def _load(trace_dir: str) -> tuple[list[dict], list[dict]]:
    """(ops, modules) of /device:TPU:0, each op {name, ts, dur (us), args}."""
    path = sorted(glob.glob(trace_dir + "/plugins/profile/*/*.trace.json.gz"))[-1]
    events = json.load(gzip.open(path))["traceEvents"]
    procs = {e["pid"]: e["args"]["name"] for e in events
             if e.get("ph") == "M" and e.get("name") == "process_name"}
    threads = {(e["pid"], e["tid"]): e["args"]["name"] for e in events
               if e.get("ph") == "M" and e.get("name") == "thread_name"}
    device = [e for e in events if e.get("ph") == "X" and procs.get(e["pid"]) == "/device:TPU:0"]
    ops = [e for e in device if threads.get((e["pid"], e["tid"])) == "XLA Ops"]
    modules = [e for e in device if threads.get((e["pid"], e["tid"])) == "XLA Modules"]
    if modules:
        return ops, modules
    # The JSON export stops at 5M events; read the raw xplane instead.
    import jax.profiler

    path = sorted(glob.glob(trace_dir + "/plugins/profile/*/*.xplane.pb"))[-1]
    plane = next(p for p in jax.profiler.ProfileData.from_file(path).planes
                 if p.name == "/device:TPU:0")
    lines = {line.name: line for line in plane.lines}

    def events_of(name):
        out = []
        for e in (lines[name].events if name in lines else ()):
            args = {k: v for k, v in getattr(e, "stats", ())}
            args.setdefault("long_name", e.name)
            out.append({"name": e.name, "ts": e.start_ns / 1e3, "dur": e.duration_ns / 1e3,
                        "args": args})
        return out

    return events_of("XLA Ops"), events_of("XLA Modules")


def _json_has_device_events(trace_dir: str) -> bool:
    paths = sorted(glob.glob(trace_dir + "/plugins/profile/*/*.trace.json.gz"))
    if not paths:
        return False
    events = json.load(gzip.open(paths[-1]))["traceEvents"]
    devices = {e["pid"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
               and e["args"]["name"] == "/device:TPU:0"}
    return any(e.get("ph") == "X" and e.get("pid") in devices for e in events)


def _union(intervals) -> float:
    total, start, end = 0.0, None, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            if end is not None:
                total += end - start
            start, end = s, e
        else:
            end = max(end, e)
    return total + ((end - start) if end is not None else 0.0)


def _matmul_kernel_flops(output: list[list[int]], operands: list[list[int]]) -> float:
    """2 * rows * K * N of a custom call shaped like x @ w.T over 2-D [N, K]
    weights (our collective-matmul kernels); 0 when it is not shaped so. Rows are
    whichever side covers them all: the AG kernel's output, the RS kernel's input."""
    # x is the first operand of rank >= 3 that 2-D weights match (scalar-prefetch
    # tables such as s32[2,4] come first in our kernels).
    for i, x in enumerate(operands):
        if len(x) < 3:
            continue
        k = x[-1]
        weights = [s for s in operands[i + 1:] if len(s) == 2 and s[1] == k]
        if weights:
            break
    else:
        return 0.0
    rows_in = math.prod(x[:-1])
    # Rows an output covers = its elements / its weight's width (an output written
    # heads first, (B, H, ..., d), has H in its leading dims).
    rows_out = max((math.prod(s) // w[0] for s, w in zip(output, weights) if s and w[0]),
                   default=0)
    return 2.0 * max(rows_in, rows_out) * k * sum(w[0] for w in weights)


def _attention_flops(operands: list[list[int]]) -> float:
    """4 * B * H * Sq * Sk * D from [B, H, S, D] q and k."""
    if len(operands) >= 3 and all(len(s) == 4 for s in operands[:2]):
        q, k = operands[0], operands[1]
        return 4.0 * q[0] * q[1] * q[2] * k[2] * q[3]
    return 0.0


def _bucket(opcode: str, category: str, name: str, output=(), operands=()) -> str:
    low = name.lower()
    if opcode == "custom-call":
        target = re.search(r'custom_call_target="([^"]*)"', name)
        target = target.group(1).lower() if target else ""
        if "bitcast" in target:  # ConcatBitcast and friends: XLA's own data movement
            return "relayout"
        if "attention" in low or "flash" in low:
            return "attention"
        if target == "tpu_custom_call" and _matmul_kernel_flops(list(output), list(operands)):
            return "cm-kernel"
        return "custom-other"
    if "convolution" in category or opcode in ("convolution", "dot"):
        return "matmul"
    if opcode.startswith(COMM) or any(c in category for c in COMM):
        return "comm"
    if opcode.startswith(RELAYOUT) or category in ("data formatting",):
        return "relayout"
    return "elementwise"


def operand_text(rest: str) -> str:
    """The operand list of an HLO line: ``rest`` up to the ')' closing ``opcode(``."""
    depth = 0
    for i, ch in enumerate(rest):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                return rest[:i]
            depth -= 1
    return rest


def _source(args: dict) -> str:
    src = args.get("source") or ""
    if not src:
        return (args.get("tf_op") or "")[-60:]
    parts = src.split("/")
    return "/".join(parts[-2:])


def _from_hlo_stats(trace_dir: str):
    """(ops, modules, busy) from ``hlo_stats.json`` (xprof_ops.py): one aggregated
    pseudo-op per HLO op -- its total self time, occurrences, FLOPs and measured
    HBM bytes -- for traces whose JSON export has no device events. The window
    comes from the xplane's module spans; busy is the summed self time."""
    import jax.profiler

    stats = json.load(open(trace_dir.rstrip("/") + "/hlo_stats.json"))["rows"]
    path = sorted(glob.glob(trace_dir + "/plugins/profile/*/*.xplane.pb"))[-1]
    plane = next(p for p in jax.profiler.ProfileData.from_file(path).planes
                 if p.name == "/device:TPU:0")
    line = next((ln for ln in plane.lines if ln.name == "XLA Modules"), None)
    modules = [{"ts": e.start_ns / 1e3, "dur": e.duration_ns / 1e3}
               for e in (line.events if line else ())]
    ops = [{"name": r["name"], "dur": r["self_us"], "n": r["occurrences"],
            "args": {"long_name": r["expression"], "hlo_category": r["category"],
                     "model_flops": r["flops"], "bytes_accessed": r["hbm_bytes"],
                     "source": r["source"]}}
           for r in stats]
    return ops, modules, sum(r["self_us"] for r in stats)


def analyze(trace_dir: str, peak_tflops: float = PEAK_TFLOPS, hbm_gbps: float = HBM_GBPS) -> dict:
    stats_path = trace_dir.rstrip("/") + "/hlo_stats.json"
    if os.path.exists(stats_path) and not _json_has_device_events(trace_dir):
        ops, modules, busy = _from_hlo_stats(trace_dir)
    else:
        ops, modules = _load(trace_dir)
        busy = _union([(e["ts"], e["ts"] + e["dur"]) for e in ops])
    window = max(m["ts"] + m["dur"] for m in modules) - min(m["ts"] for m in modules)
    rows = []
    for e in ops:
        args = e.get("args") or {}
        long_name = args.get("long_name") or e["name"]
        match = HLO_LINE.match(long_name)
        if match:
            out_text, opcode, rest = match.groups()
        else:
            out_text, opcode, rest = "", e["name"].split(".")[0], ""
        category = args.get("hlo_category") or ""
        name = e["name"] + " " + (long_name if opcode == "custom-call" else "")
        output, operands = _shapes(out_text), _shapes(operand_text(rest))
        bucket = _bucket(opcode, category, name, output, operands)
        flops = float(args.get("model_flops") or 0)
        estimated = False
        if not flops and bucket == "attention":
            flops, estimated = _attention_flops(operands), True
        elif not flops and bucket == "cm-kernel":
            flops, estimated = _matmul_kernel_flops(output, operands), True
        nbytes = float(args.get("bytes_accessed") or 0)
        compute_us = flops / (peak_tflops * 1e12) * 1e6
        memory_us = nbytes / (hbm_gbps * 1e9) * 1e6
        if bucket in ("comm", "relayout"):
            # Avoidable or hideable: the ideal is no exposed time at all.
            compute_us = memory_us = 0.0
        rows.append({
            "bucket": bucket, "opcode": opcode,
            "shape": re.sub(r"\{[^}]*\}", "", out_text)[:70], "source": _source(args),
            "dur": e["dur"], "n": e.get("n", 1),
            "flops": flops * e.get("n", 1) if estimated else flops,
            "estimated": estimated, "bytes": nbytes,
            # An op faster than its "bound" had operands in VMEM (bytes_accessed
            # counts them too): clip, so the bound never exceeds the measured time.
            "bound": min(max(compute_us * (e.get("n", 1) if estimated else 1), memory_us),
                         e["dur"]),
            "limit": ("-" if not (compute_us or memory_us)
                      else "compute" if compute_us >= memory_us else "memory"),
        })
    return {"window": window, "busy": busy, "rows": rows, "modules": len(modules)}


def report(result: dict, steps: int, top: int, peak_tflops: float) -> None:
    per = 1e3 * max(steps, 1)  # us -> ms per step
    rows, window, busy = result["rows"], result["window"], result["busy"]
    idle = window - busy
    flops = sum(r["flops"] for r in rows)
    estimated = sum(r["flops"] for r in rows if r["estimated"])
    bound = sum(r["bound"] for r in rows)
    lower = flops / (peak_tflops * 1e12) * 1e3 / max(steps, 1)
    print(f"steps {steps}, modules {result['modules']}, ops {len(rows)}")
    print(f"per step: window {window / per:.1f} ms, busy {busy / per:.1f} ms, "
          f"idle {idle / per:.1f} ms ({100 * idle / window:.1f}%)")
    print(f"per step: {flops / max(steps, 1) / 1e12:.1f} TFLOP "
          f"({100 * estimated / max(flops, 1):.0f}% estimated from custom-call shapes); "
          f"compute lower bound {lower:.1f} ms -> MFU {100 * lower / (window / per):.1f}% "
          f"of the window; sum of per-op rooflines {bound / per:.1f} ms")
    buckets = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        b = buckets[r["bucket"]]
        b["dur"] += r["dur"]; b["flops"] += r["flops"]; b["bound"] += r["bound"]; b["n"] += r.get("n", 1)
    print(f"\n{'bucket':12s} {'ms/step':>9s} {'%window':>8s} {'TFLOP/s':>8s} "
          f"{'roofline':>9s} {'gap':>8s} {'ops/step':>9s}")
    for name, b in sorted(buckets.items(), key=lambda kv: -kv[1]["dur"]):
        tf = b["flops"] / (b["dur"] * 1e-6) / 1e12 if b["flops"] else 0.0
        print(f"{name:12s} {b['dur'] / per:9.2f} {100 * b['dur'] / window:7.1f}% "
              f"{tf:8.1f} {b['bound'] / per:9.2f} {(b['dur'] - b['bound']) / per:8.2f} "
              f"{b['n'] / max(steps, 1):9.0f}")
    print(f"{'idle':12s} {idle / per:9.2f} {100 * idle / window:7.1f}%")
    print("(per-bucket times are summed op durations; async ops overlap, so the column "
          "can exceed busy)")
    groups = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        g = groups[(r["bucket"], r["opcode"], r["shape"], r["source"], r["limit"])]
        g["dur"] += r["dur"]; g["bound"] += r["bound"]; g["flops"] += r["flops"]; g["n"] += r.get("n", 1)
    print(f"\ntop {top} by gap (ms/step = measured - roofline):")
    print(f"{'gap':>7s} {'ms':>7s} {'TF/s':>6s} {'n':>5s}  {'bucket':11s} {'limit':7s} op / shape / source")
    for key, g in sorted(groups.items(), key=lambda kv: -(kv[1]["dur"] - kv[1]["bound"]))[:top]:
        bucket, opcode, shape, source, limit = key
        tf = g["flops"] / (g["dur"] * 1e-6) / 1e12 if g["flops"] else 0.0
        print(f"{(g['dur'] - g['bound']) / per:7.2f} {g['dur'] / per:7.2f} {tf:6.1f} "
              f"{g['n'] / max(steps, 1):5.0f}  {bucket:11s} {limit:7s} {opcode} {shape}  @ {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir")
    parser.add_argument("--steps", type=int, default=1,
                        help="denoising steps in the traced window (per-step numbers)")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--peak-tflops", type=float, default=PEAK_TFLOPS)
    parser.add_argument("--hbm-gbps", type=float, default=HBM_GBPS)
    parser.add_argument("--json", help="also write the per-op rows here")
    args = parser.parse_args()
    result = analyze(args.trace_dir, args.peak_tflops, args.hbm_gbps)
    report(result, args.steps, args.top, args.peak_tflops)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
