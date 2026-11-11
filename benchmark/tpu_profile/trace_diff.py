"""Where two traced TPU denoises differ: device time per HLO op kind, side by side.

    python -m benchmark.tpu_profile.trace_diff <trace_dir_a> <trace_dir_b> [--top 25]

Reads the rank-0 ``jax.profiler`` traces that main's bench writes with
``DIFFLET_BENCH_TRACE_DIR`` (as trace_breakdown / exposed_comm) and sums the
device op time ("XLA Ops" on /device:TPU:0) per op kind -- the op name without
its ``.N`` suffix, plus the profiler's ``hlo_category`` when it has one -- over
the traced window. Prints the window, the busy time, and the kinds whose totals
differ most, so a change that should remove one kind of op (for example a
relayout ``copy``) can be checked against what else moved.

When the JSON export has no device events (it stops at 5M events), the raw
xplane is read instead (as exposed_comm does); its op names are full HLO lines,
so ``--by-shape`` can key each op by opcode and output shape.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import re

SUFFIX = re.compile(r"(\.\d+)+$")
#: "%name = <shape> <opcode>(" -- an HLO instruction line (xplane op names).
HLO_LINE = re.compile(r"^%?[\w.-]+ = (.*?)\s([a-z][a-z0-9-]*)\(")


def _load_xplane(trace_dir: str):
    import jax.profiler

    path = sorted(glob.glob(trace_dir + "/plugins/profile/*/*.xplane.pb"))[-1]
    data = jax.profiler.ProfileData.from_file(path)
    plane = next(p for p in data.planes if p.name == "/device:TPU:0")
    lines = {line.name: line for line in plane.lines}

    def events(name):
        return [{"name": e.name, "ts": e.start_ns / 1e3, "dur": e.duration_ns / 1e3}
                for e in lines[name].events] if name in lines else []

    return events("XLA Ops"), events("XLA Modules")


def _load(trace_dir: str):
    path = sorted(glob.glob(trace_dir + "/plugins/profile/*/*.trace.json.gz"))[-1]
    events = json.load(gzip.open(path))["traceEvents"]
    procs = {e["pid"]: e["args"]["name"] for e in events
             if e.get("ph") == "M" and e.get("name") == "process_name"}
    threads = {(e["pid"], e["tid"]): e["args"]["name"] for e in events
               if e.get("ph") == "M" and e.get("name") == "thread_name"}
    device = [e for e in events if e.get("ph") == "X" and procs.get(e["pid"]) == "/device:TPU:0"]
    ops = [e for e in device if threads.get((e["pid"], e["tid"])) == "XLA Ops"]
    modules = [e for e in device if threads.get((e["pid"], e["tid"])) == "XLA Modules"]
    if not modules:
        return _load_xplane(trace_dir)
    return ops, modules


def _busy(intervals) -> float:
    total, start, end = 0.0, None, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            if end is not None:
                total += end - start
            start, end = s, e
        else:
            end = max(end, e)
    return total + ((end - start) if end is not None else 0.0)


def _kind(name: str, by_shape: bool) -> str:
    hlo = HLO_LINE.match(name)
    if hlo:  # an xplane op: a full HLO line
        shape, opcode = hlo.group(1), hlo.group(2)
        return f"{opcode} {shape}" if by_shape else opcode
    return SUFFIX.sub("", name)


def summarize(trace_dir: str, by_shape: bool = False) -> dict:
    ops, modules = _load(trace_dir)
    window = (max(m["ts"] + m["dur"] for m in modules) - min(m["ts"] for m in modules)) / 1e3
    kinds = collections.Counter()
    counts = collections.Counter()
    for e in ops:
        args = e.get("args", {})
        kind = _kind(e["name"], by_shape)
        category = args.get("hlo_category") or args.get("category")
        if category:
            kind = f"{kind} [{category}]"
        kinds[kind] += e["dur"] / 1e3
        counts[kind] += 1
    return {"window_ms": window,
            "busy_ms": _busy([(e["ts"], e["ts"] + e["dur"]) for e in ops]) / 1e3,
            "kinds": kinds, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--by-shape", action="store_true",
                        help="key ops by opcode and output shape (xplane traces)")
    parser.add_argument("--only", default="",
                        help="comma-separated opcodes to list (e.g. copy,reshape)")
    args = parser.parse_args()
    a, b = summarize(args.a, args.by_shape), summarize(args.b, args.by_shape)
    print(f"window ms: a {a['window_ms']:.1f}  b {b['window_ms']:.1f}  "
          f"(b - a {b['window_ms'] - a['window_ms']:+.1f})")
    print(f"busy ms:   a {a['busy_ms']:.1f}  b {b['busy_ms']:.1f}  "
          f"(b - a {b['busy_ms'] - a['busy_ms']:+.1f})")
    keys = set(a["kinds"]) | set(b["kinds"])
    if args.only:
        wanted = tuple(args.only.split(","))
        keys = {k for k in keys if k.split(" ")[0].split(" [")[0] in wanted}
    rows = sorted(keys, key=lambda k: -abs(b["kinds"][k] - a["kinds"][k]))[:args.top]
    width = 110 if args.by_shape else 60
    print(f"{'op kind':{width}s} {'a ms':>9s} {'b ms':>9s} {'b - a':>9s} {'a n':>7s} {'b n':>7s}")
    for k in rows:
        print(f"{k[:width]:{width}s} {a['kinds'][k]:9.1f} {b['kinds'][k]:9.1f} "
              f"{b['kinds'][k] - a['kinds'][k]:+9.1f} {a['counts'][k]:7d} {b['counts'][k]:7d}")
    print(f"{'(all listed kinds)' if args.only else '(all kinds)':{width}s} {sum(a['kinds'].values()):9.1f} {sum(b['kinds'].values()):9.1f} "
          f"{sum(b['kinds'].values()) - sum(a['kinds'].values()):+9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
