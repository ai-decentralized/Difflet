"""How much communication a traced TPU denoise still waits on.

    python -m benchmark.tpu_profile.exposed_comm <trace_dir> [<trace_dir> ...]

Reads the rank-0 ``jax.profiler`` trace that main's bench writes with
``DIFFLET_BENCH_TRACE_DIR`` (the natural, warm, compiled request's denoise) and
splits device time three ways:

* **ICI wait** -- ops whose time is the core waiting on the interconnect:
  synchronous collectives (``all-reduce``, ``all-gather``, ``reduce-scatter``,
  ``all-to-all``) and the ``-done`` half of asynchronous ones
  (``collective-permute-done``, ...). The ``-start`` halves only issue the
  transfer and are counted as compute;
* **local DMA wait** -- ``copy-done`` / ``slice-done``: waits on on-chip async
  copies (HBM <-> VMEM), not on other chips;
* **compute** -- every other device op;
* **idle** -- the window minus the union of all op intervals.

This is the decision input of docs/plans/2026-11-10-tpu-collective-matmul-pallas.md
(step 1): the share of a step a better-overlapped collective could still save.
"""

from __future__ import annotations

import collections
import glob
import gzip
import json
import re
import sys

#: Ops that wait on the interconnect: synchronous collectives and the -done half
#: of asynchronous ones. copy-done / slice-done are local async DMAs, not ICI.
WAIT = re.compile(r"^(all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute"
                  r"|async-collective)(-done)?(\.\d+)?$")
LOCAL = re.compile(r"^(copy|slice)-done(\.\d+)?$")
ASYNC_START = re.compile(r"-start(\.\d+)?$")
#: "%name = <shape> <opcode>(" -- the opcode of an HLO instruction line.
HLO_LINE = re.compile(r"^%?[\w.-]+ = .*?\s([a-z][a-z0-9-]*)\(")


def _union(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + ((cur_e - cur_s) if cur_e is not None else 0.0)


def _from_json(trace_dir):
    path = sorted(glob.glob(trace_dir + "/plugins/profile/*/*.trace.json.gz"))[-1]
    ev = json.load(gzip.open(path))["traceEvents"]
    procs = {e["pid"]: e["args"]["name"] for e in ev
             if e.get("ph") == "M" and e.get("name") == "process_name"}
    threads = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev
               if e.get("ph") == "M" and e.get("name") == "thread_name"}
    dev = [e for e in ev if e.get("ph") == "X" and procs.get(e["pid"]) == "/device:TPU:0"]
    ops = [e for e in dev if threads.get((e["pid"], e["tid"])) == "XLA Ops"]
    mods = [e for e in dev if threads.get((e["pid"], e["tid"])) == "XLA Modules"]
    return ops, mods


def _from_xplane(trace_dir):
    """The raw xplane, for traces whose JSON export dropped the device events
    (it stops at 5M events; a host-heavy run fills them with /host:CPU)."""
    import jax.profiler

    path = sorted(glob.glob(trace_dir + "/plugins/profile/*/*.xplane.pb"))[-1]
    data = jax.profiler.ProfileData.from_file(path)
    plane = next(p for p in data.planes if p.name == "/device:TPU:0")
    lines = {line.name: line for line in plane.lines}

    def short(text):
        # xplane names are full HLO lines: "%x.3 = bf16[..] collective-permute-done(..)";
        # keep the opcode so the patterns below see what the JSON export calls them.
        m = HLO_LINE.match(text)
        return m.group(1) if m else text

    def events(name):
        return [{"name": short(e.name), "ts": e.start_ns / 1e3, "dur": e.duration_ns / 1e3}
                for e in lines[name].events] if name in lines else []

    return events("XLA Ops"), events("XLA Modules")


def analyze(trace_dir: str) -> dict:
    ops, mods = _from_json(trace_dir)
    if not mods:
        ops, mods = _from_xplane(trace_dir)
    t0 = min(e["ts"] for e in mods)
    t1 = max(e["ts"] + e["dur"] for e in mods)
    wall = t1 - t0
    busy = _union((e["ts"], e["ts"] + e["dur"]) for e in ops)
    waits = [e for e in ops if WAIT.search(e["name"]) and not ASYNC_START.search(e["name"])]
    wait = _union((e["ts"], e["ts"] + e["dur"]) for e in waits)
    local = _union((e["ts"], e["ts"] + e["dur"]) for e in ops if LOCAL.search(e["name"]))
    by_kind = collections.Counter()
    for e in waits:
        by_kind[re.sub(r"(\.\d+)$", "", e["name"])] += e["dur"]
    by_op = collections.Counter()
    for e in ops:
        by_op[re.sub(r"(\.\d+)+$", "", e["name"])] += e["dur"]
    return {
        "top_ops": by_op.most_common(8),
        "dir": trace_dir, "window_ms": wall / 1e3, "busy_ms": busy / 1e3,
        "wait_ms": wait / 1e3, "local_ms": local / 1e3, "idle_ms": (wall - busy) / 1e3,
        "modules": len(mods), "top_wait": by_kind.most_common(5),
    }


def main() -> int:
    print(f"{'trace':24s} {'window':>9s} {'ICI wait':>16s} {'local DMA wait':>16s} "
          f"{'compute':>16s} {'idle':>14s}")
    for d in sys.argv[1:]:
        r = analyze(d)
        w = r["window_ms"]
        compute = r["busy_ms"] - r["wait_ms"] - r["local_ms"]
        print(f"{d.rstrip('/').split('/')[-1]:24s} {w:8.1f}ms "
              f"{r['wait_ms']:8.1f}ms {100 * r['wait_ms'] / w:5.1f}% "
              f"{r['local_ms']:8.1f}ms {100 * r['local_ms'] / w:5.1f}% "
              f"{compute:8.1f}ms {100 * compute / w:5.1f}% "
              f"{r['idle_ms']:7.1f}ms {100 * r['idle_ms'] / w:4.1f}%")
        print("    top waits (ms): " + ", ".join(f"{k} {v / 1e3:.1f}" for k, v in r["top_wait"]))
        print("    top ops (ms):   " + ", ".join(f"{k} {v / 1e3:.1f}" for k, v in r["top_ops"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
