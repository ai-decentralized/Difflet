"""The last device ops each chip completed, from per-rank jax.profiler traces.

    python -m benchmark.tpu_profile.trace_tail <trace_dir> [N]

For a hung run (cm_padded_repro --trace): every rank wrote a trace under
``<trace_dir>/rank<r>``. Prints, per TPU device in each trace, the last N
"XLA Ops" events (name, end time relative to that device's first op) and the
last module, so the chips' stopping points can be compared side by side.
"""
import glob
import gzip
import json
import sys


def main() -> int:
    root, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
    for path in sorted(glob.glob(root + "/rank*/plugins/profile/*/*.trace.json.gz")):
        ev = json.load(gzip.open(path))["traceEvents"]
        procs = {e["pid"]: e["args"]["name"] for e in ev
                 if e.get("ph") == "M" and e.get("name") == "process_name"}
        threads = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev
                   if e.get("ph") == "M" and e.get("name") == "thread_name"}
        print(f"== rank{path.split('/rank')[1].split('/')[0]}")
        for pid, name in sorted(procs.items(), key=lambda kv: kv[1]):
            if not name.startswith("/device:TPU"):
                continue
            ops = sorted((e for e in ev if e.get("ph") == "X" and e["pid"] == pid
                          and threads.get((pid, e["tid"])) == "XLA Ops"), key=lambda e: e["ts"] + e["dur"])
            mods = sorted((e for e in ev if e.get("ph") == "X" and e["pid"] == pid
                           and threads.get((pid, e["tid"])) == "XLA Modules"), key=lambda e: e["ts"])
            if not ops:
                print(f"  {name}: no ops")
                continue
            t0 = ops[0]["ts"]
            print(f"  {name}: {len(ops)} ops, {len(mods)} modules; last module "
                  f"{mods[-1]['name'][:60] if mods else '-'}")
            for e in ops[-n:]:
                print(f"    +{(e['ts'] + e['dur'] - t0) / 1e3:9.2f} ms  {e['name'][:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
