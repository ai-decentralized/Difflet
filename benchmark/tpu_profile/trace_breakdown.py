"""Where does a traced TPU denoise go? Busy vs idle, time per HLO category with
the matmuls' achieved TFLOP/s, and the top source lines -- from a jax.profiler
trace of rank 0 (``DIFFLET_BENCH_TRACE_DIR`` in benchmark/adapters/tpu.py).

    python -m benchmark.tpu_profile.trace_breakdown /mnt/models/profile/trace-flux

Measured with it (v5e-4, compiled, 2026-11-03): FLUX.1-dev denoise is 51% matmul
at 83% of the 197 TFLOP/s bf16 peak, 20% all-reduce, 13% attention; LTX-2 is 51%
matmul, 33% all-reduce, and the device idles 11% between steps.
"""
import collections
import glob
import gzip
import json
import sys

PEAK_TFLOPS = 197  # v5e bf16

path = sorted(glob.glob(sys.argv[1] + "/plugins/profile/*/*.trace.json.gz"))[-1]
ev = json.load(gzip.open(path))["traceEvents"]
procs = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
threads = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
dev = [e for e in ev if e.get("ph") == "X" and procs.get(e["pid"]) == "/device:TPU:0"]
ops = [e for e in dev if threads.get((e["pid"], e["tid"])) == "XLA Ops"]
mods = [e for e in dev if threads.get((e["pid"], e["tid"])) == "XLA Modules"]
t0 = min(e["ts"] for e in mods); t1 = max(e["ts"] + e["dur"] for e in mods)
wall = t1 - t0
# busy = union of op intervals
iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in ops)
busy, cur_s, cur_e = 0.0, None, None
for s, e in iv:
    if cur_e is None or s > cur_e:
        if cur_e is not None:
            busy += cur_e - cur_s
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
busy += cur_e - cur_s
mod_names = collections.Counter(m["name"][:50] for m in mods)
print(f"window {wall/1e3:.1f} ms, {len(mods)} module executions, device busy {busy/1e3:.1f} ms "
      f"({100*busy/wall:.1f}%), idle {100*(wall-busy)/wall:.1f}%")
print("modules:", mod_names.most_common(4))
cat = collections.defaultdict(float); flops = collections.defaultdict(float)
for e in ops:
    c = (e.get("args") or {}).get("hlo_category") or e["name"].split(".")[0]
    cat[c] += e["dur"]; flops[c] += float((e.get("args") or {}).get("model_flops") or 0)
print("\nper HLO category (ms, % of busy, achieved TFLOP/s where flops):")
for c, d in sorted(cat.items(), key=lambda x: -x[1])[:14]:
    tf = flops[c] / (d * 1e-6) / 1e12 if flops[c] else 0
    print(f"  {c:32s} {d/1e3:9.1f} {100*d/busy:6.1f}%" + (f"  {tf:6.1f} TF/s ({100*tf/PEAK_TFLOPS:.0f}% of {PEAK_TFLOPS})" if tf else ""))
src = collections.defaultdict(float)
for e in ops:
    a = e.get("args") or {}
    s = (a.get("source") or a.get("tf_op") or e["name"])
    s = s.split("/")[-1][:90]
    src[s] += e["dur"]
print("\ntop source lines (ms, % of busy):")
for s, d in sorted(src.items(), key=lambda x: -x[1])[:15]:
    print(f"  {d/1e3:8.1f} {100*d/busy:5.1f}%  {s}")
