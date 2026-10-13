"""Summarise nc_exec_running / cc_running events of a neuron-explorer system-trace JSON.

usage: sysexec.py <json> [t0_s t1_s]   (window relative to first_ts, seconds)
Prints: per-model exec count / total / median duration, exec timeline segments, gaps.
"""
import collections
import json
import statistics as st
import sys

d = json.load(open(sys.argv[1]))
ev = d["trace_event"]
names = d["device_names"][0] if d.get("device_names") else {}
exe = sorted([e for e in ev if e["name"] == "nc_exec_running"], key=lambda e: e["timestamp"])
cc = sorted([e for e in ev if e["name"] == "cc_running"], key=lambda e: e["timestamp"])
print("exec events", len(exe), "first", exe[0]["timestamp"] / 1e9, "last", exe[-1]["timestamp"] / 1e9)
if len(sys.argv) > 3:
    t0, t1 = float(sys.argv[2]) * 1e9, float(sys.argv[3]) * 1e9
    exe = [e for e in exe if t0 <= e["timestamp"] <= t1]
    cc = [e for e in cc if t0 <= e["timestamp"] <= t1]
by = collections.defaultdict(list)
for e in exe:
    by[e["model_name"]].append(e["duration"] / 1e6)
print(f"{'model':<52}{'n':>6}{'tot_ms':>10}{'med_ms':>9}{'min':>8}{'max':>8}")
for k, v in sorted(by.items(), key=lambda kv: -sum(kv[1])):
    print(f"{k:<52}{len(v):>6}{sum(v):>10.1f}{st.median(v):>9.3f}{min(v):>8.3f}{max(v):>8.3f}")
# gaps between consecutive executions on this core
gaps = []
for a, b in zip(exe, exe[1:]):
    gaps.append((b["timestamp"] - (a["timestamp"] + a["duration"])) / 1e6)
if gaps:
    print("execs", len(exe), "busy_ms", round(sum(e["duration"] for e in exe) / 1e6, 1),
          "gap_ms total", round(sum(gaps), 1), "median gap", round(st.median(gaps), 3))
ccby = collections.defaultdict(list)
for e in cc:
    ccby[e.get("model_name", e.get("op", "?"))].append(e["duration"] / 1e6)
for k, v in ccby.items():
    print("cc", k, len(v), round(sum(v), 1))
