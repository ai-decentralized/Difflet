"""Split a system trace into DiT forwards and summarise each (one logical core, deduped by exec_id).

usage: forwards.py <json> <block-or-graph model substring> [gap_ms]
A forward = a run of executions with inter-exec gaps < gap_ms (default 3 ms) that contains
the named model. Prints per-forward: wall (first start -> last end), busy, gaps, #execs,
named-model execs and their total, other execs total, collectives per exec (count, ms).
"""
import collections
import json
import statistics as st
import sys

d = json.load(open(sys.argv[1]))
key = sys.argv[2]
gap_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
ev = d["trace_event"]
exe = [e for e in ev if e["name"] == "nc_exec_running" and "exec_id" in e]
# one logical core: keep the lowest lnc/nc index, dedupe the 2 physical cores by exec_id
nc0 = min(e["nc_idx"] for e in exe)
seen = {}
for e in exe:
    if e["nc_idx"] != nc0:
        continue
    k = (e["model_name"], e["exec_id"])
    if k not in seen or e["duration"] > seen[k]["duration"]:
        seen[k] = e
exe = sorted(seen.values(), key=lambda e: e["timestamp"])
cc = collections.defaultdict(list)
for c in ev:
    if c["name"] == "cc_running" and c.get("nc_idx") == nc0:
        cc[c.get("exec_id")].append(c)

groups, cur = [], [exe[0]]
for a, b in zip(exe, exe[1:]):
    if (b["timestamp"] - (a["timestamp"] + a["duration"])) / 1e6 > gap_ms:
        groups.append(cur)
        cur = []
    cur.append(b)
groups.append(cur)
rows = []
for g in groups:
    if not any(key in e["model_name"] for e in g):
        continue
    start = g[0]["timestamp"]
    end = max(e["timestamp"] + e["duration"] for e in g)
    busy = sum(e["duration"] for e in g)
    named = [e for e in g if key in e["model_name"]]
    other = [e for e in g if key not in e["model_name"]]
    ccs = [c for e in named for c in cc.get(e["exec_id"], [])]
    rows.append(dict(wall=(end - start) / 1e6, busy=busy / 1e6, gaps=(end - start - busy) / 1e6,
                     n=len(g), n_named=len(named), named_ms=sum(e["duration"] for e in named) / 1e6,
                     n_other=len(other), other_ms=sum(e["duration"] for e in other) / 1e6,
                     n_cc=len(ccs), cc_ms=sum(c["duration"] for c in ccs) / 1e6,
                     t=start / 1e9))
for r in rows:
    print({k: round(v, 3) for k, v in r.items()})
if rows:
    keys = rows[0].keys()
    tail = rows[1:] if len(rows) > 2 else rows
    print("median over forwards (skip first):", {k: round(st.median(r[k] for r in tail), 3) for k in keys})
