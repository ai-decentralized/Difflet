"""Group device-profile instructions by HLO op family and engine (one physical core subgraph).

usage: instr.py <full.json> [subgraph=sg00]
Prints, per family: instruction busy time per engine (sum of durations), and the wall span.
"""
import collections
import json
import re
import sys

d = json.load(open(sys.argv[1]))
sg = sys.argv[2] if len(sys.argv) > 2 else "sg00"
ins = [i for i in d["instruction"] if i["subgraph"] == sg]
print("instructions", len(ins))


def family(i):
    n = (i.get("hlo_name") or "") + " " + (i.get("layer") or "")
    nl = n.lower()
    if "attn1" in nl and ("custom" in nl or "nki" in nl or "flash" in nl or "sdpa" in nl or "attention_kernel" in nl):
        return "attn1_kernel"
    if "attn2" in nl and ("custom" in nl or "nki" in nl or "flash" in nl or "sdpa" in nl or "attention_kernel" in nl):
        return "attn2_kernel"
    if "custom" in nl or "nki" in nl or "flash" in nl:
        return "kernel_other"
    if "all-reduce" in nl or "all_reduce" in nl:
        return "allreduce"
    m = re.search(r"(attn1|attn2|ffn|norm1|norm2|norm3)", nl)
    blk = m.group(1) if m else "top"
    if "dot" in nl:
        return blk + ":dot"
    if "transpose" in nl:
        return blk + ":transpose"
    return blk + ":other"


busy = collections.defaultdict(lambda: collections.Counter())
span = {}
names = collections.Counter()
for i in ins:
    f = family(i)
    busy[f][i["label"]] += i["duration"]
    t0, t1 = i["timestamp"], i["timestamp"] + i["duration"]
    a, b = span.get(f, (t0, t1))
    span[f] = (min(a, t0), max(b, t1))
    names[(f, (i.get("hlo_name") or "")[:80])] += i["duration"]
for f in sorted(busy, key=lambda f: -sum(busy[f].values())):
    c = busy[f]
    print(f"{f:<22} total={sum(c.values())/1e6:8.3f}ms span={(span[f][1]-span[f][0])/1e6:7.3f}ms  " +
          " ".join(f"{k}={v/1e6:.3f}" for k, v in c.most_common()))
print("\ntop hlo names by busy time:")
for (f, n), v in names.most_common(30):
    print(f"  {v/1e6:8.3f}ms {f:<18} {n}")
