import json, sys, statistics as st
d = json.load(open(sys.argv[1]))
layers = d["layers"]
rows = []
for name, rec in layers.items():
    ps = rec["per_step"]
    amax = rec["amax"]
    mn = min(ps)
    rows.append((amax / mn if mn > 0 else float("inf"), name, amax, mn, ps.index(max(ps)), ps.index(mn)))
rows.sort(reverse=True)
print("layers", len(layers))
spreads = [r[0] for r in rows]
print("spread quantiles p50 %.2f p90 %.2f p99 %.2f max %.2f" % (
    st.median(spreads), sorted(spreads)[int(0.9 * len(spreads))], sorted(spreads)[int(0.99 * len(spreads))], max(spreads)))
print("count spread>4:", sum(s > 4 for s in spreads), " >8:", sum(s > 8 for s in spreads), " >16:", sum(s > 16 for s in spreads))
print("worst 15 (spread, name, amax, min_step_amax, argmax_step, argmin_step):")
for r in rows[:15]:
    print("  %.1f  %s  amax=%.3g min=%.3g argmax=%d argmin=%d" % r)
# spread excluding step 0
rows2 = []
for name, rec in layers.items():
    ps = rec["per_step"][1:]
    rows2.append((max(ps) / min(ps) if min(ps) > 0 else float("inf"), name))
rows2.sort(reverse=True)
print("excluding step 0: p50 %.2f max %.2f" % (st.median([r[0] for r in rows2]), rows2[0][0]))
# per-suffix summary
from collections import defaultdict
by = defaultdict(list)
for s, name, *_ in rows:
    by[name.split(".", 2)[-1]].append(s)
for k in sorted(by):
    v = by[k]
    print("  %-22s n=%3d median %.2f max %.2f" % (k, len(v), st.median(v), max(v)))
# the amax values themselves
am = sorted(rec["amax"] for rec in layers.values())
print("amax quantiles min %.3g p50 %.3g max %.3g" % (am[0], am[len(am) // 2], am[-1]))
