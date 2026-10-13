"""NxD per-layer attention_cte clusters vs the rest (parquet profile, covered layers only)."""
import collections
import statistics as st

import pyarrow.parquet as pq

P = "/home/ubuntu/wan21-bench/profile/prof/nxd_parquet/profiles/global/nxd_dit@latest/"
t = pq.read_table(P + "Instruction.parquet", columns=["start_ts", "end_ts", "duration_ns", "engine", "pcore_idx", "nki_source_location"]).to_pydict()
cc = sorted(pq.read_table(P + "CcOp.parquet").to_pylist(), key=lambda c: c["start_ts"])
marks = [c["start_ts"] for c in cc if c["input_size"] == 2048]
last_cov = max(t["end_ts"])
pc = 1
att = sorted((s, e) for s, e, p, src in zip(t["start_ts"], t["end_ts"], t["pcore_idx"], t["nki_source_location"])
             if p == pc and src and "attention_cte" in src)
# Busy time per source category on Tensor engine per layer window
cat_busy = collections.defaultdict(lambda: collections.Counter())


def cat(src):
    if not src:
        return "none"
    if "attention_cte" in src:
        return "attention_cte"
    if "parallel_layers/layers.py" in src:
        return "nxd_linear"
    if "normalization" in src or "modeling_wan" in src or "module.py" in src:
        return src.rsplit("/", 1)[-1]
    return src.rsplit("/", 1)[-1]


for s, e, p, eng, src in zip(t["start_ts"], t["end_ts"], t["pcore_idx"], t["engine"], t["nki_source_location"]):
    if p != pc:
        continue
    # layer index by marker
    k = None
    for i in range(len(marks) - 1):
        if marks[i] <= s < marks[i + 1]:
            k = i
            break
    if k is None:
        continue
    cat_busy[k][(eng, cat(src))] += e - s
full = [k for k in sorted(cat_busy) if marks[k + 1] < last_cov]
print("fully covered layer windows (marker k -> k+1):", full[:3], "...", full[-3:], "n", len(full))
# attention clusters
clusters = []
for s, e in att:
    if clusters and s - clusters[-1][1] < 200_000:
        clusters[-1][1] = max(clusters[-1][1], e)
    else:
        clusters.append([s, e])
spans = [(c[1] - c[0]) / 1e6 for c in clusters if c[1] < last_cov]
print("attention_cte clusters", len(spans), "spans ms (first 12):", [round(x, 3) for x in spans[:12]])
big = [x for x in spans if x > 1.0]
small = [x for x in spans if x <= 1.0]
print("self-attn-like (>1ms) median span ms", round(st.median(big), 3) if big else None, "n", len(big))
print("cross-attn-like (<=1ms) median span ms", round(st.median(small), 3) if small else None, "n", len(small))
agg = collections.Counter()
for k in full:
    for key, v in cat_busy[k].items():
        agg[key] += v
print("\nper-layer instruction busy (sum of durations, pcore1) by (engine, source), ms:")
for (eng, c), v in agg.most_common(25):
    print(f"  {eng:<8} {c:<40} {v / len(full) / 1e6:8.3f}")
