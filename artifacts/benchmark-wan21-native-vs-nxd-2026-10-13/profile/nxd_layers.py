"""NxD whole-DiT device profile (parquet): coverage, per-block spans, attention-kernel spans.

The instruction buffer overflowed, so only a prefix of the execution is captured; this reports
which part is covered and, for the covered blocks, per-block wall spans comparable to the
TorchNeuron per-block NEFF.
"""
import collections
import re
import statistics as st
import sys

import pyarrow.parquet as pq

P = "/home/ubuntu/wan21-bench/profile/prof/nxd_parquet/profiles/global/nxd_dit@latest/"
cols = ["start_ts", "end_ts", "duration_ns", "engine", "pcore_idx", "hlo_name", "layer", "opcode",
        "spill_save_bytes", "spill_reload_bytes", "raw_flops", "flops_transpose", "hbm_read_bytes", "hbm_write_bytes"]
t = pq.read_table(P + "Instruction.parquet", columns=cols).to_pydict()
n = len(t["start_ts"])
ex = pq.read_table(P + "ExecutionInfo.parquet").to_pylist()[0]
print("execution", ex)
print("rows", n, "first", min(t["start_ts"]), "last end", max(t["end_ts"]))
print("pcores", collections.Counter(t["pcore_idx"]))
print("engines", collections.Counter(t["engine"]).most_common())
layers = collections.Counter(t["layer"])
print("n distinct layer strings", len(layers))
for k, v in list(layers.most_common(15)):
    print("  layer", repr(k)[:140], v)
names = collections.Counter((h or "")[:60] for h in t["hlo_name"][:200000])
for k, v in names.most_common(10):
    print("  hlo", k, v)

# Block index: look for 'blocks.<i>' / 'blocks[<i>]' / 'WanTransformerBlock[blocks][<i>]' in layer or hlo_name
pat = re.compile(r"blocks[\.\[\]]*(\d+)")
blk = collections.defaultdict(lambda: [10**18, 0, 0, 0, 0, 0])  # start, end, busy, spill_rd, spill_wr, xpose
attn = collections.defaultdict(lambda: [10**18, 0])
for i in range(n):
    s = (t["layer"][i] or "") + " " + (t["hlo_name"][i] or "")
    m = pat.search(s)
    if not m:
        continue
    b = int(m.group(1))
    r = blk[b]
    r[0] = min(r[0], t["start_ts"][i]); r[1] = max(r[1], t["end_ts"][i]); r[2] += t["duration_ns"][i]
    r[3] += t["spill_reload_bytes"][i] or 0; r[4] += t["spill_save_bytes"][i] or 0; r[5] += t["flops_transpose"][i] or 0
    if "custom-call" in s or "custom_call" in s:
        if "attn1" in s:
            a = attn[(b, "attn1")]
        elif "attn2" in s:
            a = attn[(b, "attn2")]
        else:
            a = attn[(b, "kernel")]
        a[0] = min(a[0], t["start_ts"][i]); a[1] = max(a[1], t["end_ts"][i])
print("\nblocks found", sorted(blk))
starts = []
for b in sorted(blk):
    r = blk[b]
    starts.append(r[0])
    print(f"block {b:2d} start={r[0]/1e6:9.3f}ms span={(r[1]-r[0])/1e6:7.3f}ms spill_rd={r[3]/1e9:.3f}GB spill_wr={r[4]/1e9:.3f}GB xposeGF={r[5]/1e9:.1f} "
          + " ".join(f"{k[1]}={(v[1]-v[0])/1e6:.3f}" for k, v in attn.items() if k[0] == b))
d = [b - a for a, b in zip(starts, starts[1:])]
if d:
    print("block-to-block start deltas ms:", [round(x / 1e6, 3) for x in d])
    print("median block period ms", round(st.median(d) / 1e6, 3))
