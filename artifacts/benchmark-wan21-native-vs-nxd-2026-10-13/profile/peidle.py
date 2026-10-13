"""PE (tensor engine) idle analysis of the TorchNeuron block profile: head/tail idle, idle gaps.

usage: peidle.py <tn_block_full.json>
"""
import json
import sys

d = json.load(open(sys.argv[1]))
total = d["summary"][0]["total_time"] * 1e9
for pc in (0, 1):
    iv = sorted((a["start_ts"], a["end_ts"]) for a in d["active_time"]
                if a["engine"] in ("tensor", "Tensor", "tensor_matrix", "TensorMatrix", "pe") and a["pcore_idx"] == pc)
    if not iv:
        engines = sorted(set(a["engine"] for a in d["active_time"]))
        print("engines in active_time:", engines)
        break
    merged = []
    for s, e in iv:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    busy = sum(e - s for s, e in merged)
    gaps = [(b[0] - a[1], a[1]) for a, b in zip(merged, merged[1:])]
    big = sorted(gaps, reverse=True)[:12]
    print(f"pcore {pc}: PE busy {busy/1e6:.3f} ms of {total/1e6:.3f}; head idle {merged[0][0]/1e6:.3f} ms; "
          f"tail idle {(total-merged[-1][1])/1e6:.3f} ms; idle gaps >50us total "
          f"{sum(g for g, _ in gaps if g > 50e3)/1e6:.3f} ms")
    print("   largest gaps (ms @ t ms):", [(round(g / 1e6, 3), round(t / 1e6, 3)) for g, t in big])
