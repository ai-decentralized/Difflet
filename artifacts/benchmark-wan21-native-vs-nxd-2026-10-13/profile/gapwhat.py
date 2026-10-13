"""What runs during the largest PE idle gaps of the TorchNeuron block profile (pcore 1 / sg01)."""
import collections
import json
import sys

d = json.load(open(sys.argv[1]))
total = d["summary"][0]["total_time"] * 1e9
pc = int(sys.argv[2]) if len(sys.argv) > 2 else 1
iv = sorted((a["start_ts"], a["end_ts"]) for a in d["active_time"] if a["engine"] == "tensor" and a["pcore_idx"] == pc)
merged = []
for s, e in iv:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])
gaps = sorted(((b[0] - a[1], a[1], b[0]) for a, b in zip(merged, merged[1:])), reverse=True)[:8]
sg = "sg01" if pc == 1 else "sg00"
ins = [i for i in d["instruction"] if i["subgraph"] == sg and i["label"] not in ("Tensor", "TensorMatrix")]
cc = d["cc_ops"]
for g, s, e in gaps:
    act = collections.Counter()
    for i in ins:
        a, b = i["timestamp"], i["timestamp"] + i["duration"]
        if b > s and a < e:
            act[(i["label"], (i.get("hlo_name") or "")[:70])] += min(b, e) - max(a, s)
    ccs = [(c["operation"], c["algorithm"], c["input_size"], round(c["timestamp"] / 1e6, 3), round(c["duration"] / 1e3, 1))
           for c in cc if c["timestamp"] + c["duration"] > s and c["timestamp"] < e]
    print(f"gap {g/1e6:.3f} ms [{s/1e6:.3f}-{e/1e6:.3f}] cc={ccs}")
    for (lab, h), v in act.most_common(6):
        print(f"     {lab:<8} {v/1e3:8.1f}us  {h}")
