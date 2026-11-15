"""Print kernel/summary.json compactly."""
import json

s = json.load(open("/mnt/nvme/difflet-mxattn/task5/kernel/summary.json"))
for k, r in s.items():
    print(k, {n: b["bit_identical"] for n, b in r["bits"].items()})
    for a, t in r["timing"].items():
        print("  ", a, [round(x, 4) for x in t["runs_ms"]], round(t["median_ms"], 4),
              "%.2f%%" % (100 * t["spread"]), round(t.get("ratio_vs_cte", 1), 4), t.get("faster"))
    for n, a in r["accuracy"].items():
        print("  ", n, a["reference"], "%.3e" % a["rel_l2"], a["nonfinite"])
