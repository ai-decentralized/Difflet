"""Task 5 device checks (b) and (c) from kernel/acc and kernel/trio: bit identity of the routed op
against the probe's fork QK-only with the VectorE exp (int16 views), the trio's per-arm medians,
spreads and ratios (J = max(2%, cte spread)), and rel L2 vs the Task 1 mirror (flat, peaked).
Writes kernel/summary.json and prints it."""
import json
import statistics
import sys
from pathlib import Path

import torch

K = Path("/mnt/nvme/difflet-mxattn/task5/kernel")
SHAPES = sys.argv[1:] or ["qwen_1024", "wan_81f"]


def same(a: Path, b: Path) -> dict:
    x, y = torch.load(a), torch.load(b)
    eq = x.shape == y.shape and torch.equal(x.view(torch.int16), y.view(torch.int16))
    return {"a": str(a), "b": str(b), "bit_identical": bool(eq),
            "differing": None if x.shape != y.shape else int((x.view(torch.int16) != y.view(torch.int16)).sum())}


out = {}
for s in SHAPES:
    row = {}
    acc = {n: json.loads((K / "acc" / f"{n}.{s}.json").read_text())
           for n in ("fvexp_flat", "routed_flat", "fvexp_peak", "cte_flat")
           if (K / "acc" / f"{n}.{s}.json").exists()}
    row["accuracy"] = {n: {k: r[k] for k in ("reference", "rel_l2", "cosine", "nonfinite")}
                       for n, r in acc.items()}
    row["bits"] = {"acc routed vs fvexp": same(K / "acc" / f"routed_flat.{s}.out.pt",
                                              K / "acc" / f"fvexp_flat.{s}.out.pt")}
    trio = K / "trio"
    if (trio / "routed" / f"{s}.out.pt").exists():
        row["bits"]["trio routed vs fvexp"] = same(trio / "routed" / f"{s}.out.pt",
                                                   trio / "fvexp" / f"{s}.out.pt")
        row["bits"]["trio fvexp vs acc fvexp"] = same(trio / "fvexp" / f"{s}.out.pt",
                                                      K / "acc" / f"fvexp_flat.{s}.out.pt")
    t = {}
    for a in ("cte", "fvexp", "routed"):
        runs = sorted((trio / a).glob(f"{s}.run*.json"))
        if not runs:
            continue
        ms = [json.loads(p.read_text())["ms"] for p in runs]
        med = statistics.median(ms)
        t[a] = {"runs_ms": ms, "median_ms": med, "spread": (max(ms) - min(ms)) / med,
                "fuser": [json.loads(p.read_text())["fuser"] for p in runs]}
    if "cte" in t:
        j = max(0.02, t["cte"]["spread"])
        for a in ("fvexp", "routed"):
            if a in t:
                r = t[a]["median_ms"] / t["cte"]["median_ms"]
                t[a].update(ratio_vs_cte=r, J=j, faster=r < 1 - j)
    row["timing"] = t
    out[s] = row
(K / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
