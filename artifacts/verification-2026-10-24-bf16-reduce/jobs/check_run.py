"""check_run.py <run dir> [<other run dir>]: step stats, dump shapes, and (with a
second dir) bf16-vs-fp32 latent agreement and identical-initial-noise check."""
import json
import statistics
import sys
from pathlib import Path

import torch

runs = [Path(p) for p in sys.argv[1:]]
for run in runs:
    steps = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()][1:]
    print(run, "steps", len(steps) + 1,
          "step med ms", round(statistics.median(s["step_s"] for s in steps) * 1000, 1),
          "fwd med ms", round(statistics.median(s["forward_s"] for s in steps) * 1000, 1))
    for pt in sorted(run.glob("*.pt")):
        t = torch.load(pt)
        print("  ", pt.name, tuple(t.shape), t.dtype, "std", round(float(t.float().std()), 4))
if len(runs) == 2:
    a, b = runs
    for pt in sorted(a.glob("initial*.pt")):
        print(pt.name, "identical:", torch.equal(torch.load(pt), torch.load(b / pt.name)))
    for pt in sorted(a.glob("final*.pt")):
        x = torch.load(pt).double().flatten()
        y = torch.load(b / pt.name).double().flatten()
        print(pt.name, "cosine", round(float(torch.nn.functional.cosine_similarity(x, y, dim=0)), 6),
              "rel_l2", round(float((y - x).norm() / x.norm()), 5))
