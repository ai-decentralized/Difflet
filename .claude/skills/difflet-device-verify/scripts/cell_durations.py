#!/usr/bin/env python3
"""Print one markdown table row per cell from every /tmp/logs/verify_matrix_*/results.json.

  python3 cell_durations.py [model-prefix]
Columns: run, cell, outcome, compile s, generate s — paste into the evidence doc.
"""
import glob, json, sys
prefix = sys.argv[1] + "/" if len(sys.argv) > 1 else ""
print("| Run | Cell | Outcome | Compile | Generate |\n|---|---|---|---:|---:|")
for p in sorted(glob.glob("/tmp/logs/verify_matrix_*/results.json")):
    run = p.split("/")[3].replace("verify_matrix_", "")
    try: d = json.load(open(p))
    except Exception: continue
    for k, v in d.get("cells", {}).items():
        if not k.startswith(prefix): continue
        c = v.get("compile") or {}; g = v.get("generate") or {}
        print(f"| `run-{run}` | `{k}` | **{v.get('outcome')}** | {c.get('duration_s') or 0:.1f} s | {g.get('duration_s') or 0:.1f} s |")
