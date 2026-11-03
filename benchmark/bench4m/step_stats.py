"""Per-step DiT forward time (steps.jsonl of a q* run): median and mean of steps 1..N (step 0
includes first-call overhead), plus step 0. Usage: step_stats.py <label>=<steps.jsonl> ..."""

import json
import statistics
import sys

for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    rows = [json.loads(line) for line in open(path)]
    fw = [r["forward_s"] for r in rows[1:]]
    print(f"{label:28s} n={len(rows)} step0 {rows[0]['forward_s'] * 1000:7.1f} ms | "
          f"steps 1+ median {statistics.median(fw) * 1000:7.1f} ms mean {statistics.mean(fw) * 1000:7.1f} ms "
          f"min {min(fw) * 1000:7.1f} max {max(fw) * 1000:7.1f}")
