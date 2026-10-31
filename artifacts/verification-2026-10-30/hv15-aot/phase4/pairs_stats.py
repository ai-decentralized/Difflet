"""pairs_stats.py <pairs.json> [<calibration.json>]: Pearson / R^2 of the TeaCache signal, and the steps
the calibrated controller would skip on this trajectory (per-step, non-accumulate decision)."""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
s = sorted(d["samples"], key=lambda r: r["step_index"])
x = np.array([r["mod_input_diff_norm"] for r in s])
y = np.array([r["noise_pred_diff_norm"] for r in s])
i = np.array([r["step_index"] for r in s])
r = np.corrcoef(x, y)[0, 1]
print(f"n={len(s)} pearson={r:.4f} linear_r2={r * r:.4f}")
win = (i >= 5) & (i < 45)
rw = np.corrcoef(x[win], y[win])[0, 1]
print(f"steps 5..44 only: n={win.sum()} pearson={rw:.4f} linear_r2={rw * rw:.4f}")
print("x (probe rel-L1)   min/median/max:", np.round([x.min(), np.median(x), x.max()], 5))
print("y (noise_pred rel-L1) min/median/max:", np.round([y.min(), np.median(y), y.max()], 5))
print("per step (i: x, y):", " ".join(f"{a}:{b:.4f},{c:.4f}" for a, b, c in zip(i, x, y)))
if len(sys.argv) > 2:
    cal = json.load(open(sys.argv[2]))
    pred = np.polyval(list(reversed(cal["poly_coef"])), x)
    skip = [int(a) for a, p in zip(i, pred) if 5 <= a < 50 - 5 and p < cal["threshold"]]
    print(f"threshold={cal['threshold']:.4f} predicted<thr in window: {len(skip)} steps {skip}")
