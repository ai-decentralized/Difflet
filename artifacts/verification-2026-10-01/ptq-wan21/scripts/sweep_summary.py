import json, sys
base = "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/artifacts/verification-2026-10-01/ptq-wan21/linear_sweep"
for t in (900, 500, 100):
    d = json.load(open(f"{base}/linear_error_t{t}.json"))
    print(f"t={t} rows={d['row_count']} forward_s={d['forward_seconds']} metrics_s={d['metrics_seconds']}")
    for s, v in d["summary"].items():
        print(f"  {s:18s} min_cos={v['min_cosine']:.6f} mean_cos={v['mean_cosine']:.6f} "
              f"max_relL2={v['max_rel_l2']:.4f} min_snr={v['min_snr_db']:.2f} mean_snr={v['mean_snr_db']:.2f} worst={v['worst_cell']}")
    # worst cells for the production scheme
    rows = sorted(d["rows"], key=lambda r: r["metrics"]["fp8-tensor-dyn"]["cosine"])[:5]
    print("  worst fp8-tensor-dyn cells:", [(r["name"], round(r["metrics"]["fp8-tensor-dyn"]["cosine"], 6), round(r["metrics"]["fp8-tensor-dyn"]["snr_db"], 1)) for r in rows])
    # per linear-type mean SNR
    by = {}
    for r in d["rows"]:
        by.setdefault(r["linear"], []).append(r["metrics"]["fp8-tensor-dyn"]["snr_db"])
    print("  mean SNR by linear (fp8-tensor-dyn):", {k: round(sum(v)/len(v), 1) for k, v in sorted(by.items())})
