import json
base = "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/artifacts/verification-2026-10-01/ptq-wan21/phase0-fixed"
for name in ("ptq_probe_report.json", "ptq_probe_report_weight_only.json"):
    r = json.load(open(f"{base}/{name}"))
    print("==", name, "spec", r["spec"]["activation"], "passed", r["passed"])
    print("  cpu_fp8_vs_cpu_bf16: cos %.7f snr %.2f" % (r["cpu_fp8_vs_cpu_bf16"]["cosine"], r["cpu_fp8_vs_cpu_bf16"]["snr_db"]))
    for k, v in r["checks"].items():
        print(f"  {k:28s} cos {v['cosine']:.7f} snr {v['snr_db']:.2f}")
    for a, rec in r["arms"].items():
        print(f"  arm {a}: compile {rec.get('compile_seconds')} load {rec.get('load_seconds')} fwd {rec.get('forward_ms', {}).get('mean', 0):.3f} ms nonfinite {rec.get('output_nonfinite')}")
