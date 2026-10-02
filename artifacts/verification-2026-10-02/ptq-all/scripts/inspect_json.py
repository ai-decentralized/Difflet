import json, sys
root = "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/benchmark/trn2/"
SKIP = {"compile_breakdown", "e2e_breakdown", "toolchain", "prompt", "device", "device_slug", "config_slug",
        "backend", "model_type", "output_kind", "dtype", "parallel", "shape", "guidance_scale", "seed", "timestamp",
        "model_id", "revision", "config_label", "quant"}
for s in sys.argv[1:]:
    d = json.load(open(f"{root}{s}.json"))
    print("====", s)
    for k in sorted(d):
        if k in SKIP:
            continue
        v = d[k]
        if isinstance(v, dict) and "samples" in v:
            v = {kk: vv for kk, vv in v.items() if kk != "samples"}
        txt = json.dumps(v)
        print(f"  {k}: {txt[:600]}")
