import json, re
base = "/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan/artifacts/verification-2026-10-01/ptq-wan21"
for d in ("ab-fixed", "ab-wo-fixed"):
    s = json.load(open(f"{base}/{d}/ab_summary.json"))
    arm = s["arms"]["fp8"]
    print(f"== {d}: compile {arm.get('compile_seconds')} s")
    for r in arm["runs"]:
        log = open(f"{base}/{d}/logs/generate_fp8_run{r['run']}.log", errors="ignore").read()
        loads = re.findall(r"total load_weights ([\d.]+)s", log)
        st = r["dit_step_ms"]
        print(f"   run {r['run']}: e2e {r['e2e_wall_seconds']} s, loads(te/tr/vae) {loads}, step {st['mean']:.1f}/{st['median']:.1f}")
    c = s["compare"].get("fp8_run1_vs_run0_control", {})
    print("   determinism:", {k: c.get("output", {}).get(k) for k in ("psnr_db", "ssim")}, "latent", c.get("latents", {}).get("cosine"))
    cmp = json.load(open(f"{base}/{d}/compare_fp8_vs_bf16_run1.json"))
    o = cmp.get("output", cmp)
    lat = cmp.get("latents", {})
    print("   fp8 vs bf16 run1: psnr", round(o.get("psnr_db", 0), 2), "ssim", round(o.get("ssim", 0), 4), "lpips", o.get("lpips"),
          "| latent cosine", lat.get("cosine"), "snr", lat.get("snr_db"), "mse", lat.get("mse"))
