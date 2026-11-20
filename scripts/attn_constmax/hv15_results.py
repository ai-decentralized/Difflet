"""HunyuanVideo-1.5 lane: collect the four arms' timed runs (hv15_arms.sh) and scores
(hv15_score.sh) into ROOT/results.json and print the table.

Per arm, from timed.log: each request's DiT step median (the driver's ``[hv15] request N: denoise
... median`` line, over its 50 steps), encode / denoise / decode seconds, rank 0's peak HBM, the
process wall time; the md5 of both mp4s. The arm's step is the mean of its two request medians,
spread = |r1 - r2| / mean; warm e2e = request 2's encode + denoise + decode (the stages run each
request in turn, so this excludes the stage loads and request 1's first-call kernel trace).

usage: hv15_results.py [ROOT]
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme/difflet-exp/constmax-models/hunyuan_video_15")
EVIDENCE_BF16_MD5 = "d3ef0f1c2ab7bd6f168180fc76cf80a0"  # 2026-11-07 doc 9b/12c bf16 (96 encode threads)


def md5(p: Path) -> str | None:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None


def arm(name: str) -> dict:
    d = ROOT / name
    log = (d / "timed.log").read_text(errors="replace") if (d / "timed.log").exists() else ""
    reqs = {}
    for m in re.finditer(r"\[hv15\] request (\d+): denoise ([\d.]+) s, (\d+) steps, median (\d+) ms \(first (\d+) ms\)", log):
        reqs.setdefault(int(m[1]), {}).update(denoise_s=float(m[2]), steps=int(m[3]), step_median_ms=int(m[4]),
                                              first_step_ms=int(m[5]))
    for m in re.finditer(r"\[hv15\] encode request (\d+): ([\d.]+) s", log):
        reqs.setdefault(int(m[1]), {})["encode_s"] = float(m[2])
    for m in re.finditer(r"\[hv15\] request (\d+): decode ([\d.]+) s", log):
        reqs.setdefault(int(m[1]), {})["decode_s"] = float(m[2])
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", log)
    peak = re.findall(r"\[hv15\] peak HBM \(rank 0\): ([\d.]+) GiB allocator \+ ([\d.]+) GiB runtime", log)
    out = {"ran": len(reqs) == 2 and all("step_median_ms" in r for r in reqs.values()),
           "requests": [reqs[k] for k in sorted(reqs)], "process_wall": wall[1] if wall else None,
           "peak_hbm_rank0_gib": [float(a) + float(b) for a, b in peak],
           "md5": [md5(d / f"warm_{i}.mp4") for i in (1, 2)]}
    if out["ran"]:
        meds = [r["step_median_ms"] for r in out["requests"]]
        out["step_ms"] = sum(meds) / 2
        out["spread_pct"] = 100 * abs(meds[0] - meds[1]) / out["step_ms"]
        last = out["requests"][1]
        out["warm_e2e_s"] = round(last["encode_s"] + last["denoise_s"] + last["decode_s"], 1)
    return out


def score(name: str) -> dict | None:
    f = ROOT / "scores" / f"{name}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    v, lat = d.get("video") or {}, d.get("latents") or {}
    return {"psnr": v.get("psnr"), "psnr_worst": v.get("psnr_worst"), "ssim": v.get("ssim"),
            "lpips": v.get("lpips"), "latent_cos": lat.get("cosine"), "latent_rel_l2": lat.get("rel_l2")}


arms = {a: arm(a) for a in "ABCD"}
step = {a: r.get("step_ms") for a, r in arms.items()}
ratio = lambda x, y: round(step[x] / step[y], 4) if step[x] and step[y] else None  # noqa: E731
res = {"model": "HunyuanVideo-1.5 480x848x121, tp2 x CFG2, 50 steps, CFG 6.0, seed 42", "arms": arms,
       "ratios": {"B/A": ratio("B", "A"), "C/A": ratio("C", "A"), "D/A": ratio("D", "A"), "D/C": ratio("D", "C")},
       "quality": {k: score(k) for k in ("B_vs_A", "C_vs_A", "D_vs_A", "D_vs_C", "A_vs_evidence_bf16")},
       "a_md5_equals_evidence_bf16": arms["A"]["md5"][1] == EVIDENCE_BF16_MD5,
       "evidence_bf16_md5": EVIDENCE_BF16_MD5}
(ROOT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
for a, r in arms.items():
    print(a, r.get("step_ms"), r.get("spread_pct"), r.get("warm_e2e_s"), [m and m[:8] for m in r["md5"]],
          [x.get("step_median_ms") for x in r["requests"]])
print(res["ratios"])
for k, v in res["quality"].items():
    print(k, v)
