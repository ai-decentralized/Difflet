"""Experiment E1, lane HV1: collect hv1_arms.sh's timed arms into ROOT/results.json.

Per arm (A bf16, B bf16 + const-max, C mxfp8, D mxfp8 + const-max): each request's DiT step median
from the orchestrator's "request:" line (median over steps 2-50, n=49), the arm's warm step = the
mean of its two request medians, spread = |r1 - r2| / mean, warm e2e = request 2's wall time, the
mp4 and latents md5s; the ratios B/A, C/A, D/A, D/C; quality (scripts/compare_difflet_runs.py on
request 2's mp4 and latents) of B, C and D against A and of D against C; whether A reproduces the
evidence's bf16 clip (docs/verification/2026-11-07-mxfp8-video-models.md 7f); A's determinism: its
two requests and A/timed_pre (an earlier run of A) give one mp4 md5 and equal latents (the latents
files' md5s differ by request because torch.save names the archive after the file).
A pair whose ROOT/scores/<pair>.json exists is not re-scored, so the four pairs can run in parts.

usage: hv1_collect.py [ROOT] [CPUS]   (CPU only; TORCH_DEVICE_BACKEND_AUTOLOAD=0 is set for the
compare subprocesses)
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme/difflet-exp/constmax-models/hunyuan_video")
CPUS = sys.argv[2] if len(sys.argv) > 2 else "78-83,174-179"
EVIDENCE_BF16_MD5 = "258cd73e2d2de99b76bf2d6a6767b271"  # 7f: bf16 544x960x129, both requests
EVIDENCE_MX_MD5 = "00c19154d6c5bb583152de73ce074045"  # 7f: mxfp8 default, both requests
REQ = re.compile(r"request: ([\d.]+) s; to the first step's end ([\d.]+) s .*step median ([\d.]+) ms "
                 r"\(n=(\d+), max ([\d.]+)\); after the last step ([\d.]+) s")


def md5(p: Path) -> str | None:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None


def arm(name: str) -> dict:
    w = ROOT / name
    log = w / "timed.log"
    if not log.exists():
        return {"arm": name, "ran": False}
    reqs = [dict(zip(("request_s", "first_step_end_s", "step_median_ms", "n", "step_max_ms",
                      "after_last_step_s"), map(float, m.groups())))
            for m in REQ.finditer(log.read_text(errors="replace"))]
    out = {"arm": name, "ran": len(reqs) == 2, "requests": reqs,
           "mp4_md5": [md5(w / f"warm_{i}.mp4") for i in (1, 2)],
           "latents_md5": [md5(w / "work" / "worker_0" / f"latents_req000{i}.pt") for i in (0, 1)]}
    start = re.search(r"start-up: .*load ([\d.]+) s", log.read_text(errors="replace"))
    out["load_s"] = float(start.group(1)) if start else None
    if len(reqs) == 2:
        a, b = (r["step_median_ms"] for r in reqs)
        out["step_ms"] = (a + b) / 2
        out["spread_pct"] = abs(a - b) / ((a + b) / 2) * 100
        out["warm_e2e_s"] = reqs[1]["request_s"]
    return out


def compare(name: str, ref: str, cand: str) -> dict | None:
    r, c = ROOT / ref, ROOT / cand
    lat = lambda w: w / "work" / "worker_0" / "latents_req0001.pt"  # noqa: E731
    if not ((r / "warm_2.mp4").exists() and (c / "warm_2.mp4").exists()):
        return None
    out = ROOT / "scores" / f"{name}.json"
    if out.exists():  # shortcut: trusts an earlier score of the same files; delete it to re-score
        return json.loads(out.read_text())
    out.parent.mkdir(exist_ok=True)
    cmd = ["taskset", "-c", CPUS, sys.executable, str(REPO / "scripts/compare_difflet_runs.py"),
           "--ref-image", str(r / "warm_2.mp4"), "--cand-image", str(c / "warm_2.mp4"), "--out", str(out)]
    if lat(r).exists() and lat(c).exists():
        cmd += ["--ref-latents", str(lat(r)), "--cand-latents", str(lat(c))]
    with open(out.with_suffix(".log"), "w") as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT,
                       env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
    return json.loads(out.read_text())


def a_deterministic() -> dict:
    import torch
    runs = [ROOT / "A", ROOT / "A" / "timed_pre"]
    mp4 = {md5(w / f"warm_{i}.mp4") for w in runs for i in (1, 2)}
    lat = [torch.load(w / "work" / "worker_0" / f"latents_req000{i}.pt", map_location="cpu",
                      weights_only=True) for w in runs for i in (0, 1)]
    return {"mp4_md5s": sorted(m for m in mp4 if m), "one_mp4_md5": len(mp4) == 1 and None not in mp4,
            "latents_equal": all(torch.equal(lat[0], t) for t in lat[1:])}


def main() -> None:
    arms = {a: arm(a) for a in "ABCD"}
    arms["A_pre"] = arm("A/timed_pre")
    step = {a: r.get("step_ms") for a, r in arms.items()}
    ratio = lambda x, y: step[x] / step[y] if step[x] and step[y] else None  # noqa: E731
    quality = {f"{c}_vs_{r}": compare(f"{c}_vs_{r}", r, c)
               for c, r in (("B", "A"), ("C", "A"), ("D", "A"), ("D", "C"))}
    res = {"model": "HunyuanVideo 1.0 (hunyuanvideo-community/HunyuanVideo e8c2aaa6)",
           "shape": "544x960x129, tp4, 50 steps, seed 42, guidance 6.0", "arms": arms,
           "b_over_a": ratio("B", "A"), "c_over_a": ratio("C", "A"), "d_over_a": ratio("D", "A"),
           "d_over_c": ratio("D", "C"), "quality": quality,
           "a_deterministic": a_deterministic(),
           "a_md5_matches_evidence": (arms["A"].get("mp4_md5") or [None])[-1] == EVIDENCE_BF16_MD5,
           "c_md5_matches_evidence": (arms["C"].get("mp4_md5") or [None])[-1] == EVIDENCE_MX_MD5,
           "evidence_md5": {"bf16": EVIDENCE_BF16_MD5, "mxfp8": EVIDENCE_MX_MD5}}
    (ROOT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "quality"}, indent=1))
    for k, q in quality.items():
        if q:
            v = q.get("video") or q.get("image") or {}
            lt = q.get("latents") or {}
            print(f"{k}: psnr {v.get('psnr')} worst {v.get('psnr_worst')} ssim {v.get('ssim')} "
                  f"lpips {v.get('lpips')} latent cos {lt.get('cosine')} rel_l2 {lt.get('rel_l2')}")


if __name__ == "__main__":
    main()
