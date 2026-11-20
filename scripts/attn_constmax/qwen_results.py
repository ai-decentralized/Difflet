"""Experiment E1, Qwen-Image lane: each arm's timing, md5s and quality into ROOT/results.json.

Per arm (A bf16, B bf16 + const-max, C mxfp8, D mxfp8 + const-max), from qwen_lane.sh's timed run
(ROOT/<arm>/timed.log, timestamped; step_times.jsonl):
  req_step_ms   each request's DiT step median as the CLI prints it (steps 2-50)
  step_ms       the median over both requests' steps 2-50 (step_times.jsonl, one rank's lines)
  spread_pct    |request 1 - request 2| / step_ms, in %
  warm_e2e_s    request 2's `request:` time plus the parent's decode+save of it (report.py's rule)
  md5           of warm_1.png and warm_2.png
and the quality lines score.sh wrote (ROOT/scores/<arm>_vs_<ref>.json). Missing arms are skipped.

usage: qwen_results.py [ROOT]
"""
import hashlib
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme/difflet-exp/constmax-models/qwen_image")
REQ = re.compile(r"\] request: ([\d.]+) s; .*?step median ([\d.]+) ms \(n=(\d+)")
STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d) ")
SAVED = re.compile(r"saved to \S*/warm_(\d)\.png")


def t(line):
    m = STAMP.match(line)
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp() if m else None


def requests_of(steps):
    """step_times.jsonl lines -> one list of step_s per request. Every tp rank appends each step
    (lines interleave: 0,0,0,0,1,1,...); a request starts where the step index drops back to 0.
    Per step, the median over the ranks (they finish together: the step ends in a collective)."""
    runs, prev = [], None
    for r in steps:
        if prev is None or r["step"] < prev:
            runs.append({})
        runs[-1].setdefault(r["step"], []).append(r["step_s"])
        prev = r["step"]
    return [[statistics.median(run[k]) for k in sorted(run)] for run in runs if len(run) == 50]


def arm(a):
    d = ROOT / a
    log = (d / "timed.log").read_text(errors="replace").splitlines() if (d / "timed.log").exists() else []
    reqs = [(t(x), REQ.search(x)) for x in log if REQ.search(x)]
    if len(reqs) < 2:
        return None
    saved = {int(m.group(1)): t(x) for x in log if (m := SAVED.search(x))}
    out = {"req_step_ms": [float(m.group(2)) for _, m in reqs[:2]],
           "req_s": [float(m.group(1)) for _, m in reqs[:2]]}
    post = 0.0
    if 1 in saved and 2 in saved and None not in saved.values() and reqs[-1][0] \
            and min(saved.values()) > reqs[-1][0]:
        post = saved[2] - saved[1]  # the parent decodes and saves after both requests
    out["parent_decode_save_s"] = round(post, 2)
    out["warm_e2e_s"] = round(out["req_s"][1] + post, 2)
    st = d / "step_times.jsonl"
    if st.exists():
        runs = requests_of([json.loads(x) for x in st.read_text().splitlines() if x.strip()])
        if len(runs) == 2:
            pooled = [s * 1e3 for r in runs for s in r[1:]]  # steps 2-50 of both requests
            out["step_ms"] = round(statistics.median(pooled), 1)
            out["step_ms_per_request"] = [round(statistics.median([s * 1e3 for s in r[1:]]), 1)
                                          for r in runs]
    base = out.get("step_ms") or statistics.mean(out["req_step_ms"])
    out.setdefault("step_ms", round(base, 1))
    a1, a2 = out["req_step_ms"]
    out["spread_pct"] = round(100 * abs(a1 - a2) / base, 2)
    out["md5"] = [hashlib.md5((d / f"warm_{i}.png").read_bytes()).hexdigest()
                  if (d / f"warm_{i}.png").exists() else None for i in (1, 2)]
    out["deterministic"] = out["md5"][0] is not None and out["md5"][0] == out["md5"][1]
    return out


def gate(a):
    """The 10-step generate's final latents (ROOT/<arm>/work_smoke/latents.pt): finite, stats."""
    p = ROOT / a / "work_smoke" / "latents.pt"
    if not p.exists():
        return None
    import torch

    x = torch.load(p, map_location="cpu").float()
    return {"shape": list(x.shape), "finite": bool(torch.isfinite(x).all()),
            "mean": round(float(x.mean()), 4), "std": round(float(x.std()), 4),
            "absmax": round(float(x.abs().max()), 3),
            "png_md5": hashlib.md5((ROOT / a / "smoke.png").read_bytes()).hexdigest()}


def main():
    res = {"root": str(ROOT), "model": "Qwen-Image", "shape": "1024x1024, tp4, 50 steps, seed 42, "
           "guidance 7.5, 'a cat sitting on a bench'", "evidence_md5": {
               "bf16": "5478687f25718aa92a77278c68808f8d", "mxfp8": "7737239be780a7877cd8b2ebe32bfb2f"},
           "gate": {a: g for a in "BD" if (g := gate(a)) is not None}, "arms": {}, "quality": {}}
    for a in "ABCD":
        r = arm(a)
        if r is not None:
            res["arms"][a] = r
    s = res["arms"]
    for x, y in (("B", "A"), ("C", "A"), ("D", "A"), ("D", "C")):
        if x in s and y in s:
            res[f"{x}_over_{y}"] = round(s[x]["step_ms"] / s[y]["step_ms"], 3)
    for f in sorted((ROOT / "scores").glob("*_vs_*.json")):
        q = json.loads(f.read_text())
        img, lat = q.get("image") or {}, q.get("latents") or {}
        res["quality"][f.stem] = {"psnr": img.get("psnr"), "ssim": img.get("ssim"),
                                  "lpips": img.get("lpips"), "latent_cos": lat.get("cosine"),
                                  "latent_rel_l2": lat.get("rel_l2"),
                                  "bit_identical": lat.get("bit_identical")}
    (ROOT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
