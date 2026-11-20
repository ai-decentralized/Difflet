"""Lane WAN of the const-max models run: one model's four arms (arms_wan.sh) into results.json.

Per arm, from ROOT/<model>/<arm>/timed.log (stamp.py-timestamped): each request's DiT step median
(n, max), the warm step (the mean of the two requests' medians) and its spread (|r1 - r2| / mean),
the warm e2e (request 2's `request:` time plus the parent's decode + save of warm_2, as
scripts/quant_bench/report.py), the expert lines, each output's md5, whether the final latents
are finite and whether the two requests are bit-identical. Quality: scripts/compare_difflet_runs.py
on request 2 (mp4 per frame: mean PSNR, worst frame, SSIM, LPIPS; latents rel L2 / cosine) for
B, C, D against A and D against C. Ratios B/A, C/A, D/A, D/C of the warm step.

usage: collect_wan.py <ROOT/<model>>   (env: TORCH_DEVICE_BACKEND_AUTOLOAD=0, a CPU taskset)
"""
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d) (.*)$")
REQUEST = re.compile(r"\] request: ([\d.]+) s; to the first step's end ([\d.]+) s .*?step median "
                     r"([\d.]+) ms \(n=(\d+), max ([\d.]+)\)")
SAVED = re.compile(r"saved to \S*/warm_(\d)\.mp4")
ROUTED = re.compile(r"const-max \(experiment E1\): (\d+) of (\d+) self-attentions routed, table (\S+)")


def md5(path: Path) -> str | None:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else None


def arm(d: Path) -> dict:
    out: dict = {"ran": False}
    log = d / "timed.log"
    if not log.exists():
        return out
    lines = []
    for raw in log.read_text(errors="replace").splitlines():
        m = STAMP.match(raw)
        lines.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp(), m.group(2))
                     if m else (None, raw))
    reqs = [m for _, x in lines if (m := REQUEST.search(x))]
    saved = {int(m.group(1)): t for t, x in lines if (m := SAVED.search(x))}
    out["requests"] = [{"request_s": float(m.group(1)), "first_step_end_s": float(m.group(2)),
                        "step_median_ms": float(m.group(3)), "n": int(m.group(4)),
                        "step_max_ms": float(m.group(5))} for m in reqs]
    out["expert_lines"] = [x for _, x in lines if "expert_calls" in x or "step median by expert" in x]
    out["routed"] = sorted({m.groups() for _, x in lines if (m := ROUTED.search(x))})
    out["start_up"] = [x for _, x in lines if "start-up:" in x]
    out["exit"] = next((int(m.group(1)) for _, x in lines if (m := re.search(r"Exit status: (\d+)", x))), None)
    env = (d / "timed.env.txt")
    out["env"] = env.read_text() if env.exists() else None
    if len(reqs) == 2:
        a, b = (float(m.group(3)) for m in reqs)
        out["ran"] = True
        out["step_ms"] = round((a + b) / 2, 1)
        out["spread_pct"] = round(abs(a - b) / ((a + b) / 2) * 100, 3)
        post = saved[2] - saved[1] if {1, 2} <= set(saved) else 0.0
        out["warm_e2e_s"] = round(float(reqs[1].group(1)) + post, 2)
        out["warm_e2e_parts"] = {"request": float(reqs[1].group(1)), "parent decode+save": round(post, 2)}
    out["md5"] = {f"warm_{i}": md5(d / f"warm_{i}.mp4") for i in (1, 2)}
    lat = [d / "work" / "worker_0" / f"latents_req000{i}.pt" for i in (0, 1)]
    out["latents_md5"] = [md5(p) for p in lat]
    if all(p.exists() for p in lat):
        t = [torch.load(p, map_location="cpu") for p in lat]
        out["latents_finite"] = all(bool(torch.isfinite(x.float()).all()) for x in t)
        out["latents_stats"] = [{"shape": list(x.shape), "mean": float(x.float().mean()),
                                 "std": float(x.float().std()), "absmax": float(x.float().abs().max())}
                                for x in t]
        out["requests_bit_identical"] = bool(torch.equal(t[0], t[1]))
    return out


def compare(root: Path, ref: str, cand: str) -> dict | None:
    r, c = root / ref, root / cand
    if not ((r / "warm_2.mp4").exists() and (c / "warm_2.mp4").exists()):
        return None
    out = root / "scores" / f"{cand}_vs_{ref}.json"
    out.parent.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(REPO / "scripts/compare_difflet_runs.py"),
                    "--ref-latents", str(r / "work/worker_0/latents_req0001.pt"),
                    "--cand-latents", str(c / "work/worker_0/latents_req0001.pt"),
                    "--ref-image", str(r / "warm_2.mp4"), "--cand-image", str(c / "warm_2.mp4"),
                    "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
    return json.loads(out.read_text())


def main(root: Path) -> None:
    arms = {a: arm(root / a) for a in "ABCD"}
    step = {a: v.get("step_ms") for a, v in arms.items()}
    ratio = lambda x, y: round(step[x] / step[y], 4) if step[x] and step[y] else None  # noqa: E731
    res = {"model": root.name, "arms": arms,
           "ratios": {"B/A": ratio("B", "A"), "C/A": ratio("C", "A"), "D/A": ratio("D", "A"),
                      "D/C": ratio("D", "C")},
           "quality": {f"{c}_vs_{r}": compare(root, r, c)
                       for r, c in (("A", "B"), ("A", "C"), ("A", "D"), ("C", "D"))}}
    (root / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    for a, v in arms.items():
        print(a, v.get("step_ms"), v.get("spread_pct"), v.get("warm_e2e_s"), v["md5"], v.get("latents_finite"),
              v.get("requests_bit_identical"), v.get("routed"))
    print(res["ratios"])
    for k, q in res["quality"].items():
        if q:
            vid, lat = q.get("video") or {}, q.get("latents") or {}
            print(k, f"psnr {vid.get('psnr')} worst {vid.get('psnr_worst')} ssim {vid.get('ssim')} "
                     f"lpips {vid.get('lpips')} cos {lat.get('cosine')} rel_l2 {lat.get('rel_l2')}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
