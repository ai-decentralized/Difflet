"""Collect every number the report page shows from the evidence files into data.json,
and convert the frame sheets to JPEG for the page."""
import glob
import json
import statistics as st
from pathlib import Path

from PIL import Image

B = Path("/home/ubuntu/wan21-bench")
R = B / "report"
WF = Path("/home/ubuntu/.claude/jobs/000cc5a1/tmp/by.json")

runs = [json.loads(l) for l in (B / "runs.jsonl").read_text().splitlines() if l.strip()]


def rank0(r):
    ps = sorted(r["probes"], key=lambda p: int(p.get("rank") or 0))
    return ps[0] if ps else None


def steps_ms(r):
    p = rank0(r)
    return [s * 1e3 for s in p["step_s"][1:]] if p else []


rows = []
for r in runs:
    s = steps_ms(r)
    p = rank0(r)
    rows.append({
        "backend": r["backend"], "name": r["name"], "guidance": r["guidance"],
        "start_utc": r["start_utc"], "wall_s": r["wall_s"], "new_cache_files": r["new_cache_files"],
        "page_cache_gib": r["page_cache_gib_at_start"], "drop": r["drop_page_cache"],
        "step_median_ms": round(st.median(s), 1) if s else None,
        "step_p90_ms": round(sorted(s)[int(0.9 * (len(s) - 1))], 1) if s else None,
        "denoise_s": round(p["denoise_s"], 2) if p else None,
        "vcs": sorted({str(q["env"].get("NEURON_RT_VIRTUAL_CORE_SIZE")) for q in r["probes"]}),
        "procs": len(r["probes"]),
    })


def pooled(backend, g):
    out = []
    for i in (1, 2, 3):
        for r in runs:
            if r["backend"] == backend and r["name"] == f"warm{i}-g{g}":
                out += steps_ms(r)
    return out


def walls(backend, g):
    return [r["wall_s"] for r in runs if r["backend"] == backend and r["name"].startswith("warm")
            and r["name"].endswith(f"g{g}")]


def one(backend, name):
    return next(r["wall_s"] for r in runs if r["backend"] == backend and r["name"] == name)


headline = {}
for b in ("neuron", "trainium"):
    headline[b] = {
        "step_g5_ms": round(st.median(pooled(b, 5)), 1), "step_g5_n": len(pooled(b, 5)),
        "step_g1_ms": round(st.median(pooled(b, 1)), 1), "step_g1_n": len(pooled(b, 1)),
        "warm_g5_s": round(st.mean(walls(b, 5)), 1), "warm_g5_runs": walls(b, 5),
        "warm_g1_s": round(st.mean(walls(b, 1)), 1), "warm_g1_runs": walls(b, 1),
        "cold_g5_s": one(b, "cold-g5"),
    }
headline["neuron"]["first_g5_s"] = one("neuron", "first-g5")
headline["neuron"]["compile_s"] = round(one("neuron", "first-g5") - one("neuron", "warm1-g5"), 1)
headline["trainium"]["compile_s"] = one("trainium", "compile")

series = {}
for b in ("neuron", "trainium"):
    for g in (5, 1):
        r = next(r for r in runs if r["backend"] == b and r["name"] == f"warm1-g{g}")
        series[f"{b}_g{g}"] = [round(x, 2) for x in steps_ms(r)]

by = json.loads(WF.read_text())
quality_metrics = json.loads((B / "quality" / "metrics.json").read_text())
oracle = json.loads((B / "quality" / "oracle" / "compare.json").read_text())

novcs = json.loads(Path(glob.glob(str(B / "vae-vcs-test/novcs/probe/probe-*.json"))[0]).read_text())

# ccop verification (fix), if present
ccop = {}
cv = B / "ccop-verify"
for d in sorted(cv.glob("*-g*")):
    if d.name.startswith("aborted"):
        continue
    ps = sorted(glob.glob(str(d / "probe" / "probe-*.json")))
    if not ps:
        continue
    probes = [json.loads(Path(p).read_text()) for p in ps]
    p0 = sorted(probes, key=lambda p: int(p.get("rank") or 0))[0]
    ccop[d.name] = {"steps_ms": [round(s * 1e3, 2) for s in p0["step_s"][1:]],
                    "denoise_s": p0["denoise_s"],
                    "flags": p0["env"]}
walls_txt = (cv / "walls.txt").read_text().splitlines() if (cv / "walls.txt").exists() else []
for line in walls_txt:
    name, rc, wall = line.split()
    ccop.setdefault(name, {})["wall_s"] = float(wall.split("=")[1])
    ccop[name]["rc"] = rc

data = {
    "rows": rows, "headline": headline, "series": series,
    "profile": by["prof"], "verify": by["verify"], "judges": by["judges"],
    "skeptic": by["skeptic"], "oracle_wf": by["oracle"], "oracle": oracle,
    "metrics": quality_metrics, "qdev": by["qdev"],
    "novcs": {"step_median_ms": round(st.median([s * 1e3 for s in novcs["step_s"][1:]]), 1),
              "env": novcs["env"]},
    "ccop": ccop,
}
(R / "data.json").write_text(json.dumps(data, indent=1, default=str))

img = R / "img"
img.mkdir(exist_ok=True)
for name in ("neuron_warm1-g5", "trainium_warm1-g5", "cpuvae_trainium_warm1-g5",
             "cpuvae_neuron_warm1-g5", "neuron_warm1-g1", "trainium_warm1-g1"):
    Image.open(B / "quality" / "sheets" / f"{name}.png").convert("RGB").save(
        img / f"{name}.jpg", quality=82, optimize=True)
print("ok", (R / "data.json").stat().st_size, sorted(p.name for p in img.iterdir()))
