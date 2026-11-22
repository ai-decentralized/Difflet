#!/usr/bin/env python3
"""Arm E (docs/plans/2026-11-22-mxfp8-constmax-teacache-trn3.md): one model's timed arms
(scripts/arm_e/run.sh; HunyuanVideo's A_cat too) into OUT/<model>/results.json, by each lane's
2026-11-18 definitions (10af3dcc^:scripts/attn_constmax/{qwen_results,collect_wan,hv1_collect,
hv15_results}.py and the collect.py of flux/ and ltx_2/ under BASE).

Per arm, from timed.log (two identical requests in one process; request 2 is the warm one):
  step_ms            (no TeaCache) the mean of the two requests' step medians as the log prints
                     them; Qwen-Image: the median over both requests' steps 2-50 in
                     step_times.jsonl
  full_step_ms       (TeaCache) the same mean: the median is a full step's time while fewer
                     than half the steps are skipped
  effective_step_ms  request 2's (request - first-step segment - after the last step) / n, the
                     n steps the median is over (steps 2..N); HunyuanVideo-1.5: denoise / N,
                     all its steps
  skipped_steps      (TeaCache) from the ranks' ``[teacache] stats`` lines; Wan 2.2, whose
                     stats cover the low-noise expert only: n - round(loop time / full step)
  warm_e2e_s         Qwen-Image, Wan: request 2 + the parent's decode and save of it (the
                     timestamps of the two "saved" lines); FLUX, HunyuanVideo: request 2 (it
                     decodes inside); LTX-2: request 2, no decode (warm_e2e_with_host_decode_s
                     adds the mp4s' mtime difference); HunyuanVideo-1.5: request 2's encode +
                     denoise + decode
  md5                of the two outputs
  problems           what makes the arm invalid: outputs or skipped steps that differ between
                     the requests, stats lines that differ or are not WORLD per request, half
                     the steps or more skipped
A_1118 is the 11-18 arm A (BASE), re-read by the same code.

usage: collect.py <model>        exits 1 when an arm has problems (env OUT, as run.sh)
       collect.py --self-check   the 11-18 arm A logs must give the plan's baseline table
       collect.py --self-test    the request and stats parsing on synthetic text
"""
import hashlib
import json
import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

BASE = Path("/mnt/nvme/difflet-exp/constmax-models")
OUT = Path(os.environ.get("OUT", "/mnt/nvme/difflet-exp/arm-e"))
WORLD = 4  # ranks printing a stats line per request: tp4; HunyuanVideo-1.5 tp2 x CFG2
#: model -> 11-18 arm A's directory, step_ms and warm_e2e_s (the plan's baseline table)
A_1118 = {"qwen_image": ("qwen_image/A", 304.7, 16.23), "flux": ("flux/bf16", 222.6, 11.55),
          "ltx_2": ("ltx_2/bf16", 1287.1, 74.81), "wan21": ("wan21/A", 10107.5, 526.24),
          "wan22": ("wan22/A", 10115.2, 425.49),
          "hunyuan_video": ("hunyuan_video/A", 18485.6, 1003.04),
          "hunyuan_video_15": ("hunyuan_video_15/A", 9917.5, 564.2)}
STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d) ")
REQ = re.compile(r"request: ([\d.]+) s; to the first step's end ([\d.]+) s .*?step median "
                 r"([\d.]+) ms \(n=(\d+), max ([\d.]+)\); after the last step ([\d.]+) s")
SAVED = re.compile(r"saved to \S*/warm_(\d)\.\w+")
STATS = re.compile(r"\[teacache\] stats: (\{[^}]*\})")


def md5(path: Path) -> str | None:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else None


def requests_of(model: str, log: str) -> list[dict]:
    """Each request's timing as its lane printed it, with the denoise loop the effective step
    is taken over: loop_s for n steps."""
    if model == "hunyuan_video_15":  # the driver's lines: the median and the loop hold every step
        found: dict[int, dict] = {}
        for m in re.finditer(r"\[hv15\] encode request (\d+): ([\d.]+) s", log):
            found.setdefault(int(m[1]), {})["encode_s"] = float(m[2])
        for m in re.finditer(r"\[hv15\] request (\d+): denoise ([\d.]+) s, (\d+) steps, "
                             r"median (\d+) ms", log):
            found.setdefault(int(m[1]), {}).update(
                denoise_s=float(m[2]), steps=int(m[3]), step_median_ms=float(m[4]),
                n=int(m[3]), loop_s=float(m[2]))
        for m in re.finditer(r"\[hv15\] request (\d+): decode ([\d.]+) s", log):
            found.setdefault(int(m[1]), {})["decode_s"] = float(m[2])
        reqs = [found[k] for k in sorted(found) if len(found[k]) == 7]
    else:  # the median is over steps 2..N: the loop is the request without both ends
        keys = ("request_s", "first_step_end_s", "step_median_ms", "n", "step_max_ms",
                "after_last_step_s")
        reqs = [dict(zip(keys, map(float, m.groups()))) for m in REQ.finditer(log)]
        for r in reqs:
            r["n"] = int(r["n"])
            r["steps"] = r["n"] + 1
            r["loop_s"] = round(r["request_s"] - r["first_step_end_s"] - r["after_last_step_s"], 2)
    for r in reqs:
        r["effective_step_ms"] = round(r["loop_s"] * 1e3 / r["n"], 1)
    return reqs


def teacache(model: str, log: str, reqs: list[dict]) -> list[str]:
    """Sets each request's skipped steps from the ranks' stats lines (log order: WORLD lines per
    request) and returns what makes the arm invalid."""
    stats = STATS.findall(log)
    problems = []
    if len(stats) != WORLD * len(reqs):
        problems.append(f"{len(stats)} [teacache] stats lines, not {WORLD} ranks x "
                        f"{len(reqs)} requests")
    # The skip decisions must agree; a float like last_delta_estimate may differ in its last digit.
    counts = {re.sub(r"'last_delta_estimate': [^,}]*", "", s) for s in stats}
    if len(counts) > 1:
        problems.append(f"[teacache] stats differ across ranks or requests: {sorted(set(stats))}")
    for i, r in enumerate(reqs):
        mine = stats[i * WORLD:(i + 1) * WORLD]
        r["stats"] = sorted(set(mine))
        if model == "wan22":  # its stats restart at the expert switch: the low-noise stage only
            r["skipped_steps"] = r["n"] - round(r["loop_s"] * 1e3 / r["step_median_ms"])
            r["skipped_steps_from"] = "derived: n - round(loop time / full step)"
        elif mine:
            r["skipped_steps"] = int(re.search(r"'skipped_steps': (\d+)", mine[0])[1])
        if r.get("skipped_steps", 0) * 2 >= r["n"]:
            problems.append(f"request {i + 1}: {r['skipped_steps']} of {r['n']} steps skipped: "
                            "the step median is not a full step")
    return problems


def qwen_pooled_step_ms(path: Path) -> float | None:
    """11-18's Qwen-Image step: the median over both requests' steps 2-50 in step_times.jsonl.
    Every rank appends each step (per step: the median over the ranks); a request starts where
    the step index drops."""
    runs: list[dict] = []
    prev = None
    for r in map(json.loads, filter(str.strip, path.read_text().splitlines())):
        if prev is None or r["step"] < prev:
            runs.append({})
        runs[-1].setdefault(r["step"], []).append(r["step_s"])
        prev = r["step"]
    steps = [[statistics.median(run[k]) for k in sorted(run)] for run in runs if len(run) == 50]
    return statistics.median(s * 1e3 for run in steps for s in run[1:]) if len(steps) == 2 else None


def parent_decode_save_s(log: str) -> float | None:
    """The stamp.py time between the two "saved" lines: the parent decodes and saves request 1,
    then request 2, after the workers. None without both stamps after the last request line."""
    def stamp(line):
        m = STAMP.match(line)
        return datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S.%f").timestamp() if m else None

    lines = log.splitlines()
    saved = {int(m[1]): stamp(x) for x in lines if (m := SAVED.search(x))}
    last = [stamp(x) for x in lines if REQ.search(x)][-1]
    if None in (saved.get(1), saved.get(2), last) or min(saved.values()) <= last:
        return None
    return round(saved[2] - saved[1], 2)


def arm(model: str, d: Path, tc: bool = False) -> dict:
    log = (d / "timed.log").read_text(errors="replace")
    reqs = requests_of(model, log)
    ext = "png" if model in ("qwen_image", "flux") else "mp4"
    outputs = [d / f"warm_{i}.{ext}" for i in (1, 2)]
    env = d / "timed.env.txt"
    out = {"dir": str(d), "env": env.read_text().splitlines()[:3] if env.exists() else None,
           "requests": reqs, "md5": [md5(p) for p in outputs], "problems": []}
    problems = out["problems"]
    if len(reqs) != 2:
        problems.append(f"{len(reqs)} complete requests in timed.log, not 2")
        return out
    if out["md5"][0] is None or out["md5"][0] != out["md5"][1]:
        problems.append(f"the two requests' outputs differ or are missing: md5 {out['md5']}")
    step = statistics.mean(r["step_median_ms"] for r in reqs)
    if tc:
        problems += teacache(model, log, reqs)
        if reqs[0].get("skipped_steps") != reqs[1].get("skipped_steps"):
            problems.append("the two requests skipped different steps")
        out["skipped_steps"], out["steps"] = reqs[1].get("skipped_steps"), reqs[1]["steps"]
    elif model == "qwen_image" and (d / "step_times.jsonl").exists():
        step = qwen_pooled_step_ms(d / "step_times.jsonl") or step
    out["full_step_ms" if tc else "step_ms"] = round(step, 2)
    out["effective_step_ms"] = reqs[1]["effective_step_ms"]
    last = reqs[1]
    if model == "hunyuan_video_15":
        out["warm_e2e_s"] = round(last["encode_s"] + last["denoise_s"] + last["decode_s"], 1)
    elif model in ("qwen_image", "wan21", "wan22"):
        post = out["parent_decode_save_s"] = parent_decode_save_s(log)
        if post is None:
            problems.append("no timestamped 'saved' lines after the requests: no warm e2e")
        else:
            out["warm_e2e_s"] = round(last["request_s"] + post, 2)
    else:
        out["warm_e2e_s"] = last["request_s"]
        if model == "ltx_2" and all(p.exists() for p in outputs):
            # the host decodes request 1, then request 2: the mtime difference is decode 2
            out["warm_e2e_with_host_decode_s"] = round(
                last["request_s"] + outputs[1].stat().st_mtime - outputs[0].stat().st_mtime, 2)
    return out


def self_check() -> int:
    bad = 0
    for model, (d, step, e2e) in A_1118.items():
        got = arm(model, BASE / d)
        ok = (abs(got.get("step_ms", 0) - step) < 0.051
              and abs(got.get("warm_e2e_s", 0) - e2e) < 0.0051)
        bad += not ok
        print(f"{model:17s} step_ms {got.get('step_ms')} (plan {step})  warm_e2e_s "
              f"{got.get('warm_e2e_s')} (plan {e2e})  {'ok' if ok else 'MISMATCH'}")
    return 1 if bad else 0


def self_test() -> None:
    stat = ("[teacache] stats: {'full_steps': 30, 'skipped_steps': 20, 'probe_calls': 0, "
            "'last_delta_estimate': 0.01, 'cache_initialized': True}")
    req = ("2026-11-22 10:00:00.0 [flux] request: 6.80 s; to the first step's end 0.50 s (prompt "
           "encode + step 1); step median 200.0 ms (n=49, max 210.0); after the last step 0.30 s")
    log = "\n".join([*[stat] * 4, req, *[stat] * 4, req])
    reqs = requests_of("flux", log)
    assert [(r["n"], r["steps"], r["loop_s"], r["effective_step_ms"]) for r in reqs] == [
        (49, 50, 6.0, 122.4)] * 2, reqs
    assert teacache("flux", log, reqs) == [] and [r["skipped_steps"] for r in reqs] == [20, 20]

    def problems(text, model="flux"):
        return " | ".join(teacache(model, text, requests_of(model, text)))

    skipped = "'skipped_steps': 20"
    assert "differ across ranks" in problems(log.replace(skipped, "'skipped_steps': 19", 1))
    assert "7 [teacache] stats lines" in problems(log.replace(stat + "\n", "", 1))
    assert "0 [teacache] stats lines" in problems(req + "\n" + req)
    assert "not a full step" in problems(log.replace(skipped, "'skipped_steps': 25"))
    # Wan 2.2: 39 steps in 250 s at 10 s a full step = 25 full, 14 skipped, whatever the stats say
    wan = "\n".join([*[stat.replace("30", "17").replace("20", "10")] * 4,
                     "[wan] request: 261.00 s; to the first step's end 11.00 s (prompt encode + "
                     "step 1); step median 10000.0 ms (n=39, max 10100.0); after the last step "
                     "0.00 s"] * 2)
    reqs = requests_of("wan22", wan)
    assert teacache("wan22", wan, reqs) == [] and [r["skipped_steps"] for r in reqs] == [14, 14]
    hv15 = "\n".join(f"[hv15] encode request {i}: 10.0 s (prompt + negative)\n"
                     f"[hv15] request {i}: denoise {d} s, 50 steps, median 9900 ms (first 9 ms)\n"
                     f"[hv15] request {i}: decode 60.0 s; [difflet] video saved to /x/warm_1.mp4"
                     for i, d in ((0, "600.0"), (1, "330.0")))
    reqs = requests_of("hunyuan_video_15", hv15 + "\n[hv15] encode request 2: 1.0 s")
    assert [(r["n"], r["effective_step_ms"], r["decode_s"]) for r in reqs] == [
        (50, 12000.0, 60.0), (50, 6600.0, 60.0)], reqs
    stamped = "\n".join([req, "2026-11-22 10:00:20.5 [vae] image saved to /x/warm_1.png",
                         "2026-11-22 10:00:21.1 [vae] image saved to /x/warm_2.png"])
    assert parent_decode_save_s(stamped) == 0.6 and parent_decode_save_s(req) is None
    print("self-test ok")


def main(model: str) -> int:
    root = OUT / model
    res = {"model": model, "arms": {name: arm(model, root / name, tc=name.startswith("E"))
                                    for name in ("A_cat", "E1", "E2")
                                    if (root / name / "timed.log").exists()},
           "A_1118": arm(model, BASE / A_1118[model][0])}
    if not res["arms"]:
        print(f"no timed.log under {root}/{{A_cat,E1,E2}}", file=sys.stderr)
        return 1
    (root / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    for name, r in {**res["arms"], "A_1118": res["A_1118"]}.items():
        print(name, {k: v for k, v in r.items() if k not in ("dir", "env", "requests")})
    return 1 if any(r["problems"] for r in res["arms"].values()) else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        sys.exit(self_check())
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    if len(sys.argv) != 2 or sys.argv[1] not in A_1118:
        sys.exit(f"usage: collect.py <{'|'.join(A_1118)}> | --self-check | --self-test")
    sys.exit(main(sys.argv[1]))
