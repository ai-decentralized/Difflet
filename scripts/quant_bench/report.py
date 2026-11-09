"""Before/after quantization table from bench.sh's logs (evidence sections 11 and 12).

Per key and arm (bf16, mx), from the timestamped logs under ROOT/<key>/<arm>/:
  compile_s    wall of `compile` into a fresh cache (/usr/bin/time), with its bf16-copy, quantize
               and DiT warmup parts
  cold_s       the lower wall of the two fresh-process `generate` runs (both kept in cold_all)
  warm_s       the second request's latency inside one process: the `request:` time, plus the
               decode the parent runs after the DP router's workers (saved-file timestamps), or
               for HunyuanVideo-1.5's staged CLI encode + denoise + decode of request 1
  step_ms      the DiT step median of that warm request
  hbm_gib      the peak device memory per rank over the warm process (ROOT/hbm_samples.txt from
               hbm_sampler.py), on the lane's cores (ROOT/<key>/lane.txt, or --lanes key=cores)
  weights_gib  the DiT weights per rank where the log prints them
Writes ROOT/report.json and ROOT/report.md (also printed). Keys without results are skipped, so an
empty root gives no rows.

usage: report.py [--root ROOT] [--lanes key=cores ...] [key ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

KEYS = ["flux", "qwen_image", "ltx_2", "wan21", "wan22", "hunyuan_video", "hunyuan_video_15",
        "wan21_official", "wan22_official", "hv15_official"]
#: HunyuanVideo-1.5's staged CLI times encode, denoise and decode per request itself.
STAGED = {"hunyuan_video_15", "hv15_official"}

STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d) (.*)$")
ELAPSED = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)")
EXIT = re.compile(r"^exit (\d+)$")
REQUEST = re.compile(r"\] request: ([\d.]+) s; to the first step's end ([\d.]+) s .*?step median "
                     r"([\d.]+) ms.*?after the last step ([\d.]+) s")
HV15_ENC = re.compile(r"\[hv15\] encode request (\d+): ([\d.]+) s")
HV15_DEN = re.compile(r"\[hv15\] request (\d+): denoise ([\d.]+) s, \d+ steps, median ([\d.]+) ms")
HV15_DEC = re.compile(r"\[hv15\] request (\d+): decode ([\d.]+) s")
SAVED = re.compile(r"saved to \S*/(warm|cold)_(\d)\.(png|mp4)")
BF16_COPY = re.compile(r"bf16 copy of (\w+): \S+ \(([\d.]+) s\)")
QUANT_DIR = re.compile(r"\] (MXFP8 transformer|FP8 DiT|MXFP8 \w+)[^:]*: (\S+/quantized/\S+)")
WARMUP = re.compile(r"'warmup forward': ([\d.]+)")
WEIGHTS = re.compile(r"DiT weights per rank: ([\d.]+) GiB")
FS_INPUTS = re.compile(r"File system inputs: (\d+)")


def core_set(spec: str) -> set[str]:
    """``"0-3,16"`` -> {"0", "1", "2", "3", "16"}."""
    cores: set[str] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cores.update(str(c) for c in range(int(lo), int(hi or lo) + 1))
    return cores


def wall(s: str) -> float:
    parts = [float(p) for p in s.split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def read(log: Path):
    """([(unix time, text)], wall seconds, exit code, file-system input GB)."""
    lines, el, rc, fs = [], None, None, None
    if not log.exists():
        return lines, el, rc, fs
    for raw in log.read_text(errors="replace").splitlines():
        m = STAMP.match(raw)
        t, text = ((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp(), m.group(2))
                   if m else (None, raw))
        lines.append((t, text))
        if (e := ELAPSED.search(text)):
            el = wall(e.group(1))
        if (x := EXIT.match(text.strip())):
            rc = int(x.group(1))
        if (f := FS_INPUTS.search(text)):
            fs = round(int(f.group(1)) * 512 / 1e9, 2)  # 512-byte blocks
    return lines, el, rc, fs


def window(lines):
    times = [t for t, _ in lines if t is not None]
    return (min(times), max(times)) if times else (None, None)


def hbm_peak(cores: set[str], lo: float, hi: float, samples) -> float | None:
    """The peak per-rank device bytes (GiB) on ``cores`` within [lo, hi]."""
    vals = [b for t, _pid, core, b in samples if lo <= t <= hi and core in cores]
    return max(vals) / 2**30 if vals else None


def arm_metrics(root: Path, key: str, arm: str, cores: set[str] | None, samples) -> dict:
    d = root / key / arm
    out: dict = {"fs_inputs_gb": {}}
    lines, el, rc, fs = read(d / "compile.log")
    out["compile_exit"], out["compile_s"], out["fs_inputs_gb"]["compile"] = rc, el, fs
    stamps: dict = {}
    for t, text in lines:
        if (m := BF16_COPY.search(text)):
            out["bf16_copy_s"] = out.get("bf16_copy_s", 0.0) + float(m.group(2))
            stamps.setdefault("copy_end", []).append(t)
        if QUANT_DIR.search(text):
            stamps.setdefault("quant_line", []).append(t)
        if (m := WARMUP.search(text)):
            out.setdefault("warmup_s", []).append(float(m.group(1)))
    first = window(lines)[0]
    # The quantized directory's line is printed once it exists (built or reused); the build
    # starts after the bf16 copy where there is one, else at the process start (the first stamped
    # line). An upper bound: it includes what the CLI does between the two (loading the copy).
    if arm == "mx" and stamps.get("quant_line") and first:
        start = max(stamps["copy_end"]) if stamps.get("copy_end") else first
        out["quantize_s_upper"] = round(max(stamps["quant_line"]) - start, 1)
    cold = []
    for i in (1, 2):
        _, cel, crc, cfs = read(d / f"cold_{i}.log")
        cold.append(cel if crc == 0 else None)
        out["fs_inputs_gb"][f"cold_{i}"] = cfs
    out["cold_all"] = cold
    ok = [c for c in cold if c is not None]
    out["cold_s"] = min(ok) if ok else None
    wl, wel, wrc, wfs = read(d / "warm.log")
    out["warm_exit"], out["warm_wall_s"], out["fs_inputs_gb"]["warm"] = wrc, wel, wfs
    reqs = [(t, REQUEST.search(x)) for t, x in wl if REQUEST.search(x)]
    saved = {int(m.group(2)): t for t, x in wl if (m := SAVED.search(x)) and m.group(1) == "warm"}
    if key in STAGED:
        enc = {int(m.group(1)): float(m.group(2)) for _, x in wl if (m := HV15_ENC.search(x))}
        den = {int(m.group(1)): (float(m.group(2)), float(m.group(3)))
               for _, x in wl if (m := HV15_DEN.search(x))}
        dec = {int(m.group(1)): float(m.group(2)) for _, x in wl if (m := HV15_DEC.search(x))}
        if 1 in enc and 1 in den and 1 in dec:
            out["warm_s"] = round(enc[1] + den[1][0] + dec[1], 2)
            out["warm_parts"] = {"encode": enc[1], "denoise": den[1][0], "decode": dec[1]}
            out["step_ms"] = den[1][1]
    elif len(reqs) >= 2:
        _, m = reqs[1]
        req_s, last_s = float(m.group(1)), float(m.group(4))
        out["step_ms"] = float(m.group(3))
        post = 0.0
        if 1 in saved and 2 in saved and min(saved.values()) > reqs[-1][0]:
            # The DP router: the workers ran both requests, then the parent decoded and saved each.
            post = saved[2] - saved[1]
            out["warm_parts"] = {"request": req_s, "parent decode+save": round(post, 2)}
        else:
            out["warm_parts"] = {"request": req_s, "after last step (in request)": last_s}
        out["warm_s"] = round(req_s + post, 2)
    lo, hi = window(wl)
    if lo and cores:
        out["hbm_gib"] = hbm_peak(cores, lo, hi, samples)
    for _, x in wl:
        if (m := WEIGHTS.search(x)):
            out["weights_gib"] = float(m.group(1))
    sz = d / "cache_size.txt"
    if sz.exists():
        text = sz.read_text().splitlines()
        out["cache"] = text[0].split()[0] if text else None
        out["cache_parts"] = {Path(p).name: s for s, p in
                              (ln.split("\t") for ln in text[1:] if "\t" in ln)}
    return out


def fmt(v, unit=""):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:,.1f}{unit}" if v >= 10 else f"{v:.2f}{unit}"
    return f"{v}{unit}"


def ratio(a, b):
    return f"{b / a:.2f}x" if a and b else "–"


def lane_cores(root: Path, key: str, lanes: dict[str, str]) -> set[str] | None:
    if key in lanes:
        return core_set(lanes[key])
    lane = root / key / "lane.txt"
    if lane.exists():
        for ln in lane.read_text().splitlines():
            name, _, value = ln.partition(" ")
            if name == "cores":
                return core_set(value.strip())
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path,
                   default=Path(os.environ.get("BENCH_ROOT", "/mnt/nvme/difflet-bench/results")))
    p.add_argument("--lanes", nargs="*", default=[], metavar="KEY=CORES",
                   help="the lane's cores per key, over ROOT/<key>/lane.txt")
    p.add_argument("keys", nargs="*")
    args = p.parse_args()
    lanes = dict(item.split("=", 1) for item in args.lanes)
    samples = []
    hbm = args.root / "hbm_samples.txt"
    if hbm.exists():
        for ln in hbm.read_text().splitlines():
            t, pid, core, b = ln.split()
            samples.append((float(t), pid, core, int(b)))
    found = sorted(d.name for d in args.root.iterdir() if (d / "bf16").is_dir()
                   or (d / "mx").is_dir()) if args.root.is_dir() else []
    keys = args.keys or [k for k in KEYS if k in found] + [k for k in found if k not in KEYS]
    report = {}
    for key in keys:
        if not (args.root / key).is_dir():
            continue
        cores = lane_cores(args.root, key, lanes)
        report[key] = {arm: arm_metrics(args.root, key, arm, cores, samples)
                       for arm in ("bf16", "mx")}
    rows = [("compile (fresh cache)", "compile_s", "s"), ("e2e cold (lower of 2)", "cold_s", "s"),
            ("e2e warm (request 2)", "warm_s", "s"), ("DiT step median", "step_ms", " ms"),
            ("peak HBM per rank", "hbm_gib", " GiB"), ("DiT weights per rank", "weights_gib", " GiB")]
    md = []
    for key, arms in report.items():
        b, m = arms["bf16"], arms["mx"]
        md += [f"\n### {key}", "| | bf16 | MXFP8 | MXFP8/bf16 |", "|---|---|---|---|"]
        md += [f"| {name} | {fmt(b.get(k), u)} | {fmt(m.get(k), u)} | {ratio(b.get(k), m.get(k))} |"
               for name, k, u in rows]
        md.append(f"| cold runs | {b.get('cold_all')} | {m.get('cold_all')} | |")
        md.append(f"| compile parts | copy {fmt(b.get('bf16_copy_s'))}, warmup {b.get('warmup_s')} "
                  f"| copy {fmt(m.get('bf16_copy_s'))}, quantize <= {fmt(m.get('quantize_s_upper'))}, "
                  f"warmup {m.get('warmup_s')} | |")
        md.append(f"| warm parts | {b.get('warm_parts')} | {m.get('warm_parts')} | |")
        md.append(f"| file system inputs (GB) | {b.get('fs_inputs_gb')} | {m.get('fs_inputs_gb')} | |")
        md.append(f"| cache | {b.get('cache')} {b.get('cache_parts')} | "
                  f"{m.get('cache')} {m.get('cache_parts')} | |")
    if args.root.is_dir():
        (args.root / "report.json").write_text(json.dumps(report, indent=1))
        (args.root / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
