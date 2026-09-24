#!/usr/bin/env python3
"""Turn a caching sweep's run logs into the paper's step-caching table rows.

Reads the tree scripts/rerun_caching_official_steps.sh writes
(cclogs/caching-official-steps/<model>/<mode>/rep*.log) and emits one JSON
document holding every column the table needs, so the table and the PSNR figure
are both generated from measurements rather than transcribed by hand.

    python scripts/collect_caching_results.py \
      --root cclogs/caching-official-steps --out cclogs/caching-official-steps/results.json

Columns, and where each comes from:

  skipped        [teacache] stats: {...}  -> skipped_steps / total steps
  loop (ms/step) [steptiming] ...         -> mean ms per scheduler step, which
                                             INCLUDES the steps the cache skipped
                                             (that is what the column means)
  e2e (s)        the run's wall time, from rep*.json
  PSNR / SSIM    scripts/psnr_compare.py, against this model's caching-off run
  signal         probe_calls from the stats line; the per-call cost needs the
                 adaptive run's probe timing and is left null for probe-free modes

Every reported number is a median over repetitions, with the spread kept so a
difference smaller than the run-to-run noise is visible as such.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

STATS_RE = re.compile(r"\[teacache\] stats: (\{.*\})")
TIMING_RE = re.compile(
    r"\[steptiming\] (?P<label>\S+): loop=(?P<loop>[\d.]+)s steps=(?P<steps>\d+) "
    r"mean=(?P<mean>[\d.]+)ms/step median_excl_step0=(?P<median>[\d.]+)ms"
)
MEDIA_SUFFIXES = {".mp4", ".mov", ".webm", ".gif", ".png", ".jpg", ".jpeg"}


def _parse_log(log: Path) -> dict[str, Any]:
    """Pull the stats and timing lines out of one run log."""
    text = log.read_text(encoding="utf-8", errors="replace")
    out: dict[str, Any] = {}

    stats = STATS_RE.findall(text)
    if stats:
        # Python dict repr, not JSON: single quotes and None.
        raw = stats[-1].replace("'", '"').replace("None", "null")
        raw = raw.replace("True", "true").replace("False", "false")
        try:
            out["stats"] = json.loads(raw)
        except json.JSONDecodeError:
            out["stats"] = {"unparsed": stats[-1]}

    timing = list(TIMING_RE.finditer(text))
    if timing:
        last = timing[-1]
        out["loop_s"] = float(last.group("loop"))
        out["steps"] = int(last.group("steps"))
        out["loop_ms_per_step"] = float(last.group("mean"))
        out["step_median_ms_excl_step0"] = float(last.group("median"))
    return out


def _median_and_spread(values: list[float]) -> dict[str, Any] | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return {
        "median": round(statistics.median(clean), 3),
        "min": round(min(clean), 3),
        "max": round(max(clean), 3),
        "n": len(clean),
    }


def _collect_mode(mode_dir: Path) -> dict[str, Any]:
    reps: list[dict[str, Any]] = []
    for log in sorted(mode_dir.glob("rep*.log")):
        rep = _parse_log(log)
        meta_path = log.with_suffix(".json")
        if meta_path.exists():
            try:
                rep.update(json.loads(meta_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        rep["log"] = str(log)
        reps.append(rep)

    row: dict[str, Any] = {"mode": mode_dir.name, "reps": reps}
    if not reps:
        return row

    row["e2e_s"] = _median_and_spread([r.get("wall_s") for r in reps])
    row["loop_ms_per_step"] = _median_and_spread([r.get("loop_ms_per_step") for r in reps])
    row["loop_s"] = _median_and_spread([r.get("loop_s") for r in reps])

    stats = next((r["stats"] for r in reps if isinstance(r.get("stats"), dict)), None)
    steps = next((r.get("steps") for r in reps if r.get("steps")), None)
    if stats:
        row["skipped_steps"] = stats.get("skipped_steps")
        row["full_steps"] = stats.get("full_steps")
        row["probe_calls"] = stats.get("probe_calls")
    elif mode_dir.name == "off":
        row["skipped_steps"] = 0
    row["total_steps"] = steps
    failures = [r for r in reps if r.get("exit_code") not in (0, None)]
    if failures:
        row["failed_reps"] = len(failures)
    return row


def collect(root: Path) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        modes = {}
        for mode_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            modes[mode_dir.name] = _collect_mode(mode_dir)
        if not modes:
            continue

        # Speedup is against this model's own caching-off loop, the table's
        # reference row; without it the ratio column has no meaning.
        baseline = modes.get("off", {}).get("loop_ms_per_step")
        if baseline:
            for name, row in modes.items():
                current = row.get("loop_ms_per_step")
                if name != "off" and current and current["median"]:
                    row["loop_speedup"] = round(baseline["median"] / current["median"], 3)
        models[model_dir.name] = {"modes": modes}
    return {"schema": "difflet-caching-official-steps-v1", "models": models}


def _attach_quality(doc: dict[str, Any], root: Path) -> None:
    """Score each cached run against its own model's caching-off output."""
    try:
        from psnr_compare import compare  # same directory
    except ImportError:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from psnr_compare import compare
        except ImportError:
            return

    for model, entry in doc["models"].items():
        off_dir = root / model / "off"
        refs = sorted(p for p in off_dir.glob("rep*.*") if p.suffix.lower() in MEDIA_SUFFIXES)
        if not refs:
            continue
        reference = refs[0]
        for name, row in entry["modes"].items():
            if name == "off":
                row["psnr_db"] = None  # the reference row has no PSNR against itself
                continue
            cands = sorted(
                p for p in (root / model / name).glob("rep*.*")
                if p.suffix.lower() in MEDIA_SUFFIXES
            )
            if not cands:
                continue
            try:
                q = compare(reference, cands[0])
            except SystemExit as exc:  # shape mismatch and friends
                row["quality_error"] = str(exc)
                continue
            row["psnr_db"] = q["psnr_db"]
            row["psnr_min_db"] = q.get("psnr_min_db")
            row["ssim"] = q.get("ssim")


def _print_table(doc: dict[str, Any]) -> None:
    header = f"{'model':<14} {'mode':<10} {'skipped':>9} {'loop ms/step':>13} {'speedup':>8} {'e2e s':>8} {'PSNR dB':>8}"
    print(header)
    print("-" * len(header))
    for model, entry in doc["models"].items():
        for name, row in entry["modes"].items():
            skipped = row.get("skipped_steps")
            total = row.get("total_steps")
            skip_s = f"{skipped}/{total}" if skipped is not None and total else "-"
            loop = row.get("loop_ms_per_step")
            loop_s = f"{loop['median']:.1f}" if loop else "-"
            spd = row.get("loop_speedup")
            spd_s = f"{spd:.2f}x" if spd else ("ref." if name == "off" else "-")
            e2e = row.get("e2e_s")
            e2e_s = f"{e2e['median']:.1f}" if e2e else "-"
            psnr = row.get("psnr_db")
            psnr_s = f"{psnr:.1f}" if psnr else ("ref." if name == "off" else "-")
            print(f"{model:<14} {name:<10} {skip_s:>9} {loop_s:>13} {spd_s:>8} {e2e_s:>8} {psnr_s:>8}")


MODE_LABELS = {
    "off": "caching off",
    "cadence2": "fixed cadence 2",
    "online": "online-delta",
    "adaptive": "calibrated adaptive",
}
MODEL_LABELS = {
    "flux": "FLUX.1-dev",
    "qwen": "Qwen-Image",
    "wan": "Wan 2.1 14B",
    "hunyuan": "HunyuanVideo",
    "ltx2": "LTX-2",
}
MODE_ORDER = ["off", "cadence2", "online", "adaptive"]


def _print_latex(doc: dict[str, Any]) -> None:
    """Emit the table body, in the column order tab:teacache already uses."""
    print("\n% --- generated by scripts/collect_caching_results.py; do not hand-edit ---")
    print("% model & mode & skipped & signal (ms/call) & loop (ms/step) & e2e (s) & PSNR (dB)")
    for model, entry in doc["models"].items():
        modes = entry["modes"]
        present = [m for m in MODE_ORDER if m in modes] + [
            m for m in modes if m not in MODE_ORDER
        ]
        label = MODEL_LABELS.get(model, model)
        print(f"\\multirow{{{len(present)}}}{{*}}{{{label}}}")
        for name in present:
            row = modes[name]
            skipped, total = row.get("skipped_steps"), row.get("total_steps")
            skip_s = f"{skipped}/{total}" if skipped is not None and total else "--"
            loop = row.get("loop_ms_per_step")
            if loop:
                spd = row.get("loop_speedup")
                loop_s = f"{loop['median']:.0f}"
                if spd:
                    loop_s = f"\\textbf{{{loop['median']:.0f}}} ({spd:.2f}$\\times$)"
            else:
                loop_s = "--"
            e2e = row.get("e2e_s")
            e2e_s = f"{e2e['median']:.0f}" if e2e else "--"
            psnr = row.get("psnr_db")
            psnr_s = "ref." if name == "off" else (f"{psnr:.1f}" if psnr else "--")
            probe = row.get("probe_calls") or 0
            signal_s = "--" if not probe else "see log"
            print(
                f" & {MODE_LABELS.get(name, name):<20} & {skip_s:>7} & {signal_s:>6} "
                f"& {loop_s} & {e2e_s} & {psnr_s} \\\\"
            )
        print("\\midrule")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("cclogs/caching-official-steps"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-quality", action="store_true", help="skip PSNR/SSIM scoring")
    p.add_argument("--latex", action="store_true", help="also emit the table body as LaTeX rows")
    args = p.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"no sweep output at {args.root}")

    doc = collect(args.root)
    if not args.no_quality:
        _attach_quality(doc, args.root)

    _print_table(doc)
    if args.latex:
        _print_latex(doc)
    out = args.out or args.root / "results.json"
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
