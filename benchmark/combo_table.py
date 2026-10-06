"""Best-combination search table: one row per combo cell measured by
benchmark/trn2/run_combo.sh, plus the per-metric winners.

    DIFFLET_BENCH_DEVICE=trn2combo python -m benchmark.combo_table --model flux_1_dev

Columns: per DiT call = real-loop mean gap between DiT calls (step_realloop);
DiT calls = steps minus TeaCache skips (x2 under sequential CFG); avg per
denoise step = denoise loop wall / steps (what layout and TeaCache buy
together); denoise loop = first DiT call to last; resident = in-process generate wall, model loaded
(median of step_realloop --generates); warm e2e = fresh-process CLI generate
with a warm page cache (median of 3), split into weight load and the rest;
PSNR/SSIM = this cell's output vs the tp4 cell's (same seed, same prompt).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from benchmark.models import (CFG_TRACK, CFG_TRACK_HOSTVAE, COMBO_LABELS, MATRIX, RING_STUDY,
                              json_path, logs_dir, reference_label, resolve)


def _row(slug: str, label: str) -> dict | None:
    cfg = resolve(slug, label)
    p = Path(json_path(cfg.config_slug))
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    st = d.get("step_latency") or {}
    tc = d.get("teacache") or {}
    # sequential CFG calls the DiT twice per step: prefer the measured count
    calls = d.get("dit_calls") or tc.get("dit_calls") or (cfg.steps if not tc else None)
    resident = d.get("resident_generate_s") or []
    warm = d.get("e2e_warm") or {}
    bd = d.get("e2e_warm_breakdown") or {}
    loop = d.get("loop_wall_s")
    return {
        "label": label, "desc": cfg.config_label.split(";")[0],
        "step_ms": st.get("mean") and st["mean"] * 1000, "n": st.get("n"),
        "calls": calls, "steps": cfg.steps,
        "resident_s": statistics.median(resident) if resident else None,
        "warm_s": warm.get("median"), "warm_n": warm.get("n"),
        "load_s": bd.get("weights_load_total_s"),
        "parity": _parity(slug, label, cfg),
        "status": d.get("status"), "loop_s": loop,
        # denoise loop / steps: the average cost of one denoise step, skipped
        # steps (TeaCache) and both CFG branches included
        "step_avg_ms": d.get("loop_step_ms") or (loop and loop / cfg.steps * 1000),
        "blocked": d.get("blocked_reason"),
    }


def ref_label(slug: str, label: str) -> str:
    """The same-workload, same-decoder reference (models.reference_label)."""
    cfg = resolve(slug, label)
    return reference_label(label, cfg.guidance_scale != MATRIX[slug].guidance_scale)


def _parity(slug: str, label: str, cfg) -> dict | None:
    ref = ref_label(slug, label)
    if label == ref:
        return {"identical": True}
    from benchmark.adapters.trainium import spec_slug
    from benchmark.output_parity import compare
    ext = ".png" if cfg.output_kind == "image" else ".mp4"
    logs = Path(logs_dir())
    a = logs / f"{spec_slug(resolve(slug, ref))}_out{ext}"
    b = logs / f"{spec_slug(cfg)}_out{ext}"
    if not (a.exists() and b.exists()):
        return None
    try:
        return compare(a, b)
    except Exception as exc:  # report, never crash the table
        return {"error": type(exc).__name__}


def _fmt(x, f="{:.1f}"):
    return "—" if x is None else f.format(x)


def _psnr(r: dict | None) -> str:
    if not r or r.get("error"):
        return "—"
    if r.get("identical"):
        return "ref" if r.get("psnr_db") is None else "identical"
    s = "∞" if r.get("psnr_db") == float("inf") else f"{r['psnr_db']:.1f} dB"
    return s + (f" / {r['ssim']:.3f}" if r.get("ssim") is not None else "")


def table(slug: str) -> list[str]:
    rows = [r for r in (_row(slug, l) for l in [*COMBO_LABELS, *CFG_TRACK, *CFG_TRACK_HOSTVAE, *RING_STUDY]) if r]
    L = ["| label | configuration | per DiT call (ms) | DiT calls | avg per denoise step (ms) | "
         "denoise loop (s) | resident (s) | warm e2e (s) | load (s) | PSNR vs ref |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] == "blocked":
            L.append(f"| `{r['label']}` | {r['desc']} | **BLOCKED** — {r['blocked']} |||||||||")
            continue
        L.append(f"| `{r['label']}` | {r['desc']} | {_fmt(r['step_ms'])} | "
                 f"{_fmt(r['calls'], '{}')}/{r['steps']} | {_fmt(r['step_avg_ms'])} | "
                 f"{_fmt(r['loop_s'], '{:.2f}')} | "
                 f"{_fmt(r['resident_s'], '{:.2f}')} | "
                 f"{_fmt(r['warm_s'])} | {_fmt(r['load_s'])} | {_psnr(r['parity'])}"
                 f"{'' if ref_label(slug, r['label']) == 'tp4' else ' (vs ' + ref_label(slug, r['label']) + ')'} |")
    return L


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    a = p.parse_args()
    print("\n".join(table(a.model)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
