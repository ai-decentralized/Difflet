"""Best-combination search table: one row per combo cell measured by
benchmark/trn2/run_combo.sh, plus the per-metric winners.

    DIFFLET_BENCH_DEVICE=trn2combo python -m benchmark.combo_table --model flux_1_dev

Columns: per-step = real-loop mean per DiT call (step_realloop); DiT calls =
steps minus TeaCache skips; loop/step = resident denoise loop wall / steps
(what TeaCache buys); resident = in-process generate wall, model loaded
(median of step_realloop --generates); warm e2e = fresh-process CLI generate
with a warm page cache (median of 3), split into weight load and the rest;
PSNR/SSIM = this cell's output vs the tp4 cell's (same seed, same prompt).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from benchmark.models import COMBO_LABELS, logs_dir, json_path, resolve


def _row(slug: str, label: str) -> dict | None:
    cfg = resolve(slug, label)
    p = Path(json_path(cfg.config_slug))
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    st = d.get("step_latency") or {}
    tc = d.get("teacache") or {}
    calls = tc.get("dit_calls") or (cfg.steps if not tc else None)
    resident = d.get("resident_generate_s") or []
    warm = d.get("e2e_warm") or {}
    bd = d.get("e2e_warm_breakdown") or {}
    loop = st.get("generate_wall_s") or st.get("loop_wall_s")
    return {
        "label": label, "desc": cfg.config_label.split(";")[0],
        "step_ms": st.get("mean") and st["mean"] * 1000, "n": st.get("n"),
        "calls": calls, "steps": cfg.steps,
        "resident_s": statistics.median(resident) if resident else None,
        "warm_s": warm.get("median"), "warm_n": warm.get("n"),
        "load_s": bd.get("weights_load_total_s"),
        "parity": _parity(slug, label, cfg),
        "status": d.get("status"), "loop_s": loop,
        "blocked": d.get("blocked_reason"),
    }


def _parity(slug: str, label: str, cfg) -> dict | None:
    if label == "tp4":
        return {"identical": True}
    from benchmark.adapters.trainium import spec_slug
    from benchmark.output_parity import compare
    ext = ".png" if cfg.output_kind == "image" else ".mp4"
    logs = Path(logs_dir())
    a = logs / f"{spec_slug(resolve(slug, 'tp4'))}_out{ext}"
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
    rows = [r for r in (_row(slug, l) for l in COMBO_LABELS) if r]
    L = ["| label | configuration | per-step (ms) | DiT calls | resident (s) | "
         "warm e2e (s) | load (s) | PSNR / SSIM vs tp4 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] == "blocked":
            L.append(f"| `{r['label']}` | {r['desc']} | **BLOCKED (HBM)** — {r['blocked']} |||||||")
            continue
        L.append(f"| `{r['label']}` | {r['desc']} | {_fmt(r['step_ms'])} | "
                 f"{_fmt(r['calls'], '{}')}/{r['steps']} | {_fmt(r['resident_s'], '{:.2f}')} | "
                 f"{_fmt(r['warm_s'])} | {_fmt(r['load_s'])} | {_psnr(r['parity'])} |")
    return L


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    a = p.parse_args()
    print("\n".join(table(a.model)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
