#!/usr/bin/env python3
"""Render one model's evidence-doc section from the per-model verification outputs.

    PYTHONPATH=$PWD python scripts/ptq_model_section.py <bf16-slug> [--evidence-dir DIR] [--results-dir DIR]

Reads benchmark/trn2/<slug>{,_fp8}.json (the harness reports),
artifacts/verification-2026-10-02/ptq-all/<slug>/{compare_fp8_vs_bf16.json,
logs/quantize.log,store_entries.txt} and prints Markdown
tables (performance, quality, quantized checkpoint, store entries). Missing inputs
render as "—" so a partial run still produces a section.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = (("", "bf16"), ("_fp8", "fp8-tensor (W8A8)"))


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _f(value, fmt="{:.1f}") -> str:
    if value is None:
        return "—"
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _stat(block: dict | None, key: str, fmt="{:.1f}", scale=1.0) -> str:
    if not block or block.get(key) is None:
        return "—"
    return fmt.format(float(block[key]) * scale)


def _transformer_load(report: dict) -> str:
    """Cold transformer weight-load seconds: the harness's per-stage load when it has one."""
    breakdown = report.get("e2e_breakdown") or {}
    for stage in breakdown.get("stages") or []:
        name = str(stage.get("stage", ""))
        if "transformer" in name or "denoise" in name or "dit" in name.lower():
            return _f(stage.get("load_s"))
    return _f(report.get("load_seconds"))


def perf_table(reports: dict[str, dict | None]) -> str:
    rows = [
        "| arm | status | compile wall s | e2e cold s | e2e warm s (median; n) | transformer load cold s | "
        "DiT step ms (median / mean; n) | peak device mem GB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, label in ARMS:
        r = reports.get(label)
        if r is None:
            rows.append(f"| {label} | _missing_ | — | — | — | — | — | — |")
            continue
        warm = r.get("e2e_warm") or {}
        step = r.get("step_latency") or {}
        rows.append(
            f"| {label} | {r.get('status', '—')} | {_f(r.get('compile_seconds'))} | "
            f"{_f(r.get('e2e_cold_seconds'))} | {_stat(warm, 'median')} ({warm.get('n', '—')}) | "
            f"{_transformer_load(r)} | {_stat(step, 'median', scale=1000.0)} / {_stat(step, 'mean', scale=1000.0)} "
            f"({step.get('n', '—')}) | {_f(r.get('peak_device_mem_gb'), '{:.2f}')} |"
        )
    return "\n".join(rows)


def ratio_lines(reports: dict[str, dict | None]) -> str:
    base = reports.get("bf16")
    if not base:
        return ""
    out = []
    for _, label in ARMS[1:]:
        r = reports.get(label)
        if not r:
            continue
        parts = []
        for key, name in (("e2e_cold_seconds", "e2e cold"), ("compile_seconds", "compile")):
            if r.get(key) and base.get(key):
                parts.append(f"{name} {float(r[key]) / float(base[key]):.2f}×")
        for key, name in (("e2e_warm", "e2e warm"), ("step_latency", "DiT step")):
            a, b = (r.get(key) or {}).get("median"), (base.get(key) or {}).get("median")
            if a and b:
                parts.append(f"{name} {float(a) / float(b):.3f}×")
        if parts:
            out.append(f"- **{label} vs bf16:** " + ", ".join(parts) + ".")
    return "\n".join(out)


def quality_table(evidence: Path) -> str:
    rows = [
        "| pair | PSNR dB | SSIM | LPIPS (net) | pixel max-abs | pixel rel-L2 | frames × H × W |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm, label in ARMS[1:]:
        cmp = _load(evidence / f"compare{arm}_vs_bf16.json")
        out = (cmp or {}).get("output") or {}
        if not out:
            rows.append(f"| {label} vs bf16 | _missing_ | — | — | — | — | — |")
            continue
        pixel = out.get("pixel") or {}
        lp = out.get("lpips")
        rows.append(
            f"| {label} vs bf16 | {_f(out.get('psnr_db'), '{:.2f}')} | {_f(out.get('ssim'), '{:.4f}')} | "
            f"{_f(lp, '{:.4f}')}{' (' + str(out.get('lpips_net')) + ')' if lp is not None else ''} | "
            f"{_f(pixel.get('max_abs'), '{:.4g}')} | {_f(pixel.get('rel_l2'), '{:.4g}')} | "
            f"{out.get('frames', '—')} × {out.get('height', '—')} × {out.get('width', '—')} |"
        )
    return "\n".join(rows)


def quantize_lines(evidence: Path) -> str:
    text = ""
    try:
        text = (evidence / "logs" / "quantize.log").read_text()
    except OSError:
        return "- quantize log: _missing_"
    out = []
    for m in re.finditer(r"\[quantize\] (\S+): (\S+) -> (\S+)", text):
        out.append(f"- quantized checkpoint `{m.group(1)}` ({m.group(2)}): `{m.group(3)}`")
    for m in re.finditer(r"linears quantized: (\d+)\s+bytes (\d+) -> (\d+)\s+\(([\d.]+)s\)", text):
        n, before, after, secs = m.groups()
        out.append(
            f"- {n} linears quantized; checkpoint bytes {int(before) / 1e9:.2f} GB → {int(after) / 1e9:.2f} GB "
            f"({int(after) / int(before):.3f}×) in {float(secs):.1f} s"
        )
    return "\n".join(out) or "- quantize log present, no summary lines matched"


def store_lines(evidence: Path) -> str:
    try:
        lines = [ln.strip() for ln in (evidence / "store_entries.txt").read_text().splitlines() if ln.strip()]
    except OSError:
        return "_store listing missing_"
    if not lines:
        return "_no store entries matched_"
    return "```\n" + "\n".join(lines) + "\n```"


def render(slug: str, results: Path, evidence: Path) -> str:
    reports = {label: _load(results / f"{slug}{arm}.json") for arm, label in ARMS}
    base = next((r for r in reports.values() if r), {}) or {}
    shape = base.get("shape") or {}
    shape_txt = "×".join(str(shape[k]) for k in ("height", "width", "num_frames") if shape.get(k) is not None)
    parallel = base.get("parallel") or {}
    header = [
        f"### `{slug}` — {base.get('model_id', slug)}",
        "",
        f"Shape {shape_txt or '—'}, steps {base.get('steps', '—')}, tp {parallel.get('tp_degree', '—')} "
        f"(cp {parallel.get('cp_degree', 1)}), seed {base.get('seed', '—')}; revision `{base.get('revision') or 'main'}`; "
        f"reports `benchmark/trn2/{slug}{{,_fp8}}.json`, evidence `{evidence.relative_to(ROOT) if evidence.is_relative_to(ROOT) else evidence}/`.",
        "",
        "**Performance** (harness: `benchmark.bench --iters 1` → compile + step latency, `benchmark.cold_warm_e2e` → "
        "true cold / warm e2e):",
        "",
        perf_table(reports),
        "",
        ratio_lines(reports),
        "",
        "**Quality** vs the bf16 output of the same prompt / seed / steps (`scripts/ptq_compare_outputs.py`):",
        "",
        quality_table(evidence),
        "",
        "**Quantized checkpoint** (`difflet quantize`):",
        "",
        quantize_lines(evidence),
        "",
        "**Shared-store entries** (A4: fp8 shards + quantized copies on disk):",
        "",
        store_lines(evidence),
        "",
    ]
    return "\n".join(header)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("slug")
    p.add_argument("--results-dir", type=Path, default=ROOT / "benchmark" / "trn2")
    p.add_argument("--evidence-dir", type=Path, default=None)
    args = p.parse_args(argv)
    evidence = args.evidence_dir or (ROOT / "artifacts" / "verification-2026-10-02" / "ptq-all" / args.slug)
    sys.stdout.write(render(args.slug, args.results_dir, evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
