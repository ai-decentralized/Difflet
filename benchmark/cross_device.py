"""Cross-device table from the per-device result JSONs -- one row per MATRIX model.

    python -m benchmark.cross_device --reference trn2 --device v5e

Reads ``benchmark/<device>/<slug>.json`` for every slug present on both sides
and prints Markdown: e2e cold, e2e warm (fresh process on both), the
resident-process request where the device has one (its trn2 counterpart is
warm e2e minus the warm load, ``compute_and_overhead_s``), DiT per-step
(the shared RealLoopStepTimer rule), the output check, and the ratios.
Every figure is read straight from the JSON the harness wrote; nothing is
typed in by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.models import MATRIX, json_path


def _load(device: str, slug: str) -> dict | None:
    path = Path(json_path(slug, device))
    return json.loads(path.read_text()) if path.exists() else None


def _mean(stats: dict | None) -> float | None:
    return None if not stats else stats.get("mean")


def _f(value: float | None, unit: str = "s", digits: int = 1) -> str:
    if value is None:
        return "—"
    if unit == "ms":
        return f"{value * 1000:.0f} ms"
    return f"{value:.{digits}f} s"


def _ratio(a: float | None, b: float | None) -> str:
    if not a or not b:
        return "—"
    return f"{a / b:.2f}×"


def _output_cell(d: dict) -> str:
    out = d.get("output") or {}
    if out.get("finite") is None:
        return out.get("note") or "—"
    shape = "×".join(str(x) for x in (out.get("shape") or []))
    return f"({shape}) {'✓' if out['finite'] else '✗ non-finite'}"


def rows(reference: str, device: str) -> list[dict]:
    result = []
    for slug in MATRIX:
        ref, dev = _load(reference, slug), _load(device, slug)
        if ref is None or dev is None:
            continue
        ref_load_warm = (ref.get("e2e_warm_breakdown") or {}).get("weights_load_total_s")
        ref_warm = _mean(ref.get("e2e_warm"))
        ref_resident = (ref.get("e2e_warm_breakdown") or {}).get("compute_and_overhead_s")
        if ref_resident is None and ref_warm is not None and ref_load_warm is not None:
            ref_resident = ref_warm - ref_load_warm
        dev_stage = dev.get("stage_seconds") or {}
        result.append({
            "slug": slug, "model_id": ref["model_id"],
            "shape": ref.get("shape"), "steps": ref.get("steps"),
            "ref_cold": ref.get("e2e_cold_seconds"), "dev_cold": dev.get("e2e_cold_seconds"),
            "ref_warm": ref_warm, "dev_warm": _mean(dev.get("e2e_warm")),
            "ref_warm_n": (ref.get("e2e_warm") or {}).get("n"),
            "dev_warm_n": (dev.get("e2e_warm") or {}).get("n"),
            "ref_load_warm": ref_load_warm,
            "dev_load_warm": (dev.get("e2e_warm_breakdown") or {}).get("weights_load_total_s"),
            "ref_resident": ref_resident, "dev_resident": _mean(dev.get("e2e_warm_resident")),
            "dev_natural": _mean(dev.get("e2e_warm_natural")),
            "ref_step": _mean(ref.get("step_latency")), "dev_step": _mean(dev.get("step_latency")),
            "ref_step_n": (ref.get("step_latency") or {}).get("n"),
            "dev_step_n": (dev.get("step_latency") or {}).get("n"),
            "dev_step_natural": _mean(dev.get("step_latency_natural")),
            "dev_encode": dev_stage.get("text_encode"), "dev_denoise": dev_stage.get("denoise"),
            "dev_decode": dev_stage.get("vae_decode"),
            "ref_compile": ref.get("compile_seconds"),
            "dev_mem": dev.get("peak_device_mem_gb"),
            "ref_output": _output_cell(ref), "dev_output": _output_cell(dev),
            "ref_status": ref.get("status"), "dev_status": dev.get("status"),
        })
    return result


def render(reference: str, device: str, table: list[dict]) -> str:
    L: list[str] = []
    a = L.append
    a(f"### {device} vs {reference} — whole request, same MATRIX rows, same protocol")
    a("")
    a("Fresh process on both sides (page cache dropped for cold, warm from the page cache, n as "
      "shown); the resident column is one more request on the process left running "
      f"(`{reference}`: warm e2e minus the warm weight load, its nearest equivalent).")
    a("")
    a(f"| model | shape / steps | {reference} cold | **{device} cold** | {reference} warm | "
      f"**{device} warm** | {reference} warm − load | **{device} resident request** | "
      f"{device} encode / denoise / decode | output ({device}) |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for r in table:
        sh = r["shape"] or {}
        dims = "×".join(str(sh[k]) for k in ("height", "width", "num_frames") if sh.get(k))
        split = " / ".join(_f(r[k]) for k in ("dev_encode", "dev_denoise", "dev_decode"))
        a(f"| {r['slug']} | {dims}, {r['steps']} st | {_f(r['ref_cold'], digits=0)} | "
          f"**{_f(r['dev_cold'], digits=0)}** | {_f(r['ref_warm'])} (n={r['ref_warm_n']}) | "
          f"**{_f(r['dev_warm'])}** (n={r['dev_warm_n']}) | {_f(r['ref_resident'])} | "
          f"**{_f(r['dev_resident'])}** | {split} | {r['dev_output']} |")
    a("")
    a(f"### {device} vs {reference} — DiT per-step (RealLoopStepTimer rule on both)")
    a("")
    a(f"| model | {reference} per-step | **{device} per-step** (synced) | {device} natural | "
      f"{device} / {reference} | {reference} AOT compile (one-time) | {device} HBM peak / chip |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for r in table:
        a(f"| {r['slug']} | {_f(r['ref_step'], 'ms')} (n={r['ref_step_n']}) | "
          f"**{_f(r['dev_step'], 'ms')}** (n={r['dev_step_n']}) | {_f(r['dev_step_natural'], 'ms')} | "
          f"{_ratio(r['dev_step'], r['ref_step'])} | {_f(r['ref_compile'], digits=0)} | "
          f"{'—' if r['dev_mem'] is None else '%.1f GB' % r['dev_mem']} |")
    a("")
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", default="trn2")
    p.add_argument("--device", default="v5e")
    p.add_argument("--json", action="store_true", help="print the row dicts instead of Markdown")
    args = p.parse_args()
    table = rows(args.reference, args.device)
    if args.json:
        print(json.dumps(table, indent=2))
    else:
        print(render(args.reference, args.device, table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
