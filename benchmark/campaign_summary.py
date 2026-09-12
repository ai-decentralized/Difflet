"""Render the parallel-topology campaign section of benchmark/<device>/RESULTS.md
from the per-cell JSONs, so every number in the report traces to a file.

    python -m benchmark.campaign_summary --labels tp4 tp2cp2 --price-per-hour 2.79 \
        [--write benchmark/trn2/RESULTS.md]

Prints markdown; with --write it replaces the block between
``<!-- campaign:begin -->`` and ``<!-- campaign:end -->`` in RESULTS.md (the
markers are inserted at the end of the file the first time).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from benchmark.cell_status import CAMPAIGN_MODELS
from benchmark.models import CONFIGS, MATRIX, UNSUPPORTED, json_path, resolve

_NAMES = {
    "flux_1_dev": "FLUX.1-dev", "qwen_image": "Qwen-Image", "ltx_2": "LTX-2",
    "hunyuan_video": "HunyuanVideo", "wan_2_1": "Wan 2.1 14B",
}
_KIND = {"image": "image", "video": "video"}
_CONFIG_TITLE = {
    "tp4": "tp4 — tensor parallel over all 4 cores (the baseline topology)",
    "tp2cp2": "tp2cp2 — tp=2 × context parallel 2, `--cp-mode ulysses`",
    "tp4sp": "tp4sp — tp=4 + Megatron sequence parallel (`--sp`)",
    "tp2cfg": "tp2cfg — tp=2 × CFG-parallel (uncond/cond branches on separate core pairs, guidance 2.0)",
}
_BEGIN, _END = "<!-- campaign:begin -->", "<!-- campaign:end -->"


def _load(slug: str, label: str) -> dict | None:
    p = Path(json_path(resolve(slug, label).config_slug))
    return json.loads(p.read_text()) if p.exists() else None


def _min(s) -> str:
    return "—" if s is None else (f"{s/60:.1f} min" if s >= 120 else f"{s:.0f} s")


def _sec(s) -> str:
    return "—" if s is None else f"{s:.0f} s"


def _ms(st) -> str:
    return "—" if not st else f"{st['mean']*1000:.1f} ms (n={st['n']})"


def _shape(d) -> str:
    sh = d.get("shape") or {}
    dims = [sh.get("height"), sh.get("width"), sh.get("num_frames")]
    return "×".join(str(x) for x in dims if x is not None)


def feature_table(label: str, price: float | None, models=CAMPAIGN_MODELS) -> list[str]:
    L = [f"### Feature: {_CONFIG_TITLE.get(label, label)}", "",
         "| model | shape / steps | compile¹ | **e2e cold**² | **e2e warm**³ | **DiT per-step**⁰ | "
         "outputs/hr (warm)⁴ | cost / 1k⁵ | guidance | output | status |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for slug in models:
        name = _NAMES.get(slug, slug)
        d = _load(slug, label)
        if (slug, label) in UNSUPPORTED:
            reason = UNSUPPORTED[(slug, label)].split(" (")[0].split(":")[0]
            L.append(f"| {name} | — | — | — | — | — | — | — | — | — | **N/A** — {reason} |")
            continue
        if d is None:
            L.append(f"| {name} | — | — | — | — | — | — | — | — | — | not measured |")
            continue
        cfg = resolve(slug, label)
        warm = (d.get("e2e_warm") or {}).get("mean")
        per_hr = 3600.0 / warm if warm else None
        cost = (price / per_hr * 1000) if (price and per_hr) else None
        out = d.get("output") or {}
        finite = out.get("finite")
        if finite is None:
            # video cells save an .mp4 (no tensor to inspect); the real-loop run
            # finite-checks its own output tensor and records it in its note
            notes = " ".join(d.get("notes") or [])
            finite = True if "finite=True" in notes else (False if "finite=False" in notes else None)
        out_s = "✓ finite" if finite else ("?" if finite is None else "**NaN/Inf**")
        st = d.get("step_latency")
        note = ""
        if st and int(st.get("n") or 0) < cfg.steps - 1:
            note = f" (n={st['n']} < {cfg.steps-1})"
        status = d.get("status", "?")
        if status == "blocked":
            status = f"**BLOCKED** — {d.get('skip_reason') or d.get('notes', [''])[-1][:80]}"
        elif status == "skipped":
            status = f"**N/A** — {d.get('skip_reason', '')[:60]}"
        cells = [
            f"[{name}]({cfg.config_slug}.md)",
            f"{_shape(d)} / {d.get('steps')}",
            _min(d.get("compile_seconds")),
            f"**{_sec(d.get('e2e_cold_seconds'))}**",
            f"**{_sec(warm)}**",
            f"**{_ms(st)}**{note}",
            f"{per_hr:.0f}" if per_hr else "—",
            f"${cost:.2f}" if cost is not None else "—",
            str(d.get("guidance_scale", "—")),
            out_s,
            status,
        ]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    return L


def speedup_table(labels: list[str], models=CAMPAIGN_MODELS) -> list[str]:
    L = ["### DiT per-step vs tp4 (lower is better; ratio = tp4 / config)", "",
         "| model | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for slug in models:
        base = _load(slug, "tp4")
        b = (base or {}).get("step_latency") or {}
        cells = []
        for label in labels:
            if (slug, label) in UNSUPPORTED:
                cells.append("N/A")
                continue
            d = _load(slug, label)
            st = (d or {}).get("step_latency") or {}
            if not st.get("mean"):
                cells.append("—")
            elif label == "tp4" or not b.get("mean"):
                cells.append(f"{st['mean']*1000:.1f} ms")
            else:
                cells.append(f"{st['mean']*1000:.1f} ms ({b['mean']/st['mean']:.2f}×)")
        L.append(f"| {_NAMES.get(slug, slug)} | " + " | ".join(cells) + " |")
    L.append("")
    return L


def toolchain_line(labels) -> str:
    for label in labels:
        for slug in CAMPAIGN_MODELS:
            d = _load(slug, label)
            if d and d.get("toolchain"):
                tc = d["toolchain"]
                return ", ".join(f"`{k}={v}`" for k, v in tc.items())
    return "(no toolchain recorded yet)"


def render(labels: list[str], price: float | None, price_note: str) -> str:
    L = [_BEGIN,
         "## 2026-09-12 parallel-topology campaign (main @ 38e863e + campaign branch)",
         "",
         "Same host class as above (**trn2.3xlarge**, 4 NeuronCores under LNC=2, 96 GB HBM, 124 GB "
         "RAM), fresh venv from `requirements-neuron.lock`; toolchain " + toolchain_line(labels) + ". "
         "Every cell is `compile` (one-time AOT, timed) → **true cold** e2e (page cache dropped) → "
         "**warm** e2e (the next process) → **DiT per-step** by the real-loop rule "
         "(`benchmark/step_realloop.py`: inter-step deltas of one real generate, device-synced, "
         "step 0 excluded — now the same method for all five models). Features ran in the order "
         "tp4 → tp2cp2 → tp4sp → tp2cfg, all models per feature, device otherwise idle. Files: "
         "`<slug>.json` (tp4) and `<slug>_<config>.json` per cell; run any cell with "
         "`python -m benchmark.{bench,cold_warm_e2e,step_realloop} --model <slug> --config <label>`.",
         ""]
    for label in labels:
        L += feature_table(label, price)
    L += speedup_table(labels)
    L += [
        "⁰ DiT per-step: mean of the inter-step deltas (n = steps − 1). ¹ compile = full `difflet "
        "compile` wall (all stages, incl. per-rank presharding); stage caches shared across "
        "features are reused, so a later feature's compile can be shorter than tp4's. ² cold = "
        "`sync; echo 3 > drop_caches` then one generate. ³ warm = the immediately following "
        "generate. ⁴ outputs/hr = 3600 / warm e2e (one image or one video per generate, batch 1, "
        "fresh process each — a served deployment with a resident model does better). "
        f"⁵ cost / 1k outputs = hourly price ÷ outputs/hr × 1000; {price_note}",
        "",
        "N/A cells are by design (the gate is named in the cell's `.md`): guidance-distilled "
        "models (FLUX, Qwen-Image, HunyuanVideo) have no second CFG branch to parallelise; LTX-2 "
        "has no CP or SP path. tp2cfg runs the two true-CFG models at guidance 2.0 (the tp4 row "
        "is single-branch at guidance 1.0), so its per-step is a two-branch step and is not a "
        "same-work comparison with tp4.",
        "",
        _END,
    ]
    return "\n".join(L)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", default=["tp4"], choices=sorted(CONFIGS))
    p.add_argument("--price-per-hour", type=float, default=None,
                   help="on-demand USD/hour for the host, for the cost column")
    p.add_argument("--price-note", default="price not recorded",
                   help="provenance of the price (region, date, source)")
    p.add_argument("--write", default=None, help="RESULTS.md to update in place")
    a = p.parse_args()
    labels = [l for l in ["tp4", "tp2cp2", "tp4sp", "tp2cfg"] if l in a.labels]
    md = render(labels, a.price_per_hour, a.price_note)
    if not a.write:
        print(md)
        return 0
    path = Path(a.write)
    text = path.read_text()
    if _BEGIN in text and _END in text:
        text = re.sub(re.escape(_BEGIN) + r".*?" + re.escape(_END), lambda _m: md, text, flags=re.S)
    else:
        text = text.rstrip("\n") + "\n\n" + md + "\n"
    path.write_text(text)
    print(f"[campaign] wrote {path} ({len(labels)} feature table(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
