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
from benchmark.models import (CONFIGS, MATRIX, ONLINE_DELTA_SWEEP, UNSUPPORTED, is_sweep_label,
                              json_path, resolve)

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
    "tp4sdpa": "tp4sdpa — tp=4 with `--attention-impl sdpa` (PyTorch SDPA through XLA instead of the attention_cte megakernel routing)",
    "tp4cfg2": "tp4cfg2 — tp=4 at guidance 2.0 (two sequential CFG branches: the same-work baseline for tp2cfg)",
    "tp4tc2": "tp4tc2 — tp=4 + TeaCache fixed cadence 2 (`--teacache-cadence 2`, warmup/cooldown 5 steps, same artifact)",
    "tp4tcod": "tp4tcod — tp=4 + TeaCache online-delta adaptive (`--teacache-online-delta 0.6`, calibration-free, same artifact)",
    "tp4tcad": "tp4tcad — tp=4 + TeaCache calibrated adaptive (`--teacache-speedup` at cadence 2's skip budget + per-model `--teacache-calibration`; probe NEFF for FLUX / Qwen-Image / HunyuanVideo, host signal for Wan / LTX-2)",
}
_CONFIG_TITLE.update({
    label: f"{label} — tp=4 + TeaCache online-delta adaptive (`--teacache-online-delta {a}`, alpha sweep)"
    for label, a in ONLINE_DELTA_SWEEP.items()
})
_BEGIN, _END = "<!-- campaign:begin -->", "<!-- campaign:end -->"


def _load(slug: str, label: str) -> dict | None:
    p = Path(json_path(resolve(slug, label).config_slug))
    return json.loads(p.read_text()) if p.exists() else None


def _min(s) -> str:
    return "—" if s is None else (f"{s/60:.1f} min" if s >= 120 else f"{s:.0f} s")


def _sec(s) -> str:
    return "—" if s is None else f"{s:.0f} s"


def _calls_per_step(d: dict) -> float:
    """DiT calls per scheduler step (2 = sequential CFG branches). From the
    JSON field when present, else from the real-loop note ('40 DiT calls timed')."""
    cps = d.get("dit_calls_per_step")
    if cps:
        return float(cps)
    m = re.search(r"(\d+) DiT calls timed", " ".join(d.get("notes") or []))
    steps = d.get("steps") or 0
    return (int(m.group(1)) / steps) if (m and steps) else 1.0


def _step_ms(d: dict):
    """Per scheduler step in ms (inter-call delta x calls per step), or None."""
    st = d.get("step_latency") or {}
    if not st.get("mean"):
        return None
    return st["mean"] * 1000 * _calls_per_step(d)


def _ms(st, d: dict | None = None) -> str:
    if not st:
        return "—"
    cps = _calls_per_step(d) if d else 1.0
    if cps > 1:
        return f"{st['mean']*1000*cps:.1f} ms ({cps:g} calls × {st['mean']*1000:.1f}, n={st['n']})"
    return f"{st['mean']*1000:.1f} ms (n={st['n']})"


def _shape(d) -> str:
    sh = d.get("shape") or {}
    dims = [sh.get("height"), sh.get("width"), sh.get("num_frames")]
    return "×".join(str(x) for x in dims if x is not None)


def _finite(d: dict):
    """Output finiteness: the adapter's tensor check when the CLI saved a .pt,
    else the real-loop run's own check of its output tensor (in its note)."""
    fin = (d.get("output") or {}).get("finite")
    if fin is None:
        notes = " ".join(d.get("notes") or [])
        fin = True if "finite=True" in notes else (False if "finite=False" in notes else None)
    return fin


def feature_table(label: str, price: float | None, models=CAMPAIGN_MODELS) -> list[str]:
    L = [f"### Feature: {_CONFIG_TITLE.get(label, label)}", "",
         "| model | shape / steps | compile¹ | **e2e cold**² | **e2e warm**³ | load cold→warm⁶ | "
         "**DiT per-step**⁰ | outputs/hr (warm)⁴ | cost / 1k⁵ | guidance | output | status |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for slug in models:
        name = _NAMES.get(slug, slug)
        d = _load(slug, label)
        if (slug, label) in UNSUPPORTED:
            reason = UNSUPPORTED[(slug, label)].split(" (")[0].split(":")[0]
            L.append(f"| {name} | — | — | — | — | — | — | — | — | — | — | **N/A** — {reason} |")
            continue
        if d is None:
            L.append(f"| {name} | — | — | — | — | — | — | — | — | — | — | not measured |")
            continue
        if d.get("status") == "blocked":
            reason = d.get("blocked_reason") or (d.get("notes") or ["blocked"])[0]
            L.append(f"| {name} | {_shape(d)} / {d.get('steps') or '—'} | — | — | — | — | — | — | — | "
                     f"{d.get('guidance_scale', '—')} | — | **BLOCKED** — {reason} |")
            continue
        cold_load = (d.get("e2e_breakdown") or {}).get("weights_load_total_s")
        warm_load = (d.get("e2e_warm_breakdown") or {}).get("weights_load_total_s")
        load_s = (f"{cold_load:.0f}→{warm_load:.0f} s" if cold_load is not None and warm_load is not None
                  else "—")
        cfg = resolve(slug, label)
        warm = (d.get("e2e_warm") or {}).get("mean")
        per_hr = 3600.0 / warm if warm else None
        cost = (price / per_hr * 1000) if (price and per_hr) else None
        finite = _finite(d)
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
            load_s,
            f"**{_ms(st, d)}**{note}",
            f"{per_hr:.0f}" if per_hr else "—",
            f"${cost:.2f}" if cost is not None else "—",
            str(d.get("guidance_scale", "—")),
            out_s,
            status,
        ]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    return L


def nxdi_table(price: float | None) -> list[str]:
    """difflet vs the native NxDI FLUX baseline (benchmark/nxdi_flux_baseline.py)."""
    p = Path(json_path("flux_1_dev_nxdi"))
    if not p.exists():
        return []
    n = json.loads(p.read_text())
    d = _load("flux_1_dev", "tp4") or {}
    rows = [("difflet `flux_1_dev` tp4", d, "flux_1_dev.md"),
            ("**native NxDI** `generate_flux.py` setup, tp4", n, "flux_1_dev_nxdi.md")]
    L = ["#### FLUX.1-dev tp4: difflet vs the native NxDI baseline (same weights, 1024², 28 steps, "
         "seed 42, guidance 3.5, bf16)", "",
         "| engine | compile | **e2e cold** | **e2e warm** | Neuron load cold→warm | host-side load (warm) | "
         "**DiT per-step** | outputs/hr (warm) | output |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, r, md in rows:
        if not r:
            continue
        warm = (r.get("e2e_warm") or {}).get("mean")
        cl = (r.get("e2e_breakdown") or {}).get("weights_load_total_s")
        wl = (r.get("e2e_warm_breakdown") or {}).get("weights_load_total_s")
        host = (r.get("e2e_warm_breakdown") or {}).get("host_pipeline_load_s")
        fin = _finite(r)
        cells = [
            f"[{name}]({md})", _min(r.get("compile_seconds")),
            f"**{_sec(r.get('e2e_cold_seconds'))}**", f"**{_sec(warm)}**",
            "—" if cl is None or wl is None else f"{cl:.0f}→{wl:.0f} s",
            "—" if host is None else f"{host:.0f} s",
            f"**{_ms(r.get('step_latency'))}**",
            f"{3600 / warm:.0f}" if warm else "—",
            "✓ finite" if fin else "?",
        ]
        L.append("| " + " | ".join(cells) + " |")
    L += ["",
          "NxDI's `NeuronFluxApplication` loads the full diffusers pipeline on the host in every "
          "process (the host-side column, inside its e2e) and runs a warm-up forward per "
          "component inside `load()`; difflet loads only the Neuron stages from presharded "
          "per-rank checkpoints. The DiT per-step (same attention_cte lineage, same compiler "
          "flags) is the like-for-like number; e2e differences are mostly load-path design. "
          "Measured by `benchmark/trn2/nxdi_flux_baseline.sh` with the campaign rules "
          "(timed compile into a fresh workdir; cold = page cache dropped; one generate per "
          "process, no warm-up; real-loop per-step).", ""]
    return L


_RE_TC_STATS = re.compile(r"\[teacache\] stats: (\{.*?\})")


def _teacache_skips(slug: str, label: str, d: dict) -> tuple[int | None, str]:
    """(skipped_steps, source): the `[teacache] stats` line from the realloop
    log (in-process generate, same stdout) or the warm generate log; else the
    real-loop DiT-call count (steps - dit_calls) -- the only evidence for
    Qwen-Image, whose pipeline prints no stats line."""
    import ast
    from benchmark.adapters.trainium import spec_slug
    cfg = resolve(slug, label)
    tc = d.get("teacache") or {}
    logs = Path("benchmark") / (d.get("device_slug") or "trn2") / "logs"
    for log in (logs / label / f"{slug}_realloop.log", logs / "warm" / f"{spec_slug(cfg)}_generate.log"):
        if log.exists():
            hits = _RE_TC_STATS.findall(log.read_text(errors="ignore"))
            if hits:
                try:
                    skipped = int(ast.literal_eval(hits[-1]).get("skipped_steps"))
                except Exception:
                    continue
                # persist the log evidence into the cell JSON so a report
                # regenerated where the logs are absent keeps its source
                if tc.get("skipped_steps_stats_line") != skipped:
                    jp = Path(json_path(cfg.config_slug))
                    if jp.exists():
                        full = json.loads(jp.read_text())
                        full.setdefault("teacache", {})["skipped_steps_stats_line"] = skipped
                        jp.write_text(json.dumps(full, indent=2))
                    tc["skipped_steps_stats_line"] = skipped
                return skipped, "stats line"
    if tc.get("skipped_steps_stats_line") is not None:
        return int(tc["skipped_steps_stats_line"]), "stats line"
    if tc.get("skipped_steps_by_calls") is not None:
        return int(tc["skipped_steps_by_calls"]), "DiT-call count"
    return None, "—"


def _parity(slug: str, label: str, d: dict) -> str:
    """PSNR of this cell's saved output vs the tp4 output (same seed).

    Computed from the two output files when both are on this host and then
    persisted into the cell JSON (``output_vs_tp4``), so a report regenerated
    on another host -- or after the logs dir is pruned -- keeps the measured
    value instead of blanking the column."""
    from benchmark.adapters.trainium import spec_slug
    from benchmark.output_parity import compare
    cfg, base = resolve(slug, label), resolve(slug, "tp4")
    ext = ".png" if cfg.output_kind == "image" else ".mp4"
    logs = Path("benchmark") / (d.get("device_slug") or "trn2") / "logs"
    a, b = logs / f"{spec_slug(base)}_out{ext}", logs / f"{spec_slug(cfg)}_out{ext}"
    r = None
    if a.exists() and b.exists():
        try:
            r = compare(a, b)
        except Exception as exc:
            return f"? ({type(exc).__name__})"
        if r.get("error"):
            return "?"
        keep = {k: r[k] for k in ("identical", "psnr_db", "ssim", "mean_abs_diff") if k in r}
        keep["source"] = f"{a.name} vs {b.name} on this host"
        if d.get("output_vs_tp4") != keep:
            d["output_vs_tp4"] = keep
            jp = Path(json_path(cfg.config_slug))
            if jp.exists():
                full = json.loads(jp.read_text())
                full["output_vs_tp4"] = keep
                jp.write_text(json.dumps(full, indent=2))
    else:
        r = d.get("output_vs_tp4")
    if not r:
        return "—"
    if r.get("identical"):
        return "**identical (no-op)**"
    psnr = r.get("psnr_db")
    if psnr is None:
        return "—"
    return (f"{psnr:.1f} dB" if psnr != float('inf') else "∞") + \
           (f", SSIM {r['ssim']:.3f}" if r.get("ssim") is not None else "")


def teacache_table(tc_labels: list[str], models=CAMPAIGN_MODELS) -> list[str]:
    """TeaCache vs the tp4 baseline: what was skipped, what it bought, what it cost."""
    L = ["### TeaCache vs tp4 (same artifact, same seed)", "",
         "| model | steps | mode | skipped steps (evidence) | warm e2e: tp4 → TC | "
         "loop ms/step: tp4 → TC⁷ | DiT call (ms) | output vs tp4 (PSNR)⁸ |",
         "|---|---:|---|---|---:|---:|---:|---|"]
    for slug in models:
        base = _load(slug, "tp4") or {}
        b_warm = (base.get("e2e_warm") or {}).get("mean")
        # tp4 has no skipped steps, so its loop per-step is its DiT call time
        # (x calls per step) when the file predates the loop-wall field
        b_loop = base.get("loop_step_ms") or _step_ms(base)
        for label in tc_labels:
            d = _load(slug, label)
            if d is None:
                L.append(f"| {_NAMES.get(slug, slug)} | — | {label} | not measured | | | | |")
                continue
            if d.get("status") == "blocked":
                reason = d.get("blocked_reason") or "blocked"
                L.append(f"| {_NAMES.get(slug, slug)} | {d.get('steps') or '—'} | {label} | "
                         f"**BLOCKED** — {reason} | — | — | — | — |")
                continue
            tc = d.get("teacache") or {}
            if tc.get("mode") == "fixed_cadence":
                mode = "cadence " + str(tc.get("cadence"))
            elif tc.get("mode") == "adaptive":
                r2 = tc.get("fit_r2")
                mode = (f"calibrated adaptive (target {tc.get('target_speedup')}×"
                        + (f", R² {r2:.2f}" if isinstance(r2, (int, float)) else "") + ")")
            else:
                mode = f"online-δ α={tc.get('online_delta_alpha')}"
            skipped, src = _teacache_skips(slug, label, d)
            steps = d.get("steps")
            skip_s = f"**{skipped}/{steps}** ({src})" if skipped is not None else "—"
            warm = (d.get("e2e_warm") or {}).get("mean")
            warm_s = (f"{b_warm:.0f} → **{warm:.0f} s** ({b_warm/warm:.2f}×)"
                      if b_warm and warm else "—")
            loop = d.get("loop_step_ms")
            loop_s = (f"{b_loop:.0f} → **{loop:.0f}** ({b_loop/loop:.2f}×)" if b_loop and loop
                      else (f"→ {loop:.0f}" if loop else "—"))
            st = d.get("step_latency") or {}
            call_s = f"{st['mean']*1000:.1f} (n={st['n']})" if st.get("mean") else "—"
            L.append(f"| {_NAMES.get(slug, slug)} | {steps} | {mode} | {skip_s} | {warm_s} | "
                     f"{loop_s} | {call_s} | {_parity(slug, label, d)} |")
    L += ["",
          "⁷ loop ms/step = denoise-loop wall (first DiT call entry → last call exit) ÷ scheduler "
          "steps, so a skipped step counts as ~0 — the per-step figure TeaCache actually changes; "
          "the DiT call column is the unchanged cost of one real call. tp4 skips nothing, so its "
          "loop figure is its DiT call time (× calls per step). ⁸ PSNR of this cell's "
          "output against the tp4 output at the same seed (pixel space; SSIM when "
          "scikit-image is installed); an identical output means the controller skipped nothing.",
          ""]
    return L


def _trace_knee(trace: list, alpha: float | None, warmup: int, steps: int, cooldown: int):
    """Offline what-if from one run's ``delta_trace`` (``[[step, rel_l1], ...]``,
    baseline = its first entry): the first eligible step whose PREVIOUS full
    step's delta is < alpha x baseline (what the controller would skip first),
    and the knee alpha = the smallest delta/baseline ratio over the window (below
    it nothing is ever skipped). Exact only up to that run's own first skip,
    because a skip changes the trajectory after it."""
    if not trace:
        return None, None
    by_step = {int(s): float(v) for s, v in trace}
    baseline = float(trace[0][1]) or 1e-12
    ratios = [(s, by_step[s - 1] / baseline) for s in range(warmup, steps - cooldown)
              if (s - 1) in by_step]
    if not ratios:
        return None, None
    knee = min(r for _, r in ratios)
    first = next((s for s, r in ratios if alpha is not None and r < alpha), None)
    return first, knee


def alpha_sweep_table(sweep_labels: list[str], models=CAMPAIGN_MODELS) -> list[str]:
    """Online-delta alpha sweep: per model, the skip / speed / quality curve over
    alpha, next to cadence 2 and the committed alpha=0.6 row as references."""
    sweep_labels = sorted(sweep_labels, key=ONLINE_DELTA_SWEEP.get)
    L = ["### TeaCache online-delta α sweep (same tp4 artifact, same seed, one host)", "",
         "| model | α (cell) | skipped / steps (evidence) | skipped step indices | "
         "loop ms/step: tp4 → TC⁷ | warm e2e | output vs tp4 (PSNR / SSIM)⁸ | "
         "trace what-if⁹ |",
         "|---|---|---:|---|---:|---:|---|---|"]
    for slug in models:
        base = _load(slug, "tp4") or {}
        b_warm = (base.get("e2e_warm") or {}).get("mean")
        b_loop = base.get("loop_step_ms") or _step_ms(base)
        # the lowest-alpha sweep run that recorded a trace = closest to the
        # unskipped trajectory, so its trace is the what-if reference
        ref_trace, ref_label = [], None
        for label in sweep_labels:
            d = _load(slug, label)
            tr = ((d or {}).get("teacache") or {}).get("stats", {}).get("delta_trace") or []
            if tr:
                ref_trace, ref_label = tr, label
                break
        rows = [("tp4tc2", "cadence 2 (ref)"), ("tp4tcod", "0.6 (committed 2026-09-13 host)")]
        rows += [(l, f"{ONLINE_DELTA_SWEEP[l]} ({l})") for l in sweep_labels]
        for label, name in rows:
            d = _load(slug, label)
            if d is None:
                if label in sweep_labels:
                    L.append(f"| {_NAMES.get(slug, slug)} | {name} | not measured | | | | | |")
                continue
            tc = d.get("teacache") or {}
            st = tc.get("stats") or {}
            steps = d.get("steps")
            skipped, src = _teacache_skips(slug, label, d)
            skip_s = f"**{skipped}/{steps}** ({src})" if skipped is not None else "—"
            idx = st.get("skipped_step_indices")
            idx_s = ", ".join(str(i) for i in idx) if idx else ("none" if skipped == 0 else "—")
            loop = d.get("loop_step_ms")
            loop_s = (f"{b_loop:.0f} → **{loop:.0f}** ({b_loop/loop:.2f}×)" if b_loop and loop
                      else (f"→ {loop:.0f}" if loop else "—"))
            warm = (d.get("e2e_warm") or {}).get("mean")
            warm_s = (f"{b_warm:.0f} → {warm:.0f} s" if b_warm and warm
                      else (f"{warm:.0f} s" if warm else "—"))
            alpha = tc.get("online_delta_alpha") if tc.get("mode") == "online_delta" else None
            if alpha is not None and ref_trace and steps:
                first, knee = _trace_knee(ref_trace, float(alpha), int(tc.get("warmup_steps", 5)),
                                          int(steps), int(tc.get("cooldown_steps", 5)))
                what_if = (f"first skip @ step {first}" if first is not None else "no skip") + \
                          (f"; knee α ≈ {knee:.2f}" if knee is not None else "") + \
                          f" (from {ref_label})"
            else:
                what_if = "—"
            L.append(f"| {_NAMES.get(slug, slug)} | {name} | {skip_s} | {idx_s} | {loop_s} | "
                     f"{warm_s} | {_parity(slug, label, d)} | {what_if} |")
    L += ["",
          "The controller (`difflet/pipeline/teacache.py`, online-delta mode) skips step *s* iff "
          "the rel-L1 delta of the last full step is < α × baseline, where the baseline is the "
          "first measured delta (step 1, inside the 5-step warmup, where the trajectory moves "
          "fastest) and never skips two steps in a row — so skips are capped at cadence 2's count "
          "(9/28, 5/20) whatever α, and α only chooses WHICH of the eligible steps go. Warm e2e "
          "is load-dominated and shown only for completeness. "
          "⁹ what-if = read off the lowest-α run's per-step delta trace: the first eligible step "
          "the rule would skip at this α, and the knee α below which it would skip nothing; "
          "exact only up to that run's own first skip (a skip changes the trajectory after it).",
          ""]
    return L


def serving_table(price: float | None, models=CAMPAIGN_MODELS) -> list[str]:
    """difflet serve (resident model) vs the CLI's per-request warm e2e, from
    benchmark/<device>/serving/<slug>_tp4.json (+ _warm.json for the load-only
    startup) written by benchmark.serve_bench."""
    sdir = Path("benchmark") / "trn2" / "serving"
    rows = []
    for slug in models:
        p = sdir / f"{slug}_tp4.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        w = sdir / f"{slug}_tp4_warm.json"
        warm = json.loads(w.read_text()) if w.exists() else None
        base = _load(slug, "tp4") or {}
        rows.append((slug, d, warm, base))
    if not rows:
        return []
    L = ["### Serving layer: `difflet serve` (resident model, tp4) vs the CLI", "",
         "| model | endpoint | startup → /ready: first (compiles) / warm⁹ | c=1 p50 / p90 / p99 | "
         "c=2 p50 | c=4 p50 | throughput (any c) | CLI warm e2e → resident speedup | "
         "NeuronCore util (c=1)¹⁰ | device mem | errors | cost / 1k¹¹ |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for slug, d, warm, base in rows:
        lv = {l["concurrency"]: l for l in d["levels"]}
        l1, l2, l4 = lv.get(1), lv.get(2), lv.get(4)
        lat = (l1 or {}).get("latency_s") or {}
        thr = (l1 or {}).get("throughput_per_hour")
        unit = "img/h" if d["kind"] == "image" else "videos/h"
        cli_warm = (base.get("e2e_warm") or {}).get("mean")
        speed = f"{cli_warm:.0f} s → {cli_warm/lat['p50']:.1f}×" if cli_warm and lat.get("p50") else "—"
        neu = ((warm or d)["levels"][0].get("neuron") if (warm or d)["levels"] else None) or {}
        if not neu and l1:
            neu = l1.get("neuron") or {}
        errs = sum(v for l in d["levels"] for k, v in l["http_codes"].items() if k != "200")
        cost = f"${price / thr * 1000:.2f}" if (price and thr) else "—"
        ready_first = d.get("ready_seconds")
        ready_warm = warm.get("ready_seconds") if warm else None
        ready_s = f"{ready_first/60:.0f} min" if ready_first else "—"
        ready_s += f" / **{ready_warm:.0f} s**" if ready_warm else " / —"
        L.append(
            f"| {_NAMES.get(slug, slug)} | `{d['endpoint']}` | {ready_s} | "
            f"**{lat.get('p50', '—')}** / {lat.get('p90', '—')} / {lat.get('p99', '—')} s | "
            f"{(l2 or {}).get('latency_s', {}).get('p50', '—')} s | "
            f"{(l4 or {}).get('latency_s', {}).get('p50', '—')} s | "
            f"**{thr:.0f} {unit}** | {speed} | "
            f"{neu.get('neuroncore_util_mean_pct', '—')}% | {neu.get('device_mem_used_gb_max', '—')} GB | "
            f"{errs} | {cost} |")
    L += ["",
          "Closed loop: c in-flight requests until 8 complete (HunyuanVideo 6), no think time, after 2 "
          "warm-up requests; latency = client wall per request including queueing; throughput = "
          "successes ÷ level wall. `difflet serve` runs **one resident worker** "
          "(`max_running_requests=1`), so c = 2 / 4 measure queueing (p50 ≈ c × service time) and "
          "throughput is flat — parallel execution needs `--dp` replicas, which need ≥ 2 cores each. "
          "Image requests are JSON on `/v1/chat/completions` (base64 PNG back); video requests are "
          "multipart on `/v1/videos/sync` (mp4 bytes back), admitted through the video service FIFO "
          "(`--max-queued-requests 8`, `--request-timeout 1800`). ⁹ Serving has its own immutable "
          "artifact generation under `~/.cache/difflet/serving/`: the first start compiles it from "
          "scratch (the CLI artifacts are not reused); the second figure is a restart against the "
          "published generation (no compile, load only) — measured after the other models had "
          "evicted this model's files from the page cache, so it is a cold-cache load; a restart "
          "right after publish, page cache warm, took 185 s for HunyuanVideo. ¹⁰ Mean over all 4 "
          "cores of neuron-monitor's "
          "`neuroncore_utilization` sampled every 1 s during the c=1 level. ¹¹ At the indicative "
          "hourly price stated above.", ""]
    return L


def serving_async_table(models=CAMPAIGN_MODELS) -> list[str]:
    """Async Videos job API (/v1/videos) vs the blocking /v1/videos/sync, from
    benchmark/<device>/serving/<slug>_tp4_async.json and <slug>_tp4_sync.json
    (same server session, same host) next to the committed <slug>_tp4.json
    (the sync pass of the campaign host)."""
    sdir = Path("benchmark") / "trn2" / "serving"
    rows = []
    for slug in models:
        a = sdir / f"{slug}_tp4_async.json"
        if not a.exists():
            continue
        s = sdir / f"{slug}_tp4_sync.json"
        c = sdir / f"{slug}_tp4.json"
        rows.append((slug, json.loads(a.read_text()),
                     json.loads(s.read_text()) if s.exists() else None,
                     json.loads(c.read_text()) if c.exists() else None))
    if not rows:
        return []
    L = ["### Serving layer: async Videos job API vs sync (same server session, tp4)", "",
         "| model | API | c=1 p50 / p90 | c=2 p50 | c=4 p50 | throughput (c=1) | "
         "create ack p50 / max | queue wait p50 (c=1) | poll + download overhead p50 | "
         "burst: admitted / N, all acked, wall, rate | errors |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]

    def p50(d, c):
        lv = {l["concurrency"]: l for l in d["levels"]}.get(c) or {}
        return (lv.get("latency_s") or {}).get("p50")

    def fmt(x, nd=1, unit=" s"):
        return f"{x:.{nd}f}{unit}" if isinstance(x, (int, float)) else "—"

    for slug, asy, syn, committed in rows:
        for name, d in (("sync (committed 2026-09-13 host)", committed),
                        ("sync `/v1/videos/sync` (this session)", syn),
                        ("**async `/v1/videos`** (this session)", asy)):
            if d is None:
                continue
            lv = {l["concurrency"]: l for l in d["levels"]}
            l1 = lv.get(1) or {}
            lat = l1.get("latency_s") or {}
            thr = l1.get("throughput_per_hour")
            errs = sum(v for l in d["levels"] for k, v in l["http_codes"].items() if k != "200")
            asy_l1 = l1.get("async") or {}
            create = asy_l1.get("create_s") or {}
            create_s = (f"{create['p50']*1000:.0f} / {create['max']*1000:.0f} ms"
                        if create.get("p50") is not None else "—")
            queue_s = fmt((asy_l1.get("queued_s") or {}).get("p50"), 2)
            over_s = fmt((asy_l1.get("overhead_s") or {}).get("p50"), 2)
            b = d.get("burst")
            if b:
                burst_s = (f"{b['admitted']} / {b['jobs']}, acked in {b['all_acked_s']*1000:.0f} ms, "
                           f"wall {b['wall_s']:.0f} s, {b['throughput_per_hour']:.0f}/h"
                           + (f" (codes {b['http_codes']})" if len(b["http_codes"]) > 1 else ""))
            else:
                burst_s = "—"
            L.append(f"| {_NAMES.get(slug, slug)} | {name} | **{fmt(lat.get('p50'), 2)}** / "
                     f"{fmt(lat.get('p90'), 2)} | {fmt(p50(d, 2), 2)} | {fmt(p50(d, 4), 2)} | "
                     f"{fmt(thr, 0, ' videos/h')} | {create_s} | {queue_s} | {over_s} | {burst_s} | "
                     f"{errs} |")
    L += ["",
          "Same server command line and the same closed-loop client as the sync rows (c in-flight "
          "until N complete, no think time). Async request = `POST /v1/videos` (returns the queued "
          "job) → poll `GET /v1/videos/{id}` every 0.25 s → `GET …/content` → `DELETE`; latency = "
          "create → content downloaded (DELETE excluded), so it is quantised by the poll interval. "
          "Both APIs share the server's single admission FIFO and its one resident worker "
          "(`difflet/serving/video_service.py`, `factory.py`), so the device time per request is "
          "identical by construction: what async changes is the client side — the create "
          "acknowledgement (the job object) instead of a held connection, server-side queueing at "
          "c > 1, and burst absorption (N creates back to back; the server admits "
          "1 + `--max-queued-requests` = 9, the rest get 429). Image models have no async endpoint.",
          ""]
    return L


def speedup_table(labels: list[str], models=CAMPAIGN_MODELS) -> list[str]:
    L = ["### DiT per-step vs tp4 (lower is better; ratio = tp4 / config; a step with two "
         "sequential CFG calls counts both calls)", "",
         "| model | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for slug in models:
        base = _load(slug, "tp4")
        b_ms = _step_ms(base) if base else None
        cells = []
        for label in labels:
            if (slug, label) in UNSUPPORTED:
                cells.append("N/A")
                continue
            d = _load(slug, label)
            ms = _step_ms(d) if d else None
            if ms is None:
                cells.append("—")
            elif label == "tp4" or not b_ms:
                cells.append(f"{ms:.1f} ms")
            else:
                cells.append(f"{ms:.1f} ms ({b_ms/ms:.2f}×)")
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
    # The alpha-sweep cells are one experiment, not campaign features: they get
    # their own table (alpha_sweep_table) and stay out of the per-feature /
    # speedup tables, which would otherwise repeat six near-identical blocks.
    feature_labels = [l for l in labels if not is_sweep_label(l)]
    sweep_labels = sorted((l for l in labels if is_sweep_label(l)), key=ONLINE_DELTA_SWEEP.get)
    for label in feature_labels:
        L += feature_table(label, price)
        if label == "tp4":
            L += nxdi_table(price)
    tc_labels = [l for l in feature_labels if l in ("tp4tc2", "tp4tcod", "tp4tcad")]
    if tc_labels:
        L += teacache_table(tc_labels)
    if sweep_labels:
        L += alpha_sweep_table(sweep_labels)
    L += serving_table(price)
    L += serving_async_table()
    L += speedup_table(feature_labels)
    L += [
        "⁰ DiT per-step: mean of the inter-step deltas (n = steps − 1). ¹ compile = full `difflet "
        "compile` wall (all stages, incl. per-rank presharding); stage caches shared across "
        "features are reused, so a later feature's compile can be shorter than tp4's. ² cold = "
        "`sync; echo 3 > drop_caches` then one generate. ³ warm = the immediately following "
        "generate. ⁴ outputs/hr = 3600 / warm e2e (one image or one video per generate, batch 1, "
        "fresh process each — a served deployment with a resident model does better). "
        f"⁵ cost / 1k outputs = hourly price ÷ outputs/hr × 1000; {price_note}. "
        "⁶ Neuron weight load summed over the pipeline's stages (from the generate log), cold vs "
        "warm — the bulk of the cold→warm gap; LTX-2's text encoder and VAE run on the host and "
        "are not in it.",
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
    order = ["tp4", "tp2cp2", "tp4sp", "tp2cfg", "tp4cfg2", "tp4sdpa", "tp4tc2", "tp4tcod",
             "tp4tcad"] + sorted(ONLINE_DELTA_SWEEP, key=ONLINE_DELTA_SWEEP.get)
    labels = [l for l in order if l in a.labels]
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
