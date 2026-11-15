#!/usr/bin/env python3
"""The MXFP8 attention plan's Task 6 (docs/plans/2026-11-14-attn-quant-model-runs.md): one key's
quality and its page dataset, from run_arms.sh's outputs under ROOT/<key>/<arm>/.

    task6.py quality <key> <arm>
        ROOT/<key>/quality.json: the outputs' md5s (determinism: the arm's two timed requests and
        its gen2 process), the arm's latents finite, the arm against bf16 (compare_difflet_runs:
        PSNR / SSIM / LPIPS on the decoded output, latents rel L2 / cosine; first timed request
        of each), and for qwen_image / flux / ltx_2 the arm and bf16 against the CPU fp32
        reference (scripts/fp32_ref/score.py's comparison, reference first, written here: score.py
        writes into the reference directory); ROOT/<key>/frames.png, bf16 above the arm, to look at.
    task6.py dataset <key> <arm>
        ROOT/page/<key>.json (the page's rows) and evidence section 15's rows on stdout: step
        times and G3 (steps.py), the block profile (the arm's breakdowns committed beside Plan
        A's bf16 ones, docs/verification/assets/2026-11-13-dit-block-profile/<key>/<arm>/,
        collect.py's execution weighting; bf16's attention phases from Plan A's captures analysed
        again with exp split out, ROOT/planA_bf16/<key>/), and the quality columns with the
        evidence's (collect.py's QUALITY).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts" / "profile_dit"), str(REPO / "scripts"), str(Path(__file__).parent)]
import collect  # noqa: E402
import compare_difflet_runs as cdr  # noqa: E402
import steps as steps_mod  # noqa: E402

ROOT = Path(os.environ.get("ROOT", "/mnt/nvme/difflet-mxattn/task6"))
FP32 = Path("/home/ubuntu/fp8-bench/fp32_ref")
FP32_KEYS = ("qwen_image", "flux", "ltx_2")
EXT = {"qwen_image": "png", "flux": "png"}
#: component groups of the plan's metrics
GROUPS = {"attention": ("attention",), "attn glue": ("attn_glue",),
          "MLP": ("mlp_up", "mlp_act", "mlp_down"), "norms + modulation": ("norm", "modulation"),
          "projections": ("qkv_proj", "out_proj")}
PHASES = ("qk_matmul", "mask_softmax", "exp", "pv_matmul", "qk_quantize", "v_quantize",
          "p_quantize", "other")
#: rows whose quality is below what the bf16 floor resolves (the plan's caveat)
BELOW_FLOOR = {"wan22_official", "hv15_official"}
#: bounded routed calls: K smoothing averages over every key, padding included (ruling BR36)
BOUNDED = {"qwen_image", "hunyuan_video_L", "hv15_official"}
#: the evidence's bf16 output md5 at this shape (section, host)
BF16_MD5 = {"qwen_image": ("5478687f", "11c, A"), "flux": ("67da9e98", "11c, A"),
            "ltx_2": ("0702ffd2", "11c, A"), "wan21_official": ("295dcb67", "12c, B"),
            "wan22_official": ("ee61e12e", "12c, B"), "hunyuan_video_L": ("258cd73e", "7f, A"),
            "hv15_official": ("d3ef0f1c", "12c, B, 96 threads")}


def _md5(path: Path) -> str | None:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else None


def _latents(work: Path) -> Path | None:
    """The first request's final latents in a work dir (batch mode: ``latents_req0000.pt``)."""
    found = sorted(work.rglob("latents_req0000.pt")) or sorted(work.rglob("latents.pt"))
    return found[0] if found else None


def _finite(path: Path | None) -> bool | None:
    if path is None:
        return None
    t = torch.load(path, map_location="cpu", weights_only=True)
    ts = [v for v in t.values() if isinstance(v, torch.Tensor)] if isinstance(t, dict) else [t]
    return all(bool(torch.isfinite(x.float()).all()) for x in ts)


def _frames(paths: list[Path], out: Path) -> None:
    """Rows of frames (first, middle, last; the image itself) per output, downscaled to 256 px."""
    from PIL import Image
    rows = []
    for p in paths:
        clip = cdr._read_frames(p)
        picks = [clip[0]] if len(clip) == 1 else [clip[0], clip[len(clip) // 2], clip[-1]]
        tiles = [Image.fromarray(f).resize((int(256 * f.shape[1] / f.shape[0]), 256)) for f in picks]
        rows.append(np.concatenate([np.asarray(t) for t in tiles], axis=1))
    width = max(r.shape[1] for r in rows)
    Image.fromarray(np.concatenate([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)))
                                    for r in rows])).save(out)


def quality(key: str, arm: str) -> dict:
    ext = EXT.get(key, "mp4")
    b, a = ROOT / key / "bf16", ROOT / key / arm
    lat_b, lat_a = _latents(b / "work_timed"), _latents(a / "work_timed")
    out = {"key": key, "arm": arm,
           "md5": {f"{d.name}/{n}": _md5(d / f"{n}.{ext}") for d in (b, a)
                   for n in ("timed_1", "timed_2", "gen2") if (d / f"{n}.{ext}").exists()},
           "latents_md5": {str(p.relative_to(ROOT)): _md5(p) for p in (lat_b, lat_a) if p},
           "arm_latents_finite": {"timed": _finite(lat_a), "gen2": _finite(_latents(a / "work_gen2"))}}
    m = out["md5"]
    out["deterministic"] = {"bf16": m.get("bf16/timed_1") == m.get("bf16/timed_2"),
                            arm: len({m.get(f"{arm}/{n}") for n in ("timed_1", "timed_2", "gen2")}) == 1}
    # a latents file's bytes hold its own name (torch.save's archive root): compare the tensors
    lats = [p for p in (lat_a, lat_a and lat_a.with_name(lat_a.name.replace("0000", "0001")),
                        _latents(a / "work_gen2")) if p and p.exists()]
    loaded = [torch.load(p, map_location="cpu", weights_only=True) for p in lats]
    same = [cdr._latent_metrics(loaded[0], x) for x in loaded[1:]]
    out["arm_latents_bit_identical"] = {str(p.relative_to(ROOT)): s for p, s in zip(lats[1:], same)}
    out["arm_vs_bf16"] = cdr.compare_runs(lat_b, lat_a, b / f"timed_1.{ext}", a / f"timed_1.{ext}")
    if key in FP32_KEYS:
        ref = FP32 / key
        for name, lat, d in (("arm_vs_fp32", lat_a, a), ("bf16_vs_fp32", lat_b, b)):
            out[name] = cdr.compare_runs(ref / "latents.pt" if lat else None, lat,
                                         ref / f"out.{ext}", d / f"timed_1.{ext}")
    _frames([b / f"timed_1.{ext}", a / f"timed_1.{ext}"], ROOT / key / "frames.png")
    (ROOT / key / "quality.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def _weighted(blocks: list[dict], field: str) -> dict:
    """{...: ms per step} of a breakdown field over the blocks, each x its executions per step."""
    acc: dict = {}
    for blk in blocks:
        collect._add(acc, blk[field], blk["execs_per_step"])
    return acc


def _phases(detail: dict) -> dict:
    """{phase: {engine: ms}} of the routed modules (attn, attn1)."""
    acc: dict = {}
    for module, phases in detail.items():
        if module in collect.ROUTED:
            collect._add(acc, phases, 1.0)
    return acc


def _config(key: str, arm: str, assets: Path, step: dict) -> dict:
    spec = collect.CONFIGS[key]
    c = collect._config(assets, key, arm, spec, (step["step_ms"], "15", "A", None))
    if arm == "bf16":  # Plan A's captures analysed again with exp split out of mask_softmax
        for blk in c["blocks"]:
            again = json.loads((ROOT / "planA_bf16" / key / f"blk{blk['blk']}.json").read_text())
            blk["attention_detail"] = again["attention_detail"]
    return c


def _q(d: dict | None) -> dict | None:
    if not d:
        return None
    dec = d.get("image") or d.get("video") or {}
    lat = d.get("latents")
    if isinstance(lat, dict) and "cosine" not in lat:  # LTX-2: {video, audio}
        lat = lat.get("video")
    return {"psnr": dec.get("psnr"), "psnr_worst": dec.get("psnr_worst"), "ssim": dec.get("ssim"),
            "lpips": dec.get("lpips"), "latents_cos": (lat or {}).get("cosine"),
            "latents_rel_l2": (lat or {}).get("rel_l2")}


def _fmt_q(q: dict | None) -> str:
    if not q:
        return "-"
    worst = f" ({q['psnr_worst']:.2f})" if q.get("psnr_worst") is not None else ""
    lp = f"{q['lpips']:.3f}" if q.get("lpips") is not None else "-"
    lat = (f" / cos {q['latents_cos']:.4f}, rel L2 {q['latents_rel_l2']:.3f}"
           if q.get("latents_cos") is not None else "")
    return f"{q['psnr']:.2f}{worst} / {q['ssim']:.3f} / {lp}{lat}"


def dataset(key: str, arm: str) -> dict:
    model, shape, layout, _, _ = collect.CONFIGS[key]
    logs = {a: ROOT / key / a / "timed.log" for a in ("bf16", arm, "mx")}
    st = {a: steps_mod.steps(p) for a, p in logs.items() if p.exists()}
    g3 = steps_mod.g3(key, st["bf16"], st[arm], st.get("mx"))
    configs = {a: _config(key, a, collect.ASSETS, st[a]) for a in ("bf16", arm)}
    out = {"key": key, "model": model, "shape": shape, "layout": layout, "arm": arm,
           "steps": st, "g3": g3, "blocks": [], "engines": [], "step_engines": [],
           "components": [], "attention_phases": [], "attention_summary": []}
    for a, c in configs.items():
        for blk in c["blocks"]:
            kb = {"arm": a, "blk": f"blk{blk['blk']}", "kind": blk["kind"]}
            out["blocks"].append(kb | {f: blk.get(f) for f in (
                "execs_per_step", "in_model_median_ms", "block_ms", "replay_spread_ms", "mfu")} | {
                "bottleneck": blk["bottleneck"]["engine_class"],
                "bottleneck_share": blk["bottleneck"]["share"],
                "psum_readout_copy_ms": blk["psum_readout_copy_ms"],
                "psum_readout_fused_ms": blk["psum_readout_fused_ms"]})
            out["engines"] += [kb | {"engine_class": e, "ms": ms} for e, ms in blk["engines_ms"].items()]
            ec = blk["bottleneck"]["engine_class"]
            out["components"] += [kb | {"group": g, "engine_class": ec, "ms": sum(
                blk["components"].get(x, {}).get(ec, 0.0) for x in parts)} for g, parts in GROUPS.items()]
            out["attention_phases"] += [kb | {"module": m or "(none)", "routed": m in collect.ROUTED,
                                              "phase": p, "engine_class": e, "ms": ms}
                                        for m, ph in blk["attention_detail"].items()
                                        for p, d in ph.items() for e, ms in d.items()]
        out["step_engines"] += [{"arm": a, "engine_class": e, "ms_per_step": ms}
                                for e, ms in _weighted(c["blocks"], "engines_ms").items()]
        per_step = _phases(_weighted(c["blocks"], "attention_detail"))
        attention = sum(_weighted(c["blocks"], "components").get("attention", {}).values())
        out["attention_summary"].append({
            "arm": a, "phases_ms_per_step": per_step, "attention_engine_ms_per_step": attention,
            "v_quantize_share": sum(per_step.get("v_quantize", {}).values()) / attention
            if attention else None, "bottleneck": c["bottleneck"]})
    qpath = ROOT / key / "quality.json"
    qual = json.loads(qpath.read_text()) if qpath.exists() else {}
    ev = {(r[1], r[4]): r for r in collect.QUALITY if r[0] == key}
    lin = next((r for (sec, against), r in ev.items() if against == "device bf16"
                and r[3] == shape), None)
    fp = ev.get(("10c", "CPU fp32")) if key in FP32_KEYS else None
    out["quality"] = {
        "arm_vs_bf16": _q(qual.get("arm_vs_bf16")), "arm_vs_fp32": _q(qual.get("arm_vs_fp32")),
        "bf16_vs_fp32_here": _q(qual.get("bf16_vs_fp32")),
        "linear_vs_bf16_evidence": dict(zip(collect._QUALITY_FIELDS, lin)) if lin else None,
        "floor_evidence": lin[12] if lin else None,
        "fp32_evidence": {"bf16_psnr": fp[10], "linear_psnr": fp[5], "linear_ssim": fp[7]} if fp else None,
        "md5": qual.get("md5"), "deterministic": qual.get("deterministic"),
        "arm_latents_finite": qual.get("arm_latents_finite"),
        "arm_latents_bit_identical": qual.get("arm_latents_bit_identical"),
        "bf16_md5_evidence": BF16_MD5[key],
        "caveat": "below the bf16 floor's resolution, not a quality reading" if key in BELOW_FLOOR else None,
        "k_smoothing": "bounded calls: K smoothing averages over all keys, padding included (BR36)"
        if key in BOUNDED else "LTX-2's routed call (video attn1) is unbounded: BR36 does not apply"
        if key == "ltx_2" else None}
    (ROOT / "page").mkdir(exist_ok=True)
    (ROOT / "page" / f"{key}.json").write_text(json.dumps(out, indent=1) + "\n")
    _rows(out)
    return out


def _rows(d: dict) -> None:
    """Section 15's rows for this key, markdown."""
    g, m, arm = d["g3"], d["model"], d["arm"]
    s = d["steps"]
    lin = (f"{g['linear_over_bf16_session']:.3f} (this session)" if "linear_over_bf16_session" in g
           else f"{g['linear_over_bf16_evidence']:.3f} ({g['evidence']['source']})"
           if "linear_over_bf16_evidence" in g else "-")
    print(f"15a\t| {m} | {s['bf16']['step_ms']:,.1f} ({100 * s['bf16']['spread']:.2f}%) | "
          f"{s[arm]['step_ms']:,.1f} ({100 * s[arm]['spread']:.2f}%) | {g['arm_over_bf16']:.3f} | "
          f"{lin} | {100 * g['J']:.1f}% | {g['rule']}: {g['threshold']:.3f} -> "
          f"{'faster' if g['faster'] else 'not faster'} | {100 * g['bf16_vs_evidence']:+.2f}% |")
    eng = ("tensor", "vector", "scalar", "gpsimd", "hbm_dma")
    for b in d["blocks"]:
        e = {r["engine_class"]: r["ms"] for r in d["engines"] if (r["arm"], r["blk"]) == (b["arm"], b["blk"])}
        print(f"15b\t| {m} | {b['arm']} | {b['blk']} ({b['kind']}) x{b['execs_per_step']} | {b['block_ms']:.2f} | "
              + " | ".join(f"{e.get(x, 0):.2f}" for x in eng)
              + f" | {b['psum_readout_copy_ms']:.2f} / {b['psum_readout_fused_ms']:.2f} | "
              f"{b['bottleneck']} {100 * b['bottleneck_share']:.1f}% |")
    for a in d["attention_summary"]:
        ph = a["phases_ms_per_step"]

        def cell(p, engines=("tensor", "vector", "scalar", "gpsimd", "hbm_dma")):
            parts = [f"{x[0].upper()} {ph.get(p, {}).get(x, 0):.1f}" for x in engines
                     if ph.get(p, {}).get(x, 0) >= 0.05]
            return ", ".join(parts) or "-"
        share = f"{100 * a['v_quantize_share']:.1f}%" if a["v_quantize_share"] else "-"
        print(f"15b-phase\t| {m} | {a['arm']} | " + " | ".join(cell(p) for p in PHASES)
              + f" | {a['attention_engine_ms_per_step']:.1f} | {share} |")
    q = d["quality"]
    ev = q["linear_vs_bf16_evidence"]
    ev_cell = (f"{ev['psnr']:.2f}" + (f" ({ev['psnr_worst']:.2f})" if ev["psnr_worst"] else "")
               + f" / {ev['ssim']:.3f} / {ev['lpips']:.3f} ({ev['section']}, {ev['host']})") if ev else "-"
    fp = q["fp32_evidence"]
    here = q["bf16_vs_fp32_here"]
    fp_cell = (f"{_fmt_q(q['arm_vs_fp32'])}; bf16 here {here['psnr'] if here else float('nan'):.2f}, "
               f"evidence bf16 {fp['bf16_psnr']:.2f} / linear {fp['linear_psnr']:.2f}") if fp else "-"
    notes = "; ".join(x for x in (q["caveat"], q["k_smoothing"]) if x)
    print(f"15c\t| {m} | {_fmt_q(q['arm_vs_bf16'])} | {ev_cell} | {q['floor_evidence'] or '-'} | "
          f"{fp_cell} | {notes or '-'} |")
    md5 = q["md5"] or {}
    ev_md5, ev_src = q["bf16_md5_evidence"]
    b1 = (md5.get("bf16/timed_1") or "-")[:8]
    lat = q["arm_latents_bit_identical"] or {}
    print(f"15d\t| {m} | `{b1}` ({'=' if b1 == ev_md5 else '!='} evidence `{ev_md5}`, {ev_src}); "
          f"two requests {'equal' if (q['deterministic'] or {}).get('bf16') else 'DIFFER'} | "
          f"`{(md5.get(f'{arm}/timed_1') or '-')[:8]}`; timed 2 and gen2 "
          f"{'equal' if (q['deterministic'] or {}).get(arm) else 'DIFFER'} | "
          f"{'bit-identical' if lat and all(v.get('bit_identical') for v in lat.values()) else 'no latents' if not lat else 'DIFFER'}"
          f" | {_finite_text(q['arm_latents_finite'])} |")


def _finite_text(f: dict | None) -> str:
    vals = list((f or {}).values())
    return ("no latents kept" if all(v is None for v in vals) else "finite" if all(vals)
            else "NON-FINITE")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("what", choices=("quality", "dataset"))
    p.add_argument("key", choices=sorted(collect.CONFIGS))
    p.add_argument("arm", choices=("both", "attn"))
    a = p.parse_args(argv)
    if a.what == "quality":
        print(json.dumps(quality(a.key, a.arm), indent=1))
    else:
        dataset(a.key, a.arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
