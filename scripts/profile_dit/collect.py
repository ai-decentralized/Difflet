#!/usr/bin/env python3
"""Merge the DiT block profile into results.json and the page's datasets (DiT block profile plan,
Task 5).

usage: collect.py [--assets DIR] [--run DIR] [--out FILE] [--page DIR] [--doc FILE]

Reads the committed breakdowns ``<assets>/<key>/<arm>/blk<i>.json``, each lane's
``<run>/<key>/<arm>/system_summary.json`` and ``blk<i>/source.json`` (joined on the NEFF name,
ruling R2), and the DiT step times and linear-MXFP8 quality rows of
``docs/verification/2026-11-07-mxfp8-video-models.md`` (sections 7f, 10, 11, 12), transcribed
below with section and host; a unit test finds every value in its section. Runs nothing on a
device. A block without a breakdown is ``pending``: its numbers stay null, never estimated.

results.json leaves out each block's ``timeline`` (it is in the committed breakdown beside it).
With ``--page``: the page's datasets as JSON row arrays in ``<page>/datasets/<id>.json``, the page
(``page/index.html`` beside this script) and ``<page>/store.json``, the Data artifact's documents
with ``<URL:id>`` where each uploaded dataset's url goes. With ``--doc``: the evidence doc's
tables, between its ``<!-- tables:begin -->`` and ``<!-- tables:end -->`` lines.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1] / "docs" / "verification" / "assets" / "2026-11-13-dit-block-profile"
RUN = Path("/mnt/nvme/difflet-prof/run")

#: key: (model, profiled shape, layout, arms, block NEFFs per arm) — the plan's configurations
CONFIGS = {
    "qwen_image": ("Qwen-Image", "1024x1024", "tp4", ("bf16", "mxfp8"), 1),
    "flux": ("FLUX.1-dev", "1024x1024", "tp4", ("bf16", "mxfp8"), 2),
    "ltx_2": ("LTX-2", "512x768x121", "tp4, CFG sequential", ("bf16", "mxfp8"), 1),
    "wan21_official": ("Wan 2.1 T2V 14B", "480x832x81", "tp4, CFG 5.0 sequential",
                       ("bf16", "mxfp8"), 1),
    "wan22_official": ("Wan 2.2 T2V A14B", "480x832x81", "tp4, CFG 4.0/3.0 sequential",
                       ("bf16", "mxfp8"), 1),
    "hunyuan_video_L": ("HunyuanVideo", "544x960x129", "tp4", ("bf16", "mxfp8"), 2),
    "hv15_official": ("HunyuanVideo-1.5", "480x848x121", "tp2 x CFG2", ("bf16",), 1),
}

#: (key, arm): (DiT step median ms, evidence section, host, steps of the run). Host A is the trn3
#: the profile ran on; section 12 ran on a second trn3, B.
STEP_TIMES = {
    ("qwen_image", "bf16"): (303, "11b", "A", 50), ("qwen_image", "mxfp8"): (240, "11b", "A", 50),
    ("flux", "bf16"): (221, "11b", "A", 50), ("flux", "mxfp8"): (183, "11b", "A", 50),
    ("ltx_2", "bf16"): (1290, "11b", "A", 50), ("ltx_2", "mxfp8"): (969, "11b", "A", 50),
    ("wan21_official", "bf16"): (10143, "12b", "B", 50),
    ("wan21_official", "mxfp8"): (9958, "12b", "B", 50),
    ("wan22_official", "bf16"): (10151, "12b", "B", 40),
    ("wan22_official", "mxfp8"): (9963, "12b", "B", 40),
    ("hunyuan_video_L", "bf16"): (18493.6, "7f", "A", 50),
    ("hunyuan_video_L", "mxfp8"): (17967.6, "7f", "A", 50),
    ("hv15_official", "bf16"): (9916, "12b", "B", 50),
}

_HV15 = "MXFP8 arm = the G1 + G3 override; HunyuanVideo-1.5 has no MXFP8 default (profiled bf16 only)"
#: The linear-MXFP8 quality rows: against "CPU fp32" (10c: with bf16's own distance) or "device
#: bf16" (7f, 11c, 12c). psnr_worst = the worst frame; floor = bf16 against bf16 at another layout.
QUALITY = [
    # key, section, host, shape, against, psnr, worst, ssim, lpips, latents cos, bf16 psnr,
    # MXFP8 - bf16 psnr (the doc's, from unrounded scores), floor, note
    ("qwen_image", "10c", "A", "1024x1024", "CPU fp32", 37.36, None, 0.992, 0.009, 0.9983, 36.34, 1.02,
     "39-46 dB", "MXFP8 1.02 dB closer to fp32 than bf16"),
    ("ltx_2", "10c", "A", "512x768x121", "CPU fp32", 16.85, 15.54, 0.485, 0.190, 0.9408, 16.55, 0.29,
     "25.9 dB / 0.909 (3a)", "both arms far from fp32 at 121 frames; MXFP8 no worse (video latents)"),
    ("hunyuan_video_L", "10c", "A", "320x512x61", "CPU fp32", 20.59, 17.42, 0.837, 0.168, 0.9809,
     20.79, -0.20, "36.64 dB / 0.957 (7a)", "no worse than bf16 within one clip; both far from fp32"),
    ("flux", "10c", "A", "1024x1024", "CPU fp32", 21.30, None, 0.885, 0.139, None, 23.36, -2.06,
     "29.4 dB on this image (6a)", "one image of a chaotic sampler: not resolved"),
    ("wan21_official", "10c", "A", "480x832x9", "CPU fp32", 33.72, 32.13, 0.947, 0.033, 0.9982,
     36.08, -2.36, "37.0 dB / 0.960 (4a)", "a measured cost; guidance 1.0 (no CFG)"),
    ("wan22_official", "10c", "A", "480x832x9", "CPU fp32", 33.46, 30.97, 0.949, 0.036, 0.9970,
     36.39, -2.93, "not measured with both experts", "a measured cost; both experts, guidance 1.0"),
    ("hv15_official", "10c", "A", "480x848x9", "CPU fp32", 17.41, 16.60, 0.446, 0.237, 0.9565, 24.89,
     -7.47, "none at 9 frames", f"the largest gap of the seven; {_HV15}"),
    ("flux", "11c", "A", "1024x1024", "device bf16", 22.59, None, 0.908, 0.098, None, None, None,
     "29.4 dB on this image (6a)", "FLUX drifts at 50 steps: 24.7 dB mean over 12 images (6b)"),
    ("qwen_image", "11c", "A", "1024x1024", "device bf16", 38.26, None, 0.992, 0.008, None, None, None,
     "39-46 dB", ""),
    ("ltx_2", "11c", "A", "512x768x121", "device bf16", 22.47, None, 0.830, 0.078, None, None, None,
     "25.9 dB / 0.909 (3a)", ""),
    ("wan21_official", "11c", "A", "480x832x9", "device bf16", 33.90, None, 0.945, 0.034, None, None,
     None, "37.0 dB / 0.960 (4a)", "guidance 1.0 (no CFG)"),
    ("wan22_official", "11c", "A", "480x832x9", "device bf16", 33.97, None, 0.950, 0.036, None, None,
     None, "not measured with both experts", "guidance 1.0 (no CFG)"),
    ("hunyuan_video_L", "11c", "A", "320x512x61", "device bf16", 31.09, None, 0.938, 0.050, None, None,
     None, "36.64 dB / 0.957 (7a)", ""),
    ("hv15_official", "11c", "A", "480x848x9", "device bf16", 17.32, None, 0.439, 0.234, None, None,
     None, "none at 9 frames", _HV15),
    ("wan21_official", "12c", "B", "480x832x81", "device bf16", 15.08, 12.96, 0.391, 0.351, None, None,
     None, "22.75 (18.31) / 0.784 / 0.098, tp8", "fails the floor-relative screen on all three"),
    ("wan22_official", "12c", "B", "480x832x81", "device bf16", 14.29, 13.27, 0.398, 0.275, None, None,
     None, "14.83 (13.74) / 0.439 / 0.254, tp8",
     "passes the screen, but its two bf16 runs are themselves 14.8 dB apart"),
    ("hv15_official", "12c", "B", "480x848x121", "device bf16", 12.65, 12.20, 0.183, 0.504, None, None,
     None, "16.94 (15.94) / 0.362 / 0.326, tp4 x CFG2", f"fails the screen on PSNR and SSIM; {_HV15}"),
    ("hunyuan_video_L", "7f", "A", "544x960x129", "device bf16", 26.46, 20.07, 0.907, 0.128, 0.98981,
     None, None, "none at this shape (320x512x61: 36.64 dB)",
     "misses the 320x512x61 screen's PSNR and LPIPS; no floor or fp32 reference at this shape"),
]
_QUALITY_FIELDS = ("key", "section", "host", "shape", "against", "psnr", "psnr_worst", "ssim",
                   "lpips", "latents_cos", "bf16_psnr", "delta_psnr", "floor", "note")

#: attention_detail modules: Plan B routes self/joint attention at head dim 128 (LTX-2's
#: audio_attn1 has 64); cross-attention stays bf16
ROUTED = {"attn", "attn1"}
CROSS = {"attn2", "audio_attn2", "audio_to_video_attn", "video_to_audio_attn"}
_SHARES = ("bottleneck", "attention_share", "addressable_attention_share", "cross_attention_share",
           "rope_share", "rope_engine")


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _blocks_text(config: dict) -> str:
    """"dual x19, single x38": each block's kind and executions per step, as far as known."""
    return ", ".join((b["kind"] or f"blk{b['blk']}") + (
        f" x{b['execs_per_step']}" if b["execs_per_step"] is not None else "")
        for b in config["blocks"])


def _add(acc: dict, d: dict, w: float) -> dict:
    """acc += w x d, for nested {key: number or dict}."""
    for k, v in d.items():
        if isinstance(v, dict):
            _add(acc.setdefault(k, {}), v, w)
        elif v is not None:
            acc[k] = acc.get(k, 0.0) + w * v
    return acc


def shares(engines: dict, components: dict, detail: dict, block_ms: float, mfu) -> dict:
    """The bottleneck (largest engine_class union; DMA is already ``hbm_dma``) and the shares of
    ``block_ms`` that attention, Plan B's routed attention and cross-attention hold on it; RoPE's
    share is on its own busiest engine (it never runs on TensorE, the usual bottleneck)."""
    ec = max(engines, key=engines.get)
    rope = components.get("rope", {})
    rope_ec = max(rope, key=rope.get) if rope else None
    attention = components.get("attention", {}).get(ec, 0.0)
    by_module = {m: sum(p.get(ec, 0.0) for p in phases.values()) for m, phases in detail.items()}
    total = sum(by_module.values())

    def part(modules):  # attention's union split by the modules' per-phase unions
        return attention * sum(by_module.get(m, 0.0) for m in modules) / total / block_ms if total \
            else 0.0
    return {"bottleneck": {"engine_class": ec, "share": engines[ec] / block_ms, "mfu": mfu},
            "attention_share": attention / block_ms,
            "addressable_attention_share": part(ROUTED), "cross_attention_share": part(CROSS),
            "rope_share": rope.get(rope_ec, 0.0) / block_ms, "rope_engine": rope_ec}


def _block(assets: Path, run: Path, key: str, arm: str, i: int, summary: dict | None) -> dict:
    source = _json(run / key / arm / f"blk{i}" / "source.json") or {}
    names = (summary or {}).get("blocks", [])
    neff = source.get("neff_name") or (names[i - 1] if len(names) >= i else None)
    system = (summary or {}).get("neffs", {}).get(neff, {})
    b = _json(assets / key / arm / f"blk{i}.json")
    block = {"blk": i, "kind": source.get("kind"), "neff_name": neff,
             "status": "ok" if b else "pending", "execs_per_step": system.get("execs_per_step"),
             "in_model_median_ms": system.get("in_model_median_ms")}
    if b:
        block |= b | shares(b["engines_ms"], b["components"], b["attention_detail"], b["block_ms"],
                            b["mfu"])
    return block


def _config(assets: Path, run: Path, key: str, arm: str, spec: tuple, step: tuple | None) -> dict:
    model, shape, layout, _, n = spec
    summary = _json(run / key / arm / "system_summary.json")
    blocks = [_block(assets, run, key, arm, i, summary) for i in range(1, n + 1)]
    known = [b["execs_per_step"] * b["in_model_median_ms"] for b in blocks
             if b["execs_per_step"] is not None]
    blocks_ms = sum(known) if blocks and len(known) == n else None
    step_ms = step[0] if step else None
    out = {"key": key, "model": model, "arm": arm, "shape": shape, "layout": layout,
           "status": "ok" if all(b["status"] == "ok" for b in blocks) else "pending",
           "step_ms": step_ms,
           "step_source": dict(zip(("section", "host", "steps"), step[1:])) if step
           else "not measured",
           "profile_step_ms": summary["step"]["wall_ms"] if summary else None,
           "blocks": blocks, "blocks_ms": blocks_ms,
           "non_block_ms": (summary or {}).get("non_block_ms"),
           "host_gap_ms": (summary or {}).get("host_gap_ms"),
           "outside_blocks_ms": step_ms - blocks_ms if None not in (step_ms, blocks_ms) else None,
           **dict.fromkeys(_SHARES)}
    weights = [b["execs_per_step"] for b in blocks]
    if out["status"] == "ok" and n == 1:
        out |= {f: blocks[0][f] for f in _SHARES}
    elif out["status"] == "ok" and None not in weights:
        # the step's blocks together: each block's times x its executions per step
        eng, comp, det, ms, mfu = {}, {}, {}, 0.0, 0.0
        for b, w in zip(blocks, weights):
            _add(eng, b["engines_ms"], w)
            _add(comp, {c: b["components"].get(c, {}) for c in ("attention", "rope")}, w)
            _add(det, b["attention_detail"], w)
            ms += w * b["block_ms"]
            mfu = None if mfu is None or b["mfu"] is None else mfu + w * b["block_ms"] * b["mfu"]
        out |= shares(eng, comp, det, ms, None if mfu is None else mfu / ms)
    return out


def collect(assets: Path = ASSETS, run: Path = RUN, configs: dict = CONFIGS,
            steps: dict = STEP_TIMES) -> dict:
    """{"configs": [one per key x arm], "quality": [rows]}; blocks keep their ``timeline``."""
    return {
        "configs": [_config(assets, run, key, arm, spec, steps.get((key, arm)))
                    for key, spec in configs.items() for arm in spec[3]],
        "quality": [dict(zip(_QUALITY_FIELDS, q)) | {
            "model": CONFIGS[q[0]][0], "profiled_shape": q[3] == CONFIGS[q[0]][1]} for q in QUALITY],
    }


def datasets(results: dict) -> dict[str, list[dict]]:
    """The page's datasets: {id: rows}, one ``timeline_<key>_<arm>_blk<i>`` per measured block."""
    out = {k: [] for k in ("overview", "blocks", "engines", "components", "attention_phases",
                           "memory", "unattributed")}
    for c in results["configs"]:
        k = {"key": c["key"], "arm": c["arm"]}
        bn = c["bottleneck"] or {}
        out["overview"].append(k | {
            f: c[f] for f in ("model", "shape", "layout", "status", "step_ms", "profile_step_ms",
                              "blocks_ms", "outside_blocks_ms", "non_block_ms", "host_gap_ms",
                              "attention_share", "addressable_attention_share",
                              "cross_attention_share", "rope_share", "rope_engine")} | {
            f"step_{f}": c["step_source"][f] if isinstance(c["step_source"], dict) else None
            for f in ("section", "host", "steps")} | {
            "blocks": _blocks_text(c),
            "bottleneck": bn.get("engine_class"), "bottleneck_share": bn.get("share"),
            "mfu": bn.get("mfu")})
        for b in c["blocks"]:
            kb = k | {"blk": f"blk{b['blk']}"}
            # a pending block has no breakdown fields: null
            bn, th = b.get("bottleneck", {}), b.get("throttle", {})
            out["blocks"].append(kb | {
                f: b.get(f) for f in ("kind", "status", "neff_name", "execs_per_step",
                                      "in_model_median_ms", "block_ms", "replay_spread_ms", "mfu",
                                      "mbu", "attention_share", "addressable_attention_share",
                                      "cross_attention_share", "rope_share", "rope_engine",
                                      "dge_notifs")} | {
                "bottleneck": bn.get("engine_class"), "bottleneck_share": bn.get("share"),
                "avg_util_limit": th.get("avg_util_limit"),
                "limited_time_share": th.get("limited_time_share"),
                "warnings": "; ".join(b.get("warnings", []))})
            if b["status"] != "ok":
                continue
            out["engines"] += [kb | {"engine_class": e, "ms": ms, "block_ms": b["block_ms"]}
                               for e, ms in b["engines_ms"].items()]
            out["components"] += [kb | {"component": comp, "engine_class": e, "ms": ms}
                                  for comp, d in b["components"].items() for e, ms in d.items()]
            out["attention_phases"] += [
                kb | {"module": m or "(no module path)", "routed": m in ROUTED, "phase": p,
                      "engine_class": e, "ms": ms}
                for m, phases in b["attention_detail"].items() for p, d in phases.items()
                for e, ms in d.items()]
            out["memory"] += [kb | {"metric": m, "engine_class": "", "value": v, "unit": "bytes"}
                              for m, v in b["memory_bytes"].items()]
            out["memory"] += [kb | {"metric": f"psum_readout_{kind}", "engine_class": e,
                                    "value": ms, "unit": "ms"}
                              for kind in ("copy", "fused")
                              for e, ms in b[f"psum_readout_{kind}_ms_by_engine"].items()]
            out["unattributed"] += [kb | {"source": s, "engine_class": e, "ms": ms}
                                    for s, d in b["unattributed_by_source"].items()
                                    for e, ms in d.items()]
            out[f"timeline_{c['key']}_{c['arm']}_blk{b['blk']}"] = b["timeline"]
    out["quality"] = results["quality"]
    return out


#: Sources titles and descriptions of the datasets (a time line's is built per block)
DESCRIBED = {
    "overview": ("Overview, one row per model and arm",
                 "DiT step median from the evidence doc (section, host, steps), the blocks' device "
                 "time per step (executions x in-model median from the system profile), the rest "
                 "outside the blocks, and the bottleneck engine with the attention, Plan B "
                 "addressable and cross-attention shares on it, RoPE's on its own busiest engine "
                 "(blocks weighted by executions)."),
    "blocks": ("Blocks, one row per block NEFF",
               "Replayed block wall time and spread (exec 4 vs exec 8), executions per step, "
               "in-model median, bottleneck, MFU/MBU, throttle, DGE notifications; pending = not "
               "yet captured."),
    "engines": ("Engine busy time per block",
                "Union of instruction intervals per physical core (semaphore waits excluded), mean "
                "over the two physical cores; DMA instructions as hbm_dma; collectives from CcOp."),
    "components": ("Component time per engine",
                   "Busy time of each block component on each engine class; unattributed = rows "
                   "with no module path."),
    "attention_phases": ("Attention phases per module",
                         "QK matmul, mask/softmax, PV matmul and other per attention module and "
                         "engine; routed = Plan B's self/joint attention (head dim 128)."),
    "memory": ("Memory traffic and PSUM read-out",
               "Bytes moved per block from the profile summary, and the VectorE/ScalarE time that "
               "reads PSUM out: pure copies (copy) and fused compute (fused), in ms."),
    "unattributed": ("Unattributed time by source",
                     "Rows with no module path, split by the source file (and function) they name."),
    "quality": ("Quality of linear MXFP8",
                "PSNR / SSIM / LPIPS of the linear-MXFP8 output against device bf16 or CPU fp32, "
                "from evidence sections 7f, 10c, 11c, 12c, with the shape each was measured at."),
}


def store(results: dict, data: dict, html: str, now: str) -> dict:
    """The Data artifact's documents; ``<URL:id>`` is replaced by each uploaded file's url."""
    docs = {"dash/meta": {"title": "DiT Block Profile"}}
    blocks = {f"timeline_{c['key']}_{c['arm']}_blk{b['blk']}": (c, b)
              for c in results["configs"] for b in c["blocks"]}
    for i in data:
        c, b = blocks.get(i, ({}, {}))
        title, description = DESCRIBED.get(i) or (
            f"Time line, {c.get('model')} {c.get('arm')} blk{b.get('blk')} ({b.get('kind')})",
            "One block replay on physical core 0: busy runs per component and engine class in µs "
            "from the block's start; gaps under max(5 µs, block/4000) merged, busy_us = busy time "
            "inside the segment.")
        docs[f"datasets/{i}"] = {
            "title": title, "description": description,
            "source": {"kind": "file", "url": f"<URL:{i}>", "name": f"{i}.json", "format": "json"},
            "fixed": True, "updated": {"at": now, "by": "claude"}, "edit": {"via": "claude", "at": now}}
    docs["files/index.html"] = {"text": html}
    return docs


def markdown(results: dict) -> str:
    """The evidence doc's tables: overview, engines per block, components and PSUM, quality."""
    def n(v, d=2):
        return "pending" if v is None else f"{v:,.{d}f}" if isinstance(v, float) else f"{v:,}"

    def pc(v):
        return "pending" if v is None else f"{100 * v:.1f}%"

    def table(head, rows):
        return "\n".join([f"| {' | '.join(head)} |", f"|{'---|' * len(head)}",
                          *(f"| {' | '.join(map(str, r))} |" for r in rows)])
    engines = ("tensor", "vector", "scalar", "gpsimd", "sync", "hbm_dma", "collective")
    over, eng, comp = [], [], []
    for c in results["configs"]:
        src, bn = c["step_source"], c["bottleneck"] or {}
        over.append([c["model"], c["shape"], c["layout"], c["arm"],
                     f"{n(c['step_ms'], 1)} ({src['section']}, {src['host']}, {src['steps']})"
                     if isinstance(src, dict) else "not measured",
                     _blocks_text(c),
                     n(c["blocks_ms"], 1), n(c["outside_blocks_ms"], 1),
                     f"{bn['engine_class']} {pc(bn['share'])}" if bn else "pending",
                     pc(bn.get("mfu")), pc(c["attention_share"]),
                     pc(c["addressable_attention_share"]), pc(c["cross_attention_share"]),
                     f"{pc(c['rope_share'])} {c['rope_engine'] or ''}".strip()])
        for b in c["blocks"]:
            name = [c["model"], c["arm"], f"blk{b['blk']}" + (f" ({b['kind']})" if b["kind"] else "")]
            if b["status"] != "ok":
                eng.append(name + ["pending"] * (len(engines) + 4))
                continue
            e, ec = b["engines_ms"], b["bottleneck"]["engine_class"]
            eng.append(name + [n(b["block_ms"]), n(b["replay_spread_ms"], 3),
                               *(n(e[k]) if k in e else "-" for k in engines),
                               f"{pc(b['throttle']['avg_util_limit'])} / "
                               f"{pc(b['throttle']['limited_time_share'])}",
                               "yes" if b["dge_notifs"] else "no"])
            top = sorted(((k, v.get(ec, 0.0)) for k, v in b["components"].items()),
                         key=lambda kv: -kv[1])[:5]
            comp.append(name + [ec, ", ".join(f"{k} {n(v)}" for k, v in top),
                                f"{n(b['psum_readout_copy_ms'])} / {n(b['psum_readout_fused_ms'])}"])
    quality = [[q["model"], q["shape"] + (" (profiled)" if q["profiled_shape"] else ""),
                f"{q['section']}, {q['host']}", q["against"],
                n(q["psnr"]) + (f" ({n(q['psnr_worst'])})" if q["psnr_worst"] is not None else ""),
                n(q["ssim"], 3), n(q["lpips"], 3),
                "-" if q["latents_cos"] is None else max(f"{q['latents_cos']:.4f}", str(q["latents_cos"]), key=len),
                "-" if q["bf16_psnr"] is None else n(q["bf16_psnr"]),
                "-" if q["delta_psnr"] is None else f"{q['delta_psnr']:+.2f}",
                q["floor"], q["note"]] for q in results["quality"]]
    return "\n\n".join([
        "**Overview** (shares of block time on the bottleneck engine; RoPE on its own busiest "
        "engine):",
        table(["model", "shape", "layout", "arm", "DiT step ms (section, host, steps)",
               "blocks per step", "blocks ms", "outside ms", "bottleneck", "MFU", "attention",
               "Plan B addressable", "cross-attention", "RoPE"], over),
        "**Engine busy time per block execution** (ms; union per physical core, mean of the two):",
        table(["model", "arm", "block", "wall ms", "replay spread ms", "TensorE", "VectorE",
               "ScalarE", "GpSimd", "Sync", "HBM DMA", "collectives",
               "throttle: avg util limit / time limited", "DGE notifications"], eng),
        "**Top components on the bottleneck engine, and PSUM read-out** (ms):",
        table(["model", "arm", "block", "bottleneck", "top five components on it",
               "PSUM read-out copy / fused (VectorE + ScalarE)"], comp),
        "**Quality of linear MXFP8** (evidence doc 2026-11-07):",
        table(["model", "shape", "section, host", "against", "PSNR dB (worst)", "SSIM", "LPIPS",
               "latents cos", "bf16 vs fp32 PSNR", "MXFP8 - bf16 dB", "floor", "note"], quality)])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--assets", type=Path, default=ASSETS)
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--out", type=Path, default=ASSETS / "results.json")
    p.add_argument("--page", type=Path, help="write the page's datasets, index.html, store.json")
    p.add_argument("--doc", type=Path, help="fill the tables of this evidence doc")
    a = p.parse_args(argv)
    results = collect(a.assets, a.run)
    lean = {**results, "configs": [{**c, "blocks": [{k: v for k, v in b.items() if k != "timeline"}
                                                     for b in c["blocks"]]}
                                   for c in results["configs"]]}
    a.out.write_text(json.dumps(lean, indent=1) + "\n")
    print(a.out)
    if a.page:
        (a.page / "datasets").mkdir(parents=True, exist_ok=True)
        data = datasets(results)
        for i, rows in data.items():
            (a.page / "datasets" / f"{i}.json").write_text(json.dumps(rows))
        shutil.copy(HERE / "page" / "index.html", a.page / "index.html")
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        docs = store(results, data, (HERE / "page" / "index.html").read_text(), now)
        (a.page / "store.json").write_text(json.dumps(docs, indent=1) + "\n")
        print(a.page / "store.json", f"{len(data)} datasets")
    if a.doc:
        head, rest = a.doc.read_text().split("<!-- tables:begin -->\n")
        tail = rest.split("<!-- tables:end -->")[1]
        a.doc.write_text(f"{head}<!-- tables:begin -->\n{markdown(results)}\n<!-- tables:end -->{tail}")
        print(a.doc)


if __name__ == "__main__":
    main()
