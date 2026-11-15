#!/usr/bin/env python3
"""The page's "MXFP8 attention: bf16 vs both" section (the MXFP8 attention plan's Task 6): the
dataset attn_runs.json from the per-model datasets, its store doc, the page with the section, and
the ArtifactData write plan (printed, and in writes.json).

    build_attn.py --out DIR [--src DIR] [--url /_blob/<id>] [--runs-version N] [--page-version N]

--src           the per-model datasets ``scripts/mx_attention/task6.py dataset`` writes (default:
                their committed copies, docs/verification/assets/2026-11-13-dit-block-profile/
                mxattn/task6/page)
--out           where attn_runs.json, the store docs (datasets__attn_runs.json,
                files__index.html.json) and writes.json go
--url           the uploaded attn_runs.json; without it the dataset doc holds <URL:attn_runs>
--runs-version  datasets/attn_runs exists at version N (the write is pinned to it); absent: a new doc
--page-version  also write files/index.html, pinned to version N; only when the section changed:
                it draws whatever rows the dataset has

The page is this directory's index.html with any earlier attention section taken out and
attn_section.html (before the Method foot) and attn_section.js (at the end of the page's script)
put in, so a rerun is idempotent; index.html is rewritten with the result, so the repository holds
the page as it is published. Published 2026-11-15: files/index.html v5, datasets/attn_runs v4
(seven rows, the committed attn_runs.json beside results.json).
"""
import argparse
import datetime
import html.parser
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PAGE = HERE / "index.html"
SRC = REPO / "docs/verification/assets/2026-11-13-dit-block-profile/mxattn/task6/page"
DS = "attn_runs"
PHASES = ("qk_matmul", "mask_softmax", "exp", "pv_matmul", "qk_quantize", "v_quantize",
          "p_quantize", "other")
HTML_AT, JS_AT = '  <details id="method">', "})();\n</script>"
HTML_FROM, JS_FROM = '  <section id="mxattn"', "  // ---- MXFP8 attention: bf16 vs both"


def row(d):
    arm, g, q, st = d["arm"], d["g3"], d["quality"], d["steps"]
    summ = {s["arm"]: s for s in d["attention_summary"]}
    session = "linear_over_bf16_session" in g
    r = {"key": d["key"], "model": d["model"], "shape": d["shape"], "layout": d["layout"], "arm": arm,
         "bf16_ms": st["bf16"]["step_ms"], "bf16_spread": st["bf16"]["spread"],
         "arm_ms": st[arm]["step_ms"], "arm_spread": st[arm]["spread"],
         "spread": max(st["bf16"]["spread"], st[arm]["spread"]), "J": g["J"], "arm_over_bf16": g["arm_over_bf16"],
         "linear_over_bf16": g["linear_over_bf16_session"] if session else g.get("linear_over_bf16_evidence"),
         "linear_source": "this session" if session
         else g["evidence"]["source"] if "linear_over_bf16_evidence" in g else None,
         "bf16_vs_evidence": g["bf16_vs_evidence"],
         "g3_rule": g["rule"], "g3_threshold": g["threshold"], "g3_faster": g["faster"]}
    kinds = sorted((b for b in d["blocks"] if b["arm"] == "bf16"), key=lambda b: b["blk"])
    assert 1 <= len(kinds) <= 2, (d["key"], len(kinds))
    for i in (1, 2):
        b = kinds[i - 1] if i <= len(kinds) else {}
        r[f"blk{i}_kind"], r[f"blk{i}_execs"] = b.get("kind"), b.get("execs_per_step")
    for a, pre in (("bf16", "bf16"), (arm, "arm")):
        blocks = {b["blk"]: b for b in d["blocks"] if b["arm"] == a}
        assert sorted(blocks) == [b["blk"] for b in kinds], (d["key"], a, sorted(blocks))
        r[f"{pre}_blocks_ms"] = sum(b["execs_per_step"] * b["in_model_median_ms"] for b in blocks.values())
        for i in (1, 2):
            r[f"{pre}_blk{i}_ms"] = blocks.get(f"blk{i}", {}).get("block_ms")
        s = summ[a]
        r[f"{pre}_bottleneck"] = s["bottleneck"]["engine_class"]
        r[f"{pre}_bottleneck_share"] = s["bottleneck"]["share"]
        r[f"{pre}_mfu"] = s["bottleneck"]["mfu"]
        r[f"{pre}_attention_ms"] = s["attention_engine_ms_per_step"]
        for p in PHASES:
            r[f"{pre}_{p}_ms"] = sum(s["phases_ms_per_step"].get(p, {}).values())
    r["v_quantize_share"] = summ[arm]["v_quantize_share"]
    vs, lin, fp = q["arm_vs_bf16"] or {}, q["linear_vs_bf16_evidence"] or {}, q["arm_vs_fp32"] or {}
    ev = q["fp32_evidence"] or {}
    r.update({"psnr": vs.get("psnr"), "psnr_worst": vs.get("psnr_worst"), "ssim": vs.get("ssim"),
              "lpips": vs.get("lpips"), "latents_cos": vs.get("latents_cos"), "latents_rel_l2": vs.get("latents_rel_l2"),
              "lin_psnr": lin.get("psnr"), "lin_psnr_worst": lin.get("psnr_worst"), "lin_ssim": lin.get("ssim"),
              "lin_lpips": lin.get("lpips"), "lin_source": f"{lin['section']}, {lin['host']}" if lin else None,
              "floor": q["floor_evidence"],
              "fp32_psnr": fp.get("psnr"), "fp32_psnr_worst": fp.get("psnr_worst"), "fp32_ssim": fp.get("ssim"),
              "fp32_lpips": fp.get("lpips"), "fp32_latents_cos": fp.get("latents_cos"),
              "bf16_fp32_psnr": (q["bf16_vs_fp32_here"] or {}).get("psnr"),
              "ev_bf16_fp32_psnr": ev.get("bf16_psnr"), "ev_linear_fp32_psnr": ev.get("linear_psnr"),
              "ev_linear_fp32_ssim": ev.get("linear_ssim"),
              "caveat": q["caveat"], "below_floor": q["caveat"] is not None,
              "k_smoothing": q["k_smoothing"], "bounded": (q["k_smoothing"] or "").startswith("bounded")})
    return r


class Tags(html.parser.HTMLParser):
    """Tag balance, and every data-source / data-field pair."""
    VOID = {"meta", "br", "hr", "img", "input", "link", "area", "base", "col", "embed", "source", "track", "wbr"}

    def __init__(self):
        super().__init__()
        self.stack, self.bad, self.marks = [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "data-source" in a or "data-field" in a:
            self.marks.append((a.get("data-source"), a.get("data-field")))
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.bad.append((tag, self.getpos()))


def tags(text):
    t = Tags()
    t.feed(text)
    t.close()
    assert not t.bad and not t.stack, (t.bad[:5], t.stack[:5])
    return t.marks


def without_section(page: str) -> str:
    """The page with the attention section taken out: each part runs from its marker to the anchor
    it was put in front of."""
    for start, anchor in ((HTML_FROM, HTML_AT), (JS_FROM, JS_AT)):
        i = page.find(start)
        if i >= 0:
            page = page[:i] + page[page.index(anchor, i):]
    assert page.count(HTML_AT) == 1 and page.count(JS_AT) == 1 and 'id="mxattn"' not in page
    return page


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--url")
    ap.add_argument("--runs-version", type=int)
    ap.add_argument("--page-version", type=int)
    a = ap.parse_args()
    assert a.url is None or re.fullmatch(r"/_blob/[0-9a-f]{32}", a.url), a.url
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    a.out.mkdir(parents=True, exist_ok=True)

    # (a) one row per model, every row with the same columns
    rows = [row(json.loads(f.read_text())) for f in sorted(a.src.glob("*.json"))]
    assert len({r["key"] for r in rows}) == len(rows)
    assert all(list(r) == list(rows[0]) for r in rows)
    with open(a.out / f"{DS}.json", "w") as f:
        json.dump(rows, f, indent=1)

    # (b) its dataset doc, as prepare.py writes them
    doc = {"title": "MXFP8 attention, bf16 against both",
           "description": "Task 6, one row per model at its Plan A profile shape: DiT step of bf16 and of the quantized arm "
                          "(both = --quant mxfp8 --attn-quant qk+pv; attn = --attn-quant qk+pv alone for HunyuanVideo-1.5), "
                          "spreads, J and G3 (reported only) against linear alone (evidence section and host, or this session); "
                          "per arm (bf16_, arm_) the blocks' time per step (executions x in-model median), each block's replay "
                          "wall time, the bottleneck engine, the attention component's engine ms per step and the routed "
                          "attention's phases in ms per step, engines added; quality against bf16 and, for Qwen-Image, FLUX and "
                          "LTX-2, CPU fp32, beside the evidence doc's linear-alone (lin_), floor and fp32 (ev_) numbers; "
                          "below_floor = not a quality reading; bounded = K smoothing over padded keys (BR36).",
           "source": {"kind": "file", "url": a.url or f"<URL:{DS}>", "name": f"{DS}.json", "format": "json"},
           "fixed": True, "updated": {"at": now, "by": "claude"}, "edit": {"via": "claude", "at": now}}
    (a.out / f"datasets__{DS}.json").write_text(json.dumps(doc))

    # (c) the page: index.html without its attention section, plus the section and its drawings
    base = without_section(PAGE.read_text())
    sec, js = (HERE / "attn_section.html").read_text(), (HERE / "attn_section.js").read_text()
    assert sec.startswith(HTML_FROM) and js.startswith(JS_FROM)
    page = base.replace(HTML_AT, sec + HTML_AT).replace(JS_AT, js + JS_AT)
    (a.out / "files__index.html.json").write_text(json.dumps({"text": page, "edit": {"via": "claude", "at": now}}))
    PAGE.write_text(page)

    # self-check: valid JSON docs, balanced tags, marks that exist, the rest byte-identical
    for f in a.out.glob("*.json"):
        json.loads(f.read_text())
    assert without_section(page) == base and page.replace(sec, "").replace(js, "") == base
    tags(page)
    cols = set(rows[0]) if rows else set()
    bad = [m for m in tags(sec) if m[0] != DS or (m[1] is not None and rows and m[1] not in cols)]
    assert not bad, bad
    used = set(re.findall(r"\br\.(\w+)", js)) | {f"{pre}_{p}_ms" for pre in ("bf16", "arm") for p in PHASES}
    assert not rows or used <= cols, sorted(used - cols)

    # (d) the write plan: the dataset first, so the page never names a missing one
    plan = [{"op": "set", "collection": "datasets", "doc_id": DS,
             "file_path": str(a.out / f"datasets__{DS}.json")}]
    if a.runs_version is not None:
        plan[0]["if_version"] = a.runs_version
    if a.page_version is not None:
        plan.append({"op": "set", "collection": "files", "doc_id": "index.html",
                     "file_path": str(a.out / "files__index.html.json"), "if_version": a.page_version})
    (a.out / "writes.json").write_text(json.dumps(plan, indent=1))
    print(f"{len(rows)} of 7 models: {', '.join(r['key'] for r in rows)}; "
          f"{DS}.json {(a.out / f'{DS}.json').stat().st_size} bytes")
    if not a.url:
        print(f"no --url: the dataset doc holds <URL:{DS}>; upload {a.out / f'{DS}.json'}, then run again with --url")
    print(json.dumps(plan, indent=1))


if __name__ == "__main__":
    main()
