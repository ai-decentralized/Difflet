"""Multi-prompt output quality for best-combination candidates (any model).

Every held-out prompt of scripts/data/m9_teacache_prompts.tsv ("holdout"
split: neither the benchmark prompt nor a calibration prompt) is generated at
seed 42 by the cell's exact ``difflet generate`` command (one process per
prompt) and saved as benchmark/<device>/logs/quality/<label>_p<i>.<png|mp4>.
(The FLUX images of 2026-10-04 were generated in-process with the same
pipeline kwargs as step_realloop's flux builder.)

    python -m benchmark.combo_quality gen --model wan_2_1 --config tp4tc2 [--prompts 4]
    python -m benchmark.combo_quality score --model wan_2_1 --configs tp4tc2 ... \
        [--ref tp4]

``score`` prints per-label mean / min PSNR vs the reference label's output of
the same prompt (video: per-frame) and writes them into each cell JSON as
``quality_holdout``.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from benchmark.models import json_path, logs_dir, resolve

PROMPTS_TSV = Path(__file__).resolve().parent.parent / "scripts" / "data" / "m9_teacache_prompts.tsv"


def holdout_prompts(n: int) -> list[str]:
    rows = PROMPTS_TSV.read_text(encoding="utf-8").strip().splitlines()[1:]
    return [r.split("\t", 1)[1].strip() for r in rows if r.startswith("holdout\t")][:n]


def _ext(cfg) -> str:
    return ".png" if cfg.output_kind == "image" else ".mp4"


def out_path(label: str, i: int, ext: str = ".png") -> Path:
    d = Path(logs_dir()) / "quality"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{label}_p{i}{ext}"


def gen(args) -> int:
    """One ``difflet generate`` per holdout prompt through the campaign adapter
    (the exact CLI the warm e2e cells run, prompt swapped). The adapter writes
    <logs>/quality/<spec_slug>_out.<ext>; it is renamed to <label>_p<i>.<ext>,
    so the cell's benchmark-prompt output is never touched."""
    from dataclasses import replace

    from benchmark.adapters.trainium import TrainiumAdapter, spec_slug

    cfg = resolve(args.model, args.config)
    ext = _ext(cfg)
    ad = TrainiumAdapter(log_dir=f"{logs_dir()}/quality/gen")
    for i, prompt in enumerate(holdout_prompts(args.prompts)):
        dst = out_path(args.config, i, ext)
        if dst.exists():
            print(f"[quality] {args.config} p{i} exists, skipping", flush=True)
            continue
        r = ad.run_generate(replace(cfg, prompt=prompt))
        produced = Path(logs_dir()) / "quality" / f"{spec_slug(cfg)}_out{ext}"
        produced.replace(dst)
        print(f"[quality] {args.config} p{i} {r['wall_seconds']:.1f}s {prompt!r} -> {dst}",
              flush=True)
    return 0


def score(args) -> int:
    from benchmark.output_parity import compare

    n = len(holdout_prompts(args.prompts))
    ext = _ext(resolve(args.model, args.ref))
    for label in args.configs:
        psnrs = []
        for i in range(n):
            a, b = out_path(args.ref, i, ext), out_path(label, i, ext)
            if a.exists() and b.exists():
                psnrs.append(compare(a, b)["psnr_db"])
        if not psnrs:
            print(f"[quality] {label}: no outputs")
            continue
        rec = {"ref": args.ref, "prompts": n, "psnr_db": [round(x, 2) for x in psnrs],
               "mean_psnr_db": round(statistics.mean(psnrs), 2),
               "min_psnr_db": round(min(psnrs), 2), "seed": 42,
               "source": "benchmark.combo_quality, holdout split of m9_teacache_prompts.tsv"}
        print(f"[quality] {label} vs {args.ref}: mean {rec['mean_psnr_db']} dB, "
              f"min {rec['min_psnr_db']} dB over {len(psnrs)} prompts")
        if label != args.ref:
            jp = Path(json_path(resolve(args.model, label).config_slug))
            if jp.exists():
                d = json.loads(jp.read_text())
                d.setdefault("quality_holdout", {})[args.ref] = rec
                jp.write_text(json.dumps(d, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--config", required=True)
    g.add_argument("--prompts", type=int, default=4)
    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--configs", nargs="+", required=True)
    s.add_argument("--ref", default="tp4")
    s.add_argument("--prompts", type=int, default=4)
    a = p.parse_args()
    return gen(a) if a.cmd == "gen" else score(a)


if __name__ == "__main__":
    raise SystemExit(main())
