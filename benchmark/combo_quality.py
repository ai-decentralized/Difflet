"""Multi-prompt output quality for best-combination candidates (flux).

One process per label: the pipeline is built once exactly as step_realloop's
flux builder does (same artifact, TeaCache and TAEF1 kwargs), then every
held-out prompt of scripts/data/m9_teacache_prompts.tsv ("holdout" split:
neither the benchmark prompt nor a calibration prompt) is generated at seed
42 and saved as benchmark/<device>/logs/quality/<label>_p<i>.png.

    python -m benchmark.combo_quality gen --model flux_1_dev --config tp4tc2 [--prompts 4]
    python -m benchmark.combo_quality score --model flux_1_dev --configs tp4tc2 ... \
        [--ref tp4]

``score`` prints per-label mean / min PSNR vs the reference label's image of
the same prompt and writes them into each cell JSON as ``quality_holdout``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from benchmark.models import json_path, logs_dir, resolve

PROMPTS_TSV = Path(__file__).resolve().parent.parent / "scripts" / "data" / "m9_teacache_prompts.tsv"


def holdout_prompts(n: int) -> list[str]:
    rows = PROMPTS_TSV.read_text(encoding="utf-8").strip().splitlines()[1:]
    return [r.split("\t", 1)[1].strip() for r in rows if r.startswith("holdout\t")][:n]


def out_path(label: str, i: int) -> Path:
    d = Path(logs_dir()) / "quality"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{label}_p{i}.png"


def gen(args) -> int:
    import torch

    from benchmark.step_realloop import _taef1_app_kwargs, _teacache_app_kwargs
    from difflet.pipeline.difflet_pipeline import DiffletPipeline
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    cfg = resolve(args.model, args.config)
    assert cfg.model_type == "flux", "combo_quality gen is flux-only"
    parallel = DiffletParallelConfig(tp_degree=cfg.tp, cp_degree=cfg.cp, cp_mode=cfg.cp_mode,
                                     cfg_parallel_enabled=cfg.cfg_parallel, sp_enabled=cfg.sp)
    pipe = DiffletPipeline.from_pretrained(
        cfg.model_id, model_type="flux", parallel=parallel, dtype=torch.bfloat16,
        height=cfg.height, width=cfg.width,
        compile_cache_dir=str(Path("~/.cache/difflet").expanduser()),
        revision=cfg.revision, skip_compile=True,
        application_kwargs={**_teacache_app_kwargs(cfg), **_taef1_app_kwargs(cfg)})
    for i, prompt in enumerate(holdout_prompts(args.prompts)):
        t0 = time.perf_counter()
        img = pipe(prompt=prompt, num_inference_steps=cfg.steps,
                   height=cfg.height or 1024, width=cfg.width or 1024,
                   guidance_scale=cfg.guidance_scale or 3.5,
                   generator=torch.Generator().manual_seed(cfg.seed)).images[0]
        img.save(out_path(args.config, i))
        print(f"[quality] {args.config} p{i} {time.perf_counter() - t0:.2f}s {prompt!r}", flush=True)
    return 0


def score(args) -> int:
    from benchmark.output_parity import compare

    n = len(holdout_prompts(args.prompts))
    for label in args.configs:
        psnrs = []
        for i in range(n):
            a, b = out_path(args.ref, i), out_path(label, i)
            if a.exists() and b.exists():
                psnrs.append(compare(a, b)["psnr_db"])
        if not psnrs:
            print(f"[quality] {label}: no images"); continue
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
