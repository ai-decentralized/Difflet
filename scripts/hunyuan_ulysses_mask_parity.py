#!/usr/bin/env python
"""Masked Ulysses correctness (paper eval E3b): HunyuanVideo DiT backbone with
Ulysses context parallelism vs the same tensor-parallel layout without CP.

One mode per process (NxD parallel_state initialises once per process):
  --mode reference  tp=2, cp=1: attention over the whole sequence on each rank
  --mode ulysses    tp=2, cp=2: the sequence split over two ranks, all-to-all to
                    a head split around attention, the text padding mask passed
                    as the joint valid-key count
Both share the TP layout, so a difference is the sequence split alone, not a
different reduction order across TP ranks.

Each mode compiles ONE program and runs it at every ``--lengths`` value: the
first L text tokens valid, the rest padding (encoder_attention_mask is a runtime
input). The same fixed-seed inputs go to both modes. Outputs are saved together
to ``--out``; ``--compare A B`` prints max |diff|, mean |diff| and cosine per
length and writes them as JSON.

    python scripts/hunyuan_ulysses_mask_parity.py --mode reference --out ref.pt
    python scripts/hunyuan_ulysses_mask_parity.py --mode ulysses --out uly.pt
    python scripts/hunyuan_ulysses_mask_parity.py --compare ref.pt uly.pt --json out.json

Reduced depth (``--layers`` dual-stream blocks, no single-stream blocks) keeps
the compile short; the eval shape 320x512x61 keeps the sequence the paper runs.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

MODES = {"reference": (2, 1), "ulysses": (2, 2)}


def run(args) -> None:
    from difflet.backends.trainium.hunyuan_video.backbone import (
        NeuronHunyuanVideoBackboneApplication,
    )
    from difflet.models.hunyuan_video.application import create_hunyuan_video_backbone_config
    from difflet.pipeline.path_resolver import resolve_model_path

    tp, cp = MODES[args.mode]
    model_dir = resolve_model_path(args.model, local_files_only=True)
    latent_frames = (args.num_frames - 1) // 4 + 1
    config = create_hunyuan_video_backbone_config(
        model_path=model_dir, world_size=tp * cp, tp_degree=tp, dtype=torch.bfloat16,
        height=args.height, width=args.width, num_frames=args.num_frames, batch_size=1,
        context_parallel_enabled=cp > 1, cp_mode="ulysses" if cp > 1 else "gather_kv",
    )
    config.num_layers = args.layers
    config.num_single_layers = args.single_layers
    out_dir = os.path.join(args.work_dir, f"compile_{args.mode}_l{args.layers}s{args.single_layers}")
    os.makedirs(out_dir, exist_ok=True)
    app = NeuronHunyuanVideoBackboneApplication(
        model_path=os.path.join(model_dir, "transformer"), config=config)
    print(f"[parity] mode={args.mode} tp={tp} cp={cp} layers={args.layers}+{args.single_layers} "
          f"shape={args.height}x{args.width}x{args.num_frames} -> {out_dir}", flush=True)
    app.compile(out_dir, debug=False)
    app.load(out_dir)

    torch.manual_seed(args.seed)
    hidden = torch.randn([1, config.in_channels, latent_frames, args.height // 8,
                          args.width // 8], dtype=torch.bfloat16)
    timestep = torch.full([1], 500.0, dtype=torch.bfloat16)
    text_len = int(getattr(config, "text_seq_len", 256))
    text = torch.randn([1, text_len, config.text_embed_dim], dtype=torch.bfloat16)
    pooled = torch.randn([1, config.pooled_projection_dim], dtype=torch.bfloat16)
    guidance = torch.full([1], 6000.0, dtype=torch.bfloat16)

    outs = {}
    for n_valid in args.lengths:
        if not 0 < n_valid <= text_len:
            raise SystemExit(f"length {n_valid} outside 1..{text_len}")
        mask = torch.zeros([1, text_len], dtype=torch.int64)
        mask[:, :n_valid] = 1
        with torch.no_grad():
            out = app.models[0](hidden, timestep, text, mask, pooled, guidance)
        outs[n_valid] = out.detach().to(torch.float32).cpu()
        print(f"[parity] {args.mode} valid={n_valid}: out {tuple(outs[n_valid].shape)} "
              f"finite={bool(torch.isfinite(outs[n_valid]).all())}", flush=True)
    torch.save({"mode": args.mode, "tp": tp, "cp": cp, "text_seq_len": text_len,
                "layers": args.layers, "single_layers": args.single_layers,
                "shape": [args.height, args.width, args.num_frames], "seed": args.seed,
                "outputs": outs, "program_dir": out_dir}, args.out)
    print(f"[parity] saved {len(outs)} outputs -> {args.out}", flush=True)


def compare(a_path: str, b_path: str, json_path: str | None) -> int:
    a, b = torch.load(a_path), torch.load(b_path)
    rows = []
    for n in sorted(set(a["outputs"]) & set(b["outputs"])):
        x, y = a["outputs"][n].flatten(), b["outputs"][n].flatten()
        d = (x - y).abs()
        rows.append({"valid_text_tokens": n, "max_abs_diff": float(d.max()),
                     "mean_abs_diff": float(d.mean()), "ref_max_abs": float(x.abs().max()),
                     "cosine": float(torch.nn.functional.cosine_similarity(x, y, dim=0))})
        print(f"[parity] valid={n}: max|diff| {rows[-1]['max_abs_diff']:.3e} mean|diff| "
              f"{rows[-1]['mean_abs_diff']:.3e} cosine {rows[-1]['cosine']:.6f}")
    # one program per mode served every length: check the masks changed the output
    distinct = len({round(float(o.sum()), 3) for o in a["outputs"].values()})
    doc = {"reference": {k: a[k] for k in ("mode", "tp", "cp", "program_dir")},
           "candidate": {k: b[k] for k in ("mode", "tp", "cp", "program_dir")},
           "layers": a["layers"], "single_layers": a["single_layers"], "shape": a["shape"],
           "text_seq_len": a["text_seq_len"], "seed": a["seed"],
           "distinct_reference_outputs": distinct, "rows": rows}
    if json_path:
        with open(json_path, "w") as fh:
            json.dump(doc, fh, indent=2)
        print(f"[parity] wrote {json_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=sorted(MODES))
    p.add_argument("--out")
    p.add_argument("--compare", nargs=2, metavar=("REF", "CAND"))
    p.add_argument("--json", default=None)
    p.add_argument("--model", default="hunyuanvideo-community/HunyuanVideo")
    p.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")],
                   default=[256, 160, 77, 20, 5])
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--single-layers", type=int, default=2)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=61)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--work-dir", default=os.path.expanduser("~/.cache/difflet/parity/hv_ulysses"))
    args = p.parse_args()
    if args.compare:
        return compare(*args.compare, args.json)
    if not (args.mode and args.out):
        p.error("--mode and --out (or --compare)")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
