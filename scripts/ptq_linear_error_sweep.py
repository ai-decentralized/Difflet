#!/usr/bin/env python3
"""Per-linear FP8 matmul error of a Wan DiT on real weights (CPU, no device).

For every FastVideo-set linear (attention q/k/v/out, FFN in/out) in every block,
capture its real input activation from one bf16 forward, then compare the FP8
matmul against the exact fp32 matmul under each PTQ scheme:

    fp8-tensor   per-tensor weights + dynamic per-tensor activations (default)
    fp8-channel  per-channel weights + dynamic activations
    bf16         the bf16 matmul itself — the noise floor to read fp8 against

(Weight-only schemes were removed on 2026-10-03; FP8 PTQ is always W8A8.)

Metrics per cell: MSE, cosine, max-abs, mean-abs, relative L2, SNR(dB). The fp8
numbers are exact for the algorithm the device runs (only fp32 accumulation order
differs), so this is the reference the on-device numbers are checked against.

    PYTHONPATH=$PWD python scripts/ptq_linear_error_sweep.py \\
        --model-dir ~/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/<sha> \\
        --height 480 --width 832 --num-frames 9 --timestep 500 --max-blocks 8 \\
        --out artifacts/ptq/wan21_linear_error.json

``--tiny`` runs the whole pipeline on a random 2-block model (smoke test, no weights).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DIFFLET_BACKEND", "cpu")  # bind difflet.ops to the torch reference

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.quant.fp8 import fp8_linear_reference, quantize_weight  # noqa: E402
from difflet.quant.metrics import tensor_error_metrics  # noqa: E402
from difflet.quant.spec import QuantSpec  # noqa: E402

SCHEMES = {
    "fp8-tensor": "tensor",
    "fp8-channel": "channel",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=None, help="HF snapshot root (contains transformer/)")
    p.add_argument("--subfolder", default="transformer")
    p.add_argument("--tiny", action="store_true", help="random 2-block model instead of --model-dir")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9, help="pixel frames (latent = (n-1)//4+1)")
    p.add_argument("--text-seq-len", type=int, default=512)
    p.add_argument("--timestep", type=float, default=500.0)
    p.add_argument("--bundle", type=Path, default=None,
                   help=".pt dict with hidden_states / timestep / encoder_hidden_states (real inputs)")
    p.add_argument("--max-blocks", type=int, default=None, help="only run/measure the first N blocks")
    p.add_argument("--m-slice", type=int, default=1024, help="tokens kept per captured activation (0 = all)")
    p.add_argument("--schemes", default=",".join(SCHEMES), help="comma list of scheme names")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    return p


def _tiny_config(wan):
    return wan.WanTransformerConfig(
        patch_size=(1, 2, 2), num_attention_heads=4, attention_head_dim=16, in_channels=16,
        out_channels=16, text_dim=32, freq_dim=32, ffn_dim=64, num_layers=2,
    )


def _load_model(args, wan):
    if args.tiny:
        torch.manual_seed(args.seed)
        return wan.WanTransformer3DModel(_tiny_config(wan)).to(torch.bfloat16).eval(), "tiny-random"
    if args.model_dir is None:
        raise SystemExit("--model-dir (or --tiny) is required")
    from difflet.backends.trainium.core.modules.checkpoint import load_state_dict
    from difflet.models.wan.checkpoint.backbone import convert_backbone_state_dict

    transformer_dir = args.model_dir / args.subfolder
    raw = json.loads((transformer_dir / "config.json").read_text())
    config = wan.WanTransformerConfig.from_diffusers_dict(raw)
    if args.max_blocks is not None:
        config.num_layers = min(config.num_layers, args.max_blocks)  # build only what we run
    started = time.perf_counter()
    model = wan.WanTransformer3DModel(config, dtype=torch.bfloat16)
    state = convert_backbone_state_dict(load_state_dict(str(transformer_dir)))
    if args.max_blocks is not None:
        state = {k: v for k, v in state.items()
                 if not k.startswith("blocks.") or int(k.split(".")[1]) < config.num_layers}
    # assign=True adopts the loaded tensors instead of copying them: the 14B
    # model would otherwise be resident twice (56 GB) during the load.
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    missing = [m for m in missing if not m.endswith(".rank")]
    if missing or unexpected:
        print(f"[sweep] WARNING load_state_dict missing={missing[:5]} unexpected={unexpected[:5]}")
    print(f"[sweep] loaded {transformer_dir} in {time.perf_counter() - started:.1f}s "
          f"({config.num_layers} blocks)", flush=True)
    return model.to(torch.bfloat16).eval(), str(transformer_dir)


def _inputs(args, model):
    cfg = model.config
    if args.bundle is not None:
        data = torch.load(args.bundle, map_location="cpu")
        return (data["hidden_states"].to(torch.bfloat16), data["timestep"].to(torch.bfloat16),
                data["encoder_hidden_states"].to(torch.bfloat16))
    g = torch.Generator().manual_seed(args.seed)
    latent_frames = (args.num_frames - 1) // 4 + 1
    if args.tiny:
        latents = torch.randn(1, cfg.in_channels, 1, 8, 8, generator=g)
        text = torch.randn(1, 8, cfg.text_dim, generator=g)
    else:
        latents = torch.randn(1, cfg.in_channels, latent_frames, args.height // 8, args.width // 8, generator=g)
        text = torch.randn(1, args.text_seq_len, cfg.text_dim, generator=g)
    timestep = torch.tensor([args.timestep])
    return latents.to(torch.bfloat16), timestep.to(torch.bfloat16), text.to(torch.bfloat16)


def _target_modules(model, spec: QuantSpec):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and spec.matches(name):
            yield name, module


def main() -> int:
    args = build_parser().parse_args()
    import difflet.models.wan.modeling_wan as wan  # after DIFFLET_BACKEND is set

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    unknown = [s for s in schemes if s not in SCHEMES]
    if unknown:
        raise SystemExit(f"unknown schemes {unknown}; known: {sorted(SCHEMES)}")

    model, model_label = _load_model(args, wan)
    spec = QuantSpec()
    captures: dict[str, torch.Tensor] = {}
    hooks = []
    limit = args.m_slice if args.m_slice and args.m_slice > 0 else None

    def make_hook(name):
        def hook(_module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            captures[name] = (x[:limit] if limit else x).to(torch.bfloat16).contiguous()
        return hook

    for name, module in _target_modules(model, spec):
        hooks.append(module.register_forward_pre_hook(make_hook(name)))

    hidden, timestep, text = _inputs(args, model)
    started = time.perf_counter()
    with torch.no_grad():
        model(hidden, timestep, text)
    forward_s = time.perf_counter() - started
    for h in hooks:
        h.remove()
    print(f"[sweep] captured {len(captures)} linears in {forward_s:.1f}s", flush=True)

    rows = []
    started = time.perf_counter()
    for name, module in _target_modules(model, spec):
        x = captures[name]
        weight = module.weight.detach()
        bias = module.bias.detach() if module.bias is not None else None
        exact = torch.nn.functional.linear(x.float(), weight.float(),
                                           bias.float() if bias is not None else None)
        metrics = {}
        bf16_out = torch.nn.functional.linear(x, weight, bias).float()
        metrics["bf16"] = tensor_error_metrics(exact, bf16_out)
        for scheme in schemes:
            granularity = SCHEMES[scheme]
            wq, scale = quantize_weight(weight, granularity)
            out = fp8_linear_reference(x.float(), wq, scale, bias)
            metrics[scheme] = tensor_error_metrics(exact, out)
        parts = name.split(".")
        block = int(parts[1]) if parts[0] == "blocks" and parts[1].isdigit() else None
        rows.append({
            "name": name, "block": block, "linear": ".".join(parts[2:]) if block is not None else name,
            "in_features": int(module.in_features), "out_features": int(module.out_features),
            "tokens": int(x.shape[0]), "metrics": metrics,
        })
    metrics_s = time.perf_counter() - started

    def summarize(scheme):
        cells = [r["metrics"][scheme] for r in rows]
        finite_snr = [c["snr_db"] for c in cells if c["snr_db"] != float("inf")]
        worst = min(rows, key=lambda r: r["metrics"][scheme]["cosine"])
        return {
            "min_cosine": min(c["cosine"] for c in cells),
            "mean_cosine": sum(c["cosine"] for c in cells) / len(cells),
            "max_mse": max(c["mse"] for c in cells),
            "max_rel_l2": max(c["rel_l2"] for c in cells),
            "mean_snr_db": sum(finite_snr) / len(finite_snr) if finite_snr else float("inf"),
            "min_snr_db": min(finite_snr) if finite_snr else float("inf"),
            "worst_cell": worst["name"],
        }

    summary = {scheme: summarize(scheme) for scheme in ["bf16", *schemes]}
    table = {
        "schema_version": 1,
        "model": model_label,
        "inputs": {"height": args.height, "width": args.width, "num_frames": args.num_frames,
                   "text_seq_len": args.text_seq_len, "timestep": args.timestep,
                   "bundle": str(args.bundle) if args.bundle else None, "tiny": args.tiny,
                   "m_slice": args.m_slice, "seed": args.seed},
        "targets": list(spec.targets),
        "row_count": len(rows),
        "forward_seconds": round(forward_s, 3),
        "metrics_seconds": round(metrics_s, 3),
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2) + "\n")

    print(f"{'scheme':18s} {'min cos':>9s} {'mean cos':>9s} {'max relL2':>10s} {'min SNR dB':>11s}  worst cell")
    for scheme, s in summary.items():
        print(f"{scheme:18s} {s['min_cosine']:9.6f} {s['mean_cosine']:9.6f} {s['max_rel_l2']:10.4f} "
              f"{s['min_snr_db']:11.2f}  {s['worst_cell']}")
    print(f"[sweep] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
