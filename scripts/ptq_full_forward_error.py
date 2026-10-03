#!/usr/bin/env python3
"""One full Wan DiT forward on the CPU, real weights, three numerics regimes.

Answers "how far is one fp8 step from bf16, and how far is bf16 itself from
fp32?" at the production shape, without the device: the same random inputs go
through the model in fp32 (reference), bf16 and fp8 W8A8
(CPU fake-quant, ``difflet.quant.fake_linear``), and every pair is compared
with cosine / rel-L2 / SNR. The per-linear sweep (``ptq_linear_error_sweep.py``)
measures each matmul in isolation; this measures the whole block stack once.

    DIFFLET_BACKEND=cpu PYTHONPATH=$PWD python scripts/ptq_full_forward_error.py \\
        --model-dir <hf snapshot> --height 480 --width 832 --num-frames 9 --timestep 500 \\
        --out artifacts/.../full_forward_error_t500.json

Needs ~90 GB of RAM for the 14B model (fp32 + bf16 copies); ``--max-blocks N``
limits the depth.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DIFFLET_BACKEND", "cpu")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from difflet.quant.fake_linear import quantize_module_  # noqa: E402
from difflet.quant.metrics import tensor_error_metrics  # noqa: E402
from difflet.quant.spec import QuantSpec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True, help="HF snapshot root (contains transformer/)")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--text-seq-len", type=int, default=512)
    p.add_argument("--timestep", type=float, default=500.0)
    p.add_argument("--max-blocks", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    return p


def _load(args, wan, dtype):
    from difflet.backends.trainium.core.modules.checkpoint import load_state_dict
    from difflet.models.wan.checkpoint.backbone import convert_backbone_state_dict

    transformer_dir = args.model_dir / "transformer"
    config = wan.WanTransformerConfig.from_diffusers_dict(json.loads((transformer_dir / "config.json").read_text()))
    if args.max_blocks is not None:
        config.num_layers = min(config.num_layers, args.max_blocks)
    model = wan.WanTransformer3DModel(config, dtype=dtype)
    state = convert_backbone_state_dict(load_state_dict(str(transformer_dir)))
    if args.max_blocks is not None:
        state = {k: v for k, v in state.items()
                 if not k.startswith("blocks.") or int(k.split(".")[1]) < config.num_layers}
    state = {k: v.to(dtype) for k, v in state.items()}
    model.load_state_dict(state, strict=False, assign=True)
    return model.to(dtype).eval()


def _first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def main() -> int:
    args = build_parser().parse_args()
    import difflet.models.wan.modeling_wan as wan

    g = torch.Generator().manual_seed(args.seed)
    latent_frames = (args.num_frames - 1) // 4 + 1
    latents = torch.randn(1, 16, latent_frames, args.height // 8, args.width // 8, generator=g)
    text = torch.randn(1, args.text_seq_len, 4096, generator=g)
    timestep = torch.tensor([args.timestep])

    outputs: dict[str, torch.Tensor] = {}
    timings: dict[str, float] = {}

    def run(name, model, dtype):
        inputs = (latents.to(dtype), timestep.to(dtype), text.to(dtype))
        started = time.perf_counter()
        with torch.no_grad():
            outputs[name] = _first(model(*inputs)).float()
        timings[name] = round(time.perf_counter() - started, 1)
        print(f"[full] {name}: forward {timings[name]}s", flush=True)

    started = time.perf_counter()
    model32 = _load(args, wan, torch.float32)
    print(f"[full] loaded fp32 in {time.perf_counter() - started:.1f}s", flush=True)
    run("fp32", model32, torch.float32)
    del model32

    model16 = _load(args, wan, torch.bfloat16)
    run("bf16", model16, torch.bfloat16)
    quantize_module_(model16, QuantSpec())
    run("fp8-tensor", model16, torch.bfloat16)
    del model16

    pairs = {}
    for a, b in itertools.combinations(outputs, 2):
        pairs[f"{b}_vs_{a}"] = tensor_error_metrics(outputs[a], outputs[b])
    report = {
        "model": str(args.model_dir), "inputs": {"height": args.height, "width": args.width,
                                                  "num_frames": args.num_frames, "text_seq_len": args.text_seq_len,
                                                  "timestep": args.timestep, "seed": args.seed,
                                                  "max_blocks": args.max_blocks},
        "forward_seconds": timings, "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'pair':32s} {'cosine':>10s} {'rel_l2':>9s} {'snr_db':>8s}")
    for name, m in pairs.items():
        print(f"{name:32s} {m['cosine']:10.6f} {m['rel_l2']:9.5f} {m['snr_db']:8.2f}")
    print(f"[full] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
