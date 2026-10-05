#!/usr/bin/env python3
"""CPU end-to-end FP8 fidelity reference for Wan 2.1: the real 20-step denoise loop with
the FP8 numerics simulated on every target linear, final latents saved for comparison.

Arms (``--arm``):
  bf16         the plain bf16 transformer (the reference);
  w8           fp8 weights (per-tensor absmax / 240), bf16 activations;
  w8a8_static  fp8 weights + activations quantized with the calibrated static scale
               (absmax * 1.25 / 240, clamp to +-240) — what the checkpoint carries;
  w8a8_dyn     fp8 weights + per-call per-tensor activation scale (the dynamic law).

The fp8 rounding is done exactly as on the device (difflet.quant.fp8 helpers, e4m3 values
within +-240 have the same grid on CPU and Trainium); the matmul runs on the dequantized
operands in bf16. Use it to tell which device result is the true fp8 fidelity:

    DIFFLET_BACKEND=cpu PYTHONPATH=$PWD python scripts/ptq_fp8_cpu_e2e.py --arm w8a8_static \\
        --model-dir <hf snapshot> --calibration act_calibration_wan21.json --out latents.pt
    python -c "from difflet.quant.metrics import compare_latents; print(compare_latents(a, b))"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DIFFLET_BACKEND", "cpu")
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

from difflet.quant import fp8  # noqa: E402
from difflet.quant.spec import QuantSpec  # noqa: E402
from ptq_calibrate_activations import _load_transformer, _prompt_embeds  # noqa: E402


def _install(model, spec, arm: str, calibration: dict | None) -> int:
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and spec.matches(n)]
    for name, module in targets:
        q, s = fp8.quantize_weight(module.weight.data, "tensor")
        module.weight.data = fp8.dequantize(q, s, module.weight.dtype)
        if arm == "w8a8_static":
            amax = calibration["layers"][name]["amax"]
            scale = torch.tensor(max(amax * fp8.STATIC_ACT_MARGIN / fp8.FP8_MAX, fp8.FP8_MIN_SCALE))

            def hook(mod, inputs, scale=scale):
                x = inputs[0]
                return (fp8.dequantize(fp8.quantize_activation_static(x, scale), scale, x.dtype),) + inputs[1:]

            module.register_forward_pre_hook(hook)
        elif arm == "w8a8_dyn":
            def hook(mod, inputs):
                x = inputs[0]
                return (fp8.fake_quant_activation(x),) + inputs[1:]

            module.register_forward_pre_hook(hook)
    return len(targets)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=("bf16", "w8", "w8a8_static", "w8a8_dyn"), required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--calibration", type=Path, default=None, help="act_calibration json (w8a8_static)")
    p.add_argument("--prompt", default="a cinematic shot of a red fox running through a snowy forest")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-seq-len", type=int, default=512)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.arm == "w8a8_static" and args.calibration is None:
        raise SystemExit("--calibration is required for w8a8_static")
    torch.set_num_threads(args.threads)
    from difflet.models.wan.pipeline import WanOrchestrator

    started = time.perf_counter()
    model = _load_transformer(args.model_dir, torch.bfloat16)
    embeds = _prompt_embeds(args.model_dir, args.prompt, args.text_seq_len, torch.bfloat16)
    hooked = 0
    if args.arm != "bf16":
        calibration = json.loads(args.calibration.read_text()) if args.calibration else None
        hooked = _install(model, QuantSpec.for_model("wan"), args.arm, calibration)
    print(f"[cpu-e2e] {args.arm}: {hooked} fp8 linears, setup {time.perf_counter() - started:.1f}s", flush=True)

    orch = WanOrchestrator(model_path=str(args.model_dir), transformer=model, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(args.seed)
    latent_frames = (args.num_frames - 1) // 4 + 1
    latents = torch.randn(1, 16, latent_frames, args.height // 8, args.width // 8, generator=g)
    started = time.perf_counter()
    with torch.no_grad():
        out = orch(prompt_embeds=embeds, latents=latents, height=args.height, width=args.width,
                   num_frames=args.num_frames, num_inference_steps=args.steps,
                   guidance_scale=1.0, output_type="latent")
    out = getattr(out, "frames", out)
    if isinstance(out, (list, tuple)):
        out = out[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out.detach().cpu(), args.out)
    print(f"[cpu-e2e] {args.arm}: {args.steps} steps in {time.perf_counter() - started:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
