"""Per-layer activation-quantization error on the REAL Wan 2.1 denoise loop (CPU, bf16 model).

At the chosen steps, for every target linear with its real input x:
  ref    = bf16 linear(x)
  dyn    = fp8 W8A8 with the dynamic per-tensor scale (Difflet law)
  static = fp8 W8A8 with the calibrated input_scale (absmax * 1.25 / 240)
  wo     = fp8 weight only (bf16 activation)
and records SNR(dB) / cosine of each against ref, plus the share of activation elements that
fall below the fp8 normal range (|x|/scale < 2^-6) or flush to zero (< 2^-9) per law.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DIFFLET_BACKEND", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
ROOT = Path("/home/ubuntu/Difflet/.claude/worktrees/verify-ptq-fp8-wan")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from difflet.quant.fp8 import (  # noqa: E402
    ACT_SCALE_MARGIN, FP8_DTYPE, FP8_MAX, STATIC_ACT_MARGIN, dequantize, quantize_weight,
)
from difflet.quant.spec import QuantSpec  # noqa: E402
import ptq_calibrate_activations as calib  # noqa: E402


def snr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    ref = ref.float(); test = test.float()
    err = (test - ref).pow(2).sum().item()
    sig = ref.pow(2).sum().item()
    return 10 * math.log10(sig / err) if err > 0 else float("inf")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--probe-steps", default="0,10,19")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    from difflet.models.wan.pipeline import WanOrchestrator

    probe_steps = {int(s) for s in args.probe_steps.split(",")}
    spec = QuantSpec.for_model("wan")
    cal = json.load(open(args.calibration))["layers"]
    t0 = time.perf_counter()
    model = calib._load_transformer(args.model_dir, torch.bfloat16)
    print(f"[err] transformer loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    embeds = calib._prompt_embeds(args.model_dir, "a cinematic shot of a red fox running through a snowy forest", 512, torch.bfloat16)
    print(f"[err] embeds {tuple(embeds.shape)}", flush=True)

    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and spec.matches(n)]
    first_name = targets[0][0]
    step = {"i": -1}
    results: dict[str, dict] = {}
    def w8(name, module):
        # Dequantized per call (no cache): 400 fp32 copies would be 40 GB.
        q, s = quantize_weight(module.weight.detach(), "tensor")
        return dequantize(q, s)

    def make_hook(name, module):
        def hook(mod, inputs):
            if name == first_name:
                step["i"] += 1
                if step["i"] in probe_steps:
                    print(f"[err] probing step {step['i']}", flush=True)
            if step["i"] not in probe_steps:
                return
            x = inputs[0].detach()
            with torch.no_grad():
                ref = torch.nn.functional.linear(x, module.weight, module.bias).float()
                w32 = w8(name, module)
                bias = module.bias.float() if module.bias is not None else None
                x32 = x.float()
                # weight only
                wo = x32 @ w32.t()
                # dynamic
                amax = x32.abs().amax()
                sd = amax / FP8_MAX * ACT_SCALE_MARGIN
                qd = (x32 * (1.0 / sd)).to(FP8_DTYPE).float() * sd
                dyn = qd @ w32.t()
                # static
                ss = torch.tensor(cal[name]["amax"] * STATIC_ACT_MARGIN / FP8_MAX)
                qs = (x32 * (1.0 / ss)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).float() * ss
                sta = qs @ w32.t()
                if bias is not None:
                    wo = wo + bias; dyn = dyn + bias; sta = sta + bias
                ax = x32.abs()
                rec = {
                    "amax": float(amax), "calib_amax": cal[name]["amax"],
                    "snr_wo": snr_db(ref, wo), "snr_dyn": snr_db(ref, dyn), "snr_static": snr_db(ref, sta),
                    "act_snr_dyn": snr_db(x32, qd), "act_snr_static": snr_db(x32, qs),
                    "sub_dyn": float((ax / sd < 2 ** -6).float().mean()), "zero_dyn": float((ax / sd < 2 ** -9).float().mean()),
                    "sub_static": float((ax / ss < 2 ** -6).float().mean()), "zero_static": float((ax / ss < 2 ** -9).float().mean()),
                    "clip_static": float((ax / ss > FP8_MAX).float().mean()),
                }
            results.setdefault(name, {})[str(step["i"])] = rec
        return hook

    for name, module in targets:
        module.register_forward_pre_hook(make_hook(name, module))

    class _Stop(Exception):
        pass

    def stop_hook(mod, inputs):
        if step["i"] > max(probe_steps):
            raise _Stop()
    dict(model.named_modules())[first_name].register_forward_pre_hook(stop_hook)

    orch = WanOrchestrator(model_path=str(args.model_dir), transformer=model, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(42)
    latents = torch.randn(1, 16, 3, 60, 104, generator=g)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            orch(prompt_embeds=embeds, latents=latents, height=480, width=832, num_frames=9,
                 num_inference_steps=args.steps, guidance_scale=1.0, output_type="latent")
    except _Stop:
        pass
    print(f"[err] loop done in {time.perf_counter() - t0:.1f}s", flush=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    # summary
    for s in sorted(probe_steps):
        rows = [v[str(s)] for v in results.values() if str(s) in v]
        if not rows:
            continue
        med = lambda k: sorted(r[k] for r in rows)[len(rows) // 2]
        print(f"[err] step {s}: n={len(rows)} median SNR wo {med('snr_wo'):.1f} dyn {med('snr_dyn'):.1f} static {med('snr_static'):.1f} dB;"
              f" act SNR dyn {med('act_snr_dyn'):.1f} static {med('act_snr_static'):.1f};"
              f" subnormal share dyn {med('sub_dyn'):.3f} static {med('sub_static'):.3f}; clip static {max(r['clip_static'] for r in rows):.2e}", flush=True)
        worst = sorted(rows, key=lambda r: r["snr_dyn"])[:3]
        print("[err]   worst dyn layers:", [(round(r['snr_dyn'], 1), round(r['snr_static'], 1)) for r in worst], flush=True)
    print(f"[err] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
