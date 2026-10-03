#!/usr/bin/env python3
"""Calibrate static per-tensor activation scales for the FP8 PTQ targets of a Wan DiT (CPU).

Runs the real denoising loop (``WanOrchestrator`` with the CPU op backend, the model's
own scheduler, UMT5 prompt embeddings computed here) on the bf16 transformer with real
weights, and records for every FP8 target linear the absmax of its input at every step.
The per-step absmax is what a dynamic per-tensor scale would have used; the max over
steps (times a margin applied at checkpoint-build time) is the static scale.

    DIFFLET_BACKEND=cpu PYTHONPATH=$PWD python scripts/ptq_calibrate_activations.py \\
        --model-dir <hf snapshot> --height 480 --width 832 --num-frames 9 --steps 20 \\
        --prompt "a cinematic shot of a red fox running through a snowy forest" \\
        --out artifacts/.../act_calibration.json

One bf16 forward of the 14B DiT at 480x832x9 takes ~2 min on the trn2 host's 12 vCPUs,
so 20 steps is ~40 min. ``--max-steps`` records only the first N steps of the schedule.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from difflet.quant.spec import QuantSpec  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True, help="HF snapshot root (transformer/, text_encoder/, tokenizer/)")
    p.add_argument("--model-type", default="wan")
    p.add_argument("--prompt", default="a cinematic shot of a red fox running through a snowy forest")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=None, help="stop after N recorded steps")
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-seq-len", type=int, default=512)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--text-pt", type=Path, default=None,
                   help="hunyuan_video: real conditioning {encoder_hidden_states, encoder_attention_mask, "
                        "pooled_projections} (the CLI's llama.pt + clip.pt contents)")
    p.add_argument("--out", type=Path, required=True)
    return p


def _load_transformer(model_dir: Path, dtype: torch.dtype):
    import difflet.models.wan.modeling_wan as wan
    from difflet.backends.trainium.core.modules.checkpoint import load_state_dict
    from difflet.models.wan.checkpoint.backbone import convert_backbone_state_dict

    transformer_dir = model_dir / "transformer"
    config = wan.WanTransformerConfig.from_diffusers_dict(json.loads((transformer_dir / "config.json").read_text()))
    model = wan.WanTransformer3DModel(config, dtype=dtype)
    state = convert_backbone_state_dict(load_state_dict(str(transformer_dir)))
    state = {k: v.to(dtype) for k, v in state.items()}
    model.load_state_dict(state, strict=False, assign=True)
    return model.to(dtype).eval()


def _prompt_embeds(model_dir: Path, prompt: str, seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    from transformers import AutoTokenizer, UMT5EncoderModel

    tok = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    enc = UMT5EncoderModel.from_pretrained(str(model_dir / "text_encoder"), torch_dtype=dtype).eval()
    ti = tok(prompt, padding="max_length", max_length=seq_len, truncation=True, return_tensors="pt",
             return_attention_mask=True)
    with torch.no_grad():
        hidden = enc(input_ids=ti.input_ids, attention_mask=ti.attention_mask).last_hidden_state
    # Wan convention: zero the padding rows (difflet.models.wan.pipeline._zero_padding_embeds).
    return (hidden * ti.attention_mask.unsqueeze(-1).to(hidden.dtype)).to(dtype)


def _load_transformer_hv(model_dir: Path, dtype: torch.dtype):
    """The real HunyuanVideo 1.0 DiT in Difflet's module layout (fused proj_out split)."""
    from types import SimpleNamespace

    import difflet.models.hunyuan_video.modeling_hunyuan_video as hv
    from difflet.backends.trainium.core.modules.checkpoint import load_state_dict
    from difflet.backends.trainium.hunyuan_video.backbone import NeuronHunyuanVideoBackboneApplication

    transformer_dir = model_dir / "transformer"
    cfg = hv.HunyuanVideoTransformerConfig.from_diffusers_dict(json.loads((transformer_dir / "config.json").read_text()))
    model = hv.HunyuanVideoTransformer3DModel(cfg).to(dtype).eval()
    state = NeuronHunyuanVideoBackboneApplication.convert_hf_to_neuron_state_dict(
        load_state_dict(str(transformer_dir)),
        SimpleNamespace(num_attention_heads=cfg.num_attention_heads, attention_head_dim=cfg.attention_head_dim,
                        num_single_layers=cfg.num_single_layers, neuron_config=SimpleNamespace(world_size=1)))
    state = {k: v.to(dtype) for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model


class _HVModelAdapter:
    """What HunyuanVideoOrchestrator calls: ``transformer(bundle)`` on a plain CPU module."""

    def __init__(self, model):
        self.model = model
        self.dtype = torch.bfloat16

    def __call__(self, bundle):
        return self.model(*bundle.as_model_inputs(), return_dict=False)


def _run_hunyuan_video(args, spec, records, step_counter) -> float:
    from difflet.models.hunyuan_video.pipeline import HunyuanVideoOrchestrator

    started = time.perf_counter()
    model = _load_transformer_hv(args.model_dir, torch.bfloat16)
    print(f"[calib] transformer loaded in {time.perf_counter() - started:.1f}s", flush=True)
    if not args.text_pt:
        raise SystemExit("--text-pt (encoder_hidden_states / encoder_attention_mask / pooled_projections) is required")
    text = torch.load(args.text_pt, map_location="cpu")
    hooked = _hook_targets(model, spec, records, step_counter, args.max_steps)
    print(f"[calib] hooked {hooked} target linears", flush=True)
    orch = HunyuanVideoOrchestrator(model_path=str(args.model_dir), transformer=_HVModelAdapter(model),
                                    dtype=torch.bfloat16, height=args.height, width=args.width, num_frames=args.num_frames)
    torch.manual_seed(args.seed)  # the CLI's noise: manual_seed(seed) then randn of the latent shape
    latent_frames = (args.num_frames - 1) // 4 + 1
    latents = torch.randn(1, 16, latent_frames, args.height // 8, args.width // 8, dtype=torch.bfloat16)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            orch(latents=latents, encoder_hidden_states=text["encoder_hidden_states"].to(torch.bfloat16),
                 encoder_attention_mask=text["encoder_attention_mask"].to(torch.int64),
                 pooled_projections=text["pooled_projections"].to(torch.bfloat16),
                 num_inference_steps=args.steps, guidance_scale=args.guidance_scale, output_type="latent")
    except _Stop:
        pass
    return time.perf_counter() - started


class _Stop(Exception):
    pass


def _hook_targets(model, spec, records, step_counter, max_steps) -> int:
    def make_hook(name):
        def hook(module, inputs):
            x = inputs[0]
            records.setdefault(name, []).append(float(x.detach().abs().amax()))
        return hook

    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and spec.matches(n)]
    for name, module in targets:
        module.register_forward_pre_hook(make_hook(name))
    if max_steps is not None and targets:
        def stop_hook(module, inputs):
            step_counter["calls"] += 1
            if step_counter["calls"] > max_steps:
                raise _Stop()
        targets[0][1].register_forward_pre_hook(stop_hook)
    return len(targets)


def main() -> int:
    args = build_parser().parse_args()
    torch.set_num_threads(args.threads)
    from difflet.models.wan.pipeline import WanOrchestrator

    spec = QuantSpec.for_model(args.model_type)
    records: dict[str, list[float]] = {}
    step_counter = {"calls": 0}
    if args.model_type == "hunyuan_video":
        elapsed = _run_hunyuan_video(args, spec, records, step_counter)
        return _write(args, records, elapsed)
    if args.model_type != "wan":
        # Other models plug in as scripts/calib_models/<model_type>.py exposing
        #   run(args, install_hooks) -> elapsed_seconds
        # where install_hooks(model) registers the absmax hooks on the loaded CPU
        # model (it returns the number of hooked linears) and run() then drives
        # the model's real denoise loop with its real conditioning.
        import importlib.util

        plugin = Path(__file__).resolve().parent / "calib_models" / f"{args.model_type}.py"
        if not plugin.exists():
            raise SystemExit(f"no calibration loop for model type {args.model_type!r} ({plugin} missing)")
        module_spec = importlib.util.spec_from_file_location(f"calib_models_{args.model_type}", plugin)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        def install_hooks(model) -> int:
            hooked = _hook_targets(model, spec, records, step_counter, args.max_steps)
            print(f"[calib] hooked {hooked} target linears", flush=True)
            return hooked

        try:
            elapsed = module.run(args, install_hooks)
        except _Stop:
            elapsed = float("nan")
        return _write(args, records, elapsed)
    started = time.perf_counter()
    model = _load_transformer(args.model_dir, torch.bfloat16)
    print(f"[calib] transformer loaded in {time.perf_counter() - started:.1f}s", flush=True)
    started = time.perf_counter()
    embeds = _prompt_embeds(args.model_dir, args.prompt, args.text_seq_len, torch.bfloat16)
    print(f"[calib] prompt embeds {tuple(embeds.shape)} in {time.perf_counter() - started:.1f}s", flush=True)

    # Per-target absmax of the input, per call (one call per step at guidance 1).
    records: dict[str, list[float]] = {}
    hooked = _hook_targets(model, spec, records, step_counter, args.max_steps)
    print(f"[calib] hooked {hooked} target linears", flush=True)

    orch = WanOrchestrator(model_path=str(args.model_dir), transformer=model, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(args.seed)
    latent_frames = (args.num_frames - 1) // 4 + 1
    latents = torch.randn(1, 16, latent_frames, args.height // 8, args.width // 8, generator=g)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            orch(prompt_embeds=embeds, latents=latents, height=args.height, width=args.width,
                 num_frames=args.num_frames, num_inference_steps=args.steps,
                 guidance_scale=args.guidance_scale, output_type="latent")
    except _Stop:
        pass
    return _write(args, records, time.perf_counter() - started)


def _write(args, records: dict[str, list[float]], elapsed: float) -> int:
    steps_recorded = max(len(v) for v in records.values()) if records else 0
    print(f"[calib] {steps_recorded} steps recorded in {elapsed:.1f}s", flush=True)

    layers = {}
    for name, values in sorted(records.items()):
        layers[name] = {"amax": max(values), "per_step": [round(v, 5) for v in values],
                        "min_step_amax": min(values), "n": len(values)}
    ratios = [v["amax"] / max(v["min_step_amax"], 1e-12) for v in layers.values()]
    summary = {
        "model_dir": str(args.model_dir), "model_type": args.model_type, "prompt": args.prompt,
        "shape": {"height": args.height, "width": args.width, "num_frames": args.num_frames},
        "steps": args.steps, "steps_recorded": steps_recorded, "guidance_scale": args.guidance_scale,
        "seed": args.seed, "num_layers": len(layers), "elapsed_seconds": round(elapsed, 1),
        "max_over_min_step_ratio": {"median": sorted(ratios)[len(ratios) // 2] if ratios else None,
                                    "max": max(ratios) if ratios else None},
        "layers": layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[calib] wrote {args.out}: {len(layers)} layers, amax spread across steps "
          f"median x{summary['max_over_min_step_ratio']['median']:.2f}, max x{summary['max_over_min_step_ratio']['max']:.2f}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
