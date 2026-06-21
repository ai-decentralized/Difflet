"""HunyuanVideo backbone compile-and-load smoke (N4).

Builds a synthetic ``transformer/`` directory containing a HunyuanVideo
config and bf16 weights derived from the diffusers reference model, then
runs the Trainium backbone application through:

  1. ``compile(dry_run=False)`` — writes ``model.pt`` + ``neuron_config.json``;
  2. artifact assertion — confirms compiled outputs are present;
  3. ``load(skip_warmup=...)`` — sharded checkpoint init on Neuron HBM (and
     optional warmup forward).

Default config matches §28 §6 (production heads / production M3 shape) at a
reduced layer count to keep iteration time bounded; env vars or CLI flags
let callers scale to full ``20+40+1``/``20+40+2``.

The diffusers reference model is used purely to generate a state dict
whose keys match Difflet's ``HunyuanVideoTransformer3DModel`` — see the
``test_hunyuan_video_transformer3d_model_matches_diffusers_*`` tests for
the equivalence proof.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int,
                        default=int(os.environ.get("DIFFLET_HUNYUAN_N4_NUM_LAYERS", "4")))
    parser.add_argument("--num-single-layers", type=int,
                        default=int(os.environ.get("DIFFLET_HUNYUAN_N4_NUM_SINGLE_LAYERS", "8")))
    parser.add_argument("--num-refiner-layers", type=int,
                        default=int(os.environ.get("DIFFLET_HUNYUAN_N4_NUM_REFINER_LAYERS", "1")))
    parser.add_argument("--height", type=int,
                        default=int(os.environ.get("DIFFLET_HUNYUAN_N4_HEIGHT", "320")))
    parser.add_argument("--width", type=int,
                        default=int(os.environ.get("DIFFLET_HUNYUAN_N4_WIDTH", "512")))
    parser.add_argument("--num-frames", type=int,
                        default=int(os.environ.get("DIFFLET_HUNYUAN_N4_FRAMES", "61")))
    parser.add_argument("--text-seq-len", type=int,
                        default=int(os.environ.get("DIFFLET_HUNYUAN_N4_TEXT_SEQ_LEN", "256")))
    parser.add_argument("--heads", type=int,
                        default=int(os.environ.get("DIFFLET_HUNYUAN_N4_HEADS", "24")))
    parser.add_argument("--head-dim", type=int,
                        default=int(os.environ.get("DIFFLET_HUNYUAN_N4_HEAD_DIM", "128")))
    parser.add_argument("--mlp-ratio", type=float,
                        default=float(os.environ.get("DIFFLET_HUNYUAN_N4_MLP_RATIO", "4.0")))
    parser.add_argument("--tp-degree", type=int,
                        default=int(os.environ.get("DIFFLET_HUNYUAN_N4_TP_DEGREE", "4")))
    parser.add_argument("--work-dir",
                        default=os.environ.get(
                            "DIFFLET_HUNYUAN_N4_WORK_DIR", ".difflet-cache/hunyuan_n4_smoke"))
    parser.add_argument("--skip-warmup", action="store_true",
                        default=os.environ.get("DIFFLET_HUNYUAN_N4_SKIP_WARMUP", "0") == "1")
    parser.add_argument("--seed", type=int, default=20260510)
    return parser.parse_args()


def _diffusers_config_dict(args: argparse.Namespace) -> dict:
    return {
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": args.heads,
        "attention_head_dim": args.head_dim,
        "num_layers": args.num_layers,
        "num_single_layers": args.num_single_layers,
        "num_refiner_layers": args.num_refiner_layers,
        "mlp_ratio": args.mlp_ratio,
        "patch_size": 2,
        "patch_size_t": 1,
        "qk_norm": "rms_norm",
        "guidance_embeds": True,
        "text_embed_dim": 4096,
        "pooled_projection_dim": 768,
        "rope_theta": 256.0,
        "rope_axes_dim": [16, 56, 56],
        "image_condition_type": None,
    }


def write_synthetic_transformer_dir(transformer_dir: Path, args: argparse.Namespace) -> None:
    """Create transformer/config.json + diffusion_pytorch_model.safetensors.

    Uses the diffusers reference module to generate a state_dict whose key
    set matches Difflet's HunyuanVideoTransformer3DModel exactly (the
    test_hunyuan_video_transformer3d_model_matches_diffusers_* tests
    prove this). We do this here, before any difflet.ops import, so that the
    process-wide backend dispatch stays free to bind to trainium below.
    """
    from diffusers.models.transformers.transformer_hunyuan_video import (
        HunyuanVideoTransformer3DModel as DiffusersModel,
    )
    from safetensors.torch import save_file

    transformer_dir.mkdir(parents=True, exist_ok=True)

    cfg = _diffusers_config_dict(args)
    (transformer_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    torch.manual_seed(args.seed)
    model = DiffusersModel(**cfg).eval()
    state_dict = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    del model

    weights_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    save_file(state_dict, str(weights_path))
    total_bytes = sum(v.element_size() * v.numel() for v in state_dict.values())
    print(f"[n4] wrote {weights_path} ({len(state_dict)} tensors, {total_bytes / 1e9:.3f} GB bf16)")


def run_compile_load(transformer_dir: Path, compiled_dir: Path, args: argparse.Namespace) -> dict:
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")

    from difflet.backends.trainium.core.config import NeuronConfig
    from difflet.backends.trainium.core.application_base import is_compiled
    from difflet.backends.trainium.hunyuan_video.backbone import (
        HunyuanVideoBackboneInferenceConfig,
        NeuronHunyuanVideoBackboneApplication,
    )
    from difflet.utils.diffusers_adapter import load_diffusers_config

    neuron_config = NeuronConfig(
        batch_size=1,
        tp_degree=args.tp_degree,
        world_size=args.tp_degree,
        torch_dtype=torch.bfloat16,
        skip_sharding=True,
    )
    cfg = HunyuanVideoBackboneInferenceConfig(
        neuron_config=neuron_config,
        load_config=load_diffusers_config(str(transformer_dir)),
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        text_seq_len=args.text_seq_len,
    )
    print(f"[n4] latent shape = [{1}, 16, {cfg.latent_frames}, {cfg.latent_height}, {cfg.latent_width}]")

    app = NeuronHunyuanVideoBackboneApplication(model_path=str(transformer_dir), config=cfg)

    print("[n4] compile(dry_run=False) ...")
    t0 = time.time()
    app.compile(str(compiled_dir), debug=False)
    compile_elapsed = time.time() - t0
    print(f"[n4] compile elapsed = {compile_elapsed:.3f}s")

    normalized = os.path.join(os.path.normpath(str(compiled_dir)), "")
    model_pt = normalized + "model.pt"
    config_json = os.path.join(normalized, "neuron_config.json")
    assert is_compiled(normalized), f"model.pt missing at {model_pt}"
    assert os.path.isfile(config_json), f"neuron_config.json missing at {config_json}"
    model_pt_bytes = os.path.getsize(model_pt)
    print(f"[n4] model.pt = {model_pt} ({model_pt_bytes / 1e6:.1f} MB)")

    print(f"[n4] load(skip_warmup={args.skip_warmup}) ...")
    t1 = time.time()
    app.load(str(compiled_dir), skip_warmup=args.skip_warmup)
    load_elapsed = time.time() - t1
    print(f"[n4] load elapsed = {load_elapsed:.3f}s")

    return {
        "config": {
            "num_layers": args.num_layers,
            "num_single_layers": args.num_single_layers,
            "num_refiner_layers": args.num_refiner_layers,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "text_seq_len": args.text_seq_len,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "mlp_ratio": args.mlp_ratio,
            "tp_degree": args.tp_degree,
            "skip_warmup": args.skip_warmup,
        },
        "latent_shape": [1, 16, cfg.latent_frames, cfg.latent_height, cfg.latent_width],
        "model_pt_bytes": model_pt_bytes,
        "compile_s": compile_elapsed,
        "load_s": load_elapsed,
    }


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    transformer_dir = work_dir / "source" / "transformer"
    compiled_dir = work_dir / "compiled" / "transformer"

    print(f"[n4] work_dir = {work_dir}")
    print(f"[n4] layers = {args.num_layers}+{args.num_single_layers}+{args.num_refiner_layers}")
    print(f"[n4] shape = {args.height}x{args.width}x{args.num_frames}, text_seq_len={args.text_seq_len}")
    print(f"[n4] heads = {args.heads}, head_dim = {args.head_dim}, mlp_ratio = {args.mlp_ratio}")

    write_synthetic_transformer_dir(transformer_dir, args)
    metrics = run_compile_load(transformer_dir, compiled_dir, args)

    metrics_path = os.environ.get("DIFFLET_HUNYUAN_N4_METRICS")
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_path).write_text(json.dumps(metrics, indent=2))
        print(f"[n4] metrics -> {metrics_path}")

    print(f"[n4] DONE compile={metrics['compile_s']:.1f}s load={metrics['load_s']:.1f}s "
          f"model.pt={metrics['model_pt_bytes'] / 1e6:.1f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
