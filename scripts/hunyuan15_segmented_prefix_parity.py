#!/usr/bin/env python3
"""Run a one-block HunyuanVideo 1.5 transformer through segmented Trainium.

This probe stitches a full prefix-1 transformer dataflow:

1. diffusers CPU frontend up to block 0 inputs
2. Trainium segmented block: pre-qkv -> masked tiled attention -> post
3. diffusers CPU final norm/projection/reshape

It is a parity gate for the segmented block inside the full transformer dataflow,
not a production NovaPipeline runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def ensure_runtime_python() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
            env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


ensure_runtime_python()

import torch  # noqa: E402

from hunyuan15_attention_capacity_probe import (  # noqa: E402
    build_attention_capacity_application,
    run_streaming_manual_stats_attention,
)
from hunyuan15_block_split_capacity_probe import (  # noqa: E402
    _cosine,
    _shape_meta,
    build_block_split_application,
)
from hunyuan15_segmented_block_parity import (  # noqa: E402
    _compile_if_needed,
    _full_attention_valid_mask,
    _make_block_inputs_from_bundle,
    _split_args,
)
from hunyuan15_transformer_parity import (  # noqa: E402
    _load_bundle_inputs,
    _load_config,
    _parse_dtype,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--cache-dir", default="/tmp/nova_hunyuan15_segmented_prefix_cache")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--timestep-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=61)
    parser.add_argument("--text-seq-len", type=int, default=1000)
    parser.add_argument("--text-seq-len-2", type=int, default=256)
    parser.add_argument("--image-seq-len", type=int, default=729)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--qk-norm", default="rms_norm")
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--patch-size-t", type=int, default=1)
    parser.add_argument("--spatial-compression-ratio", type=int, default=16)
    parser.add_argument("--temporal-compression-ratio", type=int, default=4)
    parser.add_argument("--query-tile-size", type=int, default=489)
    parser.add_argument("--key-tile-size", type=int, default=489)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument(
        "--block-compiler-args",
        default=(
            "--model-type=transformer -O1 --auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        ),
    )
    parser.add_argument(
        "--attention-compiler-args",
        default=(
            "--model-type=generic -O1 --auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        ),
    )
    return parser


def _attention_args(args: argparse.Namespace, cache_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        backend="manual-stats-masked",
        cache_dir=str(cache_dir),
        query_len=args.query_tile_size,
        key_len=args.key_tile_size,
        heads=args.heads,
        head_dim=args.head_dim,
        layout="bshd",
        tp_degree=args.tp_degree,
        force_clean=args.force_clean,
        run_stream_merge_check=False,
        stream_query_tiles=1,
        stream_key_tiles=1,
        skip_stream_reference=True,
        seed=0,
        metrics_out=None,
        compiler_args=args.attention_compiler_args,
    )


def _load_model(args: argparse.Namespace, cfg: dict[str, object]) -> torch.nn.Module:
    if int(cfg.get("num_layers", 0)) != 1:
        raise ValueError("segmented prefix parity currently expects a prefix model with num_layers=1")
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    model = HunyuanVideo15Transformer3DModel.from_pretrained(
        Path(args.model_dir) / args.transformer_subfolder,
        torch_dtype=args.dtype,
    )
    return model.eval()


def _final_projection(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    p_t = int(model.config.patch_size_t)
    p_h = int(model.config.patch_size)
    p_w = int(model.config.patch_size)
    hidden_states = model.norm_out(hidden_states.to(dtype=temb.dtype), temb)
    hidden_states = model.proj_out(hidden_states)
    hidden_states = hidden_states.reshape(
        hidden_states.shape[0],
        latent_frames // p_t,
        latent_height // p_h,
        latent_width // p_w,
        -1,
        p_t,
        p_h,
        p_w,
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def main() -> int:
    args = build_parser().parse_args()
    cache_dir = Path(args.cache_dir)
    split_args = _split_args(args, cache_dir / "segments")
    meta = _shape_meta(split_args)
    if meta["total_seq_len"] % args.query_tile_size != 0:
        raise ValueError("total_seq_len must be divisible by --query-tile-size")
    if meta["total_seq_len"] % args.key_tile_size != 0:
        raise ValueError("total_seq_len must be divisible by --key-tile-size")

    cfg = _load_config(Path(args.model_dir) / args.transformer_subfolder)
    inputs = _load_bundle_inputs(args, cfg, args.dtype)
    model = _load_model(args, cfg)

    pre_app, pre_dir = build_block_split_application(split_args, "pre-qkv", meta)
    post_app, post_dir = build_block_split_application(split_args, "post", meta)
    attn_args = _attention_args(args, cache_dir / "attention_masked")
    attn_app, attn_dir = build_attention_capacity_application(attn_args)

    compile_metrics = {
        "pre_qkv_compile_elapsed_s": _compile_if_needed(
            pre_app,
            pre_dir,
            skip_compile=args.skip_compile,
        ),
        "attention_compile_elapsed_s": _compile_if_needed(
            attn_app,
            attn_dir,
            skip_compile=args.skip_compile,
        ),
        "post_compile_elapsed_s": _compile_if_needed(
            post_app,
            post_dir,
            skip_compile=args.skip_compile,
        ),
    }

    pre_app.load(str(pre_dir), skip_warmup=True)
    attn_app.load(str(attn_dir), skip_warmup=True)
    post_app.load(str(post_dir), skip_warmup=True)

    t0 = time.perf_counter()
    hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin, attention_mask = (
        _make_block_inputs_from_bundle(args, inputs, cfg)
    )
    frontend_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    with torch.no_grad():
        query, key, value = pre_app(
            hidden_states,
            encoder_hidden_states,
            temb,
            freqs_cos,
            freqs_sin,
        )
    pre_elapsed = time.perf_counter() - t1

    attention_states, attention_metrics = run_streaming_manual_stats_attention(
        attn_app,
        query.detach().cpu(),
        key.detach().cpu(),
        value.detach().cpu(),
        query_tile_size=args.query_tile_size,
        key_tile_size=args.key_tile_size,
        collect_output=True,
        valid_mask=_full_attention_valid_mask(attention_mask, meta["latent_seq_len"]),
    )
    assert attention_states is not None

    t2 = time.perf_counter()
    with torch.no_grad():
        segmented_hidden, _segmented_context = post_app(
            hidden_states,
            encoder_hidden_states,
            temb,
            attention_states.to(dtype=torch.bfloat16),
        )
    post_elapsed = time.perf_counter() - t2

    t3 = time.perf_counter()
    with torch.no_grad():
        segmented_output = _final_projection(
            model,
            segmented_hidden.detach().cpu(),
            temb.detach().cpu(),
            latent_frames=meta["latent_frames"],
            latent_height=meta["latent_height"],
            latent_width=meta["latent_width"],
        ).detach().cpu()
    final_elapsed = time.perf_counter() - t3

    metrics = {
        "model_dir": args.model_dir,
        "transformer_subfolder": args.transformer_subfolder,
        "bundle": args.bundle,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "text_seq_len": args.text_seq_len,
        "text_seq_len_2": args.text_seq_len_2,
        "image_seq_len": args.image_seq_len,
        "query_tile_size": args.query_tile_size,
        "key_tile_size": args.key_tile_size,
        "attention_semantics": "context_masked",
        "context_valid_tokens": int(attention_mask.sum()),
        "pre_compiled_path": str(pre_dir),
        "attention_compiled_path": str(attn_dir),
        "post_compiled_path": str(post_dir),
        **meta,
        **compile_metrics,
        **attention_metrics,
        "frontend_elapsed_s": frontend_elapsed,
        "pre_qkv_forward_elapsed_s": pre_elapsed,
        "post_forward_elapsed_s": post_elapsed,
        "final_projection_elapsed_s": final_elapsed,
        "reference_skipped": args.skip_reference,
        "segmented_shape": list(segmented_output.shape),
    }

    if not args.skip_reference:
        t4 = time.perf_counter()
        with torch.no_grad():
            reference = model(
                hidden_states=inputs["hidden_states"],
                timestep=inputs["timestep"],
                encoder_hidden_states=inputs["encoder_hidden_states"],
                encoder_attention_mask=inputs["encoder_attention_mask"],
                timestep_r=inputs["timestep_r"] if bool(cfg.get("use_meanflow", False)) else None,
                encoder_hidden_states_2=inputs["encoder_hidden_states_2"],
                encoder_attention_mask_2=inputs["encoder_attention_mask_2"],
                image_embeds=inputs["image_embeds"],
                return_dict=False,
            )[0].detach().cpu()
        reference_elapsed = time.perf_counter() - t4
        diff = (segmented_output.float() - reference.float()).abs()
        metrics.update(
            {
                "reference_elapsed_s": reference_elapsed,
                "reference_shape": list(reference.shape),
                "cosine": _cosine(segmented_output, reference),
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
            }
        )

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[hunyuan15-segmented-prefix] metrics -> {metrics_path}", flush=True)
    if args.skip_reference:
        print("[hunyuan15-segmented-prefix] runtime gate = PASS", flush=True)
        return 0
    pass_gate = metrics["cosine"] >= args.min_cosine
    print(f"[hunyuan15-segmented-prefix] gate = {'PASS' if pass_gate else 'FAIL'}", flush=True)
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
