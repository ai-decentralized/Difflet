#!/usr/bin/env python3
"""Run a segmented HunyuanVideo 1.5 transformer block and compare to CPU.

This is the first stitched block-level runtime probe for the 480p closure path:

1. compiled ``pre-qkv`` segment
2. compiled manual-stats tiled attention
3. compiled ``post`` segment

The comparison uses all-valid attention masks, so the tiled attention path is
semantically equivalent to the full block only for no-mask capacity/parity
cases. It is not yet the production pipeline runtime.
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

from hunyuan15_attention_boundary_probe import _prepare_frontend  # noqa: E402
from hunyuan15_attention_capacity_probe import (  # noqa: E402
    build_attention_capacity_application,
    run_streaming_manual_stats_attention,
)
from hunyuan15_block_split_capacity_probe import (  # noqa: E402
    _cosine,
    _load_block_state_dict_from_dir,
    _make_part_inputs,
    _shape_meta,
    build_block_split_application,
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
    parser.add_argument("--cache-dir", default="/tmp/nova_hunyuan15_segmented_block_cache")
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--text-seq-len", type=int, default=7)
    parser.add_argument("--text-seq-len-2", type=int, default=3)
    parser.add_argument("--image-seq-len", type=int, default=4)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument(
        "--bundle",
        default=None,
        help=(
            "Optional HunyuanVideo 1.5 transformer-boundary safetensors bundle. "
            "When set, the script runs the diffusers frontend to obtain real block inputs."
        ),
    )
    parser.add_argument("--timestep-index", type=int, default=0)
    parser.add_argument(
        "--random-image-embeds",
        action="store_true",
        help="Replace bundle image_embeds with random values so image tokens are valid.",
    )
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--qk-norm", default="rms_norm")
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--patch-size-t", type=int, default=1)
    parser.add_argument("--spatial-compression-ratio", type=int, default=16)
    parser.add_argument("--temporal-compression-ratio", type=int, default=4)
    parser.add_argument("--query-tile-size", type=int, default=13)
    parser.add_argument("--key-tile-size", type=int, default=13)
    parser.add_argument(
        "--context-valid-tokens",
        type=int,
        default=None,
        help=(
            "Number of valid context tokens after HunyuanVideo 1.5 context reordering. "
            "Defaults to all context tokens valid."
        ),
    )
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help=(
            "Run the segmented Trainium path without CPU full-block parity. "
            "Use this for production-size capacity/runtime probes where the CPU "
            "reference would require full dense attention."
        ),
    )
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


def _split_args(args: argparse.Namespace, cache_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        cache_dir=str(cache_dir),
        model_dir=args.model_dir,
        transformer_subfolder=args.transformer_subfolder,
        block_index=args.block_index,
        part="both",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        text_seq_len=args.text_seq_len,
        text_seq_len_2=args.text_seq_len_2,
        image_seq_len=args.image_seq_len,
        heads=args.heads,
        head_dim=args.head_dim,
        mlp_ratio=args.mlp_ratio,
        qk_norm=args.qk_norm,
        patch_size=args.patch_size,
        patch_size_t=args.patch_size_t,
        spatial_compression_ratio=args.spatial_compression_ratio,
        temporal_compression_ratio=args.temporal_compression_ratio,
        tp_degree=args.tp_degree,
        force_clean=args.force_clean,
        skip_compile=args.skip_compile,
        run_parity=False,
        metrics_out=None,
        compiler_args=args.block_compiler_args,
    )


def _attention_args(args: argparse.Namespace, cache_dir: Path) -> argparse.Namespace:
    backend = (
        "manual-stats-masked"
        if args.context_valid_tokens is not None or args.bundle is not None
        else "manual-stats"
    )
    return argparse.Namespace(
        backend=backend,
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


def _load_full_cpu_block(args: argparse.Namespace) -> torch.nn.Module:
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15TransformerBlock,
    )

    block = HunyuanVideo15TransformerBlock(
        args.heads,
        args.head_dim,
        mlp_ratio=args.mlp_ratio,
        qk_norm=args.qk_norm,
    )
    block_state = _load_block_state_dict_from_dir(
        Path(args.model_dir) / args.transformer_subfolder,
        args.block_index,
        dtype=torch.bfloat16,
    )
    block.load_state_dict(
        {key.removeprefix("block."): value for key, value in block_state.items()},
        strict=True,
    )
    return block.to(dtype=torch.bfloat16).eval()


def _compile_if_needed(app: object, path: Path, *, skip_compile: bool) -> float | None:
    if skip_compile:
        return None
    t0 = time.perf_counter()
    app.compile(str(path), debug=False)
    return time.perf_counter() - t0


def _context_attention_mask(args: argparse.Namespace, context_seq_len: int) -> torch.Tensor:
    valid_tokens = context_seq_len if args.context_valid_tokens is None else int(args.context_valid_tokens)
    if valid_tokens < 0 or valid_tokens > context_seq_len:
        raise ValueError(
            f"--context-valid-tokens must be in [0, {context_seq_len}], got {valid_tokens}"
        )
    mask = torch.zeros([1, context_seq_len], dtype=torch.int64)
    mask[:, :valid_tokens] = 1
    return mask


def _full_attention_valid_mask(context_attention_mask: torch.Tensor, latent_seq_len: int) -> torch.Tensor:
    latent_valid = torch.ones(
        [context_attention_mask.shape[0], latent_seq_len],
        dtype=torch.bool,
        device=context_attention_mask.device,
    )
    return torch.cat([latent_valid, context_attention_mask.to(dtype=torch.bool)], dim=1)


def _load_frontend_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, object] | None]:
    if args.bundle is None:
        return None, None
    transformer_dir = Path(args.model_dir) / args.transformer_subfolder
    cfg = _load_config(transformer_dir)
    inputs = _load_bundle_inputs(args, cfg, args.dtype)
    if args.random_image_embeds:
        generator = torch.Generator(device="cpu").manual_seed(17)
        inputs["image_embeds"] = torch.randn(
            inputs["image_embeds"].shape,
            generator=generator,
            dtype=args.dtype,
        )
    return inputs, cfg


def _make_block_inputs_from_bundle(
    args: argparse.Namespace,
    inputs: dict[str, torch.Tensor],
    cfg: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    transformer_dir = Path(args.model_dir) / args.transformer_subfolder
    model = HunyuanVideo15Transformer3DModel.from_pretrained(
        transformer_dir,
        torch_dtype=args.dtype,
    ).eval()
    with torch.no_grad():
        hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb = _prepare_frontend(
            model,
            inputs,
            use_meanflow=bool(cfg.get("use_meanflow", False)),
        )
    freqs_cos, freqs_sin = image_rotary_emb
    return (
        hidden_states.to(dtype=torch.bfloat16).contiguous(),
        encoder_hidden_states.to(dtype=torch.bfloat16).contiguous(),
        temb.to(dtype=torch.bfloat16).contiguous(),
        freqs_cos.to(dtype=torch.bfloat16).contiguous(),
        freqs_sin.to(dtype=torch.bfloat16).contiguous(),
        attention_mask.to(dtype=torch.int64).contiguous(),
    )


def main() -> int:
    args = build_parser().parse_args()
    frontend_inputs, frontend_cfg = _load_frontend_inputs(args)
    cache_dir = Path(args.cache_dir)
    split_args = _split_args(args, cache_dir / "segments")
    meta = _shape_meta(split_args)
    if meta["total_seq_len"] % args.query_tile_size != 0:
        raise ValueError("total_seq_len must be divisible by --query-tile-size")
    if meta["total_seq_len"] % args.key_tile_size != 0:
        raise ValueError("total_seq_len must be divisible by --key-tile-size")

    pre_app, pre_dir = build_block_split_application(split_args, "pre-qkv", meta)
    post_app, post_dir = build_block_split_application(split_args, "post", meta)
    attn_cache_name = (
        "attention_masked"
        if args.context_valid_tokens is not None or args.bundle is not None
        else "attention"
    )
    attn_args = _attention_args(args, cache_dir / attn_cache_name)
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

    if frontend_inputs is None:
        hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin = _make_part_inputs(
            "pre-qkv",
            meta,
            split_args,
            dtype=torch.bfloat16,
        )
        attention_mask = _context_attention_mask(args, meta["context_seq_len"])
        encoder_hidden_states = encoder_hidden_states * attention_mask.unsqueeze(-1).to(
            dtype=encoder_hidden_states.dtype
        )
    else:
        assert frontend_cfg is not None
        (
            hidden_states,
            encoder_hidden_states,
            temb,
            freqs_cos,
            freqs_sin,
            attention_mask,
        ) = _make_block_inputs_from_bundle(args, frontend_inputs, frontend_cfg)
        if hidden_states.shape[1] != meta["latent_seq_len"]:
            raise ValueError(
                f"bundle latent seq len {hidden_states.shape[1]} does not match compile meta "
                f"{meta['latent_seq_len']}"
            )
        if encoder_hidden_states.shape[1] != meta["context_seq_len"]:
            raise ValueError(
                f"bundle context seq len {encoder_hidden_states.shape[1]} does not match compile meta "
                f"{meta['context_seq_len']}"
            )
    use_masked_attention = args.context_valid_tokens is not None or args.bundle is not None

    t0 = time.perf_counter()
    with torch.no_grad():
        query, key, value = pre_app(
            hidden_states,
            encoder_hidden_states,
            temb,
            freqs_cos,
            freqs_sin,
        )
    pre_elapsed = time.perf_counter() - t0

    attention_states, attention_metrics = run_streaming_manual_stats_attention(
        attn_app,
        query.detach().cpu(),
        key.detach().cpu(),
        value.detach().cpu(),
        query_tile_size=args.query_tile_size,
        key_tile_size=args.key_tile_size,
        collect_output=True,
        valid_mask=(
            _full_attention_valid_mask(attention_mask, meta["latent_seq_len"])
            if use_masked_attention
            else None
        ),
    )
    assert attention_states is not None

    t1 = time.perf_counter()
    with torch.no_grad():
        segmented_hidden, segmented_context = post_app(
            hidden_states,
            encoder_hidden_states,
            temb,
            attention_states.to(dtype=torch.bfloat16),
        )
    post_elapsed = time.perf_counter() - t1

    metrics = {
        "model_dir": args.model_dir,
        "transformer_subfolder": args.transformer_subfolder,
        "block_index": args.block_index,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "text_seq_len": args.text_seq_len,
        "text_seq_len_2": args.text_seq_len_2,
        "image_seq_len": args.image_seq_len,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "query_tile_size": args.query_tile_size,
        "key_tile_size": args.key_tile_size,
        "attention_semantics": "context_masked" if use_masked_attention else "all_valid_no_mask",
        "context_valid_tokens": int(attention_mask.sum()),
        "bundle": args.bundle,
        "pre_compiled_path": str(pre_dir),
        "attention_compiled_path": str(attn_dir),
        "post_compiled_path": str(post_dir),
        **meta,
        **compile_metrics,
        **attention_metrics,
        "pre_qkv_forward_elapsed_s": pre_elapsed,
        "post_forward_elapsed_s": post_elapsed,
        "reference_skipped": args.skip_reference,
    }
    if not args.skip_reference:
        full_block = _load_full_cpu_block(args)
        t2 = time.perf_counter()
        with torch.no_grad():
            reference_hidden, reference_context = full_block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                (freqs_cos, freqs_sin),
            )
        reference_elapsed = time.perf_counter() - t2

        hidden_diff = (
            segmented_hidden.detach().cpu().float() - reference_hidden.detach().cpu().float()
        ).abs()
        context_diff = (
            segmented_context.detach().cpu().float() - reference_context.detach().cpu().float()
        ).abs()
        hidden_cosine = _cosine(segmented_hidden.detach().cpu(), reference_hidden.detach().cpu())
        context_cosine = _cosine(segmented_context.detach().cpu(), reference_context.detach().cpu())
        metrics.update(
            {
                "reference_elapsed_s": reference_elapsed,
                "hidden_cosine": hidden_cosine,
                "hidden_max_abs": float(hidden_diff.max()),
                "hidden_mean_abs": float(hidden_diff.mean()),
                "context_cosine": context_cosine,
                "context_max_abs": float(context_diff.max()),
                "context_mean_abs": float(context_diff.mean()),
            }
        )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[hunyuan15-segmented-block] metrics -> {metrics_path}", flush=True)
    if args.skip_reference:
        print("[hunyuan15-segmented-block] runtime gate = PASS", flush=True)
        return 0
    pass_gate = metrics["hidden_cosine"] >= args.min_cosine and metrics["context_cosine"] >= args.min_cosine
    print(f"[hunyuan15-segmented-block] gate = {'PASS' if pass_gate else 'FAIL'}", flush=True)
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
