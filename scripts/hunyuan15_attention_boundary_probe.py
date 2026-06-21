#!/usr/bin/env python3
"""Probe HunyuanVideo 1.5 block attention at the real Q/K/V boundary.

This script is an experimental capacity probe. It runs the diffusers front half
of a HunyuanVideo 1.5 transformer on CPU, extracts the Q/K/V tensors for one
dual-stream block, then feeds those real tensors through the compiled
manual-stats streaming attention tile from ``hunyuan15_attention_capacity_probe``.

The tile path intentionally omits the Hunyuan attention mask. Use it to test
whether real projected Q/K/V tensors can be streamed at production sequence
length; do not treat it as a full transformer parity gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

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
from hunyuan15_transformer_parity import (  # noqa: E402
    _cosine,
    _load_bundle_inputs,
    _load_config,
    _make_inputs,
    _parse_dtype,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--cache-dir", default="/tmp/difflet_hunyuan15_attention_boundary_cache")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--text-seq-len", type=int, default=1000)
    parser.add_argument("--text-seq-len-2", type=int, default=256)
    parser.add_argument("--image-seq-len", type=int, default=729)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--timestep-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--query-tile-size", type=int, default=2051)
    parser.add_argument("--key-tile-size", type=int, default=2051)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument(
        "--random-image-embeds",
        action="store_true",
        help="Replace image_embeds with random values so the CPU frontend treats image tokens as valid.",
    )
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument(
        "--compiler-args",
        default=(
            "--model-type=generic -O1 --auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        ),
    )
    return parser


def _prepare_frontend(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    use_meanflow: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    hidden_states = inputs["hidden_states"]
    timestep = inputs["timestep"]
    timestep_r = inputs["timestep_r"] if use_meanflow else None
    encoder_hidden_states = inputs["encoder_hidden_states"]
    encoder_attention_mask = inputs["encoder_attention_mask"]
    encoder_hidden_states_2 = inputs["encoder_hidden_states_2"]
    encoder_attention_mask_2 = inputs["encoder_attention_mask_2"]
    image_embeds = inputs["image_embeds"]

    batch_size = hidden_states.shape[0]
    image_rotary_emb = model.rope(hidden_states)
    temb = model.time_embed(timestep, timestep_r=timestep_r)
    hidden_states = model.x_embedder(hidden_states)

    encoder_hidden_states = model.context_embedder(
        encoder_hidden_states,
        timestep,
        encoder_attention_mask,
    )
    encoder_hidden_states = encoder_hidden_states + model.cond_type_embed(
        torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long)
    )

    encoder_hidden_states_2 = model.context_embedder_2(encoder_hidden_states_2)
    encoder_hidden_states_2 = encoder_hidden_states_2 + model.cond_type_embed(
        torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long)
    )

    encoder_hidden_states_3 = model.image_embedder(image_embeds)
    is_t2v = torch.all(image_embeds == 0)
    if is_t2v:
        encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
        encoder_attention_mask_3 = torch.zeros(
            (batch_size, encoder_hidden_states_3.shape[1]),
            dtype=encoder_attention_mask.dtype,
            device=encoder_attention_mask.device,
        )
    else:
        encoder_attention_mask_3 = torch.ones(
            (batch_size, encoder_hidden_states_3.shape[1]),
            dtype=encoder_attention_mask.dtype,
            device=encoder_attention_mask.device,
        )
    encoder_hidden_states_3 = encoder_hidden_states_3 + model.cond_type_embed(
        2 * torch.ones_like(encoder_hidden_states_3[:, :, 0], dtype=torch.long)
    )

    encoder_attention_mask = encoder_attention_mask.bool()
    encoder_attention_mask_2 = encoder_attention_mask_2.bool()
    encoder_attention_mask_3 = encoder_attention_mask_3.bool()
    new_encoder_hidden_states = []
    new_encoder_attention_mask = []
    for text, text_mask, text_2, text_mask_2, image, image_mask in zip(
        encoder_hidden_states,
        encoder_attention_mask,
        encoder_hidden_states_2,
        encoder_attention_mask_2,
        encoder_hidden_states_3,
        encoder_attention_mask_3,
    ):
        new_encoder_hidden_states.append(
            torch.cat(
                [
                    image[image_mask],
                    text_2[text_mask_2],
                    text[text_mask],
                    image[~image_mask],
                    torch.zeros_like(text_2[~text_mask_2]),
                    torch.zeros_like(text[~text_mask]),
                ],
                dim=0,
            )
        )
        new_encoder_attention_mask.append(
            torch.cat(
                [
                    image_mask[image_mask],
                    text_mask_2[text_mask_2],
                    text_mask[text_mask],
                    image_mask[~image_mask],
                    text_mask_2[~text_mask_2],
                    text_mask[~text_mask],
                ],
                dim=0,
            )
        )

    return (
        hidden_states,
        torch.stack(new_encoder_hidden_states),
        temb,
        torch.stack(new_encoder_attention_mask),
        image_rotary_emb,
    )


def _extract_block_qkv(
    block: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    from diffusers.models.embeddings import apply_rotary_emb

    norm_hidden_states, *_ = block.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, *_ = block.norm1_context(encoder_hidden_states, emb=temb)
    attn = block.attn

    query = attn.to_q(norm_hidden_states)
    key = attn.to_k(norm_hidden_states)
    value = attn.to_v(norm_hidden_states)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    query = attn.norm_q(query)
    key = attn.norm_k(key)
    query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
    key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    encoder_query = attn.add_q_proj(norm_encoder_hidden_states)
    encoder_key = attn.add_k_proj(norm_encoder_hidden_states)
    encoder_value = attn.add_v_proj(norm_encoder_hidden_states)

    encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
    encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
    encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

    if attn.norm_added_q is not None:
        encoder_query = attn.norm_added_q(encoder_query)
    if attn.norm_added_k is not None:
        encoder_key = attn.norm_added_k(encoder_key)

    qkv_meta = {
        "latent_query_len": int(query.shape[1]),
        "context_query_len": int(encoder_query.shape[1]),
        "heads": int(query.shape[2]),
        "head_dim": int(query.shape[3]),
    }
    query = torch.cat([query, encoder_query], dim=1).contiguous()
    key = torch.cat([key, encoder_key], dim=1).contiguous()
    value = torch.cat([value, encoder_value], dim=1).contiguous()
    return query, key, value, qkv_meta


def _make_tile_args(args: argparse.Namespace, query: torch.Tensor, key: torch.Tensor) -> argparse.Namespace:
    if query.shape[1] % args.query_tile_size != 0:
        raise ValueError(
            f"query_len {query.shape[1]} is not divisible by --query-tile-size {args.query_tile_size}"
        )
    if key.shape[1] % args.key_tile_size != 0:
        raise ValueError(f"key_len {key.shape[1]} is not divisible by --key-tile-size {args.key_tile_size}")
    return argparse.Namespace(
        backend="manual-stats",
        cache_dir=args.cache_dir,
        query_len=args.query_tile_size,
        key_len=args.key_tile_size,
        heads=int(query.shape[2]),
        head_dim=int(query.shape[3]),
        layout="bshd",
        tp_degree=args.tp_degree,
        force_clean=args.force_compile,
        run_stream_merge_check=False,
        stream_query_tiles=1,
        stream_key_tiles=1,
        skip_stream_reference=True,
        seed=args.seed,
        metrics_out=None,
        compiler_args=args.compiler_args,
    )


def _load_inputs(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    inputs = (
        _load_bundle_inputs(args, cfg, args.dtype)
        if args.bundle is not None
        else _make_inputs(args, cfg, args.dtype)
    )
    if args.random_image_embeds:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 17)
        inputs["image_embeds"] = torch.randn(
            inputs["image_embeds"].shape,
            generator=generator,
            dtype=args.dtype,
        )
    return inputs


def main() -> int:
    args = build_parser().parse_args()
    transformer_dir = Path(args.model_dir) / args.transformer_subfolder
    cfg = _load_config(transformer_dir)
    inputs = _load_inputs(args, cfg)

    from diffusers.models.transformers.transformer_hunyuan_video15 import (
        HunyuanVideo15Transformer3DModel,
    )

    model = HunyuanVideo15Transformer3DModel.from_pretrained(
        transformer_dir,
        torch_dtype=args.dtype,
    ).eval()
    if args.block_index < 0 or args.block_index >= len(model.transformer_blocks):
        raise IndexError(f"--block-index {args.block_index} out of range for {len(model.transformer_blocks)} blocks")

    t0 = time.perf_counter()
    with torch.no_grad():
        hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb = _prepare_frontend(
            model,
            inputs,
            use_meanflow=bool(cfg.get("use_meanflow", False)),
        )
        query, key, value, qkv_meta = _extract_block_qkv(
            model.transformer_blocks[args.block_index],
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb,
        )
    qkv_elapsed = time.perf_counter() - t0

    tile_args = _make_tile_args(args, query, key)
    app, cache_dir = build_attention_capacity_application(tile_args)
    compile_elapsed = None
    if not args.skip_compile:
        t1 = time.perf_counter()
        app.compile(str(cache_dir), debug=False)
        compile_elapsed = time.perf_counter() - t1
    t2 = time.perf_counter()
    app.load(str(cache_dir), skip_warmup=True)
    load_elapsed = time.perf_counter() - t2

    stream_output, stream_metrics = run_streaming_manual_stats_attention(
        app,
        query,
        key,
        value,
        query_tile_size=args.query_tile_size,
        key_tile_size=args.key_tile_size,
        collect_output=not args.skip_reference,
    )

    metrics = {
        "model_dir": args.model_dir,
        "transformer_subfolder": args.transformer_subfolder,
        "block_index": args.block_index,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "bundle": args.bundle,
        "attention_semantics": "no_mask_capacity",
        "attention_mask_valid_tokens": int(attention_mask.sum()),
        "attention_mask_total_tokens": int(attention_mask.numel()),
        "qkv_extract_elapsed_s": qkv_elapsed,
        "compile_elapsed_s": compile_elapsed,
        "load_elapsed_s": load_elapsed,
        "compiled_path": str(cache_dir),
        "query_shape": list(query.shape),
        "key_shape": list(key.shape),
        "value_shape": list(value.shape),
        "query_tile_size": args.query_tile_size,
        "key_tile_size": args.key_tile_size,
        **qkv_meta,
        **stream_metrics,
    }

    if not args.skip_reference:
        t3 = time.perf_counter()
        reference = torch.nn.functional.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3).float(),
            key.permute(0, 2, 1, 3).float(),
            value.permute(0, 2, 1, 3).float(),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        reference_elapsed = time.perf_counter() - t3
        diff = (stream_output.float() - reference.float()).abs()
        metrics.update(
            {
                "reference_elapsed_s": reference_elapsed,
                "cosine": _cosine(stream_output, reference),
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
            }
        )

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[hunyuan15-attn-boundary] metrics -> {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
