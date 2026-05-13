#!/usr/bin/env python3
"""Run HunyuanVideo 1.5 segmented blocks with one worker process per block.

This is a runtime-lifecycle probe for the full 54-layer segmented path. The
in-process streaming mode proves that repeated Neuron ``initialize()`` calls
leak or retain device tensors; this script moves each block into a fresh Python
process so the Neuron runtime can release HBM when the worker exits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

from hunyuan15_transformer_parity import (  # noqa: E402
    _cosine,
    _load_bundle_inputs,
    _load_config,
    _parse_dtype,
    _run_reference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--cache-dir", default="/tmp/nova_hunyuan15_segmented_process_cache")
    parser.add_argument("--work-dir", default="/tmp/nova_hunyuan15_segmented_process_work")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--timestep-index", type=int, default=0)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=61)
    parser.add_argument("--text-seq-len", type=int, default=1000)
    parser.add_argument("--text-seq-len-2", type=int, default=256)
    parser.add_argument("--image-seq-len", type=int, default=729)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--reference-dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--query-tile-size", type=int, default=489)
    parser.add_argument("--key-tile-size", type=int, default=489)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--save-trainium", default=None)
    parser.add_argument("--worker-timeout-s", type=int, default=1800)
    parser.add_argument("--block-compiler-args", default=None)
    parser.add_argument("--attention-compiler-args", default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--input-tensors", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-tensors", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--block-index", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float32:
        return "fp32"
    raise ValueError(f"unsupported dtype: {dtype}")


def _load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _save_pt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def _precompile(args: argparse.Namespace):
    os.environ.setdefault("NOVA_BACKEND", "trainium")
    from nova import NovaParallelConfig, NovaPipeline

    app_kwargs: dict[str, Any] = {
        "transformer_runtime": "segmented",
        "segmented_block_load_mode": "streaming",
        "segmented_query_tile_size": args.query_tile_size,
        "segmented_key_tile_size": args.key_tile_size,
        "text_seq_len": args.text_seq_len,
        "text_seq_len_2": args.text_seq_len_2,
        "image_seq_len": args.image_seq_len,
    }
    if args.transformer_subfolder != "transformer":
        app_kwargs["transformer_subfolder"] = args.transformer_subfolder
    if args.block_compiler_args is not None:
        app_kwargs["segmented_block_compiler_args"] = args.block_compiler_args
    if args.attention_compiler_args is not None:
        app_kwargs["segmented_attention_compiler_args"] = args.attention_compiler_args

    return NovaPipeline.from_pretrained(
        args.model_dir,
        model_type="hunyuan_video_15",
        parallel=NovaParallelConfig(tp_degree=args.tp_degree),
        dtype=args.dtype,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        compile_cache_dir=args.cache_dir,
        force_compile=args.force_compile,
        skip_compile=args.skip_compile,
        load=False,
        application_kwargs=app_kwargs,
    )


def _run_worker(args: argparse.Namespace) -> int:
    if args.runtime_config is None or args.input_tensors is None or args.output_tensors is None:
        raise ValueError("--worker requires runtime config, input tensors, and output tensors")

    from nova.backends.trainium.core.config import NeuronConfig
    from nova.backends.trainium.hunyuan_video.segmented15 import (
        HunyuanVideo15AttentionTileApplication,
        HunyuanVideo15AttentionTileConfig,
        HunyuanVideo15BlockSegmentApplication,
        HunyuanVideo15BlockSegmentConfig,
        run_streaming_manual_stats_attention,
    )

    cfg = json.loads(Path(args.runtime_config).read_text(encoding="utf-8"))
    dtype = _parse_dtype(cfg["dtype"])
    neuron_config = NeuronConfig(
        batch_size=1,
        tp_degree=int(cfg["tp_degree"]),
        world_size=int(cfg["tp_degree"]),
        torch_dtype=dtype,
        skip_sharding=True,
    )

    def block_app(part: str) -> HunyuanVideo15BlockSegmentApplication:
        return HunyuanVideo15BlockSegmentApplication(
            model_path=cfg["transformer_dir"],
            config=HunyuanVideo15BlockSegmentConfig(
                neuron_config=neuron_config,
                part=part,
                latent_seq_len=int(cfg["latent_seq_len"]),
                context_seq_len=int(cfg["context_seq_len"]),
                total_seq_len=int(cfg["total_seq_len"]),
                inner_dim=int(cfg["inner_dim"]),
                heads=int(cfg["heads"]),
                head_dim=int(cfg["head_dim"]),
                mlp_ratio=float(cfg["mlp_ratio"]),
                qk_norm=str(cfg["qk_norm"]),
                source_model_dir=cfg["transformer_dir"],
                block_index=int(args.block_index),
            ),
            compiler_args=str(cfg["block_compiler_args"]),
        )

    pre = block_app("pre-qkv")
    post = block_app("post")
    attention = HunyuanVideo15AttentionTileApplication(
        model_path=cfg["transformer_dir"],
        config=HunyuanVideo15AttentionTileConfig(
            neuron_config=neuron_config,
            query_len=int(cfg["query_tile_size"]),
            key_len=int(cfg["key_tile_size"]),
            heads=int(cfg["heads"]),
            head_dim=int(cfg["head_dim"]),
        ),
        compiler_args=str(cfg["attention_compiler_args"]),
    )

    compiled_path = Path(cfg["compiled_path"])
    t0 = time.perf_counter()
    pre.load(str(compiled_path / "transformer_block_pre_qkv"), skip_warmup=True)
    attention.load(str(compiled_path / "transformer_attention_tile"), skip_warmup=True)
    post.load(str(compiled_path / "transformer_block_post"), skip_warmup=True)
    load_elapsed = time.perf_counter() - t0

    data = _load_pt(Path(args.input_tensors))
    t1 = time.perf_counter()
    with torch.no_grad():
        query, key, value = pre(
            data["hidden_states"],
            data["encoder_hidden_states"],
            data["temb"],
            data["freqs_cos"],
            data["freqs_sin"],
        )
    pre_elapsed = time.perf_counter() - t1

    attention_states, attention_metrics = run_streaming_manual_stats_attention(
        attention,
        query.detach().cpu(),
        key.detach().cpu(),
        value.detach().cpu(),
        query_tile_size=int(cfg["query_tile_size"]),
        key_tile_size=int(cfg["key_tile_size"]),
        valid_mask=data["valid_mask"],
    )

    t2 = time.perf_counter()
    with torch.no_grad():
        hidden_states, encoder_hidden_states = post(
            data["hidden_states"],
            data["encoder_hidden_states"],
            data["temb"],
            attention_states.to(dtype=dtype),
        )
    post_elapsed = time.perf_counter() - t2

    _save_pt(
        Path(args.output_tensors),
        {
            "hidden_states": hidden_states.detach().cpu(),
            "encoder_hidden_states": encoder_hidden_states.detach().cpu(),
            "metrics": {
                "block_index": int(args.block_index),
                "worker_load_elapsed_s": load_elapsed,
                "pre_elapsed_s": pre_elapsed,
                "post_elapsed_s": post_elapsed,
                "stream_tile_calls": int(attention_metrics.stream_tile_calls),
                "stream_forward_elapsed_s": float(attention_metrics.stream_forward_elapsed_s),
            },
        },
    )
    return 0


def _spawn_block_worker(
    args: argparse.Namespace,
    *,
    runtime_config: Path,
    input_tensors: Path,
    output_tensors: Path,
    block_index: int,
) -> None:
    env = os.environ.copy()
    env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.setdefault("NOVA_BACKEND", "trainium")
    env["LOCAL_WORLD_SIZE"] = str(args.tp_degree)
    cmd = [
        str(NEURON_PYTHON if NEURON_PYTHON.exists() else Path(sys.executable)),
        str(Path(__file__).resolve()),
        "--worker",
        "--runtime-config",
        str(runtime_config),
        "--input-tensors",
        str(input_tensors),
        "--output-tensors",
        str(output_tensors),
        "--block-index",
        str(block_index),
        "--model-dir",
        args.model_dir,
        "--bundle",
        args.bundle,
    ]
    subprocess.run(cmd, env=env, check=True, timeout=args.worker_timeout_s)


def main() -> int:
    args = build_parser().parse_args()
    if args.worker:
        return _run_worker(args)

    cfg = _load_config(Path(args.model_dir) / args.transformer_subfolder)
    inputs = _load_bundle_inputs(args, cfg, args.dtype)
    t_compile = time.perf_counter()
    pipe = _precompile(args)
    compile_elapsed = time.perf_counter() - t_compile
    transformer = pipe.app.transformer
    if transformer is None:
        raise RuntimeError("HunyuanVideo 1.5 segmented transformer was not constructed")

    t_frontend = time.perf_counter()
    hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb = (
        transformer._prepare_frontend(**inputs)
    )
    frontend_elapsed = time.perf_counter() - t_frontend
    freqs_cos, freqs_sin = image_rotary_emb
    valid_mask = torch.cat(
        [
            torch.ones([attention_mask.shape[0], transformer.meta["latent_seq_len"]], dtype=torch.bool),
            attention_mask.to(dtype=torch.bool),
        ],
        dim=1,
    )

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    static = {
        "temb": temb.detach().cpu(),
        "freqs_cos": freqs_cos.detach().cpu(),
        "freqs_sin": freqs_sin.detach().cpu(),
        "valid_mask": valid_mask.detach().cpu(),
    }
    runtime_config = work_dir / "runtime_config.json"
    runtime_config.write_text(
        json.dumps(
            {
                "transformer_dir": str(Path(args.model_dir) / args.transformer_subfolder),
                "compiled_path": str(pipe.compiled_path),
                "dtype": _dtype_name(args.dtype),
                "tp_degree": args.tp_degree,
                "query_tile_size": args.query_tile_size,
                "key_tile_size": args.key_tile_size,
                "block_compiler_args": transformer.pre_blocks[0].models[0].compiler_args,
                "attention_compiler_args": transformer.attention.models[0].compiler_args,
                "heads": int(transformer.config.num_attention_heads),
                "head_dim": int(transformer.config.attention_head_dim),
                "mlp_ratio": float(transformer.config.mlp_ratio),
                "qk_norm": str(transformer.config.qk_norm),
                **transformer.meta,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    block_count = int(transformer.config.num_layers)
    if args.max_blocks is not None:
        block_count = min(block_count, int(args.max_blocks))
    block_metrics = []
    t_blocks = time.perf_counter()
    for block_index in range(block_count):
        input_path = work_dir / f"block_{block_index:02d}_input.pt"
        output_path = work_dir / f"block_{block_index:02d}_output.pt"
        _save_pt(
            input_path,
            {
                "hidden_states": hidden_states.detach().cpu(),
                "encoder_hidden_states": encoder_hidden_states.detach().cpu(),
                **static,
            },
        )
        t_block = time.perf_counter()
        _spawn_block_worker(
            args,
            runtime_config=runtime_config,
            input_tensors=input_path,
            output_tensors=output_path,
            block_index=block_index,
        )
        result = _load_pt(output_path)
        hidden_states = result["hidden_states"]
        encoder_hidden_states = result["encoder_hidden_states"]
        metrics = dict(result["metrics"])
        metrics["block_wall_elapsed_s"] = time.perf_counter() - t_block
        block_metrics.append(metrics)
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        print(
            f"[hunyuan15-process] block {block_index + 1}/{block_count} "
            f"elapsed={metrics['block_wall_elapsed_s']:.3f}s",
            flush=True,
        )
    blocks_elapsed = time.perf_counter() - t_blocks

    t_final = time.perf_counter()
    with torch.no_grad():
        trainium = transformer._final_projection(hidden_states, temb.detach().cpu()).detach().cpu()
    final_elapsed = time.perf_counter() - t_final

    metrics: dict[str, Any] = {
        "model_dir": args.model_dir,
        "transformer_subfolder": args.transformer_subfolder,
        "compiled_path": str(pipe.compiled_path),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "block_count": block_count,
        "total_model_layers": int(transformer.config.num_layers),
        "query_tile_size": args.query_tile_size,
        "key_tile_size": args.key_tile_size,
        "compile_elapsed_s": compile_elapsed,
        "frontend_elapsed_s": frontend_elapsed,
        "blocks_elapsed_s": blocks_elapsed,
        "final_elapsed_s": final_elapsed,
        "trainium_shape": list(trainium.shape),
        "trainium_mean": float(trainium.float().mean()),
        "trainium_absmax": float(trainium.float().abs().max()),
        "trainium_checksum": float(trainium.float().sum()),
        "stream_tile_calls": int(sum(m["stream_tile_calls"] for m in block_metrics)),
        "stream_forward_elapsed_s": float(sum(m["stream_forward_elapsed_s"] for m in block_metrics)),
        "block_metrics": block_metrics,
        "reference_skipped": args.skip_reference,
    }
    if not args.skip_reference:
        reference, reference_elapsed = _run_reference(args, cfg, inputs)
        diff = (trainium.float() - reference.float()).abs()
        metrics.update(
            {
                "cosine": _cosine(trainium, reference),
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
                "reference_elapsed_s": reference_elapsed,
                "reference_shape": list(reference.shape),
            }
        )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if args.metrics_out:
        path = Path(args.metrics_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[hunyuan15-process] metrics -> {path}", flush=True)
    if args.save_trainium:
        torch.save(trainium, args.save_trainium)
    if args.skip_reference:
        print("[hunyuan15-process] runtime gate = PASS", flush=True)
        return 0
    pass_gate = metrics["cosine"] >= args.min_cosine
    print(f"[hunyuan15-process] gate = {'PASS' if pass_gate else 'FAIL'}", flush=True)
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
