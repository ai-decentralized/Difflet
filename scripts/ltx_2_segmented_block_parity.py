#!/usr/bin/env python3
"""Compare one compiled LTX-2 segmented block against the CPU block reference."""

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


def ensure_runtime_python() -> None:
    if Path(sys.executable) == NEURON_PYTHON or not NEURON_PYTHON.exists():
        return
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        env = os.environ.copy()
        env["PATH"] = f"{NEURON_VENV / 'bin'}:{env.get('PATH', '')}"
        env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)


def _parse_dtype(value: str):
    import torch

    normalized = value.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("DIFFLET_LTX_2_MODEL_DIR", ""),
        help="Local LTX-2 snapshot dir. Env: DIFFLET_LTX_2_MODEL_DIR.",
    )
    parser.add_argument("--cache-dir", default="/tmp/difflet_ltx2_segmented_block_probe_cache")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--audio-num-frames", type=int, default=126)
    parser.add_argument("--text-seq-len", type=int, default=1024)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", type=_parse_dtype, default="bf16")
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--min-video-cosine", type=float, default=0.99)
    parser.add_argument("--min-audio-cosine", type=float, default=0.99)
    parser.add_argument("--max-video-mean-abs", type=float, default=None)
    parser.add_argument("--max-audio-mean-abs", type=float, default=None)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--num-threads", type=int, default=0)
    return parser


def _cosine(a, b) -> float:
    import torch

    return torch.nn.functional.cosine_similarity(
        a.detach().float().reshape(-1),
        b.detach().float().reshape(-1),
        dim=0,
    ).item()


def _extract_pair(output):
    import torch

    if isinstance(output, (tuple, list)) and len(output) >= 2:
        first, second = output[0], output[1]
        if torch.is_tensor(first) and torch.is_tensor(second):
            return first, second
    raise TypeError(f"cannot extract block output pair from {type(output)!r}")


def _load_cpu_block(config, model_dir: Path, block_index: int):
    from difflet.backends.trainium.ltx_2.segmented import (
        _LTX2BlockModule,
        _load_block_state_dict_from_dir,
    )

    mx_swap = os.environ.pop("DIFFLET_LTX2_MX_ALL_E4M3", None)
    try:
        block = _LTX2BlockModule(config)
    finally:
        if mx_swap is not None:
            os.environ["DIFFLET_LTX2_MX_ALL_E4M3"] = mx_swap
    state = _load_block_state_dict_from_dir(
        model_dir / "transformer",
        block_index,
        dtype=config.neuron_config.torch_dtype,
    )
    missing, unexpected = block.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"CPU block state mismatch: missing={missing}, unexpected={unexpected}")
    return block.to(dtype=config.neuron_config.torch_dtype).eval()


def _make_inputs(block_app, seed: int):
    import torch

    torch.manual_seed(seed)
    return tuple(t.detach().cpu().contiguous() for t in block_app.model.input_generator()[0])


def main() -> int:
    ensure_runtime_python()
    args = build_parser().parse_args()
    if not args.model_dir:
        raise ValueError("--model-dir or DIFFLET_LTX_2_MODEL_DIR is required")

    import torch

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    from difflet import DiffletParallelConfig, DiffletPipeline

    model_dir = Path(args.model_dir).expanduser().resolve()
    t0 = time.perf_counter()
    pipe = DiffletPipeline.from_pretrained(
        str(model_dir),
        model_type="ltx_2",
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
        dtype=args.dtype,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        compile_cache_dir=args.cache_dir,
        force_compile=args.force_compile,
        skip_compile=args.skip_compile,
        local_files_only=True,
        load=True,
        skip_warmup=args.skip_warmup,
        application_kwargs={
            "transformer_mode": "segmented",
            "audio_num_frames": args.audio_num_frames,
            "text_seq_len": args.text_seq_len,
        },
    )
    load_elapsed = time.perf_counter() - t0
    segmented = pipe.app.transformer
    block_app = segmented.block
    if args.block_index < 0 or args.block_index >= int(segmented.config.num_layers):
        raise IndexError(f"--block-index must be in [0, {int(segmented.config.num_layers)})")
    block_app.reload_block_weights(args.block_index)

    inputs = _make_inputs(block_app, args.seed)
    cpu_block = _load_cpu_block(segmented.config, model_dir, args.block_index)
    cpu_inputs = [
        tensor.to(dtype=args.dtype) if torch.is_floating_point(tensor) else tensor
        for tensor in inputs
    ]

    t1 = time.perf_counter()
    with torch.no_grad():
        trainium_video, trainium_audio = _extract_pair(block_app(*inputs))
    trainium_elapsed = time.perf_counter() - t1

    t2 = time.perf_counter()
    with torch.no_grad():
        reference_video, reference_audio = _extract_pair(cpu_block(*cpu_inputs))
    reference_elapsed = time.perf_counter() - t2

    trainium_video = trainium_video.detach().cpu()
    trainium_audio = trainium_audio.detach().cpu()
    reference_video = reference_video.detach().cpu()
    reference_audio = reference_audio.detach().cpu()
    if trainium_video.shape != reference_video.shape:
        raise RuntimeError(
            f"video shape mismatch: trainium={tuple(trainium_video.shape)} "
            f"reference={tuple(reference_video.shape)}"
        )
    if trainium_audio.shape != reference_audio.shape:
        raise RuntimeError(
            f"audio shape mismatch: trainium={tuple(trainium_audio.shape)} "
            f"reference={tuple(reference_audio.shape)}"
        )

    video_diff = (trainium_video.float() - reference_video.float()).abs()
    audio_diff = (trainium_audio.float() - reference_audio.float()).abs()
    metrics = {
        "model_dir": str(model_dir),
        "compiled_path": str(pipe.compiled_path),
        "block_index": args.block_index,
        "tp_degree": args.tp_degree,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "audio_num_frames": args.audio_num_frames,
        "text_seq_len": args.text_seq_len,
        "load_elapsed_s": load_elapsed,
        "trainium_elapsed_s": trainium_elapsed,
        "reference_elapsed_s": reference_elapsed,
        "video_cosine": _cosine(trainium_video, reference_video),
        "audio_cosine": _cosine(trainium_audio, reference_audio),
        "video_max_abs": float(video_diff.max()),
        "audio_max_abs": float(audio_diff.max()),
        "video_mean_abs": float(video_diff.mean()),
        "audio_mean_abs": float(audio_diff.mean()),
        "trainium_video_shape": list(trainium_video.shape),
        "trainium_audio_shape": list(trainium_audio.shape),
        "reference_video_shape": list(reference_video.shape),
        "reference_audio_shape": list(reference_audio.shape),
    }
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[ltx2-segmented-block-parity] metrics -> {metrics_path}", flush=True)

    pass_gate = (
        metrics["video_cosine"] >= args.min_video_cosine
        and metrics["audio_cosine"] >= args.min_audio_cosine
    )
    if args.max_video_mean_abs is not None:
        pass_gate = pass_gate and metrics["video_mean_abs"] <= args.max_video_mean_abs
    if args.max_audio_mean_abs is not None:
        pass_gate = pass_gate and metrics["audio_mean_abs"] <= args.max_audio_mean_abs
    print(f"[ltx2-segmented-block-parity] gate = {'PASS' if pass_gate else 'FAIL'}", flush=True)
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
