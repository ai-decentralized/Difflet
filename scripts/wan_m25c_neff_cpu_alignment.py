#!/usr/bin/env python3
"""M2.5-C Wan UMT5 / DiT NEFF-vs-CPU numerical alignment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file


DEFAULT_MODEL_DIR = (
    "/home/ubuntu/.cache/huggingface/hub/"
    "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/"
    "snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all")
    parser.add_argument("--model-dir", default=os.environ.get("DIFFLET_WAN_MODEL_DIR", DEFAULT_MODEL_DIR))
    parser.add_argument("--work-dir", default=os.environ.get("DIFFLET_M25C_WORK_DIR", "/tmp/difflet_m25c"))
    parser.add_argument("--prompt", default=os.environ.get("DIFFLET_M25C_PROMPT", "a cat walking"))
    parser.add_argument("--text-seq-len", type=int, default=int(os.environ.get("DIFFLET_M25C_TEXT_SEQ_LEN", "512")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("DIFFLET_M25C_HEIGHT", "480")))
    parser.add_argument("--width", type=int, default=int(os.environ.get("DIFFLET_M25C_WIDTH", "832")))
    parser.add_argument("--video-frames", type=int, default=int(os.environ.get("DIFFLET_M25C_VIDEO_FRAMES", "9")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("DIFFLET_M25C_SEED", "20260510")))
    parser.add_argument("--tp-degree", type=int, default=int(os.environ.get("DIFFLET_M25C_TP_DEGREE", "4")))
    parser.add_argument("--text-compiled-dir", default=os.environ.get("DIFFLET_M25C_TEXT_COMPILED_DIR", ".difflet-cache/wan_text_encoder_smoke"))
    parser.add_argument("--dit-compiled-dir", default=os.environ.get("DIFFLET_M25C_DIT_COMPILED_DIR", ".difflet-cache/wan_backbone_smoke"))
    parser.add_argument("--min-mem-gb", type=float, default=float(os.environ.get("DIFFLET_M25C_MIN_MEM_GB", "20")))
    parser.add_argument("--peak-rss-max-gb", type=float, default=float(os.environ.get("DIFFLET_M25C_PEAK_RSS_MAX_GB", "115")))
    parser.add_argument("--umt5-cosine-min", type=float, default=float(os.environ.get("DIFFLET_M25C_UMT5_COSINE_MIN", "0.995")))
    parser.add_argument("--umt5-mean-abs-max", type=float, default=float(os.environ.get("DIFFLET_M25C_UMT5_MEAN_ABS_MAX", "0.03")))
    parser.add_argument("--dit-cosine-min", type=float, default=float(os.environ.get("DIFFLET_M25C_DIT_COSINE_MIN", "0.995")))
    parser.add_argument("--dit-mean-abs-max", type=float, default=float(os.environ.get("DIFFLET_M25C_DIT_MEAN_ABS_MAX", "0.05")))
    return parser.parse_args()


def latent_frames(video_frames: int) -> int:
    return (int(video_frames) - 1) // 4 + 1


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return -1.0


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def require_memory(args: argparse.Namespace, stage: str) -> None:
    available = mem_available_gb()
    print(f"[m25c:{stage}] mem_available_gb = {available:.2f}", flush=True)
    if available >= 0 and available < args.min_mem_gb:
        raise RuntimeError(
            f"Not enough available memory for {stage}: {available:.2f}GB < {args.min_mem_gb:.2f}GB"
        )


def save_metrics(args: argparse.Namespace, stage: str, metrics: dict[str, Any]) -> None:
    metrics = dict(metrics)
    metrics["stage"] = stage
    metrics["peak_rss_gb"] = peak_rss_gb()
    metrics["peak_rss_limit_gb"] = args.peak_rss_max_gb
    metrics["mem_available_gb_after"] = mem_available_gb()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / f"{stage}.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"[m25c:{stage}] peak_rss_gb = {metrics['peak_rss_gb']:.2f}", flush=True)
    if metrics["peak_rss_gb"] > args.peak_rss_max_gb:
        raise RuntimeError(
            f"{stage} peak RSS too high: {metrics['peak_rss_gb']:.2f}GB > {args.peak_rss_max_gb:.2f}GB"
        )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def iter_safetensor_shards(component_dir: Path) -> list[Path]:
    for index_name in ("diffusion_pytorch_model.safetensors.index.json", "model.safetensors.index.json"):
        index_path = component_dir / index_name
        if index_path.exists():
            index = load_json(index_path)
            return [component_dir / name for name in sorted(set(index["weight_map"].values()))]
    for name in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
        path = component_dir / name
        if path.exists():
            return [path]
    raise FileNotFoundError(f"no safetensors shard/index found under {component_dir}")


def load_sharded_state_dict(
    model: torch.nn.Module,
    component_dir: Path,
    *,
    rename_fn=None,
    dtype: torch.dtype = torch.bfloat16,
    extra_aliases: dict[str, str] | None = None,
) -> None:
    expected = set(model.state_dict().keys())
    loaded: set[str] = set()
    unexpected_all: set[str] = set()
    for shard in iter_safetensor_shards(component_dir):
        print(f"[m25c:load] shard = {shard.name}", flush=True)
        state = load_safetensors_file(str(shard), device="cpu")
        if rename_fn is not None:
            state = rename_fn(state)
        if extra_aliases:
            for dst, src in extra_aliases.items():
                if src in state and dst not in state:
                    state[dst] = state[src]
        state = {
            key: value.to(dtype) if torch.is_floating_point(value) else value
            for key, value in state.items()
        }
        _, unexpected = model.load_state_dict(state, strict=False)
        loaded.update(state.keys())
        unexpected_all.update(unexpected)
        del state
        gc.collect()
    missing = sorted(expected - loaded)
    unexpected = sorted(unexpected_all)
    if missing or unexpected:
        raise RuntimeError(
            f"sharded load mismatch for {component_dir}: missing={missing[:8]} "
            f"(n={len(missing)}), unexpected={unexpected[:8]} (n={len(unexpected)})"
        )


def run_child(args: argparse.Namespace, stage: str, *, backend: str | None = None, neuron_cores: int | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    if backend is not None:
        env["DIFFLET_BACKEND"] = backend
    if neuron_cores is not None:
        env["NEURON_RT_NUM_CORES"] = str(neuron_cores)
        env.setdefault("NEURON_RT_VIRTUAL_CORE_SIZE", "2")
    cmd = [
        sys.executable,
        __file__,
        "--stage", stage,
        "--model-dir", args.model_dir,
        "--work-dir", args.work_dir,
        "--prompt", args.prompt,
        "--text-seq-len", str(args.text_seq_len),
        "--height", str(args.height),
        "--width", str(args.width),
        "--video-frames", str(args.video_frames),
        "--seed", str(args.seed),
        "--tp-degree", str(args.tp_degree),
        "--text-compiled-dir", args.text_compiled_dir,
        "--dit-compiled-dir", args.dit_compiled_dir,
        "--min-mem-gb", str(args.min_mem_gb),
        "--peak-rss-max-gb", str(args.peak_rss_max_gb),
        "--umt5-cosine-min", str(args.umt5_cosine_min),
        "--umt5-mean-abs-max", str(args.umt5_mean_abs_max),
        "--dit-cosine-min", str(args.dit_cosine_min),
        "--dit-mean-abs-max", str(args.dit_mean_abs_max),
    ]
    print(f"[m25c] running stage {stage}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def stage_prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for path in work_dir.glob("*.pt"):
        path.unlink()
    for path in work_dir.glob("*.json"):
        path.unlink()
    model_dir = Path(args.model_dir)
    require_memory(args, "prepare")

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    tokenized = tokenizer(
        [args.prompt],
        padding="max_length",
        truncation=True,
        max_length=args.text_seq_len,
        return_tensors="pt",
    )
    torch.save(
        {
            "input_ids": tokenized["input_ids"].to(torch.int64),
            "attention_mask": tokenized["attention_mask"].to(torch.int32),
        },
        work_dir / "umt5_inputs.pt",
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    lf = latent_frames(args.video_frames)
    latent_h = args.height // 8
    latent_w = args.width // 8
    torch.save(
        {
            "hidden_states": torch.randn(1, 16, lf, latent_h, latent_w, generator=generator, dtype=torch.bfloat16) * 0.1,
            "timestep": torch.tensor([999.0], dtype=torch.bfloat16),
            "encoder_hidden_states": torch.randn(1, args.text_seq_len, 4096, generator=generator, dtype=torch.bfloat16) * 0.1,
        },
        work_dir / "dit_inputs.pt",
    )
    save_metrics(
        args,
        "prepare",
        {
            "prompt": args.prompt,
            "text_seq_len": args.text_seq_len,
            "dit_shape": [1, 16, lf, latent_h, latent_w],
            "height": args.height,
            "width": args.width,
            "video_frames": args.video_frames,
        },
    )


def stage_umt5_cpu(args: argparse.Namespace) -> None:
    from difflet.models.wan.checkpoint import convert_text_encoder_state_dict
    from difflet.models.wan.umt5.modeling_umt5 import WanUmT5Config, WanUmT5EncoderModel

    work_dir = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    require_memory(args, "umt5-cpu")
    inputs = torch.load(work_dir / "umt5_inputs.pt", map_location="cpu")
    cfg = WanUmT5Config.from_diffusers_dict(load_json(model_dir / "text_encoder" / "config.json"))
    start = time.time()
    model = WanUmT5EncoderModel(cfg).to(dtype=torch.bfloat16).eval()
    load_sharded_state_dict(
        model,
        model_dir / "text_encoder",
        rename_fn=convert_text_encoder_state_dict,
        extra_aliases={"encoder.embed_tokens.weight": "shared.weight"},
    )
    load_elapsed = time.time() - start
    with torch.no_grad():
        start = time.time()
        out = model(inputs["input_ids"], inputs["attention_mask"])
        forward_elapsed = time.time() - start
    torch.save(out.detach().cpu(), work_dir / "umt5_cpu.pt")
    save_metrics(args, "umt5-cpu", {"load_elapsed": load_elapsed, "forward_elapsed": forward_elapsed, "shape": list(out.shape)})


def stage_umt5_neff(args: argparse.Namespace) -> None:
    from difflet.backends.trainium.wan.text_encoder import NeuronWanTextEncoderApplication
    from difflet.models.wan.application import create_wan_text_encoder_config

    work_dir = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    compiled_dir = Path(args.text_compiled_dir)
    require_memory(args, "umt5-neff")
    inputs = torch.load(work_dir / "umt5_inputs.pt", map_location="cpu")
    if not (compiled_dir / "model.pt").exists():
        raise FileNotFoundError(f"missing text encoder compiled artifact: {compiled_dir / 'model.pt'}")
    config = create_wan_text_encoder_config(
        model_path=str(model_dir),
        world_size=args.tp_degree,
        tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
        text_seq_len=args.text_seq_len,
        batch_size=1,
    )
    app = NeuronWanTextEncoderApplication(model_path=str(model_dir / "text_encoder"), config=config)
    start = time.time()
    app.load(str(compiled_dir), start_rank_id=0, local_ranks_size=args.tp_degree, skip_warmup=True)
    load_elapsed = time.time() - start
    with torch.no_grad():
        start = time.time()
        out = app(inputs["input_ids"], inputs["attention_mask"])
        forward_elapsed = time.time() - start
    if isinstance(out, (tuple, list)):
        out = out[0]
    torch.save(out.detach().cpu(), work_dir / "umt5_neff.pt")
    save_metrics(args, "umt5-neff", {"load_elapsed": load_elapsed, "forward_elapsed": forward_elapsed, "shape": list(out.shape)})


def stage_dit_cpu(args: argparse.Namespace) -> None:
    from difflet.models.wan.checkpoint import convert_backbone_state_dict
    from difflet.models.wan.modeling_wan import WanTransformerConfig, WanTransformer3DModel

    work_dir = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    require_memory(args, "dit-cpu")
    inputs = torch.load(work_dir / "dit_inputs.pt", map_location="cpu")
    cfg = WanTransformerConfig.from_diffusers_dict(load_json(model_dir / "transformer" / "config.json"))
    start = time.time()
    model = WanTransformer3DModel(cfg).to(dtype=torch.bfloat16).eval()
    load_sharded_state_dict(model, model_dir / "transformer", rename_fn=convert_backbone_state_dict)
    load_elapsed = time.time() - start
    with torch.no_grad():
        start = time.time()
        out = model(inputs["hidden_states"], inputs["timestep"], inputs["encoder_hidden_states"])
        forward_elapsed = time.time() - start
    torch.save(out.detach().cpu(), work_dir / "dit_cpu.pt")
    save_metrics(args, "dit-cpu", {"load_elapsed": load_elapsed, "forward_elapsed": forward_elapsed, "shape": list(out.shape)})


def stage_dit_neff(args: argparse.Namespace) -> None:
    from difflet.backends.trainium.wan.backbone import NeuronWanBackboneApplication
    from difflet.models.wan.application import create_wan_backbone_config

    work_dir = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    compiled_dir = Path(args.dit_compiled_dir)
    require_memory(args, "dit-neff")
    inputs = torch.load(work_dir / "dit_inputs.pt", map_location="cpu")
    if not (compiled_dir / "model.pt").exists():
        raise FileNotFoundError(f"missing DiT compiled artifact: {compiled_dir / 'model.pt'}")
    config = create_wan_backbone_config(
        model_path=str(model_dir),
        world_size=args.tp_degree,
        tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
        num_frames=latent_frames(args.video_frames),
        batch_size=1,
        subfolder="transformer",
    )
    app = NeuronWanBackboneApplication(model_path=str(model_dir / "transformer"), config=config)
    start = time.time()
    app.load(str(compiled_dir), start_rank_id=0, local_ranks_size=args.tp_degree, skip_warmup=True)
    load_elapsed = time.time() - start
    with torch.no_grad():
        start = time.time()
        out = app(inputs["hidden_states"], inputs["timestep"], inputs["encoder_hidden_states"])
        forward_elapsed = time.time() - start
    if isinstance(out, (tuple, list)):
        out = out[0]
    torch.save(out.detach().cpu(), work_dir / "dit_neff.pt")
    save_metrics(args, "dit-neff", {"load_elapsed": load_elapsed, "forward_elapsed": forward_elapsed, "shape": list(out.shape)})


def compare_tensors(name: str, ref: torch.Tensor, neff: torch.Tensor, cosine_min: float, mean_abs_max: float) -> dict:
    if tuple(ref.shape) != tuple(neff.shape):
        raise RuntimeError(f"{name} shape mismatch: ref={tuple(ref.shape)} neff={tuple(neff.shape)}")
    diff = (ref.float() - neff.float()).abs()
    metrics = {
        "shape": list(ref.shape),
        "dtype_ref": str(ref.dtype),
        "dtype_neff": str(neff.dtype),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(torch.sqrt((diff * diff).mean())),
        "cosine": float(torch.nn.functional.cosine_similarity(ref.float().flatten(), neff.float().flatten(), dim=0)),
        "p99": float(diff.quantile(0.99)),
        "p999": float(diff.quantile(0.999)),
    }
    if metrics["cosine"] < cosine_min:
        raise RuntimeError(f"{name} cosine too low: {metrics['cosine']} < {cosine_min}")
    if metrics["mean_abs"] > mean_abs_max:
        raise RuntimeError(f"{name} mean_abs too high: {metrics['mean_abs']} > {mean_abs_max}")
    return metrics


def stage_compare_umt5(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    metrics = compare_tensors(
        "UMT5 NEFF",
        torch.load(work_dir / "umt5_cpu.pt", map_location="cpu"),
        torch.load(work_dir / "umt5_neff.pt", map_location="cpu"),
        args.umt5_cosine_min,
        args.umt5_mean_abs_max,
    )
    metrics["cosine_min"] = args.umt5_cosine_min
    metrics["mean_abs_max"] = args.umt5_mean_abs_max
    save_metrics(args, "compare-umt5", metrics)
    print(f"[m25c:compare-umt5] PASS {json.dumps(metrics, sort_keys=True)}", flush=True)


def stage_compare_dit(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    metrics = compare_tensors(
        "DiT NEFF",
        torch.load(work_dir / "dit_cpu.pt", map_location="cpu"),
        torch.load(work_dir / "dit_neff.pt", map_location="cpu"),
        args.dit_cosine_min,
        args.dit_mean_abs_max,
    )
    metrics["cosine_min"] = args.dit_cosine_min
    metrics["mean_abs_max"] = args.dit_mean_abs_max
    save_metrics(args, "compare-dit", metrics)
    print(f"[m25c:compare-dit] PASS {json.dumps(metrics, sort_keys=True)}", flush=True)


def stage_all(args: argparse.Namespace) -> None:
    run_child(args, "prepare")
    run_child(args, "umt5-cpu", backend="cpu")
    run_child(args, "umt5-neff", backend="trainium", neuron_cores=args.tp_degree)
    run_child(args, "compare-umt5")
    run_child(args, "dit-cpu", backend="cpu")
    run_child(args, "dit-neff", backend="trainium", neuron_cores=args.tp_degree)
    run_child(args, "compare-dit")
    print(f"[m25c] PASS: work_dir={args.work_dir}", flush=True)


def main() -> int:
    args = parse_args()
    stages = {
        "all": stage_all,
        "prepare": stage_prepare,
        "umt5-cpu": stage_umt5_cpu,
        "umt5-neff": stage_umt5_neff,
        "compare-umt5": stage_compare_umt5,
        "dit-cpu": stage_dit_cpu,
        "dit-neff": stage_dit_neff,
        "compare-dit": stage_compare_dit,
    }
    try:
        stage_fn = stages[args.stage]
    except KeyError as exc:
        raise SystemExit(f"unknown stage {args.stage!r}; choices={sorted(stages)}") from exc
    stage_fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
