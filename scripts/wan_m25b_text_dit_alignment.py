#!/usr/bin/env python3
"""M2.5-B memory-bounded Wan UMT5 / DiT numerical alignment.

The top-level `all` stage runs each heavy model in a separate subprocess:

  prepare -> UMT5 HF -> UMT5 Nova CPU -> compare -> DiT HF -> DiT Nova CPU -> compare

That avoids co-resident HF+Nova full-weight models and records per-stage peak RSS.
"""

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
    parser.add_argument("--model-dir", default=os.environ.get("NOVA_WAN_MODEL_DIR", DEFAULT_MODEL_DIR))
    parser.add_argument("--work-dir", default=os.environ.get("NOVA_M25B_WORK_DIR", "/tmp/nova_m25b"))
    parser.add_argument("--prompt", default=os.environ.get("NOVA_M25B_PROMPT", "a cat walking"))
    parser.add_argument("--text-seq-len", type=int, default=int(os.environ.get("NOVA_M25B_TEXT_SEQ_LEN", "512")))
    parser.add_argument("--dit-text-seq-len", type=int, default=int(os.environ.get("NOVA_M25B_DIT_TEXT_SEQ_LEN", "8")))
    parser.add_argument("--dit-latent-t", type=int, default=int(os.environ.get("NOVA_M25B_DIT_LATENT_T", "1")))
    parser.add_argument("--dit-latent-h", type=int, default=int(os.environ.get("NOVA_M25B_DIT_LATENT_H", "8")))
    parser.add_argument("--dit-latent-w", type=int, default=int(os.environ.get("NOVA_M25B_DIT_LATENT_W", "8")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("NOVA_M25B_SEED", "20260510")))
    parser.add_argument("--min-mem-gb", type=float, default=float(os.environ.get("NOVA_M25B_MIN_MEM_GB", "20")))
    parser.add_argument("--umt5-cosine-min", type=float, default=float(os.environ.get("NOVA_M25B_UMT5_COSINE_MIN", "0.995")))
    parser.add_argument("--umt5-mean-abs-max", type=float, default=float(os.environ.get("NOVA_M25B_UMT5_MEAN_ABS_MAX", "0.02")))
    parser.add_argument("--dit-cosine-min", type=float, default=float(os.environ.get("NOVA_M25B_DIT_COSINE_MIN", "0.995")))
    parser.add_argument("--dit-mean-abs-max", type=float, default=float(os.environ.get("NOVA_M25B_DIT_MEAN_ABS_MAX", "0.03")))
    return parser.parse_args()


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return -1.0


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def peak_rss_limit_gb() -> float:
    return float(os.environ.get("NOVA_M25B_PEAK_RSS_MAX_GB", "115"))


def require_memory(min_gb: float, stage: str) -> None:
    available = mem_available_gb()
    print(f"[m25b:{stage}] mem_available_gb = {available:.2f}", flush=True)
    if available >= 0 and available < min_gb:
        raise RuntimeError(
            f"Not enough available memory for {stage}: {available:.2f}GB < {min_gb:.2f}GB"
        )


def save_metrics(work_dir: Path, stage: str, metrics: dict[str, Any]) -> None:
    metrics = dict(metrics)
    metrics["stage"] = stage
    metrics["peak_rss_gb"] = peak_rss_gb()
    metrics["peak_rss_limit_gb"] = peak_rss_limit_gb()
    metrics["mem_available_gb_after"] = mem_available_gb()
    (work_dir / f"{stage}.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"[m25b:{stage}] peak_rss_gb = {metrics['peak_rss_gb']:.2f}", flush=True)
    if metrics["peak_rss_gb"] > metrics["peak_rss_limit_gb"]:
        raise RuntimeError(
            f"{stage} peak RSS too high: "
            f"{metrics['peak_rss_gb']:.2f}GB > {metrics['peak_rss_limit_gb']:.2f}GB"
        )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def iter_safetensor_shards(component_dir: Path) -> list[Path]:
    index_candidates = [
        component_dir / "diffusion_pytorch_model.safetensors.index.json",
        component_dir / "model.safetensors.index.json",
    ]
    for index_path in index_candidates:
        if index_path.exists():
            index = load_json(index_path)
            names = sorted(set(index["weight_map"].values()))
            return [component_dir / name for name in names]

    file_candidates = [
        component_dir / "diffusion_pytorch_model.safetensors",
        component_dir / "model.safetensors",
    ]
    for path in file_candidates:
        if path.exists():
            return [path]

    raise FileNotFoundError(f"no safetensors shard/index found under {component_dir}")


def load_sharded_state_dict(
    model: torch.nn.Module,
    component_dir: Path,
    *,
    rename_fn=None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    expected = set(model.state_dict().keys())
    loaded: set[str] = set()
    unexpected_all: set[str] = set()
    for shard in iter_safetensor_shards(component_dir):
        print(f"[m25b:load] shard = {shard.name}", flush=True)
        state = load_safetensors_file(str(shard), device="cpu")
        if rename_fn is not None:
            state = rename_fn(state)
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


def run_child(args: argparse.Namespace, stage: str, *, backend: str | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    if backend is not None:
        env["NOVA_BACKEND"] = backend
    cmd = [
        sys.executable,
        __file__,
        "--stage",
        stage,
        "--model-dir",
        args.model_dir,
        "--work-dir",
        args.work_dir,
        "--prompt",
        args.prompt,
        "--text-seq-len",
        str(args.text_seq_len),
        "--dit-text-seq-len",
        str(args.dit_text_seq_len),
        "--dit-latent-t",
        str(args.dit_latent_t),
        "--dit-latent-h",
        str(args.dit_latent_h),
        "--dit-latent-w",
        str(args.dit_latent_w),
        "--seed",
        str(args.seed),
        "--min-mem-gb",
        str(args.min_mem_gb),
        "--umt5-cosine-min",
        str(args.umt5_cosine_min),
        "--umt5-mean-abs-max",
        str(args.umt5_mean_abs_max),
        "--dit-cosine-min",
        str(args.dit_cosine_min),
        "--dit-mean-abs-max",
        str(args.dit_mean_abs_max),
    ]
    print(f"[m25b] running stage {stage}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def stage_prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    require_memory(args.min_mem_gb, "prepare")

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
    torch.save(
        {
            "hidden_states": torch.randn(
                1,
                16,
                args.dit_latent_t,
                args.dit_latent_h,
                args.dit_latent_w,
                generator=generator,
                dtype=torch.bfloat16,
            )
            * 0.1,
            "timestep": torch.tensor([999.0], dtype=torch.bfloat16),
            "encoder_hidden_states": torch.randn(
                1,
                args.dit_text_seq_len,
                4096,
                generator=generator,
                dtype=torch.bfloat16,
            )
            * 0.1,
        },
        work_dir / "dit_inputs.pt",
    )
    save_metrics(
        work_dir,
        "prepare",
        {
            "prompt": args.prompt,
            "text_seq_len": args.text_seq_len,
            "dit_shape": [1, 16, args.dit_latent_t, args.dit_latent_h, args.dit_latent_w],
            "dit_text_seq_len": args.dit_text_seq_len,
        },
    )


def stage_umt5_hf(args: argparse.Namespace) -> None:
    from transformers import UMT5EncoderModel

    work_dir = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    require_memory(args.min_mem_gb, "umt5-hf")
    inputs = torch.load(work_dir / "umt5_inputs.pt", map_location="cpu")
    start = time.time()
    model = UMT5EncoderModel.from_pretrained(
        model_dir / "text_encoder",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).eval()
    load_elapsed = time.time() - start
    with torch.no_grad():
        start = time.time()
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=False,
        )[0]
        forward_elapsed = time.time() - start
    torch.save(out.detach().cpu(), work_dir / "umt5_hf.pt")
    save_metrics(
        work_dir,
        "umt5-hf",
        {"load_elapsed": load_elapsed, "forward_elapsed": forward_elapsed, "shape": list(out.shape)},
    )


def stage_umt5_nova(args: argparse.Namespace) -> None:
    from nova.core.modules.checkpoint import load_state_dict
    from nova.models.wan.checkpoint import convert_text_encoder_state_dict
    from nova.models.wan.umt5.modeling_umt5 import WanUmT5Config, WanUmT5EncoderModel

    work_dir = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    require_memory(args.min_mem_gb, "umt5-nova")
    inputs = torch.load(work_dir / "umt5_inputs.pt", map_location="cpu")
    cfg = WanUmT5Config.from_diffusers_dict(load_json(model_dir / "text_encoder" / "config.json"))
    start = time.time()
    model = WanUmT5EncoderModel(cfg).to(dtype=torch.bfloat16).eval()
    raw = load_state_dict(str(model_dir / "text_encoder"))
    converted = convert_text_encoder_state_dict(raw)
    converted = {
        key: value.to(torch.bfloat16) if torch.is_floating_point(value) else value
        for key, value in converted.items()
    }
    if "shared.weight" in converted and "encoder.embed_tokens.weight" not in converted:
        converted["encoder.embed_tokens.weight"] = converted["shared.weight"]
    missing, unexpected = model.load_state_dict(converted, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Nova UMT5 strict load mismatch: missing={missing}, unexpected={unexpected}")
    del raw, converted
    gc.collect()
    load_elapsed = time.time() - start
    with torch.no_grad():
        start = time.time()
        out = model(inputs["input_ids"], inputs["attention_mask"])
        forward_elapsed = time.time() - start
    torch.save(out.detach().cpu(), work_dir / "umt5_nova.pt")
    save_metrics(
        work_dir,
        "umt5-nova",
        {"load_elapsed": load_elapsed, "forward_elapsed": forward_elapsed, "shape": list(out.shape)},
    )


def stage_dit_hf(args: argparse.Namespace) -> None:
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    work_dir = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    require_memory(args.min_mem_gb, "dit-hf")
    inputs = torch.load(work_dir / "dit_inputs.pt", map_location="cpu")
    start = time.time()
    model = WanTransformer3DModel.from_config(str(model_dir / "transformer")).to(dtype=torch.bfloat16).eval()
    load_sharded_state_dict(model, model_dir / "transformer")
    gc.collect()
    load_elapsed = time.time() - start
    with torch.no_grad():
        start = time.time()
        out = model(
            inputs["hidden_states"],
            inputs["timestep"],
            inputs["encoder_hidden_states"],
            return_dict=False,
        )[0]
        forward_elapsed = time.time() - start
    torch.save(out.detach().cpu(), work_dir / "dit_hf.pt")
    save_metrics(
        work_dir,
        "dit-hf",
        {"load_elapsed": load_elapsed, "forward_elapsed": forward_elapsed, "shape": list(out.shape)},
    )


def stage_dit_nova(args: argparse.Namespace) -> None:
    from nova.models.wan.checkpoint import convert_backbone_state_dict
    from nova.models.wan.modeling_wan import WanTransformerConfig, WanTransformer3DModel

    work_dir = Path(args.work_dir)
    model_dir = Path(args.model_dir)
    require_memory(args.min_mem_gb, "dit-nova")
    inputs = torch.load(work_dir / "dit_inputs.pt", map_location="cpu")
    cfg = WanTransformerConfig.from_diffusers_dict(load_json(model_dir / "transformer" / "config.json"))
    start = time.time()
    model = WanTransformer3DModel(cfg).to(dtype=torch.bfloat16).eval()
    load_sharded_state_dict(
        model,
        model_dir / "transformer",
        rename_fn=convert_backbone_state_dict,
    )
    gc.collect()
    load_elapsed = time.time() - start
    with torch.no_grad():
        start = time.time()
        out = model(
            inputs["hidden_states"],
            inputs["timestep"],
            inputs["encoder_hidden_states"],
        )
        forward_elapsed = time.time() - start
    torch.save(out.detach().cpu(), work_dir / "dit_nova.pt")
    save_metrics(
        work_dir,
        "dit-nova",
        {"load_elapsed": load_elapsed, "forward_elapsed": forward_elapsed, "shape": list(out.shape)},
    )


def compare_tensors(name: str, ref: torch.Tensor, nova: torch.Tensor, cosine_min: float, mean_abs_max: float) -> dict:
    if tuple(ref.shape) != tuple(nova.shape):
        raise RuntimeError(f"{name} shape mismatch: ref={tuple(ref.shape)} nova={tuple(nova.shape)}")
    diff = (ref.float() - nova.float()).abs()
    metrics = {
        "shape": list(ref.shape),
        "dtype_ref": str(ref.dtype),
        "dtype_nova": str(nova.dtype),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(torch.sqrt((diff * diff).mean())),
        "cosine": float(torch.nn.functional.cosine_similarity(ref.float().flatten(), nova.float().flatten(), dim=0)),
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
    ref = torch.load(work_dir / "umt5_hf.pt", map_location="cpu")
    nova = torch.load(work_dir / "umt5_nova.pt", map_location="cpu")
    metrics = compare_tensors("UMT5", ref, nova, args.umt5_cosine_min, args.umt5_mean_abs_max)
    metrics["cosine_min"] = args.umt5_cosine_min
    metrics["mean_abs_max"] = args.umt5_mean_abs_max
    save_metrics(work_dir, "compare-umt5", metrics)
    print(f"[m25b:compare-umt5] PASS {json.dumps(metrics, sort_keys=True)}", flush=True)


def stage_compare_dit(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    ref = torch.load(work_dir / "dit_hf.pt", map_location="cpu")
    nova = torch.load(work_dir / "dit_nova.pt", map_location="cpu")
    metrics = compare_tensors("DiT", ref, nova, args.dit_cosine_min, args.dit_mean_abs_max)
    metrics["cosine_min"] = args.dit_cosine_min
    metrics["mean_abs_max"] = args.dit_mean_abs_max
    save_metrics(work_dir, "compare-dit", metrics)
    print(f"[m25b:compare-dit] PASS {json.dumps(metrics, sort_keys=True)}", flush=True)


def stage_all(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    for path in work_dir.glob("*.pt"):
        path.unlink()
    for path in work_dir.glob("*.json"):
        path.unlink()

    run_child(args, "prepare")
    run_child(args, "umt5-hf")
    run_child(args, "umt5-nova", backend="cpu")
    run_child(args, "compare-umt5")
    run_child(args, "dit-hf")
    run_child(args, "dit-nova", backend="cpu")
    run_child(args, "compare-dit")
    print(f"[m25b] PASS: work_dir={work_dir}", flush=True)


def main() -> int:
    args = parse_args()
    stages = {
        "all": stage_all,
        "prepare": stage_prepare,
        "umt5-hf": stage_umt5_hf,
        "umt5-nova": stage_umt5_nova,
        "dit-hf": stage_dit_hf,
        "dit-nova": stage_dit_nova,
        "compare-umt5": stage_compare_umt5,
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
