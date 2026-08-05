#!/usr/bin/env python3
"""Decode a no-extra-DiT FLUX x0 preview from an existing cache trajectory."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.flux_cache_brake_intervention import _write_json, sha256_file  # noqa: E402

MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
MODEL_ROOT = (
    Path("/home/ubuntu/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev")
    / "snapshots"
    / MODEL_REVISION
)
DECODER_CACHE = Path("/home/ubuntu/.cache/difflet/flux/a9a7be724428f740/decoder")
SOURCE_QUALITY = Path(
    "/home/ubuntu/difflet-artifacts/flux-cache-warmup-vqa-stress-extreme-20260804/quality-input-v2.json"
)
HARDWARE_ACK = "I am decoding offline FLUX x0 previews"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-quality", default=str(SOURCE_QUALITY))
    parser.add_argument("--model-root", default=str(MODEL_ROOT))
    parser.add_argument("--decoder-cache", default=str(DECODER_CACHE))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--candidate-id",
        help="Optionally restrict a multi-candidate quality manifest to one cache arm.",
    )
    parser.add_argument("--decision-step", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--allow-hardware", action="store_true")
    parser.add_argument("--foreground-ack")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int,
    max_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


def _estimate_x0(
    previous,
    current,
    *,
    previous_sigma: float,
    current_sigma: float,
):
    denominator = current_sigma - previous_sigma
    if denominator == 0.0:
        raise ValueError("adjacent scheduler sigmas must differ")
    velocity = (current.float() - previous.float()) / denominator
    return current.float() - current_sigma * velocity


def _unpack_latents(latents, *, height: int, width: int, vae_scale_factor: int):
    batch_size, _, channels = latents.shape
    latent_height = 2 * (height // (vae_scale_factor * 2))
    latent_width = 2 * (width // (vae_scale_factor * 2))
    latents = latents.view(
        batch_size,
        latent_height // 2,
        latent_width // 2,
        channels // 4,
        2,
        2,
    )
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(
        batch_size,
        channels // 4,
        latent_height,
        latent_width,
    )


def _clone_decoder_output(value):
    import torch

    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise TypeError("VAE decoder must return one tensor")
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError("VAE decoder output must be a tensor")
    return value.detach().float().cpu().clone()


def _save_image(value, path: Path) -> None:
    from PIL import Image

    image = (value[0] / 2 + 0.5).clamp(0, 1)
    array = (
        image.permute(1, 2, 0)
        .mul(255)
        .round()
        .to(dtype=__import__("torch").uint8)
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def run(args: argparse.Namespace) -> Path:
    if not args.allow_hardware or args.foreground_ack != HARDWARE_ACK:
        raise RuntimeError(
            f"hardware execution requires --allow-hardware and --foreground-ack {HARDWARE_ACK!r}"
        )
    if args.decision_step < 2 or args.decision_step >= args.num_steps:
        raise ValueError("decision-step must be between 2 and num-steps - 1")

    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler

    from difflet.models.flux.application import create_flux_config
    from difflet.models.flux.vae.modeling_vae import NeuronVAEDecoderApplication

    os.environ.setdefault("DIFFLET_BACKEND", "trainium")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "4")
    source_path = Path(args.source_quality).expanduser().resolve()
    source = _load_json(source_path)
    comparisons = source.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("source quality manifest has no comparisons")
    if args.candidate_id:
        comparisons = [
            comparison
            for comparison in comparisons
            if str(comparison["candidate_id"]) == args.candidate_id
        ]
        if not comparisons:
            raise ValueError(f"source quality manifest has no {args.candidate_id!r} rows")
    sample_ids = [str(comparison["sample_id"]) for comparison in comparisons]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(
            "preview collection requires unique sample_ids; select one candidate-id"
        )

    output_root = Path(args.out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    image_root = output_root / "previews"
    model_root = Path(args.model_root).expanduser().resolve()
    decoder_cache = Path(args.decoder_cache).expanduser().resolve()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_root), subfolder="scheduler"
    )
    image_seq_len = (args.height // 16) * (args.width // 16)
    mu = _calculate_shift(
        image_seq_len,
        int(scheduler.config.base_image_seq_len),
        int(scheduler.config.max_image_seq_len),
        float(scheduler.config.base_shift),
        float(scheduler.config.max_shift),
    )
    scheduler.set_timesteps(
        args.num_steps,
        sigmas=np.linspace(1.0, 1 / args.num_steps, args.num_steps),
        mu=mu,
    )
    sigmas = scheduler.sigmas.detach().float().cpu()

    _, _, _, decoder_config = create_flux_config(
        str(model_root),
        world_size=4,
        backbone_tp_degree=4,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
    )
    decoder = NeuronVAEDecoderApplication(
        model_path=str(decoder_cache), config=decoder_config
    )
    load_started = time.perf_counter()
    decoder.load(str(decoder_cache), skip_warmup=bool(args.skip_warmup))
    load_seconds = time.perf_counter() - load_started

    rows = []
    preview_comparisons = []
    for comparison in comparisons:
        sample_id = str(comparison["sample_id"])
        trajectory_path = _resolve(
            source_path.parent, str(comparison["candidate"]["trajectory"])
        )
        trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=True)
        if trajectory.ndim != 4 or trajectory.shape[0] != args.num_steps:
            raise ValueError(f"unexpected trajectory shape for {sample_id}: {trajectory.shape}")
        previous = trajectory[args.decision_step - 2]
        current = trajectory[args.decision_step - 1]
        x0 = _estimate_x0(
            previous,
            current,
            previous_sigma=float(sigmas[args.decision_step - 1]),
            current_sigma=float(sigmas[args.decision_step]),
        )
        unpacked = _unpack_latents(
            x0,
            height=args.height,
            width=args.width,
            vae_scale_factor=int(decoder_config.vae_scale_factor),
        )
        vae_input = (
            unpacked / float(decoder_config.scaling_factor)
            + float(decoder_config.shift_factor)
        ).to(dtype=torch.bfloat16)
        started = time.perf_counter()
        with torch.no_grad():
            decoded = _clone_decoder_output(decoder(vae_input))
        decode_seconds = time.perf_counter() - started
        image_path = image_root / f"{sample_id}.png"
        _save_image(decoded, image_path)
        rows.append(
            {
                "sample_id": sample_id,
                "trajectory": str(trajectory_path),
                "trajectory_sha256": sha256_file(trajectory_path),
                "preview": str(image_path),
                "preview_sha256": sha256_file(image_path),
                "decode_seconds": decode_seconds,
                "x0_rms": float(x0.square().mean().sqrt()),
                "sigma_previous": float(sigmas[args.decision_step - 1]),
                "sigma_current": float(sigmas[args.decision_step]),
            }
        )
        original_candidate_image = _resolve(
            source_path.parent, str(comparison["candidate"]["image"])
        )
        preview_comparisons.append(
            {
                "sample_id": sample_id,
                "prompt_index": int(comparison["prompt_index"]),
                "prompt": str(comparison["prompt"]),
                "seed": int(comparison["seed"]),
                "candidate_id": (
                    f"x0-preview-step{args.decision_step}-"
                    f"{comparison['candidate_id']}"
                ),
                "baseline": {"image": str(original_candidate_image)},
                "candidate": {"image": str(image_path)},
            }
        )
        print(
            f"[x0-preview] {sample_id} decode={decode_seconds:.3f}s "
            f"x0_rms={rows[-1]['x0_rms']:.4f}",
            flush=True,
        )

    preview_quality_path = output_root / "quality-input.json"
    _write_json(
        preview_quality_path,
        {
            "schema": "difflet-flux-x0-preview-quality-input",
            "schema_revision": 1,
            "protocol": {
                "prompt_selection": {"split": f"x0-preview-step{args.decision_step}"}
            },
            "comparisons": preview_comparisons,
        },
        add_digest=True,
    )
    result_path = output_root / "x0-preview-run.json"
    _write_json(
        result_path,
        {
            "schema": "difflet-flux-x0-preview-run",
            "schema_revision": 1,
            "serving_claim": False,
            "opened_data": True,
            "extra_transformer_calls": 0,
            "decision_step": args.decision_step,
            "num_steps": args.num_steps,
            "source_candidate_id": args.candidate_id,
            "model_revision": MODEL_REVISION,
            "source_quality": {
                "path": str(source_path),
                "file_sha256": sha256_file(source_path),
            },
            "scheduler": {
                "mu": mu,
                "sigma_previous": float(sigmas[args.decision_step - 1]),
                "sigma_current": float(sigmas[args.decision_step]),
                "x0_formula": "current - sigma_current * (current - previous) / (sigma_current - sigma_previous)",
            },
            "decoder": {
                "compiled_path": str(decoder_cache),
                "load_seconds": load_seconds,
                "total_decode_seconds": sum(row["decode_seconds"] for row in rows),
                "online_cost_claim": False,
            },
            "quality_input": {
                "path": str(preview_quality_path),
                "file_sha256": sha256_file(preview_quality_path),
            },
            "rows": rows,
        },
        add_digest=True,
    )
    return result_path


def main() -> int:
    args = build_parser().parse_args()
    result_path = run(args)
    print(f"[x0-preview] run={result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
