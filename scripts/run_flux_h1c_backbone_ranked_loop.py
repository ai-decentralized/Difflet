#!/usr/bin/env python3
"""Run real FLUX backbone <-> resident cache-step ranked-I/O loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(
    "/home/ubuntu/.cache/huggingface/hub/"
    "models--black-forest-labs--FLUX.1-dev/snapshots/"
    "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
)
DEFAULT_BACKBONE_ARTIFACT = Path(
    "/home/ubuntu/.cache/difflet/flux/ebbfe33fed51b364/transformer"
)
DEFAULT_CACHE_ARTIFACT = Path(
    "/home/ubuntu/difflet-artifacts/flux-h1c-resident-cache-step-20260812/"
    "compiled-split-v2"
)


def ensure_runtime_python() -> None:
    try:
        import torch  # noqa: F401
        import torch_neuronx  # noqa: F401
    except (ModuleNotFoundError, OSError):
        if Path(sys.executable) != NEURON_PYTHON and NEURON_PYTHON.exists():
            env = os.environ.copy()
            env["PATH"] = (
                f"{NEURON_VENV / 'bin'}:/opt/aws/neuron/bin:{env.get('PATH', '')}"
            )
            env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
            os.execve(str(NEURON_PYTHON), [str(NEURON_PYTHON), *sys.argv], env)
        raise


ensure_runtime_python()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from diffusers.pipelines.flux.pipeline_flux import (  # noqa: E402
    calculate_shift,
    retrieve_timesteps,
)

from difflet.backends.trainium.flux.resident_cache_step_split import (  # noqa: E402
    NeuronFluxResidentCacheStepSplitApplication,
)
from difflet.models.flux.application import create_flux_config  # noqa: E402
from difflet.models.flux.modeling_flux import NeuronFluxBackboneApplication  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _schedule(model_path: Path, seq_len: int):
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_path), subfolder="scheduler"
    )
    sigmas = np.linspace(1.0, 1.0 / 50, 50)
    mu = calculate_shift(
        seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(
        scheduler, 50, "cpu", sigmas=sigmas, mu=mu
    )
    return timesteps, scheduler.sigmas[1:] - scheduler.sigmas[:-1]


def _read_rank0(ranked_output) -> tuple[torch.Tensor, torch.Tensor | None]:
    rank0 = ranked_output[0]
    selected = rank0[0].cpu()
    checksum = rank0[1].cpu() if len(rank0) > 1 else None
    return selected, checksum


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = actual.float() - expected.float()
    return (
        float(difference.abs().max().item()),
        float(
            torch.linalg.vector_norm(difference).item()
            / max(torch.linalg.vector_norm(expected.float()).item(), 1e-30)
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--backbone-artifact", type=Path, default=DEFAULT_BACKBONE_ARTIFACT
    )
    parser.add_argument("--cache-artifact", type=Path, default=DEFAULT_CACHE_ARTIFACT)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--profile-boundary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    args.backbone_artifact = args.backbone_artifact.expanduser().resolve()
    args.cache_artifact = args.cache_artifact.expanduser().resolve()
    args.result = args.result.expanduser().resolve()
    for artifact in (args.backbone_artifact, args.cache_artifact):
        if not (artifact / "model.pt").is_file():
            raise FileNotFoundError(artifact / "model.pt")

    _, _, config, _ = create_flux_config(
        model_path=str(args.model_path),
        world_size=4,
        backbone_tp_degree=4,
        dtype=torch.bfloat16,
        height=1024,
        width=1024,
    )
    backbone = NeuronFluxBackboneApplication(
        model_path=str(args.model_path / "transformer"), config=config
    )
    cache_config = type(config).load(str(args.cache_artifact))
    cache = NeuronFluxResidentCacheStepSplitApplication(
        model_path=str(args.model_path / "transformer"), config=cache_config
    )

    started = time.perf_counter()
    backbone.load(str(args.backbone_artifact), skip_warmup=True)
    backbone_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    cache.load(str(args.cache_artifact), skip_warmup=True)
    cache_load_seconds = time.perf_counter() - started

    shape = (1, 4096, 64)
    generator = torch.Generator().manual_seed(20260812)
    initial = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    encoder = torch.randn(
        (1, 512, int(config.joint_attention_dim)),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    pooled = torch.randn(
        (1, int(config.pooled_projection_dim)),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    rotary = torch.randn(
        (4096 + 512, int(config.attention_head_dim), 2),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    guidance = torch.tensor([3.5], dtype=torch.bfloat16)
    timesteps, deltas = _schedule(args.model_path, 4096)

    resident = cache.ranked_forward(
        initial, torch.tensor([20260812, 0, 0, 0], dtype=torch.int32)
    )
    host_latent = initial.clone()
    host_anchors: list[torch.Tensor] = []
    checks = []
    maximum_absolute_error = 0.0
    maximum_relative_l2_error = 0.0

    def backbone_ranked(ranked_latent, step: int):
        timestep = torch.tensor(
            [float(timesteps[step].item()) / 1000.0], dtype=torch.bfloat16
        )
        ranked_inputs = [
            [rank_outputs[0], encoder, pooled, timestep, guidance, rotary]
            for rank_outputs in ranked_latent
        ]
        return backbone.traced_model.nxd_model.forward_ranked(ranked_inputs)

    def anchor_step(step: int, ranked_latent):
        nonlocal host_latent, maximum_absolute_error, maximum_relative_l2_error
        ranked_noise = backbone_ranked(ranked_latent, step)
        delta_value = float(deltas[step].item())
        delta_packet = torch.tensor([delta_value, 0.0], dtype=torch.float32)
        next_resident = cache.ranked_anchor(ranked_noise, delta_packet)
        if not args.profile_boundary:
            host_noise, _ = _read_rank0(ranked_noise)
            host_latent = (
                host_latent.float() + delta_value * host_noise.float()
            ).to(torch.bfloat16)
            actual, _ = _read_rank0(next_resident)
            absolute, relative = _errors(actual, host_latent)
            maximum_absolute_error = max(maximum_absolute_error, absolute)
            maximum_relative_l2_error = max(maximum_relative_l2_error, relative)
            checks.append(
                {
                    "step": step,
                    "action": "real_backbone_anchor",
                    "maximum_absolute_error": absolute,
                    "relative_l2_error": relative,
                }
            )
            host_anchors.append(host_noise)
            if len(host_anchors) > 2:
                del host_anchors[0]
        return next_resident

    resident = anchor_step(0, resident)
    resident = anchor_step(1, resident)

    coefficients = torch.tensor([-1.0, 2.0], dtype=torch.float32)
    predicted_ranked = cache.ranked_forward(coefficients)
    resident = cache.ranked_scheduler(
        predicted_ranked, torch.tensor([float(deltas[2].item())], dtype=torch.float32)
    )
    if not args.profile_boundary:
        predicted = (
            host_anchors[0].float() * -1.0 + host_anchors[1].float() * 2.0
        ).to(torch.bfloat16)
        host_latent = (
            host_latent.float() + float(deltas[2].item()) * predicted.float()
        ).to(torch.bfloat16)
        actual, _ = _read_rank0(resident)
        absolute, relative = _errors(actual, host_latent)
        maximum_absolute_error = max(maximum_absolute_error, absolute)
        maximum_relative_l2_error = max(maximum_relative_l2_error, relative)
        checks.append(
            {
                "step": 2,
                "action": "resident_skip",
                "maximum_absolute_error": absolute,
                "relative_l2_error": relative,
            }
        )

    resident = anchor_step(3, resident)
    final, checksum = _read_rank0(
        cache.ranked_forward(torch.tensor([0, 0, 1], dtype=torch.int32))
    )
    final_absolute = None
    final_relative = None
    if not args.profile_boundary:
        final_absolute, final_relative = _errors(final, host_latent)
        maximum_absolute_error = max(maximum_absolute_error, final_absolute)
        maximum_relative_l2_error = max(maximum_relative_l2_error, final_relative)
    passed = (
        args.profile_boundary
        or (maximum_absolute_error == 0.0 and maximum_relative_l2_error == 0.0)
    )
    payload = {
        "schema": "difflet-flux-h1c-backbone-ranked-loop-result",
        "schema_revision": 1,
        "study_id": "flux-h1c-real-backbone-ranked-loop-20260812",
        "status": (
            "profile_captured"
            if args.profile_boundary
            else "mechanism_passed"
            if passed
            else "mechanism_failed"
        ),
        "serving_claim": False,
        "architecture_speed_claim": False,
        "mode": "boundary_profile" if args.profile_boundary else "mechanism",
        "tp_degree": 4,
        "tensor_shape": list(shape),
        "tensor_bytes": int(initial.numel() * initial.element_size()),
        "artifacts": {
            "backbone": {
                "path": str(args.backbone_artifact),
                "model_pt_sha256": _sha256(args.backbone_artifact / "model.pt"),
            },
            "cache_step": {
                "path": str(args.cache_artifact),
                "model_pt_sha256": _sha256(args.cache_artifact / "model.pt"),
            },
        },
        "load_seconds": {
            "backbone": backbone_load_seconds,
            "cache_step": cache_load_seconds,
        },
        "context_input_bytes_per_rank": {
            "encoder_hidden_states": int(encoder.numel() * encoder.element_size()),
            "pooled_projections": int(pooled.numel() * pooled.element_size()),
            "rotary_embedding": int(rotary.numel() * rotary.element_size()),
            "timestep": 2,
            "guidance": 2,
        },
        "sequence": ["anchor_0", "anchor_1", "skip_2", "anchor_3"],
        "checks": checks,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_l2_error": maximum_relative_l2_error,
        "final_maximum_absolute_error": final_absolute,
        "final_relative_l2_error": final_relative,
        "final_checksum": None if checksum is None else float(checksum.item()),
    }
    _write_json(args.result, payload)
    print(
        f"[h1c-backbone] {payload['status']} mode={payload['mode']} "
        f"max_abs={maximum_absolute_error:.6g}",
        flush=True,
    )
    print(f"[h1c-backbone] result={args.result}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
