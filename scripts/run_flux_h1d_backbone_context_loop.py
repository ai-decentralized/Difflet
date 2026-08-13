#!/usr/bin/env python3
"""Run FLUX H1d request-context staging through the real TP4 backbone loop."""

from __future__ import annotations

import argparse
import copy
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
DEFAULT_CONTEXT_ARTIFACT = Path(
    "/home/ubuntu/difflet-artifacts/flux-h1d-request-context-20260812/compiled-v2"
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

from difflet.backends.trainium.flux.request_context_stage import (  # noqa: E402
    NeuronFluxRequestContextStageApplication,
)
from difflet.backends.trainium.flux.resident_cache_step_split import (  # noqa: E402
    NeuronFluxResidentCacheStepSplitApplication,
)
from difflet.models.flux.application import create_flux_config  # noqa: E402
from difflet.models.flux.modeling_flux import (  # noqa: E402
    FluxBackboneInferenceConfig,
    NeuronFluxBackboneApplication,
)


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


def _artifact_summary(path: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "path": str(path),
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "model_pt_sha256": _sha256(path / "model.pt"),
    }


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


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = actual.float() - expected.float()
    return (
        float(difference.abs().max().item()),
        float(
            torch.linalg.vector_norm(difference).item()
            / max(torch.linalg.vector_norm(expected.float()).item(), 1e-30)
        ),
    )


def _read_rank0(ranked_output) -> tuple[torch.Tensor, torch.Tensor | None]:
    rank0 = ranked_output[0]
    selected = rank0[0].cpu()
    checksum = rank0[1].cpu() if len(rank0) > 1 else None
    return selected, checksum


def _context_tensors(config, generator: torch.Generator):
    dtype = torch.bfloat16
    return (
        torch.randn(
            (1, 512, int(config.joint_attention_dim)),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype),
        torch.randn(
            (1, int(config.pooled_projection_dim)),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype),
        torch.tensor([3.5], dtype=dtype),
        torch.randn(
            (4096 + 512, int(config.attention_head_dim), 2),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--backbone-artifact", type=Path, default=DEFAULT_BACKBONE_ARTIFACT
    )
    parser.add_argument("--cache-artifact", type=Path, default=DEFAULT_CACHE_ARTIFACT)
    parser.add_argument(
        "--context-artifact", type=Path, default=DEFAULT_CONTEXT_ARTIFACT
    )
    parser.add_argument("--compile-context", action="store_true")
    parser.add_argument("--profile-boundary", action="store_true")
    parser.add_argument("--result", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for name in (
        "model_path",
        "backbone_artifact",
        "cache_artifact",
        "context_artifact",
        "result",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
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
    compile_seconds = None
    if args.compile_context:
        if args.context_artifact.exists() and any(args.context_artifact.iterdir()):
            raise FileExistsError(
                f"context artifact directory is not empty: {args.context_artifact}"
            )
        args.context_artifact.mkdir(parents=True, exist_ok=True)
        compile_config = copy.deepcopy(config)
        compile_config.neuron_config.skip_sharding = True
        compile_config.neuron_config.save_sharded_checkpoint = False
        compile_app = NeuronFluxRequestContextStageApplication(
            model_path=str(args.model_path / "transformer"), config=compile_config
        )
        started = time.perf_counter()
        compile_app.compile(str(args.context_artifact))
        compile_seconds = time.perf_counter() - started
        del compile_app
    if not (args.context_artifact / "model.pt").is_file():
        raise FileNotFoundError(args.context_artifact / "model.pt")

    backbone = NeuronFluxBackboneApplication(
        model_path=str(args.model_path / "transformer"), config=config
    )
    cache_config = FluxBackboneInferenceConfig.load(str(args.cache_artifact))
    cache = NeuronFluxResidentCacheStepSplitApplication(
        model_path=str(args.model_path / "transformer"), config=cache_config
    )
    context_config = FluxBackboneInferenceConfig.load(str(args.context_artifact))
    context = NeuronFluxRequestContextStageApplication(
        model_path=str(args.model_path / "transformer"), config=context_config
    )

    load_seconds = {}
    for name, app, artifact in (
        ("backbone", backbone, args.backbone_artifact),
        ("cache_step", cache, args.cache_artifact),
        ("request_context", context, args.context_artifact),
    ):
        started = time.perf_counter()
        app.load(str(artifact), skip_warmup=True)
        load_seconds[name] = time.perf_counter() - started

    shape = (1, 4096, 64)
    generator = torch.Generator().manual_seed(20260812)
    initial = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    context_a = _context_tensors(config, generator)
    context_b = _context_tensors(config, generator)
    timesteps, deltas = _schedule(args.model_path, 4096)

    def stage(slot: int, values):
        return context.ranked_forward(slot, *values)

    def timestep(step: int):
        return torch.tensor(
            [float(timesteps[step].item()) / 1000.0], dtype=torch.bfloat16
        )

    def backbone_staged(ranked_latent, ranked_context, step: int):
        ranked_inputs = [
            [
                latent_outputs[0],
                context_outputs[0],
                context_outputs[1],
                timestep(step),
                context_outputs[2],
                context_outputs[3],
            ]
            for latent_outputs, context_outputs in zip(
                ranked_latent, ranked_context, strict=True
            )
        ]
        return backbone.traced_model.nxd_model.forward_ranked(ranked_inputs)

    def backbone_direct(ranked_latent, host_context, step: int):
        encoder, pooled, guidance, rotary = host_context
        ranked_inputs = [
            [latent_outputs[0], encoder, pooled, timestep(step), guidance, rotary]
            for latent_outputs in ranked_latent
        ]
        return backbone.traced_model.nxd_model.forward_ranked(ranked_inputs)

    checks: list[dict[str, Any]] = []
    maximum_absolute_error = 0.0
    maximum_relative_l2_error = 0.0

    def check(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
        nonlocal maximum_absolute_error, maximum_relative_l2_error
        absolute, relative = _errors(actual, expected)
        maximum_absolute_error = max(maximum_absolute_error, absolute)
        maximum_relative_l2_error = max(maximum_relative_l2_error, relative)
        checks.append(
            {
                "name": name,
                "maximum_absolute_error": absolute,
                "relative_l2_error": relative,
            }
        )

    staged_a = stage(0, context_a)
    if not args.profile_boundary:
        for rank, outputs in enumerate(staged_a):
            for field, actual, expected in zip(
                ("encoder", "pooled", "guidance", "rotary"),
                outputs,
                context_a,
                strict=True,
            ):
                check(f"stage_A_rank{rank}_{field}", actual.cpu(), expected)

        probe = cache.ranked_forward(
            initial, torch.tensor([20260812, 91, 0, 0], dtype=torch.int32)
        )
        direct_a, _ = _read_rank0(backbone_direct(probe, context_a, 0))
        staged_a_before, _ = _read_rank0(backbone_staged(probe, staged_a, 0))
        check("backbone_direct_vs_staged_A_before_interleave", staged_a_before, direct_a)

        staged_b = stage(1, context_b)
        direct_b, _ = _read_rank0(backbone_direct(probe, context_b, 0))
        staged_b_noise, _ = _read_rank0(backbone_staged(probe, staged_b, 0))
        check("backbone_direct_vs_staged_B", staged_b_noise, direct_b)
        _ = cache.ranked_forward(torch.tensor([-1.0, 2.0], dtype=torch.float32))
        staged_a_after, _ = _read_rank0(backbone_staged(probe, staged_a, 0))
        check("backbone_A_survives_B_and_cache_execution", staged_a_after, direct_a)

    resident = cache.ranked_forward(
        initial, torch.tensor([20260812, 0, 0, 0], dtype=torch.int32)
    )
    host_latent = initial.clone()
    host_anchors: list[torch.Tensor] = []

    def anchor_step(step: int, ranked_latent):
        nonlocal host_latent
        ranked_noise = backbone_staged(ranked_latent, staged_a, step)
        delta_value = float(deltas[step].item())
        next_resident = cache.ranked_anchor(
            ranked_noise, torch.tensor([delta_value, 0.0], dtype=torch.float32)
        )
        if not args.profile_boundary:
            host_noise, _ = _read_rank0(ranked_noise)
            host_latent = (
                host_latent.float() + delta_value * host_noise.float()
            ).to(torch.bfloat16)
            actual, _ = _read_rank0(next_resident)
            check(f"resident_anchor_{step}", actual, host_latent)
            host_anchors.append(host_noise)
            if len(host_anchors) > 2:
                del host_anchors[0]
        return next_resident

    resident = anchor_step(0, resident)
    resident = anchor_step(1, resident)
    predicted_ranked = cache.ranked_forward(torch.tensor([-1.0, 2.0], dtype=torch.float32))
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
        check("resident_skip_2", actual, host_latent)
    resident = anchor_step(3, resident)

    final_ranked = cache.ranked_forward(torch.tensor([0, 0, 1], dtype=torch.int32))
    final_checksum = None
    if not args.profile_boundary:
        final, checksum = _read_rank0(final_ranked)
        check("resident_final", final, host_latent)
        final_checksum = None if checksum is None else float(checksum.item())

    context_bytes = {
        name: int(value.numel() * value.element_size())
        for name, value in zip(
            ("encoder_hidden_states", "pooled_projections", "guidance", "rotary_embedding"),
            context_a,
            strict=True,
        )
    }
    passed = (
        args.profile_boundary
        or (maximum_absolute_error == 0.0 and maximum_relative_l2_error == 0.0)
    )
    payload = {
        "schema": "difflet-flux-h1d-request-context-staging-result",
        "schema_revision": 1,
        "study_id": "flux-h1d-request-context-staging-20260812",
        "status": (
            "profile_captured"
            if args.profile_boundary
            else "mechanism_passed"
            if passed
            else "mechanism_failed"
        ),
        "mode": "boundary_profile" if args.profile_boundary else "mechanism",
        "serving_claim": False,
        "architecture_speed_claim": False,
        "tp_degree": 4,
        "compile_seconds": compile_seconds,
        "load_seconds": load_seconds,
        "latent_noise_shape": list(shape),
        "latent_noise_tensor_bytes": int(initial.numel() * initial.element_size()),
        "request_context_bytes_per_rank": context_bytes,
        "request_context_total_bytes_tp4": sum(context_bytes.values()) * 4,
        "step_input_bytes_per_rank": {"timestep": 2},
        "request_slot_pool": {
            "proof_capacity": 2,
            "slot_token_bytes_per_rank": 4,
            "allocation_policy": "host selects a free compiled slot; a slot is not reused until request release",
        },
        "artifacts": {
            "backbone": _artifact_summary(args.backbone_artifact),
            "cache_step": _artifact_summary(args.cache_artifact),
            "request_context": _artifact_summary(args.context_artifact),
        },
        "profile_sequence": [
            "request_context_stage",
            "cache_initialize",
            "backbone_anchor_0",
            "cache_anchor_0",
            "backbone_anchor_1",
            "cache_anchor_1",
            "cache_predict",
            "cache_scheduler_skip_2",
            "backbone_anchor_3",
            "cache_anchor_3",
            "cache_finalize",
        ],
        "checks": checks,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_l2_error": maximum_relative_l2_error,
        "final_checksum": final_checksum,
    }
    _write_json(args.result, payload)
    print(
        f"[h1d] {payload['status']} mode={payload['mode']} "
        f"max_abs={maximum_absolute_error:.6g}",
        flush=True,
    )
    print(f"[h1d] result={args.result}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
