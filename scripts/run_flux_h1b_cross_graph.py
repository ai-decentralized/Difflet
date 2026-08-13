#!/usr/bin/env python3
"""Compile and run the FLUX H1b multi-NEFF resident-state spike."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
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

import torch  # noqa: E402

from difflet.backends.trainium.flux.resident_cross_graph import (  # noqa: E402
    NeuronFluxResidentCrossGraphApplication,
)
from difflet.models.flux.application import create_flux_config  # noqa: E402


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
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "model_pt_sha256": _sha256(path / "model.pt"),
    }


def _read_rank0(ranked_output) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(ranked_output, list) or not ranked_output:
        raise TypeError(f"unexpected ranked output: {type(ranked_output)!r}")
    rank0 = ranked_output[0]
    if not isinstance(rank0, list) or len(rank0) != 2:
        raise TypeError(f"unexpected per-rank output count: {type(rank0)!r} {len(rank0)}")
    return rank0[0].cpu(), rank0[1].cpu()


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = actual.float() - expected.float()
    absolute = float(difference.abs().max().item())
    relative = float(
        torch.linalg.vector_norm(difference).item()
        / max(torch.linalg.vector_norm(expected.float()).item(), 1e-30)
    )
    return absolute, relative


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--compiled-dir", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile-boundary", action="store_true")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--warmup-pairs", type=int, default=10)
    parser.add_argument("--timed-pairs", type=int, default=100)
    return parser.parse_args()


def _make_app(args):
    _, _, config, _ = create_flux_config(
        model_path=str(args.model_path),
        world_size=args.tp_degree,
        backbone_tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
    )
    config.neuron_config.skip_sharding = True
    config.neuron_config.save_sharded_checkpoint = False
    return config, NeuronFluxResidentCrossGraphApplication(
        model_path=str(args.model_path / "transformer"), config=config
    )


def _seed_state(app, zeros, anchor_a, anchor_b) -> None:
    app.ranked_forward(torch.tensor([0], dtype=torch.int32))
    app.ranked_forward(anchor_a)
    app.ranked_forward(anchor_b)


def _skip_pair(app, coefficients, step_scale):
    predicted = app.ranked_forward(coefficients)
    return app.ranked_consume(predicted, step_scale)


def _profile_boundary(app, shape, generator, args) -> dict[str, Any]:
    anchor_a = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    anchor_b = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    _seed_state(app, zeros, anchor_a, anchor_b)
    coefficients = torch.tensor([-0.25, 1.25], dtype=torch.float32)
    step_scale = torch.ones((1,), dtype=torch.float32)
    warmup_pairs = min(args.warmup_pairs, 2)
    timed_pairs = min(args.timed_pairs, 8)
    for _ in range(warmup_pairs):
        consumed = _skip_pair(app, coefficients, step_scale)
        _ = consumed[0][1].cpu().item()
    timings_ms = []
    checksums = []
    for _ in range(timed_pairs):
        started = time.perf_counter()
        consumed = _skip_pair(app, coefficients, step_scale)
        checksums.append(float(consumed[0][1].cpu().item()))
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    return {
        "mode": "boundary_profile",
        "seed_calls": ["scalar reset", "full anchor A update", "full anchor B update"],
        "warmup_skip_pairs": warmup_pairs,
        "measured_skip_pairs": timed_pairs,
        "skip_pair_entry_points": ["resident_predict", "resident_consume"],
        "large_output_materialized_on_host": False,
        "checksums": checksums,
        "latency": {
            "host_observed_pair_p50_ms": statistics.median(timings_ms),
            "host_observed_pair_p95_ms": float(
                torch.quantile(torch.tensor(timings_ms), 0.95).item()
            ),
        },
    }


def _mechanism(app, shape, generator, args) -> dict[str, Any]:
    anchor_a = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    anchor_b = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    anchor_c = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    _seed_state(app, zeros, anchor_a, anchor_b)

    records = []
    maximum_absolute_error = 0.0
    maximum_relative_l2_error = 0.0

    def check(name, actual, expected):
        nonlocal maximum_absolute_error, maximum_relative_l2_error
        absolute, relative = _errors(actual, expected)
        maximum_absolute_error = max(maximum_absolute_error, absolute)
        maximum_relative_l2_error = max(maximum_relative_l2_error, relative)
        records.append(
            {
                "name": name,
                "maximum_absolute_error": absolute,
                "relative_l2_error": relative,
            }
        )

    coefficients_ab = torch.tensor([-0.5, 1.5], dtype=torch.float32)
    expected_ab = (
        anchor_a.float() * -0.5 + anchor_b.float() * 1.5
    ).to(torch.bfloat16)
    predicted_ab, _ = _read_rank0(app.ranked_forward(coefficients_ab))
    check("cross_neff_predict_ab", predicted_ab, expected_ab)
    repeated_ab, _ = _read_rank0(app.ranked_forward(coefficients_ab))
    check("predict_is_read_only", repeated_ab, expected_ab)

    consumed_ab, checksum = _read_rank0(
        _skip_pair(app, coefficients_ab, torch.ones((1,), dtype=torch.float32))
    )
    check("ranked_predict_to_consumer", consumed_ab, expected_ab)
    checksum_error = abs(
        float(checksum.item()) - float(expected_ab.float().square().mean().item())
    )

    app.ranked_forward(anchor_c)
    coefficients_bc = torch.tensor([-1.0, 2.0], dtype=torch.float32)
    expected_bc = (
        anchor_b.float() * -1.0 + anchor_c.float() * 2.0
    ).to(torch.bfloat16)
    predicted_bc, _ = _read_rank0(app.ranked_forward(coefficients_bc))
    check("cross_neff_ring_shift", predicted_bc, expected_bc)

    app.ranked_forward(torch.tensor([91], dtype=torch.int32))
    after_reset, _ = _read_rank0(app.ranked_forward(coefficients_bc))
    check("scalar_reset_isolation", after_reset, zeros)

    timings_ms = []
    _seed_state(app, zeros, anchor_a, anchor_b)
    warmup_pairs = args.warmup_pairs
    timed_pairs = args.timed_pairs
    for _ in range(warmup_pairs):
        _, scalar = _read_rank0(
            _skip_pair(app, coefficients_ab, torch.ones((1,), dtype=torch.float32))
        )
        _ = scalar.item()
    for _ in range(timed_pairs):
        started = time.perf_counter()
        consumed = _skip_pair(
            app, coefficients_ab, torch.ones((1,), dtype=torch.float32)
        )
        _ = consumed[0][1].cpu().item()
        timings_ms.append((time.perf_counter() - started) * 1000.0)

    # The frozen F1 gate is bit-exactness of the full BF16 tensors and reset
    # isolation.  The FP32 reduction is a diagnostic scalar, not part of that
    # gate; reduction order may differ by a few ulps across TP execution.
    passed = maximum_absolute_error == 0.0 and maximum_relative_l2_error == 0.0
    return {
        "mode": "mechanism",
        "checks": records,
        "checksum_absolute_error": checksum_error,
        "mechanism_gate": {
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_l2_error": maximum_relative_l2_error,
            "passed": passed,
        },
        "latency": {
            "warmup_skip_pairs": warmup_pairs,
            "measured_skip_pairs": timed_pairs,
            "host_observed_pair_p50_ms": statistics.median(timings_ms),
            "host_observed_pair_p95_ms": float(
                torch.quantile(torch.tensor(timings_ms), 0.95).item()
            ),
        },
    }


def main() -> int:
    args = _parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    args.compiled_dir = args.compiled_dir.expanduser().resolve()
    args.result = args.result.expanduser().resolve()
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    config, app = _make_app(args)

    compile_seconds = None
    if args.compile:
        if args.compiled_dir.exists() and any(args.compiled_dir.iterdir()):
            raise FileExistsError(f"compiled directory is not empty: {args.compiled_dir}")
        args.compiled_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        app.compile(str(args.compiled_dir))
        compile_seconds = time.perf_counter() - started
    if not (args.compiled_dir / "model.pt").is_file():
        raise FileNotFoundError(args.compiled_dir / "model.pt")

    started = time.perf_counter()
    app.load(str(args.compiled_dir), skip_warmup=True)
    load_seconds = time.perf_counter() - started
    seq_len = args.height * args.width // (16 * 16)
    shape = (1, seq_len, int(config.in_channels))
    generator = torch.Generator().manual_seed(20260812)
    details = (
        _profile_boundary(app, shape, generator, args)
        if args.profile_boundary
        else _mechanism(app, shape, generator, args)
    )
    payload = {
        "schema": "difflet-flux-h1b-cross-graph-result",
        "schema_revision": 1,
        "study_id": "flux-h1b-cross-graph-state-ranked-io-20260812",
        "status": (
            "profile_captured"
            if args.profile_boundary
            else "mechanism_passed"
            if details["mechanism_gate"]["passed"]
            else "mechanism_failed"
        ),
        "serving_claim": False,
        "architecture_speed_claim": False,
        "shape": list(shape),
        "tensor_dtype": "bfloat16",
        "tensor_bytes": int(torch.zeros(shape, dtype=torch.bfloat16).numel() * 2),
        "tp_degree": args.tp_degree,
        "entry_points": [
            "resident_anchor_update",
            "resident_predict",
            "resident_consume",
            "resident_reset",
        ],
        "compile_seconds": compile_seconds,
        "load_seconds": load_seconds,
        "compiled_artifact": _artifact_summary(args.compiled_dir),
        "weight_policy": {
            "has_model_weights": False,
            "skip_sharding": True,
            "save_sharded_checkpoint": False,
        },
        "details": details,
    }
    _write_json(args.result, payload)
    print(f"[h1b] {payload['status']} mode={details['mode']}", flush=True)
    print(f"[h1b] result={args.result}", flush=True)
    return 0 if payload["status"] != "mechanism_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
