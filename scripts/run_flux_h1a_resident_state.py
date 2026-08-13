#!/usr/bin/env python3
"""Compile and run the FLUX H1a K=2 resident-state boundary spike."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
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

from difflet.backends.trainium.flux.resident_cache_state import (  # noqa: E402
    ANCHOR_ACTION,
    PREDICT_ACTION,
    RESET_ACTION,
    NeuronFluxResidentCacheStateApplication,
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


def _neuron_monitor_sample() -> dict[str, Any]:
    try:
        process = subprocess.Popen(
            ["neuron-monitor"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout is not None
        line = process.stdout.readline().strip()
        process.terminate()
        process.wait(timeout=3.0)
        return json.loads(line) if line else {"error": "empty neuron-monitor sample"}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def _artifact_summary(path: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "model_pt_sha256": _sha256(path / "model.pt")
        if (path / "model.pt").exists()
        else None,
    }


def _unwrap(value: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        raise TypeError(f"resident graph returned unexpected value: {type(value)!r}")
    selected, checksum = value[0], value[1]
    if not torch.is_tensor(selected) or not torch.is_tensor(checksum):
        raise TypeError("resident graph outputs are not tensors")
    return selected.detach().cpu(), checksum.detach().cpu()


def _reference(anchor0, anchor1, candidate, coefficients, action):
    if action == RESET_ACTION:
        zeros = torch.zeros_like(anchor0)
        return zeros, zeros, zeros
    if action == ANCHOR_ACTION:
        return candidate, anchor1, candidate
    predicted = (
        anchor0.float() * float(coefficients[0])
        + anchor1.float() * float(coefficients[1])
    ).to(torch.bfloat16)
    return predicted, anchor0, anchor1


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
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--warmup-calls", type=int, default=10)
    parser.add_argument("--timed-calls", type=int, default=100)
    parser.add_argument("--profile-sequence", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.warmup_calls < 0 or args.timed_calls < 1:
        raise SystemExit("warmup calls must be nonnegative and timed calls must be positive")
    model_path = args.model_path.expanduser().resolve()
    compiled_dir = args.compiled_dir.expanduser().resolve()
    result_path = args.result.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    _, _, config, _ = create_flux_config(
        model_path=str(model_path),
        world_size=args.tp_degree,
        backbone_tp_degree=args.tp_degree,
        dtype=torch.bfloat16,
        height=args.height,
        width=args.width,
    )
    # This state-only graph has no FLUX weights.  Without these overrides the
    # generic artifact layer may match the FLUX transformer identity and
    # hardlink its 22.7 GB shared checkpoint into this tiny experiment.
    config.neuron_config.skip_sharding = True
    config.neuron_config.save_sharded_checkpoint = False
    app = NeuronFluxResidentCacheStateApplication(
        model_path=str(model_path / "transformer"),
        config=config,
    )

    compile_seconds = None
    if args.compile:
        if compiled_dir.exists() and any(compiled_dir.iterdir()):
            raise FileExistsError(f"compiled directory is not empty: {compiled_dir}")
        compiled_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        app.compile(str(compiled_dir))
        compile_seconds = time.perf_counter() - started
    if not (compiled_dir / "model.pt").is_file():
        raise FileNotFoundError(f"compiled model is missing: {compiled_dir / 'model.pt'}")

    memory_before_load = _neuron_monitor_sample()
    started = time.perf_counter()
    app.load(str(compiled_dir), skip_warmup=True)
    load_seconds = time.perf_counter() - started
    memory_after_load = _neuron_monitor_sample()

    seq_len = args.height * args.width // (16 * 16)
    shape = (1, seq_len, int(config.in_channels))
    generator = torch.Generator().manual_seed(20260812)
    anchor_a = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    anchor_b = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    anchor_c = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    zeros = torch.zeros(shape, dtype=torch.bfloat16)

    calls = [
        ("reset", zeros, (0.0, 0.0), RESET_ACTION),
        ("anchor_a", anchor_a, (0.0, 0.0), ANCHOR_ACTION),
        ("anchor_b", anchor_b, (0.0, 0.0), ANCHOR_ACTION),
        ("predict_ab_1", zeros, (-0.5, 1.5), PREDICT_ACTION),
        ("predict_ab_2", zeros, (-0.5, 1.5), PREDICT_ACTION),
        ("anchor_c", anchor_c, (0.0, 0.0), ANCHOR_ACTION),
        ("predict_bc", zeros, (-1.0, 2.0), PREDICT_ACTION),
        ("reset_again", zeros, (0.0, 0.0), RESET_ACTION),
        ("predict_after_reset", zeros, (-1.0, 2.0), PREDICT_ACTION),
    ]
    reference0 = zeros.clone()
    reference1 = zeros.clone()
    sequence_results = []
    maximum_absolute_error = 0.0
    maximum_relative_l2_error = 0.0
    for name, candidate, coefficients, action in calls:
        expected, reference0, reference1 = _reference(
            reference0, reference1, candidate, coefficients, action
        )
        started = time.perf_counter()
        selected, checksum = _unwrap(
            app(
                candidate,
                torch.tensor(coefficients, dtype=torch.float32),
                torch.tensor([action], dtype=torch.int32),
            )
        )
        elapsed = time.perf_counter() - started
        absolute, relative = _errors(selected, expected)
        maximum_absolute_error = max(maximum_absolute_error, absolute)
        maximum_relative_l2_error = max(maximum_relative_l2_error, relative)
        sequence_results.append(
            {
                "name": name,
                "action": action,
                "latency_ms": elapsed * 1000.0,
                "maximum_absolute_error": absolute,
                "relative_l2_error": relative,
                "checksum": float(checksum.reshape(-1)[0].item()),
            }
        )

    timings_ms = []
    timed_count = min(args.timed_calls, 8) if args.profile_sequence else args.timed_calls
    warmup_count = min(args.warmup_calls, 2) if args.profile_sequence else args.warmup_calls
    coefficient_tensor = torch.tensor([-0.25, 1.25], dtype=torch.float32)
    action_tensor = torch.tensor([PREDICT_ACTION], dtype=torch.int32)
    for _ in range(warmup_count):
        selected, _ = _unwrap(app(zeros, coefficient_tensor, action_tensor))
        _ = selected.reshape(-1)[0].item()
    for _ in range(timed_count):
        started = time.perf_counter()
        selected, _ = _unwrap(app(zeros, coefficient_tensor, action_tensor))
        _ = selected.reshape(-1)[0].item()
        timings_ms.append((time.perf_counter() - started) * 1000.0)

    host_anchor0 = anchor_b
    host_anchor1 = anchor_c
    host_timings_ms = []
    for _ in range(timed_count):
        started = time.perf_counter()
        host_output = (
            host_anchor0.float() * -0.25 + host_anchor1.float() * 1.25
        ).to(torch.bfloat16)
        _ = host_output.reshape(-1)[0].item()
        host_timings_ms.append((time.perf_counter() - started) * 1000.0)

    mechanism_passed = (
        maximum_absolute_error == 0.0 and maximum_relative_l2_error == 0.0
    )
    memory_after_execution = _neuron_monitor_sample()
    result = {
        "schema": "difflet-flux-h1a-resident-state-result",
        "schema_revision": 1,
        "study_id": "flux-h1a-post-gather-resident-state-boundary-20260812",
        "status": "mechanism_passed" if mechanism_passed else "mechanism_failed",
        "serving_claim": False,
        "architecture_speed_claim": False,
        "shape": list(shape),
        "tensor_dtype": "bfloat16",
        "tensor_bytes": int(zeros.numel() * zeros.element_size()),
        "tp_degree": args.tp_degree,
        "weight_policy": {
            "has_model_weights": False,
            "skip_sharding": bool(config.neuron_config.skip_sharding),
            "save_sharded_checkpoint": bool(
                config.neuron_config.save_sharded_checkpoint
            ),
        },
        "compile_seconds": compile_seconds,
        "load_seconds": load_seconds,
        "compiled_artifact": _artifact_summary(compiled_dir),
        "neuron_monitor_before_load": memory_before_load,
        "neuron_monitor_after_load": memory_after_load,
        "neuron_monitor_after_execution": memory_after_execution,
        "sequence": sequence_results,
        "mechanism_gate": {
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_l2_error": maximum_relative_l2_error,
            "passed": mechanism_passed,
        },
        "latency": {
            "warmup_calls": warmup_count,
            "measured_calls": timed_count,
            "device_host_observed_p50_ms": statistics.median(timings_ms),
            "device_host_observed_p95_ms": float(torch.quantile(torch.tensor(timings_ms), 0.95)),
            "host_reference_p50_ms": statistics.median(host_timings_ms),
            "host_reference_p95_ms": float(
                torch.quantile(torch.tensor(host_timings_ms), 0.95)
            ),
        },
        "boundary_gate": {
            "evaluated": False,
            "reason": "requires a separate neuron-profile inspect trace",
            "large_tensor_input_in_static_signature": True,
            "large_tensor_output_in_static_signature": True,
        },
    }
    _write_json(result_path, result)
    print(
        f"[h1a] {result['status']} max_abs={maximum_absolute_error:.6g} "
        f"rel_l2={maximum_relative_l2_error:.6g} "
        f"device_p50={result['latency']['device_host_observed_p50_ms']:.3f}ms",
        flush=True,
    )
    print(f"[h1a] result={result_path}", flush=True)
    return 0 if mechanism_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
