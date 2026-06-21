#!/usr/bin/env python3
"""F3.0 denoise-loop profiler for on-chip scheduler work.

This script measures the host-side scheduler boundary before F3 moves scheduler
math into the traced Trainium graph. For cached-boundary models it replays the
latent denoise loop from a safetensors bundle; for Flux it can wrap the full
Difflet pipeline call and time transformer / scheduler / ``xm.mark_step`` phases.

Examples:
    python scripts/profile_denoise_loop.py --model qwen-image \
      --model-id /models/Qwen-Image --bundle /tmp/qwen_inputs.safetensors

    python scripts/profile_denoise_loop.py --model ltx-2 \
      --model-id /models/LTX-2 --bundle /tmp/ltx2_inputs.safetensors \
      --application-kwarg transformer_mode=segmented

    NEURON_RT_NUM_CORES=8 python scripts/profile_denoise_loop.py --model flux \
      --model-id black-forest-labs/FLUX.1-dev --num-inference-steps 28
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable

NEURON_VENV = Path("/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference")
NEURON_PYTHON = NEURON_VENV / "bin" / "python"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "cclogs" / "m7-onchip-scheduler-fold"
NEFF_GATE_TIMING_SOURCES = (
    "xla_metric:ExecuteReplicatedTime",
    "xla_metric:ExecuteTime",
    "neuron_profile:nrt_execute_max_worker_duration",
)
XLA_TO_DEVICE_TIME_METRICS = ("TransferToServerTime", "TransferToDeviceTime")
XLA_FROM_DEVICE_TIME_METRICS = ("TransferFromServerTime", "TransferFromDeviceTime")
XLA_TO_DEVICE_BYTE_METRICS = ("TransferToServerData", "TransferToDeviceData", "InboundData")
XLA_FROM_DEVICE_BYTE_METRICS = (
    "TransferFromServerData",
    "TransferFromDeviceData",
    "OutboundData",
)
XLA_EXECUTE_TIME_METRICS = ("ExecuteReplicatedTime", "ExecuteTime")
XLA_ATEN_TRANSFER_COUNTERS = (
    "aten::_to_copy",
    "aten::copy_",
    "aten::to",
)
XLA_DIRECT_METRIC_CANDIDATES = tuple(
    dict.fromkeys(
        (
            *XLA_TO_DEVICE_TIME_METRICS,
            *XLA_FROM_DEVICE_TIME_METRICS,
            *XLA_TO_DEVICE_BYTE_METRICS,
            *XLA_FROM_DEVICE_BYTE_METRICS,
            *XLA_EXECUTE_TIME_METRICS,
        )
    )
)


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
from safetensors.torch import load_file as load_safetensors_file  # noqa: E402


MODEL_ALIASES = {
    "flux": "flux",
    "hunyuan-video": "hunyuan-video",
    "hunyuan_video": "hunyuan-video",
    "hv": "hunyuan-video",
    "hunyuan-video15": "hunyuan-video15",
    "hunyuan_video15": "hunyuan-video15",
    "hv15": "hunyuan-video15",
    "ltx-2": "ltx-2",
    "ltx_2": "ltx-2",
    "qwen-image": "qwen-image",
    "qwen_image": "qwen-image",
}

MODEL_TYPES = {
    "flux": "flux",
    "hunyuan-video": "hunyuan_video",
    "hunyuan-video15": "hunyuan_video_15",
    "ltx-2": "ltx_2",
    "qwen-image": "qwen_image",
}

DEFAULT_MODEL_IDS = {
    "flux": "black-forest-labs/FLUX.1-dev",
    "hunyuan-video": "hunyuanvideo-community/HunyuanVideo",
    "hunyuan-video15": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    "ltx-2": "Lightricks/LTX-2",
    "qwen-image": "Qwen/Qwen-Image",
}


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")


def _parse_kwarg(value: str) -> tuple[str, Any]:
    key, sep, raw = value.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError("--application-kwarg entries must be key=value")
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return key, lowered == "true"
    if lowered in {"none", "null"}:
        return key, None
    try:
        return key, int(raw)
    except ValueError:
        pass
    try:
        return key, float(raw)
    except ValueError:
        return key, raw


def _application_kwargs(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        key, item = _parse_kwarg(value)
        parsed[key] = item
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle_meta(bundle_path: Path | None) -> dict[str, Any]:
    if bundle_path is None:
        return {}
    return _load_json(Path(str(bundle_path) + ".meta.json"))


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in ("sample", "images", "frames", "latents"):
            item = value.get(key)
            if torch.is_tensor(item):
                return item
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    if hasattr(value, "sample") and torch.is_tensor(value.sample):
        return value.sample
    raise TypeError(f"could not extract tensor from {type(value)!r}")


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


@dataclass(frozen=True)
class _XlaMetricSnapshot:
    available: bool
    data: dict[str, float | int]
    overhead_s: float
    error: str | None = None
    metric_names: tuple[str, ...] = ()
    counter_names: tuple[str, ...] = ()
    report_excerpt: str | None = None


class _XlaMetricsSampler:
    """Read torch_xla cumulative metrics and return per-step deltas.

    `metrics_report()` is parsed first so XLA's formatter handles units. Direct
    `metric_data()` and `counter_value()` samples are added as a fallback for
    runtimes where interesting metrics exist but are not emitted in the report.
    """

    def __init__(self) -> None:
        self._met = None
        self._parse_metrics_report = None
        self.available = False
        self.error: str | None = None
        try:
            import torch_xla.debug.metrics as met
            from torch_xla.debug.metrics_compare_utils import parse_metrics_report

            self._met = met
            self._parse_metrics_report = parse_metrics_report
            self.available = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def snapshot(self) -> _XlaMetricSnapshot:
        if not self.available or self._met is None or self._parse_metrics_report is None:
            return _XlaMetricSnapshot(False, {}, 0.0, self.error)
        start = time.perf_counter()
        try:
            metric_names = tuple(sorted(self._met.metric_names()))
            counter_names = tuple(sorted(self._met.counter_names()))
            report = self._met.metrics_report()
            data = self._parse_metrics_report(report, dehumanize=True)
            data.update(self._direct_metric_snapshot(metric_names, counter_names))
        except Exception as exc:
            return _XlaMetricSnapshot(
                False,
                {},
                time.perf_counter() - start,
                f"{type(exc).__name__}: {exc}",
            )
        return _XlaMetricSnapshot(
            True,
            data,
            time.perf_counter() - start,
            metric_names=metric_names,
            counter_names=counter_names,
            report_excerpt=report[:4000] if not data else None,
        )

    def _direct_metric_snapshot(
        self,
        metric_names: tuple[str, ...],
        counter_names: tuple[str, ...],
    ) -> dict[str, float | int]:
        if self._met is None:
            return {}
        data: dict[str, float | int] = {}
        candidates = tuple(dict.fromkeys((*XLA_DIRECT_METRIC_CANDIDATES, *metric_names)))
        for name in candidates:
            try:
                value = self._met.metric_data(name)
            except Exception:
                continue
            if value is None:
                continue
            parsed = self._parse_metric_data_tuple(name, value)
            for key, item in parsed.items():
                data.setdefault(key, item)

        counters = tuple(dict.fromkeys((*XLA_ATEN_TRANSFER_COUNTERS, *counter_names)))
        for name in counters:
            try:
                value = self._met.counter_value(name)
            except Exception:
                continue
            if value is None:
                continue
            data.setdefault(f"{name}__Value", int(value))
        return data

    @staticmethod
    def _parse_metric_data_tuple(name: str, value: Any) -> dict[str, float | int]:
        if not isinstance(value, tuple) or len(value) < 2:
            return {}
        total_samples, accumulator = value[0], value[1]
        data: dict[str, float | int] = {
            f"{name}__TotalSamples": int(total_samples),
            f"{name}__RawAccumulator": float(accumulator),
        }
        if name in XLA_TO_DEVICE_TIME_METRICS + XLA_FROM_DEVICE_TIME_METRICS:
            data[f"{name}__Accumulator_sec"] = float(accumulator)
        elif name in XLA_EXECUTE_TIME_METRICS:
            data[f"{name}__Accumulator_sec"] = float(accumulator)
        elif name in XLA_TO_DEVICE_BYTE_METRICS + XLA_FROM_DEVICE_BYTE_METRICS:
            data[f"{name}__Accumulator_bytes"] = int(round(float(accumulator)))
        return data

    @staticmethod
    def _delta(
        before: _XlaMetricSnapshot,
        after: _XlaMetricSnapshot,
        key: str,
    ) -> float | int | None:
        if not before.available or not after.available:
            return None
        if key not in before.data and key not in after.data:
            return None
        return after.data.get(key, 0) - before.data.get(key, 0)

    def _delta_metric(
        self,
        before: _XlaMetricSnapshot,
        after: _XlaMetricSnapshot,
        names: tuple[str, ...],
        suffix: str,
    ) -> tuple[str | None, float | int | None]:
        for name in names:
            value = self._delta(before, after, f"{name}__{suffix}")
            if value is not None:
                return name, value
        return None, None

    def _delta_metric_sum(
        self,
        before: _XlaMetricSnapshot,
        after: _XlaMetricSnapshot,
        names: tuple[str, ...],
        suffixes: tuple[str, ...],
    ) -> tuple[list[str], float | int | None]:
        observed: list[str] = []
        total: float | int = 0
        for name in names:
            for suffix in suffixes:
                value = self._delta(before, after, f"{name}__{suffix}")
                if value is not None:
                    observed.append(f"{name}__{suffix}")
                    total += value
                    break
        return observed, total if observed else None

    def delta(
        self,
        before: _XlaMetricSnapshot,
        after: _XlaMetricSnapshot,
    ) -> dict[str, Any]:
        if not before.available or not after.available:
            return {
                "xla_metrics_available": False,
                "xla_metrics_error": after.error or before.error or self.error,
                "xla_metrics_snapshot_overhead_s": before.overhead_s + after.overhead_s,
                "xla_metric_names": [],
                "xla_metric_names_available": list(after.metric_names),
                "xla_counter_names_available": list(after.counter_names),
            }

        to_time_name, to_time_s = self._delta_metric(
            before,
            after,
            XLA_TO_DEVICE_TIME_METRICS,
            "Accumulator_sec",
        )
        from_time_name, from_time_s = self._delta_metric(
            before,
            after,
            XLA_FROM_DEVICE_TIME_METRICS,
            "Accumulator_sec",
        )
        _, to_count = self._delta_metric(
            before,
            after,
            XLA_TO_DEVICE_TIME_METRICS,
            "TotalSamples",
        )
        _, from_count = self._delta_metric(
            before,
            after,
            XLA_FROM_DEVICE_TIME_METRICS,
            "TotalSamples",
        )
        to_byte_names, to_bytes_mb = self._delta_metric_sum(
            before,
            after,
            XLA_TO_DEVICE_BYTE_METRICS,
            ("Accumulator_mb", "Value_mb", "Accumulator_bytes", "Value"),
        )
        from_byte_names, from_bytes_mb = self._delta_metric_sum(
            before,
            after,
            XLA_FROM_DEVICE_BYTE_METRICS,
            ("Accumulator_mb", "Value_mb", "Accumulator_bytes", "Value"),
        )
        execute_name, execute_time_s = self._delta_metric(
            before,
            after,
            XLA_EXECUTE_TIME_METRICS,
            "Accumulator_sec",
        )
        _, execute_count = self._delta_metric(
            before,
            after,
            XLA_EXECUTE_TIME_METRICS,
            "TotalSamples",
        )

        transfer_time_s = None
        if to_time_s is not None or from_time_s is not None:
            transfer_time_s = float(to_time_s or 0.0) + float(from_time_s or 0.0)
        transfer_count = None
        if to_count is not None or from_count is not None:
            transfer_count = int(to_count or 0) + int(from_count or 0)
        transfer_bytes = None
        transfer_bytes = self._transfer_bytes_from_delta(
            to_byte_names,
            to_bytes_mb,
            from_byte_names,
            from_bytes_mb,
        )

        metric_names = [
            name
            for name in [
                to_time_name,
                from_time_name,
                execute_name,
                *to_byte_names,
                *from_byte_names,
            ]
            if name
        ]
        return {
            "xla_metrics_available": True,
            "xla_metrics_error": None,
            "xla_metrics_snapshot_overhead_s": before.overhead_s + after.overhead_s,
            "xla_metric_names": metric_names,
            "xla_metric_names_available": list(after.metric_names),
            "xla_counter_names_available": list(after.counter_names),
            "xla_parsed_metric_keys": sorted(after.data),
            "xla_metrics_report_excerpt": after.report_excerpt,
            "to_device_transfer_time_s": float(to_time_s) if to_time_s is not None else None,
            "from_device_transfer_time_s": (
                float(from_time_s) if from_time_s is not None else None
            ),
            "host_device_transfer_time_s": transfer_time_s,
            "to_device_transfer_count": int(to_count) if to_count is not None else None,
            "from_device_transfer_count": int(from_count) if from_count is not None else None,
            "host_device_transfer_count": transfer_count,
            "to_device_transfer_bytes": (
                self._byte_delta_from_metric_names(to_byte_names, to_bytes_mb)
            ),
            "from_device_transfer_bytes": (
                self._byte_delta_from_metric_names(from_byte_names, from_bytes_mb)
            ),
            "host_device_transfer_bytes": transfer_bytes,
            "neff_execution_time_s": (
                float(execute_time_s) if execute_time_s is not None else None
            ),
            "xla_aten_transfer_counter_delta": self._aten_counter_delta(before, after),
            "neff_execution_count": int(execute_count) if execute_count is not None else None,
            "neff_execution_time_source": (
                f"xla_metric:{execute_name}" if execute_name else None
            ),
        }

    @staticmethod
    def _byte_delta_from_metric_names(
        metric_names: list[str],
        value: float | int | None,
    ) -> int | None:
        if value is None:
            return None
        if any(name.endswith(("_mb", "__Value_mb")) for name in metric_names):
            return int(round(float(value) * 1e6))
        return int(round(float(value)))

    def _transfer_bytes_from_delta(
        self,
        to_names: list[str],
        to_value: float | int | None,
        from_names: list[str],
        from_value: float | int | None,
    ) -> int | None:
        to_bytes = self._byte_delta_from_metric_names(to_names, to_value)
        from_bytes = self._byte_delta_from_metric_names(from_names, from_value)
        if to_bytes is None and from_bytes is None:
            return None
        return int(to_bytes or 0) + int(from_bytes or 0)

    @staticmethod
    def _aten_counter_delta(
        before: _XlaMetricSnapshot,
        after: _XlaMetricSnapshot,
    ) -> int | None:
        total = 0
        observed = False
        for name in XLA_ATEN_TRANSFER_COUNTERS:
            value = _XlaMetricsSampler._delta(before, after, f"{name}__Value")
            if value is not None:
                observed = True
                total += int(value)
        return total if observed else None


def _require_tensors(tensors: dict[str, torch.Tensor], required: set[str], path: Path) -> None:
    missing = sorted(required.difference(tensors))
    if missing:
        raise KeyError(f"{path} is missing required tensors: {', '.join(missing)}")


def _meta_int(
    args: argparse.Namespace,
    meta: dict[str, Any],
    name: str,
    default: int | None = None,
) -> int:
    value = getattr(args, name)
    if value is not None:
        return int(value)
    if name in meta and meta[name] is not None:
        return int(meta[name])
    if default is not None:
        return int(default)
    raise ValueError(f"--{name.replace('_', '-')} is required when bundle meta is missing")


def _meta_float(
    args: argparse.Namespace,
    meta: dict[str, Any],
    name: str,
    default: float,
) -> float:
    value = getattr(args, name)
    if value is not None:
        return float(value)
    if name in meta and meta[name] is not None:
        return float(meta[name])
    return float(default)


def _xla_mark_step() -> float:
    try:
        import torch_xla.core.xla_model as xm
    except Exception:
        return 0.0
    start = time.perf_counter()
    xm.mark_step()
    return time.perf_counter() - start


def _summarize_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        return {
            "num_steps": 0,
            "mean_host_step_time_s": 0.0,
            "mean_device_step_time_s": 0.0,
            "mean_neff_execution_time_s": None,
            "mean_cpu_round_trip_share": 0.0,
            "max_cpu_round_trip_share": 0.0,
            "clears_10pct_gate": False,
        }
    host = [float(step["host_step_time_s"]) for step in steps]
    device = [float(step["device_step_time_s"]) for step in steps]
    neff = [
        float(step["neff_execution_time_s"])
        for step in steps
        if step.get("neff_execution_time_s") is not None
    ]
    shares = [
        float(step["cpu_round_trip_share"])
        for step in steps
        if step.get("cpu_round_trip_share") is not None
    ]
    scheduler = [float(step.get("scheduler_time_s", 0.0)) for step in steps]
    mark = [float(step.get("mark_step_time_s", 0.0)) for step in steps]
    transfer_time = [
        float(step["host_device_transfer_time_s"])
        for step in steps
        if step.get("host_device_transfer_time_s") is not None
    ]
    transfer_count = [
        int(step["host_device_transfer_count"])
        for step in steps
        if step.get("host_device_transfer_count") is not None
    ]
    transfer_bytes = [
        int(step["host_device_transfer_bytes"])
        for step in steps
        if step.get("host_device_transfer_bytes") is not None
    ]
    xla_steps = [step for step in steps if step.get("xla_metrics_available")]
    hardware_timed_steps = [
        step
        for step in steps
        if str(step.get("neff_execution_time_source") or "").startswith("xla_metric:")
    ]
    return {
        "num_steps": len(steps),
        "mean_host_step_time_s": mean(host),
        "median_host_step_time_s": median(host),
        "pstdev_host_step_time_s": pstdev(host) if len(host) > 1 else 0.0,
        "mean_device_step_time_s": mean(device),
        "mean_neff_execution_time_s": mean(neff) if neff else None,
        "mean_scheduler_time_s": mean(scheduler),
        "mean_mark_step_time_s": mean(mark),
        "mean_host_device_transfer_time_s": mean(transfer_time) if transfer_time else None,
        "total_host_device_transfer_count": sum(transfer_count) if transfer_count else None,
        "total_host_device_transfer_bytes": sum(transfer_bytes) if transfer_bytes else None,
        "xla_metrics_steps": len(xla_steps),
        "hardware_timed_steps": len(hardware_timed_steps),
        "mean_cpu_round_trip_share": mean(shares) if shares else None,
        "max_cpu_round_trip_share": max(shares) if shares else None,
        "clears_10pct_gate": bool(shares and max(shares) >= 0.10),
    }


def _step_record(
    *,
    step_index: int,
    timestep: torch.Tensor | Any,
    host_step_time_s: float,
    device_step_time_s: float,
    scheduler_time_s: float,
    mark_step_time_s: float,
    scheduler_tensor_bytes: int,
    xla_metrics_delta: dict[str, Any] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    if torch.is_tensor(timestep):
        timestep_value = float(timestep.detach().float().reshape(-1)[0].cpu())
    else:
        timestep_value = float(timestep)
    xla_metrics_delta = xla_metrics_delta or {}
    neff_execution_time_s = xla_metrics_delta.get("neff_execution_time_s")
    neff_execution_time_source = xla_metrics_delta.get("neff_execution_time_source")
    if neff_execution_time_s is None:
        neff_execution_time_s = device_step_time_s
        neff_execution_time_source = "python_transformer_call_boundary_fallback"
    cpu_share = None
    if host_step_time_s > 0 and neff_execution_time_s is not None:
        cpu_share = max(host_step_time_s - float(neff_execution_time_s), 0.0) / host_step_time_s
    record = {
        "step_index": int(step_index),
        "timestep": timestep_value,
        "host_step_time_s": host_step_time_s,
        "device_step_time_s": device_step_time_s,
        "neff_execution_time_s": neff_execution_time_s,
        "neff_execution_time_source": neff_execution_time_source,
        "neff_execution_count": xla_metrics_delta.get("neff_execution_count"),
        "scheduler_time_s": scheduler_time_s,
        "mark_step_time_s": mark_step_time_s,
        "scheduler_tensor_bytes": int(scheduler_tensor_bytes),
        "to_device_transfer_time_s": xla_metrics_delta.get("to_device_transfer_time_s"),
        "from_device_transfer_time_s": xla_metrics_delta.get("from_device_transfer_time_s"),
        "host_device_transfer_time_s": xla_metrics_delta.get("host_device_transfer_time_s"),
        "to_device_transfer_count": xla_metrics_delta.get("to_device_transfer_count"),
        "from_device_transfer_count": xla_metrics_delta.get("from_device_transfer_count"),
        "host_device_transfer_count": xla_metrics_delta.get("host_device_transfer_count"),
        "to_device_transfer_bytes": xla_metrics_delta.get("to_device_transfer_bytes"),
        "from_device_transfer_bytes": xla_metrics_delta.get("from_device_transfer_bytes"),
        "host_device_transfer_bytes": xla_metrics_delta.get("host_device_transfer_bytes"),
        "xla_aten_transfer_counter_delta": xla_metrics_delta.get(
            "xla_aten_transfer_counter_delta"
        ),
        "xla_metrics_available": bool(xla_metrics_delta.get("xla_metrics_available", False)),
        "xla_metrics_error": xla_metrics_delta.get("xla_metrics_error"),
        "xla_metrics_snapshot_overhead_s": xla_metrics_delta.get(
            "xla_metrics_snapshot_overhead_s"
        ),
        "xla_metric_names": xla_metrics_delta.get("xla_metric_names", []),
        "xla_metric_names_available": xla_metrics_delta.get(
            "xla_metric_names_available", []
        ),
        "xla_counter_names_available": xla_metrics_delta.get(
            "xla_counter_names_available", []
        ),
        "xla_parsed_metric_keys": xla_metrics_delta.get("xla_parsed_metric_keys", []),
        "xla_metrics_report_excerpt": xla_metrics_delta.get("xla_metrics_report_excerpt"),
        "cpu_round_trip_share": cpu_share,
    }
    if note is not None:
        record["note"] = note
    return record


def _load_difflet_pipeline(
    args: argparse.Namespace,
    *,
    model: str,
    model_id: str,
    height: int | None,
    width: int | None,
    num_frames: int | None,
    application_kwargs: dict[str, Any],
):
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")
    from difflet import DiffletParallelConfig, DiffletPipeline

    start = time.perf_counter()
    pipe = DiffletPipeline.from_pretrained(
        model_id,
        model_type=MODEL_TYPES[model],
        parallel=DiffletParallelConfig(tp_degree=args.tp_degree),
        dtype=args.dtype,
        height=height,
        width=width,
        num_frames=num_frames,
        compile_cache_dir=args.cache_dir,
        revision=args.revision,
        local_files_only=args.local_files_only,
        force_compile=args.force_compile,
        skip_compile=args.skip_compile,
        load=True,
        skip_warmup=args.skip_warmup,
        application_kwargs=application_kwargs,
    )
    return pipe, time.perf_counter() - start


def _load_direct_app(
    args: argparse.Namespace,
    *,
    model: str,
    source_dir: str,
    compiled_dir: str,
    height: int | None,
    width: int | None,
    num_frames: int | None,
    application_kwargs: dict[str, Any],
):
    os.environ.setdefault("DIFFLET_BACKEND", "trainium")
    from difflet import DiffletParallelConfig

    parallel = DiffletParallelConfig(tp_degree=args.tp_degree)
    shape = {"height": height, "width": width, "num_frames": num_frames}
    start = time.perf_counter()
    if model == "hunyuan-video":
        from difflet.models.hunyuan_video.application import NeuronHunyuanVideoApplication

        app = NeuronHunyuanVideoApplication(
            model_path=source_dir,
            parallel=parallel,
            dtype=args.dtype,
            shape=shape,
            **application_kwargs,
        )
    elif model == "hunyuan-video15":
        from difflet.models.hunyuan_video.application import NeuronHunyuanVideoApplication

        app = NeuronHunyuanVideoApplication(
            model_path=source_dir,
            parallel=parallel,
            dtype=args.dtype,
            shape=shape,
            model_version="1.5",
            **application_kwargs,
        )
    elif model == "qwen-image":
        from difflet.models.qwen_image.application import NeuronQwenImageApplication

        app = NeuronQwenImageApplication(
            model_path=source_dir,
            parallel=parallel,
            dtype=args.dtype,
            shape=shape,
            **application_kwargs,
        )
    elif model == "ltx-2":
        from difflet.models.ltx_2.application import NeuronLTX2Application

        app = NeuronLTX2Application(
            model_path=source_dir,
            parallel=parallel,
            dtype=args.dtype,
            shape=shape,
            **application_kwargs,
        )
    else:
        raise ValueError(f"direct app loading is not supported for {model}")
    app.load(compiled_dir, skip_warmup=args.skip_warmup)
    load_elapsed = time.perf_counter() - start
    return app, load_elapsed


def _load_app_or_pipeline(
    args: argparse.Namespace,
    *,
    model: str,
    model_id: str,
    height: int | None,
    width: int | None,
    num_frames: int | None,
    application_kwargs: dict[str, Any],
):
    if args.source_dir or args.compiled_dir:
        if not args.source_dir or not args.compiled_dir:
            raise ValueError("--source-dir and --compiled-dir must be passed together")
        app, load_elapsed = _load_direct_app(
            args,
            model=model,
            source_dir=args.source_dir,
            compiled_dir=args.compiled_dir,
            height=height,
            width=width,
            num_frames=num_frames,
            application_kwargs=application_kwargs,
        )
        return app, load_elapsed, args.compiled_dir
    pipe, load_elapsed = _load_difflet_pipeline(
        args,
        model=model,
        model_id=model_id,
        height=height,
        width=width,
        num_frames=num_frames,
        application_kwargs=application_kwargs,
    )
    return pipe, load_elapsed, str(pipe.compiled_path)


def _app_from_target(target: Any) -> Any:
    return getattr(target, "app", target)


def _call_target(target: Any, *args: Any, **kwargs: Any) -> Any:
    return target(*args, **kwargs)


def _run_qwen_image(args: argparse.Namespace, meta: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    from difflet.models.qwen_image.application import QwenImageDiTInputBundle

    bundle_path = Path(args.bundle)
    tensors = load_safetensors_file(str(bundle_path), device="cpu")
    _require_tensors(
        tensors,
        {
            "latents_init",
            "timesteps",
            "encoder_hidden_states",
            "encoder_hidden_states_mask",
            "guidance",
        },
        bundle_path,
    )
    height = _meta_int(args, meta, "height")
    width = _meta_int(args, meta, "width")
    text_seq_len = _meta_int(args, meta, "text_seq_len")
    app_kwargs = {"text_seq_len": text_seq_len, **_application_kwargs(args.application_kwarg)}
    target, load_elapsed, compiled_path = _load_app_or_pipeline(
        args,
        model="qwen-image",
        model_id=args.model_id,
        height=height,
        width=width,
        num_frames=None,
        application_kwargs=app_kwargs,
    )
    app = _app_from_target(target)
    latents = tensors["latents_init"].to(dtype=args.dtype).contiguous()
    timesteps = tensors["timesteps"].contiguous()
    scheduler_loop = app.pipeline
    steps: list[dict[str, Any]] = []
    xla_metrics = _XlaMetricsSampler()

    for step_index, timestep in enumerate(timesteps):
        timestep_batch = (
            timestep.to(dtype=args.dtype).reshape(1).expand(latents.shape[0]).contiguous()
        )
        bundle = QwenImageDiTInputBundle(
            hidden_states=latents.to(dtype=args.dtype).contiguous(),
            timestep=timestep_batch,
            encoder_hidden_states=tensors["encoder_hidden_states"]
            .to(dtype=args.dtype)
            .contiguous(),
            encoder_hidden_states_mask=tensors["encoder_hidden_states_mask"]
            .to(torch.bool)
            .contiguous(),
            guidance=tensors["guidance"].to(dtype=args.dtype).contiguous(),
        )
        xla_before = xla_metrics.snapshot()
        host_start = time.perf_counter()
        device_start = time.perf_counter()
        noise_pred = _first_tensor(_call_target(target, bundle))
        device_elapsed = time.perf_counter() - device_start
        scheduler_start = time.perf_counter()
        next_latents = scheduler_loop._scheduler_step(  # noqa: SLF001
            noise_pred,
            timestep,
            latents,
            int(timesteps.shape[0]),
        )
        scheduler_elapsed = time.perf_counter() - scheduler_start
        mark_elapsed = _xla_mark_step() if args.mark_step else 0.0
        host_elapsed = time.perf_counter() - host_start
        xla_after = xla_metrics.snapshot()
        scheduler_bytes = _tensor_bytes((noise_pred, timestep, latents, next_latents))
        latents = next_latents
        steps.append(
            _step_record(
                step_index=step_index,
                timestep=timestep,
                host_step_time_s=host_elapsed,
                device_step_time_s=device_elapsed,
                scheduler_time_s=scheduler_elapsed,
                mark_step_time_s=mark_elapsed,
                scheduler_tensor_bytes=scheduler_bytes,
                xla_metrics_delta=xla_metrics.delta(xla_before, xla_after),
            )
        )

    return {
        "load_compile_elapsed_s": load_elapsed,
        "compiled_path": compiled_path,
        "height": height,
        "width": width,
        "num_frames": None,
        "text_seq_len": text_seq_len,
        "steps": steps,
        "final_latents_shape": list(latents.shape),
    }, target


def _run_hunyuan_video(
    args: argparse.Namespace,
    meta: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    from difflet.models.hunyuan_video.application import HunyuanVideoDiTInputBundle

    bundle_path = Path(args.bundle)
    tensors = load_safetensors_file(str(bundle_path), device="cpu")
    _require_tensors(
        tensors,
        {
            "latents_init",
            "timesteps",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "pooled_projections",
            "guidance",
        },
        bundle_path,
    )
    height = _meta_int(args, meta, "height")
    width = _meta_int(args, meta, "width")
    num_frames = _meta_int(args, meta, "num_frames")
    text_seq_len = int(tensors["encoder_hidden_states"].shape[1])
    app_kwargs = {"text_seq_len": text_seq_len, **_application_kwargs(args.application_kwarg)}
    target, load_elapsed, compiled_path = _load_app_or_pipeline(
        args,
        model="hunyuan-video",
        model_id=args.model_id,
        height=height,
        width=width,
        num_frames=num_frames,
        application_kwargs=app_kwargs,
    )
    app = _app_from_target(target)
    latents = tensors["latents_init"].to(dtype=args.dtype).contiguous()
    timesteps = tensors["timesteps"].contiguous()
    scheduler_loop = app.pipeline
    steps: list[dict[str, Any]] = []
    xla_metrics = _XlaMetricsSampler()

    for step_index, timestep in enumerate(timesteps):
        timestep_batch = (
            timestep.to(dtype=args.dtype).reshape(1).expand(latents.shape[0]).contiguous()
        )
        bundle = HunyuanVideoDiTInputBundle(
            hidden_states=latents.to(dtype=args.dtype).contiguous(),
            timestep=timestep_batch,
            encoder_hidden_states=tensors["encoder_hidden_states"]
            .to(dtype=args.dtype)
            .contiguous(),
            encoder_attention_mask=tensors["encoder_attention_mask"].to(torch.int64).contiguous(),
            pooled_projections=tensors["pooled_projections"].to(dtype=args.dtype).contiguous(),
            guidance=tensors["guidance"].to(dtype=args.dtype).contiguous(),
        )
        xla_before = xla_metrics.snapshot()
        host_start = time.perf_counter()
        device_start = time.perf_counter()
        noise_pred = _first_tensor(_call_target(target, bundle))
        device_elapsed = time.perf_counter() - device_start
        scheduler_start = time.perf_counter()
        next_latents = scheduler_loop._scheduler_step(  # noqa: SLF001
            noise_pred,
            timestep,
            latents,
            int(timesteps.shape[0]),
        )
        scheduler_elapsed = time.perf_counter() - scheduler_start
        mark_elapsed = _xla_mark_step() if args.mark_step else 0.0
        host_elapsed = time.perf_counter() - host_start
        xla_after = xla_metrics.snapshot()
        scheduler_bytes = _tensor_bytes((noise_pred, timestep, latents, next_latents))
        latents = next_latents
        steps.append(
            _step_record(
                step_index=step_index,
                timestep=timestep,
                host_step_time_s=host_elapsed,
                device_step_time_s=device_elapsed,
                scheduler_time_s=scheduler_elapsed,
                mark_step_time_s=mark_elapsed,
                scheduler_tensor_bytes=scheduler_bytes,
                xla_metrics_delta=xla_metrics.delta(xla_before, xla_after),
            )
        )

    return {
        "load_compile_elapsed_s": load_elapsed,
        "compiled_path": compiled_path,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "text_seq_len": text_seq_len,
        "steps": steps,
        "final_latents_shape": list(latents.shape),
    }, target


def _run_hunyuan_video15(
    args: argparse.Namespace,
    meta: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    from difflet.models.hunyuan_video.application import HunyuanVideo15DiTInputBundle

    bundle_path = Path(args.bundle)
    tensors = load_safetensors_file(str(bundle_path), device="cpu")
    _require_tensors(
        tensors,
        {
            "hidden_states",
            "timesteps",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "encoder_hidden_states_2",
            "encoder_attention_mask_2",
            "image_embeds",
        },
        bundle_path,
    )
    height = _meta_int(args, meta, "height")
    width = _meta_int(args, meta, "width")
    num_frames = _meta_int(args, meta, "num_frames")
    text_seq_len = int(tensors["encoder_hidden_states"].shape[1])
    text_seq_len_2 = int(tensors["encoder_hidden_states_2"].shape[1])
    image_seq_len = int(tensors["image_embeds"].shape[1])
    app_kwargs = {
        "text_seq_len": text_seq_len,
        "text_seq_len_2": text_seq_len_2,
        "image_seq_len": image_seq_len,
        **_application_kwargs(args.application_kwarg),
    }
    target, load_elapsed, compiled_path = _load_app_or_pipeline(
        args,
        model="hunyuan-video15",
        model_id=args.model_id,
        height=height,
        width=width,
        num_frames=num_frames,
        application_kwargs=app_kwargs,
    )
    app = _app_from_target(target)
    latents = tensors["hidden_states"].to(dtype=args.dtype).contiguous()
    timesteps = tensors["timesteps"].contiguous()
    timestep_r = tensors.get("timestep_r", torch.ones([latents.shape[0]], dtype=args.dtype))
    scheduler_loop = app.pipeline
    steps: list[dict[str, Any]] = []
    xla_metrics = _XlaMetricsSampler()

    for step_index, timestep in enumerate(timesteps):
        timestep_batch = (
            timestep.to(dtype=args.dtype).reshape(1).expand(latents.shape[0]).contiguous()
        )
        bundle = HunyuanVideo15DiTInputBundle(
            hidden_states=latents.to(dtype=args.dtype).contiguous(),
            timestep=timestep_batch,
            encoder_hidden_states=tensors["encoder_hidden_states"]
            .to(dtype=args.dtype)
            .contiguous(),
            encoder_attention_mask=tensors["encoder_attention_mask"].to(torch.int64).contiguous(),
            timestep_r=timestep_r.to(dtype=args.dtype).contiguous(),
            encoder_hidden_states_2=tensors["encoder_hidden_states_2"]
            .to(dtype=args.dtype)
            .contiguous(),
            encoder_attention_mask_2=tensors["encoder_attention_mask_2"]
            .to(torch.int64)
            .contiguous(),
            image_embeds=tensors["image_embeds"].to(dtype=args.dtype).contiguous(),
        )
        xla_before = xla_metrics.snapshot()
        host_start = time.perf_counter()
        device_start = time.perf_counter()
        noise_pred = _first_tensor(_call_target(target, bundle))
        device_elapsed = time.perf_counter() - device_start
        scheduler_start = time.perf_counter()
        next_latents = scheduler_loop._scheduler_step(  # noqa: SLF001
            noise_pred,
            timestep,
            latents,
            int(timesteps.shape[0]),
        )
        scheduler_elapsed = time.perf_counter() - scheduler_start
        mark_elapsed = _xla_mark_step() if args.mark_step else 0.0
        host_elapsed = time.perf_counter() - host_start
        xla_after = xla_metrics.snapshot()
        scheduler_bytes = _tensor_bytes((noise_pred, timestep, latents, next_latents))
        latents = next_latents
        steps.append(
            _step_record(
                step_index=step_index,
                timestep=timestep,
                host_step_time_s=host_elapsed,
                device_step_time_s=device_elapsed,
                scheduler_time_s=scheduler_elapsed,
                mark_step_time_s=mark_elapsed,
                scheduler_tensor_bytes=scheduler_bytes,
                xla_metrics_delta=xla_metrics.delta(xla_before, xla_after),
                note=(
                    "cached HunyuanVideo-1.5 boundary loop updates the transformer "
                    "hidden_states tensor used by the compiled graph"
                ),
            )
        )

    return {
        "load_compile_elapsed_s": load_elapsed,
        "compiled_path": compiled_path,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "text_seq_len": text_seq_len,
        "text_seq_len_2": text_seq_len_2,
        "image_seq_len": image_seq_len,
        "steps": steps,
        "final_latents_shape": list(latents.shape),
    }, target


def _run_ltx_2(args: argparse.Namespace, meta: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    from difflet.models.ltx_2.application import LTX2DiTInputBundle

    bundle_path = Path(args.bundle)
    tensors = load_safetensors_file(str(bundle_path), device="cpu")
    _require_tensors(
        tensors,
        {
            "latents_init",
            "audio_latents_init",
            "timesteps",
            "encoder_hidden_states",
            "audio_encoder_hidden_states",
            "encoder_attention_mask",
            "audio_encoder_attention_mask",
            "video_coords",
            "audio_coords",
        },
        bundle_path,
    )
    height = _meta_int(args, meta, "height")
    width = _meta_int(args, meta, "width")
    num_frames = _meta_int(args, meta, "num_frames")
    text_seq_len = int(tensors["encoder_hidden_states"].shape[1])
    audio_text_seq_len = int(tensors["audio_encoder_hidden_states"].shape[1])
    audio_num_frames = int(meta.get("audio_num_frames") or tensors["audio_latents_init"].shape[1])
    frame_rate = _meta_float(args, meta, "frame_rate", 24.0)
    app_kwargs = {
        "text_seq_len": text_seq_len,
        "audio_text_seq_len": audio_text_seq_len,
        "audio_num_frames": audio_num_frames,
        "frame_rate": frame_rate,
        **_application_kwargs(args.application_kwarg),
    }
    target, load_elapsed, compiled_path = _load_app_or_pipeline(
        args,
        model="ltx-2",
        model_id=args.model_id,
        height=height,
        width=width,
        num_frames=num_frames,
        application_kwargs=app_kwargs,
    )
    app = _app_from_target(target)
    latents = tensors["latents_init"].to(dtype=args.dtype).contiguous()
    audio_latents = tensors["audio_latents_init"].to(dtype=args.dtype).contiguous()
    timesteps = tensors["timesteps"].contiguous()
    scheduler_loop = app.pipeline
    steps: list[dict[str, Any]] = []
    xla_metrics = _XlaMetricsSampler()

    for step_index, timestep in enumerate(timesteps):
        timestep_batch = (
            timestep.to(dtype=args.dtype).reshape(1).expand(latents.shape[0]).contiguous()
        )
        bundle = LTX2DiTInputBundle(
            hidden_states=latents.to(dtype=args.dtype).contiguous(),
            audio_hidden_states=audio_latents.to(dtype=args.dtype).contiguous(),
            encoder_hidden_states=tensors["encoder_hidden_states"]
            .to(dtype=args.dtype)
            .contiguous(),
            audio_encoder_hidden_states=tensors["audio_encoder_hidden_states"]
            .to(dtype=args.dtype)
            .contiguous(),
            timestep=timestep_batch,
            sigma=timestep_batch,
            encoder_attention_mask=tensors["encoder_attention_mask"].to(torch.bool).contiguous(),
            audio_encoder_attention_mask=tensors["audio_encoder_attention_mask"]
            .to(torch.bool)
            .contiguous(),
            video_coords=tensors["video_coords"].to(torch.float32).contiguous(),
            audio_coords=tensors["audio_coords"].to(torch.float32).contiguous(),
        )
        xla_before = xla_metrics.snapshot()
        host_start = time.perf_counter()
        device_start = time.perf_counter()
        output = _call_target(target, bundle)
        device_elapsed = time.perf_counter() - device_start
        if isinstance(output, (tuple, list)) and len(output) >= 2:
            noise_pred_video = _first_tensor(output[0])
            noise_pred_audio = _first_tensor(output[1])
        else:
            raise TypeError("LTX-2 transformer output must contain video and audio tensors")
        scheduler_start = time.perf_counter()
        next_latents = scheduler_loop._scheduler_step(  # noqa: SLF001
            scheduler_loop.scheduler,
            noise_pred_video,
            timestep,
            latents,
            int(timesteps.shape[0]),
        )
        audio_scheduler = (
            getattr(scheduler_loop, "audio_scheduler", None) or scheduler_loop.scheduler
        )
        next_audio_latents = scheduler_loop._scheduler_step(  # noqa: SLF001
            audio_scheduler,
            noise_pred_audio,
            timestep,
            audio_latents,
            int(timesteps.shape[0]),
        )
        scheduler_elapsed = time.perf_counter() - scheduler_start
        mark_elapsed = _xla_mark_step() if args.mark_step else 0.0
        host_elapsed = time.perf_counter() - host_start
        xla_after = xla_metrics.snapshot()
        scheduler_bytes = _tensor_bytes(
            (
                noise_pred_video,
                noise_pred_audio,
                timestep,
                latents,
                audio_latents,
                next_latents,
                next_audio_latents,
            )
        )
        latents = next_latents
        audio_latents = next_audio_latents
        steps.append(
            _step_record(
                step_index=step_index,
                timestep=timestep,
                host_step_time_s=host_elapsed,
                device_step_time_s=device_elapsed,
                scheduler_time_s=scheduler_elapsed,
                mark_step_time_s=mark_elapsed,
                scheduler_tensor_bytes=scheduler_bytes,
                xla_metrics_delta=xla_metrics.delta(xla_before, xla_after),
            )
        )

    return {
        "load_compile_elapsed_s": load_elapsed,
        "compiled_path": compiled_path,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "text_seq_len": text_seq_len,
        "audio_text_seq_len": audio_text_seq_len,
        "audio_num_frames": audio_num_frames,
        "frame_rate": frame_rate,
        "steps": steps,
        "final_latents_shape": list(latents.shape),
        "final_audio_latents_shape": list(audio_latents.shape),
    }, target


def _wrap_method(
    obj: Any,
    name: str,
    wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> Callable[[], None]:
    original = getattr(obj, name)
    setattr(obj, name, wrapper_factory(original))

    def restore() -> None:
        setattr(obj, name, original)

    return restore


def _run_flux(args: argparse.Namespace, meta: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    del meta
    height = args.height or 1024
    width = args.width or 1024
    app_kwargs = _application_kwargs(args.application_kwarg)
    pipe, load_elapsed = _load_difflet_pipeline(
        args,
        model="flux",
        model_id=args.model_id,
        height=height,
        width=width,
        num_frames=None,
        application_kwargs=app_kwargs,
    )
    flux_pipe = pipe.app.pipe
    steps: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    step_index = -1
    xla_metrics = _XlaMetricsSampler()

    def finalize_active_step(note: str | None = None) -> None:
        nonlocal active
        if active is None:
            return
        host_elapsed = time.perf_counter() - active["host_start"]
        xla_after = xla_metrics.snapshot()
        steps.append(
            _step_record(
                step_index=active["step_index"],
                timestep=active["timestep"],
                host_step_time_s=host_elapsed,
                device_step_time_s=active["device_step_time_s"],
                scheduler_time_s=active["scheduler_time_s"],
                mark_step_time_s=active["mark_step_time_s"],
                scheduler_tensor_bytes=active["scheduler_tensor_bytes"],
                xla_metrics_delta=xla_metrics.delta(active["xla_metrics_before"], xla_after),
                note=note,
            )
        )
        active = None

    def wrap_transformer(original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*call_args: Any, **call_kwargs: Any) -> Any:
            nonlocal active
            start = time.perf_counter()
            result = original(*call_args, **call_kwargs)
            elapsed = time.perf_counter() - start
            if active is not None:
                active["device_step_time_s"] += elapsed
            return result

        return wrapped

    def wrap_scheduler_step(original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*call_args: Any, **call_kwargs: Any) -> Any:
            nonlocal active, step_index
            finalize_active_step(note="finalized at next scheduler.step")
            if active is None:
                step_index += 1
                timestep = call_args[1] if len(call_args) > 1 else call_kwargs.get("timestep", 0.0)
                active = {
                    "step_index": step_index,
                    "timestep": timestep,
                    "xla_metrics_before": xla_metrics.snapshot(),
                    "host_start": time.perf_counter(),
                    "device_step_time_s": 0.0,
                    "scheduler_time_s": 0.0,
                    "mark_step_time_s": 0.0,
                    "scheduler_tensor_bytes": _tensor_bytes((call_args, call_kwargs)),
                }
            start = time.perf_counter()
            result = original(*call_args, **call_kwargs)
            active["scheduler_time_s"] += time.perf_counter() - start
            active["scheduler_tensor_bytes"] += _tensor_bytes(result)
            return result

        return wrapped

    restore_transformer = _wrap_method(flux_pipe.transformer, "forward", wrap_transformer)
    restore_scheduler = _wrap_method(flux_pipe.scheduler, "step", wrap_scheduler_step)

    restore_mark_step: Callable[[], None] | None = None
    try:
        import torch_xla.core.xla_model as xm

        if args.mark_step:
            original_mark_step = xm.mark_step

            def marked_step(*call_args: Any, **call_kwargs: Any) -> Any:
                nonlocal active
                start = time.perf_counter()
                result = original_mark_step(*call_args, **call_kwargs)
                elapsed = time.perf_counter() - start
                if active is not None:
                    active["mark_step_time_s"] += elapsed
                    finalize_active_step()
                return result

            xm.mark_step = marked_step

            def restore_mark() -> None:
                xm.mark_step = original_mark_step

            restore_mark_step = restore_mark
    except Exception:
        restore_mark_step = None

    try:
        call_start = time.perf_counter()
        with torch.no_grad():
            result = pipe(
                prompt=args.prompt,
                height=height,
                width=width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                output_type=args.output_type,
            )
        call_elapsed = time.perf_counter() - call_start
        finalize_active_step(note="finalized without observing xm.mark_step")
    finally:
        restore_transformer()
        restore_scheduler()
        if restore_mark_step is not None:
            restore_mark_step()

    return {
        "load_compile_elapsed_s": load_elapsed,
        "compiled_path": str(pipe.compiled_path),
        "height": height,
        "width": width,
        "num_frames": None,
        "text_seq_len": None,
        "call_elapsed_s": call_elapsed,
        "steps": steps,
        "output_type": args.output_type,
        "result_type": type(result).__name__,
    }, pipe


RUNNERS: dict[str, Callable[[argparse.Namespace, dict[str, Any]], tuple[dict[str, Any], Any]]] = {
    "flux": _run_flux,
    "hunyuan-video": _run_hunyuan_video,
    "hunyuan-video15": _run_hunyuan_video15,
    "ltx-2": _run_ltx_2,
    "qwen-image": _run_qwen_image,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_ALIASES),
        help="Model loop to profile.",
    )
    parser.add_argument("--model-id", default=None, help="HF model id or local model directory.")
    parser.add_argument(
        "--bundle",
        default=None,
        help="Cached DiT-boundary safetensors bundle. Required except for --model flux.",
    )
    parser.add_argument("--cache-dir", default=".difflet-cache/f3_denoise_loop")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Direct model source dir containing component config.json files.",
    )
    parser.add_argument(
        "--compiled-dir",
        default=None,
        help="Direct compiled app dir containing component model.pt artifacts.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--text-seq-len", type=int, default=None)
    parser.add_argument("--frame-rate", type=float, default=None)
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--mark-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call or wrap xm.mark_step per denoise step when torch_xla is available.",
    )
    parser.add_argument("--application-kwarg", action="append", default=[])
    parser.add_argument(
        "--prompt",
        default="A cinematic shot of a small robot walking through rain.",
    )
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument(
        "--output-type",
        default="latent",
        help="Flux output_type. Use latent for denoise-loop-only measurement.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = MODEL_ALIASES[args.model]
    if model != "flux" and args.bundle is None:
        raise SystemExit(f"--bundle is required for --model {model}")

    bundle_path = Path(args.bundle) if args.bundle is not None else None
    meta = _bundle_meta(bundle_path)
    args.model_id = args.model_id or meta.get("model_id") or DEFAULT_MODEL_IDS[model]

    print(
        f"[f3-profile] model={model} model_id={args.model_id} tp={args.tp_degree} "
        f"dtype={args.dtype} bundle={bundle_path}",
        flush=True,
    )
    run_started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    runner = RUNNERS[model]
    measurements, _pipe = runner(args, meta)
    steps = measurements["steps"]
    summary = _summarize_steps(steps)
    profile = {
        "schema": "difflet-f3-denoise-loop-profile-v2",
        "created_utc": run_started,
        "model": model,
        "model_id": args.model_id,
        "bundle": str(bundle_path) if bundle_path is not None else None,
        "bundle_meta": meta,
        "tp_degree": int(args.tp_degree),
        "dtype": str(args.dtype),
        "mark_step_enabled": bool(args.mark_step),
        "measurement_notes": [
            "device_step_time_s is the timed Trainium transformer call boundary.",
            "device_step_time_s is not hardware-only execution time; it may include "
            "runtime blocking, synchronization, and transfer-related waiting.",
            "difflet.utils.benchmark.LatencyCollector is a forward-hook wall-clock timer "
            "in this repository, so it is not treated as a Neuron hardware counter.",
            "F3.0b samples torch_xla.debug.metrics before and after each step; transfer "
            "time/count/bytes are per-step deltas when the runtime exposes the relevant "
            "Transfer* or InboundData/OutboundData metrics.",
            "neff_execution_time_s prefers XLA ExecuteReplicatedTime/ExecuteTime deltas; "
            "when unavailable it falls back to device_step_time_s and marks "
            "neff_execution_time_source accordingly.",
            "neuron-profile inspect summaries may replace fallback rows with "
            "neuron_profile:nrt_execute_max_worker_duration before gate auditing.",
            "cpu_round_trip_share = max(host_step_time_s - neff_execution_time_s, 0) / "
            "host_step_time_s. Treat fallback-source rows as wall-clock-only evidence.",
        ],
        "neff_gate_timing_sources": list(NEFF_GATE_TIMING_SOURCES),
        "latency_collector_status": {
            "module": "difflet.utils.benchmark.LatencyCollector",
            "usable_as_gate_neff_counter": False,
            "reason": "forward_hook_wall_clock_timer_not_neuron_hardware_counter",
        },
        **measurements,
        "summary": summary,
        "abort_gate": {
            "threshold": 0.10,
            "passes": bool(summary["hardware_timed_steps"] and summary["clears_10pct_gate"]),
            "metric": "max_cpu_round_trip_share",
            "requires_hardware_timed_steps": True,
            "hardware_timed_steps": int(summary["hardware_timed_steps"]),
        },
    }

    out_path = (
        Path(args.metrics_out)
        if args.metrics_out
        else Path(args.out_dir) / f"profile_{model.replace('-', '_')}_pre_f3.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "metrics_out": str(out_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
