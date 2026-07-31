#!/usr/bin/env python3
"""Evaluate FLUX cache candidates with per-sample, worst-case quality gates.

This script is intentionally offline: it reads the tensor/image manifests from
``collect_flux_cache_ab.py`` and never initializes Neuron. Metrics are computed
for every prompt×seed comparison before aggregation. Cosine/PSNR/SSIM use
PyTorch; LPIPS is loaded lazily from the optional ``lpips`` package and fails
closed when unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_flux_cache_ab import (  # noqa: E402
    QUALITY_INPUT_SCHEMA,
    SPEEDUP_CANDIDATES_SCHEMA,
)

QUALITY_CURVE_SCHEMA = "quality-curve-v2"
DEFAULT_MIN_SPEEDUP = 1.5
DEFAULT_MIN_TRAJECTORY_COSINE = 0.9999
DEFAULT_MIN_FINAL_LATENT_COSINE = 0.9995
DEFAULT_MIN_PSNR_DB = 30.0
DEFAULT_MAX_LPIPS = 0.10

LPIPSFunction = Callable[[Any, Any], float]


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"{name} does not exist: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON: {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return document


def _check_keys(
    value: Mapping[str, Any],
    name: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")


def _strict_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a {qualifier}finite number")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def _strict_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return value


_IDENTITY_FIELDS = (
    "model",
    "model_id",
    "shape_label",
    "num_steps",
    "scheduler_class",
    "guidance_scale",
    "prompt_count",
    "seed_count",
    "sample_count",
)
_COMMON_FIELDS = {
    *_IDENTITY_FIELDS,
    "hardware_measured",
    "started_at",
    "completed_at",
}


def _validate_identity(document: Mapping[str, Any], name: str) -> dict[str, Any]:
    identity = {
        "model": _strict_string(document["model"], f"{name}.model"),
        "model_id": _strict_string(document["model_id"], f"{name}.model_id"),
        "shape_label": _strict_string(document["shape_label"], f"{name}.shape_label"),
        "num_steps": _strict_positive_int(document["num_steps"], f"{name}.num_steps"),
        "scheduler_class": _strict_string(document["scheduler_class"], f"{name}.scheduler_class"),
        "guidance_scale": _finite_float(document["guidance_scale"], f"{name}.guidance_scale"),
        "prompt_count": _strict_positive_int(document["prompt_count"], f"{name}.prompt_count"),
        "seed_count": _strict_positive_int(document["seed_count"], f"{name}.seed_count"),
        "sample_count": _strict_positive_int(document["sample_count"], f"{name}.sample_count"),
    }
    _strict_bool(document["hardware_measured"], f"{name}.hardware_measured")
    _strict_string(document["started_at"], f"{name}.started_at")
    _strict_string(document["completed_at"], f"{name}.completed_at")
    expected_samples = identity["prompt_count"] * identity["seed_count"]
    if identity["sample_count"] != expected_samples:
        raise ValueError(
            f"{name}.sample_count must equal prompt_count * seed_count " f"({expected_samples})"
        )
    return identity


def _validate_candidate_definition(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    _check_keys(
        value,
        name,
        required={"candidate_id", "policy", "predictor"},
    )
    candidate_id = _strict_string(value["candidate_id"], f"{name}.candidate_id")
    if not isinstance(value["policy"], dict) or not isinstance(value["predictor"], dict):
        raise ValueError(f"{name}.policy and predictor must be JSON objects")
    from difflet.pipeline.cache import build_policy, build_predictor

    build_policy(value["policy"])
    build_predictor(value["predictor"])
    return {
        "candidate_id": candidate_id,
        "policy": dict(value["policy"]),
        "predictor": dict(value["predictor"]),
    }


def _validate_artifacts(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    _check_keys(
        value,
        name,
        required={"trajectory", "final_latent", "image"},
    )
    return {
        key: _strict_string(value[key], f"{name}.{key}")
        for key in ("trajectory", "final_latent", "image")
    }


def _validate_quality_input(
    document: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    _check_keys(
        document,
        "quality input",
        required={"schema", *_COMMON_FIELDS, "candidates", "comparisons"},
    )
    if document["schema"] != QUALITY_INPUT_SCHEMA:
        raise ValueError(
            f"quality input schema must be {QUALITY_INPUT_SCHEMA!r}, " f"got {document['schema']!r}"
        )
    identity = _validate_identity(document, "quality input")
    definitions: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(_strict_list(document["candidates"], "quality input.candidates")):
        definition = _validate_candidate_definition(value, f"quality input.candidates[{index}]")
        candidate_id = definition["candidate_id"]
        if candidate_id in definitions:
            raise ValueError(f"duplicate quality candidate_id: {candidate_id!r}")
        definitions[candidate_id] = definition

    comparisons: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(
        _strict_list(document["comparisons"], "quality input.comparisons")
    ):
        name = f"quality input.comparisons[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a JSON object")
        _check_keys(
            value,
            name,
            required={
                "sample_id",
                "prompt_index",
                "prompt",
                "seed",
                "candidate_id",
                "baseline",
                "candidate",
            },
        )
        candidate_id = _strict_string(value["candidate_id"], f"{name}.candidate_id")
        if candidate_id not in definitions:
            raise ValueError(f"{name} references unknown candidate {candidate_id!r}")
        sample_id = _strict_string(value["sample_id"], f"{name}.sample_id")
        key = (candidate_id, sample_id)
        if key in seen:
            raise ValueError(f"duplicate quality comparison for {key!r}")
        seen.add(key)
        prompt_index = value["prompt_index"]
        seed = value["seed"]
        if isinstance(prompt_index, bool) or not isinstance(prompt_index, int) or prompt_index < 0:
            raise ValueError(f"{name}.prompt_index must be a nonnegative integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"{name}.seed must be a nonnegative integer")
        comparisons.append(
            {
                "sample_id": sample_id,
                "prompt_index": int(prompt_index),
                "prompt": _strict_string(value["prompt"], f"{name}.prompt"),
                "seed": int(seed),
                "candidate_id": candidate_id,
                "baseline": _validate_artifacts(value["baseline"], f"{name}.baseline"),
                "candidate": _validate_artifacts(value["candidate"], f"{name}.candidate"),
            }
        )
    expected = identity["sample_count"] * len(definitions)
    if len(comparisons) != expected:
        raise ValueError(f"quality input has {len(comparisons)} comparisons, expected {expected}")
    by_candidate: dict[str, dict[str, tuple[Any, ...]]] = {
        candidate_id: {} for candidate_id in definitions
    }
    for comparison in comparisons:
        by_candidate[comparison["candidate_id"]][comparison["sample_id"]] = (
            comparison["prompt_index"],
            comparison["prompt"],
            comparison["seed"],
            comparison["baseline"],
        )
    canonical = next(iter(by_candidate.values()))
    if len(canonical) != identity["sample_count"]:
        raise ValueError("quality input does not contain the declared sample matrix")
    for candidate_id, matrix in by_candidate.items():
        if matrix != canonical:
            raise ValueError(
                f"candidate {candidate_id!r} does not use the same baseline sample matrix"
            )
    if len({row[0] for row in canonical.values()}) != identity["prompt_count"]:
        raise ValueError("quality input prompt_count does not match the sample matrix")
    if len({row[2] for row in canonical.values()}) != identity["seed_count"]:
        raise ValueError("quality input seed_count does not match the sample matrix")
    coordinates = {(row[0], row[2]) for row in canonical.values()}
    if len(coordinates) != identity["sample_count"]:
        raise ValueError("quality input repeats a prompt_index/seed combination")
    prompt_labels: dict[int, str] = {}
    for prompt_index, prompt, _, _ in canonical.values():
        previous = prompt_labels.setdefault(prompt_index, prompt)
        if previous != prompt:
            raise ValueError("one prompt_index maps to multiple prompt strings")
    return identity, definitions, comparisons


def _validate_speedup_input(
    document: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _check_keys(
        document,
        "speedup candidates",
        required={"schema", *_COMMON_FIELDS, "baseline", "candidates"},
    )
    if document["schema"] != SPEEDUP_CANDIDATES_SCHEMA:
        raise ValueError(
            f"speedup schema must be {SPEEDUP_CANDIDATES_SCHEMA!r}, " f"got {document['schema']!r}"
        )
    identity = _validate_identity(document, "speedup candidates")
    baseline = document["baseline"]
    if not isinstance(baseline, dict):
        raise ValueError("speedup candidates.baseline must be a JSON object")
    _check_keys(baseline, "speedup candidates.baseline", required={"total_s", "samples"})
    baseline_total = _finite_float(
        baseline["total_s"],
        "speedup candidates.baseline.total_s",
        positive=True,
    )
    baseline_samples = _validate_timing_samples(
        baseline["samples"],
        "speedup candidates.baseline.samples",
        with_runner_stats=False,
    )
    if len(baseline_samples) != identity["sample_count"]:
        raise ValueError("speedup baseline does not contain the declared sample count")
    if not math.isclose(
        sum(sample["elapsed_s"] for sample in baseline_samples),
        baseline_total,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("speedup baseline total_s disagrees with its sample timings")
    baseline_ids = {sample["sample_id"] for sample in baseline_samples}

    candidates: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(
        _strict_list(document["candidates"], "speedup candidates.candidates")
    ):
        name = f"speedup candidates.candidates[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a JSON object")
        _check_keys(
            value,
            name,
            required={
                "candidate_id",
                "policy",
                "predictor",
                "total_s",
                "measured_speedup",
                "hardware_measured",
                "runner_stats",
                "samples",
            },
        )
        candidate_id = _strict_string(value["candidate_id"], f"{name}.candidate_id")
        if candidate_id in candidates:
            raise ValueError(f"duplicate speedup candidate_id: {candidate_id!r}")
        if not isinstance(value["policy"], dict) or not isinstance(value["predictor"], dict):
            raise ValueError(f"{name}.policy and predictor must be JSON objects")
        if not isinstance(value["runner_stats"], dict):
            raise ValueError(f"{name}.runner_stats must be a JSON object")
        samples = _validate_timing_samples(
            value["samples"],
            f"{name}.samples",
            with_runner_stats=True,
        )
        if {sample["sample_id"] for sample in samples} != baseline_ids:
            raise ValueError(f"{name}.samples do not match the baseline sample identifiers")
        total_s = _finite_float(value["total_s"], f"{name}.total_s", positive=True)
        if not math.isclose(
            sum(sample["elapsed_s"] for sample in samples),
            total_s,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{name}.total_s disagrees with its sample timings")
        measured_speedup = _finite_float(
            value["measured_speedup"],
            f"{name}.measured_speedup",
            positive=True,
        )
        if not math.isclose(
            measured_speedup,
            baseline_total / total_s,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{name}.measured_speedup disagrees with measured total_s values")
        aggregate_stats: dict[str, int] = {}
        for sample in samples:
            for key, stat in sample["runner_stats"].items():
                if isinstance(stat, bool) or not isinstance(stat, int):
                    continue
                aggregate_stats[key] = aggregate_stats.get(key, 0) + stat
        if value["runner_stats"] != aggregate_stats:
            raise ValueError(f"{name}.runner_stats disagrees with per-sample runner statistics")
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "policy": dict(value["policy"]),
            "predictor": dict(value["predictor"]),
            "total_s": total_s,
            "measured_speedup": measured_speedup,
            "hardware_measured": _strict_bool(
                value["hardware_measured"], f"{name}.hardware_measured"
            ),
            "runner_stats": dict(value["runner_stats"]),
            "samples": samples,
        }
    return identity, candidates


def _validate_timing_samples(
    value: Any,
    name: str,
    *,
    with_runner_stats: bool,
) -> list[dict[str, Any]]:
    rows = _strict_list(value, name)
    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    required = {"sample_id", "elapsed_s"}
    if with_runner_stats:
        required.add("runner_stats")
    for index, row in enumerate(rows):
        row_name = f"{name}[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{row_name} must be a JSON object")
        _check_keys(row, row_name, required=required)
        sample_id = _strict_string(row["sample_id"], f"{row_name}.sample_id")
        if sample_id in identifiers:
            raise ValueError(f"{name} contains duplicate sample_id {sample_id!r}")
        identifiers.add(sample_id)
        normalized_row = {
            "sample_id": sample_id,
            "elapsed_s": _finite_float(row["elapsed_s"], f"{row_name}.elapsed_s", positive=True),
        }
        if with_runner_stats:
            if not isinstance(row["runner_stats"], dict):
                raise ValueError(f"{row_name}.runner_stats must be a JSON object")
            normalized_row["runner_stats"] = dict(row["runner_stats"])
        normalized.append(normalized_row)
    return normalized


def _artifact_path(manifest_path: Path, relative: str, name: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"{name} must be relative to the quality manifest")
    root = manifest_path.parent.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the quality manifest directory") from error
    if not resolved.is_file():
        raise ValueError(f"{name} does not exist: {resolved}")
    return resolved


def _load_tensor(path: Path, name: str):
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"could not load {name} tensor {path}: {error}") from error
    if not torch.is_tensor(value):
        raise ValueError(f"{name} artifact must contain one tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} tensor must have a floating dtype")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"{name} tensor contains non-finite values")
    return value.detach().cpu()


def _load_image(path: Path, name: str):
    import numpy as np
    import torch
    from PIL import Image

    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    except (OSError, ValueError) as error:
        raise ValueError(f"could not load {name} image {path}: {error}") from error
    return torch.from_numpy(array).permute(2, 0, 1).div_(255.0)


def tensor_cosine(lhs: Any, rhs: Any) -> float:
    """Use the cache architecture's existing flattened cosine implementation."""

    from difflet.pipeline.latent_metrics import _cosine

    if tuple(lhs.shape) != tuple(rhs.shape):
        raise ValueError(
            f"cosine tensors must have identical shapes, got {lhs.shape} and {rhs.shape}"
        )
    value = _finite_float(_cosine(lhs, rhs), "cosine")
    return max(-1.0, min(1.0, value))


def trajectory_cosine(lhs: Any, rhs: Any, *, num_steps: int) -> float:
    """Return the worst per-step latent cosine across the full trajectory."""

    if lhs.ndim < 2 or rhs.ndim < 2:
        raise ValueError("trajectory tensors must include step and latent dimensions")
    if lhs.shape[0] != num_steps or rhs.shape[0] != num_steps:
        raise ValueError(
            f"trajectory tensors must contain {num_steps} steps, got "
            f"{lhs.shape[0]} and {rhs.shape[0]}"
        )
    if tuple(lhs.shape) != tuple(rhs.shape):
        raise ValueError("trajectory tensors must have identical shapes")
    return min(tensor_cosine(lhs[step], rhs[step]) for step in range(num_steps))


def psnr_db(lhs: Any, rhs: Any) -> float:
    import torch

    if tuple(lhs.shape) != tuple(rhs.shape):
        raise ValueError("PSNR images must have identical shapes")
    mse = float(torch.mean((lhs.float() - rhs.float()).pow(2)).item())
    # A finite ceiling keeps the output strict JSON while preserving the exact
    # match semantics (120 dB corresponds to an MSE floor of 1e-12).
    return float(10.0 * math.log10(1.0 / max(mse, 1e-12)))


def ssim(lhs: Any, rhs: Any) -> float:
    """Compute channel-averaged Gaussian-window SSIM for RGB [0,1] tensors."""

    import torch
    import torch.nn.functional as functional

    if tuple(lhs.shape) != tuple(rhs.shape) or lhs.ndim != 3:
        raise ValueError("SSIM images must be identically shaped CHW tensors")
    height, width = int(lhs.shape[-2]), int(lhs.shape[-1])
    window = min(11, height, width)
    if window % 2 == 0:
        window -= 1
    if window <= 0:
        raise ValueError("SSIM images must have non-empty spatial dimensions")
    sigma = 1.5
    coordinate = torch.arange(window, dtype=torch.float32) - (window - 1) / 2
    kernel_1d = torch.exp(-(coordinate.pow(2)) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = int(lhs.shape[0])
    kernel = kernel_2d.expand(channels, 1, window, window).contiguous()
    left = lhs.float().unsqueeze(0)
    right = rhs.float().unsqueeze(0)
    padding = window // 2
    mu_left = functional.conv2d(left, kernel, padding=padding, groups=channels)
    mu_right = functional.conv2d(right, kernel, padding=padding, groups=channels)
    mu_left_sq = mu_left.pow(2)
    mu_right_sq = mu_right.pow(2)
    mu_product = mu_left * mu_right
    sigma_left = (
        functional.conv2d(left * left, kernel, padding=padding, groups=channels) - mu_left_sq
    )
    sigma_right = (
        functional.conv2d(right * right, kernel, padding=padding, groups=channels) - mu_right_sq
    )
    sigma_cross = (
        functional.conv2d(left * right, kernel, padding=padding, groups=channels) - mu_product
    )
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_product + c1) * (2 * sigma_cross + c2)) / (
        (mu_left_sq + mu_right_sq + c1) * (sigma_left + sigma_right + c2)
    )
    value = _finite_float(score.mean().item(), "SSIM")
    return max(-1.0, min(1.0, value))


def _validate_lpips_images(lhs: Any, rhs: Any) -> None:
    if (
        lhs.ndim != 3
        or rhs.ndim != 3
        or tuple(lhs.shape) != tuple(rhs.shape)
        or min(*lhs.shape[-2:], *rhs.shape[-2:]) < 64
    ):
        raise ValueError(
            "LPIPS images must be identically shaped CHW tensors with spatial "
            "dimensions of at least 64x64"
        )


def build_lpips_function(net: str = "alex") -> LPIPSFunction:
    """Load LPIPS lazily so tensor-only helpers and unit tests stay lightweight."""

    try:
        import lpips
    except ImportError as error:
        raise RuntimeError(
            "LPIPS evaluation requires the optional 'lpips' package; "
            "install it with: pip install -e '.[cache-eval]'"
        ) from error
    model = lpips.LPIPS(net=net)
    model.eval()

    def metric(lhs: Any, rhs: Any) -> float:
        import torch

        _validate_lpips_images(lhs, rhs)
        with torch.inference_mode():
            value = model(
                lhs.float().unsqueeze(0).mul(2.0).sub(1.0),
                rhs.float().unsqueeze(0).mul(2.0).sub(1.0),
            )
        result = _finite_float(value.reshape(-1)[0].item(), "LPIPS")
        if result < 0.0:
            raise ValueError("LPIPS must be nonnegative")
        return result

    return metric


def _failed_gates(
    metrics: Mapping[str, float],
    *,
    hardware_measured: bool,
    thresholds: Mapping[str, float],
) -> list[str]:
    failures: list[str] = []
    if not hardware_measured:
        failures.append("hardware_measured")
    for metric, threshold_name, comparison in (
        ("measured_speedup", "min_speedup", "min"),
        ("trajectory_cosine", "min_trajectory_cosine", "min"),
        ("final_latent_cosine", "min_final_latent_cosine", "min"),
        ("psnr_db", "min_psnr_db", "min"),
        ("lpips", "max_lpips", "max"),
    ):
        value = float(metrics[metric])
        threshold = float(thresholds[threshold_name])
        if (comparison == "min" and value < threshold) or (
            comparison == "max" and value > threshold
        ):
            failures.append(metric)
    return failures


def metric_config(lpips_net: str) -> dict[str, Any]:
    if lpips_net not in ("alex", "vgg", "squeeze"):
        raise ValueError("lpips_net must be one of: alex, vgg, squeeze")
    return {
        "trajectory_cosine": "minimum-per-step-flattened-v1",
        "final_latent_cosine": "flattened-v1",
        "psnr": {
            "data_range": 1.0,
            "mse_floor": 1e-12,
        },
        "ssim": {
            "data_range": 1.0,
            "max_window": 11,
            "sigma": 1.5,
        },
        "lpips": {
            "package": "lpips",
            "version": "0.1",
            "net": lpips_net,
            "input_range": "minus-one-to-one",
            "minimum_spatial_size": [64, 64],
        },
    }


def evaluate(
    quality_path: Path,
    speedup_path: Path,
    *,
    thresholds: Mapping[str, float],
    lpips_function: LPIPSFunction,
    lpips_net: str = "alex",
) -> dict[str, Any]:
    """Validate both manifests, compute metrics, and return quality-curve-v2."""

    import torch

    quality_document = _load_json(quality_path, "quality input")
    speedup_document = _load_json(speedup_path, "speedup candidates")
    identity, definitions, comparisons = _validate_quality_input(quality_document)
    speed_identity, speed_candidates = _validate_speedup_input(speedup_document)
    if identity != speed_identity:
        raise ValueError("quality and speedup experiment identities do not match")
    if set(definitions) != set(speed_candidates):
        raise ValueError("quality and speedup candidate sets do not match")
    for candidate_id, definition in definitions.items():
        speed = speed_candidates[candidate_id]
        if definition["policy"] != speed["policy"] or definition["predictor"] != speed["predictor"]:
            raise ValueError(
                f"candidate {candidate_id!r} policy/predictor differs between manifests"
            )

    per_candidate: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in definitions
    }
    for comparison in comparisons:
        baseline_paths = {
            key: _artifact_path(
                quality_path,
                comparison["baseline"][key],
                f"{comparison['sample_id']} baseline {key}",
            )
            for key in ("trajectory", "final_latent", "image")
        }
        candidate_paths = {
            key: _artifact_path(
                quality_path,
                comparison["candidate"][key],
                f"{comparison['sample_id']} candidate {key}",
            )
            for key in ("trajectory", "final_latent", "image")
        }
        baseline_trajectory = _load_tensor(baseline_paths["trajectory"], "baseline trajectory")
        candidate_trajectory = _load_tensor(candidate_paths["trajectory"], "candidate trajectory")
        baseline_final = _load_tensor(baseline_paths["final_latent"], "baseline final latent")
        candidate_final = _load_tensor(candidate_paths["final_latent"], "candidate final latent")
        if baseline_trajectory.ndim < 1 or not torch.equal(baseline_final, baseline_trajectory[-1]):
            raise ValueError(
                f"{comparison['sample_id']} baseline final latent does not "
                "match the final trajectory step"
            )
        if candidate_trajectory.ndim < 1 or not torch.equal(
            candidate_final, candidate_trajectory[-1]
        ):
            raise ValueError(
                f"{comparison['sample_id']} candidate final latent does not "
                "match the final trajectory step"
            )
        baseline_image = _load_image(baseline_paths["image"], "baseline")
        candidate_image = _load_image(candidate_paths["image"], "candidate")
        lpips_value = _finite_float(lpips_function(baseline_image, candidate_image), "LPIPS")
        if lpips_value < 0.0:
            raise ValueError("LPIPS must be nonnegative")
        sample_metrics = {
            "trajectory_cosine": trajectory_cosine(
                baseline_trajectory,
                candidate_trajectory,
                num_steps=identity["num_steps"],
            ),
            "final_latent_cosine": tensor_cosine(baseline_final, candidate_final),
            "psnr_db": psnr_db(baseline_image, candidate_image),
            "ssim": ssim(baseline_image, candidate_image),
            "lpips": lpips_value,
        }
        per_candidate[comparison["candidate_id"]].append(
            {
                "sample_id": comparison["sample_id"],
                "prompt_index": comparison["prompt_index"],
                "seed": comparison["seed"],
                **sample_metrics,
            }
        )

    candidate_rows: list[dict[str, Any]] = []
    for candidate_id, definition in definitions.items():
        samples = per_candidate[candidate_id]
        if len(samples) != identity["sample_count"]:
            raise ValueError(
                f"candidate {candidate_id!r} has {len(samples)} samples, "
                f"expected {identity['sample_count']}"
            )
        speed = speed_candidates[candidate_id]
        worst = {
            "trajectory_cosine": min(sample["trajectory_cosine"] for sample in samples),
            "final_latent_cosine": min(sample["final_latent_cosine"] for sample in samples),
            "psnr_db": min(sample["psnr_db"] for sample in samples),
            "ssim": min(sample["ssim"] for sample in samples),
            "lpips": max(sample["lpips"] for sample in samples),
        }
        hardware_measured = bool(
            quality_document["hardware_measured"] is True
            and speedup_document["hardware_measured"] is True
            and speed["hardware_measured"] is True
        )
        gate_metrics = {
            "measured_speedup": speed["measured_speedup"],
            **worst,
        }
        failures = _failed_gates(
            gate_metrics,
            hardware_measured=hardware_measured,
            thresholds=thresholds,
        )
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "policy": definition["policy"],
                "predictor": definition["predictor"],
                "hardware_measured": hardware_measured,
                "measured_speedup": speed["measured_speedup"],
                "runner_stats": speed["runner_stats"],
                "per_sample": samples,
                "worst_sample": worst,
                "passes_gate": not failures,
                "failed_gates": failures,
            }
        )

    return {
        "schema": QUALITY_CURVE_SCHEMA,
        **identity,
        "hardware_measured": bool(
            quality_document["hardware_measured"] is True
            and speedup_document["hardware_measured"] is True
        ),
        "aggregation": "worst-sample",
        "metric_config": metric_config(lpips_net),
        "thresholds": dict(thresholds),
        "candidates": candidate_rows,
        "passing_candidate_ids": [
            candidate["candidate_id"] for candidate in candidate_rows if candidate["passes_gate"]
        ],
    }


def default_thresholds(
    *,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    min_trajectory_cosine: float = DEFAULT_MIN_TRAJECTORY_COSINE,
    min_final_latent_cosine: float = DEFAULT_MIN_FINAL_LATENT_COSINE,
    min_psnr_db: float = DEFAULT_MIN_PSNR_DB,
    max_lpips: float = DEFAULT_MAX_LPIPS,
) -> dict[str, float]:
    speedup = _finite_float(min_speedup, "min_speedup", positive=True)
    trajectory = _finite_float(min_trajectory_cosine, "min_trajectory_cosine")
    final = _finite_float(min_final_latent_cosine, "min_final_latent_cosine")
    psnr = _finite_float(min_psnr_db, "min_psnr_db")
    lpips_value = _finite_float(max_lpips, "max_lpips")
    if not -1.0 <= trajectory <= 1.0:
        raise ValueError("min_trajectory_cosine must be between -1 and 1")
    if not -1.0 <= final <= 1.0:
        raise ValueError("min_final_latent_cosine must be between -1 and 1")
    if psnr < 0.0:
        raise ValueError("min_psnr_db must be nonnegative")
    if lpips_value < 0.0:
        raise ValueError("max_lpips must be nonnegative")
    return {
        "min_speedup": speedup,
        "min_trajectory_cosine": trajectory,
        "min_final_latent_cosine": final,
        "min_psnr_db": psnr,
        "max_lpips": lpips_value,
    }


def _write_json(path: Path, document: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"output already exists: {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-input", required=True)
    parser.add_argument("--speedup-candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-speedup", type=float, default=DEFAULT_MIN_SPEEDUP)
    parser.add_argument(
        "--min-trajectory-cosine",
        type=float,
        default=DEFAULT_MIN_TRAJECTORY_COSINE,
    )
    parser.add_argument(
        "--min-final-latent-cosine",
        type=float,
        default=DEFAULT_MIN_FINAL_LATENT_COSINE,
    )
    parser.add_argument("--min-psnr-db", type=float, default=DEFAULT_MIN_PSNR_DB)
    parser.add_argument("--max-lpips", type=float, default=DEFAULT_MAX_LPIPS)
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        thresholds = default_thresholds(
            min_speedup=args.min_speedup,
            min_trajectory_cosine=args.min_trajectory_cosine,
            min_final_latent_cosine=args.min_final_latent_cosine,
            min_psnr_db=args.min_psnr_db,
            max_lpips=args.max_lpips,
        )
        result = evaluate(
            Path(args.quality_input).expanduser().resolve(),
            Path(args.speedup_candidates).expanduser().resolve(),
            thresholds=thresholds,
            lpips_function=build_lpips_function(args.lpips_net),
            lpips_net=args.lpips_net,
        )
        output = Path(args.out).expanduser().resolve()
        _write_json(output, result, force=bool(args.force))
    except (FileExistsError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", flush=True)
        return 2
    print(
        f"[cache-quality] {len(result['passing_candidate_ids'])}/"
        f"{len(result['candidates'])} candidates passed -> {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
