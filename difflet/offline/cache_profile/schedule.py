"""Pure schedule-cost and anchor-placement algorithms.

The functions in this module are label-free. They operate on hardware timing
points, scheduler configuration, and full-compute trajectories; semantic
metrics are intentionally handled by the independent qualification layer.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def nearest_rank(values: Sequence[float], quantile: float) -> float:
    """Return the deterministic nearest-rank quantile."""

    if not values:
        raise ValueError("nearest-rank quantile requires at least one value")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def finite_positive(value: Any, name: str) -> float:
    """Parse one strictly positive finite scalar."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def fit_affine_step_cost(points: Sequence[tuple[int, float]]) -> dict[str, Any]:
    """Fit aggregate latency = intercept + incremental cost * real steps."""

    grouped: dict[int, list[float]] = {}
    for real_steps, latency_s in points:
        if (
            isinstance(real_steps, bool)
            or not isinstance(real_steps, int)
            or real_steps <= 0
        ):
            raise ValueError("hardware calibration real-step counts must be positive integers")
        grouped.setdefault(real_steps, []).append(
            finite_positive(latency_s, "hardware calibration latency")
        )
    if len(grouped) < 2:
        raise ValueError("hardware calibration requires at least two real-step counts")

    counts = np.asarray(sorted(grouped), dtype=np.float64)
    latencies = np.asarray(
        [sum(grouped[int(count)]) / len(grouped[int(count)]) for count in counts],
        dtype=np.float64,
    )
    centered = counts - float(counts.mean())
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        raise ValueError("hardware calibration real-step counts have no variance")
    incremental = float(
        np.dot(centered, latencies - float(latencies.mean())) / denominator
    )
    intercept = float(latencies.mean() - incremental * counts.mean())
    if not math.isfinite(incremental) or incremental <= 0.0:
        raise ValueError("hardware calibration must have positive incremental real-step cost")
    if not math.isfinite(intercept) or intercept < 0.0:
        raise ValueError("hardware calibration must have a nonnegative latency intercept")

    predicted = intercept + incremental * counts
    residual_sum = float(np.square(latencies - predicted).sum())
    total_sum = float(np.square(latencies - float(latencies.mean())).sum())
    r_squared = 1.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    return {
        "fit_method": "ordinary_least_squares_over_static_profiles",
        "latency_formula": (
            "aggregate_latency_s=intercept_s+incremental_real_step_s*real_steps"
        ),
        "intercept_s": intercept,
        "incremental_real_step_s": incremental,
        "fit_r_squared": r_squared,
        "points": [
            {
                "real_steps": int(count),
                "aggregate_latency_s": float(latency),
            }
            for count, latency in zip(counts, latencies)
        ],
    }


def derive_hardware_budget(
    *,
    num_steps: int,
    warmup_steps: int,
    cooldown_steps: int,
    baseline_latency_s: float,
    target_speedup: float,
    intercept_s: float,
    incremental_real_step_s: float,
    dynamic_step_reserve: int,
) -> dict[str, Any]:
    """Derive one static anchor budget from a minimum hardware speed target."""

    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if (
        isinstance(dynamic_step_reserve, bool)
        or not isinstance(dynamic_step_reserve, int)
        or dynamic_step_reserve < 0
    ):
        raise ValueError("dynamic_step_reserve must be a nonnegative integer")
    baseline = finite_positive(baseline_latency_s, "baseline_latency_s")
    target = finite_positive(target_speedup, "target_speedup")
    if target <= 1.0:
        raise ValueError("target_speedup must be greater than one")
    intercept = float(intercept_s)
    incremental = finite_positive(
        incremental_real_step_s,
        "incremental_real_step_s",
    )
    if not math.isfinite(intercept) or intercept < 0.0:
        raise ValueError("intercept_s must be a nonnegative finite number")

    maximum_latency = baseline / target
    raw_total_budget = (maximum_latency - intercept) / incremental
    total_real_budget = min(num_steps, math.floor(raw_total_budget + 1e-12))
    static_anchor_budget = total_real_budget - dynamic_step_reserve
    required_static = warmup_steps + cooldown_steps
    if cooldown_steps == 0:
        required_static += 1
    if static_anchor_budget < required_static:
        raise ValueError(
            "target speed leaves too few static anchors after the dynamic-step reserve"
        )
    predicted_static_latency = intercept + incremental * static_anchor_budget
    predicted_reserved_latency = intercept + incremental * total_real_budget
    return {
        "selection_rule": "largest_total_real_step_budget_meeting_minimum_target_speedup",
        "budget_formula": (
            "floor((baseline_latency_s/target_speedup-intercept_s)"
            "/incremental_real_step_s)"
        ),
        "target_speedup": target,
        "maximum_aggregate_latency_s": maximum_latency,
        "total_real_step_budget": total_real_budget,
        "dynamic_step_reserve": dynamic_step_reserve,
        "static_anchor_budget": static_anchor_budget,
        "predicted_static_latency_s": predicted_static_latency,
        "predicted_static_speedup": baseline / predicted_static_latency,
        "predicted_reserved_latency_s": predicted_reserved_latency,
        "predicted_reserved_speedup": baseline / predicted_reserved_latency,
        "fail_closed_steps_exempt_from_speed_target": True,
    }


def scheduler_sigmas(generation: Mapping[str, Any]) -> tuple[float, ...]:
    """Reproduce the registered FlowMatch Euler sigma schedule without model imports."""

    if generation.get("scheduler_class") != "FlowMatchEulerDiscreteScheduler":
        raise ValueError("only FlowMatchEulerDiscreteScheduler is supported")
    config = generation.get("scheduler_config")
    if not isinstance(config, dict) or config.get("use_dynamic_shifting") is not True:
        raise ValueError("registered scheduler must use dynamic shifting")
    if config.get("time_shift_type") != "exponential":
        raise ValueError("registered scheduler must use exponential time shifting")
    if any(
        bool(config.get(name))
        for name in (
            "invert_sigmas",
            "shift_terminal",
            "use_beta_sigmas",
            "use_exponential_sigmas",
            "use_karras_sigmas",
        )
    ):
        raise ValueError("registered scheduler enables an unsupported sigma transform")
    steps = int(generation["num_steps"])
    height = int(generation["height"])
    width = int(generation["width"])
    if height % 16 or width % 16:
        raise ValueError("FLUX dimensions must be divisible by 16")
    image_seq_len = (height // 16) * (width // 16)
    base_len = int(config["base_image_seq_len"])
    max_len = int(config["max_image_seq_len"])
    base_shift = float(config["base_shift"])
    max_shift = float(config["max_shift"])
    slope = (max_shift - base_shift) / (max_len - base_len)
    mu = image_seq_len * slope + (base_shift - slope * base_len)
    raw = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
    shifted = math.exp(mu) / (math.exp(mu) + (1.0 / raw - 1.0))
    return tuple(float(value) for value in np.concatenate([shifted, np.zeros(1, np.float32)]))


def relative_prediction_error(
    gram: Any,
    a: int,
    b: int,
    target: int,
    *,
    norm_floor: float,
) -> float:
    """Evaluate order-1 index extrapolation error from a velocity Gram matrix."""

    if not 1 <= a < b < target:
        raise ValueError("prediction indices must satisfy 1 <= a < b < target")
    ratio = (target - b) / (b - a)
    coefficients = (-ratio, 1.0 + ratio, -1.0)
    indices = (a - 1, b - 1, target - 1)
    squared = 0.0
    for left, left_index in enumerate(indices):
        for right, right_index in enumerate(indices):
            squared += (
                coefficients[left]
                * coefficients[right]
                * float(gram[left_index, right_index])
            )
    numerator = math.sqrt(max(squared, 0.0))
    denominator = max(
        math.sqrt(max(float(gram[target - 1, target - 1]), 0.0)),
        norm_floor,
    )
    return numerator / denominator


def gap_cap(last_anchor: int, *, boundary: int, middle_cap: int, tail_cap: int) -> int:
    """Return the registered phase-specific maximum anchor gap."""

    return middle_cap if last_anchor < boundary else tail_cap


def optimize_mask(
    segment_costs: Mapping[tuple[int, int, int], float],
    *,
    num_steps: int,
    warmup_steps: int,
    anchor_budget: int,
    phase_boundary: int,
    middle_gap_cap: int,
    tail_gap_cap: int,
) -> tuple[tuple[int, ...], float]:
    """Find the minimum-cost exact-budget anchor path with lexical tie-breaking."""

    if warmup_steps < 2 or anchor_budget <= warmup_steps or anchor_budget > num_steps:
        raise ValueError("anchor budget is incompatible with warmup")
    prefix = tuple(range(warmup_steps))
    final_step = num_steps - 1

    @lru_cache(maxsize=None)
    def solve(a: int, b: int, anchors_left: int) -> tuple[float, tuple[int, ...]] | None:
        if anchors_left == 1:
            cap = gap_cap(
                b,
                boundary=phase_boundary,
                middle_cap=middle_gap_cap,
                tail_cap=tail_gap_cap,
            )
            if final_step <= b or final_step - b > cap:
                return None
            key = (a, b, final_step)
            if key not in segment_costs:
                return None
            return float(segment_costs[key]), (final_step,)

        cap = gap_cap(
            b,
            boundary=phase_boundary,
            middle_cap=middle_gap_cap,
            tail_cap=tail_gap_cap,
        )
        maximum = min(final_step - (anchors_left - 1), b + cap)
        best: tuple[float, tuple[int, ...]] | None = None
        for candidate in range(b + 1, maximum + 1):
            key = (a, b, candidate)
            if key not in segment_costs:
                continue
            suffix = solve(b, candidate, anchors_left - 1)
            if suffix is None:
                continue
            result = (float(segment_costs[key]) + suffix[0], (candidate, *suffix[1]))
            if (
                best is None
                or result[0] < best[0]
                or (
                    math.isclose(result[0], best[0], rel_tol=0.0, abs_tol=1e-15)
                    and result[1] < best[1]
                )
            ):
                best = result
        return best

    result = solve(warmup_steps - 2, warmup_steps - 1, anchor_budget - warmup_steps)
    if result is None:
        raise ValueError(f"no valid anchor mask exists for budget {anchor_budget}")
    anchors = (*prefix, *result[1])
    if len(anchors) != anchor_budget or anchors[-1] != final_step:
        raise RuntimeError("optimizer produced an invalid anchor count")
    return anchors, result[0]


def materialized_path_segments(
    anchors: Sequence[int],
    segment_costs: Mapping[tuple[int, int, int], float],
    *,
    warmup_steps: int,
) -> list[dict[str, Any]]:
    """Describe only post-warmup segments scored by the optimizer."""

    rows = []
    for index in range(warmup_steps, len(anchors)):
        a, b, c = anchors[index - 2 : index + 1]
        key = (a, b, c)
        if key not in segment_costs:
            raise ValueError(f"optimized path is missing segment cost {key}")
        rows.append(
            {
                "previous_anchor": a,
                "anchor": b,
                "next_anchor": c,
                "prompt_q95_cost": float(segment_costs[key]),
            }
        )
    return rows


def trajectory_gram(path: Path, deltas: Sequence[float]) -> Any:
    """Build a compact velocity Gram matrix from one full-compute trajectory."""

    import torch

    trajectory = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(trajectory) or trajectory.shape[0] != len(deltas):
        raise ValueError(f"trajectory has an unexpected shape: {path}")
    states = trajectory.float()
    delta = torch.tensor(deltas[1:], dtype=torch.float32)
    reshape = (len(delta),) + (1,) * (states.ndim - 1)
    velocities = (states[1:] - states[:-1]) / delta.reshape(reshape)
    flat = velocities.reshape(len(delta), -1)
    return torch.mm(flat, flat.t()).double().numpy()


__all__ = [
    "derive_hardware_budget",
    "finite_positive",
    "fit_affine_step_cost",
    "gap_cap",
    "materialized_path_segments",
    "nearest_rank",
    "optimize_mask",
    "relative_prediction_error",
    "scheduler_sigmas",
    "trajectory_gram",
]
