"""Pure schedule-cost and anchor-placement algorithms.

The functions in this module are label-free. They operate on scheduler
configuration and full-compute trajectories; semantic metrics are intentionally
handled by the independent qualification layer.
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


def optimize_budget_frontier(
    segment_costs: Mapping[tuple[int, int, int], float],
    *,
    num_steps: int,
    warmup_steps: int,
    minimum_anchor_budget: int,
    phase_boundary: int,
    middle_gap_cap: int,
    tail_gap_cap: int,
) -> tuple[tuple[int, tuple[int, ...], float], ...]:
    """Optimize every feasible budget at or above a frozen search floor.

    Budget is deliberately not selected here.  This function is a label-free
    candidate generator; an independent closed-loop quality gate chooses the
    minimum qualified budget from the generated ascending ladder.
    """

    if (
        isinstance(minimum_anchor_budget, bool)
        or not isinstance(minimum_anchor_budget, int)
        or minimum_anchor_budget <= warmup_steps
        or minimum_anchor_budget > num_steps
    ):
        raise ValueError("minimum anchor budget is incompatible with the horizon")
    frontier: list[tuple[int, tuple[int, ...], float]] = []
    for anchor_budget in range(minimum_anchor_budget, num_steps + 1):
        try:
            anchors, objective = optimize_mask(
                segment_costs,
                num_steps=num_steps,
                warmup_steps=warmup_steps,
                anchor_budget=anchor_budget,
                phase_boundary=phase_boundary,
                middle_gap_cap=middle_gap_cap,
                tail_gap_cap=tail_gap_cap,
            )
        except ValueError as error:
            if str(error).startswith("no valid anchor mask exists for budget "):
                continue
            raise
        frontier.append((anchor_budget, anchors, objective))
    if not frontier:
        raise ValueError("no structurally feasible static anchor budget exists")
    return tuple(frontier)


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
    "gap_cap",
    "materialized_path_segments",
    "nearest_rank",
    "optimize_budget_frontier",
    "optimize_mask",
    "relative_prediction_error",
    "scheduler_sigmas",
    "trajectory_gram",
]
