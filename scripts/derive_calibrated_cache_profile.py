"""Derive a calibrated_linear cache profile from registered calibration trajectories.

For the base candidate's static schedule, solve the reuse weights of every
skipped step jointly so the teacher-forced final-latent deviation
``||sum_t dsigma_t (vhat_t - v_t)||^2`` (expected over calibration prompts) is
minimized. The objective is a quadratic form in the mean velocity Gram matrix,
so the solve is closed-form; the emitted candidate replays the frozen table via
CalibratedLinearPredictor with an unchanged runtime operator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from difflet.offline.cache_profile.schedule import scheduler_sigmas
from difflet.pipeline.cache.profile import canonical_sha256, load_phased_candidate

RIDGE = 1e-8


def _mean_velocity_gram(quality_input: dict, base_dir: Path, delta_sigma: np.ndarray):
    import torch

    paths = sorted(
        {comparison["baseline"]["trajectory"] for comparison in quality_input["comparisons"]}
    )
    total = None
    for relative in paths:
        states = torch.load(base_dir / relative, map_location="cpu", weights_only=True).float()
        # States are x_1..x_N, so differences recover dsigma_k * v_k for k = 1..N-1.
        diffs = (states[1:] - states[:-1]).reshape(states.shape[0] - 1, -1).numpy()
        velocities = diffs.astype(np.float64) / delta_sigma[1 : states.shape[0]].reshape(-1, 1)
        gram = velocities @ velocities.T
        total = gram if total is None else total + gram
    return total / len(paths), len(paths)


def _skipped_segments(anchors: list[int], num_steps: int):
    anchor_set = set(anchors)
    last_two: list[int] = []
    for step in range(num_steps):
        if step in anchor_set:
            last_two = (last_two + [step])[-2:]
        elif len(last_two) == 2:
            yield last_two[0], last_two[1], step


def solve_global_weights(
    mean_gram: np.ndarray, anchors: list[int], num_steps: int, delta_sigma: np.ndarray
) -> dict[int, list[list[float]]]:
    """Jointly minimize the final-deviation quadratic over all reuse weights."""

    # Velocity k occupies gram row k-1 (states are x_1..x_N).
    triples = list(_skipped_segments(anchors, num_steps))
    n_var = 2 * len(triples)
    design = np.zeros((mean_gram.shape[0], n_var))
    offset = np.zeros(mean_gram.shape[0])
    for j, (a, b, t) in enumerate(triples):
        design[a - 1, 2 * j] = delta_sigma[t]
        design[b - 1, 2 * j + 1] = delta_sigma[t]
        offset[t - 1] -= delta_sigma[t]
    system = design.T @ mean_gram @ design
    system[np.diag_indices(n_var)] += RIDGE * np.trace(system) / n_var
    solution = np.linalg.solve(system, -(design.T @ mean_gram @ offset))
    return {
        t: [[a, solution[2 * j]], [b, solution[2 * j + 1]]]
        for j, (a, b, t) in enumerate(triples)
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-input", required=True, type=Path)
    parser.add_argument("--base-candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    base = load_phased_candidate(args.base_candidate)
    if base.policy["type"] != "phased_static":
        raise SystemExit("calibrated weights require a phased_static base candidate")
    base_document = json.loads(args.base_candidate.read_text(encoding="utf-8"))
    quality_input = json.loads(args.quality_input.read_text(encoding="utf-8"))
    generation = quality_input["protocol"]["generation"]
    num_steps = int(base.policy["num_steps"])
    if int(generation["num_steps"]) != num_steps:
        raise SystemExit("quality input and base candidate disagree on num_steps")

    sigmas = np.array(scheduler_sigmas(generation))
    delta_sigma = np.diff(sigmas)
    mean_gram, trajectory_count = _mean_velocity_gram(
        quality_input, args.quality_input.parent, delta_sigma
    )
    anchors = list(base.policy["static_anchor_steps"])
    weights = solve_global_weights(mean_gram, anchors, num_steps, delta_sigma)

    payload = {
        key: value for key, value in base_document.items() if key != "sha256"
    }
    payload["candidate_id"] = f"{base.candidate_id}-calibrated-global"
    payload["predictor"] = {
        "type": "calibrated_linear",
        "coord": "index",
        "weights": {str(step): entry for step, entry in sorted(weights.items())},
    }
    document = {**payload, "sha256": canonical_sha256(payload)}
    args.out.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    load_phased_candidate(args.out)  # fail closed before reporting success
    print(
        f"wrote {args.out} (candidate {payload['candidate_id']}, "
        f"{len(weights)} skipped steps, {trajectory_count} calibration trajectories)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
