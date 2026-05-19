"""Decode-free per-step latent metric harness (M5.0.3.2).

Candidate-aware latent-space telemetry following the metrics-JSON
convention of ``scripts/qwen_image_research_baseline.py`` /
``scripts/hunyuan_vae_parity.py`` (per-step lists + ``*_mean`` /
``*_pstdev`` summary + ``final_latents_*``). Operates purely on latent
tensors — **no VAE decode** — so it can run inside any denoise loop at
negligible cost.

It additionally records the cross-candidate divergence signal the
M5.1 latent prefix cache needs: for each step, the cosine of every
candidate against candidate 0, and a derived ``shared_prefix`` —
the leading run of steps where all candidates stay within a cosine
threshold of candidate 0 (the "candidates still share their prefix"
fraction; cclog 44 §1's ≥20%-duplication trigger is read off this).

Design note: cclog 43 §3.2 said "uplift
``nova/utils/benchmark.py:LatencyCollector``", but that file is a
verbatim NxDI fork (rebase-tracked, black-excluded). This Nova-authored
collector lives here instead, beside ``precision_schedule.py``, keeping
the M5 candidate-runtime surface together and out of the fork.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

DEFAULT_FORK_COSINE_THRESHOLD = 0.9995


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            _flat(a).unsqueeze(0), _flat(b).unsqueeze(0)
        ).item()
    )


@dataclass
class LatentMetricCollector:
    """Per-step, per-candidate latent statistics collector.

    Usage inside a denoise loop::

        coll = LatentMetricCollector(fork_cosine_threshold=0.9995)
        for step, t in enumerate(timesteps):
            latents = denoise_step(latents)          # [N, ...]
            coll.record(step, latents, reference=ref_latents)
        coll.write_json(path)

    ``latents`` is ``[N, ...]`` with ``N`` = active candidates (``N=1``
    is the legacy no-candidate case and is fully supported — the
    cross-candidate fields collapse to the trivial ``[1.0]``).
    """

    fork_cosine_threshold: float = DEFAULT_FORK_COSINE_THRESHOLD
    steps: list[dict[str, Any]] = field(default_factory=list)
    _final: torch.Tensor | None = field(default=None, repr=False)

    def record(
        self,
        step_index: int,
        latents: torch.Tensor,
        *,
        reference: torch.Tensor | None = None,
    ) -> None:
        if latents.ndim < 1:
            raise ValueError("latents must have at least a candidate dim")
        n = latents.shape[0]
        per_candidate_mean = [float(latents[i].float().mean()) for i in range(n)]
        per_candidate_std = [float(latents[i].float().std()) for i in range(n)]
        # Cross-candidate cosine vs candidate 0 (the divergence signal).
        cross = [_cosine(latents[i], latents[0]) for i in range(n)]
        min_cross = min(cross)
        entry: dict[str, Any] = {
            "step": int(step_index),
            "num_candidates": int(n),
            "candidate_mean": per_candidate_mean,
            "candidate_std": per_candidate_std,
            "cross_candidate_cosine_vs_0": cross,
            "min_cross_candidate_cosine": min_cross,
            "candidates_converged": bool(
                min_cross >= self.fork_cosine_threshold
            ),
        }
        if reference is not None:
            entry["cosine_vs_reference"] = [
                _cosine(latents[i], reference[i] if reference.shape[0] == n else reference)
                for i in range(n)
            ]
        self.steps.append(entry)
        self._final = latents.detach().cpu()

    @property
    def shared_prefix_steps(self) -> int:
        """Leading run of steps where all candidates are still converged."""

        count = 0
        for entry in self.steps:
            if entry["candidates_converged"]:
                count += 1
            else:
                break
        return count

    @property
    def shared_prefix_fraction(self) -> float:
        if not self.steps:
            return 0.0
        return self.shared_prefix_steps / len(self.steps)

    def to_metrics_dict(self) -> dict[str, Any]:
        n_steps = len(self.steps)
        min_cross = [s["min_cross_candidate_cosine"] for s in self.steps]
        metrics: dict[str, Any] = {
            "schema": "nova-latent-metrics-v1",
            "num_steps": n_steps,
            "num_candidates": self.steps[-1]["num_candidates"] if self.steps else 0,
            "fork_cosine_threshold": self.fork_cosine_threshold,
            "per_step": self.steps,
            "per_step_min_cross_candidate_cosine": min_cross,
            "min_cross_candidate_cosine_mean": mean(min_cross) if min_cross else 1.0,
            "min_cross_candidate_cosine_pstdev": (
                pstdev(min_cross) if len(min_cross) > 1 else 0.0
            ),
            "shared_prefix_steps": self.shared_prefix_steps,
            "shared_prefix_fraction": self.shared_prefix_fraction,
        }
        if self._final is not None:
            metrics["final_latents_shape"] = list(self._final.shape)
            metrics["final_latents_mean"] = float(self._final.float().mean())
            metrics["final_latents_std"] = float(self._final.float().std())
        return metrics

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_metrics_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__ = ["DEFAULT_FORK_COSINE_THRESHOLD", "LatentMetricCollector"]
