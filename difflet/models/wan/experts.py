"""Wan 2.2 A14B's two experts, read from the checkpoint. Torch-free (the CLI driver's checks).

A two-expert checkpoint holds a high-noise expert (``transformer``) for timesteps at or above
``boundary_ratio`` x ``num_train_timesteps`` (``model_index.json``) and a low-noise expert
(``transformer_2``) below it, the rule of diffusers' WanPipeline and of Wan2.2's own
``generate.py``. ``WanOrchestrator`` applies it (``pipeline.py``); this module answers, before
anything loads, whether a checkpoint can run that way.
"""

from __future__ import annotations

import json
import os

HIGH_NOISE = "transformer"
LOW_NOISE = "transformer_2"

#: The config fields that shape a DiT forward's inputs and output. The experts must agree on
#: these: one orchestrator feeds both the same latents and prompt embeddings.
CONTRACT_FIELDS = ("in_channels", "out_channels", "patch_size", "text_dim")


def _read_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def has_low_noise_expert(model_dir: str | os.PathLike) -> bool:
    return os.path.isfile(os.path.join(model_dir, LOW_NOISE, "config.json"))


def read_boundary_ratio(model_dir: str | os.PathLike) -> float | None:
    """The checkpoint's ``boundary_ratio`` (``model_index.json``), or None (Wan 2.1)."""
    index = _read_json(os.path.join(model_dir, "model_index.json")) or {}
    value = index.get("boundary_ratio")
    return None if value is None else float(value)


def expert_contract_diff(model_dir: str | os.PathLike) -> list[str]:
    """The ``CONTRACT_FIELDS`` on which the two experts' ``config.json`` differ, as
    ``"field (high vs low)"``. Other fields (``num_layers``, metadata) are not compared: each
    expert is built from its own config."""
    high = _read_json(os.path.join(model_dir, HIGH_NOISE, "config.json")) or {}
    low = _read_json(os.path.join(model_dir, LOW_NOISE, "config.json")) or {}
    return [f"{name} ({high.get(name)!r} vs {low.get(name)!r})"
            for name in CONTRACT_FIELDS if high.get(name) != low.get(name)]


def dual_expert_checkpoint_misuse(model_dir: str | os.PathLike) -> str | None:
    """Why ``model_dir`` cannot run both experts, else None: without ``transformer_2`` or
    ``boundary_ratio`` the run would silently use the high-noise expert for every step."""
    if not has_low_noise_expert(model_dir):
        return (f"--dual-expert needs a two-expert checkpoint (Wan 2.2 A14B): {model_dir} has no "
                f"{LOW_NOISE}/config.json")
    if read_boundary_ratio(model_dir) is None:
        return (f"--dual-expert needs the checkpoint's boundary_ratio (model_index.json) to "
                f"switch experts: {model_dir} has none")
    differ = expert_contract_diff(model_dir)
    if differ:
        return f"--dual-expert: {HIGH_NOISE} and {LOW_NOISE} differ in " + ", ".join(differ)
    return None


__all__ = [
    "CONTRACT_FIELDS",
    "HIGH_NOISE",
    "LOW_NOISE",
    "dual_expert_checkpoint_misuse",
    "expert_contract_diff",
    "has_low_noise_expert",
    "read_boundary_ratio",
]
