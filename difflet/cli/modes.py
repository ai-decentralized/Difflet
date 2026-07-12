"""Runtime-mode table: latency / throughput / mixed → per-model-class dp/cfg/cp.

Spec: docs/superpowers/specs/2026-07-06-dp-replication-routing-design.md.
Rules (encoded, not per-cell): distilled ⇒ cfg forced 1; LTX-2 ⇒ cp capped 1;
distilled mixed backfills the freed cfg lane with cp=2. When HunyuanVideo-1.5 /
Qwen-Image gain true CFG, only their MODEL_CLASS entry flips.
"""

from __future__ import annotations

from dataclasses import dataclass

MODES = ("latency", "throughput", "mixed")

MODEL_CLASS: dict[str, str] = {
    "black-forest-labs/FLUX.1-dev": "distilled",
    "hunyuanvideo-community/HunyuanVideo": "distilled",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": "distilled",
    "Qwen/Qwen-Image": "distilled",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "true_cfg",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": "true_cfg",
    "Lightricks/LTX-2": "true_cfg_no_cp",
}


@dataclass(frozen=True)
class ModeConfig:
    dp: int
    cfg_parallel: bool
    cp_degree: int


_BASE = {
    ("latency", "distilled"): ModeConfig(dp=1, cfg_parallel=False, cp_degree=4),
    ("latency", "true_cfg"): ModeConfig(dp=1, cfg_parallel=True, cp_degree=1),
    ("throughput", "distilled"): ModeConfig(dp=4, cfg_parallel=False, cp_degree=1),
    ("throughput", "true_cfg"): ModeConfig(dp=4, cfg_parallel=False, cp_degree=1),
    ("mixed", "distilled"): ModeConfig(dp=2, cfg_parallel=False, cp_degree=2),
    ("mixed", "true_cfg"): ModeConfig(dp=2, cfg_parallel=True, cp_degree=1),
}


def resolve_mode(model_id: str, mode: str | None, args) -> ModeConfig | None:
    if mode is None:
        return None
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    if model_id not in MODEL_CLASS:
        raise ValueError(f"unknown model {model_id!r}")
    klass = MODEL_CLASS[model_id]
    base = _BASE[(mode, "true_cfg" if klass == "true_cfg_no_cp" else klass)]
    cp = 1 if klass == "true_cfg_no_cp" else base.cp_degree
    return ModeConfig(
        dp=args.dp if getattr(args, "dp", None) is not None else base.dp,
        cfg_parallel=True if getattr(args, "cfg_parallel", False) else base.cfg_parallel,
        cp_degree=args.cp_degree if getattr(args, "cp_degree", None) is not None else cp,
    )
