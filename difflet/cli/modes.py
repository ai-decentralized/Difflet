"""Runtime-mode table: latency / throughput / mixed → per-model-class dp/cfg/cp.

Spec: docs/superpowers/specs/2026-07-06-dp-replication-routing-design.md.
Rules (encoded, not per-cell): distilled ⇒ cfg forced 1; LTX-2 ⇒ cp capped 1;
distilled mixed backfills the freed cfg lane with cp=2. When HunyuanVideo-1.5 /
Qwen-Image gain true CFG, only their registry ``ModelCapabilities`` flips.

The dp/cp degrees below assume the four-core trn2.3xlarge. They are a static
preset, not a plan: they do not consult the host's core count or the model's
head count. ``difflet/planner/`` supersedes them.
"""

from __future__ import annotations

from dataclasses import dataclass

MODES = ("latency", "throughput", "mixed")


def model_class(model_id: str) -> str:
    """Classify a model by the parallel lanes its guidance path can use.

    Derived from ``difflet.registry`` rather than tabulated here, so a new model
    is classified by the capabilities it declares once:

    - ``distilled``      -- no second CFG branch, so the cfg lane is free for cp
    - ``true_cfg``       -- two CFG branches and context parallelism available
    - ``true_cfg_no_cp`` -- two CFG branches but no cp to fall back on (LTX-2)
    """

    from difflet.registry import resolve_model

    capabilities = resolve_model(model_id).require_capabilities()
    if capabilities.is_distilled:
        return "distilled"
    return "true_cfg" if capabilities.supports_cp else "true_cfg_no_cp"


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
    try:
        klass = model_class(model_id)
    except ValueError as exc:
        raise ValueError(f"unknown model {model_id!r}") from exc
    base = _BASE[(mode, "true_cfg" if klass == "true_cfg_no_cp" else klass)]
    cp = 1 if klass == "true_cfg_no_cp" else base.cp_degree
    return ModeConfig(
        dp=args.dp if getattr(args, "dp", None) is not None else base.dp,
        cfg_parallel=True if getattr(args, "cfg_parallel", False) else base.cfg_parallel,
        cp_degree=args.cp_degree if getattr(args, "cp_degree", None) is not None else cp,
    )
