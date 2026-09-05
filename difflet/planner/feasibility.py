"""Enumerate the parallel configurations a given model and host can actually run.

The search space is small enough to enumerate outright. ``world_size`` must equal
the available NeuronCore count and every axis divides it, so a four-core host has
single-digit legal configurations and even a 64-core one has a few dozen. No ILP,
no dynamic programming, no pruning heuristics -- generate the lattice, apply the
rules, and keep what survives.

The rules come from four places and this module deliberately re-uses rather than
re-states them:

1. **Mutual exclusion** -- by *constructing* a ``DiffletParallelConfig`` and
   catching ``ValueError``. Restating "cp>1 excludes cfg" here would create a
   second copy that can drift from what the runtime enforces; asking the real
   type means the planner can never propose something the pipeline would reject.
2. **Model capability** -- ``ModelCapabilities`` from the registry (P1).
3. **Divisibility** -- ``heads % tp`` and, for ulysses,
   ``heads % (tp * cp)``. The latter is documented in ``DiffletParallelConfig``
   but enforced only by ``_ulysses_check_heads`` inside the attention layer, so
   before this module a bad ulysses config was discovered at compile time.
4. **Known-bad cells** -- configurations that are legal and still fail, from
   ``scripts/verify_cli.py``'s expected-failure table. Reported separately from
   infeasible ones: "the compiler crashes on this" is different information from
   "this configuration is meaningless", and only the first can be fixed upstream.
5. **Memory infeasibility** -- for models whose whole footprint is co-resident
   on the device (``staged=False``: Flux and LTX-2 load as one pipeline), the
   ``dp * cfg * cp`` replicated weight bytes are a physical upper bound on
   device HBM. A candidate over that bound cannot run, so it is rejected
   outright rather than flagged (D35: the 2026-08-20 traversal measured the
   planner recommending exactly such a config -- flux tp1cp4 at ~135 GB
   against a 96 GB device). Staged models keep the advisory-only treatment
   (D22): their components are never all resident, so the same arithmetic
   over-counts and must not reject.
"""

from __future__ import annotations

from dataclasses import dataclass

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.registry import ModelCapabilities

# Configurations that pass every rule above and still do not work, keyed by
# (registry model name, candidate label). A ``None`` label denies every
# configuration for that model. Sourced from EXPECTED_FAIL_CELLS in
# scripts/verify_cli.py -- keep the two in step when a cell is fixed.
KNOWN_BAD: dict[tuple[str, str | None], str] = {
    ("hunyuan_video", "tp2cp2"): (
        "neuronx-cc 2.25.3371 dies with NCC_INLA001/NCC_IBIR243 on the CP-2 DiT graph"
    ),
    ("hunyuan_video", "dp2tp2"): (
        "a single replica's weights exceed one Trainium2 chip's HBM at dp=2"
    ),
    ("hunyuan_video_15", None): (
        "HunyuanVideo 1.5 is a scaffold: only `difflet download` is implemented"
    ),
}

# Ulysses all-to-alls the sequence shard into a head shard and that kernel has
# no path for an attention mask; HunyuanVideo always carries one (its Llama
# text encoder emits padded, variable-length sequences). Mirrors
# ULYSSES_UNSUPPORTED in scripts/verify_cli.py -- keep the two in step.
# Qwen-Image reaches the branch with attention_mask=None and is unaffected.
ULYSSES_UNSUPPORTED: frozenset[str] = frozenset({"hunyuan_video"})

# Serving pins that are not expressible as ModelCapabilities, from
# difflet/serving/options.py and the resident adapters.
_SERVING_REJECTS_CFG_PARALLEL = "serving does not expose the true-CFG request path"
_SERVING_QWEN_CP = "Qwen-Image serving requires cp_degree=1"


@dataclass(frozen=True)
class Candidate:
    """A parallel configuration that survived every feasibility rule."""

    parallel: DiffletParallelConfig

    @property
    def label(self) -> str:
        return config_label(self.parallel)

    @property
    def world_size(self) -> int:
        return self.parallel.world_size


@dataclass(frozen=True)
class Rejection:
    """A configuration that was ruled out, and why.

    ``kind`` groups reasons so the CLI can order them usefully and so callers can
    distinguish "never going to work" from "broken today".
    """

    label: str
    reason: str
    kind: str  # exclusivity | capability | divisibility | degenerate | serving | known-bad


@dataclass(frozen=True)
class FeasibilityReport:
    model_name: str
    cores: int
    feasible: tuple[Candidate, ...]
    rejected: tuple[Rejection, ...]

    def labels(self) -> tuple[str, ...]:
        return tuple(candidate.label for candidate in self.feasible)


def config_label(parallel: DiffletParallelConfig) -> str:
    """A short, stable name for a configuration.

    Matches the keys ``scripts/verify_cli.py`` uses for its matrix (``tp4``,
    ``tp2cp2``, ``tp2cfg``, ``tp4sp``, ``dp2tp2``, ``tp2cp2ulysses``) so the two
    can be compared cell by cell.
    """

    parts: list[str] = []
    if parallel.dp_degree > 1:
        parts.append(f"dp{parallel.dp_degree}")
    parts.append(f"tp{parallel.tp_degree}")
    if parallel.cp_degree > 1:
        parts.append(f"cp{parallel.cp_degree}")
    if parallel.cfg_parallel_enabled:
        parts.append("cfg")
    if parallel.sp_enabled:
        parts.append("sp")
    if parallel.cp_degree > 1 and parallel.cp_mode != "gather_kv":
        parts.append(parallel.cp_mode)
    return "".join(parts)


def divisors(value: int) -> tuple[int, ...]:
    return tuple(d for d in range(1, value + 1) if value % d == 0)


def enumerate_candidates(
    *,
    model_name: str,
    capabilities: ModelCapabilities,
    cores: int,
    serving: bool = False,
    max_dp: int | None = None,
    weight_bytes: int = 0,
    weight_budget: int = 0,
    staged: bool = True,
) -> FeasibilityReport:
    """All configurations that exactly fill ``cores``, split into kept and rejected.

    Only configurations whose ``world_size`` equals ``cores`` are considered.
    Under-filling the host is legal but never what anyone wants -- it leaves
    cores idle -- and enumerating those would bury the useful candidates.

    Rejections are recorded only for configurations that *do* fill the host, so
    the report explains real choices rather than arithmetic. Each label is
    reported once, with the first reason found; the check order runs from most
    fundamental (mutual exclusion) to most contingent (a compiler bug).

    ``weight_bytes``/``weight_budget``/``staged`` drive the memory rule: when
    the model is not staged (whole footprint co-resident) and its replicated
    weight bytes cannot fit the budget, the candidate is rejected as
    physically unrunnable. Zero for either byte count disables the rule, which
    keeps callers without weight data (tests, out-of-tree models) on the old
    behavior.
    """

    if cores < 1:
        raise ValueError(f"cores must be >= 1, got {cores}")

    feasible: list[Candidate] = []
    rejected: dict[str, Rejection] = {}
    heads = capabilities.num_attention_heads

    for parallel in _lattice(cores=cores, capabilities=capabilities, max_dp=max_dp):
        label = config_label(parallel)
        if label in rejected:
            continue
        reason = _reject(
            parallel,
            model_name=model_name,
            capabilities=capabilities,
            heads=heads,
            serving=serving,
            weight_bytes=weight_bytes,
            weight_budget=weight_budget,
            staged=staged,
        )
        if reason is None:
            if not any(existing.label == label for existing in feasible):
                feasible.append(Candidate(parallel))
        else:
            rejected[label] = reason

    feasible.sort(key=lambda candidate: _sort_key(candidate.parallel))
    order = {
        "exclusivity": 0,
        "capability": 1,
        "divisibility": 2,
        "degenerate": 3,
        "serving": 4,
        "memory": 5,
        "known-bad": 6,
    }
    ordered = sorted(rejected.values(), key=lambda r: (order.get(r.kind, 9), r.label))
    return FeasibilityReport(
        model_name=model_name,
        cores=cores,
        feasible=tuple(feasible),
        rejected=tuple(ordered),
    )


def _lattice(
    *, cores: int, capabilities: ModelCapabilities, max_dp: int | None
) -> list[DiffletParallelConfig]:
    """Every (dp, cfg, cp, tp, sp, cp_mode) tuple whose world_size fills the host.

    Tuples that ``DiffletParallelConfig`` itself rejects are yielded as the
    exception they raise, not skipped -- ``_reject`` turns that into a reported
    reason so the user sees *why* an obvious-looking config is unavailable.
    """

    # Offer every mode the model wires, plus gather_kv as the cp==1 placeholder.
    cp_modes = sorted(capabilities.cp_modes) or ["gather_kv"]
    out: list = []
    for dp in divisors(cores):
        if max_dp is not None and dp > max_dp:
            continue
        for cfg in (1, 2):
            for cp in divisors(cores):
                for tp in divisors(cores):
                    if dp * cfg * cp * tp != cores:
                        continue
                    for cp_mode in cp_modes if cp > 1 else ["gather_kv"]:
                        out.append(_build(tp=tp, cp=cp, cfg=cfg, dp=dp, sp=False, cp_mode=cp_mode))
                    # SP is only *legal* at cp==1, but emitting one cp>1 variant
                    # per (tp, cp) means the report explains the exclusion
                    # instead of silently omitting it. One representative is
                    # enough: repeating it per cp_mode would be pure noise.
                    out.append(_build(tp=tp, cp=cp, cfg=cfg, dp=dp, sp=True, cp_mode="gather_kv"))
    return out


def _build(*, tp: int, cp: int, cfg: int, dp: int, sp: bool, cp_mode: str):
    """Construct the real config, or an ``_ExcludedConfig`` if the type refuses it.

    The stand-in carries the runtime's own error message forward, so the
    rejection the user reads is the one the pipeline would have raised.
    """

    try:
        return DiffletParallelConfig(
            tp_degree=tp,
            cp_degree=cp,
            cfg_parallel_enabled=cfg == 2,
            cp_mode=cp_mode,
            sp_enabled=sp,
            dp_degree=dp,
        )
    except ValueError as exc:
        return _ExcludedConfig(
            tp_degree=tp,
            cp_degree=cp,
            cfg_parallel_enabled=cfg == 2,
            cp_mode=cp_mode,
            sp_enabled=sp,
            dp_degree=dp,
            error=str(exc),
        )


@dataclass(frozen=True)
class _ExcludedConfig:
    """Stand-in for a tuple ``DiffletParallelConfig.__post_init__`` refused.

    Mirrors the field names so ``config_label`` works on it unchanged.
    """

    tp_degree: int
    cp_degree: int
    cfg_parallel_enabled: bool
    cp_mode: str
    sp_enabled: bool
    dp_degree: int
    error: str

    @property
    def world_size(self) -> int:
        cfg = 2 if self.cfg_parallel_enabled else 1
        return self.dp_degree * cfg * self.cp_degree * self.tp_degree


def _reject(
    parallel,
    *,
    model_name: str,
    capabilities: ModelCapabilities,
    heads: int,
    serving: bool,
    weight_bytes: int = 0,
    weight_budget: int = 0,
    staged: bool = True,
) -> Rejection | None:
    label = config_label(parallel)

    if isinstance(parallel, _ExcludedConfig):
        return Rejection(label, parallel.error, "exclusivity")

    if parallel.cfg_parallel_enabled and not capabilities.supports_cfg_parallel:
        return Rejection(
            label,
            "guidance-distilled: one forward pass with guidance baked into the "
            "timestep embedding, so there is no second CFG branch to split",
            "capability",
        )
    if parallel.cp_degree > 1 and not capabilities.supports_cp:
        return Rejection(label, "model does not wire context parallelism", "capability")
    if parallel.cp_degree > 1 and parallel.cp_mode not in capabilities.cp_modes:
        return Rejection(label, f"model does not wire cp_mode={parallel.cp_mode!r}", "capability")
    if parallel.cp_mode == "ulysses" and model_name in ULYSSES_UNSUPPORTED:
        return Rejection(
            label,
            "ulysses attention has no kernel path for this model's attention mask",
            "capability",
        )
    if parallel.sp_enabled and not capabilities.supports_sp:
        return Rejection(label, "model does not wire sequence parallelism", "capability")
    if parallel.sp_enabled and parallel.tp_degree == 1:
        return Rejection(
            label,
            "SP shards the norm/modulation/residual regions across the "
            "tensor-parallel group; at tp=1 that group has one member, so it is "
            "a no-op that only adds collectives",
            "degenerate",
        )

    if heads % parallel.tp_degree:
        return Rejection(
            label,
            f"tp_degree={parallel.tp_degree} does not divide {heads} attention heads",
            "divisibility",
        )
    if parallel.cp_mode == "ulysses":
        shard = parallel.tp_degree * parallel.cp_degree
        if heads % shard:
            return Rejection(
                label,
                f"ulysses shards heads over cp on top of the TP head shard, so it "
                f"needs heads % (tp * cp) == 0: {heads} % {shard} != 0",
                "divisibility",
            )

    if weight_bytes and weight_budget and not staged:
        copies = (
            parallel.dp_degree * (2 if parallel.cfg_parallel_enabled else 1) * parallel.cp_degree
        )
        resident = copies * weight_bytes
        if resident > weight_budget:
            return Rejection(
                label,
                f"{copies} resident copies of the {weight_bytes / 1e9:.1f} GB footprint "
                f"need ~{resident / 1e9:.0f} GB of device HBM, over the "
                f"{weight_budget / 1e9:.0f} GB budget -- the whole pipeline is "
                f"co-resident on this model, so this cannot run",
                "memory",
            )

    if serving:
        if parallel.cfg_parallel_enabled:
            return Rejection(label, _SERVING_REJECTS_CFG_PARALLEL, "serving")
        if parallel.dp_degree > 1:
            return Rejection(label, "resident serving adapters require dp_degree=1", "serving")
        if model_name == "qwen_image" and parallel.cp_degree > 1:
            return Rejection(label, _SERVING_QWEN_CP, "serving")

    known_bad = KNOWN_BAD.get((model_name, label)) or KNOWN_BAD.get((model_name, None))
    if known_bad:
        return Rejection(label, known_bad, "known-bad")

    return None


def _sort_key(parallel: DiffletParallelConfig) -> tuple:
    """Order candidates the way a reader scans them: fewest moving parts first."""

    axes = sum(
        (
            parallel.dp_degree > 1,
            parallel.cp_degree > 1,
            parallel.cfg_parallel_enabled,
            parallel.sp_enabled,
        )
    )
    return (axes, -parallel.tp_degree, parallel.cp_degree, parallel.cp_mode)
