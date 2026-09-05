"""Analytic per-step cost model for a parallel configuration.

Why analytic at all: every configuration is a distinct compile-cache key, and one
AOT compile costs 1484s for Flux at 1024x1024. Measuring five candidates is a
two-hour errand, so the default path has to predict. Measurements, where they
exist, override predictions rather than seeding a search.

The model is deliberately *relative*. Rather than estimate absolute FLOPs and
device throughput -- both of which are guesses on this accelerator -- it fits a
single scalar ``compute_seconds_single_core`` to a real measurement of the same
model and shape, then extrapolates across configurations using terms whose
*ratios* are known from the code:

    T_step = C * compute_share(config) + comm_bytes(config) / bandwidth

``compute_share`` and ``comm_bytes`` are derived below from what each strategy
actually does. With two or more measurements at the same (model, shape, host),
``bandwidth`` is fitted too, at which point nothing in the prediction rests on an
assumed constant. That is why the benchmark plan asks for more than one
configuration per model.

Everything the model cannot know is named and defaulted in one place, and
:class:`Prediction` carries its own provenance so the CLI never presents a fitted
number and a guessed one as equals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.planner.measurements import Measurement
from difflet.planner.model_profile import ModelProfile, SequenceLengths

# --------------------------------------------------------------------- constants

BF16_BYTES = 2

# Share of per-step compute in the regions Megatron-style SP shards: the norms,
# modulation/AdaLN projections, and residual adds that are otherwise replicated
# across the tensor-parallel group. Everything else already scales with tp.
# An estimate; SP's entire predicted benefit in this model rides on it, because
# SP is communication-neutral (see comm_bytes).
REPLICATED_COMPUTE_SHARE = 0.08

# Tensor parallelism shrinks each core's GEMM tiles, so speedup is sublinear.
# Modeled as a mild logarithmic tax rather than a fitted curve.
TP_EFFICIENCY_TAX = 0.03
CP_EFFICIENCY_TAX = 0.015

# Ring attention moves the same bytes as gather-KV but overlaps the rotation with
# compute, so only part of it lands on the critical path. Estimated.
RING_OVERLAP_RETENTION = 0.7

# Effective bytes/second for a collective inside one Neuron device. An
# assumption, not a measurement -- it is replaced by a fitted value as soon as
# two measurements exist for a (model, shape, host). Predictions that still rest
# on it are reported as such.
ASSUMED_COLLECTIVE_BYTES_PER_SECOND = 100e9

# Fallback throughput used only when a model has no measurement at all. Flagged
# as uncalibrated wherever it is used.
ASSUMED_CORE_FLOPS = 1.6e14
ASSUMED_MFU = 0.35


@dataclass(frozen=True)
class CommBreakdown:
    """Bytes each strategy moves per rank per denoise step."""

    tensor_parallel: float = 0.0
    context_parallel: float = 0.0
    cfg_parallel: float = 0.0

    @property
    def total(self) -> float:
        return self.tensor_parallel + self.context_parallel + self.cfg_parallel


@dataclass(frozen=True)
class Calibration:
    """What the fit produced, and how much to trust it."""

    compute_seconds_single_core: float
    bandwidth_bytes_per_second: float
    # measured-fit | measured-anchor | uncalibrated
    kind: str
    anchor_labels: tuple[str, ...] = ()

    @property
    def bandwidth_is_assumed(self) -> bool:
        return self.kind != "measured-fit"


@dataclass(frozen=True)
class Prediction:
    step_seconds: float
    compute_seconds: float
    comm_seconds: float
    comm: CommBreakdown
    # measured | predicted | predicted-uncalibrated
    evidence: str
    measurement: Measurement | None = None


def compute_share(parallel: DiffletParallelConfig) -> float:
    """Per-step compute for one rank, relative to a single core doing all of it.

    Three effects, in the order they apply:

    - **CFG parallelism** halves the work: each rank runs one of the two guidance
      branches, so per-rank batch is 1 instead of 2.
    - **TP and CP** shard the per-token work. CP shards the sequence, so it
      shrinks *everything* per-token including the norms; TP shards only the
      layer-internal work unless SP is on.
    - **SP** extends the TP division to the otherwise-replicated
      norm/modulation/residual regions.

    The efficiency tax reflects that halving a GEMM does not halve its time once
    the tiles get small.
    """

    tp, cp = parallel.tp_degree, parallel.cp_degree
    cfg_div = 2.0 if parallel.cfg_parallel_enabled else 1.0
    replicated_div = float(tp) if parallel.sp_enabled else 1.0

    sharded = (1.0 - REPLICATED_COMPUTE_SHARE) / (tp * cp)
    replicated = REPLICATED_COMPUTE_SHARE / (cp * replicated_div)
    tax = 1.0 + TP_EFFICIENCY_TAX * math.log2(tp) + CP_EFFICIENCY_TAX * math.log2(cp)
    return (sharded + replicated) * tax / cfg_div


def comm_bytes(
    parallel: DiffletParallelConfig,
    *,
    profile: ModelProfile,
    seq: SequenceLengths,
    latent_channels: int = 64,
) -> CommBreakdown:
    """Bytes one rank moves per denoise step, per strategy.

    **Tensor parallel.** Two all-reduces per block -- after the attention output
    projection and after the MLP down projection -- each over a
    ``[B, S/cp, hidden]`` activation. A ring all-reduce moves ``2(tp-1)/tp`` of
    its payload per rank.

    With SP the row-parallel all-reduce becomes a reduce-scatter plus a later
    all-gather: ``(tp-1)/tp`` each, summing to exactly the same ``2(tp-1)/tp``.
    **SP is communication-neutral in this model**, which is the honest reading of
    what the code does; its predicted win comes only from the compute term.

    **Context parallel.** Per attention, with local K/V of
    ``L = B * (H/tp) * (S/cp) * d``:

    - ``gather_kv`` all-gathers K and V: ``2 * (cp-1) * L``.
    - ``ring`` moves the same volume but pipelines it against compute, so only
      ``RING_OVERLAP_RETENTION`` of it is charged.
    - ``ulysses`` does four all-to-alls (q, k, v out, then the inverse on the
      attention output): ``4 * (cp-1)/cp * L``.

    So ulysses/gather-KV is ``2/cp``: a tie at cp=2, half at cp=4, a quarter at
    cp=8. The Ulysses design doc's "a factor of cp less" compares
    ``O(S*H*d)`` against ``O(S/cp*H*d)`` and drops both the ``(cp-1)/cp`` factors
    and the tensor counts; this accounting keeps them.

    **CFG parallel.** One all-gather of the output latent per step over an axis
    of size 2. Small, but not zero.
    """

    dims = profile.dims
    tp, cp = parallel.tp_degree, parallel.cp_degree
    tokens_per_rank = seq.joint / cp

    tp_bytes = 0.0
    if tp > 1:
        activation = tokens_per_rank * dims.hidden_size * BF16_BYTES
        per_all_reduce = 2.0 * (tp - 1) / tp * activation
        tp_bytes = dims.total_blocks * 2.0 * per_all_reduce

    cp_bytes = 0.0
    if cp > 1:
        heads_per_rank = dims.num_attention_heads / tp
        local_kv = tokens_per_rank * heads_per_rank * dims.attention_head_dim * BF16_BYTES
        if parallel.cp_mode == "ulysses":
            per_attention = 4.0 * (cp - 1) / cp * local_kv
        else:
            per_attention = 2.0 * (cp - 1) * local_kv
            if parallel.cp_mode == "ring":
                per_attention *= RING_OVERLAP_RETENTION
        cp_bytes = dims.total_blocks * per_attention

    cfg_bytes = 0.0
    if parallel.cfg_parallel_enabled:
        cfg_bytes = seq.image * latent_channels * BF16_BYTES

    return CommBreakdown(
        tensor_parallel=tp_bytes, context_parallel=cp_bytes, cfg_parallel=cfg_bytes
    )


def calibrate(
    anchors: tuple[Measurement, ...],
    *,
    profile: ModelProfile,
    seq: SequenceLengths,
    parallel_of,
) -> Calibration:
    """Fit the model's free parameters to whatever measurements exist.

    - **Two or more anchors**: least-squares fit of both ``C`` and ``1/bandwidth``
      over ``T_i = C * share_i + bytes_i / bandwidth``. Nothing rests on an
      assumed constant.
    - **One anchor**: solve ``C`` with bandwidth assumed. If the assumed
      bandwidth already accounts for the whole measured step, fall back to
      attributing it all to compute rather than emitting a negative ``C``.
    - **None**: a nominal ``C`` from a parameter-count estimate, flagged
      ``uncalibrated``.

    ``parallel_of`` maps a measurement's label back to its
    ``DiffletParallelConfig``; the caller owns that mapping because it is the one
    holding the feasibility report.
    """

    rows: list[tuple[float, float, float, str]] = []
    for anchor in anchors:
        parallel = parallel_of(anchor.label)
        if parallel is None or not anchor.step_latency_seconds:
            continue
        share = compute_share(parallel)
        total_bytes = comm_bytes(parallel, profile=profile, seq=seq).total
        rows.append((share, total_bytes, float(anchor.step_latency_seconds), anchor.label))

    if len(rows) >= 2:
        fitted = _least_squares(rows)
        if fitted is not None:
            compute, inv_bandwidth = fitted
            return Calibration(
                compute_seconds_single_core=compute,
                bandwidth_bytes_per_second=1.0 / inv_bandwidth,
                kind="measured-fit",
                anchor_labels=tuple(row[3] for row in rows),
            )

    if rows:
        share, total_bytes, measured, label = rows[0]
        comm_seconds = total_bytes / ASSUMED_COLLECTIVE_BYTES_PER_SECOND
        compute = (measured - comm_seconds) / share
        if compute <= 0:
            compute = measured / share
        return Calibration(
            compute_seconds_single_core=compute,
            bandwidth_bytes_per_second=ASSUMED_COLLECTIVE_BYTES_PER_SECOND,
            kind="measured-anchor",
            anchor_labels=(label,),
        )

    return Calibration(
        compute_seconds_single_core=_nominal_compute_seconds(profile, seq),
        bandwidth_bytes_per_second=ASSUMED_COLLECTIVE_BYTES_PER_SECOND,
        kind="uncalibrated",
    )


def predict(
    parallel: DiffletParallelConfig,
    *,
    profile: ModelProfile,
    seq: SequenceLengths,
    calibration: Calibration,
    measurement: Measurement | None = None,
) -> Prediction:
    """Per-step time for one configuration, measured if possible."""

    breakdown = comm_bytes(parallel, profile=profile, seq=seq)
    comm_seconds = breakdown.total / calibration.bandwidth_bytes_per_second
    compute_seconds = calibration.compute_seconds_single_core * compute_share(parallel)

    if measurement is not None and measurement.step_latency_seconds:
        return Prediction(
            step_seconds=float(measurement.step_latency_seconds),
            compute_seconds=compute_seconds,
            comm_seconds=comm_seconds,
            comm=breakdown,
            evidence="measured",
            measurement=measurement,
        )

    evidence = "predicted-uncalibrated" if calibration.kind == "uncalibrated" else "predicted"
    return Prediction(
        step_seconds=compute_seconds + comm_seconds,
        compute_seconds=compute_seconds,
        comm_seconds=comm_seconds,
        comm=breakdown,
        evidence=evidence,
    )


def _least_squares(rows: list[tuple[float, float, float, str]]) -> tuple[float, float] | None:
    """Fit ``T = C * share + bytes * inv_bw`` with both coefficients non-negative.

    Two unknowns and a handful of points, so the normal equations are solved
    directly. A degenerate system (all anchors sharing a configuration shape) or
    a fit that wants a negative coefficient means the data cannot support the
    two-parameter model; return ``None`` and let the caller fall back to the
    single-anchor path rather than report a nonsense bandwidth.
    """

    sxx = sum(share * share for share, _, _, _ in rows)
    sxy = sum(share * byte for share, byte, _, _ in rows)
    syy = sum(byte * byte for _, byte, _, _ in rows)
    sxt = sum(share * time for share, _, time, _ in rows)
    syt = sum(byte * time for _, byte, time, _ in rows)

    determinant = sxx * syy - sxy * sxy
    if abs(determinant) < 1e-30:
        return None
    compute = (sxt * syy - syt * sxy) / determinant
    inv_bandwidth = (syt * sxx - sxt * sxy) / determinant
    if compute <= 0 or inv_bandwidth <= 0:
        return None
    return compute, inv_bandwidth


def _nominal_compute_seconds(profile: ModelProfile, seq: SequenceLengths) -> float:
    """Order-of-magnitude single-core step time from parameter count.

    Only reached for a model with no measurements at all. A transformer block is
    roughly ``12 * hidden^2`` parameters (4h^2 attention, 8h^2 MLP), a forward
    pass is ~2 FLOPs per parameter per token, and attention adds ``4 * S^2 * h``
    per block. Every constant here is nominal, which is why callers surface this
    as ``predicted-uncalibrated`` rather than as a time.
    """

    dims = profile.dims
    hidden = dims.hidden_size
    params = 12.0 * hidden * hidden * dims.total_blocks
    dense_flops = 2.0 * params * seq.joint
    attention_flops = 4.0 * seq.joint * seq.joint * hidden * dims.total_blocks
    return (dense_flops + attention_flops) / (ASSUMED_CORE_FLOPS * ASSUMED_MFU)
