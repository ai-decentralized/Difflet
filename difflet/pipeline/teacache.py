"""TeaCache controller primitives.

The controller is deliberately pipeline-layer code. It keeps the compiled DiT
graph unchanged and only decides whether the host loop should call the
transformer or reuse a cached residual.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

CALIBRATION_SCHEMA = "difflet-m9-teacache-calibration-v1"


@dataclass(frozen=True)
class TeaCacheCalibration:
    model: str
    shape_label: str
    num_steps: int
    poly_coef: tuple[float, ...]
    threshold: float
    warmup_steps: int = 5
    cooldown_steps: int = 5
    target_speedup: float | None = None
    fit_r2: float | None = None
    n_samples: int | None = None
    mod_input_source: str = "block0_modulated_input"
    # cclog 78: when the probe decides to skip, commit to skipping this step
    # plus the next (skip_run_length - 1) steps WITHOUT re-running the probe
    # NEFF. 1 = legacy (probe every step). >1 amortizes the ~51 ms probe
    # dispatch across a run of skips, exploiting local smoothness of the
    # denoise trajectory.
    skip_run_length: int = 1
    # cclog 83: original-TeaCache decision mode. When True, the per-step
    # ``predict_delta(diff_norm)`` is accumulated and a full step runs only when
    # the running sum crosses ``threshold`` (then the accumulator resets). This
    # is the mechanism vLLM-Omni / the TeaCache paper use with a relative-L1
    # signal + a rescaling polynomial. When False (default), the legacy per-step
    # ``predict_delta < threshold`` decision is used (cclog 72-80).
    accumulate: bool = False
    # cclog 84: fixed-cadence mode. When > 0, the controller ignores the probe
    # signal entirely and skips every ``cadence``-th step inside the
    # [warmup, num_steps - cooldown) window (cadence=2 -> skip every other step).
    # Probe-free + calibration-free; used to test whether a weak-signal model is
    # better served by a uniform skip than by its (weak) adaptive controller.
    cadence: int = 0
    # cclog 91: generic online-delta mode (probe-free, calibration-free, 0-per-model).
    # When > 0, skip a step iff the PREVIOUS full step's measured output rel-L1 delta
    # is below ``online_delta_alpha * baseline`` (baseline = first post-warmup full
    # step's delta). Uses the real noise_pred trajectory the pipeline already has — no
    # per-model probe/signal. Data-driven (adaptive), unlike the blind fixed cadence;
    # it skips only genuinely-flat steps. No two skips in a row (must re-measure).
    online_delta_alpha: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TeaCacheCalibration":
        schema = data.get("schema")
        if schema != CALIBRATION_SCHEMA:
            raise ValueError(f"unsupported TeaCache calibration schema: {schema!r}")
        return cls(
            model=str(data["model"]),
            shape_label=str(data["shape_label"]),
            num_steps=int(data["num_steps"]),
            poly_coef=tuple(float(item) for item in data["poly_coef"]),
            threshold=float(data["threshold"]),
            warmup_steps=int(data.get("warmup_steps", 5)),
            cooldown_steps=int(data.get("cooldown_steps", 5)),
            target_speedup=(
                float(data["target_speedup"]) if data.get("target_speedup") is not None else None
            ),
            fit_r2=float(data["fit_r2"]) if data.get("fit_r2") is not None else None,
            n_samples=int(data["n_samples"]) if data.get("n_samples") is not None else None,
            mod_input_source=str(data.get("mod_input_source", "block0_modulated_input")),
            skip_run_length=int(data.get("skip_run_length", 1)),
            accumulate=bool(data.get("accumulate", False)),
            cadence=int(data.get("cadence", 0)),
            online_delta_alpha=float(data.get("online_delta_alpha", 0.0)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "TeaCacheCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CALIBRATION_SCHEMA,
            "model": self.model,
            "shape_label": self.shape_label,
            "num_steps": int(self.num_steps),
            "poly_coef": list(self.poly_coef),
            "threshold": float(self.threshold),
            "warmup_steps": int(self.warmup_steps),
            "cooldown_steps": int(self.cooldown_steps),
            "target_speedup": self.target_speedup,
            "fit_r2": self.fit_r2,
            "n_samples": self.n_samples,
            "mod_input_source": self.mod_input_source,
            "skip_run_length": int(self.skip_run_length),
            "accumulate": bool(self.accumulate),
            "cadence": int(self.cadence),
            "online_delta_alpha": float(self.online_delta_alpha),
        }

    def predict_delta(self, mod_input_diff_norm: float) -> float:
        x = float(mod_input_diff_norm)
        total = 0.0
        power = 1.0
        for coef in self.poly_coef:
            total += float(coef) * power
            power *= x
        return total


class TeaCacheController:
    """Host-side TeaCache state machine.

    `should_skip` only returns true after at least two full DiT steps have
    populated both `prev_noise_pred` and `cached_residual`.
    """

    def __init__(self, calibration: TeaCacheCalibration) -> None:
        self.calibration = calibration
        self.prev_mod_input: torch.Tensor | None = None
        self.prev_noise_pred: torch.Tensor | None = None
        self.cached_residual: torch.Tensor | None = None
        self.full_steps = 0
        self.skipped_steps = 0
        self.probe_calls = 0
        self.last_delta_estimate: float | None = None
        # Steps still committed to skipping without re-probing (cclog 78).
        self._skip_run_remaining = 0
        # Running sum of rescaled per-step estimates (cclog 83 accumulate mode).
        self._accum = 0.0
        # cclog 91 online-delta mode: last full step's measured output rel-L1 delta,
        # the baseline (first post-warmup full delta), and a no-two-skips-in-a-row latch.
        self._last_full_delta: float | None = None
        self._baseline_delta: float | None = None
        self._just_skipped = False

    def reset(self) -> None:
        self.prev_mod_input = None
        self.prev_noise_pred = None
        self.cached_residual = None
        self.full_steps = 0
        self.skipped_steps = 0
        self.probe_calls = 0
        self.last_delta_estimate = None
        self._skip_run_remaining = 0
        self._accum = 0.0
        self._last_full_delta = None
        self._baseline_delta = None
        self._just_skipped = False

    def needs_signal(self) -> bool:
        """Whether the controller needs the probe's per-step signal at all.

        False in fixed-cadence mode (cclog 84): the skip decision is purely
        index-based, so the pipeline can skip the probe NEFF dispatch entirely.
        Also False in online-delta mode (cclog 91): the decision uses the real
        noise_pred trajectory the pipeline already has — no per-model probe.
        """
        return (
            int(self.calibration.cadence) <= 0 and float(self.calibration.online_delta_alpha) <= 0.0
        )

    def needs_probe(self) -> bool:
        """Whether the next step requires a fresh probe NEFF call.

        Returns False while a committed skip-run is in flight — the caller can
        then skip the ~51 ms probe dispatch entirely (cclog 78).
        """
        return self._skip_run_remaining <= 0

    def should_skip(
        self,
        step_index: int,
        mod_input_now: torch.Tensor,
        *,
        diff_norm: float | None = None,
    ) -> bool:
        """Decide whether to skip this DiT step.

        ``mod_input_now`` is the modulated input tensor at the current step.
        ``diff_norm`` is the L2-norm distance between ``mod_input_now`` and the
        previous full step's ``mod_input``. When provided (cclog 72 T1 probe
        NEFF path), the controller skips its own host-side
        ``||mod_input_now - prev_mod_input||`` computation — the probe NEFF has
        already computed this on device and shipped the scalar back. When
        ``None`` (legacy / calibration path), the controller falls back to
        host-side diff against the cached ``prev_mod_input`` tensor.
        """
        step_index = int(step_index)
        if step_index < int(self.calibration.warmup_steps):
            self._skip_run_remaining = 0
            self._accum = 0.0
            return False
        if step_index >= int(self.calibration.num_steps) - int(self.calibration.cooldown_steps):
            self._skip_run_remaining = 0
            self._accum = 0.0
            return False
        if self.prev_noise_pred is None or self.cached_residual is None:
            self._skip_run_remaining = 0
            self._accum = 0.0
            return False

        # cclog 84: fixed-cadence mode — skip purely by step index, no probe/signal.
        if int(self.calibration.cadence) > 0:
            pos = step_index - int(self.calibration.warmup_steps)
            return (pos % int(self.calibration.cadence)) == (int(self.calibration.cadence) - 1)

        # cclog 91: online-delta mode — skip iff the PREVIOUS full step's measured
        # output rel-L1 delta was below alpha*baseline. Probe-free, 0-per-model,
        # data-driven. No two skips in a row (the latch forces a re-measure).
        if float(self.calibration.online_delta_alpha) > 0.0:
            if self._just_skipped:
                self._just_skipped = False
                return False
            if self._last_full_delta is None or self._baseline_delta is None:
                return False
            thresh = float(self.calibration.online_delta_alpha) * self._baseline_delta
            skip = self._last_full_delta < thresh
            self.last_delta_estimate = self._last_full_delta
            return skip

        # Committed skip-run (cclog 78): skip without a fresh probe decision.
        # The pipeline will not have run the probe this step (needs_probe()
        # returned False), so diff_norm / mod_input may be None here.
        if self._skip_run_remaining > 0:
            self._skip_run_remaining -= 1
            return True

        # The host-side prev_mod_input copy is only needed for the host-side
        # diff branch. When the device probe supplies diff_norm directly, the
        # host copy is never read — skip the requirement so the device path can
        # avoid materializing the 63 MB mod_input tensor on host every step.
        prev_mod_input = self.prev_mod_input
        if diff_norm is None and prev_mod_input is None:
            return False

        if diff_norm is None:
            import torch

            assert prev_mod_input is not None
            diff_norm_value = float(
                torch.linalg.vector_norm(
                    mod_input_now.detach().float().cpu() - prev_mod_input.float().cpu()
                ).item()
            )
        else:
            diff_norm_value = float(diff_norm)

        # cclog 83 accumulate mode (original TeaCache / vLLM-Omni): rescale the
        # per-step signal through the polynomial and accumulate; run a full step
        # only when the running sum crosses the threshold, then reset.
        if self.calibration.accumulate:
            # vLLM-Omni / original TeaCache accumulate abs(rescale_func(rel_l1)):
            # the rescaling polynomial can dip negative, and abs() keeps the
            # accumulator monotone so the threshold is always eventually crossed.
            self._accum += abs(float(self.calibration.predict_delta(diff_norm_value)))
            self.last_delta_estimate = self._accum
            if self._accum < float(self.calibration.threshold):
                return True
            self._accum = 0.0
            return False

        self.last_delta_estimate = self.calibration.predict_delta(diff_norm_value)
        skip = self.last_delta_estimate < float(self.calibration.threshold)
        if skip:
            # Commit to this skip plus the next (skip_run_length - 1) steps
            # without re-probing.
            self._skip_run_remaining = max(int(self.calibration.skip_run_length) - 1, 0)
        else:
            self._skip_run_remaining = 0
        return skip

    def skip_noise_pred(self, mod_input: torch.Tensor | None = None) -> torch.Tensor:
        if self.prev_noise_pred is None or self.cached_residual is None:
            raise RuntimeError("TeaCache skip requested before residual cache was initialized")
        self.skipped_steps += 1
        self._just_skipped = True  # online-delta: never skip two in a row (re-measure next)
        noise_pred = self.prev_noise_pred + self.cached_residual
        self.prev_noise_pred = noise_pred.detach()
        if mod_input is not None:
            self.prev_mod_input = mod_input.detach().float().cpu()
        return noise_pred

    def record_full_step(
        self, noise_pred: torch.Tensor, mod_input: torch.Tensor | None = None
    ) -> None:
        noise_pred = noise_pred.detach()
        if self.prev_noise_pred is not None:
            self.cached_residual = noise_pred - self.prev_noise_pred
            # cclog 91 online-delta: measure this full step's output rel-L1 change.
            if float(self.calibration.online_delta_alpha) > 0.0:
                prev = self.prev_noise_pred
                denom = prev.abs().mean().clamp_min(1e-8)
                self._last_full_delta = float((noise_pred - prev).abs().mean() / denom)
                if self._baseline_delta is None:
                    self._baseline_delta = self._last_full_delta
        self.prev_noise_pred = noise_pred
        self._just_skipped = False
        # In device-probe mode the caller passes mod_input=None — the device
        # probe owns prev_mod_input, so the 63 MB host copy is skipped.
        if mod_input is not None:
            self.prev_mod_input = mod_input.detach().float().cpu()
        self.full_steps += 1

    def note_probe(self) -> None:
        """Record that a probe NEFF call was made this step (for stats)."""
        self.probe_calls += 1

    def stats(self) -> dict[str, Any]:
        return {
            "full_steps": int(self.full_steps),
            "skipped_steps": int(self.skipped_steps),
            "probe_calls": int(self.probe_calls),
            "last_delta_estimate": self.last_delta_estimate,
            "cache_initialized": self.cached_residual is not None,
        }


def load_teacache_calibration_or_raise(
    path: str | Path | None,
    *,
    model: str,
    shape_label: str,
) -> TeaCacheCalibration:
    if path is None:
        raise FileNotFoundError(
            "TeaCache was requested but no calibration JSON was provided for "
            f"{model}/{shape_label}. Run scripts/calibrate_teacache.py and commit the "
            "result under cclogs/m9-teacache/ before enabling teacache_speedup."
        )
    calibration = TeaCacheCalibration.from_json(path)
    if calibration.model != model:
        raise ValueError(
            f"TeaCache calibration model mismatch: expected {model!r}, "
            f"got {calibration.model!r}"
        )
    if calibration.shape_label != shape_label:
        raise ValueError(
            "TeaCache calibration shape mismatch: "
            f"expected {shape_label!r}, got {calibration.shape_label!r}"
        )
    return calibration
