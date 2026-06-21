"""Generic TeaCache method auto-selection (cclog 91).

Given a gate trajectory — the per-step block-0 modulated-input *signal* and the
true noise_pred *output delta* — pick the cache method WITHOUT per-model code:

  * **adaptive** (original probe TeaCache) when the signal predicts the output
    (Pearson >= ADAPTIVE_PEARSON). The only case worth a per-model probe.
  * **online_delta** (generic, 0-per-model) when the output trajectory itself is
    predictable step-to-step (lag-1 autocorrelation >= ONLINE_AUTOCORR) — the
    controller skips flat steps using the measured noise_pred delta, no probe.
  * **cadence** (blind) fallback when neither holds.

Emits a ``TeaCacheCalibration`` the unified controller dispatches on (probe via
``poly_coef``/``accumulate``; online via ``online_delta_alpha``; blind via
``cadence``). Measured rule (cclog 91): HV-1.0 P(sig,δ)=0.984 -> adaptive;
Qwen P=0.30 but δ-autocorr=0.93 -> online_delta (beats its old fixed-cadence).
"""

from __future__ import annotations

from difflet.pipeline.teacache import TeaCacheCalibration

ADAPTIVE_PEARSON = 0.8
ONLINE_AUTOCORR = 0.7
DEFAULT_ALPHA = 0.6


def pearson(x, y) -> float | None:
    n = min(len(x), len(y))
    if n < 3:
        return None
    x, y = list(x[:n]), list(y[:n])
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x) ** 0.5
    syy = sum((b - my) ** 2 for b in y) ** 0.5
    return (sxy / (sxx * syy)) if sxx > 0 and syy > 0 else None


def lag1_autocorr(deltas) -> float | None:
    return pearson(deltas[:-1], deltas[1:]) if len(deltas) > 3 else None


def decide_method(probe_pearson: float | None, delta_autocorr: float | None) -> str:
    if probe_pearson is not None and probe_pearson >= ADAPTIVE_PEARSON:
        return "adaptive"
    if delta_autocorr is not None and delta_autocorr >= ONLINE_AUTOCORR:
        return "online_delta"
    return "cadence"


def build_calibration(
    *,
    model: str,
    shape_label: str,
    num_steps: int,
    signals: list[float],
    deltas: list[float],
    warmup: int = 2,
    cooldown: int = 2,
    alpha: float = DEFAULT_ALPHA,
    cadence: int = 2,
) -> TeaCacheCalibration:
    """Auto-pick the method from the gate trajectory and return its calibration."""
    pp = pearson(signals, deltas) if signals and deltas else None
    ac = lag1_autocorr(deltas) if deltas else None
    method = decide_method(pp, ac)
    common = dict(
        model=model, shape_label=shape_label, num_steps=int(num_steps),
        warmup_steps=warmup, cooldown_steps=cooldown, fit_r2=pp,
        poly_coef=(0.0,), threshold=0.0,
    )
    if method == "adaptive":
        import numpy as np

        sig = np.asarray(signals, dtype=float)
        dlt = np.asarray(deltas, dtype=float)
        scale = float(sig.max()) or 1.0
        desc = np.polyfit(sig / scale, dlt, min(4, len(sig) - 1))  # normalize then rescale
        deg = len(desc) - 1
        asc = tuple(float(desc[deg - k]) / (scale ** k) for k in range(deg + 1))
        return TeaCacheCalibration(**{**common, "poly_coef": asc, "threshold": 0.2, "accumulate": True})
    if method == "online_delta":
        return TeaCacheCalibration(**{**common, "online_delta_alpha": float(alpha)})
    return TeaCacheCalibration(**{**common, "cadence": int(cadence)})


def _rel_l1(cur, prev) -> float:
    import torch

    cur = cur.detach().float() if torch.is_tensor(cur) else torch.as_tensor(cur, dtype=torch.float32)
    prev = prev.detach().float() if torch.is_tensor(prev) else torch.as_tensor(prev, dtype=torch.float32)
    denom = prev.abs().mean().clamp_min(1e-8)
    return float((cur - prev).abs().mean() / denom)


def run_gate(
    *,
    model: str,
    shape_label: str,
    num_steps: int,
    init_latent,
    step_fn,
    advance_fn,
    warmup: int = 2,
    cooldown: int = 2,
    alpha: float = DEFAULT_ALPHA,
    cadence: int = 2,
):
    """Generic CPU-shadow gate: run a denoise loop, capture (signal, output) per step,
    auto-select the cache method, and return ``(TeaCacheCalibration, summary)``.

    This is the model-agnostic core the per-model gate scripts share (cclog 91). The
    caller supplies three thin callables (the only model-specific glue):

      * ``init_latent() -> latent``
      * ``step_fn(step_index, latent) -> (noise_pred, block0_signal)`` — runs the DiT
        forward and returns its output plus the block-0 modulated-input signal tensor
        (e.g. captured by a forward_pre_hook). Use REAL conditioning (cclog 89).
      * ``advance_fn(latent, noise_pred, step_index) -> latent`` — one scheduler step.

    Per-step signal = rel-L1 change of the block-0 modulated input; delta = rel-L1
    change of the noise_pred. ``build_calibration`` then picks adaptive / online_delta /
    cadence from Pearson(signal, delta) + lag-1 autocorr(delta).
    """
    latent = init_latent()
    prev_np = prev_sig = None
    signals: list[float] = []
    deltas: list[float] = []
    for i in range(int(num_steps)):
        noise_pred, sig = step_fn(i, latent)
        if prev_np is not None:
            deltas.append(_rel_l1(noise_pred, prev_np))
            # sig is None in the fully-0-per-model path (no block-0 hook): the gate
            # then can only pick online_delta / cadence from the output trajectory.
            if sig is not None and prev_sig is not None:
                signals.append(_rel_l1(sig, prev_sig))
        prev_np, prev_sig = noise_pred, sig
        latent = advance_fn(latent, noise_pred, i)

    cal = build_calibration(
        model=model, shape_label=shape_label, num_steps=num_steps,
        signals=signals, deltas=deltas, warmup=warmup, cooldown=cooldown,
        alpha=alpha, cadence=cadence,
    )
    pp = pearson(signals, deltas)
    ac = lag1_autocorr(deltas)
    method = ("adaptive" if cal.accumulate else "online_delta" if cal.online_delta_alpha > 0 else "cadence")
    summary = {"method": method, "probe_pearson": pp, "delta_autocorr": ac, "n_pairs": len(deltas)}
    return cal, summary


__all__ = ["pearson", "lag1_autocorr", "decide_method", "build_calibration", "run_gate",
           "ADAPTIVE_PEARSON", "ONLINE_AUTOCORR", "DEFAULT_ALPHA"]
