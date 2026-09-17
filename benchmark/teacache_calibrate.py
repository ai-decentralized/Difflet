"""Calibrate the TeaCache adaptive controller for a campaign cell (tp4tcad).

Three commands, all in-process with the SAME pipelines step_realloop drives (so
the recorded signal is exactly the scalar the controller sees at inference):

  python -m benchmark.teacache_calibrate placeholder --model flux_1_dev
      writes a calibration that never skips (threshold -1) so the adaptive
      artifact can be compiled and loaded before a real calibration exists
      (flux constructs its application from the calibration at compile time).

  python -m benchmark.teacache_calibrate collect --model flux_1_dev --prompt-index 0
      loads the ADAPTIVE artifact with the placeholder, patches
      TeaCacheController to record, per denoise step, the signal the pipeline
      hands it (``diff_norm``: the block-0 modulated-input rel-L1 -- from the
      probe NEFF for flux / qwen_image / hunyuan_video, from the host shadow /
      host CPU transformer for wan / ltx_2) and the rel-L1 change of the noise
      prediction, runs ONE generate of a calibration prompt (never the
      benchmark prompt), and appends the trajectory to
      benchmark/<device>/teacache_calib/<slug>_tp4tcad_pairs.json.
      One prompt per process: the staged models reload the DiT per
      _run_stage_internal call and repeated loads leak Neuron RT resources.

  python -m benchmark.teacache_calibrate fit --model flux_1_dev
      fits delta ~ poly(signal) (degree 4, the TeaCache paper's rescaling
      polynomial) on every recorded trajectory, picks the accumulate-mode
      threshold whose simulated skip count on those trajectories is closest to
      cadence 2's skip budget (models.cadence2_skips: 9 of 28 for FLUX, 5 of
      20 for the rest -- so tp4tcad is compared with tp4tc2 at equal speed) and
      writes benchmark/<device>/teacache_calib/<slug>_tp4tcad.json, the file
      models.resolve() names for tp4tcad, with the fit statistics (Pearson,
      R^2, n, prompts) the report shows next to the speedup.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from benchmark.models import (TEACACHE_COOLDOWN, TEACACHE_WARMUP, cadence2_skips, resolve,
                              results_dir)

# scripts/data/m9_teacache_prompts.tsv "calib" split: none of them is the
# benchmark prompt, so the calibration never sees what it is scored on.
PROMPTS_TSV = Path(__file__).resolve().parent.parent / "scripts" / "data" / "m9_teacache_prompts.tsv"

# (calibration model label, shape label) the pipelines check the JSON against
# (difflet/models/*/pipeline.py _teacache_shape_label, wan/ltx_2 build_probe_free_controller).
_MODEL_LABEL = {"flux": "flux", "qwen_image": "qwen_image", "hunyuan_video": "hunyuan_video",
                "wan": "wan", "ltx_2": "ltx_2"}
_SIGNAL_SOURCE = {
    "flux": "fused probe NEFF: on-device rel-L1 of the block-0 modulated input vs the previous step",
    "qwen_image": "fused probe NEFF: on-device rel-L1 of the block-0 modulated input vs the previous step",
    "hunyuan_video": "probe NEFF (teacache_mod_input_with_delta): on-device L2 norm of the block-0 "
                     "modulated-input change vs the previous step (not rel-L1 -- the fit absorbs "
                     "the scale)",
    "wan": "host CPU shadow (WanTeacacheCPUShadow): rel-L1 of the block-0 modulated input",
    "ltx_2": "host CPU transformer (teacache_mod_input): rel-L1 of the block-0 modulated input",
}


def calibration_prompts() -> list[str]:
    rows = PROMPTS_TSV.read_text(encoding="utf-8").strip().splitlines()[1:]
    return [line.split("\t", 1)[1].strip() for line in rows if line.startswith("calib\t")]


def shape_label(cfg) -> str:
    if cfg.num_frames is None:
        return f"{cfg.height}x{cfg.width}"
    return f"{cfg.height}x{cfg.width}x{cfg.num_frames}"


def calib_dir(cfg) -> Path:
    d = Path(cfg.teacache_calibration).parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def placeholder_path(cfg) -> Path:
    return calib_dir(cfg) / f"{cfg.slug}_tp4tcad_placeholder.json"


def pairs_path(cfg) -> Path:
    return calib_dir(cfg) / f"{cfg.slug}_tp4tcad_pairs.json"


def write_placeholder(cfg) -> Path:
    """A schema-valid calibration whose controller never skips: poly 0 < -1 is
    False at every step, target_speedup None passes every 'requested <= target'
    check. Used to build/load the adaptive artifact and for record-only runs."""
    from difflet.pipeline.teacache import CALIBRATION_SCHEMA
    doc = {
        "schema": CALIBRATION_SCHEMA,
        "model": _MODEL_LABEL[cfg.model_type], "shape_label": shape_label(cfg),
        "num_steps": int(cfg.steps), "poly_coef": [0.0], "threshold": -1.0,
        "warmup_steps": TEACACHE_WARMUP, "cooldown_steps": TEACACHE_COOLDOWN,
        "target_speedup": None, "fit_r2": None, "n_samples": 0,
        "mod_input_source": "block0_modulated_input", "accumulate": False,
        "placeholder": "record-only: never skips; replaced by `fit`",
    }
    p = placeholder_path(cfg)
    p.write_text(json.dumps(doc, indent=2) + "\n")
    return p


# --------------------------------------------------------------------------- collect

def _record_only(rec: dict):
    """Class-level patch of TeaCacheController: record the signal the pipeline
    passes (diff_norm) and the rel-L1 output change per step, never skip.
    record_full_step still runs the original so prev_mod_input (the host-signal
    models' 'previous step') keeps advancing exactly as in a real run."""
    from difflet.pipeline.teacache import TeaCacheController

    orig_skip = TeaCacheController.should_skip
    orig_full = TeaCacheController.record_full_step

    def should_skip(self, step_index, mod_input_now, *, diff_norm=None):
        rec["step"] = int(step_index)
        if diff_norm is not None:
            rec["signals"][int(step_index)] = float(diff_norm)
        return False

    def record_full_step(self, noise_pred, mod_input=None):
        cur = noise_pred.detach().float().cpu()
        prev = rec["prev"]
        if prev is not None:
            denom = prev.abs().mean().clamp_min(1e-8)
            rec["deltas"][rec["step"]] = float((cur - prev).abs().mean() / denom)
        rec["prev"] = cur
        rec["full_steps"] += 1
        return orig_full(self, noise_pred, mod_input)

    TeaCacheController.should_skip = should_skip
    TeaCacheController.record_full_step = record_full_step

    def restore():
        TeaCacheController.should_skip = orig_skip
        TeaCacheController.record_full_step = orig_full
    return restore


def collect(args) -> int:
    from benchmark import step_realloop as sr

    base = resolve(args.model, "tp4tcad")
    prompts = calibration_prompts()
    prompt = prompts[args.prompt_index]
    placeholder = write_placeholder(base)
    # The placeholder as the calibration (never skips), speedup 1.0 (<= any
    # target), a calibration prompt, and a config label of its own so the
    # staged models' work dir never collides with the benchmark cell's.
    cfg = replace(base, teacache_speedup=1.0, teacache_calibration=str(placeholder),
                  prompt=prompt, seed=args.seed, config=f"tp4tcad_calib{args.prompt_index}")
    cache = Path("~/.cache/difflet").expanduser()
    print(f"[calibrate] {base.config_slug}: collect prompt {args.prompt_index} "
          f"({prompt!r}) seed {args.seed} on the adaptive artifact (record-only)", flush=True)

    from difflet.ops.attention_config import attention_implementation
    rec = {"signals": {}, "deltas": {}, "prev": None, "step": -1, "full_steps": 0}
    with attention_implementation(cfg.attention_impl):
        t0 = time.perf_counter()
        run, cls, method, out_finite = sr._BUILDERS[args.model](cfg, cache)
        load_s = time.perf_counter() - t0
        restore = _record_only(rec)
        try:
            t1 = time.perf_counter()
            result = run()
            gen_s = time.perf_counter() - t1
        finally:
            restore()
        finite = out_finite(result)

    steps = sorted(s for s in rec["deltas"] if s in rec["signals"])
    traj = [{"step": s, "signal": rec["signals"][s], "delta": rec["deltas"][s]} for s in steps]
    if rec["full_steps"] != cfg.steps:
        print(f"[calibrate] ERROR {rec['full_steps']} full steps for {cfg.steps} scheduler steps "
              "-- the record-only run must never skip", file=sys.stderr)
        return 2
    if len(traj) < cfg.steps - 2:
        print(f"[calibrate] ERROR only {len(traj)} (signal, delta) pairs for {cfg.steps} steps: "
              f"signals at {sorted(rec['signals'])}, deltas at {sorted(rec['deltas'])}",
              file=sys.stderr)
        return 2
    pp = pairs_path(base)
    doc = json.loads(pp.read_text()) if pp.exists() else {
        "schema": "difflet-bench-teacache-pairs-v1", "model": _MODEL_LABEL[base.model_type],
        "slug": base.slug, "shape_label": shape_label(base), "num_steps": int(base.steps),
        "signal_source": _SIGNAL_SOURCE[base.model_type],
        "delta": "rel-L1 of the noise prediction vs the previous full step",
        "trajectories": []}
    doc["trajectories"] = [t for t in doc["trajectories"] if t["prompt_index"] != args.prompt_index]
    doc["trajectories"].append({
        "prompt_index": args.prompt_index, "prompt": prompt, "seed": args.seed,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "load_s": round(load_s, 1), "generate_s": round(gen_s, 1), "output_finite": finite,
        "n_pairs": len(traj), "trajectory": traj})
    doc["trajectories"].sort(key=lambda t: t["prompt_index"])
    pp.write_text(json.dumps(doc, indent=2) + "\n")
    sig = [t["signal"] for t in traj]
    dlt = [t["delta"] for t in traj]
    print(f"[calibrate] {base.config_slug}: prompt {args.prompt_index}: {len(traj)} pairs, "
          f"signal [{min(sig):.4f}, {max(sig):.4f}] delta [{min(dlt):.4f}, {max(dlt):.4f}], "
          f"generate {gen_s:.1f}s, finite={finite} -> {pp}", flush=True)
    return 0


# --------------------------------------------------------------------------- fit

def _repo_relative(p: Path) -> str:
    root = Path(results_dir()).resolve().parent.parent
    try:
        return str(p.resolve().relative_to(root))
    except ValueError:
        return str(p)


def _pearson(x, y) -> float:
    import numpy as np
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def _simulate_skips(poly, traj_signals: dict, *, steps: int, threshold: float) -> int:
    """The controller's accumulate branch (difflet/pipeline/teacache.py
    should_skip): outside [warmup, steps - cooldown) the accumulator resets and
    nothing is skipped; inside, accumulate |poly(signal)| and skip while the sum
    is below the threshold, resetting when it crosses."""
    accum, skips = 0.0, 0
    for step in range(steps):
        if step < TEACACHE_WARMUP or step >= steps - TEACACHE_COOLDOWN:
            accum = 0.0
            continue
        x = traj_signals.get(step)
        if x is None:
            accum = 0.0
            continue
        accum += abs(float(poly(x)))
        if accum < threshold:
            skips += 1
        else:
            accum = 0.0
    return skips


def fit(args) -> int:
    import numpy as np

    from difflet.pipeline.teacache import TeaCacheCalibration

    cfg = resolve(args.model, "tp4tcad")
    pp = pairs_path(cfg)
    doc = json.loads(pp.read_text())
    trajs = doc["trajectories"]
    if not trajs:
        print(f"[calibrate] no trajectories in {pp}", file=sys.stderr)
        return 2
    xs = [p["signal"] for t in trajs for p in t["trajectory"]]
    ys = [p["delta"] for t in trajs for p in t["trajectory"]]
    pearson = _pearson(xs, ys)
    coef_desc = np.polyfit(np.asarray(xs), np.asarray(ys), args.degree)   # highest degree first
    poly = np.poly1d(coef_desc)
    pred = poly(np.asarray(xs))
    ss_res = float(np.sum((np.asarray(ys) - pred) ** 2))
    ss_tot = float(np.sum((np.asarray(ys) - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    coef_asc = tuple(float(c) for c in coef_desc[::-1])

    target = cadence2_skips(cfg.steps)
    per_traj = [{p["step"]: p["signal"] for p in t["trajectory"]} for t in trajs]
    # the largest threshold worth trying skips the whole window on every prompt
    window = range(TEACACHE_WARMUP, cfg.steps - TEACACHE_COOLDOWN)
    hi = max(sum(abs(float(poly(s.get(i, 0.0)))) for i in window) for s in per_traj) * 1.01 + 1e-9
    best_thr, best_err, best_sim = 0.0, float("inf"), None
    for thr in np.linspace(0.0, hi, 2001):
        sims = [_simulate_skips(poly, s, steps=cfg.steps, threshold=float(thr)) for s in per_traj]
        err = abs(float(np.mean(sims)) - target)
        if err < best_err - 1e-12:           # first (smallest) threshold at the best error
            best_thr, best_err, best_sim = float(thr), err, sims
    calib = TeaCacheCalibration(
        model=doc["model"], shape_label=doc["shape_label"], num_steps=int(cfg.steps),
        poly_coef=coef_asc, threshold=best_thr,
        warmup_steps=TEACACHE_WARMUP, cooldown_steps=TEACACHE_COOLDOWN,
        target_speedup=float(cfg.teacache_speedup), fit_r2=float(r2), n_samples=len(xs),
        accumulate=True,
    )
    out = calib.to_dict()
    out.update({
        "signal_pearson": float(pearson),
        "poly_degree": int(args.degree),
        "signal_source": doc["signal_source"],
        "target_skips": int(target),
        "target_rule": "cadence 2's skip count at warmup/cooldown 5 (tp4tc2), so tp4tcad is "
                       "compared at the same skip budget",
        "simulated_skips_per_prompt": best_sim,
        "calibration_prompts": [t["prompt"] for t in trajs],
        "calibration_seeds": [t["seed"] for t in trajs],
        "pairs_file": _repo_relative(pp),
        "hardware_measured": True,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    Path(cfg.teacache_calibration).write_text(json.dumps(out, indent=2) + "\n")
    print(f"[calibrate] {cfg.config_slug}: n={len(xs)} pairs from {len(trajs)} prompt(s); "
          f"Pearson(signal, delta)={pearson:.3f} R^2(deg {args.degree})={r2:.3f}; "
          f"signal [{min(xs):.4f}, {max(xs):.4f}]; threshold={best_thr:.5f} -> simulated skips "
          f"{best_sim} (target {target}/{cfg.steps}, speedup {cfg.teacache_speedup}) "
          f"-> {cfg.teacache_calibration}", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    ph = sub.add_parser("placeholder")
    ph.add_argument("--model", required=True)
    co = sub.add_parser("collect")
    co.add_argument("--model", required=True)
    co.add_argument("--prompt-index", type=int, default=0)
    co.add_argument("--seed", type=int, default=42)
    ft = sub.add_parser("fit")
    ft.add_argument("--model", required=True)
    ft.add_argument("--degree", type=int, default=4)
    a = p.parse_args()
    if a.cmd == "placeholder":
        print(write_placeholder(resolve(a.model, "tp4tcad")))
        return 0
    if a.cmd == "collect":
        return collect(a)
    return fit(a)


if __name__ == "__main__":
    sys.exit(main())
