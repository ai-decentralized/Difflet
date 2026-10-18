"""Whole-pipeline CUDA gate: Difflet's Wan 2.2 video vs diffusers' from the same noise (F1).

    DIFFLET_BACKEND=cuda python scripts/cuda_e2e_wan_vs_diffusers.py --model-path <snapshot>
    DIFFLET_RUN_CUDA_NUMERICAL=1 pytest tests/numerical/test_cuda_wan_e2e_vs_diffusers.py -v

GPU-only: the result JSON comes from the script on the pod. The metrics are over the whole
``(F, H, W, 3)`` video in [0, 1] on both sides. The floors are the initial F1 floors
(25 dB / 0.99, as F3 and F5 started); Task 5 tightens them from the first measurement, leaving
headroom. Both pipelines run bf16, so this is "the same clip", not bit equality.

The script runs Difflet once per ``--timestep-dtypes`` variant (F1b task 2) against one
diffusers run; the floors judge the ``float32`` variant (the shipped default) by name when the
result has it, else the first listed dtype's. The top-level ``metrics`` are the first listed
dtype's, which need not be ``float32``. Every variant's PSNR and cosine are printed
(``-rA`` shows them), not gated.

**Seeds and the noise floor (F9 task 1).** A result run with ``--seeds`` carries ``per_seed``:
the floors judge EVERY seed's gated variant (a failure names the seed), the top-level keys must
be the first seed's, and the noise floor (diffusers vs its own bf16-rounded timestep,
``--noise-floor``) and the min/mean/max ``aggregate`` are printed under ``-rA``. A pre-F9 result
(no ``per_seed``) is gated exactly as before.

**The gate relative to the noise floor (F9b).** F9 measured the floor at 28.62 / 26.74 / 23.79 dB
over seeds 0-2 — its own worst seed is under the 25 dB floor, so seed luck could fail a correct
build. When the result carries the floor on every seed, the gate is stated the way D14 states
the parity gates, relative to what the reference costs itself, in three parts:

- gate A, the mean over seeds: Difflet's mean PSNR is at least the floor's mean minus
  ``20·log10(1.15)`` = 1.21 dB (D14's 1.15x deviation ratio in dB; a mean of dB values is the
  geometric mean of RMSE ratios), and the mean cosine deviation ``1 - cosine`` is at most
  1.15² = 1.3225x the floor's mean deviation (``1 - cosine`` is quadratic in the error —
  ``≈ ε²/2`` — so the square is the same 1.15x on the linear error);
- gate B, every seed: at least the floor's WORST seed minus 1.21 dB, and a cosine deviation of
  at most 1.3225x the floor's worst — against the floor's spread, not matched by seed, because
  a half-ulp perturbation moves diffusers by 5 dB between seeds;
- gate C, a loose absolute floor on every seed, ``LOOSE_PSNR_DB`` / ``LOOSE_COSINE``
  (20 dB / 0.98), so a collapsed floor cannot pass a bad build; a floor whose own worst seed is
  under it, or not finite, fails the relative test as "the oracle moved" rather than judging.

The 25 dB / 0.99 floors then step aside (``test_video_is_the_same_clip_as_diffusers`` applies
gate C and says so). Without a floor — a legacy JSON, or ``--seeds`` without ``--noise-floor``,
as ``run_gates.sh wan``'s e2e — the 25 dB / 0.99 floors judge every seed as before and the
relative test is skipped with that reason. A floor on some seeds only is a malformed result.
With n = 3 seeds the floor's mean carries a standard error of about 1.4 dB, so the F9 run's
2.0 dB headroom on gate A is about one standard error of the difference: the relative gate
reduces the seed-luck risk F9 named, it does not remove it; more seeds narrow it.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.numerical, pytest.mark.slow]

MIN_PSNR_DB = 25.0
MIN_COSINE = 0.99
# The relative gate (F9b): D14's deviation ratio, in dB for PSNR and squared on 1 - cosine
# (quadratic in the error); the loose absolute floor that holds whatever the noise floor says.
MAX_DEVIATION_RATIO = 1.15
MARGIN_DB = 20 * math.log10(MAX_DEVIATION_RATIO)  # 1.21 dB
COSINE_DEVIATION_RATIO = MAX_DEVIATION_RATIO**2  # 1.3225x on 1 - cosine
LOOSE_PSNR_DB = 20.0
LOOSE_COSINE = 0.98
DEFAULT_RESULT = "/tmp/logs/cuda_e2e_wan.json"


def _result() -> dict:
    path = Path(os.environ.get("DIFFLET_CUDA_WAN_E2E_RESULT", DEFAULT_RESULT))
    if not path.is_file():
        pytest.skip(
            f"no recorded e2e result at {path}; run scripts/cuda_e2e_wan_vs_diffusers.py first"
        )
    return json.loads(path.read_text())


@pytest.fixture(autouse=True)
def _opt_in():
    if os.environ.get("DIFFLET_RUN_CUDA_NUMERICAL") != "1":
        pytest.skip("set DIFFLET_RUN_CUDA_NUMERICAL=1 to run the CUDA numerical gate")


def _gated_metrics(payload: dict, variants: dict | None = None) -> dict:
    """The metrics the floors judge: the ``float32`` variant's (the shipped default) when the
    result has variants including it, else the first listed dtype's (legacy results: the
    top-level metrics). ``variants``: one seed's (``per_seed.<seed>.variants``) instead of the
    top-level ones."""
    per_seed = variants is not None
    if variants is None:
        variants = payload.get("variants") or {}
    if "float32" in variants:
        return variants["float32"]["metrics"]
    first = (payload.get("timestep_dtypes") or [None])[0]
    if first in variants:
        return variants[first]["metrics"]
    assert not per_seed, f"a seed's variants {sorted(variants)} hold no gated dtype"
    return payload["metrics"]


def _floor_misses(m: dict, min_psnr_db: float, min_cosine: float) -> list[str]:
    misses = []
    if not m["psnr_db"] >= min_psnr_db:
        misses.append(f"PSNR {m['psnr_db']:.2f} dB below {min_psnr_db}")
    if not m["cosine"] >= min_cosine:
        misses.append(f"pixel cosine {m['cosine']:.5f} below {min_cosine}")
    return misses


def _noise_floors(payload: dict) -> dict[str, dict] | None:
    """Per seed, the noise floor's metrics (diffusers vs its bf16-rounded timestep), or None
    when the result carries none (a legacy JSON, or ``--seeds`` without ``--noise-floor``). A
    floor on some seeds only is a malformed result."""
    per_seed = payload.get("per_seed")
    if not per_seed:
        return None
    floors = {seed: entry.get("noise_floor") for seed, entry in per_seed.items()}
    if all(floor is None for floor in floors.values()):
        return None
    missing = sorted(seed for seed, floor in floors.items() if floor is None)
    assert not missing, f"noise floor recorded on some seeds but not on {missing}"
    return {seed: floor["metrics"] for seed, floor in floors.items()}


def _absolute_floors(payload: dict) -> tuple[float, float]:
    """(PSNR, cosine) floors every seed must clear: the loose ones when the noise floor is
    recorded (the relative gate judges), the 25 dB / 0.99 ones otherwise. Says which."""
    if _noise_floors(payload) is None:
        print(f"no noise floor recorded: the {MIN_PSNR_DB} dB / {MIN_COSINE} floors judged")
        return MIN_PSNR_DB, MIN_COSINE
    print(
        f"noise floor recorded: the {MIN_PSNR_DB} dB / {MIN_COSINE} floors are superseded by the "
        f"relative gate; the loose floors {LOOSE_PSNR_DB} dB / {LOOSE_COSINE} judged every seed"
    )
    return LOOSE_PSNR_DB, LOOSE_COSINE


def test_video_is_the_same_clip_as_diffusers():
    payload = _result()
    assert payload["model"] == "wan", payload["model"]
    min_psnr_db, min_cosine = _absolute_floors(payload)
    per_seed = payload.get("per_seed")
    if not per_seed:
        misses = _floor_misses(_gated_metrics(payload), min_psnr_db, min_cosine)
        assert not misses, "; ".join(misses)
        return
    failures = []
    for seed, entry in per_seed.items():
        m = _gated_metrics(payload, entry["variants"])
        failures += [f"seed {seed}: {miss}" for miss in _floor_misses(m, min_psnr_db, min_cosine)]
    assert not failures, "; ".join(failures)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def test_video_is_as_close_to_diffusers_as_diffusers_is_to_itself():
    """The relative gate (F9b): gate A on the mean over seeds, gate B on every seed against the
    floor's worst seed; see the module docstring. Skipped when no noise floor is recorded."""
    payload = _result()
    floors = _noise_floors(payload)
    if floors is None:
        pytest.skip(
            "no noise floor recorded (run with --seeds ... --noise-floor): "
            f"the {MIN_PSNR_DB} dB / {MIN_COSINE} floors judged"
        )
    ratio = f"{COSINE_DEVIATION_RATIO:g}x"
    difflet = {
        seed: _gated_metrics(payload, entry["variants"])
        for seed, entry in payload["per_seed"].items()
    }
    n = len(floors)
    floor_psnr = [m["psnr_db"] for m in floors.values()]
    floor_dev = [1 - m["cosine"] for m in floors.values()]
    psnr = [m["psnr_db"] for m in difflet.values()]
    dev = [1 - m["cosine"] for m in difflet.values()]

    # The oracle itself: a floor that is not finite, or whose worst seed is under the loose
    # floor, cannot scale anything — diffusers moved, not Difflet.
    assert all(
        math.isfinite(v) for v in floor_psnr + floor_dev
    ), f"noise floor not finite: {floors}"
    assert min(floor_psnr) >= LOOSE_PSNR_DB and max(floor_dev) <= 1 - LOOSE_COSINE, (
        f"the oracle moved: noise floor worst seed PSNR {min(floor_psnr):.2f} dB, cosine "
        f"{1 - max(floor_dev):.5f} is under the loose floor {LOOSE_PSNR_DB} dB / {LOOSE_COSINE}"
    )

    # Gate A: the mean over seeds.
    required_a = _mean(floor_psnr) - MARGIN_DB
    allowed_a = COSINE_DEVIATION_RATIO * _mean(floor_dev)
    print(
        f"relative gate A (mean over {n} seeds): PSNR {_mean(psnr):.2f} dB, required >= "
        f"{required_a:.2f} dB (noise floor mean {_mean(floor_psnr):.2f} dB - {MARGIN_DB:.2f} dB); "
        f"cosine deviation {_mean(dev):.5f}, allowed <= {allowed_a:.5f} "
        f"({ratio} the floor's {_mean(floor_dev):.5f}; {MAX_DEVIATION_RATIO:g}x on the linear error)"
    )
    failures = []
    if not _mean(psnr) >= required_a:
        failures.append(
            f"mean PSNR {_mean(psnr):.2f} dB below {required_a:.2f} "
            f"(noise floor mean {_mean(floor_psnr):.2f} dB - {MARGIN_DB:.2f} dB)"
        )
    if not _mean(dev) <= allowed_a:
        failures.append(
            f"mean cosine deviation {_mean(dev):.5f} above {allowed_a:.5f} "
            f"({ratio} the noise floor's {_mean(floor_dev):.5f})"
        )

    # Gate B: every seed against the floor's worst seed.
    required_b = min(floor_psnr) - MARGIN_DB
    allowed_b = COSINE_DEVIATION_RATIO * max(floor_dev)
    print(
        f"relative gate B (every seed): PSNR >= {required_b:.2f} dB (noise floor min "
        f"{min(floor_psnr):.2f} dB - {MARGIN_DB:.2f} dB); cosine deviation <= {allowed_b:.5f} "
        f"({ratio} the floor's worst {max(floor_dev):.5f})"
    )
    for seed, m in difflet.items():
        if not m["psnr_db"] >= required_b:
            failures.append(
                f"seed {seed}: PSNR {m['psnr_db']:.2f} dB below {required_b:.2f} "
                f"(noise floor min {min(floor_psnr):.2f} dB - {MARGIN_DB:.2f} dB)"
            )
        if not 1 - m["cosine"] <= allowed_b:
            failures.append(
                f"seed {seed}: cosine deviation {1 - m['cosine']:.5f} above {allowed_b:.5f} "
                f"({ratio} the noise floor's worst {max(floor_dev):.5f})"
            )
    assert not failures, "; ".join(failures)


def test_every_timestep_dtype_variant_is_recorded():
    payload = _result()
    variants = payload.get("variants")
    assert variants, "no variants: the result predates F1b task 2; re-run the e2e script"
    for name, variant in variants.items():
        for key in ("metrics", "timings", "image"):
            assert key in variant, (name, key)
        m = variant["metrics"]
        print(f"timestep_dtype={name}: PSNR {m['psnr_db']:.2f} dB, cosine {m['cosine']:.5f}")
    first = payload["timestep_dtypes"][0]
    assert (
        first in variants
    ), f"timestep_dtypes[0]={first!r} is not among the variants {list(variants)}"
    assert (
        payload["metrics"] == variants[first]["metrics"]
    ), f"the top-level metrics are not the first listed dtype's ({first})"
    assert payload["timings"]["difflet"] == variants[first]["timings"], first
    per_seed = payload.get("per_seed")
    if per_seed is None:
        return
    seeds = [str(seed) for seed in payload["seeds"]]
    assert seeds == list(per_seed), f"seeds {payload['seeds']} vs per_seed {list(per_seed)}"
    assert str(payload["seed"]) == seeds[0], (payload["seed"], seeds)
    for seed, entry in per_seed.items():
        missing = set(payload["timestep_dtypes"]) - set(entry["variants"])
        assert not missing, f"seed {seed}: no variant(s) {sorted(missing)}"
        for name, variant in entry["variants"].items():
            for key in ("metrics", "timings"):
                assert key in variant, (seed, name, key)
            m = variant["metrics"]
            print(
                f"seed {seed} timestep_dtype={name}: PSNR {m['psnr_db']:.2f} dB, "
                f"cosine {m['cosine']:.5f}"
            )
    first_seed = per_seed[seeds[0]]["variants"]
    for name, variant in variants.items():
        assert (
            variant["metrics"] == first_seed[name]["metrics"]
        ), f"the top-level variants.{name} are not the first seed's ({seeds[0]})"


def _spread_line(agg: dict) -> str:
    p, c = agg["psnr_db"], agg["cosine"]
    return (
        f"PSNR min {p['min']:.2f} / mean {p['mean']:.2f} / max {p['max']:.2f} dB, "
        f"cosine min {c['min']:.5f} / mean {c['mean']:.5f} / max {c['max']:.5f}"
    )


def _check_spread(agg: dict, entries: list[dict], what: str) -> None:
    for key in ("psnr_db", "cosine"):
        values = [e["metrics"][key] for e in entries]
        expected = {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}
        for stat, value in expected.items():
            assert math.isclose(
                agg[key][stat], value, rel_tol=1e-9, abs_tol=1e-12
            ), f"aggregate {what} {key} {stat} {agg[key][stat]} != {value} from per_seed"


def test_noise_floor_and_aggregate_are_recorded():
    """Recorded, not gated: per seed the noise floor (diffusers vs its bf16-rounded timestep),
    then the min/mean/max over seeds — checked against the per-seed numbers, then printed."""
    payload = _result()
    per_seed = payload.get("per_seed")
    if per_seed is None:
        print("single-seed result (no per_seed): no noise floor recorded")
        return
    for seed, entry in per_seed.items():
        floor = entry.get("noise_floor")
        if floor is None:
            print(f"seed {seed} noise floor: not run (--noise-floor not given)")
            continue
        m = floor["metrics"]
        print(
            f"seed {seed} noise floor (diffusers vs its bf16-rounded timestep): "
            f"PSNR {m['psnr_db']:.2f} dB, cosine {m['cosine']:.5f}"
        )
    aggregate = payload.get("aggregate")
    assert aggregate, "per_seed without an aggregate: re-run the e2e script"
    entries = list(per_seed.values())
    n = len(entries)
    for name in payload["timestep_dtypes"]:
        agg = aggregate["variants"][name]
        _check_spread(agg, [e["variants"][name] for e in entries], f"variants.{name}")
        print(f"aggregate over {n} seeds, timestep_dtype={name}: {_spread_line(agg)}")
    floors = [e.get("noise_floor") for e in entries]
    if aggregate.get("noise_floor") is None:
        assert all(f is None for f in floors), "per-seed noise floors without an aggregate"
        print(f"aggregate over {n} seeds, noise floor: not run")
        return
    assert all(f is not None for f in floors), "a noise-floor aggregate with a seed missing one"
    _check_spread(aggregate["noise_floor"], floors, "noise_floor")
    print(f"aggregate over {n} seeds, noise floor: {_spread_line(aggregate['noise_floor'])}")


def test_result_records_the_run():
    payload = _result()
    for key in (
        "device",
        "torch",
        "model_path",
        "prompt",
        "steps",
        "guidance_scale",
        "guidance_scale_2",
        "seed",
        "shape",
        "video_shape",
        "images",
        "timings",
    ):
        assert key in payload, key
    for side in ("difflet", "diffusers"):
        for key in ("load_s", "generate_s", "peak_alloc_gb"):
            assert key in payload["timings"][side], (side, key)
