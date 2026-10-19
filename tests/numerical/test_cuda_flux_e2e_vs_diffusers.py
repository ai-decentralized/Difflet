"""Whole-pipeline CUDA gate: Difflet's Flux image vs diffusers' from the same noise (F5).

    DIFFLET_BACKEND=cuda python scripts/cuda_e2e_flux_vs_diffusers.py --model-path <snapshot>
    DIFFLET_RUN_CUDA_NUMERICAL=1 pytest tests/numerical/test_cuda_flux_e2e_vs_diffusers.py -v

GPU-only: the result JSON comes from the script on the pod. The floors are the initial F5
floors (25 dB / 0.99, as F3 started Qwen-Image); F5 task 5 tightens them from the first
measurement, leaving headroom. Both pipelines run bf16, so this is "the same picture", not
bit equality.

**The gate relative to the noise floor (F4f).** The F4f pod measured the metric's own
sensitivity on Flux: diffusers against itself with its bf16 gemm reductions flipped lands at
30.92 / 36.62 / 37.61 dB over seeds 0-2 (``--noise-floor``), so on seed 0 the 35 dB floor fails
diffusers' own rounding noise and any rounding-level change of Difflet's (F4f's fused
projection and ``--torch-compile`` both read 31.5 dB there and 39.6-43.8 dB on seeds 1-2).
When the result carries ``noise_floor`` the gate is stated the way F9b states Wan's (D14's
shape): PSNR at least the floor's minus ``20·log10(1.15)`` = 1.21 dB, the cosine deviation
``1 - cosine`` at most 1.15² = 1.3225x the floor's (quadratic in the error: 1.15x on the linear
error), and a loose absolute floor of ``LOOSE_PSNR_DB`` / ``LOOSE_COSINE`` (20 dB / 0.98) that the
floor itself must clear ("the oracle moved" otherwise). The 35 dB / 0.998 floors then step
aside; without a floor they judge as before and the relative test is skipped with that reason.

**Over seeds (F9c).** A result run with ``--seeds`` carries ``per_seed`` (a pre-F9c JSON is read
as its one top-level seed). The absolute test judges EVERY seed (a failure names it). With the
floor on every seed and two or more seeds, the relative test is F9b's for Wan: gate A, the mean
PSNR over seeds at least the floor's mean minus 1.21 dB and the mean cosine deviation at most
1.3225x the floor's mean; gate B, every seed at least the floor's WORST seed minus 1.21 dB and
at most 1.3225x the floor's worst deviation — against the floor's spread, not matched by seed
(F4f's floor spans 6.7 dB over seeds 0-2); "the oracle moved" when the floor's worst seed is
under the loose floor. One seed keeps the statement above (gates A and B coincide). A floor on
some seeds only is a malformed result. ``test_seeds_and_aggregate_are_recorded`` checks the
first-seed top-level keys and the min/mean/max ``aggregate`` against the per-seed numbers and
prints them (``-rA``).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.numerical, pytest.mark.slow]

MIN_PSNR_DB = 35.0
MIN_COSINE = 0.998
# The relative gate (F4f, as F9b for Wan): D14's deviation ratio in dB for PSNR, squared on
# 1 - cosine; the loose absolute floor that holds whatever the noise floor says.
MAX_DEVIATION_RATIO = 1.15
MARGIN_DB = 20 * math.log10(MAX_DEVIATION_RATIO)  # 1.21 dB
COSINE_DEVIATION_RATIO = MAX_DEVIATION_RATIO**2  # 1.3225x on 1 - cosine
LOOSE_PSNR_DB = 20.0
LOOSE_COSINE = 0.98
DEFAULT_RESULT = "/tmp/logs/cuda_e2e_flux.json"


def _result() -> dict:
    path = Path(os.environ.get("DIFFLET_CUDA_FLUX_E2E_RESULT", DEFAULT_RESULT))
    if not path.is_file():
        pytest.skip(
            f"no recorded e2e result at {path}; run scripts/cuda_e2e_flux_vs_diffusers.py first"
        )
    return json.loads(path.read_text())


@pytest.fixture(autouse=True)
def _opt_in():
    if os.environ.get("DIFFLET_RUN_CUDA_NUMERICAL") != "1":
        pytest.skip("set DIFFLET_RUN_CUDA_NUMERICAL=1 to run the CUDA numerical gate")


def _per_seed(payload: dict) -> dict[str, dict]:
    """Per seed, ``{metrics, noise_floor, ...}``: a ``--seeds`` result's ``per_seed`` (F9c), else
    the one top-level seed's (a pre-F9c JSON)."""
    per_seed = payload.get("per_seed")
    if per_seed:
        return per_seed
    return {
        str(payload.get("seed")): {
            "metrics": payload["metrics"],
            "noise_floor": payload.get("noise_floor"),
        }
    }


def _noise_floors(payload: dict) -> dict[str, dict] | None:
    """Per seed, the noise floor's metrics (diffusers vs its perturbed self, ``--noise-floor``),
    or None when the result carries none (a pre-F4f JSON, or the flag not given). A floor on
    some seeds only is a malformed result."""
    floors = {seed: entry.get("noise_floor") for seed, entry in _per_seed(payload).items()}
    if all(floor is None for floor in floors.values()):
        return None
    missing = sorted(seed for seed, floor in floors.items() if floor is None)
    assert not missing, f"noise floor recorded on some seeds but not on {missing}"
    return {seed: floor["metrics"] for seed, floor in floors.items()}


def _absolute_floors(payload: dict) -> tuple[float, float]:
    """(PSNR, cosine) floors every seed must clear: the loose ones when the noise floor is
    recorded (the relative gate judges), the 35 dB / 0.998 ones otherwise. Says which."""
    if _noise_floors(payload) is None:
        print(f"no noise floor recorded: the {MIN_PSNR_DB} dB / {MIN_COSINE} floors judged")
        return MIN_PSNR_DB, MIN_COSINE
    print(
        f"noise floor recorded: the {MIN_PSNR_DB} dB / {MIN_COSINE} floors are superseded by the "
        f"relative gate; the loose floors {LOOSE_PSNR_DB} dB / {LOOSE_COSINE} judged every seed"
    )
    return LOOSE_PSNR_DB, LOOSE_COSINE


def test_image_is_the_same_picture_as_diffusers():
    payload = _result()
    assert payload["model"] == "flux", payload["model"]
    min_psnr_db, min_cosine = _absolute_floors(payload)
    failures = []
    for seed, entry in _per_seed(payload).items():
        m = entry["metrics"]
        if not m["psnr_db"] >= min_psnr_db:
            failures.append(f"seed {seed}: PSNR {m['psnr_db']:.2f} dB below {min_psnr_db}")
        if not m["cosine"] >= min_cosine:
            failures.append(f"seed {seed}: pixel cosine {m['cosine']:.5f} below {min_cosine}")
    assert not failures, "; ".join(failures)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def test_image_is_as_close_to_diffusers_as_diffusers_is_to_itself():
    """The relative gate (F4f; over seeds F9c): against the recorded noise floor — one seed
    against its own floor, two or more by gates A and B; see the module docstring. Skipped when
    no noise floor is recorded."""
    payload = _result()
    floors = _noise_floors(payload)
    if floors is None:
        pytest.skip(
            "no noise floor recorded (run with --noise-floor): "
            f"the {MIN_PSNR_DB} dB / {MIN_COSINE} floors judged"
        )
    difflet = {seed: entry["metrics"] for seed, entry in _per_seed(payload).items()}
    n = len(floors)
    ratio = f"{COSINE_DEVIATION_RATIO:g}x"
    floor_psnr = [m["psnr_db"] for m in floors.values()]
    floor_dev = [1 - m["cosine"] for m in floors.values()]

    # The oracle itself: a floor that is not finite, or whose worst seed is under the loose
    # floor, cannot scale anything — diffusers moved, not Difflet.
    assert all(
        math.isfinite(v) for v in floor_psnr + floor_dev
    ), f"noise floor not finite: {floors}"
    worst = "noise floor" if n == 1 else "noise floor worst seed"
    assert min(floor_psnr) >= LOOSE_PSNR_DB and max(floor_dev) <= 1 - LOOSE_COSINE, (
        f"the oracle moved: {worst} PSNR {min(floor_psnr):.2f} dB, cosine "
        f"{1 - max(floor_dev):.5f} is under the loose floor {LOOSE_PSNR_DB} dB / {LOOSE_COSINE}"
    )

    if n == 1:
        # F4f's statement: the one seed against its own floor (gates A and B coincide).
        (m,) = difflet.values()
        (floor_p,), (floor_d,) = floor_psnr, floor_dev
        required = floor_p - MARGIN_DB
        allowed = COSINE_DEVIATION_RATIO * floor_d
        dev = 1 - m["cosine"]
        print(
            f"relative gate: PSNR {m['psnr_db']:.2f} dB, required >= {required:.2f} dB (noise "
            f"floor {floor_p:.2f} dB - {MARGIN_DB:.2f} dB); cosine deviation {dev:.5f}, allowed "
            f"<= {allowed:.5f} ({ratio} the floor's {floor_d:.5f}; {MAX_DEVIATION_RATIO:g}x on "
            "the linear error)"
        )
        failures = []
        if not m["psnr_db"] >= required:
            failures.append(
                f"PSNR {m['psnr_db']:.2f} dB below {required:.2f} "
                f"(noise floor {floor_p:.2f} dB - {MARGIN_DB:.2f} dB)"
            )
        if not dev <= allowed:
            failures.append(
                f"cosine deviation {dev:.5f} above {allowed:.5f} ({ratio} the noise floor's "
                f"{floor_d:.5f})"
            )
        assert not failures, "; ".join(failures)
        return

    psnr = [m["psnr_db"] for m in difflet.values()]
    dev = [1 - m["cosine"] for m in difflet.values()]

    # Gate A: the mean over seeds.
    required_a = _mean(floor_psnr) - MARGIN_DB
    allowed_a = COSINE_DEVIATION_RATIO * _mean(floor_dev)
    print(
        f"relative gate A (mean over {n} seeds): PSNR {_mean(psnr):.2f} dB, required >= "
        f"{required_a:.2f} dB (noise floor mean {_mean(floor_psnr):.2f} dB - {MARGIN_DB:.2f} dB); "
        f"cosine deviation {_mean(dev):.5f}, allowed <= {allowed_a:.5f} "
        f"({ratio} the floor's {_mean(floor_dev):.5f}; {MAX_DEVIATION_RATIO:g}x on the linear "
        "error)"
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


def _spread_line(agg: dict) -> str:
    p, c = agg["psnr_db"], agg["cosine"]
    return (
        f"PSNR min {p['min']:.2f} / mean {p['mean']:.2f} / max {p['max']:.2f} dB, "
        f"cosine min {c['min']:.5f} / mean {c['mean']:.5f} / max {c['max']:.5f}"
    )


def _check_spread(agg: dict, entries: list[dict], what: str) -> None:
    for key in ("psnr_db", "cosine"):
        values = [e["metrics"][key] for e in entries]
        expected = {"min": min(values), "mean": _mean(values), "max": max(values)}
        for stat, value in expected.items():
            assert math.isclose(
                agg[key][stat], value, rel_tol=1e-9, abs_tol=1e-12
            ), f"aggregate {what} {key} {stat} {agg[key][stat]} != {value} from per_seed"


def test_seeds_and_aggregate_are_recorded():
    """Recorded, not gated (F9c): a ``--seeds`` result's first seed on top, every seed's
    metrics and noise floor, and the min/mean/max over seeds — checked against the per-seed
    numbers, then printed."""
    payload = _result()
    per_seed = payload.get("per_seed")
    if per_seed is None:
        print("single-seed result (no per_seed): the top-level seed judged")
        return
    seeds = [str(seed) for seed in payload["seeds"]]
    assert seeds == list(per_seed), f"seeds {payload['seeds']} vs per_seed {list(per_seed)}"
    assert str(payload["seed"]) == seeds[0], (payload["seed"], seeds)
    first = per_seed[seeds[0]]
    for key in ("metrics", "timings", "noise_floor"):
        assert payload[key] == first[key], f"the top-level {key} are not the first seed's"
    for seed, entry in per_seed.items():
        for key in ("metrics", "timings", "noise_floor", "images"):
            assert key in entry, (seed, key)
        for side in ("difflet", "diffusers"):
            for key in ("load_s", "generate_s", "peak_alloc_gb"):
                assert key in entry["timings"][side], (seed, side, key)
        m = entry["metrics"]
        print(
            f"seed {seed} difflet vs diffusers: PSNR {m['psnr_db']:.2f} dB, "
            f"cosine {m['cosine']:.5f}"
        )
        floor = entry["noise_floor"]
        if floor is None:
            print(f"seed {seed} noise floor: not run (--noise-floor not given)")
            continue
        f = floor["metrics"]
        print(
            f"seed {seed} noise floor (diffusers vs its perturbed self): "
            f"PSNR {f['psnr_db']:.2f} dB, cosine {f['cosine']:.5f}"
        )
    aggregate = payload.get("aggregate")
    assert aggregate, "per_seed without an aggregate: re-run the e2e script"
    entries = list(per_seed.values())
    n = len(entries)
    _check_spread(aggregate["difflet"], entries, "difflet")
    print(f"aggregate over {n} seeds, difflet: {_spread_line(aggregate['difflet'])}")
    floors = [e["noise_floor"] for e in entries]
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
        "seed",
        "shape",
        "images",
    ):
        assert key in payload, key
