"""Whole-pipeline CUDA gate: Difflet's Wan 2.2 video vs diffusers' from the same noise.

    DIFFLET_BACKEND=cuda python scripts/cuda_e2e_wan_vs_diffusers.py --model-path <snapshot>
    DIFFLET_RUN_CUDA_NUMERICAL=1 pytest tests/numerical/test_cuda_wan_e2e_vs_diffusers.py -v

GPU-only: the result JSON comes from the script on the pod. The metrics are over the whole
``(F, H, W, 3)`` video in [0, 1] on both sides. The floors are the initial F1 floors
(25 dB / 0.99, as F3 and F5 started); Task 5 tightens them from the first measurement, leaving
headroom. Both pipelines run bf16, so this is "the same clip", not bit equality.

The script runs Difflet once per ``--timestep-dtypes`` variant against one
diffusers run; the floors judge the ``float32`` variant (the shipped default) by name when the
result has it, else the first listed dtype's. The top-level ``metrics`` are the first listed
dtype's, which need not be ``float32``. Every variant's PSNR and cosine are printed
(``-rA`` shows them), not gated.

**Seeds and the noise floor.** A result run with ``--seeds`` carries ``per_seed``:
the floors judge EVERY seed's gated variant (a failure names the seed), the top-level keys must
be the first seed's, and the noise floor (diffusers vs its own bf16-rounded timestep,
``--noise-floor``) and the min/mean/max ``aggregate`` are printed under ``-rA`` — recorded, never
gated. A pre-F9 result (no ``per_seed``) is gated exactly as before.
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
    if variants is None:
        variants = payload.get("variants") or {}
    if "float32" in variants:
        return variants["float32"]["metrics"]
    first = (payload.get("timestep_dtypes") or [None])[0]
    if first in variants:
        return variants[first]["metrics"]
    return payload["metrics"]


def _floor_misses(m: dict) -> list[str]:
    misses = []
    if not m["psnr_db"] >= MIN_PSNR_DB:
        misses.append(f"PSNR {m['psnr_db']:.2f} dB below {MIN_PSNR_DB}")
    if not m["cosine"] >= MIN_COSINE:
        misses.append(f"pixel cosine {m['cosine']:.5f} below {MIN_COSINE}")
    return misses


def test_video_is_the_same_clip_as_diffusers():
    payload = _result()
    assert payload["model"] == "wan", payload["model"]
    per_seed = payload.get("per_seed")
    if not per_seed:
        m = _gated_metrics(payload)
        assert m["psnr_db"] >= MIN_PSNR_DB, f"PSNR {m['psnr_db']:.2f} dB below {MIN_PSNR_DB}"
        assert m["cosine"] >= MIN_COSINE, f"pixel cosine {m['cosine']:.5f} below {MIN_COSINE}"
        return
    failures = []
    for seed, entry in per_seed.items():
        m = _gated_metrics(payload, entry["variants"])
        failures += [f"seed {seed}: {miss}" for miss in _floor_misses(m)]
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
