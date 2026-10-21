"""The Flux CUDA e2e gate (tests/numerical/test_cuda_flux_e2e_vs_diffusers.py) on fabricated
results and the measured GPU results, run in a pytest subprocess on the CPU host."""

import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
GATE = REPO / "tests" / "numerical" / "test_cuda_flux_e2e_vs_diffusers.py"
RELATIVE = "test_image_is_as_close_to_diffusers_as_diffusers_is_to_itself"
ABSOLUTE = "test_image_is_the_same_picture_as_diffusers"


def _gate(payload: dict, tmp_path) -> tuple[dict[str, str], str]:
    result = tmp_path / "result.json"
    result.write_text(json.dumps(payload))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(GATE),
            "-v",
            "-rA",
            "-p",
            "no:cacheprovider",
            "--import-mode=importlib",
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DIFFLET_BACKEND": "cpu",
            "DIFFLET_RUN_CUDA_NUMERICAL": "1",
            "DIFFLET_CUDA_FLUX_E2E_RESULT": str(result),
        },
    )
    outcomes = {}
    for line in proc.stdout.splitlines():
        progress = re.match(r"^.*?::(\S+) (PASSED|FAILED|SKIPPED|ERROR)\b", line)
        if progress:
            outcomes[progress[1]] = progress[2]
    assert outcomes, proc.stdout[-3000:] + proc.stderr[-2000:]
    return outcomes, proc.stdout


def _payload(psnr_db: float, cosine: float, *, floor: tuple[float, float] | None = None) -> dict:
    payload = {
        "model": "flux",
        "model_id": "black-forest-labs/FLUX.1-dev",
        "model_path": "/snap",
        "prompt": "a cat",
        "steps": 28,
        "guidance_scale": 3.5,
        "seed": 0,
        "shape": [1024, 1024],
        "device": "fake",
        "torch": "x",
        "metrics": {"psnr_db": psnr_db, "cosine": cosine, "mean_abs": 0.0, "max_abs": 0.0},
        "images": {"difflet": "/a.png", "diffusers": "/b.png"},
        "timings": {
            "difflet": {"load_s": 1.0, "generate_s": 1.0, "peak_alloc_gb": 1.0},
            "diffusers": {"load_s": 1.0, "generate_s": 1.0, "peak_alloc_gb": 1.0},
        },
        "noise_floor": None,
    }
    if floor is not None:
        payload["noise_floor"] = {
            "perturbation": "allow_bf16_reduced_precision_reduction flipped",
            "metrics": {"psnr_db": floor[0], "cosine": floor[1], "mean_abs": 0.0, "max_abs": 0.0},
            "timings": {"load_s": 1.0, "generate_s": 1.0, "peak_alloc_gb": None},
        }
    return payload


def test_without_a_noise_floor_the_35_db_floors_judge_and_the_relative_test_skips(tmp_path):
    out, stdout = _gate(_payload(35.0, 0.998), tmp_path)
    assert out[ABSOLUTE] == "PASSED" and out[RELATIVE] == "SKIPPED", out
    assert "no noise floor recorded: the 35.0 dB / 0.998 floors judged" in stdout
    out, stdout = _gate(_payload(34.9, 0.999), tmp_path)
    assert out[ABSOLUTE] == "FAILED" and "PSNR 34.90 dB below 35.0" in stdout


def test_with_the_f4f_seed0_floor_the_relative_gate_judges(tmp_path):
    # The f4f run's seed 0 numbers: Difflet (fused projection) 31.56 dB / 0.99851 against a floor
    # at 30.92 dB (the measured one; its cosine 0.99827 is rounded to 0.99808 here for a round
    # allowance of 0.00254) — under the 35 dB floor, at the floor's own level.
    out, stdout = _gate(_payload(31.56, 0.99851, floor=(30.92, 0.99808)), tmp_path)
    assert out[ABSOLUTE] == "PASSED" and out[RELATIVE] == "PASSED", out
    assert (
        "relative gate: PSNR 31.56 dB, required >= 29.71 dB (noise floor 30.92 dB - 1.21 dB)"
        in stdout
    )
    assert "the 35.0 dB / 0.998 floors are superseded by the relative gate" in stdout
    # 1.4 dB under the floor's minus the margin fails and names the numbers.
    out, stdout = _gate(_payload(29.5, 0.9985, floor=(30.92, 0.99808)), tmp_path)
    assert out[RELATIVE] == "FAILED" and "PSNR 29.50 dB below 29.71" in stdout
    # The cosine half: deviation 0.004 against 1.3225 x 0.00192 = 0.00254 allowed.
    out, stdout = _gate(_payload(33.0, 0.996, floor=(30.92, 0.99808)), tmp_path)
    assert out[RELATIVE] == "FAILED" and "cosine deviation 0.00400 above 0.00254" in stdout


def test_a_collapsed_floor_fails_as_the_oracle_moved_and_the_loose_floors_still_judge(tmp_path):
    out, stdout = _gate(_payload(33.0, 0.999, floor=(18.0, 0.95)), tmp_path)
    assert out[RELATIVE] == "FAILED" and "the oracle moved" in stdout
    assert out[ABSOLUTE] == "PASSED"
    out, stdout = _gate(_payload(19.5, 0.97, floor=(18.0, 0.95)), tmp_path)
    assert out[ABSOLUTE] == "FAILED" and "PSNR 19.50 dB below 20.0" in stdout


def test_measured_flux_results_keep_their_verdicts(tmp_path):
    f4f = _MEASURED / "f4f"
    for seed in (0, 1, 2):
        payload = json.loads((f4f / f"floor/e2e_floor_seed{seed}.json").read_text())
        out, _ = _gate(payload, tmp_path)
        assert set(out.values()) == {"PASSED"}, (seed, out)
    # The seed-0 runs without a floor (the gate runner's e2e, eager and compiled) fail the
    # single-seed 35 dB floor as recorded; F5c's seed-0 JSON (41.58 dB, no floor) passes.
    for name in ("gates-flux", "gates-flux-compiled"):
        out, _ = _gate(json.loads((f4f / name / "cuda_e2e_flux.json").read_text()), tmp_path)
        assert out[ABSOLUTE] == "FAILED" and out[RELATIVE] == "SKIPPED", (name, out)
    out, _ = _gate(
        json.loads((_MEASURED / "flux/f5c/cuda_e2e_flux.json").read_text()),
        tmp_path,
    )
    assert out[ABSOLUTE] == "PASSED" and out[RELATIVE] == "SKIPPED", out


# --- the gate over seeds ------------------------------------------------------------------
# A `--seeds` result: gate A on the mean over seeds within 1.21 dB of the floor's mean, gate B on
# every seed within 1.21 dB of the floor's worst seed (cosine deviation within 1.3225x the same
# way), the loose 20 dB / 0.98 floor on every seed, as F9b gates Wan.

RECORD = "test_seeds_and_aggregate_are_recorded"
# The f4f run, two decimals: Difflet (fused projection, eager) and the noise floor, seeds 0-2.
F4F_DIFFLET = {0: (31.56, 0.99851), 1: (39.56, 0.99976), 2: (43.79, 0.99991)}
F4F_FLOOR = {0: (30.92, 0.99827), 1: (36.62, 0.99952), 2: (37.61, 0.99964)}
KEYS = ("psnr_db", "cosine")


def _spread(values: list[float]) -> dict:
    return {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}


def _with_seeds(payload: dict, per_seed: dict[str, dict], seeds: list[int]) -> dict:
    """``payload`` with ``per_seed`` and its aggregate; the pre-F9c keys the first seed's."""
    entries = list(per_seed.values())

    def agg(pick):
        return {key: _spread([pick(e)["metrics"][key] for e in entries]) for key in KEYS}

    first = per_seed[str(seeds[0])]
    floors = [e["noise_floor"] for e in entries]
    return {
        **payload,
        "seed": seeds[0],
        "seeds": seeds,
        "metrics": first["metrics"],
        "timings": first["timings"],
        "noise_floor": first["noise_floor"],
        "per_seed": per_seed,
        "aggregate": {
            "difflet": agg(lambda e: e),
            "noise_floor": None if floors[0] is None else agg(lambda e: e["noise_floor"]),
        },
    }


def _seeded(
    difflet: dict[int, tuple[float, float]], *, floor: dict[int, tuple[float, float]] | None = None
) -> dict:
    """A multi-seed result: ``difflet[seed]`` is that seed's (PSNR, cosine), ``floor[seed]`` its
    noise floor (None: run without ``--noise-floor``)."""
    seeds = list(difflet)
    per_seed = {}
    for seed in seeds:
        one = _payload(*difflet[seed], floor=None if floor is None else floor[seed])
        per_seed[str(seed)] = {
            "metrics": one["metrics"],
            "timings": one["timings"],
            "noise_floor": one["noise_floor"],
            "images": {"difflet": f"/d{seed}.png", "diffusers": f"/r{seed}.png"},
        }
    return _with_seeds(_payload(*difflet[seeds[0]]), per_seed, seeds)


def test_over_seeds_the_relative_gate_judges_the_mean_and_every_seed(tmp_path):
    out, stdout = _gate(_seeded(F4F_DIFFLET, floor=F4F_FLOOR), tmp_path)
    assert set(out.values()) == {"PASSED"}, out
    assert RECORD in out
    # Gate A: the mean over seeds, 1.21 dB (a 1.15x RMSE ratio) under the floor's mean.
    assert (
        "relative gate A (mean over 3 seeds): PSNR 38.30 dB, required >= 33.84 dB "
        "(noise floor mean 35.05 dB - 1.21 dB); cosine deviation 0.00061, allowed <= 0.00113 "
        "(1.3225x the floor's 0.00086; 1.15x on the linear error)" in stdout
    ), stdout
    # Gate B: every seed against the floor's worst seed.
    assert (
        "relative gate B (every seed): PSNR >= 29.71 dB (noise floor min 30.92 dB - 1.21 dB); "
        "cosine deviation <= 0.00229 (1.3225x the floor's worst 0.00173)" in stdout
    ), stdout
    assert (
        "the 35.0 dB / 0.998 floors are superseded by the relative gate; the loose floors "
        "20.0 dB / 0.98 judged every seed" in stdout
    ), stdout
    # Recorded: every seed, its floor, and the spread over seeds.
    assert "seed 1 difflet vs diffusers: PSNR 39.56 dB, cosine 0.99976" in stdout, stdout
    assert "seed 2 noise floor (diffusers vs its perturbed self): PSNR 37.61 dB" in stdout
    assert (
        "aggregate over 3 seeds, difflet: PSNR min 31.56 / mean 38.30 / max 43.79 dB, "
        "cosine min 0.99851 / mean 0.99939 / max 0.99991" in stdout
    ), stdout
    assert (
        "aggregate over 3 seeds, noise floor: PSNR min 30.92 / mean 35.05 / max 37.61 dB, "
        "cosine min 0.99827 / mean 0.99914 / max 0.99964" in stdout
    ), stdout


def test_over_seeds_a_mean_short_of_the_floor_s_mean_fails_gate_a(tmp_path):
    # 33.5 dB on every seed clears seed 0's floor (gate B, and F4f's single-seed gate) but not
    # the mean: seeds 1 and 2 put diffusers 36.6-37.6 dB from itself.
    difflet = {seed: (33.5, 0.9995) for seed in F4F_FLOOR}
    out, stdout = _gate(_seeded(difflet, floor=F4F_FLOOR), tmp_path)
    assert out[RELATIVE] == "FAILED" and out[ABSOLUTE] == "PASSED", out
    assert "mean PSNR 33.50 dB below 33.84 (noise floor mean 35.05 dB - 1.21 dB)" in stdout
    assert "dB below 29.71" not in stdout, stdout  # every seed clears gate B


def test_over_seeds_one_seed_under_the_floor_s_worst_fails_gate_b_and_names_it(tmp_path):
    difflet = {0: (29.5, 0.9985), 1: (42.0, 0.9998), 2: (44.0, 0.9999)}
    out, stdout = _gate(_seeded(difflet, floor=F4F_FLOOR), tmp_path)
    assert out[RELATIVE] == "FAILED" and out[ABSOLUTE] == "PASSED", out
    assert "seed 0: PSNR 29.50 dB below 29.71 (noise floor min 30.92 dB - 1.21 dB)" in stdout
    assert "seed 1: PSNR" not in stdout and "mean PSNR" not in stdout, stdout


def test_over_seeds_the_cosine_deviation_is_judged_the_same_way(tmp_path):
    difflet = {0: (35.0, 0.9975), 1: (40.0, 0.9990), 2: (42.0, 0.9995)}
    out, stdout = _gate(_seeded(difflet, floor=F4F_FLOOR), tmp_path)
    assert out[RELATIVE] == "FAILED", out
    assert (
        "mean cosine deviation 0.00133 above 0.00113 (1.3225x the noise floor's 0.00086)" in stdout
    ), stdout
    assert (
        "seed 0: cosine deviation 0.00250 above 0.00229 (1.3225x the noise floor's worst 0.00173)"
        in stdout
    ), stdout
    assert "seed 1: cosine" not in stdout and "dB below" not in stdout, stdout


def test_over_seeds_a_collapsed_worst_seed_fails_as_the_oracle_moved(tmp_path):
    floor = {**F4F_FLOOR, 0: (18.0, 0.95)}
    out, stdout = _gate(_seeded(F4F_DIFFLET, floor=floor), tmp_path)
    assert out[RELATIVE] == "FAILED" and out[ABSOLUTE] == "PASSED", out
    assert (
        "the oracle moved: noise floor worst seed PSNR 18.00 dB, cosine 0.95000 is under the "
        "loose floor 20.0 dB / 0.98" in stdout
    ), stdout


def test_over_seeds_without_a_floor_the_35_db_floors_judge_every_seed(tmp_path):
    difflet = {0: (36.0, 0.999), 1: (34.5, 0.999), 2: (40.0, 0.9995)}
    out, stdout = _gate(_seeded(difflet), tmp_path)
    assert out[ABSOLUTE] == "FAILED" and out[RELATIVE] == "SKIPPED", out
    assert out[RECORD] == "PASSED", out
    assert "seed 1: PSNR 34.50 dB below 35.0" in stdout, stdout
    assert "seed 0: PSNR 36" not in stdout and "seed 2: PSNR 40" not in stdout, stdout
    assert "seed 0 noise floor: not run (--noise-floor not given)" in stdout, stdout
    assert "aggregate over 3 seeds, noise floor: not run" in stdout, stdout
    out, _ = _gate(_seeded({0: (36.0, 0.999), 1: (35.5, 0.999)}), tmp_path)
    assert out[ABSOLUTE] == "PASSED" and out[RELATIVE] == "SKIPPED", out


def test_over_seeds_a_floor_on_some_seeds_only_is_malformed(tmp_path):
    payload = _seeded(F4F_DIFFLET, floor=F4F_FLOOR)
    payload["per_seed"]["1"]["noise_floor"] = None
    out, stdout = _gate(payload, tmp_path)
    assert out[RELATIVE] == "FAILED" and out[RECORD] == "FAILED", out
    assert "noise floor recorded on some seeds but not on ['1']" in stdout, stdout


def test_over_seeds_the_first_seed_is_on_top_and_the_aggregate_matches(tmp_path):
    payload = _seeded(F4F_DIFFLET, floor=F4F_FLOOR)
    payload["metrics"] = payload["per_seed"]["1"]["metrics"]
    out, stdout = _gate(payload, tmp_path)
    assert out[RECORD] == "FAILED" and "the top-level metrics are not the first seed's" in stdout

    payload = _seeded(F4F_DIFFLET, floor=F4F_FLOOR)
    payload["seeds"] = [0, 1]
    out, _ = _gate(payload, tmp_path)
    assert out[RECORD] == "FAILED", out
    del payload["seeds"]
    out, stdout = _gate(payload, tmp_path)
    assert out[RECORD] == "FAILED" and "per_seed without seeds" in stdout, stdout
    assert out[RELATIVE] == "PASSED" and out[ABSOLUTE] == "PASSED", out

    payload = _seeded(F4F_DIFFLET, floor=F4F_FLOOR)
    payload["aggregate"]["difflet"]["psnr_db"]["mean"] = 40.0
    out, stdout = _gate(payload, tmp_path)
    assert out[RECORD] == "FAILED" and "aggregate difflet psnr_db mean 40.0" in stdout, stdout
    assert out[RELATIVE] == "PASSED" and out[ABSOLUTE] == "PASSED", out

    payload = _seeded(F4F_DIFFLET, floor=F4F_FLOOR)
    del payload["aggregate"]
    out, _ = _gate(payload, tmp_path)
    assert out[RECORD] == "FAILED", out


def test_a_single_seed_result_keeps_f4f_s_statement(tmp_path):
    out, stdout = _gate(_payload(31.56, 0.99851, floor=(30.92, 0.99808)), tmp_path)
    assert out[RECORD] == "PASSED" and "single-seed result (no per_seed)" in stdout, stdout
    assert "relative gate A" not in stdout, stdout
    # A one-seed `--seeds` result reads the same as the pre-F9c JSON.
    one = _seeded({3: (31.56, 0.99851)}, floor={3: (30.92, 0.99808)})
    out, stdout = _gate(one, tmp_path)
    assert set(out.values()) == {"PASSED"}, out
    assert (
        "relative gate: PSNR 31.56 dB, required >= 29.71 dB (noise floor 30.92 dB - 1.21 dB)"
        in stdout
    ), stdout


def test_the_measured_f4f_floor_runs_merged_over_seeds_pass(tmp_path):
    # The three single-seed floor runs of the f4f run as one three-seed result: what the F9c
    # script writes from one load per side.
    f4f = _MEASURED / "f4f/floor"
    runs = [json.loads((f4f / f"e2e_floor_seed{seed}.json").read_text()) for seed in (0, 1, 2)]
    per_seed = {
        str(run["seed"]): {key: run[key] for key in ("metrics", "timings", "noise_floor", "images")}
        for run in runs
    }
    out, stdout = _gate(_with_seeds(runs[0], per_seed, [0, 1, 2]), tmp_path)
    assert set(out.values()) == {"PASSED"}, out
    assert "relative gate A (mean over 3 seeds): PSNR 38.30 dB, required >= 33.84 dB" in stdout
    assert "relative gate B (every seed): PSNR >= 29.71 dB" in stdout, stdout


def test_measured_f9c_runs_over_seeds_pass(tmp_path):
    # The f9c3 run (Workstation Edition): seeds 0-7 with the floor, eager and compiled, and
    # the flux gate run's three-seed e2e; each passes gates A and B with the headroom the
    # status log states.
    f9c = _MEASURED / "f9c"
    expected = {
        "eager/cuda_e2e_flux_noise.json": "PSNR 36.79 dB, required >= 32.40 dB",
        "compiled/cuda_e2e_flux_noise.json": "PSNR 38.31 dB, required >= 32.40 dB",
        "gates-flux/cuda_e2e_flux.json": "PSNR 38.30 dB, required >= 33.84 dB",
    }
    for name, line in expected.items():
        out, stdout = _gate(json.loads((f9c / name).read_text()), tmp_path)
        assert set(out.values()) == {"PASSED"}, (name, out)
        assert line in stdout, (name, stdout)
    out, stdout = _gate(json.loads((f9c / "eager/cuda_e2e_flux_noise.json").read_text()), tmp_path)
    assert "relative gate B (every seed): PSNR >= 23.02 dB (noise floor min 24.24 dB" in stdout
    assert "seed 7 difflet vs diffusers: PSNR 24.89 dB" in stdout, stdout


# --- measured results the tests above are pinned against ---------------------------------------
# PSNR and cosine of the GPU runs, at full precision, transcribed from the result JSONs the e2e
# script wrote on the GPU host; the payload around them comes from the builders above. The JSON
# files themselves are kept with the rest of the GPU evidence, outside this repository.


def _without(payload: dict, *keys: str) -> dict:
    return {key: value for key, value in payload.items() if key not in keys}


def _measured_results() -> dict[str, dict]:
    return {
        # seed 0, with the noise floor
        "f4f/floor/e2e_floor_seed0.json": _payload(
            31.564294225201547, 0.9985057519682408, floor=(30.92373250148661, 0.9982712401987284)
        ),
        # seed 1, with the noise floor
        "f4f/floor/e2e_floor_seed1.json": {
            **_payload(
                39.55547969517981, 0.9997568624459019, floor=(36.62104024843132, 0.9995219231179735)
            ),
            "seed": 1,
        },
        # seed 2, with the noise floor
        "f4f/floor/e2e_floor_seed2.json": {
            **_payload(
                43.78866577298002, 0.999912451915674, floor=(37.605905468138545, 0.9996355033656944)
            ),
            "seed": 2,
        },
        # seed 0
        "f4f/gates-flux-compiled/cuda_e2e_flux.json": _payload(
            31.405094605784907, 0.9984494606556629
        ),
        # seed 0
        "f4f/gates-flux/cuda_e2e_flux.json": _payload(31.564294225201547, 0.9985057519682408),
        # 8 seeds, with the noise floor
        "f9c/compiled/cuda_e2e_flux_noise.json": _seeded(
            {
                0: (42.27261684836233, 0.9998749736584387),
                1: (41.31239820407713, 0.9998384193666154),
                2: (40.52270489155631, 0.999814000537145),
                3: (43.01444868958021, 0.9998991624433271),
                4: (36.879382207109884, 0.9994979624603094),
                5: (39.76220318313252, 0.9998030505440206),
                6: (34.23529655785738, 0.9991522001808376),
                7: (28.496203492372814, 0.9968378439737349),
            },
            floor={
                0: (30.92373250148661, 0.9982712401987284),
                1: (36.62104024843132, 0.9995219231179733),
                2: (37.605905468138545, 0.9996355033656947),
                3: (31.899727701907665, 0.9986830330018996),
                4: (33.04803157471056, 0.9987632044874011),
                5: (39.436461590790095, 0.999787891031054),
                6: (35.14129406333446, 0.9993118192728915),
                7: (24.23772558683694, 0.9915854725793107),
            },
        ),
        # 8 seeds, with the noise floor
        "f9c/eager/cuda_e2e_flux_noise.json": _seeded(
            {
                0: (31.564294225201547, 0.9985057519682408),
                1: (39.55547969517981, 0.9997568624459017),
                2: (43.78866577298002, 0.999912451915674),
                3: (41.27145138169099, 0.9998486348324223),
                4: (35.60975010632689, 0.9993133925446605),
                5: (44.13913925750629, 0.9999280588457766),
                6: (33.463866758067574, 0.9989883058547875),
                7: (24.89339956491908, 0.992760177207342),
            },
            floor={
                0: (30.92373250148661, 0.9982712401987284),
                1: (36.62104024843132, 0.9995219231179733),
                2: (37.605905468138545, 0.9996355033656947),
                3: (31.899727701907665, 0.9986830330018996),
                4: (33.04803157471056, 0.9987632044874011),
                5: (39.436461590790095, 0.999787891031054),
                6: (35.14129406333446, 0.9993118192728915),
                7: (24.23772558683694, 0.9915854725793107),
            },
        ),
        # 3 seeds, with the noise floor
        "f9c/gates-flux/cuda_e2e_flux.json": _seeded(
            {
                0: (31.564294225201547, 0.9985057519682408),
                1: (39.55547969517981, 0.9997568624459017),
                2: (43.78866577298002, 0.999912451915674),
            },
            floor={
                0: (30.92373250148661, 0.9982712401987284),
                1: (36.62104024843132, 0.9995219231179733),
                2: (37.605905468138545, 0.9996355033656947),
            },
        ),
        # seed 0
        "flux/f5c/cuda_e2e_flux.json": _payload(40.35257866675994, 0.9998033805951199),
    }


class _Measured:
    """Path-like view of ``_measured_results()``: ``(_MEASURED / run / result).read_text()``."""

    def __init__(self, rel: str = "") -> None:
        self.rel = rel

    def __truediv__(self, part: str) -> "_Measured":
        return _Measured(f"{self.rel}/{part}" if self.rel else part)

    def read_text(self) -> str:
        return json.dumps(_measured_results()[self.rel])


_MEASURED = _Measured()
