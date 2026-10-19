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
    # The f4f run, seed 0: diffusers vs its perturbed self 30.92 dB / 0.99808; Difflet (fused
    # projection) 31.56 dB / 0.99851 — under the 35 dB floor, at the floor's own level.
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
