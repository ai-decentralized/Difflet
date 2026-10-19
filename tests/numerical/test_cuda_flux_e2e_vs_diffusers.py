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


def _noise_floor(payload: dict) -> dict | None:
    """The noise floor's metrics (diffusers vs its perturbed self, ``--noise-floor``), or None
    when the result carries none (a pre-F4f JSON, or the flag not given)."""
    floor = payload.get("noise_floor")
    return None if floor is None else floor["metrics"]


def _absolute_floors(payload: dict) -> tuple[float, float]:
    if _noise_floor(payload) is None:
        print(f"no noise floor recorded: the {MIN_PSNR_DB} dB / {MIN_COSINE} floors judged")
        return MIN_PSNR_DB, MIN_COSINE
    print(
        f"noise floor recorded: the {MIN_PSNR_DB} dB / {MIN_COSINE} floors are superseded by the "
        f"relative gate; the loose floors {LOOSE_PSNR_DB} dB / {LOOSE_COSINE} judged"
    )
    return LOOSE_PSNR_DB, LOOSE_COSINE


def test_image_is_the_same_picture_as_diffusers():
    payload = _result()
    assert payload["model"] == "flux", payload["model"]
    min_psnr_db, min_cosine = _absolute_floors(payload)
    m = payload["metrics"]
    assert m["psnr_db"] >= min_psnr_db, f"PSNR {m['psnr_db']:.2f} dB below {min_psnr_db}"
    assert m["cosine"] >= min_cosine, f"pixel cosine {m['cosine']:.5f} below {min_cosine}"


def test_image_is_as_close_to_diffusers_as_diffusers_is_to_itself():
    """The relative gate (F4f): against the recorded noise floor; see the module docstring.
    Skipped when no noise floor is recorded."""
    payload = _result()
    floor = _noise_floor(payload)
    if floor is None:
        pytest.skip(
            "no noise floor recorded (run with --noise-floor): "
            f"the {MIN_PSNR_DB} dB / {MIN_COSINE} floors judged"
        )
    m = payload["metrics"]
    floor_psnr, floor_dev = floor["psnr_db"], 1 - floor["cosine"]
    assert all(
        math.isfinite(v) for v in (floor_psnr, floor_dev)
    ), f"noise floor not finite: {floor}"
    assert floor_psnr >= LOOSE_PSNR_DB and floor_dev <= 1 - LOOSE_COSINE, (
        f"the oracle moved: noise floor PSNR {floor_psnr:.2f} dB, cosine {floor['cosine']:.5f} "
        f"is under the loose floor {LOOSE_PSNR_DB} dB / {LOOSE_COSINE}"
    )
    required = floor_psnr - MARGIN_DB
    allowed = COSINE_DEVIATION_RATIO * floor_dev
    dev = 1 - m["cosine"]
    ratio = f"{COSINE_DEVIATION_RATIO:g}x"
    print(
        f"relative gate: PSNR {m['psnr_db']:.2f} dB, required >= {required:.2f} dB (noise floor "
        f"{floor_psnr:.2f} dB - {MARGIN_DB:.2f} dB); cosine deviation {dev:.5f}, allowed <= "
        f"{allowed:.5f} ({ratio} the floor's {floor_dev:.5f}; {MAX_DEVIATION_RATIO:g}x on the "
        "linear error)"
    )
    failures = []
    if not m["psnr_db"] >= required:
        failures.append(
            f"PSNR {m['psnr_db']:.2f} dB below {required:.2f} "
            f"(noise floor {floor_psnr:.2f} dB - {MARGIN_DB:.2f} dB)"
        )
    if not dev <= allowed:
        failures.append(
            f"cosine deviation {dev:.5f} above {allowed:.5f} ({ratio} the noise floor's {floor_dev:.5f})"
        )
    assert not failures, "; ".join(failures)


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
