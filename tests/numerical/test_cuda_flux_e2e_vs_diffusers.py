"""Whole-pipeline CUDA gate: Difflet's Flux image vs diffusers' from the same noise (F5).

    DIFFLET_BACKEND=cuda python scripts/cuda_e2e_flux_vs_diffusers.py --model-path <snapshot>
    DIFFLET_RUN_CUDA_NUMERICAL=1 pytest tests/numerical/test_cuda_flux_e2e_vs_diffusers.py -v

GPU-only: the result JSON comes from the script on the pod. The floors are the initial F5
floors (25 dB / 0.99, as F3 started Qwen-Image); F5 task 5 tightens them from the first
measurement, leaving headroom. Both pipelines run bf16, so this is "the same picture", not
bit equality.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.numerical, pytest.mark.slow]

MIN_PSNR_DB = 25.0
MIN_COSINE = 0.99
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


def test_image_is_the_same_picture_as_diffusers():
    payload = _result()
    assert payload["model"] == "flux", payload["model"]
    m = payload["metrics"]
    assert m["psnr_db"] >= MIN_PSNR_DB, f"PSNR {m['psnr_db']:.2f} dB below {MIN_PSNR_DB}"
    assert m["cosine"] >= MIN_COSINE, f"pixel cosine {m['cosine']:.5f} below {MIN_COSINE}"


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
