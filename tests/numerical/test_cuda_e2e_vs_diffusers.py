"""Whole-pipeline CUDA gate: Difflet's Qwen-Image image vs diffusers' from the same noise.

    DIFFLET_BACKEND=cuda python scripts/cuda_e2e_vs_diffusers.py --model-path <snapshot>
    DIFFLET_RUN_CUDA_NUMERICAL=1 pytest tests/numerical/test_cuda_e2e_vs_diffusers.py -v

The floors were set from the first GPU run (2026-10-17, RTX PRO 6000, torch 2.11.0+cu128:
PSNR 41.40 dB, pixel cosine 0.99972 at 50 steps, seed 0); the initial F3 floors were 25 dB /
0.99. Both pipelines run bf16, so this is "the same picture", not bit equality — headroom is
~6 dB and ~1.7e-3 in cosine, enough for a kernel change, not for a lost mask.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.numerical, pytest.mark.slow]

MIN_PSNR_DB = 35.0
MIN_COSINE = 0.998
DEFAULT_RESULT = "/tmp/logs/cuda_e2e_qwen_image.json"


def _result() -> dict:
    path = Path(os.environ.get("DIFFLET_CUDA_E2E_RESULT", DEFAULT_RESULT))
    if not path.is_file():
        pytest.skip(f"no recorded e2e result at {path}; run scripts/cuda_e2e_vs_diffusers.py first")
    return json.loads(path.read_text())


@pytest.fixture(autouse=True)
def _opt_in():
    if os.environ.get("DIFFLET_RUN_CUDA_NUMERICAL") != "1":
        pytest.skip("set DIFFLET_RUN_CUDA_NUMERICAL=1 to run the CUDA numerical gate")


def test_image_is_the_same_picture_as_diffusers():
    m = _result()["metrics"]
    assert m["psnr_db"] >= MIN_PSNR_DB, f"PSNR {m['psnr_db']:.2f} dB below {MIN_PSNR_DB}"
    assert m["cosine"] >= MIN_COSINE, f"pixel cosine {m['cosine']:.5f} below {MIN_COSINE}"


def test_result_records_the_run():
    payload = _result()
    for key in ("device", "torch", "model_path", "prompt", "steps", "seed", "shape", "images"):
        assert key in payload, key
