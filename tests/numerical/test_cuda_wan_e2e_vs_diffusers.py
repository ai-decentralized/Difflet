"""Whole-pipeline CUDA gate: Difflet's Wan 2.2 video vs diffusers' from the same noise.

    DIFFLET_BACKEND=cuda python scripts/cuda_e2e_wan_vs_diffusers.py --model-path <snapshot>
    DIFFLET_RUN_CUDA_NUMERICAL=1 pytest tests/numerical/test_cuda_wan_e2e_vs_diffusers.py -v

GPU-only: the result JSON comes from the script on the pod. The metrics are over the whole
``(F, H, W, 3)`` video in [0, 1] on both sides. The floors are the initial F1 floors
(25 dB / 0.99, as F3 and F5 started); Task 5 tightens them from the first measurement, leaving
headroom. Both pipelines run bf16, so this is "the same clip", not bit equality.

The script runs Difflet once per ``--timestep-dtypes`` variant against one
diffusers run; the floors judge the top-level ``metrics``, which are the first variant's
(``float32``, production, in the pod command). Every variant's PSNR and cosine are printed
(``-rA`` shows them), not gated.
"""

from __future__ import annotations

import json
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


def test_video_is_the_same_clip_as_diffusers():
    payload = _result()
    assert payload["model"] == "wan", payload["model"]
    m = payload["metrics"]
    assert m["psnr_db"] >= MIN_PSNR_DB, f"PSNR {m['psnr_db']:.2f} dB below {MIN_PSNR_DB}"
    assert m["cosine"] >= MIN_COSINE, f"pixel cosine {m['cosine']:.5f} below {MIN_COSINE}"


def test_every_timestep_dtype_variant_is_recorded():
    payload = _result()
    variants = payload.get("variants")
    assert variants, "no variants: the result predates F1b task 2; re-run the e2e script"
    for name, variant in variants.items():
        for key in ("metrics", "timings", "image"):
            assert key in variant, (name, key)
        m = variant["metrics"]
        print(f"timestep_dtype={name}: PSNR {m['psnr_db']:.2f} dB, cosine {m['cosine']:.5f}")
    first = next(iter(variants))
    assert (
        payload["metrics"] == variants[first]["metrics"]
    ), f"the gated top-level metrics are not the first variant's ({first})"
    assert payload["timings"]["difflet"] == variants[first]["timings"], first


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
