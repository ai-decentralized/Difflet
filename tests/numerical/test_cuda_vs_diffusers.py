"""CUDA DiT numerical alignment against Hugging Face diffusers (CUDA plan Phase 4, D14).

The oracle is the *upstream* implementation, never another difflet backend. The
numbers come from one GPU script that runs three forwards on identical seeded
inputs — diffusers fp32, diffusers bf16 (the control), difflet bf16 through the
CUDA ops — and records them:

    DIFFLET_BACKEND=cuda python tests/numerical/cuda_parity_gpu.py --model-path <hf snapshot>
    DIFFLET_RUN_CUDA_NUMERICAL=1 pytest tests/numerical/test_cuda_vs_diffusers.py -v

The gates are the TPU harness's relative ones (tests/numerical/test_tpu_vs_diffusers.py):
an absolute 0.999 cosine is unreachable for bf16 itself, so difflet is judged
against what bf16 alone costs upstream.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.numerical, pytest.mark.slow]

MAX_ERROR_RATIO_VS_CONTROL = 1.15
MAX_SAME_DTYPE_DISTANCE_RATIO = 1.2
MIN_ABSOLUTE_COSINE = 0.995

DEFAULT_RESULT = "/tmp/logs/cuda_parity_qwen_image.json"


def _result() -> dict:
    path = Path(os.environ.get("DIFFLET_CUDA_PARITY_RESULT", DEFAULT_RESULT))
    if not path.is_file():
        pytest.skip(f"no recorded parity result at {path}; run cuda_parity_gpu.py first")
    return json.loads(path.read_text())


@pytest.fixture(autouse=True)
def _opt_in():
    if os.environ.get("DIFFLET_RUN_CUDA_NUMERICAL") != "1":
        pytest.skip("set DIFFLET_RUN_CUDA_NUMERICAL=1 to run the CUDA numerical gate")


def test_sits_inside_the_noise_bf16_already_creates():
    payload = _result()
    entry = payload["results"]["sdpa"]
    same_dtype = entry["vs_bf16"]
    assert same_dtype["cosine"] >= MIN_ABSOLUTE_COSINE, (
        f"cosine {same_dtype['cosine']:.8f} against the same-dtype upstream reference is "
        f"below the {MIN_ABSOLUTE_COSINE} floor -- a break, not rounding"
    )
    ratio = same_dtype["rel_l1"] / payload["control"]["rel_l1"]
    assert ratio <= MAX_SAME_DTYPE_DISTANCE_RATIO, (
        f"difflet is {same_dtype['rel_l1']:.3e} from the same-dtype reference, {ratio:.2f}x "
        f"the {payload['control']['rel_l1']:.3e} that reference is from fp32"
    )


def test_adds_little_on_top_of_what_bf16_costs():
    payload = _result()
    control = payload["control"]
    entry = payload["results"]["sdpa"]
    ratio = entry["rel_l1"] / control["rel_l1"]
    assert ratio <= MAX_ERROR_RATIO_VS_CONTROL, (
        f"relative error {entry['rel_l1']:.3e} is {ratio:.3f}x the {control['rel_l1']:.3e} "
        f"that bf16 alone costs upstream, above the {MAX_ERROR_RATIO_VS_CONTROL}x allowance"
    )


def test_result_records_the_host():
    payload = _result()
    for key in ("device", "torch", "model_path", "shape"):
        assert key in payload, key


def test_padded_prompt_sits_inside_the_noise_bf16_creates():
    # F3: a real prompt is ~16 valid tokens of 1024; the DiT must mask the padding like
    # diffusers does. Same three gates against the padded-prompt reference and control.
    payload = _result()
    if "masked" not in payload.get("results", {}) or "control_masked" not in payload:
        pytest.skip("no padded-prompt case recorded; re-run cuda_parity_gpu.py (F3)")
    entry = payload["results"]["masked"]
    control = payload["control_masked"]
    same_dtype = entry["vs_bf16"]
    assert same_dtype["cosine"] >= MIN_ABSOLUTE_COSINE, same_dtype
    assert same_dtype["rel_l1"] / control["rel_l1"] <= MAX_SAME_DTYPE_DISTANCE_RATIO, (
        same_dtype,
        control,
    )
    assert entry["rel_l1"] / control["rel_l1"] <= MAX_ERROR_RATIO_VS_CONTROL, (entry, control)
