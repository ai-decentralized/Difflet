"""CUDA Wan DiT numerical alignment against Hugging Face diffusers, per expert (CUDA plan F1
task 4).

The oracle is the *upstream* ``WanTransformer3DModel``, never another difflet backend. The
numbers come from one GPU script that runs, for each expert (``transformer`` at timestep 900,
``transformer_2`` at 500), three forwards on identical seeded inputs — diffusers fp32,
diffusers bf16 (the control), difflet bf16 through the CUDA ops — and records them:

    DIFFLET_BACKEND=cuda python tests/numerical/cuda_parity_wan_gpu.py --model-path <hf snapshot>
    DIFFLET_RUN_CUDA_NUMERICAL=1 pytest tests/numerical/test_cuda_wan_vs_diffusers.py -v

The gates are the Qwen-Image / Flux ones (tests/numerical/test_cuda_flux_vs_diffusers.py),
applied to every expert the JSON records: an absolute 0.999 cosine is unreachable for bf16
itself, so difflet is judged against what bf16 alone costs upstream.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.numerical, pytest.mark.slow]

# The same thresholds as test_cuda_vs_diffusers.py (not importable under importlib mode).
MAX_ERROR_RATIO_VS_CONTROL = 1.15
MAX_SAME_DTYPE_DISTANCE_RATIO = 1.2
MIN_ABSOLUTE_COSINE = 0.995

DEFAULT_RESULT = "/tmp/logs/cuda_parity_wan.json"


def _path() -> Path:
    return Path(os.environ.get("DIFFLET_CUDA_WAN_PARITY_RESULT", DEFAULT_RESULT))


def _result() -> dict:
    path = _path()
    if not path.is_file():
        pytest.skip(f"no recorded parity result at {path}; run cuda_parity_wan_gpu.py first")
    return json.loads(path.read_text())


def _recorded_experts() -> list:
    """The experts the JSON holds, read at collection time to parametrise the gates."""
    try:
        experts = sorted(json.loads(_path().read_text()).get("experts", {}))
    except (OSError, ValueError, AttributeError):
        experts = []
    reason = f"no recorded expert in {_path()}; run cuda_parity_wan_gpu.py first"
    return experts or [pytest.param("none", marks=pytest.mark.skip(reason=reason))]


EXPERTS = _recorded_experts()


@pytest.fixture(autouse=True)
def _opt_in():
    if os.environ.get("DIFFLET_RUN_CUDA_NUMERICAL") != "1":
        pytest.skip("set DIFFLET_RUN_CUDA_NUMERICAL=1 to run the CUDA numerical gate")


@pytest.mark.parametrize("expert", EXPERTS)
def test_sits_inside_the_noise_bf16_already_creates(expert):
    record = _result()["experts"][expert]
    same_dtype = record["results"]["sdpa"]["vs_bf16"]
    control = record["control"]
    assert same_dtype["cosine"] >= MIN_ABSOLUTE_COSINE, (
        f"[{expert}] cosine {same_dtype['cosine']:.8f} against the same-dtype upstream "
        f"reference is below the {MIN_ABSOLUTE_COSINE} floor -- a break, not rounding"
    )
    ratio = same_dtype["rel_l1"] / control["rel_l1"]
    assert ratio <= MAX_SAME_DTYPE_DISTANCE_RATIO, (
        f"[{expert}] difflet is {same_dtype['rel_l1']:.3e} from the same-dtype reference, "
        f"{ratio:.2f}x the {control['rel_l1']:.3e} that reference is from fp32"
    )


@pytest.mark.parametrize("expert", EXPERTS)
def test_adds_little_on_top_of_what_bf16_costs(expert):
    record = _result()["experts"][expert]
    control = record["control"]
    entry = record["results"]["sdpa"]
    ratio = entry["rel_l1"] / control["rel_l1"]
    assert ratio <= MAX_ERROR_RATIO_VS_CONTROL, (
        f"[{expert}] relative error {entry['rel_l1']:.3e} is {ratio:.3f}x the "
        f"{control['rel_l1']:.3e} that bf16 alone costs upstream, above the "
        f"{MAX_ERROR_RATIO_VS_CONTROL}x allowance"
    )


def test_result_records_the_host():
    payload = _result()
    assert payload["model"] == "wan", payload["model"]
    for key in ("device", "torch", "model_path", "shape"):
        assert key in payload, key
    assert payload["experts"], "no expert recorded"
    for expert, record in payload["experts"].items():
        assert "timestep" in record, expert
