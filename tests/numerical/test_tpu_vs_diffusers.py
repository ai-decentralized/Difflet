"""TPU DiT numerical alignment against Hugging Face diffusers.

The oracle is the *upstream* implementation, never another difflet backend:
two independently wrong implementations can agree, which is exactly what a
TPU-vs-Trainium comparison would hide.

Producing the numbers is a two-stage, hardware-heavy job, so this test reads
what the harness recorded rather than recomputing it:

    # stage 1 -- CPU only, wants ~57-80 GB of host RAM, run it alone
    python tests/numerical/tpu_oracle_reference.py wan
    python tests/numerical/tpu_oracle_reference.py qwen_image

    # stage 2 -- 4 chips, writes /mnt/models/oracle_<model>_result.json
    DIFFLET_BACKEND=tpu python tests/numerical/tpu_oracle_compare.py wan
    DIFFLET_BACKEND=tpu python tests/numerical/tpu_oracle_compare.py qwen_image

    DIFFLET_RUN_TPU_NUMERICAL=1 pytest tests/numerical/test_tpu_vs_diffusers.py

**Why the gate is relative and not an absolute cosine.** The other numerical
gates in this directory use `cosine >= 0.999`, which works for a *trajectory*
of latents. Applied to a raw DiT output in bf16 it is not reachable: measured
on this host, diffusers' own bf16 forward scores 0.99873 against diffusers'
own fp32 forward for Wan. The format cannot do better, so a 0.999 gate would
fail upstream itself and tell us nothing about difflet.

So each stage-2 run also records a control -- the same upstream model in bf16
-- and the gates below are stated relative to it:

  1. difflet's deviation from fp32 must be within a small factor of what the
     dtype alone costs (<= 1.15x), i.e. sharding, the fused kernel and XLA's
     op choices must not add materially on top of bf16.
  2. difflet must sit inside the noise ball bf16 already creates: its distance
     from the *same-dtype* upstream run must not exceed the distance that run
     is from fp32.
  3. A loose absolute cosine floor, so that a catastrophic break still fails
     even if it somehow satisfies the ratios.

**On (2), and why it is not an absolute cosine.** difflet scores 0.99891
(Wan) and 0.99995 (Qwen-Image) against the same-dtype reference, so a 0.999
gate would fail Wan. That is worth being explicit about rather than quietly
loosening: the residual is consistent with the same arithmetic accumulated in
a different order, not with a defect. Two implementations deviating from fp32
*independently* by e would sit about 1.41e apart; measured, difflet is 0.91e
(Wan) and 0.38e (Qwen-Image) from the control -- closer than independence
would give, which is the signature of correlated rounding rather than
divergent math. tp=4 changes reduction order, the cross-rank qk-norm sums
sum-of-squares across ranks, and the TPU MXU accumulates in fp32 where CPU
bf16 GEMM does not. A real sharding, rotary or norm bug lands at 0.9x, not at
0.9989, and it would also not produce prompt-faithful video. The residual is
nevertheless *unexplained at the 1e-3 level* and is not claimed to be
understood.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.numerical,
    pytest.mark.slow,
]

#: How much error difflet may add on top of what bf16 already costs. Measured
#: 1.042x (Wan) and 1.021x (Qwen-Image); 1.15 leaves headroom for run-to-run
#: variation without admitting a real regression.
MAX_ERROR_RATIO_VS_CONTROL = 1.15

#: difflet's distance from the same-dtype reference, as a fraction of that
#: reference's own distance from fp32. Measured 0.91 (Wan) and 0.38
#: (Qwen-Image); 1.2 keeps difflet inside the format's noise ball.
MAX_SAME_DTYPE_DISTANCE_RATIO = 1.2

#: Loose absolute floor. Not the real gate -- upstream's own bf16 scores
#: 0.99873 for Wan, so 0.999 is unreachable here -- but a catastrophic break
#: (wrong sharding, wrong rotary) lands two orders of magnitude below this.
MIN_ABSOLUTE_COSINE = 0.995

MODELS = ("wan", "qwen_image")


def _result(model: str) -> dict:
    path = Path(
        os.environ.get(
            f"DIFFLET_TPU_ORACLE_{model.upper()}",
            f"/mnt/models/oracle_{model}_result.json",
        )
    )
    if not path.is_file():
        pytest.skip(f"no recorded oracle result at {path}; run the harness first")
    return json.loads(path.read_text())


@pytest.fixture(autouse=True)
def _opt_in():
    if os.environ.get("DIFFLET_RUN_TPU_NUMERICAL") != "1":
        pytest.skip("set DIFFLET_RUN_TPU_NUMERICAL=1 to run the TPU numerical gate")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("attention", ["fused", "sdpa"])
def test_sits_inside_the_noise_bf16_already_creates(model: str, attention: str):
    payload = _result(model)
    entry = payload["results"][attention]
    same_dtype = entry.get("vs_bf16")
    assert same_dtype is not None, (
        "the recorded result has no bf16 control; re-run stage 1, which now "
        "computes one"
    )
    assert same_dtype["cosine"] >= MIN_ABSOLUTE_COSINE, (
        f"{model}/{attention}: cosine {same_dtype['cosine']:.8f} against the "
        f"same-dtype upstream reference is below the {MIN_ABSOLUTE_COSINE} "
        f"floor -- this is a break, not rounding"
    )
    ratio = same_dtype["rel_l1"] / payload["control"]["rel_l1"]
    assert ratio <= MAX_SAME_DTYPE_DISTANCE_RATIO, (
        f"{model}/{attention}: difflet is {same_dtype['rel_l1']:.3e} from the "
        f"same-dtype reference, {ratio:.2f}x the {payload['control']['rel_l1']:.3e} "
        f"that reference is from fp32 -- outside the format's own noise"
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("attention", ["fused", "sdpa"])
def test_adds_little_on_top_of_what_bf16_costs(model: str, attention: str):
    payload = _result(model)
    control = payload["control"]
    entry = payload["results"][attention]
    ratio = entry["rel_l1"] / control["rel_l1"]
    assert ratio <= MAX_ERROR_RATIO_VS_CONTROL, (
        f"{model}/{attention}: relative error {entry['rel_l1']:.3e} is "
        f"{ratio:.3f}x the {control['rel_l1']:.3e} that bf16 alone costs "
        f"upstream, above the {MAX_ERROR_RATIO_VS_CONTROL}x allowance"
    )


@pytest.mark.parametrize("model", MODELS)
def test_the_fused_kernel_is_not_worse_than_sdpa(model: str):
    """The fused Pallas kernel replaced SDPA for speed; prove it was free.

    The two are expected to differ only in the noise -- measured 4.0e-6 (Wan)
    and 1.7e-6 (Qwen-Image) of cosine -- so this allows a small band rather
    than demanding the fused path win.
    """
    results = _result(model)["results"]
    fused, sdpa = results["fused"], results["sdpa"]
    assert fused["kernel_available"], (
        "the fused kernel was not available in the recorded run, so this "
        "comparison comes from two SDPA runs and proves nothing"
    )
    assert fused["cosine"] >= sdpa["cosine"] - 1e-4, (
        f"{model}: fused attention scored {fused['cosine']:.8f} against SDPA's "
        f"{sdpa['cosine']:.8f} -- a real accuracy regression, not noise"
    )
