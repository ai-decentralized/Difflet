"""Neuron DiT numerical alignment against Hugging Face diffusers (Wan2.1, W2 / W3 gates).

The neuron twin of ``test_tpu_vs_diffusers.py``: the oracle is the upstream diffusers model,
and producing the numbers is a hardware-heavy two-stage job, so this test reads what the device
check recorded rather than recomputing it:

    # stage 1 -- CPU only, ~57 GB of host RAM for the 14B, run it alone
    python tests/numerical/tpu_oracle_reference.py wan --frames 9 --timesteps 996,500 \\
        --snapshot <snapshot> --out <oracle.pt>

    # stage 2 -- 4 NeuronCores under torchrun, one result-rank<r>.json per rank in --out-dir
    torchrun --standalone --nproc-per-node 4 tests/manual/check_neuron_wan_preflight.py \\
        --model-dir <snapshot> --ref <oracle.pt> --exec-mode eager --out-dir <eager-dir> ...

    DIFFLET_RUN_NEURON_NUMERICAL=1 DIFFLET_NEURON_ORACLE_EAGER=<eager-dir> \\
        [DIFFLET_NEURON_ORACLE_COMPILE=<compile-dir>] \\
        pytest tests/numerical/test_neuron_vs_diffusers.py

Each exec mode (``eager``, ``compile``) is a parameter; a mode without a recorded directory is
skipped. The thresholds are the TPU gate's, imported from ``tests.numerical.neuron_parity``
(why they are relative to a bf16 control and not an absolute cosine: see
``test_tpu_vs_diffusers.py``). Every rank's result and every recorded timestep is judged.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from tests.numerical.neuron_parity import (
    MAX_RATIO_VS_CONTROL,
    MAX_SAME_DTYPE_RATIO,
    MIN_COSINE,
)

pytestmark = [
    pytest.mark.numerical,
    pytest.mark.slow,
]

MODES = ("eager", "compile")


def _results(mode: str) -> list[dict]:
    """The per-rank results the device check wrote for ``mode``, rank order."""
    raw = os.environ.get(f"DIFFLET_NEURON_ORACLE_{mode.upper()}")
    if not raw:
        pytest.skip(f"no recorded {mode} run; set DIFFLET_NEURON_ORACLE_{mode.upper()} to the "
                    "device check's --out-dir")
    paths = sorted(Path(raw).glob("result-rank*.json"))
    if not paths:
        pytest.skip(f"no result-rank*.json under {raw}; run the device check first")
    results = sorted((json.loads(p.read_text()) for p in paths), key=lambda r: r["rank"])
    assert [r["rank"] for r in results] == list(range(results[0]["world_size"])), (
        f"{raw}: results for ranks {[r['rank'] for r in results]} of "
        f"world size {results[0]['world_size']}"
    )
    modes = {r["exec_mode"] for r in results}
    assert modes == {mode}, f"{raw} holds a {sorted(modes)} run, not {mode}"
    return results


@pytest.fixture(autouse=True)
def _opt_in():
    if os.environ.get("DIFFLET_RUN_NEURON_NUMERICAL") != "1":
        pytest.skip("set DIFFLET_RUN_NEURON_NUMERICAL=1 to run the neuron numerical gate")


def _entries(results: list[dict]):
    for result in results:
        assert result["parity"], f"rank {result['rank']}: no parity entry recorded"
        for timestep, entry in result["parity"].items():
            yield f"rank {result['rank']} t={timestep}", entry


@pytest.mark.parametrize("mode", MODES)
def test_sits_inside_the_noise_bf16_already_creates(mode: str):
    for where, entry in _entries(_results(mode)):
        same_dtype = entry["vs_fp32"].get("vs_bf16")
        assert same_dtype is not None, (
            f"{mode} {where}: no comparison with the bf16 control recorded"
        )
        assert same_dtype["cosine"] >= MIN_COSINE, (
            f"{mode} {where}: cosine {same_dtype['cosine']:.8f} against the same-dtype "
            f"upstream reference is below the {MIN_COSINE} floor -- a break, not rounding"
        )
        control = entry["control"]
        ratio = same_dtype["rel_l1"] / control["rel_l1"]
        assert ratio <= MAX_SAME_DTYPE_RATIO, (
            f"{mode} {where}: difflet is {same_dtype['rel_l1']:.3e} from the same-dtype "
            f"reference, {ratio:.2f}x the {control['rel_l1']:.3e} that reference is from "
            f"fp32 -- outside the format's own noise"
        )


@pytest.mark.parametrize("mode", MODES)
def test_adds_little_on_top_of_what_bf16_costs(mode: str):
    for where, entry in _entries(_results(mode)):
        fp32, control = entry["vs_fp32"], entry["control"]
        assert fp32["cosine"] >= MIN_COSINE, (
            f"{mode} {where}: cosine {fp32['cosine']:.8f} against the fp32 reference is "
            f"below the {MIN_COSINE} floor"
        )
        ratio = fp32["rel_l1"] / control["rel_l1"]
        assert ratio <= MAX_RATIO_VS_CONTROL, (
            f"{mode} {where}: relative error {fp32['rel_l1']:.3e} is {ratio:.3f}x the "
            f"{control['rel_l1']:.3e} that bf16 alone costs upstream, above the "
            f"{MAX_RATIO_VS_CONTROL}x allowance"
        )


@pytest.mark.parametrize("mode", MODES)
def test_device_run_was_clean_and_ranks_agree(mode: str):
    """Every rank passed its own checks with nothing on the CPU, the ranks' outputs are
    bit-identical, and rank 0's saved outputs (what W3 compares against) are those bytes."""
    results = _results(mode)
    raw = os.environ[f"DIFFLET_NEURON_ORACLE_{mode.upper()}"]
    for result in results:
        failed = [c["name"] for c in result["checks"] if not c["ok"]]
        assert result["ok"] and not failed, f"{mode} rank {result['rank']}: failed {failed}"
        assert result["device_evidence"]["fallbacks"] == [], (
            f"{mode} rank {result['rank']}: CPU fallbacks "
            f"{result['device_evidence']['fallbacks']}"
        )
        assert result["warmup_fallbacks"] == [], (
            f"{mode} rank {result['rank']}: CPU fallbacks in the warm-up "
            f"{result['warmup_fallbacks']}"
        )
        assert result["device_evidence"]["unwarmed_shapes"] == []
    timesteps = set(results[0]["outputs"])
    for timestep in timesteps:
        hashes = {r["outputs"][timestep]["sha256"] for r in results}
        assert len(hashes) == 1, f"{mode} t={timestep}: outputs differ across ranks {hashes}"
        saved = torch.load(Path(raw) / f"out-t{timestep}.pt", weights_only=True)
        digest = hashlib.sha256(saved.contiguous().numpy().tobytes()).hexdigest()
        assert digest in hashes, f"{mode} t={timestep}: out-t{timestep}.pt is not the output"
