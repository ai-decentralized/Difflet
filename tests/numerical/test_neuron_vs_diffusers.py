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
skipped. The recorded metrics are judged by ``tests.numerical.neuron_parity.check``, the
function the device check itself calls, so there is one rule: the TPU gate's thresholds (why
they are relative to a bf16 control and not an absolute cosine: see
``test_tpu_vs_diffusers.py``), with the same-dtype limit gated only when the result is less
accurate than the control (decision D65). Every rank's result and every recorded timestep is
judged. ``test_device_run_was_clean_and_ranks_agree`` reads the verdicts the device check
recorded, so point it at a run made under the current rule.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from tests.numerical import neuron_parity as parity
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
def test_adds_little_on_top_of_what_bf16_costs(mode: str):
    """Every rank's recorded metrics at every timestep pass ``parity.check``, the one rule the
    device check applied too: cosine floor, rel_l1 vs fp32 within 1.15x the bf16 control, and
    the distance to upstream bf16 within 1.2x the control whenever the result is less accurate
    than the control (D65; at or below 1.0x it is reported, not gated)."""
    for where, entry in _entries(_results(mode)):
        fp32, control = entry["vs_fp32"], entry["control"]
        assert fp32.get("vs_bf16") is not None, (
            f"{mode} {where}: no comparison with the bf16 control recorded"
        )
        notes: list[str] = []
        ok, reasons = parity.check(fp32, control=control, notes=notes)
        assert ok, (
            f"{mode} {where}: {'; '.join(reasons)} (rel_l1 {fp32['rel_l1']:.3e} = "
            f"{fp32['rel_l1'] / control['rel_l1']:.3f}x the {control['rel_l1']:.3e} that bf16 "
            f"alone costs upstream; thresholds max_ratio_vs_control {MAX_RATIO_VS_CONTROL}, "
            f"max_same_dtype_ratio {MAX_SAME_DTYPE_RATIO}, min_cosine {MIN_COSINE})"
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
