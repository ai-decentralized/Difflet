"""Joint-MMDiT ring-attention vs gather-KV parity gate (device-only).

Skipped by default — requires a multi-core CP launch (cp_degree > 1) on
Trainium2. Set DIFFLET_RUN_JOINT_RING_NEFF=1 to enable.

This is the Task 3 go/no-go correctness gate for ``joint_ring_attention``. The
real work lives in ``scripts/joint_ring_spike.sh``: it compiles a tiny SPMD probe
at tp=2/cp=2 that, on each rank, scatters a synthetic image K,V across the cp
ring (per-rank ``S_img/cp``), keeps the text K,V replicated (``S_txt``), forms
this rank's joint query ``S_q = S_img/cp + S_txt``, and computes BOTH:
  * candidate = ``joint_ring_attention(...)`` (collective_permute ring), and
  * reference = one full joint ``attention`` over the gathered ``[image ‖ text]``.
It then asserts the two match (cosine >= 0.999). This test shells out to that
driver so the gate runs the genuine SPMD ring path; running the SPMD launch
inside pytest directly is not possible (one parallel_state per process).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.numerical,
    pytest.mark.neuron,
    pytest.mark.slow,
]


def test_joint_ring_matches_gather_kv_neff():
    if os.environ.get("DIFFLET_RUN_JOINT_RING_NEFF") != "1":
        pytest.skip("set DIFFLET_RUN_JOINT_RING_NEFF=1 to run the joint ring parity gate")

    repo_root = Path(__file__).resolve().parents[2]
    driver = repo_root / "scripts" / "joint_ring_spike.sh"
    assert driver.exists(), f"missing parity driver: {driver}"

    # The driver compiles+runs the joint-ring probe at tp=2/cp=2 and asserts
    # cosine >= 0.999 (override via DIFFLET_JOINT_RING_COSINE_MIN). It exits
    # non-zero on mismatch or any compile/runtime failure, failing this test.
    result = subprocess.run(
        ["bash", str(driver)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"joint_ring_spike.sh failed (rc={result.returncode}).\n"
        f"--- stdout tail ---\n{result.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{result.stderr[-2000:]}"
    )
    assert "PASS — joint ring matches gather_kv" in result.stdout, result.stdout[-3000:]
