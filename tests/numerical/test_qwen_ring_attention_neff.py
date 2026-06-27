"""Qwen-Image ring-attention vs gather-KV parity gate (device-only).

Skipped by default — requires a multi-core CP launch (cp_degree > 1) on
Trainium2. Set DIFFLET_RUN_QWEN_RING_NEFF=1 to enable.

The real work lives in ``scripts/qwen_ring_parity_smoke.sh``: it runs the
Qwen-Image DiT backbone twice on a fixed-seed input — once cp_mode=gather_kv,
once cp_mode=ring, each in its own process (neuronx_distributed parallel_state
initializes once per process) — then asserts the two output tensors match
(cosine >= 0.999). This test shells out to that driver so the gate runs the
genuine SPMD ring path; running the SPMD launch inside pytest directly is not
possible (one parallel_state per process).

DEVICE PARITY DEFERRED: Qwen-Image transformer weights are not cached on the
compile box and the disk cannot fit them. Per-model on-device Qwen parity is
deferred by project decision; generic joint-ring correctness is already proven
by Task 3 (Task 3 validated joint_ring_attention on trn2 at tp=2/cp=2 with
cosine 0.999972). This test is committed unrun — set DIFFLET_RUN_QWEN_RING_NEFF=1
to execute it on a box with the real weights and Trainium2 hardware.
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


def test_qwen_ring_matches_gather_kv_neff():
    if os.environ.get("DIFFLET_RUN_QWEN_RING_NEFF") != "1":
        pytest.skip(
            "set DIFFLET_RUN_QWEN_RING_NEFF=1 to run the Qwen-Image ring parity gate"
        )

    repo_root = Path(__file__).resolve().parents[2]
    driver = repo_root / "scripts" / "qwen_ring_parity_smoke.sh"
    assert driver.exists(), f"missing parity driver: {driver}"

    # The driver compiles+runs both cp_modes at tp=2/cp=2 and asserts cosine >= 0.999
    # (override via DIFFLET_QWEN_PARITY_COSINE_MIN). It exits non-zero on mismatch
    # or any compile/runtime failure, which fails this test.
    result = subprocess.run(
        ["bash", str(driver)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"qwen_ring_parity_smoke.sh failed (rc={result.returncode}).\n"
        f"--- stdout tail ---\n{result.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{result.stderr[-2000:]}"
    )
    assert "PASS — ring matches gather_kv" in result.stdout, result.stdout[-3000:]
