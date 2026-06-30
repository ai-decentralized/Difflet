"""Flux ring-attention vs gather-KV parity gate (device-only).

Skipped by default — requires a multi-core CP launch (cp_degree > 1) on
Trainium2. Set DIFFLET_RUN_FLUX_RING_NEFF=1 to enable.

The real work lives in ``scripts/flux_ring_parity_smoke.sh``: it runs the Flux DiT
backbone twice on a fixed-seed input — once cp_mode=gather_kv, once cp_mode=ring,
each in its own process (neuronx_distributed parallel_state initializes once per
process) — then asserts the two output latents match (cosine >= 0.999). This test
shells out to that driver so the gate runs the genuine SPMD ring path; running the
SPMD launch inside pytest directly is not possible (one parallel_state per process).
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


def test_flux_ring_matches_gather_kv_neff():
    if os.environ.get("DIFFLET_RUN_FLUX_RING_NEFF") != "1":
        pytest.skip("set DIFFLET_RUN_FLUX_RING_NEFF=1 to run the Flux ring parity gate")

    repo_root = Path(__file__).resolve().parents[2]
    driver = repo_root / "scripts" / "flux_ring_parity_smoke.sh"
    assert driver.exists(), f"missing parity driver: {driver}"

    # The driver compiles+runs both cp_modes at tp=2/cp=2 and asserts cosine >= 0.999
    # (override via DIFFLET_FLUX_PARITY_COSINE_MIN). It exits non-zero on mismatch or
    # any compile/runtime failure, which fails this test.
    result = subprocess.run(
        ["bash", str(driver)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"flux_ring_parity_smoke.sh failed (rc={result.returncode}).\n"
        f"--- stdout tail ---\n{result.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{result.stderr[-2000:]}"
    )
    assert "PASS — ring matches gather_kv" in result.stdout, result.stdout[-3000:]
