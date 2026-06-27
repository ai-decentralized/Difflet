"""HunyuanVideo ring-attention vs gather-KV parity gates (device-only).

Two tests, two gates:

  DIFFLET_RUN_HUNYUAN_RING_NEFF=1
    Per-layer (backbone single forward) parity — Task 4.
    Shells out to ``scripts/hunyuan_ring_parity_smoke.sh``: runs the
    HunyuanVideo DiT backbone twice (gather_kv / ring) on a fixed-seed
    input and asserts cosine >= 0.999 on the single output latent.

  DIFFLET_RUN_HUNYUAN_RING_E2E=1
    End-to-end trajectory parity — Task 6.
    Shells out to ``scripts/hunyuan_ring_e2e_parity_smoke.sh``: runs a
    short denoising loop (default 3 steps) in both modes and asserts
    per-step cosine >= 0.999 across the full trajectory.

Both tests require cp_degree > 1 on Trainium2 (NxD parallel_state).
Running the SPMD launch inside pytest directly is not possible (one
parallel_state per process); these tests shell out to the drivers which
run each mode in its own process.

DEVICE PARITY DEFERRED: HunyuanVideo transformer weights are not cached on
the compile box and the disk cannot fit them. Generic joint-ring correctness
is already proven by Task 3 (joint_ring_attention validated on trn2 at
tp=2/cp=2 with cosine 0.999972). Both tests are committed unrun — run on a
box with the weights and the required Trainium2 hardware.
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


def test_hunyuan_ring_matches_gather_kv_neff():
    if os.environ.get("DIFFLET_RUN_HUNYUAN_RING_NEFF") != "1":
        pytest.skip(
            "set DIFFLET_RUN_HUNYUAN_RING_NEFF=1 to run the HunyuanVideo ring parity gate"
        )

    repo_root = Path(__file__).resolve().parents[2]
    driver = repo_root / "scripts" / "hunyuan_ring_parity_smoke.sh"
    assert driver.exists(), f"missing parity driver: {driver}"

    # The driver compiles+runs both cp_modes at tp=2/cp=2 and asserts cosine >= 0.999
    # (override via DIFFLET_HUNYUAN_PARITY_COSINE_MIN). It exits non-zero on mismatch
    # or any compile/runtime failure, which fails this test.
    result = subprocess.run(
        ["bash", str(driver)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"hunyuan_ring_parity_smoke.sh failed (rc={result.returncode}).\n"
        f"--- stdout tail ---\n{result.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{result.stderr[-2000:]}"
    )
    assert "PASS — ring matches gather_kv" in result.stdout, result.stdout[-3000:]


def test_hunyuan_ring_e2e_trajectory_matches_gather_kv():
    """End-to-end trajectory parity: ring vs gather-KV over a short denoise loop.

    Runs the HunyuanVideo backbone through a short denoising loop (default 3
    steps, tunable via DIFFLET_HUNYUAN_E2E_STEPS) in both cp_modes and asserts
    per-step cosine >= 0.999 across the full latent trajectory.  Mirrors the
    per-layer gate above but exercises the full multi-step denoising path so
    that any accumulated scheduling or latent-trajectory divergence would surface.

    Gate: DIFFLET_RUN_HUNYUAN_RING_E2E=1  (distinct from DIFFLET_RUN_HUNYUAN_RING_NEFF=1)

    DEVICE PARITY DEFERRED: HunyuanVideo transformer weights are not cached on
    the compile box and the disk cannot fit them. Committed unrun — set the gate
    on a box with the real weights and Trainium2 hardware.
    """
    if os.environ.get("DIFFLET_RUN_HUNYUAN_RING_E2E") != "1":
        pytest.skip(
            "set DIFFLET_RUN_HUNYUAN_RING_E2E=1 to run the HunyuanVideo ring e2e "
            "trajectory parity gate (requires Trainium2 + model weights)"
        )

    repo_root = Path(__file__).resolve().parents[2]
    driver = repo_root / "scripts" / "hunyuan_ring_e2e_parity_smoke.sh"
    assert driver.exists(), f"missing e2e parity driver: {driver}"

    # The driver compiles+runs both cp_modes at tp=2/cp=2, steps through a short
    # denoising loop in each mode (each in its own process), then asserts per-step
    # cosine >= 0.999 over the full trajectory
    # (override threshold via DIFFLET_HUNYUAN_E2E_COSINE_MIN).
    # Exits non-zero on mismatch or any compile/runtime failure → fails this test.
    result = subprocess.run(
        ["bash", str(driver)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"hunyuan_ring_e2e_parity_smoke.sh failed (rc={result.returncode}).\n"
        f"--- stdout tail ---\n{result.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{result.stderr[-2000:]}"
    )
    assert "PASS — ring matches gather_kv e2e trajectory" in result.stdout, result.stdout[-3000:]
