"""The difflet-cuda-verify skill's scripts on a CPU host: every shell wrapper parses (`bash -n`)
and answers `--help`, the pod tool answers `--help`, and `run_gates.sh --dry-run` prints each
model's gate sequence with the runbooks' env and flags. Nothing here touches a GPU, a pod or
the network."""

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
SKILL = ROOT / ".claude" / "skills" / "difflet-cuda-verify"
SHELL_SCRIPTS = ["pod_setup.sh", "ship_branch.sh", "run_gates.sh"]
POD_TOOL = ROOT / "scripts" / "yottalabs_pod.py"


@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_shell_script_parses(name):
    script = SKILL / "scripts" / name
    assert script.is_file(), script
    proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("name", SHELL_SCRIPTS)
def test_shell_script_help(name):
    proc = subprocess.run(
        ["bash", str(SKILL / "scripts" / name), "--help"], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert name in proc.stdout and "usage" in proc.stdout.lower()


def test_pod_tool_help():
    proc = subprocess.run(
        [sys.executable, str(POD_TOOL), "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "YOTTALABS_API_KEY_FILE": "/nonexistent"},
    )
    assert proc.returncode == 0, proc.stderr
    for sub in [
        "gpus",
        "create",
        "status",
        "wait-running",
        "pause",
        "resume",
        "delete",
        "ssh-config",
    ]:
        assert sub in proc.stdout
    sub = subprocess.run(
        [sys.executable, str(POD_TOOL), "create", "--help"], capture_output=True, text=True
    )
    assert sub.returncode == 0 and "--wait-for-capacity" in sub.stdout and "--dry-run" in sub.stdout


def dry_run(model, tmp_path):
    snap = tmp_path / "hf" / "hub" / "models--x--y" / "snapshots" / "abc"
    out = tmp_path / "logs" / model
    proc = subprocess.run(
        [
            "bash",
            str(SKILL / "scripts" / "run_gates.sh"),
            model,
            str(snap),
            "--out",
            str(out),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={k: v for k, v in os.environ.items() if k != "HF_HOME"},
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout, snap, out


def step_lines(stdout):
    return [line for line in stdout.splitlines() if line.startswith("[step ")]


def test_run_gates_dry_run_qwen(tmp_path):
    stdout, snap, out = dry_run("qwen", tmp_path)
    assert "DIFFLET_BACKEND=cuda" in stdout and "HF_HUB_OFFLINE=1" in stdout
    assert f"HF_HOME={tmp_path / 'hf'}" in stdout  # derived from the snapshot path
    steps = step_lines(stdout)
    assert [s.split("]")[0] for s in steps] == [
        "[step smoke",
        "[step parity",
        "[step parity-gate",
        "[step e2e",
        "[step e2e-gate",
    ]
    assert "scripts/cuda_smoke.sh" in steps[0]
    assert "tests/numerical/cuda_parity_gpu.py" in steps[1] and str(snap) in steps[1]
    assert f"DIFFLET_CUDA_PARITY_RESULT={out}/cuda_parity_qwen_image.json" in steps[2]
    assert "DIFFLET_RUN_CUDA_NUMERICAL=1" in steps[2]
    assert "tests/numerical/test_cuda_vs_diffusers.py" in steps[2]
    assert "scripts/cuda_e2e_vs_diffusers.py" in steps[3]
    assert f"DIFFLET_CUDA_E2E_RESULT={out}/cuda_e2e_qwen_image.json" in steps[4]


def test_run_gates_dry_run_flux(tmp_path):
    stdout, _, out = dry_run("flux", tmp_path)
    steps = step_lines(stdout)
    assert len(steps) == 5
    assert "scripts/cuda_smoke_flux.sh" in steps[0] and "--steps 28" in steps[0]
    assert "tests/numerical/cuda_parity_flux_gpu.py" in steps[1]
    assert f"DIFFLET_CUDA_FLUX_PARITY_RESULT={out}/cuda_parity_flux.json" in steps[2]
    assert "scripts/cuda_e2e_flux_vs_diffusers.py" in steps[3]
    assert f"DIFFLET_CUDA_FLUX_E2E_RESULT={out}/cuda_e2e_flux.json" in steps[4]


def test_run_gates_dry_run_wan(tmp_path):
    stdout, _, out = dry_run("wan", tmp_path)
    steps = step_lines(stdout)
    assert len(steps) == 5
    assert "scripts/cuda_smoke_wan.sh" in steps[0] and f"CUDA_SMOKE_LOGDIR={out}" in steps[0]
    assert "--num-frames 9" in steps[0] and "--steps 40" in steps[0]
    assert "tests/numerical/cuda_parity_wan_gpu.py" in steps[1] and "--expert both" in steps[1]
    assert "--cache-dir" in steps[1] and "--reference" in steps[1]  # rounding block + reuse
    assert f"DIFFLET_CUDA_WAN_PARITY_RESULT={out}/cuda_parity_wan.json" in steps[2]
    assert "--timestep-dtypes float32,model" in steps[3]
    assert f"DIFFLET_CUDA_WAN_E2E_RESULT={out}/cuda_e2e_wan.json" in steps[4]


def test_run_gates_rejects_an_unknown_model(tmp_path):
    proc = subprocess.run(
        ["bash", str(SKILL / "scripts" / "run_gates.sh"), "hunyuan", str(tmp_path), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert proc.returncode == 2 and "qwen|flux|wan" in proc.stderr
