"""scripts/cuda_smoke_wan.sh on a CPU host: it reaches ``difflet run`` (``--help`` exits
before any load) and writes no timings JSON without a generate. The run itself is GPU-only."""

import json
import os
import pathlib
import subprocess
import sys

SCRIPT = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "cuda_smoke_wan.sh"


def test_help_reaches_difflet_run_for_wan(tmp_path):
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON_BIN": sys.executable, "CUDA_SMOKE_LOGDIR": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "usage: difflet run" in proc.stdout and "--num-frames" in proc.stdout
    logs = list(tmp_path.glob("cuda-smoke-wan-*.log"))
    assert len(logs) == 1 and not list(tmp_path.glob("cuda-smoke-wan-*.json"))


def test_timings_are_parsed_from_the_cli_lines(tmp_path):
    log = tmp_path / "cuda-smoke-wan-x.log"
    log.write_text(
        "[difflet] loaded in 61.2 s\n"
        "[difflet] wrote /tmp/wan.mp4 (9 frames, 480x832) in 95.4 s\n"
    )
    out = tmp_path / "t.json"
    (tmp_path / "t.gpu.txt").write_text("1200\n70310\n69000\n")
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--parse-timings", str(log), str(out), "--steps", "40"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON_BIN": sys.executable, "CUDA_SMOKE_LOGDIR": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    data = json.loads(out.read_text())
    assert data["load_s"] == 61.2 and data["generate_s"] == 95.4
    assert data["output"] == "/tmp/wan.mp4" and data["steps"] == 40
    assert data["per_step_s"] == round(95.4 / 40, 2)
    assert data["peak_gpu_mem_mib"] == 70310
