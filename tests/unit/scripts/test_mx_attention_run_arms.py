"""scripts/mx_attention/run_arms.sh and steps.py (the MXFP8 attention plan's Task 6): the
commands a lane runs, driven with a recording PY and a fake fuser on PATH (no device, no model),
as test_quant_bench.py drives bench.sh, and steps.py's step times and G3 row."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MX = REPO / "scripts" / "mx_attention"
sys.path.insert(0, str(MX))
import steps  # noqa: E402

#: A `-m difflet.cli.main` command is recorded with the environment it sees; for --requests it
#: prints two torchrun request lines at the arm's step (STEP_MS: {"bf16"|"mx"|"both": ms}).
FAKE_PY = """#!{exe}
import json, os, sys
if sys.argv[1:3] == ["-m", "difflet.cli.main"]:
    args = sys.argv[3:]
    env = {{k: os.environ.get(k) for k in ("NEURON_RT_VISIBLE_CORES", "HF_HUB_CACHE",
           "DIFFLET_NKI_TRACE_CACHE", "DIFFLET_QUANT_TARGETS", "DIFFLET_ATTN_QUANT")}}
    with open(os.environ["PY_LOG"], "a") as f:
        f.write(json.dumps({{"argv": args, "env": env}}) + "\\n")
    if "--requests" in args:
        arm = ("both" if "--attn-quant" in args else "mx") if "--quant" in args else "bf16"
        for ms in [json.loads(os.environ.get("STEP_MS", "{{}}")).get(arm, 100)] * 2:
            print(f"[x] request: 5.00 s; to the first step's end 0.5 s (prompt encode + step 1); "
                  f"step median {{ms}} ms (n=49, max 1); after the last step 0.00 s")
    sys.exit(0)
os.execv(sys.executable, [sys.executable] + sys.argv[1:])
"""
FAKE_FUSER = """#!/usr/bin/env bash
case " $FUSER_BUSY " in *" ${1#/dev/neuron} "*) echo " 4242"; exit 0 ;; esac
exit 1
"""
needs_tools = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("taskset") and os.path.exists("/usr/bin/time")),
    reason="run_arms.sh runs its steps under taskset and /usr/bin/time")


def _run(tmp_path: Path, *args: str, **env: str):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, text in (("py", FAKE_PY.format(exe=sys.executable)), ("fuser", FAKE_FUSER)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "py.jsonl"
    full = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "PY": str(bin_dir / "py"),
            "ROOT": str(tmp_path / "root"), "PY_LOG": str(log), "FUSER_BUSY": "", **env}
    cpu = str(min(os.sched_getaffinity(0)))
    done = subprocess.run(["bash", str(MX / "run_arms.sh"), *args[:4], cpu], env=full,
                          capture_output=True, text=True, timeout=300)
    lines = [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []
    return done, lines


def _flag(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


@needs_tools
def test_each_step_runs_benchs_command_for_the_arm_on_the_lanes_device(tmp_path):
    for step in ("warm", "timed", "gen2"):
        done, _ = _run(tmp_path, step, "wan22_official", "both", "3")
        assert done.returncode == 0, done.stderr
    _, lines = _run(tmp_path, "warm", "hv15_official", "attn", "5")
    wan = [ln for ln in lines if "--dual-expert" in ln["argv"]]
    assert [_flag(ln["argv"], "--steps") for ln in wan] == ["2", "40", "40"]
    for ln in wan:
        argv = ln["argv"]
        assert (_flag(argv, "--quant"), _flag(argv, "--attn-quant")) == ("mxfp8", "qk+pv")
        assert (_flag(argv, "--num-frames"), _flag(argv, "--guidance-scale-2")) == ("81", "3.0")
        assert _flag(argv, "--cache-dir") == str(tmp_path / "root" / "cache" / "video")
        assert "--keep-work-dir" in argv and _flag(argv, "--seed") == "42"
        assert ln["env"]["NEURON_RT_VISIBLE_CORES"] == "12-15"
        assert ln["env"]["HF_HUB_CACHE"].endswith("/.cache/huggingface/hub")  # hub_of
        assert ln["env"]["DIFFLET_NKI_TRACE_CACHE"] is None  # bf16 Wan only
    assert "--requests" in wan[1]["argv"] and "--prompt" in wan[2]["argv"]
    requests = (tmp_path / "root" / "wan22_official" / "both" / "requests.jsonl").read_text()
    assert [json.loads(r)["steps"] for r in requests.splitlines()] == [40, 40]
    [hv15] = [ln for ln in lines if "--dual-expert" not in ln["argv"]]
    assert "--quant" not in hv15["argv"] and _flag(hv15["argv"], "--attn-quant") == "qk+pv"
    assert hv15["env"]["DIFFLET_QUANT_TARGETS"] is None  # the former default follows --quant
    assert hv15["env"]["HF_HUB_CACHE"] == "/mnt/nvme/difflet-quant-bench-hub"
    runs = (tmp_path / "root" / "wan22_official" / "runs.log").read_text()
    assert runs.count(" start: device 3 ") == 3 and runs.count(" end: exit 0;") == 3


@needs_tools
def test_a_busy_device_runs_nothing(tmp_path):
    done, lines = _run(tmp_path, "timed", "qwen_image", "bf16", "4", FUSER_BUSY="4")
    assert done.returncode == 3 and lines == []
    assert "BUSY, not run" in (tmp_path / "root" / "qwen_image" / "runs.log").read_text()


@needs_tools
@pytest.mark.parametrize("bf16, both, linear", [(302.9, 200.0, False), (302.9, 240.0, True),
                                                (310.0, 200.0, True)])
def test_the_lane_runs_linear_alone_only_when_the_plan_asks(tmp_path, bf16, both, linear):
    """Qwen-Image: evidence 302.9 / 239.6 ms (linear / bf16 0.791). Both at 0.66 is far from
    it; at 0.79 within 2J; a bf16 2.3% off the evidence's asks for it too."""
    done, lines = _run(tmp_path, "lane", "qwen_image", "both", "2",
                       STEP_MS=json.dumps({"bf16": bf16, "both": both, "mx": 239.0}))
    assert done.returncode == 0, done.stderr
    arms = [(_flag(ln["argv"], "--quant"), _flag(ln["argv"], "--attn-quant")) for ln in lines]
    timed = [(None, None), ("mxfp8", "qk+pv")] + [("mxfp8", None)] * linear
    assert arms == timed + [("mxfp8", "qk+pv")]  # then gen2
    assert ["--requests" in ln["argv"] for ln in lines] == [True] * len(timed) + [False]
    runs = (tmp_path / "root" / "qwen_image" / "runs.log").read_text()
    assert f'"linear_run_needed": {str(linear).lower()}' in runs


def test_steps_take_the_median_of_the_requests_and_refuse_a_failed_log(tmp_path):
    log = tmp_path / "timed.log"
    log.write_text("[hv15] request 0: denoise 495.9 s, 50 steps, median 9910 ms (first 9950 ms)\n"
                   "[hv15] request 1: denoise 496.1 s, 50 steps, median 9930 ms (first 9940 ms)\n"
                   "exit 0\n")
    s = steps.steps(log)
    assert s["requests"] == [9910.0, 9930.0] and s["step_ms"] == 9920.0
    assert s["spread"] == pytest.approx(20 / 9920)
    row = steps.g3("hv15_official", s, {**s, "step_ms": 9000.0})
    assert row["rule"] == "attn / bf16 < 1 - J" and row["faster"] and not row["linear_run_needed"]
    log.write_text(log.read_text().replace("exit 0", "exit 1"))
    with pytest.raises(ValueError, match="did not exit 0"):
        steps.steps(log)
