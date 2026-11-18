"""scripts/quant_bench: bench.sh's command lines and the environment each command sees (a
recording PY in place of the venv's python, so no device and no model), and quality.py's
reading of an arm. A bench lane runs for hours, so a wrong flag, a leaked shell variable or a
stale output from an earlier run is the expensive kind of bug here."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
QB = REPO / "scripts" / "quant_bench"
PINS = {"wan21_official": "38ec498cb3208fb688890f8cc7e94ede2cbd7f68",
        "wan22_official": "5be7df9619b54f4e2667b2755bc6a756675b5cd7",
        "hv15_official": "286be7ce72277246578a3e3cc2487e95ddae5bcf"}
#: Shell variables that would make an arm's compile, threads or quantized set something other
#: than the bench's: cache locations that win over --cache-dir, thread counts that change
#: HunyuanVideo-1.5's host-CPU encode, and a leaked override.
LEAKS = ("DIFFLET_COMPILE_CACHE", "TORCH_NEURONX_NEFF_CACHE_DIR",
         "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "TORCH_NEURONX_HLO_CACHE_DIR",
         "TORCHINDUCTOR_CACHE_DIR", "DIFFLET_NEURON_WEIGHT_STORE", "DIFFLET_SHARED_WEIGHTS_DIR",
         "OMP_NUM_THREADS", "MKL_NUM_THREADS", "DIFFLET_QUANT_TARGETS")

#: The bench's PY: a `-m difflet.cli.main` command is recorded (its argv and the LEAKS it
#: sees) and writes "<DRY_RUN>" to its --output, or fails without writing when the output
#: path contains DRY_FAIL; anything else (stamp.py, the request lines) runs on python.
FAKE_PY = """#!{exe}
import json, os, sys
if sys.argv[1:3] == ["-m", "difflet.cli.main"]:
    args = sys.argv[3:]
    with open(os.environ["DRY_LOG"], "a") as f:
        f.write(json.dumps({{"argv": args, "env": {{k: os.environ.get(k) for k in {leaks!r}}}}}) + "\\n")
    out = args[args.index("--output") + 1] if "--output" in args else None
    if out and os.environ.get("DRY_FAIL") and os.environ["DRY_FAIL"] in out:
        sys.exit(1)
    if out:
        open(out, "w").write(os.environ.get("DRY_RUN", ""))
    sys.exit(0)
os.execv(sys.executable, [sys.executable] + sys.argv[1:])
"""

needs_tools = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("taskset") and os.path.exists("/usr/bin/time")),
    reason="bench.sh runs its steps under taskset and /usr/bin/time")


def _bench(tmp_path: Path, *keys: str, **env: str):
    fake = tmp_path / "py"
    fake.write_text(FAKE_PY.format(exe=sys.executable, leaks=LEAKS))
    fake.chmod(0o755)
    log = tmp_path / "cmds.jsonl"
    full = {**os.environ, "PY": str(fake), "BENCH_ROOT": str(tmp_path / "results"),
            "CACHE_ROOT": str(tmp_path / "cache"), "HF_HUB_CACHE": str(tmp_path / "hub"),
            "DRY_LOG": str(log), **env}
    cpu = str(min(os.sched_getaffinity(0)))
    done = subprocess.run(["bash", str(QB / "bench.sh"), "--cores", "0-3", "--cpus", cpu, *keys],
                          env=full, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    lines = [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []
    return tmp_path / "results", lines


def _flag(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


@needs_tools
def test_no_shell_cache_thread_or_targets_variable_reaches_a_command(tmp_path):
    root, lines = _bench(tmp_path, "wan21_official", **{k: "/leak" for k in LEAKS})
    assert len(lines) == 8  # per arm: compile, two cold runs, the warm process
    for line in lines:
        assert line["env"] == {k: None for k in LEAKS}, line["argv"][0]
    lane = (root / "wan21_official" / "lane.txt").read_text().split("\n")
    assert lane[:2] == ["cores 0-3", f"cpus {min(os.sched_getaffinity(0))}"]
    assert lane[2] == "threads 1"  # nproc under the lane's taskset: torch's default count


@needs_tools
def test_the_official_keys_pin_their_snapshots_and_settings(tmp_path):
    root, lines = _bench(tmp_path, *PINS)
    for key, rev in PINS.items():
        mine = [ln["argv"] for ln in lines if rev in ln["argv"]]
        assert len(mine) == 8, key
    wan22 = [ln["argv"] for ln in lines if "Wan-AI/Wan2.2-T2V-A14B-Diffusers" in ln["argv"]
             and ln["argv"][0] == "generate"]
    neg = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"  # noqa: E501
    for argv in wan22:
        assert (_flag(argv, "--steps"), _flag(argv, "--guidance-scale"),
                _flag(argv, "--guidance-scale-2"), _flag(argv, "--num-frames")) == (
            "40", "4.0", "3.0", "81")
        assert _flag(argv, "--negative-prompt") == neg and "--dual-expert" in argv
    for arm in ("bf16", "mx"):
        requests = (root / "wan22_official" / arm / "requests.jsonl").read_text().splitlines()
        assert [json.loads(r)["negative_prompt"] for r in requests] == [neg, neg]
        assert [json.loads(r)["steps"] for r in requests] == [40, 40]


@needs_tools
def test_only_the_hv15_mxfp8_arm_sets_the_former_default(tmp_path):
    from tests.unit.backends._neuron_hv15_toy import HV15_MEASURED_TARGETS

    _, lines = _bench(tmp_path, "hv15_official")
    for line in lines:
        want = ";".join(HV15_MEASURED_TARGETS) if "--quant" in line["argv"] else None
        assert line["env"]["DIFFLET_QUANT_TARGETS"] == want, line["argv"][0]
    assert sum("--quant" in ln["argv"] for ln in lines) == 4


@needs_tools
def test_a_rerun_archives_the_previous_arm_instead_of_keeping_its_outputs(tmp_path):
    """A rerun that fails partway must not leave the earlier run's files where quality.py and
    the CPU fp32 scores read the arm: the earlier run is moved under discarded/ first."""
    root, _ = _bench(tmp_path, "wan21_official", DRY_RUN="first")
    arm = root / "wan21_official" / "bf16"
    assert (arm / "cold_1.mp4").read_text() == "first"
    _bench(tmp_path, "wan21_official", DRY_RUN="second", DRY_FAIL="bf16/cold_1")
    assert not (arm / "cold_1.mp4").exists()  # the rerun's cold_1 failed: no output
    assert (arm / "cold_2.mp4").read_text() == "second"
    [archived] = (root / "discarded").glob("wan21_official-bf16-*")
    assert (archived / "cold_1.mp4").read_text() == "first"


def test_quality_reads_no_output_of_a_failed_step(tmp_path):
    arm = tmp_path / "wan21_official" / "bf16"
    arm.mkdir(parents=True)
    for step, rc in (("cold_1", 1), ("cold_2", 0)):
        (arm / f"{step}.mp4").write_bytes(b"clip")
        (arm / f"{step}.log").write_text(f"... Exit status: {rc}\nexit {rc}\n")
    done = subprocess.run([sys.executable, str(QB / "quality.py"), "--root", str(tmp_path),
                           "wan21_official"], capture_output=True, text=True, timeout=300,
                          env={**os.environ, "PYTHONPATH": str(REPO)})
    assert done.returncode == 0, done.stderr
    md5 = json.loads((tmp_path / "quality.json").read_text())["wan21_official"]["bf16"]["md5"]
    assert md5["cold_1"] is None and md5["cold_2"] is not None
