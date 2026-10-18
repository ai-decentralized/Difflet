"""The difflet-cuda-verify skill's scripts on a CPU host: every shell wrapper parses (`bash -n`)
and answers `--help`, the pod tool answers `--help`, and `run_gates.sh --dry-run` prints each
model's gate sequence with the runbooks' env and flags; the skill's files exist, its frontmatter
parses, every script its text names is in the repo, and its evals mirror the Trainium schema.
Nothing here touches a GPU, a pod or the network."""

import json
import os
import pathlib
import py_compile
import re
import subprocess
import sys

import pytest
import yaml

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


def run_gates_for_real(args, env_extra=None):
    """run_gates.sh without --dry-run, for the early exits that stop before any GPU step."""
    env = {k: v for k, v in os.environ.items() if k != "HF_HOME"}
    return subprocess.run(
        ["bash", str(SKILL / "scripts" / "run_gates.sh"), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**env, **(env_extra or {})},
    )


def test_run_gates_snapshot_without_transformer_still_writes_the_done_marker(tmp_path):
    snap = tmp_path / "hf" / "hub" / "models--Qwen--Qwen-Image" / "snapshots" / "abc"
    snap.mkdir(parents=True)  # a partial download: no transformer/
    out = tmp_path / "logs" / "qwen"
    proc = run_gates_for_real(["qwen", str(snap), "--out", str(out)])
    assert proc.returncode == 2 and "has no transformer/" in proc.stderr
    lines = (out / "gates.txt").read_text().splitlines()
    assert lines[-1].startswith("GATES DONE qwen overall_rc=2 (") and "transformer/" in lines[-1]


def test_run_gates_auto_without_a_snapshot_still_writes_the_done_marker(tmp_path):
    out = tmp_path / "logs" / "wan"
    proc = run_gates_for_real(
        ["wan", "auto", "--out", str(out)], {"HF_HOME": str(tmp_path / "empty-hf")}
    )
    assert proc.returncode == 2 and "no models--Wan-AI" in proc.stderr
    assert (
        (out / "gates.txt")
        .read_text()
        .splitlines()[-1]
        .startswith("GATES DONE wan overall_rc=2 (no models--Wan-AI")
    )


def test_pod_setup_ends_with_its_done_marker_even_on_failure(tmp_path):
    proc = subprocess.run(  # not a Difflet checkout: exits 2 before touching anything
        ["bash", str(SKILL / "scripts" / "pod_setup.sh")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "REPO": str(tmp_path)},
    )
    assert proc.returncode == 2 and "not a Difflet checkout" in proc.stderr
    assert proc.stdout.splitlines()[-1] == "SETUP DONE rc=2"


# --- the skill's text (F8 task 2) ---------------------------------------------------------

SKILL_FILES = [
    "SKILL.md",
    "references/feature-catalog.md",
    "references/pod-runbook.md",
    "references/failure-triage.md",
    "references/gotchas.md",
    "references/evidence-doc-template.md",
    "evals/evals.json",
    "scripts/hf_device_login.py",
]
SKILL_DOCS = ["SKILL.md", *[f for f in SKILL_FILES if f.startswith("references/")]]
# Where a bare script name may live (`$S/run_gates.sh`, `cuda_smoke.py`, a gate test's name).
SCRIPT_DIRS = [SKILL / "scripts", ROOT / "scripts", ROOT / "tests" / "numerical"]
# A path token: optional directories, then a .sh / .py file name; never a glob (`*_x.py`).
SCRIPT_TOKEN = re.compile(r"(?<![\w.*-])((?:[\w$~.-]+/)*)([\w-]+\.(?:sh|py))\b")


@pytest.mark.parametrize("rel", SKILL_FILES)
def test_skill_file_exists(rel):
    path = SKILL / rel
    assert path.is_file() and path.stat().st_size > 0, path


def test_skill_frontmatter_parses():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front = yaml.safe_load(text.split("---\n", 2)[1])
    assert front["name"] == "difflet-cuda-verify"
    assert isinstance(front["description"], str) and "CUDA" in front["description"]
    assert len(text.splitlines()) <= 200


def named_scripts(rel):
    """(token, resolved path or None) for every script path or name a skill doc mentions."""
    text = (SKILL / rel).read_text(encoding="utf-8")
    found = []
    for match in SCRIPT_TOKEN.finditer(text):
        dirs, name = match.groups()
        token = dirs + name
        if dirs.startswith("~") or dirs.startswith("<"):
            continue  # a path on the pod (`~/hf_device_login.py`), not in the repo
        if dirs in ("", "$S/"):
            candidates = [d / name for d in SCRIPT_DIRS]
        elif "scripts/" in dirs or dirs.startswith("tests/"):
            candidates = [ROOT / token, SKILL / token]
        else:
            continue
        found.append((token, next((c for c in candidates if c.is_file()), None)))
    return found


@pytest.mark.parametrize("rel", SKILL_DOCS)
def test_every_script_the_skill_names_exists(rel):
    found = named_scripts(rel)
    missing = sorted({token for token, path in found if path is None})
    assert not missing, f"{rel} names scripts that are not in the repo: {missing}"


def test_skill_md_names_the_pod_tool_and_the_wrappers():
    tokens = {token for token, _ in named_scripts("SKILL.md")}
    for name in ["scripts/yottalabs_pod.py", "pod_setup.sh", "ship_branch.sh", "run_gates.sh"]:
        assert any(t.endswith(name) for t in tokens), name


@pytest.mark.parametrize("rel", SKILL_DOCS)
def test_skill_docs_keep_prose_lines_short(rel):
    long_lines = [
        n
        for n, line in enumerate((SKILL / rel).read_text(encoding="utf-8").splitlines(), 1)
        if len(line) > 100 and not line.lstrip().startswith("|")
    ]
    assert not long_lines, f"{rel}: lines over 100 characters: {long_lines}"


def test_hf_device_login_compiles(tmp_path):
    py_compile.compile(
        str(SKILL / "scripts" / "hf_device_login.py"),
        cfile=str(tmp_path / "hf_device_login.pyc"),
        doraise=True,
    )


def test_evals_mirror_the_trainium_schema():
    ours = json.loads((SKILL / "evals" / "evals.json").read_text(encoding="utf-8"))
    theirs = json.loads(
        (ROOT / ".claude/skills/difflet-device-verify/evals/evals.json").read_text(encoding="utf-8")
    )
    assert ours["skill_name"] == "difflet-cuda-verify"
    assert len(ours["evals"]) == 2
    assert [e["id"] for e in ours["evals"]] == [0, 1]
    for case in ours["evals"]:
        assert set(case) == set(theirs["evals"][0]), set(case) ^ set(theirs["evals"][0])
        assert case["prompt"] and case["expected_output"] and len(case["expectations"]) >= 5
