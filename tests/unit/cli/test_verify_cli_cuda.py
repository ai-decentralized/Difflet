"""scripts/verify_cli_cuda.py — the CUDA column of the CLI matrix. Pure logic: no subprocess."""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))

import verify_cli_cuda as v  # noqa: E402
from verify_cli import Status, StepResult  # noqa: E402


def test_commands_select_the_cuda_backend_and_tp1(tmp_path):
    cmds = v.build_cmds(
        "Qwen/Qwen-Image",
        prompt="a cat",
        output=str(tmp_path / "q.png"),
        steps=4,
        seed=0,
        cache_dir=str(tmp_path),
    )
    assert cmds["download"][-2:] == ["--model-id", "Qwen/Qwen-Image"]
    compile_cmd = cmds["compile"]
    assert compile_cmd[compile_cmd.index("--tp-degree") + 1] == "1"
    gen = cmds["generate"]
    assert gen[gen.index("--steps") + 1] == "4" and gen[gen.index("--seed") + 1] == "0"
    assert v.cuda_env()["DIFFLET_BACKEND"] == "cuda"


def test_outcome_requires_every_step_to_pass():
    ok = StepResult(Status.PASS, 1.0)
    assert v.cell_outcome({"download": ok, "compile": ok, "generate": ok}) is Status.PASS
    bad = StepResult(Status.FAIL, 1.0, "rc=1")
    assert v.cell_outcome({"download": ok, "compile": ok, "generate": bad}) is Status.FAIL


def test_main_writes_results_json_and_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(
        v, "run_step", lambda cmd, log_fh, *, timeout, env: StepResult(Status.PASS, 0.5, cmd=cmd)
    )
    rc = v.main(["--run-root", str(tmp_path), "--steps", "2"])
    assert rc == 0
    data = json.loads(next(tmp_path.rglob("results.json")).read_text())
    assert data["backend"] == "cuda" and data["cells"][0]["outcome"] == "PASS"
    assert data["cells"][0]["steps"]["generate"]["status"] == "PASS"


def test_main_exit_code_is_one_on_a_failed_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(
        v,
        "run_step",
        lambda cmd, log_fh, *, timeout, env: StepResult(Status.FAIL, 0.1, "rc=1", cmd=cmd),
    )
    assert v.main(["--run-root", str(tmp_path), "--steps", "2"]) == 1


def test_flux_cell_is_in_the_matrix():
    assert ("Qwen/Qwen-Image", "tp1") in v.CELLS
    assert ("black-forest-labs/FLUX.1-dev", "tp1") in v.CELLS


def _generate_steps(tmp_path, monkeypatch, argv):
    seen = {}

    def fake(cmd, log_fh, *, timeout, env):
        if "generate" in cmd:
            seen[cmd[cmd.index("--model-id") + 1]] = cmd[cmd.index("--steps") + 1]
        return StepResult(Status.PASS, 0.5, cmd=cmd)

    monkeypatch.setattr(v, "run_step", fake)
    assert v.main(["--run-root", str(tmp_path), *argv]) == 0
    return seen, json.loads(next(tmp_path.rglob("results.json")).read_text())


def test_each_cell_generates_at_its_own_default_steps(tmp_path, monkeypatch):
    # Qwen keeps its 50 (diffusers' default); Flux runs FluxPipeline's 28.
    seen, data = _generate_steps(tmp_path, monkeypatch, [])
    assert seen == {"Qwen/Qwen-Image": "50", "black-forest-labs/FLUX.1-dev": "28"}
    by_model = {c["model_id"]: c for c in data["cells"]}
    assert by_model["Qwen/Qwen-Image"]["generate_steps"] == 50
    assert by_model["black-forest-labs/FLUX.1-dev"]["generate_steps"] == 28


def test_explicit_steps_override_every_cell(tmp_path, monkeypatch):
    seen, data = _generate_steps(tmp_path, monkeypatch, ["--steps", "3"])
    assert set(seen.values()) == {"3"} and data["generate_steps"] == 3
