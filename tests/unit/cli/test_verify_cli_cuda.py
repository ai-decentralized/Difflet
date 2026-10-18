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


def _passing_step(cmd, log_fh, *, timeout, env):
    """A PASS that writes the generate output, as a real ``difflet generate`` does."""
    if "generate" in cmd:
        pathlib.Path(cmd[cmd.index("--output") + 1]).write_bytes(b"x")
    return StepResult(Status.PASS, 0.5, cmd=cmd)


def test_main_writes_results_json_and_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "run_step", _passing_step)
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


WAN = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


def test_wan_cell_is_in_the_matrix_at_40_steps_writing_an_mp4(tmp_path):
    assert (WAN, "tp1") in v.CELLS
    # diffusers' Wan 2.2 default; the registry's serving default_steps=2 is a Trainium smoke.
    assert v.CELL_STEPS[WAN] == 40
    assert v.cell_output(tmp_path, WAN, "tp1") == tmp_path / "wan2.2-t2v-a14b-diffusers_tp1.mp4"
    assert v.cell_output(tmp_path, "Qwen/Qwen-Image", "tp1") == tmp_path / "qwen-image_tp1.png"


def test_generate_is_given_the_per_model_output_path(tmp_path, monkeypatch):
    outputs = {}

    def fake(cmd, log_fh, *, timeout, env):
        if "generate" in cmd:
            outputs[cmd[cmd.index("--model-id") + 1]] = cmd[cmd.index("--output") + 1]
        return _passing_step(cmd, log_fh, timeout=timeout, env=env)

    monkeypatch.setattr(v, "run_step", fake)
    assert v.main(["--run-root", str(tmp_path)]) == 0
    assert outputs[WAN].endswith("_tp1.mp4")
    assert outputs["black-forest-labs/FLUX.1-dev"].endswith("_tp1.png")


def test_a_generate_that_writes_no_output_fails_the_cell(tmp_path, monkeypatch):
    def fake(cmd, log_fh, *, timeout, env):
        # Exit 0 everywhere, but the Wan generate writes only the .pt fallback.
        if "generate" in cmd:
            out = pathlib.Path(cmd[cmd.index("--output") + 1])
            model_id = cmd[cmd.index("--model-id") + 1]
            (out.with_suffix(".pt") if model_id == WAN else out).write_bytes(b"x")
        return StepResult(Status.PASS, 0.5, cmd=cmd)

    monkeypatch.setattr(v, "run_step", fake)
    assert v.main(["--run-root", str(tmp_path)]) == 1
    data = json.loads(next(tmp_path.rglob("results.json")).read_text())
    by_model = {c["model_id"]: c for c in data["cells"]}
    assert by_model["Qwen/Qwen-Image"]["outcome"] == "PASS"
    wan = by_model[WAN]
    assert wan["outcome"] == "FAIL" and wan["steps"]["generate"]["status"] == "FAIL"
    assert ".mp4" in wan["steps"]["generate"]["reason"]


def _generate_steps(tmp_path, monkeypatch, argv):
    seen = {}

    def fake(cmd, log_fh, *, timeout, env):
        if "generate" in cmd:
            seen[cmd[cmd.index("--model-id") + 1]] = cmd[cmd.index("--steps") + 1]
        return _passing_step(cmd, log_fh, timeout=timeout, env=env)

    monkeypatch.setattr(v, "run_step", fake)
    assert v.main(["--run-root", str(tmp_path), *argv]) == 0
    return seen, json.loads(next(tmp_path.rglob("results.json")).read_text())


def test_each_cell_generates_at_its_own_default_steps(tmp_path, monkeypatch):
    # Qwen keeps its 50 (diffusers' default); Flux runs FluxPipeline's 28, Wan WanPipeline's 40.
    seen, data = _generate_steps(tmp_path, monkeypatch, [])
    assert seen == {"Qwen/Qwen-Image": "50", "black-forest-labs/FLUX.1-dev": "28", WAN: "40"}
    by_model = {c["model_id"]: c for c in data["cells"]}
    assert by_model["Qwen/Qwen-Image"]["generate_steps"] == 50
    assert by_model["black-forest-labs/FLUX.1-dev"]["generate_steps"] == 28
    assert by_model[WAN]["generate_steps"] == 40


def test_explicit_steps_override_every_cell(tmp_path, monkeypatch):
    seen, data = _generate_steps(tmp_path, monkeypatch, ["--steps", "3"])
    assert set(seen.values()) == {"3"} and data["generate_steps"] == 3
