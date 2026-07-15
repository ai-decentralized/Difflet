from __future__ import annotations
import sys


def test_sets_neuron_rt_num_cores(monkeypatch):
    captured = {}
    monkeypatch.setattr("subprocess.run", lambda cmd, env, check: captured.update({"env": env}))
    monkeypatch.delenv("NEURON_RT_NUM_CORES", raising=False)
    from difflet.cli.runner import run_stage

    run_stage("wan", "transformer", num_cores=8, virtual_core_size=None, cli_args=[])
    assert captured["env"]["NEURON_RT_NUM_CORES"] == "8"


def test_respects_existing_neuron_rt_num_cores(monkeypatch):
    captured = {}
    monkeypatch.setattr("subprocess.run", lambda cmd, env, check: captured.update({"env": env}))
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "16")
    from difflet.cli.runner import run_stage

    run_stage("wan", "transformer", num_cores=4, virtual_core_size=None, cli_args=[])
    assert captured["env"]["NEURON_RT_NUM_CORES"] == "16"


def test_sets_virtual_core_size_when_provided(monkeypatch):
    captured = {}
    monkeypatch.setattr("subprocess.run", lambda cmd, env, check: captured.update({"env": env}))
    from difflet.cli.runner import run_stage

    run_stage("hunyuan_video", "clip", num_cores=1, virtual_core_size=2, cli_args=[])
    assert captured["env"]["NEURON_RT_VIRTUAL_CORE_SIZE"] == "2"


def test_does_not_set_virtual_core_size_when_none(monkeypatch):
    captured = {}
    monkeypatch.setattr("subprocess.run", lambda cmd, env, check: captured.update({"env": env}))
    from difflet.cli.runner import run_stage

    run_stage("wan", "transformer", num_cores=4, virtual_core_size=None, cli_args=[])
    assert "NEURON_RT_VIRTUAL_CORE_SIZE" not in captured["env"]


def test_builds_correct_command(monkeypatch):
    captured = {}
    monkeypatch.setattr("subprocess.run", lambda cmd, env, check: captured.update({"cmd": cmd}))
    from difflet.cli.runner import run_stage

    run_stage(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "transformer",
        num_cores=4,
        virtual_core_size=None,
        cli_args=["--model-id", "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "--tp-degree", "4"],
    )
    cmd = captured["cmd"]
    assert cmd[0] == sys.executable
    assert "-m" in cmd
    assert "difflet.cli.stage" in cmd
    assert "--orchestrator" in cmd
    assert "Wan-AI/Wan2.2-T2V-A14B-Diffusers" in cmd
    assert "--stage" in cmd
    assert "transformer" in cmd
    assert "--model-id" in cmd


def test_passes_check_true(monkeypatch):
    captured = {}
    monkeypatch.setattr("subprocess.run", lambda cmd, env, check: captured.update({"check": check}))
    from difflet.cli.runner import run_stage

    run_stage("wan", "vae", num_cores=1, virtual_core_size=None, cli_args=[])
    assert captured["check"] is True


def test_strict_environment_preserves_inherited_visible_cores(monkeypatch):
    captured = {}
    monkeypatch.setattr("subprocess.run", lambda cmd, env, check: captured.update({"env": env}))
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "4-7")
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "8")
    monkeypatch.setenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1")
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    monkeypatch.setenv("WORLD_SIZE", "8")
    from difflet.cli.runner import run_stage

    run_stage(
        "Qwen/Qwen-Image",
        "generate",
        num_cores=4,
        virtual_core_size=2,
        cli_args=[],
        strict_environment=True,
    )

    assert captured["env"]["NEURON_RT_VISIBLE_CORES"] == "4,5,6,7"
    assert captured["env"]["NEURON_RT_NUM_CORES"] == "4"
    assert captured["env"]["NEURON_RT_VIRTUAL_CORE_SIZE"] == "2"
    assert "NEURON_LOGICAL_NC_CONFIG" not in captured["env"]
    assert [
        captured["env"][name] for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK")
    ] == ["1", "1", "0", "0"]
