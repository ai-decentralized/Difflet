from __future__ import annotations
import os
import sys

import pytest

# What run_stage's torchrun branch adds for the neuron ranks (compile.configure_compile_cache).
NEURON_CACHE_VARS = (
    "TORCH_NEURONX_NEFF_CACHE_DIR",
    "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR",
    "TORCH_NEURONX_HLO_CACHE_DIR",
    "NKI_ENABLE_TRACE_CACHE",
)


def test_sets_neuron_rt_num_cores(monkeypatch):
    captured = {}
    monkeypatch.setattr("subprocess.run", lambda cmd, env, check: captured.update({"env": env}))
    monkeypatch.delenv("NEURON_RT_NUM_CORES", raising=False)
    # a 16-core host, so 8 cores is a genuine under-fill (host-size probing is
    # patched: this suite also runs on 4-core dev boxes and CI containers)
    monkeypatch.setattr(
        "difflet.cli.runner.resolve_available_neuron_core_ids",
        lambda required_num_cores: tuple(range(16)),
    )
    from difflet.cli.runner import run_stage

    run_stage("wan", "transformer", num_cores=8, virtual_core_size=None, cli_args=[])
    assert captured["env"]["NEURON_RT_NUM_CORES"] == "8"


def test_whole_device_stage_leaves_num_cores_unset(monkeypatch):
    captured = {}
    monkeypatch.setattr("subprocess.run", lambda cmd, env, check: captured.update({"env": env}))
    monkeypatch.delenv("NEURON_RT_NUM_CORES", raising=False)
    monkeypatch.setattr(
        "difflet.cli.runner.resolve_available_neuron_core_ids",
        lambda required_num_cores: tuple(range(4)),
    )
    from difflet.cli.runner import run_stage

    # An explicit NEURON_RT_NUM_CORES equal to the whole device is rejected by
    # some driver builds; the default allocation is what it describes anyway.
    run_stage("wan", "transformer", num_cores=4, virtual_core_size=None, cli_args=[])
    assert "NEURON_RT_NUM_CORES" not in captured["env"]


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
    # num_cores (4) covers the whole inherited visible set (4-7), so the
    # stale inherited NEURON_RT_NUM_CORES=8 is dropped rather than forwarded —
    # an explicit whole-device count is rejected by some driver builds.
    assert "NEURON_RT_NUM_CORES" not in captured["env"]
    assert captured["env"]["NEURON_RT_VIRTUAL_CORE_SIZE"] == "2"
    assert "NEURON_LOGICAL_NC_CONFIG" not in captured["env"]
    assert [
        captured["env"][name] for name in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK")
    ] == ["1", "1", "0", "0"]


@pytest.fixture
def captured_launch(monkeypatch, tmp_path):
    """Record the launch instead of spawning it: ``api`` is "run" for the single-process
    ``subprocess.run(cmd, env=env, check=True)``, "popen" for the torchrun branch
    (``_run_torchrun``). No neuron cache variable in the shell; DIFFLET_COMPILE_CACHE is
    ``tmp_path``."""
    captured = {}

    def fake_run(cmd, env, check):
        captured.update(api="run", cmd=cmd, env=env)

    class FakePopen:
        def __init__(self, cmd, env):
            captured.update(api="popen", cmd=cmd, env=env)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("subprocess.Popen", FakePopen)
    for name in NEURON_CACHE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    return captured


def test_torchrun_launch_exports_the_neuron_cache_locations(monkeypatch, tmp_path, captured_launch):
    """torch-neuronx fixes its HLO lock directory (TORCH_NEURONX_HLO_CACHE_DIR, read into
    hlo_cache.CACHE_DIR) when a rank imports torch, before any Difflet code runs there: the
    launcher must hand the ranks every location."""
    from difflet.cli.runner import run_stage

    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")

    run_stage("pkg.mod:Orch", "toy", num_cores=2, virtual_core_size=None, cli_args=[])

    assert captured_launch["api"] == "popen"
    env = captured_launch["env"]
    assert {name: env[name] for name in NEURON_CACHE_VARS} == {
        "TORCH_NEURONX_NEFF_CACHE_DIR": str(tmp_path / "neuron" / "neff"),
        "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": str(tmp_path / "neuron" / "neff_local"),
        "TORCH_NEURONX_HLO_CACHE_DIR": str(tmp_path / "neuron" / "hlo"),
        "NKI_ENABLE_TRACE_CACHE": "0",
    }
    assert not any(name in os.environ for name in NEURON_CACHE_VARS)  # the launcher's own env
    assert not (tmp_path / "neuron").exists()  # torch-neuronx creates the directories


def test_torchrun_launch_keeps_cache_locations_the_shell_exported(
    monkeypatch, tmp_path, captured_launch
):
    from difflet.cli.runner import run_stage

    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("TORCH_NEURONX_HLO_CACHE_DIR", "/shell/hlo")

    run_stage("pkg.mod:Orch", "toy", num_cores=2, virtual_core_size=None, cli_args=[])

    env = captured_launch["env"]
    assert env["TORCH_NEURONX_HLO_CACHE_DIR"] == "/shell/hlo"
    assert env["TORCH_NEURONX_NEFF_CACHE_DIR"] == str(tmp_path / "neuron" / "neff")


@pytest.mark.parametrize("backend", [None, "trainium", "tpu", "cpu"])
def test_single_process_launch_adds_no_neuron_cache_variables(
    monkeypatch, captured_launch, backend
):
    from difflet.cli.runner import run_stage

    if backend is None:
        monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    else:
        monkeypatch.setenv("DIFFLET_BACKEND", backend)

    run_stage("Org/Model", "transformer", num_cores=4, virtual_core_size=None, cli_args=[])

    assert captured_launch["api"] == "run"
    assert [name for name in NEURON_CACHE_VARS if name in captured_launch["env"]] == []
