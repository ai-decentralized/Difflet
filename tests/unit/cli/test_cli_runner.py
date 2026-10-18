from __future__ import annotations
import sys


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


# ------------------------------------------------------------------ run_torchrun (CUDA F2)

_RANK_ARGV = ["-m", "difflet.cli.main", "generate", "--model-id", "x", "--cfg-parallel"]


def test_run_torchrun_launches_a_standalone_torchrun_of_the_module(monkeypatch):
    import subprocess

    captured = {}

    def fake(cmd, env, check):
        captured.update(cmd=cmd, env=env, check=check)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("subprocess.run", fake)
    from difflet.cli.runner import run_torchrun

    assert run_torchrun(2, list(_RANK_ARGV), env={"DIFFLET_BACKEND": "cuda"}) == 0
    assert captured["cmd"] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        *_RANK_ARGV,
    ]
    assert captured["env"] == {"DIFFLET_BACKEND": "cuda"}
    # The return code is the caller's to judge: no CalledProcessError.
    assert captured["check"] is False


def test_run_torchrun_returns_the_child_return_code(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        "subprocess.run", lambda cmd, env, check: subprocess.CompletedProcess(cmd, 3)
    )
    from difflet.cli.runner import run_torchrun

    assert run_torchrun(2, list(_RANK_ARGV)) == 3


def test_run_torchrun_inherits_the_environment_by_default(monkeypatch):
    import os
    import subprocess

    captured = {}
    monkeypatch.setenv("DIFFLET_TORCHRUN_PROBE", "1")
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, env, check: captured.update(env=env) or subprocess.CompletedProcess(cmd, 0),
    )
    from difflet.cli.runner import run_torchrun

    run_torchrun(2, list(_RANK_ARGV))
    assert captured["env"] == dict(os.environ)
    assert captured["env"]["DIFFLET_TORCHRUN_PROBE"] == "1"


def test_torchrun_parses_the_command_as_a_module_launch(monkeypatch):
    """The launcher's own parser reads the command: ``-m`` is torchrun's ``--module`` and
    everything after the module name reaches the ranks untouched."""
    import subprocess

    captured = {}
    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, env, check: captured.update(cmd=cmd) or subprocess.CompletedProcess(cmd, 0),
    )
    from difflet.cli.runner import run_torchrun

    run_torchrun(2, list(_RANK_ARGV))
    from torch.distributed.run import get_args_parser

    cmd = captured["cmd"]
    assert cmd[1:3] == ["-m", "torch.distributed.run"]
    parsed = get_args_parser().parse_args(cmd[3:])
    assert parsed.standalone is True and parsed.nproc_per_node == "2"
    assert parsed.module is True and parsed.training_script == "difflet.cli.main"
    assert parsed.training_script_args == ["generate", "--model-id", "x", "--cfg-parallel"]
