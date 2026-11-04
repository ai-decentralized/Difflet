"""The CLI's compile / generate / run refuse the tpu backend, with the reason.

TorchTPU compiles just in time and caches the executables itself, so there is no
ahead-of-time artifact for the staged Neuron-toolchain flow to build or load.
"""

import importlib

import pytest

cli = importlib.import_module("difflet.cli.main")  # the module; difflet.cli re-exports main()

WAN_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


@pytest.mark.parametrize("command", ["compile", "generate", "run"])
@pytest.mark.parametrize("how", ["flag", "env"])
def test_aot_commands_refuse_tpu_with_an_explanation(monkeypatch, capsys, command, how):
    argv = [command, "--model-id", WAN_ID]
    if command != "compile":
        argv += ["--prompt", "p", "--output", "o.mp4"]
    if how == "flag":
        argv += ["--backend", "tpu"]
    else:
        monkeypatch.setenv("DIFFLET_BACKEND", "tpu")
    monkeypatch.setattr(cli, "_get_orchestrator",
                        lambda args: pytest.fail("no orchestrator may be built"))

    with pytest.raises(SystemExit) as exc:
        cli.main(argv)

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert f"'difflet {command}' is not supported on the tpu backend" in err
    assert "compiles just in time" in err
    assert "difflet serve" in err
