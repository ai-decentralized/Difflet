"""The CLI's DP capacity check counts the inherited NEURON_RT_VISIBLE_CORES, the cores the
router hands out, and a router refusal ends in the CLI's usual ``Error:`` line.

trn3, 2026-11-08: ``--dp 2`` at tp4 on NEURON_RT_VISIBLE_CORES=20-23 passed the early check
whenever NEURON_RT_NUM_CORES said 8 or neuron-ls was missing, and the router's refusal then
surfaced as a traceback (after a full compile, for ``difflet run``).
"""

import importlib
import json

import pytest

# difflet.cli.__init__ re-exports main() (the function), shadowing the submodule attribute.
cli_main = importlib.import_module("difflet.cli.main")

_WAN = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


@pytest.fixture(autouse=True)
def _trainium_without_neuron_ls(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "trainium")
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    monkeypatch.delenv("NEURON_RT_NUM_CORES", raising=False)

    def no_neuron_ls():
        raise AssertionError("the budget must come from the environment, not neuron-ls")

    monkeypatch.setattr("difflet.planner.hardware.detect_hardware", no_neuron_ls)


def _args(command="generate", dp=2, tp=4, extra=()):
    args = cli_main._build_parser().parse_args(
        [command, "--model-id", _WAN, "--dp", str(dp), "--tp-degree", str(tp),
         "--prompt", "x", "--output", "x.mp4", *extra]
    )
    args.command = command
    return args


def test_the_inherited_visible_cores_set_the_budget_before_num_cores(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "20-23")
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "8")
    assert cli_main._resolve_total_cores(_args()) == (4, "NEURON_RT_VISIBLE_CORES=20-23")
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "0,2,4-5")
    assert cli_main._resolve_total_cores(_args())[0] == 4


def test_a_malformed_visible_cores_value_falls_back_to_num_cores(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "abc")
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "8")
    assert cli_main._resolve_total_cores(_args()) == (8, "NEURON_RT_NUM_CORES")


@pytest.mark.parametrize("command", ["generate", "run"])
@pytest.mark.parametrize("num_cores", [None, "8"])
def test_dp_on_too_few_inherited_cores_fails_the_early_check(
    monkeypatch, capsys, command, num_cores
):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "20-23")
    if num_cores is not None:
        monkeypatch.setenv("NEURON_RT_NUM_CORES", num_cores)
    with pytest.raises(SystemExit) as exc:
        cli_main._validate_dp(_args(command))  # 2 x 4 = 8 > 4
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "needs 8 NeuronCores but only 4 are available (NEURON_RT_VISIBLE_CORES=20-23)" in err


def test_dp_that_fits_the_inherited_cores_passes(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "20-27")
    monkeypatch.setenv("NEURON_RT_NUM_CORES", "4")  # the visible cores win
    cli_main._validate_dp(_args())


def test_a_router_refusal_prints_an_error_and_exits_1(monkeypatch, tmp_path, capsys):
    # The backstop: the real router refuses before starting any worker.
    from difflet.cli.dp import router

    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "20-23")
    monkeypatch.setattr(
        router.subprocess, "Popen", lambda *a, **k: pytest.fail("no worker may start")
    )
    req = tmp_path / "r.jsonl"
    req.write_text(
        "".join(json.dumps({"prompt": "x", "output": f"x{i}.png"}) + "\n" for i in range(2))
    )
    args = _args(extra=["--work-dir", str(tmp_path / "work")])
    args.prompt = args.output = None
    args.requests = str(req)
    with pytest.raises(SystemExit) as exc:
        cli_main._dispatch_dp(args)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: [dp-router] dp=2 x 4 cores per replica needs 8 NeuronCores")
    assert "Traceback" not in err
