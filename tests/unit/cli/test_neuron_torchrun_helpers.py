"""The driver helpers the neuron CLI orchestrators share (``difflet/cli/orchestrators/torchrun.py``),
torch-free. The stage bodies (``run_generate_stage``, ``run_vae_stage``) run in the Wan,
Qwen-Image and LTX-2 CLI tests through their fake pipeline and CPU decoders."""

import argparse

import pytest

from difflet.cli.orchestrators import torchrun as tr


def _args(**overrides):
    base = dict(tp_degree=None, cp_degree=None, cp_mode="gather_kv", cfg_parallel=False,
                sp_enabled=False, shapes=None, host_vae=False, device_vae=False,
                teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None,
                teacache_calibration=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_unsupported_options_name_teacache_and_several_shapes():
    assert tr.torchrun_unsupported_options(_args()) == []
    assert tr.torchrun_unsupported_options(_args(shapes="480x832x9")) == []
    assert tr.torchrun_unsupported_options(
        _args(teacache_cadence=2, shapes="480x832x9, 480x832x13")
    ) == ["--teacache-cadence", "--shapes with more than one shape"]


def test_refusals_print_one_line_and_exit_2(monkeypatch, capsys):
    tr.refuse(None)
    with pytest.raises(SystemExit) as exc:
        tr.refuse("bad pair")
    assert exc.value.code == 2
    assert capsys.readouterr().err == "Error: bad pair; nothing was launched.\n"
    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    tr.refuse_torchrun_conflicts(["ignored off the torchrun backends"])
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    with pytest.raises(SystemExit):
        tr.refuse_torchrun_conflicts(["first", "second"])
    assert "Error: first; nothing was launched." in capsys.readouterr().err


def test_shared_conflicts_cores_and_decode_placement():
    assert tr.vae_placement_conflict(_args(host_vae=True, device_vae=True)) == (
        "--host-vae and --device-vae are mutually exclusive")
    assert tr.vae_placement_conflict(_args(host_vae=True)) is None
    assert tr.cfg_cp_conflict(_args(cfg_parallel=True, cp_degree=2)) == (
        "--cfg-parallel and --cp-degree > 1 are mutually exclusive")
    assert tr.cfg_cp_conflict(_args(cfg_parallel=True)) is None
    assert tr.replica_cores(_args()) == 4
    assert tr.replica_cores(_args(tp_degree=2, cp_degree=2)) == 4
    assert tr.replica_cores(_args(tp_degree=2, cfg_parallel=True)) == 4
    for default in (True, False):
        assert tr.decodes_on_device(_args(), default) is default
        assert tr.decodes_on_device(_args(device_vae=True), default) is True
        assert tr.decodes_on_device(_args(host_vae=True, device_vae=True), default) is False


def test_hbm_checks_before_torchrun(capsys):
    gib = 1024**3

    def fits(model_dir, *, tp_degree):
        estimate = 40 * gib // tp_degree
        return estimate <= 20 * gib, estimate, 20 * gib

    def missing():
        raise OSError("not local")

    tr.refuse_dit_overflow("Toy", missing, fits, 1)  # weights not local: the app decides
    tr.refuse_dit_overflow("Toy", lambda: "/m", fits, 2)  # 20 GiB: fits
    with pytest.raises(SystemExit) as exc:
        tr.refuse_dit_overflow("Toy", lambda: "/m", fits, 1)
    assert exc.value.code == 2
    assert capsys.readouterr().err == (
        "Error: the Toy DiT needs at least 40.0 GiB of bf16 weights per core at tp=1, over the "
        "20.0 GiB budget of one NeuronCore (every DP replica is its own TP group); use "
        "--tp-degree 2 or more; nothing was launched.\n")
    assert tr.device_encoder_overflow(missing, fits, 1) is None
    assert tr.device_encoder_overflow(lambda: "/m", fits, 2) is None
    assert tr.device_encoder_overflow(lambda: "/m", fits, 1) == (
        "needs at least 40.0 GiB per core beside the DiT at tp=1, over the 20.0 GiB budget")


def test_stage_keys_and_parallel_request(monkeypatch):
    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    assert tr.torchrun_stage_keys({"a": 1}) == {"a": 1}
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    assert tr.torchrun_stage_keys({"a": 1}) == {"a": 1, "backend": "neuron",
                                                "exec_mode": "compile"}
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "eager")
    assert tr.torchrun_stage_keys({"a": 1})["exec_mode"] == "eager"
    p = tr.torchrun_parallel(_args(tp_degree=2, cp_degree=2, cp_mode=None, sp_enabled=None))
    assert (p.tp_degree, p.cp_degree, p.cp_mode, p.cfg_parallel_enabled, p.sp_enabled) == (
        2, 2, "gather_kv", False, False)
