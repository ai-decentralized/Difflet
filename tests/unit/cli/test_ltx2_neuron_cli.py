"""The LTX-2 CLI driver on a torchrun backend (neuron): what it refuses before torchrun starts,
the stage launch it plans, and the arguments it forwards. Torch-free checks; no stage runs."""

import argparse

import pytest

from difflet.cli.orchestrators import ltx_2 as cli


def _args(**overrides):
    base = dict(
        model_id="Lightricks/LTX-2", tp_degree=4, cp_degree=1, cfg_parallel=False,
        sp_enabled=False, height=480, width=704, num_frames=49, steps=20, guidance_scale=1.0,
        seed=42, prompt="a fox", negative_prompt=None, output="/tmp/o.mp4", cache_dir="/tmp/c",
        work_dir=None, keep_work_dir=False, revision="47da56e2", shapes=None,
        teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None,
        teacache_calibration=None, device_text_encoder=False, requests_dir=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.parametrize(
    ("overrides", "flag"),
    [
        ({"teacache_cadence": 2}, "--teacache-cadence"),
        ({"shapes": "480x704x49,512x768x121"}, "--shapes with more than one shape"),
        ({"cp_degree": 2}, "--cp-degree > 1"),
        ({"sp_enabled": True}, "--sp"),
    ],
)
def test_driver_lists_what_the_ranks_do_not_run(overrides, flag):
    assert flag in cli._torchrun_unsupported_options(_args(**overrides))
    assert cli._torchrun_unsupported_options(_args()) == []


def test_cfg_parallel_needs_guidance_when_generating():
    assert cli._torchrun_conflicts(_args(cfg_parallel=True), generating=True)
    assert not cli._torchrun_conflicts(_args(cfg_parallel=True), generating=False)
    assert not cli._torchrun_conflicts(_args(cfg_parallel=True, guidance_scale=3.0),
                                       generating=True)
    # The Trainium default guidance (3.5) applies when none is given.
    assert not cli._torchrun_conflicts(_args(cfg_parallel=True, guidance_scale=None),
                                       generating=True)


def test_rejection_exits_2_before_anything_launches(monkeypatch, capsys):
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    launched = []
    monkeypatch.setattr(cli.runner, "run_stage", lambda *a, **k: launched.append(a))
    orch = cli.LTX2Orchestrator(_args(cp_degree=2))
    with pytest.raises(SystemExit) as exc:
        orch.generate()
    assert exc.value.code == 2 and launched == []
    assert "--cp-degree > 1" in capsys.readouterr().err
    orch = cli.LTX2Orchestrator(_args(cfg_parallel=True, guidance_scale=1.0))
    with pytest.raises(SystemExit) as exc:
        orch.generate()
    assert exc.value.code == 2 and launched == []


def test_compile_on_neuron_is_one_generate_stage_on_tp_times_cfg_cores(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.delenv("DIFFLET_COMPILE_CACHE", raising=False)
    launched = []
    monkeypatch.setattr(cli.runner, "run_stage",
                        lambda model, stage, **k: launched.append((model, stage, k)))
    monkeypatch.setattr(cli.LTX2Orchestrator, "_check_dit_fits", lambda self: None)
    cli.LTX2Orchestrator(_args(tp_degree=2, cfg_parallel=True)).compile()
    [(model, stage, kwargs)] = launched
    assert (model, stage, kwargs["num_cores"]) == ("Lightricks/LTX-2", "generate", 4)
    forwarded = kwargs["cli_args"]
    assert forwarded[forwarded.index("--stage-mode") + 1] == "compile"
    assert "--cfg-parallel" in forwarded
    assert forwarded[forwarded.index("--revision") + 1] == "47da56e2"


def test_decode_placement(monkeypatch, tmp_path):
    """Host decode by default (the Trainium placement); --device-vae adds a one-core vae stage
    after the generate stage, to compile and to generate; --host-vae with it is refused."""
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.delenv("DIFFLET_COMPILE_CACHE", raising=False)
    launched, decoded = [], []
    monkeypatch.setattr(cli.runner, "run_stage",
                        lambda model, stage, **k: launched.append((stage, k["num_cores"])))
    monkeypatch.setattr(cli.LTX2Orchestrator, "_check_dit_fits", lambda self: None)
    monkeypatch.setattr(cli.LTX2Orchestrator, "_decode_on_host",
                        lambda self, work_dir: decoded.append(str(work_dir)))
    assert not cli._decodes_on_device(_args())
    cli.LTX2Orchestrator(_args(work_dir=str(tmp_path / "w"))).generate()
    assert launched == [("generate", 4)] and decoded == [str(tmp_path / "w")]
    launched.clear()
    cli.LTX2Orchestrator(_args(device_vae=True)).compile()
    assert launched == [("generate", 4), ("vae", 1)]
    launched.clear()
    cli.LTX2Orchestrator(_args(device_vae=True, work_dir=str(tmp_path / "w2"))).generate()
    assert launched == [("generate", 4), ("vae", 1)] and len(decoded) == 1
    assert cli._torchrun_conflicts(_args(device_vae=True, host_vae=True), generating=False)


def test_stage_cache_inputs_name_the_stage():
    orch = cli.LTX2Orchestrator(_args())
    generate = orch._stage_cache_inputs(_args())
    vae = orch._stage_cache_inputs(_args(stage="vae"))
    assert generate["component"] == "ltx_2_generate" and generate["tp"] == 4
    assert vae["component"] == "ltx_2_vae" and "tp" not in vae


def test_dit_fit_check_refuses_tp1(monkeypatch, tmp_path, capsys):
    """tp1 needs about 35 GiB of bf16 DiT per core, over a 24 GiB core: refused in the driver."""
    import json
    import struct

    transformer = tmp_path / "transformer"
    transformer.mkdir()
    # A header-only safetensors file declaring 18.9 B bf16 elements (no tensor data is read).
    header = json.dumps({"w": {"dtype": "BF16", "shape": [18_876_000_000],
                               "data_offsets": [0, 0]}}).encode()
    (transformer / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", "24")
    monkeypatch.setattr(cli.LTX2Orchestrator, "_model_dir", lambda self: str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        cli.LTX2Orchestrator(_args(tp_degree=1))._check_dit_fits()
    assert exc.value.code == 2
    assert "--tp-degree 2 or more" in capsys.readouterr().err
    cli.LTX2Orchestrator(_args(tp_degree=4))._check_dit_fits()  # 8.8 GiB per core: fits


def _header_only_checkpoint(directory, name, numel):
    import json
    import struct

    directory.mkdir(parents=True, exist_ok=True)
    header = json.dumps({name: {"dtype": "BF16", "shape": [numel], "data_offsets": [0, 0]}})
    (directory / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header.encode())


def test_device_text_encoder_placement(monkeypatch, tmp_path, capsys):
    """--device-text-encoder is forwarded at tp4, where Gemma-3 (11.8 B) and the DiT (18.9 B)
    fit beside each other, and dropped with a note at tp2, where they do not."""
    _header_only_checkpoint(tmp_path / "transformer", "w", 18_876_000_000)
    _header_only_checkpoint(tmp_path / "text_encoder", "language_model.model.w", 11_770_000_000)
    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", "24")
    monkeypatch.setattr(cli.LTX2Orchestrator, "_model_dir", lambda self: str(tmp_path))
    orch = cli.LTX2Orchestrator(_args(device_text_encoder=True))
    orch._place_text_encoder()
    assert orch.args.device_text_encoder is True
    assert "--device-text-encoder" in orch._shared_cli_args("generate")
    orch = cli.LTX2Orchestrator(_args(device_text_encoder=True, tp_degree=2))
    orch._place_text_encoder()
    assert orch.args.device_text_encoder is False
    assert "encoding on rank 0's host instead" in capsys.readouterr().out
    assert "--device-text-encoder" not in orch._shared_cli_args("generate")


def test_stage_dispatch_knows_ltx_2():
    from difflet.cli import stage

    assert stage._ORCHESTRATOR_MAP["Lightricks/LTX-2"].endswith("ltx_2.LTX2Orchestrator")
