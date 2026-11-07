"""The LTX-2 CLI driver on a torchrun backend (neuron): what it refuses before torchrun starts,
the stage launch it plans, and the arguments it forwards. Torch-free checks; no stage runs."""

import argparse

import pytest

from difflet.cli.orchestrators import ltx_2 as cli


def _args(**overrides):
    base = dict(
        model_id="Lightricks/LTX-2", tp_degree=4, cp_degree=1, cp_mode="gather_kv",
        cfg_parallel=False,
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
        ({"cp_degree": 2, "cp_mode": "ring"}, "--cp-mode ring"),
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
    orch = cli.LTX2Orchestrator(_args(cp_degree=2, cp_mode="ring"))
    with pytest.raises(SystemExit) as exc:
        orch.generate()
    assert exc.value.code == 2 and launched == []
    assert "--cp-mode ring" in capsys.readouterr().err
    orch = cli.LTX2Orchestrator(_args(cp_degree=2, cfg_parallel=True, guidance_scale=3.0))
    with pytest.raises(SystemExit) as exc:
        orch.generate()
    assert exc.value.code == 2 and launched == []
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


def test_context_parallel_launch_is_tp_times_cp_ranks(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.delenv("DIFFLET_COMPILE_CACHE", raising=False)
    launched = []
    monkeypatch.setattr(cli.runner, "run_stage",
                        lambda model, stage, **k: launched.append((stage, k)))
    monkeypatch.setattr(cli.LTX2Orchestrator, "_check_dit_fits", lambda self: None)
    cli.LTX2Orchestrator(_args(tp_degree=2, cp_degree=2, cp_mode="ulysses")).compile()
    [(stage, kwargs)] = launched
    assert stage == "generate" and kwargs["num_cores"] == 4
    forwarded = kwargs["cli_args"]
    assert forwarded[forwarded.index("--cp-degree") + 1] == "2"
    assert forwarded[forwarded.index("--cp-mode") + 1] == "ulysses"
    inputs = cli.LTX2Orchestrator(_args())._stage_cache_inputs(
        _args(tp_degree=2, cp_degree=2, cp_mode="ulysses"))
    assert (inputs["cp"], inputs["cp_mode"]) == (2, "ulysses")


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


def test_device_text_encoder_is_the_default_and_host_opts_out(monkeypatch, tmp_path):
    _header_only_checkpoint(tmp_path / "transformer", "w", 18_876_000_000)
    _header_only_checkpoint(tmp_path / "text_encoder", "language_model.model.w", 11_770_000_000)
    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", "24")
    monkeypatch.setattr(cli.LTX2Orchestrator, "_model_dir", lambda self: str(tmp_path))
    orch = cli.LTX2Orchestrator(_args())
    orch._place_text_encoder()
    assert orch.args.device_text_encoder is True
    orch = cli.LTX2Orchestrator(_args(host_text_encoder=True))
    orch._place_text_encoder()
    assert orch.args.device_text_encoder is False
    assert "--device-text-encoder" not in orch._shared_cli_args("generate")


def test_stage_dispatch_knows_ltx_2():
    from difflet.cli import stage

    assert stage._ORCHESTRATOR_MAP["Lightricks/LTX-2"].endswith("ltx_2.LTX2Orchestrator")


def test_stage_and_dp_worker_lines_carry_quant():
    """A stage launch or a DP worker that dropped --quant would run the bf16 DiT against the
    fp8 stage key: both lines carry the flags, and only when given (a bf16 line is unchanged)."""
    from difflet.cli.dp.router import worker_cli_args

    args = _args(quant="mxfp8", quant_compute="dequant", dp=2)
    stage = cli.LTX2Orchestrator(args)._shared_cli_args("generate", work_dir="/w")
    assert stage[stage.index("--quant") + 1] == "mxfp8"
    assert stage[stage.index("--quant-compute") + 1] == "dequant"
    worker = worker_cli_args(args)
    assert worker[worker.index("--quant") + 1] == "mxfp8"
    assert worker[worker.index("--quant-compute") + 1] == "dequant"
    for line in (cli.LTX2Orchestrator(_args())._shared_cli_args("generate", work_dir="/w"),
                 worker_cli_args(_args())):
        assert "--quant" not in line and "--quant-compute" not in line


def test_quant_is_prepared_before_the_fit_checks_and_written_on_compile_only(monkeypatch,
                                                                              tmp_path):
    """The quantized directory is resolved (written by ``compile``, only read by ``generate``)
    before the DiT fit check and the text encoder placement, so both count the fp8 bytes, and
    before any rank starts, so none writes; and after the compile cache is rooted at
    --cache-dir, so the directory lands where the ranks, which inherit that root, resolve it
    (``checkpoint.default_cache_root`` reads DIFFLET_COMPILE_CACHE when the directory is made:
    rooted afterwards, compile would write under ~/.cache/difflet and the ranks refuse with
    'run difflet compile first')."""
    import os

    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    # setenv first records the variable's original state: the driver exports it with a plain
    # os.environ write, which a delenv of an unset variable would not undo.
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", "")
    monkeypatch.delenv("DIFFLET_COMPILE_CACHE")
    monkeypatch.setattr(cli.runner, "run_stage", lambda *a, **k: None)
    monkeypatch.setattr(cli.LTX2Orchestrator, "_decode_on_host", lambda self, work_dir: None)
    calls = []
    monkeypatch.setattr(cli.LTX2Orchestrator, "_prepare_quant",
                        lambda self, *, create: calls.append(
                            ("prepare_quant", create, os.environ.get("DIFFLET_COMPILE_CACHE"))))
    monkeypatch.setattr(cli.LTX2Orchestrator, "_check_dit_fits",
                        lambda self: calls.append(("check_dit_fits",)))
    monkeypatch.setattr(cli.LTX2Orchestrator, "_place_text_encoder",
                        lambda self: calls.append(("place_text_encoder",)))
    cache_dir = str(tmp_path / "c")
    cli.LTX2Orchestrator(_args(quant="mxfp8", cache_dir=cache_dir)).compile()
    assert calls == [("prepare_quant", True, os.path.abspath(cache_dir)), ("check_dit_fits",),
                     ("place_text_encoder",)]
    calls.clear()
    monkeypatch.delenv("DIFFLET_COMPILE_CACHE")  # compile exported it; generate exports again
    cli.LTX2Orchestrator(_args(quant="mxfp8", cache_dir=cache_dir,
                               work_dir=str(tmp_path / "w"))).generate()
    assert calls == [("prepare_quant", False, os.path.abspath(cache_dir)), ("check_dit_fits",),
                     ("place_text_encoder",)]


def test_quant_refuses_before_the_weights_are_downloaded(monkeypatch, capsys):
    """--quant needs the weights on the driver (``compile`` writes the quantized directory from
    them, ``generate`` resolves it and counts its bytes), so a launch before `difflet download`
    is refused here, naming the fix, instead of starting tp x cp ranks that each report the
    missing weights. The bf16 path leaves them to the ranks, as before."""
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")

    def not_local(self):
        raise OSError("not local")

    monkeypatch.setattr(cli.LTX2Orchestrator, "_model_dir", not_local)
    for create in (True, False):
        with pytest.raises(SystemExit) as exc:
            cli.LTX2Orchestrator(_args(quant="mxfp8"))._prepare_quant(create=create)
        assert exc.value.code == 2
        assert capsys.readouterr().err == (
            "Error: --quant: the Lightricks/LTX-2 weights are not local; run `difflet download "
            "--model-id Lightricks/LTX-2` first; nothing was launched.\n")
    cli.LTX2Orchestrator(_args())._prepare_quant(create=True)
    assert capsys.readouterr() == ("", "")


def _stored_checkpoint(directory, name, nbytes):
    """A safetensors file whose one fp8 tensor's ``data_offsets`` span ``nbytes`` of payload:
    what the fit checks count for a quantized directory (its bytes as stored)."""
    import json
    import struct

    directory.mkdir(parents=True, exist_ok=True)
    header = json.dumps({name: {"dtype": "F8_E4M3", "shape": [nbytes],
                                "data_offsets": [0, nbytes]}})
    (directory / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header.encode() + bytes(nbytes))


def _budget_gib(nbytes: int) -> str:
    """A DIFFLET_NEURON_HBM_GIB whose fit budget (``FIT_FRACTION`` of it) is ``nbytes``."""
    from difflet.backends.neuron.hbm import FIT_FRACTION, GIB

    return repr(nbytes / FIT_FRACTION / GIB)


def test_fit_checks_count_the_quantized_directory_as_stored(monkeypatch, tmp_path, capsys):
    """With --quant the DiT is counted as its quantized directory stores it (fp8 weights and
    block scales), not as the bf16 source: the source refuses tp1 (35 GiB per core) and moves
    Gemma-3 to the host at tp2; a directory of 2,000 stored bytes passes both checks under a
    24 GiB budget. Those bytes are what is compared with the budget: one under them refuses
    the tp, naming MXFP8 weights, and one that holds them but not Gemma-3 beside them moves
    Gemma-3 to the host."""
    from difflet.backends.neuron.hbm import quantized_dir_bytes

    _header_only_checkpoint(tmp_path / "transformer", "w", 18_876_000_000)
    _header_only_checkpoint(tmp_path / "text_encoder", "language_model.model.w", 11_770_000_000)
    _stored_checkpoint(tmp_path / "quantized", "w", 2_000)
    assert quantized_dir_bytes(str(tmp_path / "quantized"), tp_degree=1) == 2_000
    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", "24")
    monkeypatch.setattr(cli.LTX2Orchestrator, "_model_dir", lambda self: str(tmp_path))
    orch = cli.LTX2Orchestrator(_args(tp_degree=1, quant="mxfp8"))
    with pytest.raises(SystemExit) as exc:
        orch._check_dit_fits()
    assert exc.value.code == 2 and "GiB of bf16 weights per core" in capsys.readouterr().err
    orch._quantized_dit = str(tmp_path / "quantized")
    orch._check_dit_fits()
    orch = cli.LTX2Orchestrator(_args(tp_degree=2, quant="mxfp8"))
    orch._place_text_encoder()
    assert orch.args.device_text_encoder is False
    assert "encoding on rank 0's host instead" in capsys.readouterr().out
    orch = cli.LTX2Orchestrator(_args(tp_degree=2, quant="mxfp8"))
    orch._quantized_dit = str(tmp_path / "quantized")
    orch._place_text_encoder()
    assert orch.args.device_text_encoder is True
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", _budget_gib(1_500))
    orch = cli.LTX2Orchestrator(_args(tp_degree=1, quant="mxfp8"))
    orch._quantized_dit = str(tmp_path / "quantized")
    with pytest.raises(SystemExit) as exc:
        orch._check_dit_fits()
    assert exc.value.code == 2
    assert "GiB of MXFP8 weights per core at tp=1" in capsys.readouterr().err
    small = tmp_path / "small"  # Gemma-3 of 500 bf16 elements: 1,000 bytes beside the DiT
    _header_only_checkpoint(small / "text_encoder", "language_model.model.w", 500)
    monkeypatch.setattr(cli.LTX2Orchestrator, "_model_dir", lambda self: str(small))
    monkeypatch.setenv("DIFFLET_NEURON_HBM_GIB", _budget_gib(2_500))
    orch = cli.LTX2Orchestrator(_args(tp_degree=1, quant="mxfp8"))
    orch._quantized_dit = str(tmp_path / "quantized")
    orch._check_dit_fits()  # 2,000 stored bytes fit a 2,500-byte budget ...
    orch._place_text_encoder()  # ... but not with Gemma-3's 1,000 beside them
    assert orch.args.device_text_encoder is False
    assert "encoding on rank 0's host instead" in capsys.readouterr().out
