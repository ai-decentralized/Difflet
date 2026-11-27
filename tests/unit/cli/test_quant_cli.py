"""The CLI's MXFP8 flags (``difflet/cli/quant.py``) and ``--attention mx_attention``: parsing on
compile/generate/run, the driver's refusals, the spec and its forwarding to the stage parser and
the DP workers (the attention switch through the environment), and the DiT stage keys
(additive: a bf16 command line's key is unchanged)."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import pytest

from difflet.cli import quant as Q
from difflet.cli.stage import _build_stage_parser

QWEN = "Qwen/Qwen-Image"
FLUX = "black-forest-labs/FLUX.1-dev"
LTX2 = "Lightricks/LTX-2"
WAN21 = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
WAN22 = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
HV1 = "hunyuanvideo-community/HunyuanVideo"


def _parse(*flags):
    cli_main = importlib.import_module("difflet.cli.main")
    return cli_main._build_parser().parse_args(list(flags))


@pytest.mark.parametrize("command", ["compile", "generate", "run"])
def test_flags_parse_on_every_command(command):
    args = _parse(command, "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8",
                  "--quant-compute", "dequant")
    assert (args.quant, args.quant_compute) == ("mxfp8", "dequant")
    args = _parse(command, "--model-id", QWEN)
    assert (args.quant, args.quant_compute) == (None, None)
    with pytest.raises(SystemExit):  # the per-tensor static recipe is gone
        _parse(command, "--model-id", QWEN, "--quant", "fp8")


def test_the_help_names_the_attention_qkv_a_default_keeps_bf16():
    """The help says the DiT's attention q/k/v run in MXFP8, so it names the q/k/v a model's
    default table keeps bf16: HunyuanVideo's image q/k/v and Wan's self-attention q/k/v."""
    parser = argparse.ArgumentParser()
    Q.add_quant_flags(parser)
    help_text = next(a.help for a in parser._actions if a.dest == "quant")
    assert "HunyuanVideo: its image q/k/v too" in help_text
    assert "Wan: its self-attention q/k/v too" in help_text


def _refused(args, match, capsys):
    with pytest.raises(SystemExit):
        Q.validate_quant(args)
    assert match in capsys.readouterr().err


def test_refusals(capsys, monkeypatch):
    monkeypatch.delenv("DIFFLET_BACKEND", raising=False)
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    monkeypatch.delenv("UNSAFE_FP8FNCAST", raising=False)
    base = dict(model_id=QWEN, backend="neuron", quant="mxfp8", quant_compute=None,
                exec_mode=None, teacache_speedup=None)
    Q.validate_quant(argparse.Namespace(**base))  # the happy path
    Q.validate_quant(argparse.Namespace(**{**base, "model_id": FLUX}))
    _refused(argparse.Namespace(**{**base, "quant": None, "quant_compute": "nki"}),
             "--quant-compute needs --quant", capsys)
    _refused(argparse.Namespace(**{**base, "backend": "trainium"}), "neuron backend", capsys)
    _refused(argparse.Namespace(**{**base, "backend": None}), "neuron backend", capsys)
    _refused(argparse.Namespace(**{**base, "model_id": "MiniMaxAI/MiniMax-H3"}),
             "not available for", capsys)
    _refused(argparse.Namespace(**{**base, "exec_mode": "eager"}), "compile", capsys)
    _refused(argparse.Namespace(**{**base, "teacache_speedup": 1.5}), "TeaCache", capsys)
    monkeypatch.setenv("UNSAFE_FP8FNCAST", "1")
    _refused(argparse.Namespace(**base), "UNSAFE_FP8FNCAST", capsys)


def test_the_backend_env_counts_as_the_flag(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    Q.validate_quant(argparse.Namespace(model_id=QWEN, backend=None, quant="mxfp8",
                                        quant_compute=None))


def test_spec_from_args():
    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8",
                  "--tp-degree", "2")
    spec = Q.quant_spec(args)
    assert spec.format == "mxfp8" and spec.compute == "nki"
    assert spec.tp_degree == 2 and spec.k_shards("transformer_blocks.0.img_mlp.net.2") == 2
    args.quant_compute = "dequant"
    assert Q.quant_spec(args).compute == "dequant"
    assert Q.quant_spec(_parse("generate", "--model-id", QWEN)) is None
    assert Q.quant_spec(_parse("generate", "--model-id", QWEN, "--quant", "mxfp8")).tp_degree == 4


def test_the_tp_flag_wins_over_the_drivers_default_tp():
    """The row-parallel weights are quantized per rank K range, so the spec's tp is the launch's:
    --tp-degree when given, else the driver's default (4 unless it passes its own). The default
    is a fallback, never an override: a driver that passed its default constant would quantize
    for one tp and launch its stage, which reads the flag, for another."""
    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8")
    assert Q.quant_spec(args).tp_degree == 4
    assert Q.quant_spec(args, default_tp_degree=2).tp_degree == 2
    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8",
                  "--tp-degree", "8")
    assert Q.quant_spec(args).tp_degree == 8
    assert Q.quant_spec(args, default_tp_degree=2).tp_degree == 8


@pytest.mark.parametrize("tp_flags", [(), ("--tp-degree", "2")])
def test_the_stage_line_carries_the_tp_the_driver_quantized_for(monkeypatch, tp_flags):
    """Every rank rebuilds the spec from the stage flags, so the stage line names the tp the
    driver's prepare_quant used, flag or default; otherwise generate's ranks would refuse the
    directory compile just wrote ('built with another quantization spec')."""
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "compile")
    for model, stage_line in ((QWEN, lambda o: o._shared_cli_args("generate", work_dir="/w")),
                              (FLUX, lambda o: o._stage_cli_args("generate", work_dir="/w"))):
        args = cli_main._build_parser().parse_args(
            ["generate", "--model-id", model, "--quant", "mxfp8", *tp_flags])
        orch = cli_main._get_orchestrator(args)
        stage_args, _ = _build_stage_parser().parse_known_args(
            ["--orchestrator", model, "--stage", "generate", *stage_line(orch)])
        assert Q.quant_spec(stage_args) == Q.quant_spec(args)
        assert Q.quant_spec(args).tp_degree == (2 if tp_flags else 4)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    """The tiny diffusers-layout Qwen-Image DiT's transformer directory (4 heads x 32, a
    512-wide MLP: every target's K per rank is 128 at the driver's default tp4). torch and
    diffusers load here only; the rest of this module is as torch-free as the driver."""
    pytest.importorskip("diffusers")
    from tests.unit.backends._neuron_qwen_toy import write_tiny_qwen_model

    root = tmp_path_factory.mktemp("tiny_qwen_quant")
    write_tiny_qwen_model(root)
    return str(root / "transformer")


@pytest.fixture
def _trace_cache_restored(monkeypatch):
    """No DIFFLET_NKI_TRACE_CACHE, and whatever a test sets undone afterwards: setenv first
    records the variable's original state (delenv of an unset variable records nothing, and
    a value leaked into later tests' compile-cache layout)."""
    monkeypatch.setenv("DIFFLET_NKI_TRACE_CACHE", "")
    monkeypatch.delenv("DIFFLET_NKI_TRACE_CACHE")


def test_prepare_quant_writes_on_create_and_refuses_when_missing(
        tmp_path, monkeypatch, capsys, tiny_dir, _trace_cache_restored):
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    args = _parse("compile", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8")
    with pytest.raises(SystemExit) as exc:  # generate's resolve-only call before any compile
        Q.prepare_quant(args, {"transformer": tiny_dir}, create=False, tag="t")
    assert exc.value.code == 1  # the default refusal; a driver passes its own (below)
    assert capsys.readouterr().err.startswith("Error: --quant: transformer: ")
    out = Q.prepare_quant(args, {"transformer": tiny_dir}, create=True, tag="t")
    assert (Path(out["transformer"]) / "difflet_quant.json").is_file()
    assert Path(out["transformer"]).is_relative_to(tmp_path)
    assert capsys.readouterr().out.startswith(f"[t] MXFP8 transformer (nki): {out['transformer']}")
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ  # the trace cache is on by default
    # Resolve-only now finds it, and a second component is prepared under its own name.
    both = Q.prepare_quant(args, {"transformer": tiny_dir, "twin": tiny_dir}, create=False,
                           tag="t")
    assert both == {"transformer": out["transformer"], "twin": out["transformer"]}


def test_prepare_quant_refuses_a_component_without_weights(tmp_path, monkeypatch, capsys,
                                                           _trace_cache_restored):
    """A component directory without safetensors (a partial download) is refused on the driver,
    naming the component, as a stale directory is: not a traceback from the fingerprint."""
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    empty = tmp_path / "snapshot" / "transformer"
    empty.mkdir(parents=True)
    args = _parse("compile", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8")
    with pytest.raises(SystemExit) as exc:
        Q.prepare_quant(args, {"transformer": str(empty)}, create=True, tag="t")
    assert exc.value.code == 1
    assert capsys.readouterr().err == f"Error: --quant: transformer: no .safetensors under {empty}\n"


def test_each_driver_refuses_quant_the_way_it_refuses_everything_else(
        monkeypatch, tmp_path, capsys, tiny_dir, _trace_cache_restored):
    """A refused --quant exits as the driver's other refusals do, so a wrapper that tells a
    driver refusal from a launch failure keeps telling them apart: Qwen-Image through `refuse`
    (exit 2, '; nothing was launched.'), FLUX as its own checks (exit 1, no suffix)."""
    import difflet.pipeline.path_resolver as resolver

    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "compile")
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    snapshot = str(Path(tiny_dir).parent)
    monkeypatch.setattr(resolver, "resolve_model_path", lambda *a, **k: snapshot)
    for model, code, suffix in ((QWEN, 2, "; nothing was launched.\n"), (FLUX, 1, "\n")):
        args = cli_main._build_parser().parse_args(
            ["generate", "--model-id", model, "--quant", "mxfp8"])
        with pytest.raises(SystemExit) as exc:  # generate before any compile wrote the directory
            cli_main._get_orchestrator(args)._prepare_quant(create=False)
        err = capsys.readouterr().err
        assert exc.value.code == code, model
        assert err.startswith("Error: --quant: transformer: ") and err.endswith(suffix), model
        assert err.count("nothing was launched") == (1 if code == 2 else 0), model


def test_prepare_quant_is_off_without_the_flag_or_the_weights(tmp_path, capsys,
                                                              _trace_cache_restored):
    """None and no line when --quant is off, or when the weights are not downloaded yet (the
    ranks report that); the trace-cache setting is never touched (on by default)."""
    missing = {"transformer": str(tmp_path / "not-downloaded" / "transformer")}
    assert Q.prepare_quant(_parse("compile", "--model-id", QWEN), missing, create=True,
                           tag="t") is None
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ
    args = _parse("compile", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8",
                  "--quant-compute", "dequant")
    assert Q.prepare_quant(args, missing, create=True, tag="t") is None
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ
    args = _parse("compile", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8")
    assert Q.prepare_quant(args, missing, create=True, tag="t") is None
    assert Q.prepare_quant(args, {}, create=True, tag="t") is None
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ
    assert capsys.readouterr() == ("", "")


def test_flags_are_forwarded_only_when_given_and_consumed_by_the_stage_parser():
    assert Q.quant_cli_args(argparse.Namespace()) == []
    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8",
                  "--quant-compute", "dequant")
    forwarded = Q.quant_cli_args(args)
    assert forwarded == ["--quant", "mxfp8", "--quant-compute", "dequant"]
    parsed, extra = _build_stage_parser().parse_known_args(
        ["--orchestrator", QWEN, "--stage", "generate", *forwarded])
    assert extra == [] and parsed.quant == "mxfp8" and parsed.quant_compute == "dequant"
    assert Q.quant_cli_args(_parse("generate", "--model-id", QWEN, "--quant", "mxfp8")) == \
        ["--quant", "mxfp8"]

    from difflet.cli.dp.router import worker_cli_args

    args.dp = 2
    worker = worker_cli_args(args)
    assert "--quant" in worker and "--quant-compute" in worker


def test_qwen_orchestrator_forwards_and_keys_quant(monkeypatch):
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "compile")

    def key(*flags):
        args = cli_main._build_parser().parse_args(
            ["generate", "--model-id", QWEN, "--cache-dir", "/c", "--tp-degree", "2", *flags])
        orch = cli_main._get_orchestrator(args)
        return (orch._stage_cache_inputs("generate", args),
                orch._stage_compiled_dir("generate", args),
                orch._shared_cli_args("generate", work_dir="/w"))

    plain, plain_dir, plain_cli = key()
    assert not any(k.startswith("quant") for k in plain)
    assert "--quant" not in plain_cli
    mx, mx_dir, mx_cli = key("--quant", "mxfp8")
    added = sorted(k for k in mx if k not in plain)
    assert added == ["quant", "quant_compute", "quant_schema", "quant_targets_sha256",
                     "quant_tp_degree"]
    assert mx["quant"] == "mxfp8" and mx["quant_tp_degree"] == 2
    assert {k: v for k, v in mx.items() if k not in added} == plain
    assert mx_dir != plain_dir and mx_dir.parent == plain_dir.parent
    assert mx_cli[mx_cli.index("--quant") + 1] == "mxfp8"
    dequant, dequant_dir, _ = key("--quant", "mxfp8", "--quant-compute", "dequant")
    assert sorted(k for k in mx if mx[k] != dequant.get(k)) == ["quant_compute"]
    assert dequant_dir != mx_dir


def test_application_kwargs_carry_the_spec(monkeypatch):
    from difflet.cli.orchestrators.qwen_image import _torchrun_application_kwargs

    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--quant", "mxfp8")
    kwargs = _torchrun_application_kwargs(args)
    assert kwargs["quant"]["format"] == "mxfp8" and kwargs["quant"]["compute"] == "nki"
    assert kwargs["quant"]["tp_degree"] == 4 and "quant_calibration" not in kwargs
    assert "quant" not in _torchrun_application_kwargs(_parse("generate", "--model-id", QWEN))


def test_quant_leaves_the_nki_trace_cache_setting_alone(monkeypatch, tmp_path,
                                                        _trace_cache_restored):
    """The trace cache is on by default (neuron.compile), so no --quant path sets
    DIFFLET_NKI_TRACE_CACHE, and a user's opt-out reaches the ranks unchanged."""
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "compile")

    def prepare(*flags):
        args = cli_main._build_parser().parse_args(
            ["generate", "--model-id", QWEN, "--cache-dir", str(tmp_path), "--tp-degree", "4",
             *flags])
        orch = cli_main._get_orchestrator(args)
        # the model is not downloaded: _prepare_quant returns after the environment choice
        orch._model_dir = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError())
        orch._prepare_quant(create=False)

    prepare()
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ
    prepare("--quant", "mxfp8", "--quant-compute", "dequant")
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ
    prepare("--quant", "mxfp8")
    assert "DIFFLET_NKI_TRACE_CACHE" not in os.environ
    monkeypatch.setenv("DIFFLET_NKI_TRACE_CACHE", "0")
    prepare("--quant", "mxfp8")
    assert os.environ["DIFFLET_NKI_TRACE_CACHE"] == "0"


def test_quant_targets_env_replaces_the_layer_set_and_keys_the_spec(monkeypatch):
    monkeypatch.delenv("DIFFLET_EXEC_MODE", raising=False)
    monkeypatch.delenv(Q.QUANT_TARGETS_ENV, raising=False)
    args = _parse("generate", "--model-id", QWEN, "--quant", "mxfp8")
    default = Q.quant_spec(args)
    monkeypatch.setenv(Q.QUANT_TARGETS_ENV,
                       r"transformer_blocks\.\d+\.(img_mlp|txt_mlp)\.net\.0\.proj;;"
                       r"transformer_blocks\.\d+\.(img_mlp|txt_mlp)\.net\.2")
    narrowed = Q.quant_spec(args)
    assert len(narrowed.targets) == 2 and narrowed.targets != default.targets
    assert narrowed.matches("transformer_blocks.3.img_mlp.net.2")
    assert not narrowed.matches("transformer_blocks.3.attn.to_q")
    assert default.matches("transformer_blocks.3.attn.to_q")
    assert narrowed.cache_fields()["quant_targets_sha256"] != default.cache_fields()["quant_targets_sha256"]
    assert narrowed.weights_identity() != default.weights_identity()
    monkeypatch.setenv(Q.QUANT_TARGETS_ENV, "")
    assert Q.quant_spec(args).targets == default.targets


def test_ltx2_orchestrator_forwards_and_keys_quant(monkeypatch):
    """The LTX-2 generate stage's key gains the spec's fields with --quant, at the tp the launch
    uses (--tp-degree 2), and its stage line carries the flags so the ranks rebuild the same
    spec; the vae stage's key is unchanged (the decoder reads latents, whatever the DiT's
    precision) and a bf16 line keeps its key and its words."""
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "compile")

    def key(*flags):
        args = cli_main._build_parser().parse_args(
            ["generate", "--model-id", LTX2, "--cache-dir", "/c", "--tp-degree", "2", *flags])
        orch = cli_main._get_orchestrator(args)
        return (orch._stage_cache_inputs(args), orch._stage_compiled_dir(args),
                orch._shared_cli_args("generate", work_dir="/w"), orch)

    plain, plain_dir, plain_cli, _ = key()
    assert not any(k.startswith("quant") for k in plain)
    assert "--quant" not in plain_cli
    mx, mx_dir, mx_cli, orch = key("--quant", "mxfp8")
    added = sorted(k for k in mx if k not in plain)
    assert added == ["quant", "quant_compute", "quant_schema", "quant_targets_sha256",
                     "quant_tp_degree"]
    assert mx["quant"] == "mxfp8" and mx["quant_tp_degree"] == 2
    assert {k: v for k, v in mx.items() if k not in added} == plain
    assert mx_dir != plain_dir and mx_dir.parent == plain_dir.parent
    assert mx_cli[mx_cli.index("--quant") + 1] == "mxfp8"
    stage_args, extra = _build_stage_parser().parse_known_args(
        ["--orchestrator", LTX2, "--stage", "generate", *mx_cli])
    assert extra == [] and Q.quant_spec(stage_args) == Q.quant_spec(orch.args)
    assert Q.quant_spec(stage_args).tp_degree == 2
    vae = orch._stage_cache_inputs(argparse.Namespace(**{**vars(orch.args), "stage": "vae"}))
    assert not any(k.startswith("quant") for k in vae)
    dequant, dequant_dir, dequant_cli, _ = key("--quant", "mxfp8", "--quant-compute", "dequant")
    assert sorted(k for k in mx if mx[k] != dequant.get(k)) == ["quant_compute"]
    assert dequant_dir != mx_dir
    assert dequant_cli[dequant_cli.index("--quant-compute") + 1] == "dequant"


def test_ltx2_application_kwargs_carry_the_spec():
    from difflet.cli.orchestrators.ltx_2 import _torchrun_application_kwargs

    args = _parse("generate", "--model-id", LTX2, "--backend", "neuron", "--quant", "mxfp8")
    kwargs = _torchrun_application_kwargs(args)
    assert kwargs["quant"]["format"] == "mxfp8" and kwargs["quant"]["compute"] == "nki"
    assert kwargs["quant"]["tp_degree"] == 4 and "quant_checkpoint" not in kwargs
    assert "quant" not in _torchrun_application_kwargs(_parse("generate", "--model-id", LTX2))


@pytest.mark.parametrize("model", [WAN21, WAN22])
def test_wan_orchestrator_forwards_and_keys_quant(monkeypatch, model):
    """The Wan transformer stage's key gains the spec's fields with --quant, at the tp the
    launch uses (--tp-degree 2), under either Wan id (one target table, two cache prefixes),
    and its stage line carries the flags so the ranks rebuild the same spec; the vae stage's
    key is unchanged (the decoder reads latents, whatever the DiT's precision) and a bf16 line
    keeps its key and its words."""
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "compile")

    def key(*flags):
        args = cli_main._build_parser().parse_args(
            ["generate", "--model-id", model, "--cache-dir", "/c", "--tp-degree", "2", *flags])
        orch = cli_main._get_orchestrator(args)
        return (orch._stage_cache_inputs("transformer", args),
                orch._stage_compiled_dir("transformer", args),
                orch._shared_cli_args("generate", work_dir="/w"), orch)

    plain, plain_dir, plain_cli, _ = key()
    assert not any(k.startswith("quant") for k in plain)
    assert "--quant" not in plain_cli
    mx, mx_dir, mx_cli, orch = key("--quant", "mxfp8")
    added = sorted(k for k in mx if k not in plain)
    assert added == ["quant", "quant_compute", "quant_schema", "quant_targets_sha256",
                     "quant_tp_degree"]
    assert mx["quant"] == "mxfp8" and mx["quant_tp_degree"] == 2
    assert {k: v for k, v in mx.items() if k not in added} == plain
    assert mx_dir != plain_dir and mx_dir.parent == plain_dir.parent
    assert mx_cli[mx_cli.index("--quant") + 1] == "mxfp8"
    stage_args, extra = _build_stage_parser().parse_known_args(
        ["--orchestrator", model, "--stage", "transformer", *mx_cli])
    assert extra == [] and Q.quant_spec(stage_args) == Q.quant_spec(orch.args)
    assert Q.quant_spec(stage_args).tp_degree == 2
    assert not any(k.startswith("quant") for k in orch._stage_cache_inputs("vae", orch.args))
    dequant, dequant_dir, dequant_cli, _ = key("--quant", "mxfp8", "--quant-compute", "dequant")
    assert sorted(k for k in mx if mx[k] != dequant.get(k)) == ["quant_compute"]
    assert dequant_dir != mx_dir
    assert dequant_cli[dequant_cli.index("--quant-compute") + 1] == "dequant"


def test_wan_application_kwargs_carry_the_spec():
    """The ranks build TorchNeuronWanApplication with the spec's dict beside --dual-expert's
    ``enable_transformer_2``: the application resolves each expert's quantized directory
    itself, read-only, from the spec, so no directory is named (``quant_checkpoint`` could
    serve one expert only and the application refuses it with two)."""
    from difflet.cli.orchestrators.wan import _torchrun_application_kwargs

    args = _parse("generate", "--model-id", WAN22, "--backend", "neuron", "--quant", "mxfp8",
                  "--dual-expert")
    kwargs = _torchrun_application_kwargs(args)
    assert kwargs["quant"]["format"] == "mxfp8" and kwargs["quant"]["compute"] == "nki"
    assert kwargs["quant"]["tp_degree"] == 4 and "quant_checkpoint" not in kwargs
    assert kwargs["enable_transformer_2"] is True
    assert "quant" not in _torchrun_application_kwargs(_parse("generate", "--model-id", WAN22))


def test_hunyuan_video_orchestrator_forwards_and_keys_quant(monkeypatch):
    """The HunyuanVideo generate stage's key gains the spec's fields with --quant, at the tp
    the launch uses (--tp-degree 2), and its stage line carries the flags so the ranks rebuild
    the same spec; the vae stage's key is unchanged (the decoder reads latents, whatever the
    DiT's precision) and a bf16 line keeps its key and its words."""
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "compile")

    def key(*flags):
        args = cli_main._build_parser().parse_args(
            ["generate", "--model-id", HV1, "--cache-dir", "/c", "--tp-degree", "2", *flags])
        orch = cli_main._get_orchestrator(args)
        return (orch._stage_cache_inputs("generate", args),
                orch._stage_compiled_dir("generate", args),
                orch._shared_cli_args("generate", work_dir="/w"), orch)

    plain, plain_dir, plain_cli, _ = key()
    assert not any(k.startswith("quant") for k in plain)
    assert "--quant" not in plain_cli
    mx, mx_dir, mx_cli, orch = key("--quant", "mxfp8")
    added = sorted(k for k in mx if k not in plain)
    assert added == ["quant", "quant_compute", "quant_schema", "quant_targets_sha256",
                     "quant_tp_degree"]
    assert mx["quant"] == "mxfp8" and mx["quant_tp_degree"] == 2
    assert {k: v for k, v in mx.items() if k not in added} == plain
    assert mx_dir != plain_dir and mx_dir.parent == plain_dir.parent
    assert mx_cli[mx_cli.index("--quant") + 1] == "mxfp8"
    stage_args, extra = _build_stage_parser().parse_known_args(
        ["--orchestrator", HV1, "--stage", "generate", *mx_cli])
    assert extra == [] and Q.quant_spec(stage_args) == Q.quant_spec(orch.args)
    assert Q.quant_spec(stage_args).tp_degree == 2
    assert not any(k.startswith("quant") for k in orch._stage_cache_inputs("vae", orch.args))
    dequant, dequant_dir, dequant_cli, _ = key("--quant", "mxfp8", "--quant-compute", "dequant")
    assert sorted(k for k in mx if mx[k] != dequant.get(k)) == ["quant_compute"]
    assert dequant_dir != mx_dir
    assert dequant_cli[dequant_cli.index("--quant-compute") + 1] == "dequant"


def test_hunyuan_video_application_kwargs_carry_the_spec():
    """The ranks build TorchNeuronHunyuanVideoApplication with the spec's dict: it resolves
    the quantized directory itself, read-only, from the spec, so no directory is named
    (``quant_checkpoint``), and every other keyword is the bf16 launch's."""
    from difflet.cli.orchestrators.hunyuan_video import _torchrun_application_kwargs

    args = _parse("generate", "--model-id", HV1, "--backend", "neuron", "--quant", "mxfp8")
    kwargs = _torchrun_application_kwargs(args)
    assert kwargs["quant"]["format"] == "mxfp8" and kwargs["quant"]["compute"] == "nki"
    assert kwargs["quant"]["tp_degree"] == 4 and "quant_checkpoint" not in kwargs
    plain = _torchrun_application_kwargs(_parse("generate", "--model-id", HV1))
    assert "quant" not in plain
    assert {k: v for k, v in kwargs.items() if k != "quant"} == plain


@pytest.fixture(scope="module")
def tiny_ltx2_dir(tmp_path_factory):
    """The tiny diffusers-layout LTX-2 snapshot in the MXFP8 geometry (every target's K per
    rank a multiple of 128 at tp1); torch and diffusers load here only."""
    pytest.importorskip("diffusers")
    from tests.unit.backends._neuron_ltx2_toy import TINY_FP8_LTX2, write_tiny_ltx2_model

    root = tmp_path_factory.mktemp("tiny_ltx2_quant")
    write_tiny_ltx2_model(root, config=TINY_FP8_LTX2)
    return str(root)


def test_ltx2_driver_refuses_quant_the_way_it_refuses_everything_else(
        monkeypatch, tmp_path, capsys, tiny_ltx2_dir, _trace_cache_restored):
    """A refused --quant exits as the LTX-2 driver's other refusals do (``refuse``: exit 2,
    '; nothing was launched.'), naming the component; ``compile`` writes the directory under the
    compile cache and keeps it for the fit checks (``_quantized_dit``)."""
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "compile")
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))

    def orchestrator(command, tp):
        args = cli_main._build_parser().parse_args(
            [command, "--model-id", LTX2, "--quant", "mxfp8", "--tp-degree", str(tp)])
        orch = cli_main._get_orchestrator(args)
        orch._model_dir = lambda: tiny_ltx2_dir
        return orch

    with pytest.raises(SystemExit) as exc:  # generate before any compile wrote the directory
        orchestrator("generate", 1)._prepare_quant(create=False)
    err = capsys.readouterr().err
    assert exc.value.code == 2
    assert err.startswith("Error: --quant: transformer: ") and err.endswith(
        "; nothing was launched.\n")
    compiling = orchestrator("compile", 1)
    assert compiling._quantized_dit is None
    compiling._prepare_quant(create=True)
    assert (Path(compiling._quantized_dit) / "difflet_quant.json").is_file()
    assert Path(compiling._quantized_dit).is_relative_to(tmp_path)
    assert capsys.readouterr().out.startswith(
        f"[ltx_2] MXFP8 transformer (nki): {compiling._quantized_dit}")
    generating = orchestrator("generate", 1)
    generating._prepare_quant(create=False)
    assert generating._quantized_dit == compiling._quantized_dit


def test_ltx2_driver_refuses_a_dead_target_pattern_before_writing(
        monkeypatch, tmp_path, capsys, tiny_ltx2_dir, _trace_cache_restored):
    """A target pattern that matches no linear weight (a DIFFLET_QUANT_TARGETS typo) is refused
    on the driver, naming the pattern, before the DiT is quantized and before torchrun starts:
    the writer would otherwise quantize less than the spec claims, print the MXFP8 line, and
    the refusal would land in every rank (``quantize_module_``) after minutes of quantizing."""
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "compile")
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    live = r"transformer_blocks\.\d+\.attn1\.(to_q|to_k|to_v)"
    dead = r"transformer_blocks\.\d+\.audio_attn9\.to_q"
    monkeypatch.setenv(Q.QUANT_TARGETS_ENV, f"{live};{dead}")
    args = cli_main._build_parser().parse_args(
        ["compile", "--model-id", LTX2, "--quant", "mxfp8", "--tp-degree", "1"])
    orch = cli_main._get_orchestrator(args)
    orch._model_dir = lambda: tiny_ltx2_dir
    with pytest.raises(SystemExit) as exc:
        orch._prepare_quant(create=True)
    out, err = capsys.readouterr()
    assert exc.value.code == 2 and out == ""
    assert err.startswith("Error: --quant: transformer: ") and "match no linear weight" in err
    assert err.endswith(f": {dead}; nothing was launched.\n")
    assert orch._quantized_dit is None
    assert not (tmp_path / "quantized").exists()


# ------------------------------------------------------------------ --attention mx_attention


def _isolate_env(monkeypatch, *names):
    """Unset ``names`` now and restore them after the test, even when the code under test
    assigns os.environ itself (delenv of an absent name records nothing to undo)."""
    for name in names:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


_SWITCHES = ("DIFFLET_ATTN_QUANT", "DIFFLET_EXP_ATTN_CONSTMAX")


@pytest.mark.parametrize("command", ["compile", "generate", "run"])
def test_mx_attention_is_an_attention_kernel_and_attn_quant_is_gone(command):
    """The MXFP8 kernel is the third ``--attention`` value; the former ``--attn-quant`` flag is
    refused by the CLI parser, and the stage parser does not declare it either."""
    assert _parse(command, "--model-id", QWEN, "--attention", "mx_attention").attention == \
        "mx_attention"
    assert _parse(command, "--model-id", QWEN).attention is None  # absent: the environment's
    for bad in (["--attn-quant", "qk+pv"], ["--attn-quant", "pv"], ["--attention", "qk+pv"],
                ["--attention", "mxfp8"]):
        with pytest.raises(SystemExit):
            _parse(command, "--model-id", QWEN, *bad)
    parsed, extra = _build_stage_parser().parse_known_args(
        ["--orchestrator", QWEN, "--stage", "generate", "--attn-quant", "qk+pv"])
    assert extra == ["--attn-quant", "qk+pv"] and not hasattr(parsed, "attn_quant")


def test_the_attention_help_carries_mx_attentions_cost_and_scope():
    """A user who finds mx_attention in --help learns what runs in MXFP8, where it runs, that
    it is slower and costs quality on some models, and that it is independent of --quant."""
    cli_main = importlib.import_module("difflet.cli.main")
    assert cli_main.ATTENTION_CHOICES == ("cte", "const-max", "mx_attention")
    generate = next(a for a in cli_main._build_parser()._actions
                    if isinstance(a, argparse._SubParsersAction)).choices["generate"]
    help_text = " ".join(next(a.help for a in generate._actions if a.dest == "attention").split())
    for words in ("'mx_attention': Difflet's MXFP8 attention kernel for self/joint attention",
                  "QK^T and PV on nc_matmul_mx", "the softmax and cross-attention stay bf16",
                  "trn3, compile mode, models with MXFP8 target layers", "slower than cte",
                  "costs quality on some models", "independent of --quant",
                  "DIFFLET_ATTN_QUANT=mxfp8"):
        assert words in help_text, words


@pytest.mark.parametrize("attention, ambient, want", [
    ("mx_attention", (None, "1"), ("mxfp8", None)),     # exports its switch, clears const-max's
    ("mx_attention", ("off", None), ("mxfp8", None)),   # over an exported value
    ("const-max", ("mxfp8", None), (None, "1")),        # exports its switch, clears the MX one
    ("cte", ("mxfp8", "1"), (None, None)),              # clears both
    (None, ("mxfp8", None), ("mxfp8", None)),           # absent: the environment as it was
    (None, (None, "1"), (None, "1")),
    (None, (None, None), (None, None)),
])
def test_attention_exports_one_kernels_switch_and_clears_the_other(monkeypatch, attention,
                                                                   ambient, want):
    """``--attention`` picks one kernel, so DIFFLET_ATTN_QUANT and DIFFLET_EXP_ATTN_CONSTMAX are
    never both left set by the flag; without it the ambient environment is kept."""
    cli_main = importlib.import_module("difflet.cli.main")
    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE", *_SWITCHES)
    for name, value in zip(_SWITCHES, ambient):
        if value is not None:
            monkeypatch.setenv(name, value)
    cli_main._apply_backend_flags(argparse.Namespace(backend="neuron", exec_mode=None,
                                                     attention=attention, model_id=QWEN))
    assert tuple(os.environ.get(name) for name in _SWITCHES) == want


def test_mx_attention_on_another_backend_exits_before_anything_is_exported(monkeypatch, capsys):
    cli_main = importlib.import_module("difflet.cli.main")
    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE", *_SWITCHES)
    monkeypatch.setenv("DIFFLET_EXP_ATTN_CONSTMAX", "1")
    with pytest.raises(SystemExit) as exc:
        cli_main._apply_backend_flags(argparse.Namespace(backend="trainium", exec_mode=None,
                                                         attention="mx_attention", model_id=QWEN))
    assert exc.value.code == 2
    assert "--attention mx_attention runs on the neuron backend only" in capsys.readouterr().err
    assert "DIFFLET_ATTN_QUANT" not in os.environ and "DIFFLET_BACKEND" not in os.environ
    assert os.environ["DIFFLET_EXP_ATTN_CONSTMAX"] == "1"


def test_mx_attention_is_independent_of_quant_and_refused_where_it_cannot_run(capsys,
                                                                              monkeypatch):
    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE", "UNSAFE_FP8FNCAST",
                 *_SWITCHES)
    base = dict(model_id=QWEN, backend="neuron", quant=None, quant_compute=None,
                attention="mx_attention", exec_mode=None, teacache_speedup=None)
    Q.validate_quant(argparse.Namespace(**base))  # without --quant
    Q.validate_quant(argparse.Namespace(**{**base, "quant": "mxfp8"}))  # with it
    Q.validate_quant(argparse.Namespace(**{**base, "teacache_speedup": 1.5}))  # --quant's rule
    for other in ("cte", "const-max", None):  # the other kernels: none of its checks
        Q.validate_quant(argparse.Namespace(**{**base, "attention": other, "backend": None,
                                               "exec_mode": "eager"}))
    _refused(argparse.Namespace(**{**base, "backend": "trainium"}),
             "--attention mx_attention runs on the PyTorch-native neuron backend only", capsys)
    _refused(argparse.Namespace(**{**base, "model_id": "MiniMaxAI/MiniMax-H3"}),
             "--attention mx_attention is not available for", capsys)
    monkeypatch.setenv("DIFFLET_ATTN_QUANT", "mxfp8")  # exported, no flag: H3 refuses it as well
    _refused(argparse.Namespace(**{**base, "model_id": "MiniMaxAI/MiniMax-H3", "attention": None}),
             "DIFFLET_ATTN_QUANT=mxfp8 (--attention mx_attention) is not available for", capsys)
    monkeypatch.delenv("DIFFLET_ATTN_QUANT")
    _refused(argparse.Namespace(**{**base, "exec_mode": "eager"}),
             "--attention mx_attention needs the compiled DiT", capsys)
    monkeypatch.setenv("UNSAFE_FP8FNCAST", "1")
    _refused(argparse.Namespace(**base), "UNSAFE_FP8FNCAST", capsys)


@pytest.mark.parametrize("cp_mode, cp_degree, refused", [
    ("ring", 2, True), ("ulysses", 2, True), ("gather_kv", 2, False), ("ring", 1, False),
    ("ulysses", None, False)])
def test_mx_attention_is_refused_where_context_parallelism_keeps_attention_bf16(
        capsys, monkeypatch, cp_mode, cp_degree, refused):
    """Ring and Ulysses send every self/joint attention call through their bf16 paths: the
    kernel would quantize nothing (and still fork the stage key)."""
    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE", "UNSAFE_FP8FNCAST",
                 *_SWITCHES)
    args = argparse.Namespace(model_id=QWEN, backend="neuron", quant=None, quant_compute=None,
                              attention="mx_attention", exec_mode=None, teacache_speedup=None,
                              cp_mode=cp_mode, cp_degree=cp_degree)
    if refused:
        _refused(args, f"--attention mx_attention with --cp-mode {cp_mode}", capsys)
    else:
        Q.validate_quant(args)
    Q.validate_quant(argparse.Namespace(**{**vars(args), "attention": None, "quant": "mxfp8"}))


@pytest.mark.parametrize("value, refused", [
    ("qk+pv", "the qk / pv variants are gone"), ("qk", "'qk'"), ("pv", "'pv'"),
    ("both", "'both'"), ("", None), ("off", None), ("mxfp8", None)])
def test_the_driver_checks_an_exported_switch_before_anything_launches(
        capsys, monkeypatch, value, refused):
    """Without ``--attention`` an exported value is kept: the driver refuses a bad one (the
    former variants too) in one line, as its other refusals, not as a traceback from a cache key
    or a rank's import."""
    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE", "UNSAFE_FP8FNCAST",
                 *_SWITCHES)
    monkeypatch.setenv("DIFFLET_ATTN_QUANT", value)
    args = argparse.Namespace(model_id=QWEN, backend="neuron", quant=None, quant_compute=None,
                              attention=None, exec_mode=None, teacache_speedup=None)
    if refused:
        _refused(args, refused, capsys)
    else:
        Q.validate_quant(args)


def test_the_driver_refuses_both_attention_switches_exported(capsys, monkeypatch):
    """An ambient pair (``--attention`` clears one of them) names both variables, before any
    rank's import of the neuron attention would refuse it."""
    _isolate_env(monkeypatch, "DIFFLET_BACKEND", "DIFFLET_EXEC_MODE", "UNSAFE_FP8FNCAST",
                 *_SWITCHES)
    monkeypatch.setenv("DIFFLET_ATTN_QUANT", "mxfp8")
    monkeypatch.setenv("DIFFLET_EXP_ATTN_CONSTMAX", "1")
    args = argparse.Namespace(model_id=QWEN, backend="neuron", quant=None, quant_compute=None,
                              attention=None, exec_mode=None, teacache_speedup=None)
    _refused(args, "DIFFLET_ATTN_QUANT=mxfp8 and DIFFLET_EXP_ATTN_CONSTMAX=1 select different "
             "attention kernels", capsys)
    monkeypatch.setenv("DIFFLET_ATTN_QUANT", "off")
    Q.validate_quant(args)


def test_mx_attention_reaches_stages_and_dp_workers_through_the_environment():
    """Nothing is forwarded on a stage or worker line (as for const-max): stages and DP workers
    inherit the exported DIFFLET_ATTN_QUANT."""
    args = _parse("generate", "--model-id", QWEN, "--backend", "neuron", "--attention",
                  "mx_attention")
    assert Q.quant_cli_args(args) == []

    from difflet.cli.dp.router import worker_cli_args

    args.dp = 2
    worker = worker_cli_args(args)
    assert "--attention" not in worker and "--attn-quant" not in worker
    assert _parse("generate", *worker).attention is None  # the worker keeps the environment's


_ORDER_PROBE = """
import importlib, os, sys
m = importlib.import_module("difflet.cli.main")
def stop(args):  # the orchestrator: the first code that imports a backend module
    pre = "difflet.backends.neuron.ops_impl.attention" in sys.modules
    # What the import resolves is under test, not the kernel: with the switch on, the import
    # loads the MX op, which needs torch-neuronx (absent on a CPU host such as CI).
    from difflet.backends.neuron.ops_impl import mx_attention
    mx_attention.load_op = lambda: None
    from difflet.backends.neuron.ops_impl import attention
    print("RESULT", pre, os.environ.get("DIFFLET_ATTN_QUANT"),
          os.environ.get("DIFFLET_EXP_ATTN_CONSTMAX"), attention.ATTN_QUANT, attention._CONSTMAX)
    raise SystemExit(0)
m._get_orchestrator = stop
m.main(sys.argv[1:])
"""


@pytest.mark.parametrize("flag, env, expect", [
    ("mx_attention", {}, ("mxfp8", "None", "mxfp8", "False")),  # the flag sets the switch
    ("mx_attention", {"DIFFLET_ATTN_QUANT": "off", "DIFFLET_EXP_ATTN_CONSTMAX": "1"},
     ("mxfp8", "None", "mxfp8", "False")),                       # over the exported values
    ("cte", {"DIFFLET_ATTN_QUANT": "mxfp8"}, ("None", "None", "off", "False")),
    ("const-max", {"DIFFLET_ATTN_QUANT": "mxfp8"}, ("None", "1", "off", "True")),
    (None, {"DIFFLET_ATTN_QUANT": "mxfp8"}, ("mxfp8", "None", "mxfp8", "False")),  # kept
    (None, {}, ("None", "None", "off", "False")),
])
def test_attention_sets_the_switch_before_any_backend_module_is_imported(flag, env, expect):
    """ATTN_QUANT and _CONSTMAX are resolved once, when ops_impl/attention.py is imported:
    main()'s _apply_backend_flags exports --attention before anything imports a backend (the
    orchestrator, the stage launches)."""
    import subprocess
    import sys

    environ = {k: v for k, v in os.environ.items() if k not in _SWITCHES}
    environ.update(DIFFLET_BACKEND="cpu", PYTHONPATH=str(Path(__file__).parents[3]), **env)
    argv = ["generate", "--model-id", QWEN, "--backend", "neuron", "--tp-degree", "4",
            "--prompt", "p", "--output", "/nonexistent/o.png",
            *(["--attention", flag] if flag else [])]
    proc = subprocess.run([sys.executable, "-c", _ORDER_PROBE, *argv], env=environ,
                          capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT")]
    assert lines, proc.stderr[-3000:]
    pre, *resolved = lines[-1].split()[1:]
    assert pre == "False" and tuple(resolved) == expect


def test_the_stage_entry_keeps_the_inherited_switch(monkeypatch):
    """A stage line carries no attention flag: the stage imports the backend under the
    DIFFLET_ATTN_QUANT its driver exported."""
    from difflet.cli import stage

    monkeypatch.setenv("DIFFLET_ATTN_QUANT", "mxfp8")
    seen = []

    def load(name):
        seen.append(os.environ.get("DIFFLET_ATTN_QUANT"))
        raise SystemExit(0)

    monkeypatch.setattr(stage, "_load_orchestrator_class", load)
    with pytest.raises(SystemExit):
        stage.main(["--orchestrator", QWEN, "--stage", "generate"])
    assert seen == ["mxfp8"]


def test_the_attention_switch_values(monkeypatch):
    from difflet.quant import spec as S

    monkeypatch.delenv("DIFFLET_ATTN_QUANT", raising=False)
    assert S.attn_quant_mode() == "off"
    assert S.ATTN_QUANT_VALUES == ("off", "mxfp8")
    for value, expect in (("off", "off"), ("", "off"), ("  ", "off"), (" OFF ", "off"),
                          ("mxfp8", "mxfp8"), (" MXFP8 ", "mxfp8")):  # empty: `VAR= cmd`
        monkeypatch.setenv("DIFFLET_ATTN_QUANT", value)
        assert S.attn_quant_mode() == expect
    for bad in ("qk", "pv", "qk+pv", "fp8", "both", "on"):
        monkeypatch.setenv("DIFFLET_ATTN_QUANT", bad)
        with pytest.raises(ValueError) as exc:
            S.attn_quant_mode()
        message = str(exc.value)
        assert "DIFFLET_ATTN_QUANT" in message and repr(bad) in message
        assert all(v in message for v in ("unset", "empty", "'off'", "'mxfp8'"))
        assert ("variants are gone" in message) == (bad in ("qk", "pv", "qk+pv"))


def _dit_key(model, monkeypatch, value):
    cli_main = importlib.import_module("difflet.cli.main")
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    monkeypatch.setenv("DIFFLET_EXEC_MODE", "compile")
    if value is None:
        monkeypatch.delenv("DIFFLET_ATTN_QUANT", raising=False)
    else:
        monkeypatch.setenv("DIFFLET_ATTN_QUANT", value)
    args = cli_main._build_parser().parse_args(
        ["generate", "--model-id", model, "--cache-dir", "/c", "--tp-degree", "2"])
    orch = cli_main._get_orchestrator(args)
    stage = {QWEN: "generate", HV1: "generate", WAN21: "transformer"}.get(model)
    if stage is None:  # LTX-2: one keyed stage
        return orch._stage_cache_inputs(args), orch._stage_compiled_dir(args)
    return orch._stage_cache_inputs(stage, args), orch._stage_compiled_dir(stage, args)


@pytest.mark.parametrize("model", [QWEN, WAN21, HV1, LTX2])
def test_the_dit_stage_key_tells_off_and_mxfp8_apart(monkeypatch, model):
    """Additive: off (unset or 'off') keeps today's key and dir; mxfp8 adds ``attn_quant`` and
    nothing else, so the two are two artifacts (with or without --quant: the field sits beside
    the spec's)."""
    unset, unset_dir = _dit_key(model, monkeypatch, None)
    off, off_dir = _dit_key(model, monkeypatch, "off")
    assert off == unset and off_dir == unset_dir and "attn_quant" not in off
    key, key_dir = _dit_key(model, monkeypatch, "mxfp8")
    assert {k: v for k, v in key.items() if k != "attn_quant"} == off
    assert key["attn_quant"] == "mxfp8" and key_dir.parent == off_dir.parent
    assert key_dir != off_dir


def test_quant_cache_fields_adds_mxfp8_attention_beside_the_spec(monkeypatch):
    from difflet.quant.spec import quant_cache_fields

    monkeypatch.delenv("DIFFLET_ATTN_QUANT", raising=False)
    spec = Q.quant_spec(_parse("generate", "--model-id", QWEN, "--quant", "mxfp8"))
    linear = quant_cache_fields(spec)
    assert quant_cache_fields(None) == {} and "attn_quant" not in linear
    for off in ("off", ""):  # empty is off: the keys stay today's
        monkeypatch.setenv("DIFFLET_ATTN_QUANT", off)
        assert quant_cache_fields(None) == {} and quant_cache_fields(spec) == linear
    monkeypatch.setenv("DIFFLET_ATTN_QUANT", "mxfp8")
    assert quant_cache_fields(None) == {"attn_quant": "mxfp8"}
    assert quant_cache_fields(spec) == {**linear, "attn_quant": "mxfp8"}
