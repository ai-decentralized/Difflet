"""--quant plumbing: parser, validation, quantize command, Wan stage identity, DP."""

from __future__ import annotations

import argparse
import json
import sys
import types

import pytest

from difflet.quant.spec import QuantSpec

import importlib

cli_main = importlib.import_module("difflet.cli.main")

WAN = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


@pytest.mark.parametrize("command", ["compile", "generate", "run", "serve"])
def test_quant_flags_parse_on_every_device_command(command):
    parser = cli_main._build_parser()
    extra = ["--prompt", "p", "--output", "o.mp4"] if command in ("generate", "run") else []
    args = parser.parse_args(
        [command, "--model-id", WAN, "--quant", "fp8", "--quant-granularity", "channel",
         "--quant-act", "none", *extra]
    )
    assert (args.quant, args.quant_granularity, args.quant_act) == ("fp8", "channel", "none")
    default = parser.parse_args([command, "--model-id", WAN, *extra])
    assert default.quant is None and default.quant_granularity == "tensor" and default.quant_act == "dynamic"
    with pytest.raises(SystemExit):
        parser.parse_args([command, "--model-id", WAN, "--quant", "int8", *extra])


def test_quantize_subcommand_defaults_to_fp8():
    args = cli_main._build_parser().parse_args(["quantize", "--model-id", WAN])
    assert args.command == "quantize" and args.quant == "fp8" and args.force is False
    args = cli_main._build_parser().parse_args(
        ["quantize", "--model-id", WAN, "--quant-granularity", "channel", "--force", "--cache-dir", "/c"]
    )
    assert args.quant_granularity == "channel" and args.force and args.cache_dir == "/c"


def test_validate_quant_rejects_unwired_models_and_probe_teacache():
    cli_main._validate_quant(argparse.Namespace(model_id=WAN, quant="fp8"))
    hv15 = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
    cli_main._validate_quant(argparse.Namespace(model_id=hv15, quant=None))
    with pytest.raises(SystemExit):
        cli_main._validate_quant(argparse.Namespace(model_id=hv15, quant="fp8"))
    with pytest.raises(SystemExit):
        cli_main._validate_quant(argparse.Namespace(model_id=WAN, quant="fp8", teacache_speedup=1.5))
    cli_main._validate_quant(argparse.Namespace(model_id=WAN, quant="fp8", teacache_cadence=2))


def test_main_dispatches_quantize_and_exits_with_its_code(monkeypatch):
    seen = {}

    def fake_run(args):
        seen["args"] = args
        return 3

    monkeypatch.setattr("difflet.cli.quantize.run", fake_run)
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["quantize", "--model-id", WAN, "--quant-granularity", "channel"])
    assert exc.value.code == 3
    assert seen["args"].quant == "fp8" and seen["args"].quant_granularity == "channel"


def test_validate_quant_accepts_every_wired_model_and_rejects_hunyuan_video_15(capsys):
    for model_id in (
        "black-forest-labs/FLUX.1-dev",
        "Qwen/Qwen-Image",
        "hunyuanvideo-community/HunyuanVideo",
        "Lightricks/LTX-2",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    ):
        cli_main._validate_quant(argparse.Namespace(model_id=model_id, quant="fp8", teacache_speedup=None))
    with pytest.raises(SystemExit):
        cli_main._validate_quant(argparse.Namespace(
            model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
            quant="fp8", teacache_speedup=None))
    err = capsys.readouterr().err
    assert "does not support --quant" in err and "flux" in err and "ltx_2" in err


# ---------------------------------------------------------------- Wan orchestrator


def _wan_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id=WAN, tp_degree=4, cp_degree=1, cp_mode="gather_kv", height=480, width=832,
        num_frames=9, cache_dir=None, force=False, revision=None, prompt="a cat",
        output="/tmp/w.mp4", steps=2, guidance_scale=1.0, seed=42, work_dir=None,
        keep_work_dir=False, teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None, quant=None,
        quant_granularity="tensor", quant_act="dynamic", shapes=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_wan_stage_identity_is_unchanged_for_bf16_and_extended_for_fp8(monkeypatch):
    from difflet.cli.orchestrators import wan as wan_orch

    monkeypatch.setattr(wan_orch, "stage_toolchain_versions", lambda: {"python": "3.12"})
    bf16 = wan_orch.WanOrchestrator(_wan_args())
    inputs = bf16._stage_cache_inputs("transformer", bf16.args)
    assert set(inputs) == {
        "component", "model_id", "tp", "cp", "cp_mode", "cfg_parallel", "sp", "dtype",
        "text_seq_len", "shapes", "toolchain",
    }  # additive-only: no "quant" key for bf16, every existing artifact keeps its hash

    fp8 = wan_orch.WanOrchestrator(_wan_args(quant="fp8", quant_act="none"))
    fp8_inputs = fp8._stage_cache_inputs("transformer", fp8.args)
    assert fp8_inputs["quant"] == {
        "format": "fp8_e4m3", "weight_granularity": "tensor", "activation": "none",
        "targets": list(__import__("difflet.quant.spec", fromlist=["DEFAULT_TARGETS"]).DEFAULT_TARGETS),
    }
    from difflet.backends.trainium.core.quant import QUANT_LAYER_SCHEMA

    # The quantized-layer schema version keys the NEFF too (fp8 only, additive).
    assert fp8_inputs["quant_layer_schema"] == QUANT_LAYER_SCHEMA
    assert {k: v for k, v in fp8_inputs.items() if k not in ("quant", "quant_layer_schema")} == inputs
    assert "quant" not in fp8._stage_cache_inputs("vae", fp8.args)
    assert "quant_layer_schema" not in fp8._stage_cache_inputs("vae", fp8.args)
    assert fp8._stage_compiled_dir("transformer", fp8.args) != bf16._stage_compiled_dir(
        "transformer", bf16.args
    )
    assert fp8._stage_compiled_dir("vae", fp8.args) == bf16._stage_compiled_dir("vae", bf16.args)


def test_wan_shared_cli_args_forward_quant_flags_only_when_set():
    from difflet.cli.orchestrators.wan import WanOrchestrator

    plain = WanOrchestrator(_wan_args())._shared_cli_args(stage_mode="compile")
    assert "--quant" not in plain
    quant = WanOrchestrator(_wan_args(quant="fp8", quant_granularity="channel"))._shared_cli_args(
        stage_mode="compile"
    )
    idx = quant.index("--quant")
    assert quant[idx : idx + 6] == ["--quant", "fp8", "--quant-granularity", "channel", "--quant-act", "dynamic"]


def test_wan_transformer_stage_passes_quant_kwargs_to_the_application(monkeypatch, tmp_path):
    from difflet.cli.orchestrators import wan as wan_orch

    captured = {}

    class FakeApp:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def compile(self, path):
            captured["compiled"] = path

    fake_module = types.ModuleType("difflet.models.wan.application")
    fake_module.NeuronWanApplication = FakeApp
    monkeypatch.setitem(sys.modules, "difflet.models.wan.application", fake_module)
    monkeypatch.setattr(
        "difflet.pipeline.path_resolver.resolve_model_path", lambda *a, **k: str(tmp_path)
    )
    monkeypatch.setattr(wan_orch, "stage_toolchain_versions", lambda: {})
    monkeypatch.setattr(wan_orch.WanOrchestrator, "_finish_stage_compile", lambda *a, **k: None)

    args = _wan_args(quant="fp8", quant_act="none", cache_dir=str(tmp_path / "cache"))
    args.stage_mode = "compile"
    wan_orch.WanOrchestrator(args)._stage_transformer(args)
    assert captured["quant"]["activation"] == "none"
    assert captured["quant_cache_dir"] == str(tmp_path / "cache")
    assert "compiled" in captured

    captured.clear()
    args = _wan_args()
    args.stage_mode = "compile"
    wan_orch.WanOrchestrator(args)._stage_transformer(args)
    assert captured["quant"] is None


def test_stage_parser_and_dp_router_carry_quant_flags():
    from difflet.cli import stage
    from difflet.cli.dp.router import worker_cli_args

    args, _ = stage._build_stage_parser().parse_known_args(
        ["--orchestrator", WAN, "--stage", "transformer", "--quant", "fp8", "--quant-act", "none"]
    )
    assert args.quant == "fp8" and args.quant_act == "none" and args.quant_granularity == "tensor"

    argv = worker_cli_args(_wan_args(quant="fp8", quant_granularity="channel"))
    assert argv[argv.index("--quant") :][:6] == [
        "--quant", "fp8", "--quant-granularity", "channel", "--quant-act", "dynamic",
    ]
    assert "--quant" not in worker_cli_args(_wan_args())


# ------------------------------------------------------------------ quantize cmd


def test_quantize_command_builds_the_checkpoint_copy(monkeypatch, tmp_path, capsys):
    import torch
    from safetensors.torch import save_file

    from difflet.cli import quantize as quantize_cmd
    from difflet.quant.checkpoint import read_manifest

    model_dir = tmp_path / "snap"
    for sub in ("transformer", "transformer_2"):
        (model_dir / sub).mkdir(parents=True)
        save_file(
            {"blocks.0.attn1.to_q.weight": torch.randn(8, 8, dtype=torch.bfloat16),
             "proj_out.weight": torch.randn(4, 8, dtype=torch.bfloat16)},
            str(model_dir / sub / "diffusion_pytorch_model.safetensors"),
        )
        (model_dir / sub / "config.json").write_text(json.dumps({}))
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", lambda *a, **k: str(model_dir))
    assert quantize_cmd.transformer_subfolders(str(model_dir)) == ["transformer", "transformer_2"]

    args = argparse.Namespace(model_id=WAN, revision=None, quant="fp8", quant_granularity="tensor",
                              quant_act="dynamic", cache_dir=str(tmp_path / "cache"), force=False)
    assert quantize_cmd.run(args) == 0
    out = capsys.readouterr().out
    assert "[quantize] transformer:" in out and "[quantize] transformer_2:" in out
    dirs = sorted((tmp_path / "cache" / "quantized").rglob("difflet_quant.json"))
    assert len(dirs) == 2
    assert read_manifest(dirs[0].parent)["report"]["num_quantized"] == 1

    assert quantize_cmd.run(argparse.Namespace(model_id=WAN, revision=None, quant=None)) == 1

    def missing(*a, **k):
        raise OSError("no snapshot")

    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", missing)
    assert quantize_cmd.run(args) == 1



# --------------------------------------------------------------- FLUX orchestrator


def _flux_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="black-forest-labs/FLUX.1-dev", tp_degree=4, cp_degree=1, cp_mode="gather_kv",
        sp_enabled=False, height=1024, width=1024, num_frames=None, cache_dir="/tmp/cache",
        force=False, revision=None, prompt="a cat", output="/tmp/f.png", steps=2,
        guidance_scale=3.5, seed=42, teacache_cadence=None, teacache_online_delta=None,
        teacache_speedup=None, teacache_calibration=None, taef1=False, taef1_path=None,
        quant=None, quant_granularity="tensor", quant_act="dynamic", shapes=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_flux_application_kwargs_add_quant_only_when_set():
    from difflet.cli.orchestrators import flux as flux_orch

    plain = flux_orch.FluxOrchestrator(_flux_args())._application_kwargs()
    assert "quant" not in plain and "quant_cache_dir" not in plain
    fp8 = flux_orch.FluxOrchestrator(_flux_args(quant="fp8", quant_act="none"))._application_kwargs()
    assert fp8["quant"] == QuantSpec.for_model("flux", activation="none").to_dict()
    assert fp8["quant"]["targets"] != list(QuantSpec().targets)  # FLUX targets, not Wan's
    assert fp8["quant_cache_dir"] == "/tmp/cache"
    assert {k: v for k, v in fp8.items() if k not in ("quant", "quant_cache_dir")} == plain


def test_pipeline_cache_kwargs_key_the_layer_schema_for_fp8_only():
    """DiffletPipeline models (FLUX, LTX-2): the cache key gains quant + the layer
    schema; quant_cache_dir is runtime-only; bf16 kwargs are untouched."""
    from difflet.backends.trainium.core.quant import QUANT_LAYER_SCHEMA
    from difflet.pipeline.difflet_pipeline import _cache_application_kwargs

    assert _cache_application_kwargs({}, model_name="flux") is None
    bf16 = _cache_application_kwargs({"taef1": True}, model_name="flux")
    assert bf16 == {"taef1": True}
    spec = QuantSpec.for_model("flux").to_dict()
    fp8 = _cache_application_kwargs({"taef1": True, "quant": spec, "quant_cache_dir": "/c"}, model_name="flux")
    assert fp8["quant"] == spec and fp8["quant_layer_schema"] == QUANT_LAYER_SCHEMA
    assert fp8["taef1"] is True



def test_quantize_command_uses_the_model_targets_and_rejects_unwired_models(monkeypatch, tmp_path, capsys):
    import torch
    from safetensors.torch import save_file

    from difflet.cli import quantize as quantize_cmd
    from difflet.quant.checkpoint import read_manifest

    model_dir = tmp_path / "flux"
    (model_dir / "transformer").mkdir(parents=True)
    save_file(
        {"transformer_blocks.0.attn.to_q.weight": torch.randn(8, 8, dtype=torch.bfloat16),
         "blocks.0.attn1.to_q.weight": torch.randn(8, 8, dtype=torch.bfloat16),   # a Wan-style name: not a FLUX target
         "proj_out.weight": torch.randn(4, 8, dtype=torch.bfloat16)},
        str(model_dir / "transformer" / "diffusion_pytorch_model.safetensors"),
    )
    (model_dir / "transformer" / "config.json").write_text(json.dumps({}))
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", lambda *a, **k: str(model_dir))

    args = argparse.Namespace(model_id="black-forest-labs/FLUX.1-dev", revision=None, quant="fp8",
                              quant_granularity="tensor", quant_act="dynamic",
                              cache_dir=str(tmp_path / "cache"), force=False)
    assert quantize_cmd.run(args) == 0
    dest = next((tmp_path / "cache" / "quantized").rglob("difflet_quant.json")).parent
    manifest = read_manifest(dest)
    assert manifest["report"]["quantized"] == ["transformer_blocks.0.attn.to_q"]  # FLUX targets, not Wan's
    assert manifest["spec"]["targets"] == list(QuantSpec.for_model("flux").targets)

    unwired = argparse.Namespace(model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
                                 revision=None, quant="fp8", quant_granularity="tensor", quant_act="dynamic",
                                 cache_dir=str(tmp_path / "cache"), force=False)
    assert quantize_cmd.run(unwired) == 1
    assert "does not support --quant" in capsys.readouterr().err



# ---------------------------------------------------------- Qwen-Image orchestrator


def _qwen_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        model_id="Qwen/Qwen-Image", tp_degree=4, cp_degree=1, cp_mode="gather_kv", sp_enabled=False,
        height=1024, width=1024, num_frames=None, cache_dir="/tmp/cache", force=False, revision=None,
        prompt="a cat", output="/tmp/q.png", steps=2, guidance_scale=4.0, seed=42, work_dir=None,
        keep_work_dir=False, teacache_cadence=None, teacache_online_delta=None, teacache_speedup=None,
        teacache_calibration=None, quant=None, quant_granularity="tensor", quant_act="dynamic", shapes=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_qwen_stage_identity_is_unchanged_for_bf16_and_extended_for_fp8(monkeypatch):
    from difflet.backends.trainium.core.quant import QUANT_LAYER_SCHEMA
    from difflet.cli.orchestrators import qwen_image as qwen_orch

    monkeypatch.setattr(qwen_orch, "stage_toolchain_versions", lambda: {"toolchain": "x"})
    bf16 = qwen_orch.QwenImageOrchestrator(_qwen_args())
    inputs = bf16._stage_cache_inputs("generate", bf16.args)
    assert "quant" not in inputs and "quant_layer_schema" not in inputs
    fp8 = qwen_orch.QwenImageOrchestrator(_qwen_args(quant="fp8", quant_act="none"))
    fp8_inputs = fp8._stage_cache_inputs("generate", fp8.args)
    assert fp8_inputs["quant"] == QuantSpec.for_model("qwen_image", activation="none").to_dict()
    assert fp8_inputs["quant_layer_schema"] == QUANT_LAYER_SCHEMA
    assert {k: v for k, v in fp8_inputs.items() if k not in ("quant", "quant_layer_schema")} == inputs
    for stage in ("text", "vae"):
        assert "quant" not in fp8._stage_cache_inputs(stage, fp8.args)
    assert fp8._quant_app_kwargs(fp8.args) == {"quant": fp8_inputs["quant"], "quant_cache_dir": "/tmp/cache"}
    assert bf16._quant_app_kwargs(bf16.args) == {}
