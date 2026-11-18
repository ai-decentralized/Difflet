"""Adding a model to the CLI is one registry record: no table in difflet/cli to edit."""

import argparse
import importlib

import pytest

from difflet import registry
from difflet.cli import stage
from difflet.cli.orchestrators.base import ModelOrchestrator


main = importlib.import_module("difflet.cli.main")  # difflet.cli exports a main() function


class _ToyOrchestrator(ModelOrchestrator):
    def download(self): ...
    def compile(self): ...
    def generate(self): ...


@pytest.fixture
def toy_model():
    registry.registered_models()  # the built-ins first, so the toy does not shadow their loading

    @registry.register_model(
        name="cli_toy",
        application_factory="tests.unit.cli.test_registry_cli:_ToyOrchestrator",
        hf_paths=("org/cli-toy",),
        backends=("trainium",),
        cli_model_ids=("org/cli-toy",),
        cli_orchestrator="tests.unit.cli.test_registry_cli:_ToyOrchestrator",
    )
    class _Toy:
        pass

    yield "org/cli-toy"
    registry._REGISTRY.pop("cli_toy")


def test_a_registered_model_is_offered_and_dispatched_by_the_cli(toy_model, monkeypatch):
    assert main.cli_model_types()[toy_model] == "cli_toy"
    # Compared by name: the registry's import of this module may be a second copy.
    assert stage._load_orchestrator_class(toy_model).__name__ == "_ToyOrchestrator"

    monkeypatch.setenv("DIFFLET_BACKEND", "trainium")
    orch = main._get_orchestrator(argparse.Namespace(model_id=toy_model, torch_compile=False))
    assert type(orch).__name__ == "_ToyOrchestrator" and orch.args.model_id == toy_model


def test_an_unregistered_model_id_is_refused_with_the_registered_ones(capsys):
    args = main._build_parser().parse_args(["generate", "--model-id", "org/unknown", "--prompt", "p"])
    with pytest.raises(SystemExit):
        main.check_command_args(args)
    err = capsys.readouterr().err
    assert "Unknown model-id 'org/unknown'" in err and "Qwen/Qwen-Image" in err


def test_the_cli_offers_exactly_what_it_did_before_the_tables_moved():
    assert main.cli_model_types() == {
        "black-forest-labs/FLUX.1-dev": "flux",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "wan",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers": "wan",
        "hunyuanvideo-community/HunyuanVideo": "hunyuan_video",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v": "hunyuan_video_15",
        "Qwen/Qwen-Image": "qwen_image",
        "MiniMaxAI/MiniMax-H3": "minimax_h3",
        "Lightricks/LTX-2": "ltx_2",
    }


@pytest.mark.parametrize("flag, says", [(["--sp"], "does not support --sp"),
                                         (["--cfg-parallel"], "is guidance-distilled")])
def test_minimax_parallel_flags_are_refused_with_a_message_not_a_traceback(flag, says, monkeypatch, capsys):
    # MiniMax-H3 declared no capabilities: these ended in "ValueError: model 'minimax_h3'
    # declares no capabilities".
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    args = main._build_parser().parse_args(
        ["generate", "--model-id", "MiniMaxAI/MiniMax-H3", "--prompt", "p", "--output", "o.mp4", *flag])
    with pytest.raises(SystemExit):
        main.check_command_args(args)
    assert says in capsys.readouterr().err


@pytest.mark.parametrize("model_id", ["hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
                                      "MiniMaxAI/MiniMax-H3"])
def test_a_flag_the_model_never_reads_is_refused_up_front(model_id, monkeypatch, capsys):
    # HV1.5 never read --tp-reduce-dtype (the run stayed bf16 with no warning); MiniMax-H3
    # refused it only after dispatch, from inside its orchestrator.
    monkeypatch.setenv("DIFFLET_BACKEND", "neuron")
    args = main._build_parser().parse_args(
        ["generate", "--model-id", model_id, "--prompt", "p", "--output", "o.mp4", "--tp-reduce-dtype", "fp32"])
    with pytest.raises(SystemExit):
        main.check_command_args(args)
    assert f"{model_id} does not take --tp-reduce-dtype" in capsys.readouterr().err


def test_plan_refuses_a_model_without_planner_geometry_instead_of_a_traceback(capsys):
    # MiniMax-H3 ended in "ValueError: no planner dimensions registered for model 'minimax_h3'".
    from difflet.cli import plan

    with pytest.raises(SystemExit):
        plan.run(argparse.Namespace(model_id="MiniMaxAI/MiniMax-H3"))
    assert "difflet plan has no geometry for MiniMaxAI/MiniMax-H3" in capsys.readouterr().err
