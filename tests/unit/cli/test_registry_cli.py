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
