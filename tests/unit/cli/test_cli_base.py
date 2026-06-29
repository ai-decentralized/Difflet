from __future__ import annotations
import argparse
import pytest


def test_abstract_orchestrator_cannot_be_instantiated():
    from difflet.cli.orchestrators.base import ModelOrchestrator
    with pytest.raises(TypeError):
        ModelOrchestrator(argparse.Namespace())


def test_run_calls_download_compile_generate_in_order():
    from difflet.cli.orchestrators.base import ModelOrchestrator
    order = []

    class Concrete(ModelOrchestrator):
        def download(self): order.append("download")
        def compile(self): order.append("compile")
        def generate(self): order.append("generate")

    Concrete(argparse.Namespace()).run()
    assert order == ["download", "compile", "generate"]


def test_run_stage_internal_raises_not_implemented():
    from difflet.cli.orchestrators.base import ModelOrchestrator
    class Concrete(ModelOrchestrator):
        def download(self): pass
        def compile(self): pass
        def generate(self): pass

    orch = Concrete(argparse.Namespace())
    with pytest.raises(NotImplementedError):
        orch._run_stage_internal("clip", argparse.Namespace())
