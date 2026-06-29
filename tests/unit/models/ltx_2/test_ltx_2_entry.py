"""CPU-only unit coverage for difflet.models.ltx_2.entry and the package __init__."""

import os


import pytest

import difflet.models.ltx_2 as ltx_2_pkg
from difflet.models.ltx_2.entry import create_ltx_2_application
from difflet.models.ltx_2.application import NeuronLTX2Application
from difflet.pipeline.parallel_config import DiffletParallelConfig


@pytest.fixture(autouse=True)
def _force_cpu_backend():
    prev = os.environ.get("DIFFLET_BACKEND")
    os.environ["DIFFLET_BACKEND"] = "cpu"
    yield
    if prev is None:
        os.environ.pop("DIFFLET_BACKEND", None)
    else:
        os.environ["DIFFLET_BACKEND"] = prev


def test_package_reexports_public_symbols():
    for name in (
        "LTX2DiTInputBundle",
        "LTX2Orchestrator",
        "LTX2PipelineOutput",
        "NeuronLTX2Application",
        "create_ltx_2_transformer_config",
        "validate_ltx_2_dit_inputs",
    ):
        assert hasattr(ltx_2_pkg, name)


def test_entry_rejects_non_trainium_backend(tmp_path):
    with pytest.raises(NotImplementedError, match="trainium backend"):
        create_ltx_2_application(
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(tp_degree=1),
            dtype="bf16",
            shape={"height": 256, "width": 512, "num_frames": 17},
            backend="cpu",
        )


def test_entry_rejects_context_parallel(tmp_path):
    with pytest.raises(NotImplementedError, match="CP is deferred"):
        create_ltx_2_application(
            model_path=str(tmp_path),
            parallel=DiffletParallelConfig(tp_degree=1, cp_degree=2),
            dtype="bf16",
            shape={"height": 256, "width": 512, "num_frames": 17},
        )


def test_entry_builds_application(tmp_path):
    app = create_ltx_2_application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="fp32",
        shape={"height": 256, "width": 512, "num_frames": 17},
    )
    assert isinstance(app, NeuronLTX2Application)
    assert app.transformer is None  # empty model dir -> no transformer/config.json
