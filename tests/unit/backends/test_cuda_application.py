"""CUDA component lifecycle (CUDA plan Phase 3): manifest-only compile, eager load.

No GPU needed: the base takes ``device=`` and these tests run it on "cpu". The same
module is run with ``device="cuda"`` on the box in Task 8 (plan Review Focus 4).
"""

import inspect
import json

import pytest
import torch
import torch.nn as nn

from difflet.backends.cuda.core import application_base as ab
from difflet.backends.cuda.ops_impl import linear as L
from difflet.backends.cuda.ops_impl import parallel_mesh as pm
from difflet.pipeline.parallel_mesh import MeshSpec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def _clean_mesh():
    pm.destroy_parallel_mesh()
    yield
    pm.destroy_parallel_mesh()


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = L.ColumnParallelLinear(8, 16, bias=True, gather_output=False)
        self.down = L.RowParallelLinear(16, 8, bias=True, input_is_parallel=True)
        self.register_buffer("rope", torch.arange(8, dtype=torch.float32))

    def forward(self, x):
        return self.down(torch.relu(self.up(x))) + self.rope


class _App(ab.CudaApplicationBase):
    def __init__(self, device=DEVICE):
        super().__init__(config=MeshSpec(tp=1), device=device, dtype=torch.float32)

    def build_module(self):
        pm.init_parallel_mesh(self.config)
        return _Block()

    def get_example_inputs(self):
        return (torch.zeros(2, 8),)


# ------------------------------------------------------------------ contract


def test_signatures_match_what_the_pipeline_reflects_on():
    assert "debug" in inspect.signature(ab.CudaApplicationBase.compile).parameters
    load_params = inspect.signature(ab.CudaApplicationBase.load).parameters
    for name in ("start_rank_id", "local_ranks_size", "skip_warmup"):
        assert name in load_params
    assert callable(ab.CudaApplicationBase.has_compiled_artifacts)


def test_forward_before_load_is_an_explicit_error():
    app = _App()
    with pytest.raises(RuntimeError, match="not loaded"):
        app(torch.zeros(2, 8))


# ------------------------------------------------------------------ manifest


def test_compile_writes_manifest_and_has_compiled_artifacts_sees_it(tmp_path):
    app = _App()
    assert app.has_compiled_artifacts(tmp_path) is False
    app.compile(str(tmp_path))
    body = json.loads((tmp_path / ab.MANIFEST_FILE_NAME).read_text())
    assert body["format"] == "eager"
    assert body["mesh"] == {"dp": 1, "cfg": 1, "cp": 1, "tp": 1}
    assert body["world_size"] == 1
    assert body["dtype"] == "torch.float32"
    assert "torch" in body
    assert app.has_compiled_artifacts(tmp_path) is True
    assert app.is_compiled is True


def test_has_compiled_artifacts_is_false_without_a_manifest(tmp_path):
    assert _App().has_compiled_artifacts(tmp_path) is False


def test_has_compiled_artifacts_survives_a_corrupt_manifest(tmp_path):
    (tmp_path / ab.MANIFEST_FILE_NAME).write_text("{not json")
    assert _App().has_compiled_artifacts(tmp_path) is False


def test_has_compiled_artifacts_rejects_other_mesh_or_dtype(tmp_path):
    app = _App()
    app.compile(str(tmp_path))
    body = json.loads((tmp_path / ab.MANIFEST_FILE_NAME).read_text())
    other_mesh = dict(body, mesh={"dp": 1, "cfg": 1, "cp": 1, "tp": 2}, world_size=2)
    (tmp_path / ab.MANIFEST_FILE_NAME).write_text(json.dumps(other_mesh))
    assert app.has_compiled_artifacts(tmp_path) is False
    other_dtype = dict(body, dtype="torch.bfloat16")
    (tmp_path / ab.MANIFEST_FILE_NAME).write_text(json.dumps(other_dtype))
    assert app.has_compiled_artifacts(tmp_path) is False
    with pytest.raises(ValueError, match="dtype"):
        app.load(str(tmp_path))


def test_load_rejects_a_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        _App().load(str(tmp_path))


# ---------------------------------------------------------------- lifecycle


def test_load_moves_all_params_and_buffers_to_device(tmp_path):
    app = _App()
    app.compile(str(tmp_path))
    app.load(str(tmp_path), skip_warmup=True)
    devices = {p.device.type for p in app.module.parameters()} | {
        b.device.type for b in app.module.buffers()
    }
    assert devices == {DEVICE}, devices
    assert app.is_loaded is True


def test_load_then_forward_runs_the_module(tmp_path):
    app = _App()
    app.compile(str(tmp_path))
    app.load(str(tmp_path))  # warmup runs the example inputs
    x = torch.randn(2, 8, device=DEVICE)
    out = app(x)
    assert out.shape == (2, 8)
    assert torch.allclose(out, app.module(x))


def test_load_accepts_and_ignores_trainium_rank_range(tmp_path):
    app = _App()
    app.compile(str(tmp_path))
    app.load(str(tmp_path), start_rank_id=0, local_ranks_size=1, skip_warmup=True)
    assert app.is_loaded
