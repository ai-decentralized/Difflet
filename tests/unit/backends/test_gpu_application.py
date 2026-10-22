"""The generic GPU flow: the component lifecycle (manifest-only compile, eager load) and the
application contract every model shares, parametrized over the models with a GPU spec.

No GPU needed: the base takes ``device=`` and these tests run it on "cpu" when no CUDA
device is present. A model's own behaviour is tested next to the model.
"""

import inspect
import json

import pytest
import torch
import torch.nn as nn

from difflet.backends.gpu import application as ab
from difflet.backends.cuda.ops_impl import linear as L
from difflet.backends.cuda.ops_impl import parallel_mesh as pm
from difflet.models.spec import gpu_model_spec, gpu_model_specs
from difflet.pipeline.parallel_config import DiffletParallelConfig
from difflet.pipeline.parallel_mesh import MeshSpec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = [spec.model_type for spec in gpu_model_specs()]
SHAPE = {"height": 64, "width": 64}
# Tiny transformer/config.json per model: enough to build the components, never loaded.
TINY_CONFIGS = {
    "qwen_image": dict(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=1,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=64,
        guidance_embeds=False,
        axes_dims_rope=[8, 12, 12],
    ),
    "flux": dict(
        patch_size=1,
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        pooled_projection_dim=8,
        guidance_embeds=True,
        axes_dims_rope=[2, 2, 4],
    ),
    "wan": dict(
        patch_size=[1, 2, 2],
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=32,
        ffn_dim=32,
        num_layers=1,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
    ),
}


@pytest.fixture(autouse=True)
def _cpu_backend(monkeypatch):
    monkeypatch.setenv("DIFFLET_BACKEND", "cpu")


@pytest.fixture(autouse=True)
def _clean_mesh():
    pm.destroy_parallel_mesh()
    yield
    pm.destroy_parallel_mesh()


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = L.ColumnParallelLinear(8, 16, bias=True, gather_output=False)
        self.blocks = nn.ModuleList([nn.ReLU()])
        self.down = L.RowParallelLinear(16, 8, bias=True, input_is_parallel=True)
        self.register_buffer("rope", torch.arange(8, dtype=torch.float32))

    def forward(self, x):
        return self.down(self.blocks[0](self.up(x))) + self.rope


class _App(ab.GpuComponent):
    def __init__(self, device=DEVICE, **kwargs):
        super().__init__(config=MeshSpec(tp=1), device=device, dtype=torch.float32, **kwargs)

    def build_module(self):
        pm.init_parallel_mesh(self.config)
        return _Block()

    def get_example_inputs(self):
        return (torch.zeros(2, 8),)

    def repeated_blocks(self, module):
        return [module.blocks]


def _build(tmp_path, model, *, parallel=None, **kwargs):
    return ab.create_gpu_application(
        model,
        model_path=str(tmp_path),
        parallel=parallel or DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape=dict(SHAPE),
        **kwargs,
    )


# ------------------------------------------------------------------ contract


def test_signatures_match_what_the_pipeline_reflects_on():
    for cls in (ab.GpuComponent, ab.GpuApplication):
        assert "debug" in inspect.signature(cls.compile).parameters
        load_params = inspect.signature(cls.load).parameters
        for name in ("start_rank_id", "local_ranks_size", "skip_warmup"):
            assert name in load_params
        assert callable(cls.has_compiled_artifacts)


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


@pytest.mark.parametrize("torch_compile", [False, True])
def test_torch_compile_wraps_the_repeated_blocks_only_when_asked(tmp_path, torch_compile):
    from torch._dynamo.eval_frame import OptimizedModule

    app = _App(torch_compile=torch_compile)
    app.compile(str(tmp_path))
    app.load(str(tmp_path), skip_warmup=True)
    assert isinstance(app.module.blocks[0], OptimizedModule) is torch_compile


# ------------------------------------------------------------- applications


@pytest.mark.parametrize("model", MODELS)
def test_registry_routes_cuda_to_an_application_that_needs_its_transformer(tmp_path, model):
    from difflet.registry import resolve_model

    spec = gpu_model_spec(model)
    app = resolve_model(spec.cli_model_ids[0]).create_application(
        model_path=str(tmp_path),
        parallel=DiffletParallelConfig(tp_degree=1),
        dtype="bf16",
        shape=dict(SHAPE),
        backend="cuda",
    )
    assert type(app).__name__ == spec.application.rpartition(":")[2]
    assert app.dtype is torch.bfloat16 and app.device == torch.device("cuda")
    # An empty model dir: no transformer/config.json.
    assert app.transformer is None and app.has_compiled_artifacts(str(tmp_path)) is False
    for step in (app.compile, app.load):
        with pytest.raises(NotImplementedError, match="transformer/config.json"):
            step(str(tmp_path))
    with pytest.raises(RuntimeError, match="load"):
        app.generate("a cat")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "parallel,expected",
    [
        ({"tp_degree": 2}, "tensor parallelism.*not supported"),
        ({"tp_degree": 1, "cp_degree": 2}, "context parallelism.*not supported"),
        ({"tp_degree": 1, "sp_enabled": True}, "sequence parallelism.*not supported"),
    ],
)
def test_unsupported_parallel_modes_are_refused_before_any_load(
    tmp_path, model, parallel, expected
):
    with pytest.raises(NotImplementedError, match=expected):
        _build(tmp_path, model, parallel=DiffletParallelConfig(**parallel))


@pytest.mark.parametrize("model", MODELS)
def test_cfg_parallel_is_refused_without_a_cond_uncond_pair(tmp_path, model):
    spec = gpu_model_spec(model)
    parallel = DiffletParallelConfig(tp_degree=1, cfg_parallel_enabled=True)
    if spec.accepts_cfg_parallel:
        assert _build(tmp_path, model, parallel=parallel).parallel is parallel
    else:
        with pytest.raises(NotImplementedError, match=f"does not apply to {spec.display_name}"):
            _build(tmp_path, model, parallel=parallel)


@pytest.mark.parametrize(
    "model,key",
    [
        (spec.model_type, key)
        for spec in gpu_model_specs()
        for key in spec.refused_application_kwargs
        + tuple(prefix + "x" for prefix in spec.refused_application_kwarg_prefixes)
    ],
)
def test_refused_kwargs_are_refused_only_when_they_carry_a_request(tmp_path, model, key):
    with pytest.raises(NotImplementedError, match="not supported"):
        _build(tmp_path, model, **{key: 1})
    # The CLI and DiffletPipeline pass several of them with their "off" value.
    _build(tmp_path, model, **{key: None})
    _build(tmp_path, model, **{key: False})


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("kwargs", [{}, {"torch_compile": True}])
def test_torch_compile_reaches_every_transformer_and_is_off_by_default(tmp_path, model, kwargs):
    for name in ("transformer", "transformer_2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text(json.dumps(TINY_CONFIGS[model]))
    app = _build(tmp_path, model, device="cpu", **kwargs)
    expected = bool(kwargs)
    assert app.torch_compile is expected
    assert {t.torch_compile for _, t in app.named_transformers()} == {expected}


@pytest.mark.parametrize("model", MODELS)
def test_call_with_a_prompt_dispatches_to_generate(tmp_path, model, monkeypatch):
    app = _build(tmp_path, model)
    monkeypatch.setattr(app, "generate", lambda **kw: kw)
    assert app(prompt="a cat", seed=0) == {"prompt": "a cat", "seed": 0}
