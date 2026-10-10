"""Unit tests for the neuron backend's per-block compile helper; no Neuron hardware needed."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from difflet.backends.neuron import compile as neuron_compile  # noqa: E402


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.inner = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])  # nested stack

    def forward(self, x):
        return self.proj(x)


class Other(nn.Module):
    def forward(self, x):
        return x


class Model(nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        self.embed = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([Block() for _ in range(depth)])
        self.mixed = nn.ModuleList([Block(), Other()])
        self.single = nn.ModuleList([Block()])


@pytest.fixture
def compile_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(nn.Module, "compile", lambda self, **kw: calls.append((self, kw)))
    return calls


def test_finds_outermost_uniform_stacks_only():
    assert neuron_compile.repeated_block_lists(Model()) == ["blocks"]


def test_eager_mode_compiles_nothing(compile_calls):
    assert neuron_compile.compile_blocks(Model(), mode="eager") == []
    assert compile_calls == []


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        neuron_compile.compile_blocks(Model(), mode="reduce-overhead")


def test_compiles_each_block_in_place_with_neuron_options(compile_calls):
    model = Model(depth=3)
    assert neuron_compile.compile_blocks(model) == ["blocks"]
    assert [module for module, _ in compile_calls] == list(model.blocks)
    for _, options in compile_calls:
        assert options == {"backend": "neuron", "dynamic": False, "fullgraph": True}


def test_explicit_block_lists(compile_calls):
    model = Model()
    assert neuron_compile.compile_blocks(model, block_lists=["mixed"], fullgraph=False) == ["mixed"]
    assert [module for module, _ in compile_calls] == list(model.mixed)
    assert all(options["fullgraph"] is False for _, options in compile_calls)


def test_explicit_name_must_be_a_module_list():
    with pytest.raises(TypeError, match="not an nn.ModuleList"):
        neuron_compile.compile_blocks(Model(), block_lists=["embed"])


def test_no_stack_found_raises():
    with pytest.raises(ValueError, match="no repeated block stack"):
        neuron_compile.compile_blocks(nn.Sequential(nn.Linear(2, 2)))


def test_parameter_names_are_unchanged():
    model = Model()
    before = list(model.state_dict())
    neuron_compile.compile_blocks(model)  # nn.Module.compile is lazy: nothing compiles until called
    assert list(model.state_dict()) == before


# ---- persistent cache locations -------------------------------------------------------

from difflet.backends.neuron import cache as neuron_cache  # noqa: E402

_CACHE_VARS = list(neuron_cache.CACHE_ENV_DIRS)


@pytest.fixture
def clean_cache_env(monkeypatch, tmp_path):
    for var in _CACHE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DIFFLET_COMPILE_CACHE", str(tmp_path))
    return tmp_path


def test_cache_env_defaults_under_difflet_cache(clean_cache_env):
    env = neuron_cache.neuron_cache_env()
    assert env == {
        "TORCH_NEURONX_NEFF_CACHE_DIR": str(clean_cache_env / "_neuron" / "neff"),
        "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR": str(clean_cache_env / "_neuron" / "neff_local"),
        "TORCH_NEURONX_HLO_CACHE_DIR": str(clean_cache_env / "_neuron" / "hlo"),
    }


def test_cache_env_keeps_variables_already_set(clean_cache_env, monkeypatch):
    monkeypatch.setenv("TORCH_NEURONX_NEFF_CACHE_DIR", "/shared/neff")
    env = neuron_cache.neuron_cache_env(root=clean_cache_env / "other")
    assert env["TORCH_NEURONX_NEFF_CACHE_DIR"] == "/shared/neff"
    assert env["TORCH_NEURONX_HLO_CACHE_DIR"] == str(clean_cache_env / "other" / "_neuron" / "hlo")


def test_apply_exports_and_creates_directories(clean_cache_env):
    import os

    with pytest.warns(UserWarning):  # torch is already imported in this test process
        env = neuron_cache.apply_neuron_cache_env()
    for var, path in env.items():
        assert os.environ[var] == path
        assert os.path.isdir(path)


def test_apply_warns_when_torch_is_already_imported(clean_cache_env):
    with pytest.warns(UserWarning, match="TORCH_NEURONX_HLO_CACHE_DIR"):
        neuron_cache.apply_neuron_cache_env()


def test_cache_module_imports_without_torch():
    import subprocess
    import sys

    code = "import sys, difflet.backends.neuron.cache; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
