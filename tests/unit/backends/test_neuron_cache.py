"""Unit tests for the neuron backend's persistent compile-cache locations; no Neuron hardware needed."""

import os
import subprocess
import sys

import pytest

from difflet.backends.neuron import cache as neuron_cache

_CACHE_VARS = list(neuron_cache.CACHE_ENV_DIRS)


@pytest.fixture
def clean_cache_env(monkeypatch, tmp_path):
    for var in _CACHE_VARS:
        # setenv first so monkeypatch records the original value (or its absence) and
        # teardown undoes apply_neuron_cache_env's os.environ writes.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
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
    pytest.importorskip("torch")  # torch in sys.modules makes apply_neuron_cache_env warn
    with pytest.warns(UserWarning):
        env = neuron_cache.apply_neuron_cache_env()
    for var, path in env.items():
        assert os.environ[var] == path
        assert os.path.isdir(path)


def test_apply_warns_when_torch_is_already_imported(clean_cache_env):
    pytest.importorskip("torch")
    with pytest.warns(UserWarning, match="TORCH_NEURONX_HLO_CACHE_DIR"):
        neuron_cache.apply_neuron_cache_env()


def test_cache_module_imports_without_torch():
    code = "import sys, difflet.backends.neuron.cache; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
