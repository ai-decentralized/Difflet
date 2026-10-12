"""Per-model NEURON_RT_VIRTUAL_CORE_SIZE defaults (compiler_flags.VIRTUAL_CORE_DEFAULTS): LTX-2
traces its DiT for VC2 since 2026-10-05 and carries it in the cache key."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from difflet.backends.trainium.core import compiler_flags as cf


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # apply_virtual_core_env writes os.environ directly: work on a copy so nothing leaks
    # into later tests (test_cli_runner asserts the variable is absent).
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in ("DIFFLET_VIRTUAL_CORE_SIZE", "DIFFLET_STRIDED_DMA", "DIFFLET_TENSORIZER_EXTRA",
                 "NEURON_RT_VIRTUAL_CORE_SIZE"):
        os.environ.pop(name, None)


def test_ltx2_defaults_virtual_core_2_and_strided_dma(monkeypatch):
    assert cf.virtual_core_size("ltx_2") == 2
    assert cf.virtual_core_cache_inputs("ltx_2") == {"virtual_core_size": 2}
    assert cf.tensorizer_cache_inputs("ltx_2") == {"tensorizer_extras": ["--vectorize-strided-dma"]}
    for model in ("wan", "flux", "qwen_image", "hunyuan_video", None):
        assert cf.virtual_core_size(model) is None
        assert cf.virtual_core_cache_inputs(model) == {}
    monkeypatch.setenv("DIFFLET_VIRTUAL_CORE_SIZE", "1")
    assert cf.virtual_core_size("ltx_2") is None
    assert cf.virtual_core_cache_inputs("ltx_2") == {}


def test_apply_virtual_core_env_touches_only_models_with_a_default(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VIRTUAL_CORE_SIZE", "7")
    assert cf.apply_virtual_core_env("flux") is None
    assert os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] == "7"  # untouched
    assert cf.apply_virtual_core_env("ltx_2") == 2
    assert os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] == "2"
    monkeypatch.setenv("DIFFLET_VIRTUAL_CORE_SIZE", "1")
    assert cf.apply_virtual_core_env("ltx_2") is None
    assert "NEURON_RT_VIRTUAL_CORE_SIZE" not in os.environ


def test_cache_spec_key_carries_the_ltx2_virtual_core_size(monkeypatch):
    from difflet.pipeline.compile_cache import CacheSpec
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    spec = CacheSpec(model_id="m", model_path="/x", model_name="ltx_2",
                     parallel=DiffletParallelConfig(tp_degree=4), dtype="bfloat16",
                     height=480, width=704, num_frames=49)
    inputs = spec.cache_inputs()
    assert inputs["virtual_core_size"] == 2
    assert inputs["tensorizer_extras"] == ["--vectorize-strided-dma"]
    monkeypatch.setenv("DIFFLET_VIRTUAL_CORE_SIZE", "1")
    monkeypatch.setenv("DIFFLET_STRIDED_DMA", "0")
    legacy = spec.cache_inputs()
    assert "virtual_core_size" not in legacy and "tensorizer_extras" not in legacy  # the pre-2026-10-05 key


class _Args(SimpleNamespace):
    def __getattr__(self, name):  # unknown CLI fields read as unset
        return None


def test_ltx2_orchestrator_compile_exports_vc2_before_tracing(monkeypatch):
    from difflet.cli.orchestrators.ltx_2 import LTX2Orchestrator

    seen = {}
    monkeypatch.setattr("difflet.pipeline.path_resolver.resolve_model_path", lambda *a, **k: "/x")
    monkeypatch.setattr(
        "difflet.pipeline.difflet_pipeline.DiffletPipeline.precompile",
        staticmethod(lambda *a, **k: seen.update({"vc": os.environ.get("NEURON_RT_VIRTUAL_CORE_SIZE")})),
    )
    args = _Args(tp_degree=4, cp_degree=1, cp_mode="gather_kv", cfg_parallel=False, sp_enabled=False,
                 height=480, width=704, num_frames=49, revision=None, cache_dir=None, quant=None)
    LTX2Orchestrator(args).compile()
    assert seen["vc"] == "2"
    monkeypatch.setenv("DIFFLET_VIRTUAL_CORE_SIZE", "1")
    LTX2Orchestrator(args).compile()
    assert seen["vc"] is None
