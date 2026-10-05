"""Shared tensorizer options (difflet.backends.trainium.core.compiler_flags), the backbones'
compiler arguments that consume them, and their cache-key contribution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from difflet.backends.trainium.core import compiler_flags as cf


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("DIFFLET_STRIDED_DMA", "DIFFLET_TENSORIZER_EXTRA", "DIFFLET_WAN_TENSORIZER_EXTRA"):
        monkeypatch.delenv(name, raising=False)


def test_wan_defaults_strided_dma_on_the_others_off():
    assert cf.tensorizer_options("wan") == "--enable-ccop-compute-overlap --vectorize-strided-dma"
    assert cf.tensorizer_cache_inputs("wan") == {"tensorizer_extras": ["--vectorize-strided-dma"]}
    for model in ("flux", "qwen_image", "hunyuan_video", "ltx_2", None, "unknown"):
        assert cf.tensorizer_options(model) == "--enable-ccop-compute-overlap"
        assert cf.tensorizer_cache_inputs(model) == {}


@pytest.mark.parametrize("value,expected", [("0", False), ("false", False), ("off", False), ("1", True), ("yes", True)])
def test_env_forces_strided_dma_either_way(monkeypatch, value, expected):
    monkeypatch.setenv("DIFFLET_STRIDED_DMA", value)
    for model in ("wan", "flux"):
        assert (cf.STRIDED_DMA_FLAG in cf.tensorizer_options(model)) is expected


def test_generic_and_model_extras_append_without_duplicates(monkeypatch):
    monkeypatch.setenv("DIFFLET_TENSORIZER_EXTRA", "--foo --vectorize-strided-dma")
    monkeypatch.setenv("DIFFLET_WAN_TENSORIZER_EXTRA", "--bar --foo")
    assert cf.tensorizer_extras("wan") == ["--vectorize-strided-dma", "--foo", "--bar"]
    assert cf.tensorizer_extras("flux") == ["--foo", "--vectorize-strided-dma"]  # no model env for flux


def test_cache_spec_key_is_unchanged_at_historical_defaults_and_moves_with_extras(monkeypatch):
    from difflet.pipeline.compile_cache import CacheSpec
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    def spec(model_name):
        return CacheSpec(model_id="m", model_path="/x", model_name=model_name,
                         parallel=DiffletParallelConfig(tp_degree=4), dtype="bfloat16",
                         height=480, width=832, num_frames=9)

    assert "tensorizer_extras" not in spec("flux").cache_inputs()
    assert spec("wan").cache_inputs()["tensorizer_extras"] == ["--vectorize-strided-dma"]
    monkeypatch.setenv("DIFFLET_STRIDED_DMA", "0")
    assert "tensorizer_extras" not in spec("wan").cache_inputs()  # the pre-2026-10-05 key
    monkeypatch.setenv("DIFFLET_STRIDED_DMA", "1")
    assert spec("flux").cache_inputs()["tensorizer_extras"] == ["--vectorize-strided-dma"]


def _compiler_args(klass):
    app = klass.__new__(klass)
    app.config = SimpleNamespace(neuron_config=SimpleNamespace(
        world_size=4, quantized=False, quantization_dtype=None))
    return app.get_compiler_args()


def test_wan_backbone_compiler_args_default_to_strided_dma(monkeypatch):
    pytest.importorskip("neuronx_distributed")
    from difflet.backends.trainium.wan.backbone import NeuronWanBackboneApplication as K

    assert "--tensorizer-options='--enable-ccop-compute-overlap --vectorize-strided-dma'" in _compiler_args(K)
    monkeypatch.setenv("DIFFLET_STRIDED_DMA", "0")
    assert "--vectorize-strided-dma" not in _compiler_args(K)
    monkeypatch.delenv("DIFFLET_STRIDED_DMA")
    monkeypatch.setenv("DIFFLET_WAN_TENSORIZER_EXTRA", "--extra-flag")
    assert "--vectorize-strided-dma --extra-flag'" in _compiler_args(K)


@pytest.mark.parametrize("module,cls", [
    ("difflet.backends.trainium.qwen_image.transformer", "NeuronQwenImageTransformerApplication"),
    ("difflet.backends.trainium.ltx_2.transformer", "NeuronLTX2TransformerApplication"),
    ("difflet.backends.trainium.hunyuan_video.backbone", "NeuronHunyuanVideoBackboneApplication"),
])
def test_other_backbones_keep_strided_dma_off_until_opted_in(monkeypatch, module, cls):
    pytest.importorskip("neuronx_distributed")
    import importlib

    klass = getattr(importlib.import_module(module), cls)
    assert "--tensorizer-options='--enable-ccop-compute-overlap'" in _compiler_args(klass)
    monkeypatch.setenv("DIFFLET_STRIDED_DMA", "1")
    assert "--vectorize-strided-dma" in _compiler_args(klass)
