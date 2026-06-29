"""Cover the lazy public exports of the top-level difflet package."""

import pytest

import difflet


def test_lazy_parallel_config():
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    assert difflet.DiffletParallelConfig is DiffletParallelConfig


def test_lazy_candidate_config():
    from difflet.pipeline.parallel_config import CandidateConfig

    assert difflet.CandidateConfig is CandidateConfig


def test_lazy_register_model():
    from difflet.registry import register_model

    assert difflet.register_model is register_model


def test_lazy_current_backend():
    from difflet.backends import current_backend

    assert difflet.current_backend is current_backend


def test_lazy_difflet_pipeline():
    from difflet.pipeline.difflet_pipeline import DiffletPipeline

    assert difflet.DiffletPipeline is DiffletPipeline


def test_all_public_attrs_resolve():
    for name in difflet.__all__:
        assert getattr(difflet, name) is not None


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        difflet.does_not_exist_zzz
