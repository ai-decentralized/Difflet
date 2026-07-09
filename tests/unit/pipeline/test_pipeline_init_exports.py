"""Cover the lazy public exports of difflet.pipeline.__init__."""

import pytest

import difflet.pipeline as pkg


def test_lazy_parallel_config():
    from difflet.pipeline.parallel_config import DiffletParallelConfig

    assert pkg.DiffletParallelConfig is DiffletParallelConfig


def test_lazy_candidate_config():
    from difflet.pipeline.parallel_config import CandidateConfig

    assert pkg.CandidateConfig is CandidateConfig


def test_lazy_teacache_calibration():
    from difflet.pipeline.teacache import TeaCacheCalibration

    assert pkg.TeaCacheCalibration is TeaCacheCalibration


def test_lazy_teacache_controller():
    from difflet.pipeline.teacache import TeaCacheController

    assert pkg.TeaCacheController is TeaCacheController


def test_lazy_difflet_pipeline():
    from difflet.pipeline.difflet_pipeline import DiffletPipeline

    assert pkg.DiffletPipeline is DiffletPipeline


def test_all_exports_listed():
    for name in pkg.__all__:
        assert getattr(pkg, name) is not None


def test_unknown_attr_raises():
    with pytest.raises(AttributeError):
        pkg.does_not_exist
