"""Extra tests for difflet.pipeline.difflet_pipeline internal helpers.

These exercise the pure-Python glue (teacache kwarg merge, dtype default,
compile/load signature dispatch, artifact readiness) without touching the
hardware-bound from_pretrained path.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from difflet.pipeline import difflet_pipeline as dp


def test_default_dtype_is_bfloat16():
    assert dp._default_dtype() is torch.bfloat16


def test_merge_teacache_kwargs_none_when_empty():
    assert dp._merge_teacache_kwargs(None, teacache_speedup=None, teacache_calibration_path=None) is None


def test_merge_teacache_kwargs_injects_values():
    merged = dp._merge_teacache_kwargs(
        {"foo": 1}, teacache_speedup=1.5, teacache_calibration_path="/cal.json"
    )
    assert merged["teacache_speedup"] == 1.5
    assert merged["teacache_calibration_path"] == "/cal.json"
    assert merged["foo"] == 1


def test_merge_teacache_kwargs_conflict_raises():
    with pytest.raises(ValueError):
        dp._merge_teacache_kwargs(
            {"teacache_speedup": 2.0},
            teacache_speedup=1.5,
            teacache_calibration_path=None,
        )


def test_merge_teacache_kwargs_same_value_no_conflict():
    merged = dp._merge_teacache_kwargs(
        {"teacache_speedup": 1.5}, teacache_speedup=1.5, teacache_calibration_path=None
    )
    assert merged["teacache_speedup"] == 1.5


def test_compile_app_passes_debug_when_supported():
    calls = {}

    class App:
        def compile(self, path, debug=False):
            calls["path"] = path
            calls["debug"] = debug

    dp._compile_app(App(), Path("/out"), debug=True)
    assert calls == {"path": "/out", "debug": True}


def test_compile_app_omits_debug_when_unsupported():
    calls = {}

    class App:
        def compile(self, path):
            calls["path"] = path

    dp._compile_app(App(), Path("/out"), debug=True)
    assert calls == {"path": "/out"}


def test_compiled_artifacts_ready_defaults_true_without_checker():
    assert dp._compiled_artifacts_ready(SimpleNamespace(), Path("/x")) is True


def test_compiled_artifacts_ready_uses_checker():
    app = SimpleNamespace(has_compiled_artifacts=lambda p: False)
    assert dp._compiled_artifacts_ready(app, Path("/x")) is False
    app2 = SimpleNamespace(has_compiled_artifacts=lambda p: True)
    assert dp._compiled_artifacts_ready(app2, Path("/x")) is True


def test_qualified_profile_identity_uses_resolved_snapshot_revision():
    kwargs = dp._bind_qualified_profile_identity(
        {"cache_profile_file": "/profile.json"},
        model_id="black-forest-labs/FLUX.1-dev",
        model_path="/cache/models--flux/snapshots/resolved-hash",
        revision="requested-tag",
    )
    assert kwargs["cache_runtime_model_id"] == "black-forest-labs/FLUX.1-dev"
    assert kwargs["cache_runtime_model_revision"] == "resolved-hash"


def test_qualified_profile_identity_rejects_conflicting_override():
    with pytest.raises(ValueError, match="conflicts"):
        dp._bind_qualified_profile_identity(
            {
                "cache_profile_file": "/profile.json",
                "cache_runtime_model_revision": "wrong",
            },
            model_id="black-forest-labs/FLUX.1-dev",
            model_path="/cache/models--flux/snapshots/resolved-hash",
            revision=None,
        )


def test_qualified_profile_does_not_change_compiled_graph_cache_key():
    kwargs = dp._cache_application_kwargs(
        {
            "cache_profile_file": "/profile.json",
            "cache_profile_qualification_file": "/qualification.json",
            "cache_runtime_model_id": "model",
            "cache_runtime_model_revision": "revision",
        }
    )
    assert kwargs is None


def _fake_backend():
    return SimpleNamespace(
        resolve_load_rank_range=lambda start_rank_id, local_ranks_size: (
            start_rank_id or 0,
            local_ranks_size or 1,
        )
    )


def test_load_app_passes_supported_kwargs():
    calls = {}

    class App:
        def load(self, path, start_rank_id=None, local_ranks_size=None, skip_warmup=False):
            calls.update(
                path=path,
                start_rank_id=start_rank_id,
                local_ranks_size=local_ranks_size,
                skip_warmup=skip_warmup,
            )

    dp._load_app(
        App(),
        Path("/c"),
        backend=_fake_backend(),
        start_rank_id=3,
        local_ranks_size=2,
        skip_warmup=True,
    )
    assert calls["path"] == "/c"
    assert calls["start_rank_id"] == 3
    assert calls["local_ranks_size"] == 2
    assert calls["skip_warmup"] is True


def test_load_app_minimal_signature():
    calls = {}

    class App:
        def load(self, path):
            calls["path"] = path

    dp._load_app(
        App(),
        Path("/c"),
        backend=_fake_backend(),
        start_rank_id=None,
        local_ranks_size=None,
        skip_warmup=False,
    )
    assert calls == {"path": "/c"}
